"""Reference pictures for 3D asset modelling, drawn on the integrated GPU with the pipeline's Qwen-Image drawer.

A modelling agent copies a pictured target far better than it designs from words, so each asset gets a small set of
views before (or while) it is modelled: front, side, top and three-quarter on a plain light-grey background, a 2x2
sheet of the four, an optional close-up of its material, and notes.md (prompts, seed, sizes, seconds, what the views
agree on). The drawer is tools/qwen_image.py, run unchanged in --serve mode (the model stays loaded across assets).

    python tools/assets/refs/ref_draw.py draw ASSET_ID --desc "..." [--style "..."] [--material "..."]
                                         [--recipe hero|separate|sheet] [--seed N] [--also OTHER_ID ...]
    python tools/assets/refs/ref_draw.py batch assets.json      # [{"id", "desc", "style", "material", "also": []}]
    python tools/assets/refs/ref_draw.py note ASSET_ID "how far the views agree"   # fills notes.md's review section

Recipes (see README.md beside this file for the comparison):
  sheet     (default) one 1536 px 2x2 "turnaround sheet" picture, cut along its emptiest row/columns into the four
            views, each cropped square around its object. One picture = one design: the most consistent views.
  hero      the three-quarter view first, then front/side/top each drawn with the hero picture as a reference input
            (Qwen's multi-reference conditioning). Same design, but the "front" and "side" came out as angled views.
  separate  four independent prompts sharing one description and seed: true views, but four different designs.
  extras    only the extra views again, beside a set already drawn (its views and source_*.png as references).

Device and lock: the drawer runs on REFS_DEVICE (the line's image station passes ASSETS_IMAGE_DEVICE: card 05 by
default; on the iGPU one CU is left for the desktop), from that device's Python environment, and holds its lock file (flock -n, polled, never a blocking wait) from its start
to its exit; it is started on the first picture and stopped when the run ends.

Environment (defaults in brackets):
  REFS_DIR        where sets are written                 [~/ue/library/_refs]
  REFS_DEVICE     igpu, or a card's PCI bus (05, 65)      [$ASSETS_IMAGE_DEVICE, else igpu]
  REFS_PY         the Python that runs the drawer         [~/tts-ab/.venv-charlora-igpu/bin/python on the iGPU,
                                                           ~/tts-ab/.venv-qwenimage/bin/python on a card]
  REFS_HIP_INDEX  the helper that maps a device to an index [~/bin/hip-index]
  REFS_LOCK       the device's lock file                  [/tmp/gpu-<device>.lock]
  REFS_SIZE       picture side in pixels                  [1024]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
TOOLS = Path(__file__).resolve().parents[2]
DRAWER = TOOLS / "qwen_image.py"
OUT = Path(os.environ.get("REFS_DIR", HOME / "ue" / "library" / "_refs")).expanduser()
# the device: "igpu", or an XTX card's PCI bus ("05", "65"); the line sets ASSETS_IMAGE_DEVICE for its image station
DEVICE = os.environ.get("REFS_DEVICE") or os.environ.get("ASSETS_IMAGE_DEVICE") or "igpu"
PY = Path(os.environ.get("REFS_PY") or (HOME / "tts-ab" / (".venv-charlora-igpu" if DEVICE == "igpu" else ".venv-qwenimage")
                                         / "bin" / "python")).expanduser()
HIP_INDEX = Path(os.environ.get("REFS_HIP_INDEX", HOME / "bin" / "hip-index")).expanduser()
LOCK = Path(os.environ.get("REFS_LOCK", f"/tmp/gpu-{DEVICE}.lock"))
SIZE = int(os.environ.get("REFS_SIZE", "1024"))
VIEWS = ("front", "side", "top", "three_quarter")

LOOK = ("Studio product reference photograph for a 3D modeller. Orthographic-looking view with no perspective "
        "distortion, the whole object centred and filling most of the frame, nothing cropped. Plain seamless light "
        "grey background, soft even lighting from all sides, no cast shadows, no props, no text, no labels, no "
        "watermark, no people, no hands.")
VIEW_WORDS = {
    "front": "Straight-on front view (front elevation), the camera level and looking directly at the object's front",
    "side": "Exact side view (side elevation, profile), the camera level and looking at the object's left side "
            "at ninety degrees",
    "top": "Top view (plan view), the camera directly above looking straight down at the object",
    "three_quarter": "Three-quarter view from the front left, the camera slightly above, showing the front, the "
                     "left side and a little of the top",
    "back": "Straight-on back view (rear elevation), the camera level and looking directly at the object's back",
    "right": "Exact side view from the object's right side at ninety degrees (right elevation, profile), the camera "
             "level",
    "rear_three_quarter": "Rear three-quarter view: the camera behind the object and off to its right, slightly "
                          "above, so the back end is nearest the camera and the front end points away from it",
}


def log(msg: str) -> None:
    sys.stderr.write(f"[ref_draw] {msg}\n")
    sys.stderr.flush()


# --- the drawer on the iGPU -----------------------------------------------------------------------

class Drawer:
    """qwen_image.py --serve on the iGPU, started on the first draw and stopped by close(). Holds the lock file
    for its whole life (the model sits in iGPU memory while it lives)."""

    def __init__(self, poll: float = 15.0):
        self.proc = None
        self.lock_fd = None
        self.poll = poll
        self.times: list[float] = []
        self.peak_gb = 0.0
        self.load_s = None

    def _take_lock(self) -> None:
        fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        t, said = time.time(), False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not said:
                    log(f"{LOCK} is held by another job: polling every {self.poll:.0f} s")
                    said = True
                time.sleep(self.poll)
        if said:
            log(f"got {LOCK} after {time.time() - t:.0f} s")
        self.lock_fd = fd

    def start(self) -> None:
        if self.proc:
            return
        self._take_lock()
        hip = subprocess.run([str(HIP_INDEX), DEVICE], capture_output=True, text=True, check=True).stdout.strip()
        env = dict(os.environ, HIP_VISIBLE_DEVICES=hip)
        if DEVICE == "igpu":
            env["HSA_CU_MASK"] = f"{hip}:0-38"                                       # one CU left for the desktop
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        t = time.time()
        self.proc = subprocess.Popen([str(PY), "-I", str(DRAWER), "--serve"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        ln = self.proc.stdout.readline()
        if not ln:
            raise RuntimeError(f"the drawer died while loading (rc {self.proc.wait()})")
        self.load_s = round(time.time() - t, 1)
        log(f"drawer loaded on {DEVICE} (HIP {hip}) in {self.load_s} s")

    def draw(self, job: dict) -> dict:
        self.start()
        self.proc.stdin.write(json.dumps(job) + "\n")
        self.proc.stdin.flush()
        while True:
            ln = self.proc.stdout.readline()
            if not ln:
                raise RuntimeError(f"the drawer died on {job['id']} (rc {self.proc.wait()})")
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("id") == job["id"]:
                break
        if r.get("error"):
            raise RuntimeError(f"{job['id']}: {r['error']}")
        self.times.append(r["seconds"])
        self.peak_gb = max(self.peak_gb, r.get("peak_gb") or 0)
        log(f"{job['id']}: {r['seconds']} s, peak {r.get('peak_gb')} GiB")
        return r

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self.lock_fd is not None:
            os.close(self.lock_fd)          # (releases the flock)
            self.lock_fd = None


# --- picture handling -------------------------------------------------------------------------------

def _edges(a):
    """Where the picture has detail: the object (the seamless background is smooth, shadows and light falloff too)."""
    import numpy as np
    g = a.astype(np.float32).mean(axis=2)
    e = np.zeros_like(g, dtype=bool)
    e[:, 1:] |= np.abs(np.diff(g, axis=1)) > 10
    e[1:, :] |= np.abs(np.diff(g, axis=0)) > 10
    return e


def _gutter(counts, lo: float = 0.3, hi: float = 0.7) -> int:
    """The emptiest line between lo and hi of the length (nearest the middle among the emptiest)."""
    import numpy as np
    n = len(counts)
    i0, i1 = int(lo * n), int(hi * n)
    c = np.convolve(counts[i0:i1], np.ones(9) / 9, mode="same")
    best = c.min()
    cand = np.nonzero(c <= best + 0.5)[0] + i0
    return int(cand[np.argmin(np.abs(cand - n / 2))])


def square_crop(im, margin: float = 0.06):
    """The object's box (by its edges) grown by `margin`, made square around the object's centre and padded with the
    background colour where the square leaves the picture; resized to SIZE. Returns (picture, fill fraction)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.convert("RGB"))
    ys, xs = np.nonzero(_edges(a))
    if len(xs) < 200:
        return im.resize((SIZE, SIZE), Image.LANCZOS), 0.0
    x0, x1 = np.percentile(xs, [0.3, 99.7])
    y0, y1 = np.percentile(ys, [0.3, 99.7])
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3), a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    bg = tuple(int(c) for c in np.median(border, axis=0))
    box = [round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)]
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), bg)
    canvas.paste(im.crop((max(0, box[0]), max(0, box[1]), min(im.width, box[2]), min(im.height, box[3]))),
                 (max(0, -box[0]), max(0, -box[1])))
    return canvas.resize((SIZE, SIZE), Image.LANCZOS), round(max(x1 - x0, y1 - y0) / side, 2)


def place(src: Path, dest: Path) -> None:
    """A single-view drawing as a view: kept as drawn (the drawer already fills the frame), only resized to SIZE."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    (im if im.size == (SIZE, SIZE) else im.resize((SIZE, SIZE), Image.LANCZOS)).save(dest)


def cut_sheet(src: Path, out_dir: Path) -> dict:
    """A 2x2 turnaround picture cut into its four views (front, side / top, three-quarter). The cuts go along the
    emptiest row and, in each half, the emptiest column near the middle (a view often reaches over the midline), and
    each view is cropped square around its object. Returns {view: path}."""
    import numpy as np
    from PIL import Image
    im = Image.open(src).convert("RGB")
    e = _edges(np.asarray(im))
    h, w = e.shape
    ry = _gutter(e.sum(axis=1))
    paths = {}
    for row, (y0, y1) in enumerate(((0, ry), (ry, h))):
        cx = _gutter(e[y0:y1].sum(axis=0))
        for col, (x0, x1) in enumerate(((0, cx), (cx, w))):
            v = VIEWS[row * 2 + col]
            pic, fill = square_crop(im.crop((x0, y0, x1, y1)))
            q = out_dir / f"{v}.png"
            pic.save(q)
            paths[v] = q
            log(f"{v}: cell {x0}-{x1} x {y0}-{y1}, object {fill:.0%} of its square")
    return paths


def make_sheet(views: dict, dest: Path) -> None:
    from PIL import Image, ImageDraw
    tile, lab = 640, 28
    sheet = Image.new("RGB", (2 * tile, 2 * (tile + lab)), (235, 235, 235))
    d = ImageDraw.Draw(sheet)
    for i, v in enumerate(VIEWS):
        x, y = (i % 2) * tile, (i // 2) * (tile + lab)
        sheet.paste(Image.open(views[v]).convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y + lab))
        d.text((x + 8, y + 7), v.replace("_", "-"), fill=(30, 30, 30))
    sheet.save(dest)


# --- prompts ------------------------------------------------------------------------------------------

def view_prompt(desc: str, style: str, view: str) -> str:
    return f"{VIEW_WORDS[view]} of {desc}. {style + '. ' if style else ''}{LOOK}"


def hero_follow_prompt(desc: str, style: str, view: str) -> str:
    return (f"The exact same object as in the reference picture, with the identical design, parts, colours, markings "
            f"and proportions, now shown in a {VIEW_WORDS[view][0].lower()}{VIEW_WORDS[view][1:]}. The object: {desc}. "
            f"{style + '. ' if style else ''}{LOOK}")


SOURCE_WORDS = ("the object shown in the reference pictures, redrawn clean and whole: keep its exact shape, parts, "
                "colours, markings and wear as the reference pictures show them, ignore their background, people and "
                "lighting, and invent only what they do not show, in the same style. The object")


def sheet_prompt(desc: str, style: str, sourced: bool = False) -> str:
    what = f"{SOURCE_WORDS}: {desc}" if sourced else desc
    return (f"Orthographic turnaround reference sheet of one single object for a 3D modeller: {what}. "
            f"{style + '. ' if style else ''}The same object shown four times in a two-by-two grid with wide empty "
            f"space between the views: top left a straight-on front view, top right an exact side view in profile, "
            f"bottom left a top view looking straight down, bottom right a three-quarter view from the front left. "
            f"Identical design, colours and proportions in all four views, each view the same scale. Plain seamless "
            f"light grey background, soft even lighting, no shadows, no text, no labels, no grid lines, no people.")


def material_prompt(material: str) -> str:
    return (f"Close-up macro photograph of a flat surface of {material}, the material filling the whole frame edge to "
            f"edge, seen straight on, soft even diffuse lighting, sharp fine detail of the surface texture, wear and "
            f"colour variation. No object edges, no background, no text, no people.")


# --- one asset ------------------------------------------------------------------------------------------

def extra_prompt(desc: str, style: str, view: str, sourced: bool, plain: bool = True) -> str:
    """An extra view drawn beside the sheet, with the sheet's views (and the sources) as reference pictures. `view`
    is a VIEW_WORDS key or free words for the view; plain=False leaves out the plain-background studio look (for a
    view into or out of the object, such as an interior)."""
    words = VIEW_WORDS.get(view, view)
    return (f"The exact same object as in the reference pictures, with the identical design, parts, colours, markings "
            f"and wear{' (the clean turnaround views are its design; the other pictures show it in use)' if sourced else ''}, "
            f"now shown in a {words[0].lower()}{words[1:]}. The object: {desc}. {style + '. ' if style else ''}"
            f"{LOOK if plain else 'No text, no labels, no people.'}")


def draw_asset(d: Drawer, aid: str, desc: str, style: str = "", material: str = "", recipe: str = "sheet",
               seed: int = 11, also: list | None = None, out_root: Path | None = None,
               sources: list | None = None, extras: list | None = None) -> Path:
    """sources: pictures of the object from elsewhere (a book's keyframes, a video frame): copied in as
    source_<n>_<name>.png and given to the sheet as reference pictures (sheet recipe only). extras: further views
    [{"name", "view", "refs": ["sources" | "view:<name>" | path, ...], "plain": true}], each drawn with those reference pictures."""
    out = (out_root or OUT) / aid
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    src = []
    for i, sp in enumerate(sources or ()):
        dst = out / f"source_{i + 1}_{Path(sp).stem}.png"
        dst.write_bytes(Path(sp).read_bytes())
        src.append(dst)
    size = [SIZE, SIZE]
    rows, views = [], {}
    t0 = time.time()

    def job(name, prompt, refs=(), sz=None):
        p = raw / f"{name}.png"
        p.unlink(missing_ok=True)
        j = {"id": f"{aid}_{name}", "prompt": prompt, "seed": seed, "path": str(p), "size": sz or size}
        if refs:
            j["refs"] = [str(r) for r in refs]
        r = d.draw(j)
        rows.append((name, prompt, [Path(x).name for x in refs], r["seconds"], j["size"]))
        return p

    if recipe == "extras":          # only the extras again, beside the set already there
        views = {v: out / f"{v}.png" for v in VIEWS}
        src = sorted(out.glob("source_*.png"))
        material = ""
    elif recipe == "sheet":
        p = job("turnaround", sheet_prompt(desc, style, bool(src)), refs=src, sz=[SIZE * 3 // 2 // 32 * 32] * 2)
        for v, q in cut_sheet(p, raw).items():
            views[v] = out / f"{v}.png"
            views[v].write_bytes(q.read_bytes())
    else:
        order = ("three_quarter", "front", "side", "top")
        hero = None
        for v in order:
            if recipe == "hero" and hero is not None:
                p = job(v, hero_follow_prompt(desc, style, v), refs=[hero])
            else:
                p = job(v, view_prompt(desc, style, v))
            if v == "three_quarter":
                hero = p
            views[v] = out / f"{v}.png"
            place(p, views[v])
    if recipe != "extras":
        make_sheet(views, out / "sheet.png")
    for x in extras or ():
        refs = []
        for r in x.get("refs", ["view:three_quarter"]):
            refs += src if r == "sources" else [views[r[5:]]] if r.startswith("view:") else [Path(r)]
        p = job(x["name"], extra_prompt(desc, style, x.get("view", x["name"]), bool(src), x.get("plain", True)),
                refs=refs)
        views[x["name"]] = out / f"{x['name']}.png"
        place(p, views[x["name"]])
    if material:
        mp = job("material", material_prompt(material))
        (out / "material.png").write_bytes(mp.read_bytes())

    if recipe == "extras":
        old = (out / "notes.md").read_text().split("## How far the views agree")
        new = []
        for name, prompt, refs, sec, sz in rows:
            new += [f"### {name}, redrawn ({sz[0]}x{sz[1]}, {sec} s, seed {seed}"
                    f"{', references: ' + ', '.join(refs) if refs else ''})", "", prompt, ""]
        (out / "notes.md").write_text(old[0] + "\n".join(new) + "\n## How far the views agree" + "".join(old[1:]))
        log(f"{aid}: extras redrawn in {out}")
        return out
    notes = [f"# Reference set: {aid}", "",
             f"- description: {desc}", f"- style: {style or '(none)'}", f"- material: {material or '(none)'}",
             f"- recipe: {recipe}", f"- seed: {seed}", f"- picture size: {size[0]}x{size[1]} "
             f"(each view square; raw drawings and sheet cells in raw/)",
             f"- drawer: tools/qwen_image.py --serve (Qwen-Image-2.1 + 4-step Lightning) on {DEVICE}; "
             f"{time.time() - t0:.0f} s for the set", "", "## Prompts", ""]
    for name, prompt, refs, sec, sz in rows:
        notes += [f"### {name} ({sz[0]}x{sz[1]}, {sec} s{', references: ' + ', '.join(refs) if refs else ''})", "",
                  prompt, ""]
    notes += ["## How far the views agree", "", "(not reviewed yet)", ""]
    (out / "notes.md").write_text("\n".join(notes))
    for other in also or ():
        link = (out_root or OUT) / other
        if other != aid and not link.exists():
            link.symlink_to(aid)
            log(f"{other} -> {aid}")
    log(f"{aid}: set written to {out}")
    return out


def set_note(aid: str, text: str) -> None:
    p = OUT / aid / "notes.md"
    s = p.read_text()
    head = s.split("## How far the views agree")[0]
    p.write_text(head + "## How far the views agree\n\n" + text.strip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("draw")
    a1.add_argument("asset_id")
    a1.add_argument("--desc", required=True)
    a1.add_argument("--style", default="")
    a1.add_argument("--material", default="", help="also draw material.png: a close-up of this surface")
    a1.add_argument("--recipe", choices=("hero", "separate", "sheet", "extras"), default="sheet")
    a1.add_argument("--seed", type=int, default=11)
    a1.add_argument("--also", nargs="*", default=[], help="other ids to link to this set (a modeller's id)")
    a1.add_argument("--out", default="", help="write under this folder instead of REFS_DIR")
    a1.add_argument("--sources", nargs="*", default=[], help="pictures the object is drawn from (reference pictures)")
    a2 = sub.add_parser("batch")
    a2.add_argument("jobs", help='JSON list of {"id", "desc", "style", "material", "recipe", "seed", "also", '
                                 '"sources", "extras"}')
    a2.add_argument("--out", default="")
    a3 = sub.add_parser("note")
    a3.add_argument("asset_id")
    a3.add_argument("text")
    a4 = sub.add_parser("picture", help="one plain picture (a backdrop's view): <id>/picture.png")
    a4.add_argument("asset_id")
    a4.add_argument("--desc", required=True)
    a4.add_argument("--size", nargs=2, type=int, default=[1664, 832])
    a4.add_argument("--seed", type=int, default=11)
    a4.add_argument("--sources", nargs="*", default=[], help="pictures it is drawn from (reference pictures)")
    a = ap.parse_args()
    if a.cmd == "picture":
        dest = OUT / a.asset_id / "picture.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        d = Drawer()
        try:
            d.draw({"id": a.asset_id, "prompt": a.desc, "seed": a.seed, "path": str(dest), "size": list(a.size),
                    **({"refs": list(a.sources)} if a.sources else {})})
        finally:
            d.close()
        return 0 if dest.exists() else 1
    if a.cmd == "note":
        set_note(a.asset_id, a.text)
        return 0
    items = json.loads(Path(a.jobs).read_text()) if a.cmd == "batch" else [
        {"id": a.asset_id, "desc": a.desc, "style": a.style, "material": a.material, "recipe": a.recipe,
         "seed": a.seed, "also": a.also, "sources": a.sources}]
    out_root = Path(a.out) if a.out else None
    d = Drawer()
    failed = 0
    try:
        for it in items:
            try:
                draw_asset(d, it["id"], it["desc"], it.get("style", ""), it.get("material", ""),
                           it.get("recipe", "sheet"), int(it.get("seed", 11)), it.get("also"), out_root,
                           it.get("sources"), it.get("extras"))
            except RuntimeError as exc:
                log(f"!!! {it['id']}: {exc}")
                failed += 1
                if d.proc and d.proc.poll() is not None:
                    d.proc = None
    finally:
        d.close()
    if d.times:
        log(f"{len(d.times)} pictures, {sum(d.times) / len(d.times):.1f} s each on average "
            f"(load {d.load_s} s, peak {d.peak_gb} GiB on {DEVICE})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
