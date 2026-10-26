"""The contact view of the staged moments: every handled prop close up, see-through, with its contact parts in colour
(user 10-22: "print out the 3D models for those interactions"). CPU/OpenGL only (Blender Workbench), seconds a view.

    python3 tools/contacts_view.py <book> <chapter stem> [moment ids...] [--out sheet.jpg]

Without ids: one moment per handled prop (holds, contacts, recipes) plus the moments whose set has a jointed prop
(the bellows). Each moment: its start / middle / end frame (a clip) from the staging camera's side, square on, and
from above. Colours: RED = the grip a hand must close on, BLUE = the hands (fists), YELLOW = the prop's contact
proxy (striking face, blade, jaws...), grey = the meshes (see-through). The worst check numbers of the moment
(render/checks.json) are printed under its row; a moment with a handled prop and no grip in reach is flagged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blockout_rig as rig  # noqa: E402

BLENDER = os.environ.get("BLOCKOUT_BLENDER", os.path.expanduser("~/opt/blender/current/blender"))
COLORS = {"grip": (0.95, 0.1, 0.1), "fist": (0.15, 0.35, 1.0), "prop": (1.0, 0.85, 0.1), "body": (0.6, 0.6, 0.6)}
SZ = 480


def pick(cd: Path, plan: dict) -> list:
    """One moment per handled prop and per jointed set prop (the first staged one)."""
    seen, out = set(), []
    props = {}
    pj = cd.parent / "stage" / "props" / "props.json"
    if pj.exists():
        reg = json.loads(pj.read_text())
        props = reg.get("props", reg)
    for e in plan["entries"]:
        if not e.get("shot") or e.get("route") != "stage":
            continue
        s = json.loads(Path(e["shot"]).read_text())
        names = set()
        for f in s.get("figures", []):
            names |= {p for p in (f.get("holds") or {}).values() if isinstance(p, str)}
        names |= {c.get("prop") for c in s.get("contacts") or [] if c.get("prop")}
        names |= {p.get("prop") for p in (s.get("set") or {}).get("props", [])
                  if isinstance(props.get(p.get("prop")), dict) and props[p.get("prop")].get("joints")}
        new = names - seen
        if new and (cd / "flow3d" / "stage" / e["id"] / "render" / "scene.json").exists():
            seen |= new
            out.append((e["id"], sorted(new)))
    return out


def views(frame: dict) -> list:
    """Three cameras on the hands + grips (or the props when nobody holds one): the staging side, square on, above."""
    pts = [np.asarray(p.get("c") or p.get("a"), float) for p in frame["prims"]
           if p.get("tag") in ("grip", "fist") and (p.get("c") or p.get("a")) is not None]
    if not pts:
        pts = [np.asarray(p["M"], float)[:3, 3] for p in frame["prims"] if p.get("t") == "propmesh"]
    if not pts:
        return []
    c = np.mean(pts, 0)
    span = float(np.ptp(np.array(pts), 0).max()) if len(pts) > 1 else 0.3
    dist = max(0.9, 1.5 * span + 0.5)
    cam = np.asarray(frame["camera"]["pos"], float)
    d0 = cam - c
    d0[2] = 0
    d0 = d0 / (np.linalg.norm(d0) or 1)
    side = np.array([-d0[1], d0[0], 0.0])
    out = []
    for lab, d in (("staging side", d0 + [0, 0, 0.25]), ("square on", side + [0, 0, 0.2]), ("above", d0 * 0.35 + [0, 0, 1])):
        d = d / np.linalg.norm(d)
        pos = c + d * dist
        out.append((lab, {"pos": pos.tolist(), "R": rig.look_at(pos, c).tolist(), "lens": 35.0}))
    return out


def render(scene: dict, idxs: list, tmp: Path, tag: str) -> list:
    frames, labels = [], []
    for i in idxs:
        fr = scene["frames"][i]
        for lab, cam in views(fr):
            n = f"{tag}_{i}_{lab.replace(' ', '_')}"
            frames.append({**{k: v for k, v in fr.items() if k not in ("layers", "blend")},
                           "name": n, "camera": cam, "size": [SZ, SZ], "show_proxies": True, "colors": COLORS,
                           "clay": str(tmp / f"{n}.png"), "depth": str(tmp / f".{n}.npy")})
            labels.append((str(tmp / f"{n}.png"), f"frame {i} {lab}", i))
    if not frames:
        return []
    sp = tmp / f"{tag}.json"
    sp.write_text(json.dumps({"size": [SZ, SZ], "samples": 1, "engine": "workbench", "xray": 0.45, "frames": frames}))
    r = subprocess.run([BLENDER, "--background", "--factory-startup", "-t", "8", "--python",
                        str(HERE / "blockout_scene.py"), "--", str(sp)], capture_output=True, text=True, timeout=900)
    if r.returncode:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
    return [x for x in labels if Path(x[0]).exists()]


def worst(rd: Path) -> str:
    cj = rd / "checks.json"
    if not cj.exists():
        return "no checks"
    d = json.loads(cj.read_text())
    w = d.get("worst") or {}
    keys = [k for k in w if any(t in k for t in ("grip", "fist", "gap", "pen", "face", "handle", "contact"))]
    return f"violations {d.get('violations', 0)}  " + "  ".join(f"{k} {w[k]:.1f}" for k in keys[:5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("stem")
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    book = Path(a.book).expanduser().resolve()
    cd = book / "work" / "video" / a.stem
    plan = json.loads((cd / "flow3d" / "plan.json").read_text())
    todo = [(i, []) for i in a.ids] if a.ids else pick(cd, plan)
    tmp = Path(tempfile.mkdtemp(prefix="contacts_"))
    rows = []
    for mid, props in todo:
        rd = cd / "flow3d" / "stage" / mid / "render"
        scene = json.loads((rd / "scene.json").read_text())
        n = len(scene["frames"])
        idxs = [0, n // 2, n - 1] if n > 2 else list(range(n))
        imgs = render(scene, idxs, tmp, mid)
        print(f"{mid} {props}: {len(imgs)} view(s)", flush=True)
        rows.append((f"{mid}  {', '.join(props)}  |  {worst(rd)}", imgs))
    lines = []                                       # one row per frame: its three views
    for title, imgs in rows:
        by = {}
        for p, lab, i in imgs:
            by.setdefault(i, []).append((p, lab))
        lines += [(title if k == 0 else "", v) for k, v in enumerate(by.values())] or [(title + "  (nothing rendered)", [])]
    cols = max((len(v) for _, v in lines), default=1)
    sheet = Image.new("RGB", (cols * SZ, len(lines) * (SZ + 22) + 22 * len(rows)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    y = 0
    for title, imgs in lines:
        if title:
            d.text((8, y + 5), title, fill=(255, 255, 0))
            y += 22
        for ci, (p, lab) in enumerate(imgs):
            sheet.paste(Image.open(p).convert("RGB").resize((SZ, SZ)), (ci * SZ, y))
            d.text((ci * SZ + 6, y + 4), lab, fill=(255, 255, 255))
        y += SZ + 22
    out = Path(a.out) if a.out else book / "out" / f"{book.name}_{a.stem}_contacts.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
