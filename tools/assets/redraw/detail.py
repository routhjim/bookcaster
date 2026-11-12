"""Detail passes on an accepted key: small people redrawn in their own enlarged crop, look-alike pieces in theirs.

    python tools/assets/redraw/detail.py plan  RUN.json [--accepted accepted.json] [--dry] [--pieces flagged|pairs|off]
    python tools/assets/redraw/detail.py paste RUN.json [--accepted accepted.json]
    python tools/assets/redraw/detail.py block  RUN.json      (off by default: colour-blocked stills for a run)
    python tools/assets/redraw/detail.py lights RUN.json      (off by default: engine pixels for tiny lit sources)
    python tools/assets/redraw/detail.py depth IN.png OUT.png [--inverse] [--clip 2 98]

Why (11-12): the image model's latent grid is 1/16 of the picture, so at 1152x640 a head a fifth of the frame's
height is about 8 cells. Below about a third of the frame's height a person's identity, outfit and facial hair
fall apart whatever the character model's weight, and a veil or a moustache is decided by chance. A second draw
of that person alone, in a head-and-shoulders crop enlarged to the model's comfortable size, gives the head the
cells it has in a close shot; the result is colour-matched to its surroundings and pasted back through the
person's own engine mask, so the room, the staging and the other people are not touched.

"plan" writes, per accepted key, the crop inputs and one image job per person who needs a pass (and per piece,
see below) into <out>/detail/ and prints the job file; nothing is drawn here. "paste" puts the drawn crops back,
farthest person first, nearer people's masks protecting them, and writes <out>/detail/<key id>__d.png (the key
itself is never overwritten); the
caller runs the usual checks on it and keeps it only when no check got worse (not_worse()).

Pieces: a large piece the look check found drawn as another piece ("flagged", the default: it costs nothing on
keys that pass) or every piece of a look-alike pair in frame ("pairs") is redrawn in its own crop through its
engine mask with words naming only that piece. Off by default and for a box test first: colour_block() (the
still's pieces pulled toward their own base colour before the first draw), prep_depth() (inverse depth,
percentile-clipped, for the control at its own recommended strength) and composite_lit() (the engine's own
bright pixels for lit sources too small to draw).

Geometry, colour matching and pasting are pure functions on numpy arrays (tools/tests/test_redraw_detail_1112.py).
The image server already takes everything a pass needs (init picture, per-pixel strength mask, a prepared depth
control, one character model, references); it has no true inpainting (the control branch's keep-mask is not
wired), so the surroundings are kept by the paste, not by the draw.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# A person's head (top to chin) as a share of the frame's height.
HEAD_MAX = _env("REDRAW_DETAIL_HEAD_MAX", 0.33)   # at or above: the base draw holds them (held at 0.36-0.41, lost at 0.31)
HEAD_MIN = _env("REDRAW_DETAIL_HEAD_MIN", 0.05)   # below: a sliver, nothing a crop could fix but colour
CROP_FACTOR = _env("REDRAW_DETAIL_CROP", 3.0)     # crop side = head height x this: the head a third of the crop
CROP_DROP = _env("REDRAW_DETAIL_DROP", 0.45)      # crop centre this many head heights under the head's middle (shoulders in)
SIZE = int(_env("REDRAW_DETAIL_SIZE", 1024))      # the crop enlarged to this square (a multiple of 32)
STRENGTH = _env("REDRAW_DETAIL_STRENGTH", 0.5)    # img2img strength inside the person's mask
STRENGTH_BG = _env("REDRAW_DETAIL_STRENGTH_BG", 0.1)
FEATHER = _env("REDRAW_DETAIL_FEATHER", 0.012)    # paste edge, of the crop's side
GROW = _env("REDRAW_DETAIL_GROW", 0.01)           # the paste mask grown by this share of the crop's side first
VISIBLE_MIN = _env("REDRAW_DETAIL_VISIBLE", 0.25)  # a person with less of their body in view is left alone
PIECE_PAD = _env("REDRAW_PIECE_PAD", 1.25)        # a piece's crop = its box x this
PIECE_STRENGTH = _env("REDRAW_PIECE_STRENGTH", 0.55)
PIECE_AREA = _env("REDRAW_PIECE_AREA", 0.01)      # a piece smaller than this share of the frame is not redrawn
LIT_SMALL = _env("REDRAW_LIT_SMALL", 0.004)       # a lit source under this share of the frame: the engine's pixels
STEP_S_PER_MP = _env("REDRAW_DETAIL_S_PER_MP", 0.0)   # 0: estimated from the run's own log when asked


# ---------------------------------------------------------------- colour

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 or 0..1 sRGB (..., 3) -> CIE Lab (D65)."""
    c = rgb.astype(np.float64)
    if c.max() > 1.0:
        c = c / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """CIE Lab -> uint8 sRGB."""
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) * 27 / 24389) * np.array([0.95047, 1.0, 1.08883])
    m = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    c = np.clip(xyz @ m.T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.rint(c * 255), 0, 255).astype(np.uint8)


def colour_match(new: np.ndarray, old: np.ndarray, ring: np.ndarray, spread: tuple = (0.6, 1.6)) -> np.ndarray:
    """The redrawn crop brought to the key's light: mean and spread per Lab channel measured over the ring (the
    part of the crop OUTSIDE the paste mask, which both pictures share) and applied to the whole redrawn crop.
    `ring` is a boolean (h, w); too small a ring (under 2 % of the crop) leaves the crop as drawn."""
    if ring.sum() < 0.02 * ring.size:
        return new
    a, b = srgb_to_lab(new), srgb_to_lab(old)
    out = a.copy()
    for ch in range(3):
        ma, mb = a[..., ch][ring].mean(), b[..., ch][ring].mean()
        sa, sb = a[..., ch][ring].std(), b[..., ch][ring].std()
        g = min(spread[1], max(spread[0], sb / sa)) if sa > 1e-6 else 1.0
        out[..., ch] = (a[..., ch] - ma) * g + mb
    return lab_to_srgb(out)


# ---------------------------------------------------------------- masks and pasting

def _blur(m: np.ndarray, r: float) -> np.ndarray:
    """A separable box blur run three times (close to a Gaussian of sigma r/sqrt(3)... good enough for an edge)."""
    r = int(round(r))
    if r < 1:
        return m.astype(np.float64)
    out = m.astype(np.float64)
    k = 2 * r + 1
    for _ in range(3):
        for ax in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[ax] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=ax)
            out = (np.take(c, range(k, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - k), axis=ax)) / k
    return out


def _grow(m: np.ndarray, r: int) -> np.ndarray:
    """A boolean mask dilated by r pixels (square), by a running maximum."""
    if r < 1:
        return m
    out = m.copy()
    for ax in (0, 1):
        acc = out.copy()
        for s in range(1, r + 1):
            acc |= np.roll(out, s, ax) | np.roll(out, -s, ax)
        out = acc
    return out


def paste_alpha(mask: np.ndarray, protect: np.ndarray | None = None, grow: float = GROW, feather: float = FEATHER) -> np.ndarray:
    """The paste weight (0..1) for a redrawn crop: the person's own mask, grown a little and feathered, zero
    wherever a nearer person's mask lies (`protect`) and fading to zero at the crop's own border, so a crop cut
    by the frame or by its square never leaves a seam."""
    side = max(mask.shape)
    a = _blur(_grow(mask.astype(bool), int(round(grow * side))), feather * side)
    a = np.clip(a, 0, 1)
    if protect is not None:
        a = a * (1 - np.clip(_blur(_grow(protect.astype(bool), int(round(grow * side))), feather * side / 2), 0, 1))
    b = max(1, int(round(feather * side)))
    h, w = a.shape
    yy, xx = np.indices((h, w))
    edge = np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx))
    return a * np.clip(edge / b, 0, 1)


def feather_paste(base: np.ndarray, crop: np.ndarray, box: tuple, alpha: np.ndarray) -> np.ndarray:
    """`crop` (already at the box's pixel size) laid into `base` at `box` = (x0, y0, x1, y1) with weight `alpha`."""
    x0, y0, x1, y1 = box
    out = base.astype(np.float64).copy()
    a = alpha[..., None]
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + crop.astype(np.float64) * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- geometry

def head_of(j2: dict) -> tuple | None:
    """(centre x, top y, chin y) of a head in 0..1 frame units from the person's projected joints, or None."""
    if not j2 or "chin" not in j2 or "head_top" not in j2:
        return None
    xs = [j2[k][0] for k in ("FACIAL_L_Eye", "FACIAL_R_Eye", "head", "chin", "head_top") if k in j2]
    return float(np.mean(xs)), float(j2["head_top"][1]), float(j2["chin"][1])


def size_class(hh: float | None, visible: float = 1.0) -> str:
    """"large" (the base draw holds them), "detail" (gets a pass), "sliver" or "unknown" (left alone)."""
    if hh is None or hh <= 0:
        return "unknown"
    if visible < VISIBLE_MIN or hh < HEAD_MIN:
        return "sliver"
    return "large" if hh >= HEAD_MAX else "detail"


def crop_box(cx: float, top: float, chin: float, w: int, h: int, factor: float = CROP_FACTOR, drop: float = CROP_DROP) -> tuple:
    """A square head-and-shoulders crop in pixels (x0, y0, x1, y1): side = head height x factor, centred `drop`
    head heights under the head's middle; slid back inside the frame when it would cross an edge, and shrunk to
    the frame's shorter side when it is larger than the frame. Never leaves the frame."""
    hh = (chin - top) * h
    side = int(round(min(factor * hh, w, h)))
    side = max(32, side - side % 2)
    mx, my = cx * w, (top + chin) / 2 * h + drop * hh
    x0 = int(round(min(max(mx - side / 2, 0), w - side)))
    y0 = int(round(min(max(my - side / 2, 0), h - side)))
    return x0, y0, x0 + side, y0 + side


def box_px(b: list, w: int, h: int, pad: float = 1.0, mult: int = 32, long_side: int = SIZE) -> tuple:
    """A normalised box grown by `pad` about its centre, clamped to the frame -> ((x0, y0, x1, y1) px, (W, H) to
    enlarge it to: its long side at `long_side`, both sides multiples of `mult`)."""
    cx, cy, bw, bh = (b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h, (b[2] - b[0]) * w * pad, (b[3] - b[1]) * h * pad
    x0, y0 = int(max(0, math.floor(cx - bw / 2))), int(max(0, math.floor(cy - bh / 2)))
    x1, y1 = int(min(w, math.ceil(cx + bw / 2))), int(min(h, math.ceil(cy + bh / 2)))
    s = long_side / max(x1 - x0, y1 - y0)
    tw, th = (max(mult, int(round((x1 - x0) * s / mult)) * mult), max(mult, int(round((y1 - y0) * s / mult)) * mult))
    return (x0, y0, x1, y1), (tw, th)


def paint_order(people: list) -> list:
    """People farthest from the camera first, so a nearer person's pass is pasted last, over theirs. Each item
    needs "key" and "distance_m" (missing distances sort as nearest: never painted over)."""
    return sorted(people, key=lambda p: -(p.get("distance_m") if p.get("distance_m") is not None else -1.0))


def nearer_than(p: dict, people: list) -> list:
    """The keys of the people nearer the camera than `p` (their masks protect them from p's paste)."""
    d = p.get("distance_m")
    return [q["key"] for q in people if q["key"] != p["key"] and d is not None and q.get("distance_m") is not None
            and q["distance_m"] < d]


def not_worse(base: dict, new: dict) -> bool:
    """True when the detail key fails nothing the base key passed. Both are {who or "set": [failed checks]}."""
    return all(set(new.get(k) or []) <= set(base.get(k) or []) for k in set(base) | set(new))


# ---------------------------------------------------------------- control and lights (off by default)

def prep_depth(d: np.ndarray, inverse: bool = True, clip: tuple = (2.0, 98.0), mask: np.ndarray | None = None) -> np.ndarray:
    """A depth picture stretched for the control: optionally inverse depth (near things get the range), clipped at
    the given percentiles (taken inside `mask` when given, so a crop's own relief uses the full range) -> uint8.
    Input: any depth-like array where LARGER = FARTHER (metres or a grey picture with far = white)."""
    x = d.astype(np.float64)
    if x.ndim == 3:
        x = x[..., 0]
    if inverse:
        x = 1.0 / np.maximum(x - x.min() + 0.05 * (np.ptp(x) + 1e-9), 1e-9)       # near = large
    else:
        x = x.max() - x                                                            # near = large, linear
    sel = x[mask] if mask is not None and mask.any() else x
    lo, hi = np.percentile(sel, clip[0]), np.percentile(sel, clip[1])
    return np.clip(np.rint((x - lo) / max(hi - lo, 1e-9) * 255), 0, 255).astype(np.uint8)


def composite_lit(key: np.ndarray, still: np.ndarray, lit: np.ndarray, rise: float = 1.5, floor: float = 0.55) -> np.ndarray:
    """The engine's own bright pixels of a lit source laid onto the key: inside the source's mask grown upward
    `rise` times its height (the flame stands above the wick), wherever the engine still is brighter than `floor`
    of its own brightest there, feathered. For sources too small for the model to draw."""
    ys, xs = np.nonzero(lit)
    if not len(ys):
        return key
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    y0 = max(0, int(y0 - rise * (y1 - y0)))
    reg = np.zeros(lit.shape, bool)
    reg[y0:y1, max(0, x0 - 2):x1 + 2] = True
    lum = still.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    top = lum[reg].max()
    a = np.clip((lum - floor * top) / max((1 - floor) * top, 1e-9), 0, 1) * reg
    a = np.clip(np.maximum(a, _blur(a, 1)), 0, 1)[..., None]      # the core kept whole, a soft rim round it
    return np.clip(np.rint(key * (1 - a) + still * a), 0, 255).astype(np.uint8)


COLOURS = json.loads((HERE / "detail_colours.json").read_text()) if (HERE / "detail_colours.json").exists() else {}


def base_colour(look: str) -> tuple | None:
    """The first colour word of a piece's look line -> Lab, from the data table (longest words first)."""
    t = look.lower()
    for w in sorted(COLOURS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}\b", t):
            return tuple(COLOURS[w])
    return None


def colour_block(still: np.ndarray, pieces: list, amount: float = 0.6) -> np.ndarray:
    """The engine still with each large piece pulled toward its own base colour: inside each (mask, Lab) the
    pixels' a/b move `amount` of the way to the target's and their mean lightness the same way, their own
    shading kept. So two look-alike pieces differ in the source the first draw starts from."""
    lab = srgb_to_lab(still)
    for mask, target in pieces:
        m = mask.astype(bool)
        if not m.any() or target is None:
            continue
        for ch in (1, 2):
            lab[..., ch][m] += amount * (target[ch] - lab[..., ch][m].mean())
        lab[..., 0][m] += amount * (target[0] - lab[..., 0][m].mean())
    return lab_to_srgb(lab)


# ---------------------------------------------------------------- pieces

def family(name: str) -> str:
    """A set item's kind family: its name without a repeat number and without a leading qualifier the data table
    lists (so "armchair_2" and "wingback_armchair" are both "armchair")."""
    n = re.sub(r"_\d+$", "", name.lower())
    for q in COLOURS.get("__qualifiers__", []):
        n = n.replace(q + "_", "")
    return n


def _area(t: dict) -> float:
    b = t.get("box") or [0, 0, 0, 0]
    return t.get("mask_area") or (b[2] - b[0]) * (b[3] - b[1]) * (t.get("visible_frac") or 1.0)


def lookalike_pairs(things: list) -> list:
    """Names of the large pieces in frame that share a kind family with another large piece in frame."""
    big = [t for t in things if t.get("large") and _area(t) >= PIECE_AREA]
    fam: dict = {}
    for t in big:
        fam.setdefault(family(t["name"]), []).append(t["name"])
    return sorted(n for v in fam.values() if len(v) > 1 for n in v)


# ---------------------------------------------------------------- jobs

def _img(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def _mask(path, size=None):
    from PIL import Image
    im = Image.open(path).convert("L")
    if size:
        im = im.resize(size, Image.NEAREST)
    return np.asarray(im) > 127


def _save(arr, path, size=None, mode=None):
    from PIL import Image
    im = Image.fromarray(arr if arr.dtype == np.uint8 else (arr.astype(np.uint8) * 255))
    if mode:
        im = im.convert(mode)
    if size:
        im = im.resize(size, Image.LANCZOS if (mode or im.mode) == "RGB" else Image.BILINEAR)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def piece_mask(path, nbox: list, w: int, h: int) -> np.ndarray:
    """A piece's engine mask at the frame's size; without one, its box as a rectangle (people protect themselves)."""
    if path and Path(path).exists():
        return _mask(path, (w, h))
    m = np.zeros((h, w), bool)
    m[int(nbox[1] * h):int(math.ceil(nbox[3] * h)), int(nbox[0] * w):int(math.ceil(nbox[2] * w))] = True
    return m


def person_prompt(who: dict, look: str, wear_add: str | None, facing: str, style: str) -> str:
    """Words for one person alone, from their own facts only."""
    wear = "; ".join(x for x in (who.get("wear", ""), look, wear_add or "") if x)
    trig = f" ({who['trigger']})" if who.get("trigger") else ""
    return (f"A photographed frame from a period film. A head-and-shoulders view of one {who.get('noun', 'man')}{trig}: "
            f"{wear}. {facing} Nobody else is in the picture. {style}").replace("  ", " ").strip()


def plan(run: dict, accepted: dict, dry: bool, pieces: str) -> tuple:
    """-> (jobs, rows): image jobs for every pass, and one row per key {id, people: {key: class}, pieces: [...]}."""
    sys.path.insert(0, str(HERE))
    import redraw as R
    out = Path(run["out"]) / "detail"
    meta = json.loads((Path(run["out"]) / "meta.json").read_text())
    truths = {Path(sd).name: (Path(sd), json.loads((Path(sd) / "truth.json").read_text())) for sd in run["scenes"]}
    one_w = float((run.get("lora_w") or {}).get("1", 0.8))
    cast = run.get("cast") or {}
    jobs, rows = [], []
    for kid, rec in accepted.items():
        m = meta.get(rec.get("id", kid)) or meta.get(kid)
        if not m:
            continue
        sd, truth = truths[m["scene"]]
        sh = truth["shots"][m["cam"]]
        g = sd / "guides"
        wear = R.scene_wear(sd)
        ppl = [dict(p, key=k) for k, p in sh["people"].items() if any(q["key"] == k for q in m["people"])]
        row = {"id": kid, "path": rec["path"], "people": {}, "pieces": []}
        key_im = None if dry else _img(rec["path"])
        H, W = (640, 1152) if key_im is None else key_im.shape[:2]
        for p in paint_order(ppl):
            hd = head_of(p.get("joints2d") or {})
            cls = size_class((hd[2] - hd[1]) if hd else None, p.get("visible_frac", 1.0))
            row["people"][p["key"]] = cls
            if cls != "detail":
                continue
            box = crop_box(hd[0], hd[1], hd[2], W, H)
            who = dict(cast.get(p["key"]) or {})
            wv = wear.get(p["key"]) or {}
            look, _ = R.look_words(R.entry_facts(wv.get("id")), "her" if who.get("noun") == "woman" else "his")
            pos = "her" if who.get("noun") == "woman" else "his"
            facing = ("She " if pos == "her" else "He ") + R.facing_words(p.get("head_to_camera_deg", 0.0), pos) + "."
            d = out / kid
            j = {"id": f"{kid}__d_{p['key']}", "seed": int(run.get("seed", 7)), "size": [SIZE, SIZE],
                 "path": str(d / f"{p['key']}_draw.png"),
                 "prompt": person_prompt(who, look, wv.get("wear_add"), facing, run.get("look", "")),
                 "refs": [x for x in (who.get("sheet"), None if wv.get("face_covered") else who.get("face_ref")) if x],
                 "loras": [[who["lora"], one_w, [0.0, 0.0, 1.0, 1.0]]] if who.get("lora") else [],
                 "init": {"image": str(d / f"{p['key']}_in.png"), "strength": STRENGTH,
                          "mask": str(d / f"{p['key']}_mask.png"), "strength_bg": STRENGTH_BG},
                 "control": {"image": str(d / f"{p['key']}_depth.png"), "type": "depth",
                             "scale": float((run.get("control") or {}).get("scale", 0.8)), "prepared": True},
                 "_detail": {"key": kid, "who": p["key"], "box": box, "protect": nearer_than(p, ppl),
                             "mask_src": str(g / f"{m['cam']}_mask_{p['key']}.png"),
                             "masks": {q["key"]: str(g / f"{m['cam']}_mask_{q['key']}.png") for q in ppl}}}
            if not dry:
                x0, y0, x1, y1 = box
                _save(key_im[y0:y1, x0:x1], j["init"]["image"], (SIZE, SIZE), "RGB")
                pm = _mask(j["_detail"]["mask_src"], (W, H))
                _save(pm[y0:y1, x0:x1], j["init"]["mask"], (SIZE, SIZE), "L")
                dp = np.asarray(__import__("PIL.Image", fromlist=["x"]).open(g / f"{m['cam']}_depth.png").convert("L").resize((W, H)))
                _save(prep_depth(255 - dp[y0:y1, x0:x1].astype(np.float64), inverse=False, mask=pm[y0:y1, x0:x1])
                      if os.environ.get("REDRAW_DETAIL_DEPTH_LOCAL", "1") == "1" else dp[y0:y1, x0:x1],
                      j["control"]["image"], (SIZE, SIZE), "L")
            jobs.append(j)
        want = set()
        if pieces == "pairs":
            want = set(lookalike_pairs(sh.get("things") or []))
        elif pieces == "flagged":
            want = {x[5:].strip() for x in (rec.get("fails") or {}).get("set", []) if x.startswith("look ")}
        for t in sh.get("things") or []:
            area = t.get("mask_area") or (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]) * (t.get("visible_frac") or 1.0)
            if t["name"] not in want or area < PIECE_AREA:
                continue
            box, size = box_px(t["box"], W, H, PIECE_PAD)
            d = out / kid
            j = {"id": f"{kid}__p_{t['name']}", "seed": int(run.get("seed", 7)), "size": list(size),
                 "path": str(d / f"piece_{t['name']}_draw.png"),
                 "prompt": f"A photographed frame from a period film. {t['words'][0].upper() + t['words'][1:]}, "
                           f"{t.get('place') or 'in the room'}. Only this one piece of furniture fills the picture. "
                           f"{run.get('look', '')}".strip(),
                 "refs": [], "loras": [],
                 "init": {"image": str(d / f"piece_{t['name']}_in.png"), "strength": PIECE_STRENGTH,
                          "mask": str(d / f"piece_{t['name']}_mask.png"), "strength_bg": STRENGTH_BG},
                 "control": {"image": str(d / f"piece_{t['name']}_depth.png"), "type": "depth",
                             "scale": float((run.get("control") or {}).get("scale", 0.8)), "prepared": True},
                 "_detail": {"key": kid, "piece": t["name"], "box": box, "mask_src": t.get("mask"), "nbox": t["box"],
                             "protect": [q["key"] for q in ppl],
                             "masks": {q["key"]: str(g / f"{m['cam']}_mask_{q['key']}.png") for q in ppl}}}
            if not dry:
                x0, y0, x1, y1 = box
                _save(key_im[y0:y1, x0:x1], j["init"]["image"], size, "RGB")
                _save(piece_mask(t.get("mask"), t["box"], W, H)[y0:y1, x0:x1], j["init"]["mask"], size, "L")
                dp = np.asarray(__import__("PIL.Image", fromlist=["x"]).open(g / f"{m['cam']}_depth.png").convert("L").resize((W, H)))
                _save(dp[y0:y1, x0:x1], j["control"]["image"], size, "L")
            row["pieces"].append(t["name"])
            jobs.append(j)
        rows.append(row)
    return jobs, rows


def paste(jobs: list, accepted: dict, out_dir) -> dict:
    """Every drawn crop of a key laid back onto it: pieces first, then people farthest first (the jobs' order).
    -> {key id: path of <key>__d.png} for keys with at least one drawn crop."""
    from PIL import Image
    done = {}
    by_key: dict = {}
    for j in jobs:
        by_key.setdefault(j["_detail"]["key"], []).append(j)
    for kid, js in by_key.items():
        base = _img(accepted[kid]["path"])
        H, W = base.shape[:2]
        n = 0
        for j in sorted(js, key=lambda j: 0 if "piece" in j["_detail"] else 1):
            if not Path(j["path"]).exists():
                continue
            dt = j["_detail"]
            x0, y0, x1, y1 = dt["box"]
            new = np.asarray(Image.open(j["path"]).convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS))
            m = (piece_mask(dt.get("mask_src"), dt["nbox"], W, H) if "piece" in dt else _mask(dt["mask_src"], (W, H)))[y0:y1, x0:x1]
            prot = np.zeros_like(m)
            for k in dt["protect"]:
                if Path(dt["masks"][k]).exists():
                    prot |= _mask(dt["masks"][k], (W, H))[y0:y1, x0:x1]
            old = base[y0:y1, x0:x1]
            a = paste_alpha(m, prot)
            new = colour_match(new, old, a < 0.02)
            base = feather_paste(base, new, dt["box"], a)
            n += 1
        if n:
            out = Path(out_dir) / f"{Path(accepted[kid]['path']).stem}__d.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(base).save(out)
            done[kid] = str(out)
    return done


def block_stills(run: dict) -> dict:
    """colour_block() on every shot's engine still (large pieces with a colour word in their look and a mask or
    box), written to <out>/detail/blocked/<scene>__<cam>.png -> {"scene/cam": path}. redraw.py starts from these
    when the run says "blocked_stills": true. Off by default: it changes the source of every key."""
    done = {}
    for sd in map(Path, run["scenes"]):
        truth = json.loads((sd / "truth.json").read_text())
        for cam, sh in truth["shots"].items():
            still = _img(sh["still"])
            H, W = still.shape[:2]
            people = np.zeros((H, W), bool)
            for k in sh.get("people") or {}:
                mp = sd / "guides" / f"{cam}_mask_{k}.png"
                if mp.exists():
                    people |= _mask(mp, (W, H))
            ps = [(piece_mask(t.get("mask"), t["box"], W, H) & ~people, base_colour(t.get("words", "")))
                  for t in sh.get("things") or [] if t.get("large") and _area(t) >= PIECE_AREA]
            ps = [x for x in ps if x[1] is not None]
            if ps:
                o = Path(run["out"]) / "detail" / "blocked" / f"{sd.name}__{cam}.png"
                _save(colour_block(still, ps), o)
                done[f"{sd.name}/{cam}"] = str(o)
    return done


def lit_keys(run: dict, accepted: dict) -> dict:
    """composite_lit() on each accepted key for its lit sources smaller than LIT_SMALL of the frame -> {id: path
    of <out>/detail/<key>__l.png}. Off by default (a separate command), for a box test: with it a candle can be a lit thing."""
    from PIL import Image
    meta = json.loads((Path(run["out"]) / "meta.json").read_text())
    done = {}
    for kid, rec in accepted.items():
        m = meta.get(rec.get("id", kid)) or meta.get(kid)
        if not m:
            continue
        sd = next(Path(x) for x in run["scenes"] if Path(x).name == m["scene"])
        sh = json.loads((sd / "truth.json").read_text())["shots"][m["cam"]]
        small = [t for t in sh.get("things") or [] if (t.get("lit") or t.get("flame_in_view")) and _area(t) < LIT_SMALL]
        if not small:
            continue
        key = _img(rec["path"])
        H, W = key.shape[:2]
        still = np.asarray(Image.open(sh["still"]).convert("RGB").resize((W, H), Image.LANCZOS))
        for t in small:
            key = composite_lit(key, still, piece_mask(t.get("lit_mask") or t.get("mask"), t["box"], W, H))
        o = Path(run["out"]) / "detail" / f"{Path(rec['path']).stem}__l.png"
        o.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(key).save(o)
        done[kid] = str(o)
    return done


def seconds_per_draw(run_out: Path) -> float | None:
    """The run's own median seconds per picture, from the image server's result lines in its logs."""
    secs = []
    for f in list(run_out.glob("*.log")) + list(run_out.glob("*result*.json*")) + list(run_out.parent.glob(f"pipeline_{run_out.name}.log")):
        try:
            for ln in f.read_text(errors="ignore").splitlines():
                mm = re.search(r'"seconds":\s*([0-9.]+)', ln)
                if mm:
                    secs.append(float(mm.group(1)))
        except OSError:
            pass
    return float(np.median(secs)) if secs else None


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "paste", "block", "lights"):
        s = sub.add_parser(name)
        s.add_argument("run")
        s.add_argument("--accepted")
        s.add_argument("--dry", action="store_true", help="plan: build and print the jobs, write nothing")
        s.add_argument("--pieces", choices=("flagged", "pairs", "off"), default="flagged")
    s = sub.add_parser("depth")
    s.add_argument("src"), s.add_argument("dst")
    s.add_argument("--inverse", action="store_true")
    s.add_argument("--clip", nargs=2, type=float, default=(2.0, 98.0))
    a = ap.parse_args()
    if a.cmd == "depth":
        from PIL import Image
        d = np.asarray(Image.open(a.src).convert("L")).astype(np.float64)
        _save(prep_depth(255 - d, inverse=a.inverse, clip=tuple(a.clip)), a.dst, mode="L")     # guides: near = white
        return
    run = json.loads(Path(a.run).read_text())
    out = Path(run["out"])
    ap_ = Path(a.accepted or out / "accepted.json")
    acc = json.loads(ap_.read_text()) if ap_.exists() else {}
    jp = out / "detail" / "detail_jobs.jsonl"
    if a.cmd == "block":
        print(json.dumps(block_stills(run), indent=1))
    elif a.cmd == "lights":
        print(json.dumps(lit_keys(run, acc), indent=1))
    elif a.cmd == "plan":
        jobs, rows = plan(run, acc, a.dry, a.pieces)
        per = seconds_per_draw(out)
        base_mp = 1152 * 640 / 1e6
        for r in rows:
            n = sum(1 for c in r["people"].values() if c == "detail")
            est = None if per is None else round(per / base_mp * (n * SIZE * SIZE / 1e6 + 0.8 * len(r["pieces"])), 1)
            print(json.dumps({"id": r["id"], "people": r["people"], "passes": n, "pieces": r["pieces"], "est_added_s": est}))
        if not a.dry:
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text("".join(json.dumps(j) + "\n" for j in jobs))
            print(jp, len(jobs))
        else:
            print(f"dry: {len(jobs)} jobs over {len(rows)} keys; base draw {per} s")
    else:
        jobs = [json.loads(ln) for ln in jp.read_text().splitlines() if ln.strip()]
        for k, p in paste(jobs, acc, out / "detail").items():
            print(k, p)


if __name__ == "__main__":
    main()
