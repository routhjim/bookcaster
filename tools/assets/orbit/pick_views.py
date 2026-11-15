"""Pick the four multi-view inputs (front, left, back, right) from the orbit frames by their reconstructed azimuths
and write RGBA cut-outs (full-resolution frame + its mask), plus a shape_mv.py runs file.

    python3 -I pick_views.py RECON_DIR OUT_DIR --front-clip c [--back-clip m] [--seeds 0 1]

front: among --front-clip frames, the one whose mask is most mirror-symmetric (left-right). The sides are the frames
nearest front +- 90 degrees; which sign is the subject's left is found from the image: a forward-facing subject's
head is the end of the mask that leads in the turn, so both assignments are written as runs ("lr" and "rl") and the
caller keeps the mesh that fits the frames better (fit.py's IoU). back: the frame nearest front + 180 (from
--back-clip when given). Needs only numpy and PIL (small arrays; bookkeeping, no image model).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("recon")
ap.add_argument("out")
ap.add_argument("--front-clip", default="c")
ap.add_argument("--back-clip", default=None)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
a = ap.parse_args()
R, O = Path(a.recon), Path(a.out)
O.mkdir(parents=True, exist_ok=True)
C = json.load(open(R / "cams.json"))
fr = C["frames"]


def tag(p):
    m = re.match(r"\d+_([a-zA-Z]+)", Path(p).name)
    return m.group(1) if m else ""


def mask(i):
    return np.asarray(Image.open(R / "masks" / f"m_{i:03d}.png")) > 127


def sym(i):
    m = mask(i)
    xs = np.nonzero(m.any(0))[0]
    if not len(xs):
        return 0
    cx = int(round(m.nonzero()[1].mean()))
    w = min(cx, m.shape[1] - cx)
    l, r = m[:, cx - w:cx], m[:, cx:cx + w][:, ::-1]
    return (l & r).sum() / max((l | r).sum(), 1)


cand = [f["i"] for f in fr if tag(f["path"]) == a.front_clip]
scores = {i: sym(i) for i in cand}
front = max(scores, key=scores.get)
az = np.array([f["az"] for f in fr])
fa = az[front]


def nearest(target, pool=None):
    ids = [f["i"] for f in fr if pool is None or tag(f["path"]) == pool]
    return min(ids, key=lambda i: abs(az[i] - target))


plus, minus = nearest(fa + 90), nearest(fa - 90)
back = nearest(fa + 180, a.back_clip) if a.back_clip else min(
    (nearest(fa + 180), nearest(fa - 180)), key=lambda i: min(abs(az[i] - fa - 180), abs(az[i] - fa + 180)))


def cut(i, name):
    p = fr[i]["path"]
    im = Image.open(p).convert("RGB")
    m = Image.open(R / "masks" / f"m_{i:03d}.png").convert("L").resize(im.size, Image.BILINEAR)
    rgba = Image.merge("RGBA", (*im.split(), m))
    q = O / f"{name}.png"
    rgba.save(q)
    return str(q)


views = {"front": cut(front, "front"), "back": cut(back, "back"), "plus": cut(plus, "side_plus"),
         "minus": cut(minus, "side_minus")}
runs = []
for s in a.seeds:
    runs.append({"name": f"lr_s{s}", "seed": s, "views": {"front": views["front"], "back": views["back"],
                                                         "left": views["plus"], "right": views["minus"]}})
    runs.append({"name": f"rl_s{s}", "seed": s, "views": {"front": views["front"], "back": views["back"],
                                                         "left": views["minus"], "right": views["plus"]}})
pick = {"front": [front, fr[front]["path"], float(fa)], "back": [back, fr[back]["path"], float(az[back])],
        "plus90": [plus, fr[plus]["path"], float(az[plus])], "minus90": [minus, fr[minus]["path"], float(az[minus])],
        "symmetry": {str(k): round(float(v), 3) for k, v in scores.items()}}
json.dump(pick, open(O / "pick.json", "w"), indent=1)
json.dump({"out_dir": str(O / "meshes"), "steps": 30, "octree": 320, "faces": 200000, "runs": runs},
          open(O / "runs.json", "w"), indent=1)
print(json.dumps({k: pick[k][:2] if isinstance(pick[k], list) else None for k in ("front", "back", "plus90",
                                                                               "minus90")}))
