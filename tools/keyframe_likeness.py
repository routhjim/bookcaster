"""Likeness + prop-visibility scores for a drawn keyframe (CPU; DINOv2-small via transformers).

    HIP_VISIBLE_DEVICES= ~/tts-ab/.venv-hy3d/bin/python -I tools/keyframe_likeness.py KEYFRAME.png EXPECT.json \
        --ref WHO=/abs/outfit_sheet.png [--ref ...] [--still COLOR_STILL.png] [--json out.json]

- likeness[who]: cosine similarity of DINOv2 CLS embeddings between the figure's crop in the keyframe (its
  blockout bbox from EXPECT.json, padded 15 %) and the character's reference (outfit sheet / cut-out);
  ~0.3 unrelated, ~0.6+ same costume.
- hot[...]: for emissive props (glowing metal, fire set dressing) the staging put in this frame (prims tagged
  hot/spark in the still's stills.json, projected to the screen): the fraction of their bright orange/yellow
  pixels in the colour still that is still bright and warm in the drawing (does the glowing blade survive as a
  blade-shaped glow, not a smear elsewhere). None when the frame stages no such prop in view: a warm bright
  wall or face is not a glowing prop, and brightness alone never scores.
Prints / writes {"likeness": {who: s}, "hot": f | null}.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
from PIL import Image

MODEL = os.environ.get("LIKENESS_MODEL", "facebook/dinov2-small")


def embed(images):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL).eval()
    with torch.inference_mode():
        x = proc(images=images, return_tensors="pt")
        e = m(**x).last_hidden_state[:, 0]
    e = e / e.norm(dim=-1, keepdim=True)
    return e.numpy()


def hot_mask(rgb):
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    return (r > 200) & (g > 90) & (b < 140) & (r > b + 90)


TITLES = {"doctor": "dr", "mister": "mr", "missus": "mrs", "saint": "st", "professor": "prof",
          "captain": "capt", "colonel": "col"}
HOT_TAGS = ("hot", "spark")
HOT_PAD = 0.03          # of the frame width, around each staged glowing prim


def canon(name):
    """One spelling per person (tts_reader.video.canon's rule): a title written out or abbreviated, with or
    without its full stop, is the same name."""
    return " ".join(TITLES.get(w, w) for w in re.findall(r"[a-z0-9']+", str(name).lower()))


def _pts(pr):
    if "a" in pr and "b" in pr:
        return [pr["a"], pr["b"]]
    c = np.asarray(pr.get("c", [0, 0, 0]), float)
    M = np.asarray(pr["M"], float) if "M" in pr else np.zeros((3, 3))
    return [c] + [c + sg * M[:, i] for i in range(3) for sg in (-1, 1)]


def hot_region(still, W, H):
    """Boolean H x W mask of where this frame's staged glowing prims are on screen (each prim's projected box,
    padded), or None when the still has no scene record or no such prim in view."""
    d, base = os.path.dirname(os.path.abspath(still)), os.path.basename(still)
    sj = os.path.join(d, "stills.json")
    if not os.path.exists(sj) or "_" not in base:
        return None
    sc = json.load(open(sj))
    fr = next((f for f in sc.get("frames", []) if f.get("name") == base.rsplit("_", 1)[0]), None)
    cam = (fr or {}).get("camera")
    if not cam:
        return None
    Rm, C = np.asarray(cam["R"], float), np.asarray(cam["pos"], float)
    fpx = float(cam["lens"]) / 36.0 * W
    m, pad = np.zeros((H, W), bool), HOT_PAD * W
    for pr in fr.get("prims", []):
        if pr.get("tag") not in HOT_TAGS:
            continue
        q = [Rm.T @ (np.asarray(p, float) - C) for p in _pts(pr)]
        q = [v for v in q if v[2] < -1e-3]
        if not q:
            continue
        xs = [W / 2 + fpx * v[0] / -v[2] for v in q]
        ys = [H / 2 - fpx * v[1] / -v[2] for v in q]
        x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            continue
        m[int(max(0, y0)):int(min(H, y1)) + 1, int(max(0, x0)):int(min(W, x1)) + 1] = True
    return m if m.any() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("expect")
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--still")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ex = json.load(open(a.expect))
    W, H = ex["size"]
    im = Image.open(a.image).convert("RGB").resize((W, H))
    refs = {canon(k): v for k, v in (r.split("=", 1) for r in a.ref)}
    crops, names, refims = [], [], []
    for fg in ex["figures"]:
        if canon(fg["who"]) not in refs or not fg.get("bbox"):
            continue
        x0, y0, x1, y1 = fg["bbox"]
        px, py = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        crops.append(im.crop((max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))))
        refims.append(Image.open(refs[canon(fg["who"])]).convert("RGB"))
        names.append(fg["who"])
    out = {"likeness": {}, "hot": None}
    if crops:
        e = embed(crops + refims)
        n = len(crops)
        for i, w in enumerate(names):
            out["likeness"][w] = round(float(e[i] @ e[n + i]), 4)
    if a.still:
        st = np.asarray(Image.open(a.still).convert("RGB").resize((W, H)))
        reg = hot_region(a.still, W, H)
        m = hot_mask(st) & reg if reg is not None else np.zeros((H, W), bool)
        if m.sum() > 50:
            d = np.asarray(im)
            dr, dg, db = (d[..., i].astype(np.float32) for i in range(3))
            warm = (dr > 170) & (dr > db + 70) & (dg > 60)
            out["hot"] = round(float((warm & m).sum() / m.sum()), 4)
    s = json.dumps(out)
    if a.json:
        open(a.json, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
