"""Likeness + prop-visibility scores for a drawn keyframe (CPU; DINOv2-small via transformers).

    HIP_VISIBLE_DEVICES= ~/tts-ab/.venv-hy3d/bin/python -I tools/keyframe_likeness.py KEYFRAME.png EXPECT.json \
        --ref WHO=/abs/outfit_sheet.png [--ref ...] [--still COLOR_STILL.png] [--json out.json]

- likeness[who]: cosine similarity of DINOv2 CLS embeddings between the figure's crop in the keyframe (its
  blockout bbox from EXPECT.json, padded 15 %) and the character's reference (outfit sheet / cut-out);
  ~0.3 unrelated, ~0.6+ same costume.
- hot[...]: for emissive props (glowing metal, fire set dressing) in the colour still (bright orange/yellow
  pixels): the fraction of that region that is still bright and warm in the drawing (does the glowing
  blade survive as a blade-shaped glow, not a smear elsewhere). None if the still has no hot region.
Prints / writes {"likeness": {who: s}, "hot": f | null}.
"""

from __future__ import annotations

import argparse
import json
import os

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
    refs = dict(r.split("=", 1) for r in a.ref)
    crops, names, refims = [], [], []
    for fg in ex["figures"]:
        if fg["who"] not in refs:
            continue
        x0, y0, x1, y1 = fg["bbox"]
        px, py = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        crops.append(im.crop((max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))))
        refims.append(Image.open(refs[fg["who"]]).convert("RGB"))
        names.append(fg["who"])
    out = {"likeness": {}, "hot": None}
    if crops:
        e = embed(crops + refims)
        n = len(crops)
        for i, w in enumerate(names):
            out["likeness"][w] = round(float(e[i] @ e[n + i]), 4)
    if a.still:
        st = np.asarray(Image.open(a.still).convert("RGB").resize((W, H)))
        m = hot_mask(st)
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
