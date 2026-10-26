"""Face-level likeness + AGE check for a drawn keyframe against a character's reference sheet (any book).

    python3 tools/face_likeness.py KEYFRAME.png --ref /abs/sheet.png --age 70 --gender woman \
        [--side left|right|any] [--json out.json] [--headshot-out /abs/headshot.png]

1. faces (rtmlib wholebody face keypoints, ~/tts-ab/.venv-pose): every face in the keyframe; the
   character's face = the one on --side (or the biggest); the ref sheet's head crop = its largest face
   (saved as --headshot-out: a HEADSHOT reference to draw with).
2. score (CLIP ViT-L/14 + DINOv2-small, ~/tts-ab/.venv-hy3d, CPU):
   - face_sim: DINOv2 cosine between the keyframe face crop and the sheet's head crop;
   - age: CLIP zero-shot over age bands ("a painted portrait of a {gender} in her/his 20s ... 80s, with
     {skin}") -> expected age (probability-weighted band centres) for the keyframe face AND the sheet's
     face (the drawer's reference truth); age_gap = keyframe - sheet.
Pass rule (used to rank or reject seeds): |age_gap| <= 10 years and face_sim >= 0.45.
Prints/writes {"face_box", "ref_box", "face_sim", "age": {...}, "pass"}.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

POSE_PY = os.path.expanduser("~/tts-ab/.venv-pose/bin/python")
HY_PY = os.path.expanduser("~/tts-ab/.venv-hy3d/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = [25, 35, 45, 55, 65, 75, 85]


def faces(path):
    import cv2
    import numpy as np
    from rtmlib import Wholebody
    img = cv2.imread(path)
    wb = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    kps, scores = wb(img)
    out = []
    for k, s in zip(kps, scores):
        f = np.asarray(k)[23:91][np.asarray(s)[23:91] > 0.3]
        if len(f) < 10:
            continue
        x0, y0 = f.min(0)
        x1, y1 = f.max(0)
        w, h = x1 - x0, y1 - y0
        # the face landmarks cover brows-to-chin: pad up for the forehead/hair
        box = [float(x0 - 0.25 * w), float(y0 - 0.55 * h), float(x1 + 0.25 * w), float(y1 + 0.2 * h)]
        lm = np.concatenate([np.asarray(k)[23:91], np.asarray(s)[23:91, None]], axis=1)   # 68 landmarks + scores
        out.append({"box": [max(0.0, b) for b in box], "area": float(w * h),
                    "lm": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in lm]})
    return {"size": [img.shape[1], img.shape[0]], "faces": out}


def score(kf, kbox, ref, rbox, age_bands_text, gender):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    a = Image.open(kf).convert("RGB").crop(tuple(kbox))
    b = Image.open(ref).convert("RGB").crop(tuple(rbox))
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    m = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    with torch.inference_mode():
        e = m(**proc(images=[a, b], return_tensors="pt")).last_hidden_state[:, 0]
        e = e / e.norm(dim=-1, keepdim=True)
    sim = float(e[0] @ e[1])
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    texts = age_bands_text
    with torch.inference_mode():
        x = cp(text=texts, images=[a, b], return_tensors="pt", padding=True)
        lo = cm(**x).logits_per_image.softmax(-1).numpy()
    ages = [float((p * np.array(BANDS)).sum()) for p in lo]
    # wrinkles: a direct smooth-vs-deeply-wrinkled contrast (the age bands alone follow the grey hair)
    wt = [f"a close-up portrait of a middle-aged {gender} with smooth, unlined skin",
          f"a close-up portrait of an elderly {gender} with a deeply wrinkled, lined, sagging face"]
    with torch.inference_mode():
        x2 = cp(text=wt, images=[a, b], return_tensors="pt", padding=True)
        lw = cm(**x2).logits_per_image.softmax(-1).numpy()
    return {"face_sim": round(sim, 4), "age": {"keyframe": round(ages[0], 1), "sheet": round(ages[1], 1),
                                               "keyframe_probs": [round(float(v), 3) for v in lo[0]]},
            "wrinkled": {"keyframe": round(float(lw[0][1]), 3), "sheet": round(float(lw[1][1]), 3)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--age", type=float)
    ap.add_argument("--gender", default="person")
    ap.add_argument("--side", default="any")
    ap.add_argument("--json")
    ap.add_argument("--headshot-out")
    ap.add_argument("--_faces", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_score", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a._faces:                                   # inside the pose venv
        print(json.dumps({"kf": faces(a.image), "ref": faces(a.ref)}))
        return 0
    if a._score:                                   # inside the hy3d venv
        d = json.loads(a._score)
        print(json.dumps(score(a.image, d["kbox"], a.ref, d["rbox"], d["texts"], a.gender)))
        return 0
    r = subprocess.run([POSE_PY, "-I", os.path.abspath(__file__), a.image, "--ref", a.ref, "--_faces"],
                       capture_output=True, text=True)
    F = json.loads(r.stdout.strip().splitlines()[-1])
    kfs = F["kf"]["faces"]
    if not kfs:
        out = {"pass": False, "fail": "no face found in the keyframe"}
        print(json.dumps(out))
        return 2
    W = F["kf"]["size"][0]
    if a.side == "left":
        kf = min(kfs, key=lambda f: f["box"][0])
    elif a.side == "right":
        kf = max(kfs, key=lambda f: f["box"][2])
    else:
        kf = max(kfs, key=lambda f: f["area"])
    rf = max(F["ref"]["faces"], key=lambda f: f["area"]) if F["ref"]["faces"] else None
    if rf is None:
        sys.exit("no face in the reference sheet")
    if a.headshot_out:
        from PIL import Image
        x0, y0, x1, y1 = rf["box"]
        w, h = x1 - x0, y1 - y0
        Image.open(a.ref).convert("RGB").crop((max(0, x0 - 0.5 * w), max(0, y0 - 0.3 * h), x1 + 0.5 * w,
                                               y1 + 0.9 * h)).save(a.headshot_out)
    g = a.gender
    pron = "her" if g in ("woman", "girl") else "his" if g in ("man", "boy") else "their"
    texts = [f"a painted portrait of a {g} in {pron} {d}s, {'smooth young' if d < 40 else 'lined' if d < 60 else 'deeply wrinkled, aged'} skin"
             for d in (20, 30, 40, 50, 60, 70, 80)]
    env = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([HY_PY, "-I", os.path.abspath(__file__), a.image, "--ref", a.ref, "--gender", g, "--_score",
                        json.dumps({"kbox": kf["box"], "rbox": rf["box"], "texts": texts})],
                       capture_output=True, text=True, env=env)
    S = json.loads(r.stdout.strip().splitlines()[-1])
    target = a.age if a.age is not None else S["age"]["sheet"]
    gap = S["age"]["keyframe"] - target
    out = {"face_box": kf["box"], "ref_box": rf["box"], **S, "target_age": target, "age_gap": round(gap, 1),
           "pass": abs(gap) <= 10 and S["face_sim"] >= 0.45 and
                   S["wrinkled"]["keyframe"] >= 0.6 * S["wrinkled"]["sheet"]}
    s = json.dumps(out)
    if a.json:
        open(a.json, "w").write(s)
    print(s)
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
