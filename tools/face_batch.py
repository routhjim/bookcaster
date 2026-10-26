"""Resident batch face checker: tools/face_likeness.py's checks for MANY (picture, reference) pairs with ONE model load
per stage (DWPose in the pose venv, DINOv2 + CLIP in the hy3d venv), instead of ~48 s of loads per call.

    python3 tools/face_batch.py JOBS.json OUT.json
    JOBS: [{"id", "image", "ref", "gender", "age" (optional), "side": any|left|right}]
    OUT:  {id: {face_box, ref_box, face_sim, age{keyframe, sheet}, wrinkled{...}, target_age, age_gap, pass,
                face_px, skipped?}}

Faces narrower than MIN_FACE_PX, or with too few face keypoints (back of the head, profile in shadow), are SKIPPED
(no verdict) - judging a 30 px back-of-head face redrew a whole frame wrongly (10-20, Sherlock action-025031).
Same pass rule as face_likeness: |age gap| <= 10, face_sim >= 0.45, wrinkles >= 0.6 x the sheet's.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POSE_PY = os.path.expanduser("~/tts-ab/.venv-pose/bin/python")
HY_PY = os.path.expanduser("~/tts-ab/.venv-hy3d/bin/python")
MIN_FACE_PX = 48


def _faces_stage(paths):
    """pose venv: face boxes for every picture with ONE DWPose instance."""
    sys.path.insert(0, HERE)
    import rtmlib
    real = rtmlib.Wholebody
    inst = {}
    rtmlib.Wholebody = lambda *a, **k: inst.setdefault("wb", real(*a, **k))   # face_likeness.faces builds one per call
    import face_likeness as FL
    return {p: FL.faces(p) for p in paths}


def _score_stage(items):
    """hy3d venv: DINOv2 + CLIP loaded once; items: [{key, image, kbox, ref, rbox, texts, gender}]."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dm = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    bands = [25, 35, 45, 55, 65, 75, 85]
    out = {}
    for it in items:
        a = Image.open(it["image"]).convert("RGB").crop(tuple(it["kbox"]))
        b = Image.open(it["ref"]).convert("RGB").crop(tuple(it["rbox"]))
        g = it["gender"]
        with torch.inference_mode():
            e = dm(**proc(images=[a, b], return_tensors="pt")).last_hidden_state[:, 0]
            e = e / e.norm(dim=-1, keepdim=True)
            lo = cm(**cp(text=it["texts"], images=[a, b], return_tensors="pt", padding=True)).logits_per_image.softmax(-1).numpy()
            wt = [f"a close-up portrait of a middle-aged {g} with smooth, unlined skin",
                  f"a close-up portrait of an elderly {g} with a deeply wrinkled, lined, sagging face"]
            lw = cm(**cp(text=wt, images=[a, b], return_tensors="pt", padding=True)).logits_per_image.softmax(-1).numpy()
        ages = [float((p * np.array(bands)).sum()) for p in lo]
        out[it["key"]] = {"face_sim": round(float(e[0] @ e[1]), 4),
                          "age": {"keyframe": round(ages[0], 1), "sheet": round(ages[1], 1)},
                          "wrinkled": {"keyframe": round(float(lw[0][1]), 3), "sheet": round(float(lw[1][1]), 3)}}
    return out


def run(jobs: list[dict]) -> dict:
    imgs = sorted({j["image"] for j in jobs} | {j["ref"] for j in jobs})
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8")}
    r = subprocess.run([POSE_PY, "-I", __file__, "--_faces"], input=json.dumps(imgs), capture_output=True, text=True, env=env)
    F = json.loads(r.stdout.strip().splitlines()[-1])
    res, items = {}, []
    for j in jobs:
        kfs = [f for f in F[j["image"]]["faces"]]
        rfs = F[j["ref"]]["faces"]
        if not kfs or not rfs:
            res[j["id"]] = {"skipped": "no face found" if not kfs else "no face in the sheet", "pass": None}
            continue
        side = j.get("side", "any")
        kf = (min(kfs, key=lambda f: f["box"][0]) if side == "left" else
              max(kfs, key=lambda f: f["box"][2]) if side == "right" else max(kfs, key=lambda f: f["area"]))
        px = kf["box"][2] - kf["box"][0]
        if px < MIN_FACE_PX:
            res[j["id"]] = {"face_box": kf["box"], "face_px": round(px), "skipped": f"face {px:.0f} px < {MIN_FACE_PX}",
                            "pass": None}
            continue
        rf = max(rfs, key=lambda f: f["area"])
        g = j.get("gender", "person")
        pron = "her" if g in ("woman", "girl") else "his" if g in ("man", "boy") else "their"
        texts = [f"a painted portrait of a {g} in {pron} {d}s, "
                 f"{'smooth young' if d < 40 else 'lined' if d < 60 else 'deeply wrinkled, aged'} skin"
                 for d in (20, 30, 40, 50, 60, 70, 80)]
        items.append({"key": j["id"], "image": j["image"], "kbox": kf["box"], "ref": j["ref"], "rbox": rf["box"],
                      "texts": texts, "gender": g})
        res[j["id"]] = {"face_box": kf["box"], "ref_box": rf["box"], "face_px": round(px), "side": side}
    if items:
        env2 = {**env, "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": ""}
        r = subprocess.run([HY_PY, "-I", __file__, "--_score"], input=json.dumps(items), capture_output=True, text=True,
                           env=env2)
        S = json.loads(r.stdout.strip().splitlines()[-1])
        for j in jobs:
            if j["id"] in S:
                s = S[j["id"]]
                target = j.get("age") if j.get("age") is not None else s["age"]["sheet"]
                gap = s["age"]["keyframe"] - target
                res[j["id"]].update(s, target_age=target, age_gap=round(gap, 1),
                                    **{"pass": abs(gap) <= 10 and s["face_sim"] >= 0.45
                                       and s["wrinkled"]["keyframe"] >= 0.6 * s["wrinkled"]["sheet"]})
    return res


if __name__ == "__main__":
    if "--_faces" in sys.argv:
        print(json.dumps(_faces_stage(json.loads(sys.stdin.read()))))
    elif "--_score" in sys.argv:
        print(json.dumps(_score_stage(json.loads(sys.stdin.read()))))
    else:
        out = run(json.load(open(sys.argv[1])))
        json.dump(out, open(sys.argv[2], "w"), indent=1)
        print(json.dumps({"n": len(out), "skipped": sum(1 for v in out.values() if v.get("skipped"))}))
