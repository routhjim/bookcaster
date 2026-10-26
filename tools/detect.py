"""Cheap, non-reasoning picture checks for bookcaster's keyframes (plan 10-18, phase 2), on the CPU so
they never compete with a render: Grounding DINO counts people, faces and creatures (a vision model's
counting is wrong ~40% of the time and it misses a stray head under a dragon's wing), and DINOv2 says how
alike a figure in the frame is to a character's reference sheet (an identity drift alarm).

    python detect.py  < jobs.jsonl  > results.jsonl
    {"id", "image", "count": ["person", "face", "dragon"], "same": {"Edric": "/refs/fig-edric.png"}}
    -> {"id", "counts": {"person": 2, ...}, "boxes": {"person": [[x0, y0, x1, y1] (0-1), ...]},
        "same": {"Edric": 0.71}}   (cosine of DINOv2 CLS features, best-matching person box vs the sheet)

Runs in the drawing environment (transformers + torch); HIP_VISIBLE_DEVICES="" keeps it off the GPUs.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

BOX_THRESHOLD = float(os.environ.get("DETECT_BOX_THRESHOLD", "0.35"))
TEXT_THRESHOLD = float(os.environ.get("DETECT_TEXT_THRESHOLD", "0.30"))
DETECTOR = os.environ.get("DETECT_MODEL", "IDEA-Research/grounding-dino-tiny")
EMBEDDER = os.environ.get("DETECT_EMBED", "facebook/dinov2-small")
DEVICE = os.environ.get("DETECT_DEVICE", "cpu")  # "cuda" on a GPU (pin it with HIP_VISIBLE_DEVICES)


def load():
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(int(os.environ.get("DETECT_THREADS", "8")))
    dp = AutoProcessor.from_pretrained(DETECTOR)
    dm = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).eval().to(DEVICE)
    ep = AutoImageProcessor.from_pretrained(EMBEDDER)
    em = AutoModel.from_pretrained(EMBEDDER).eval().to(DEVICE)
    return dp, dm, ep, em


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area else 0.0


def detect(dp, dm, im, labels: list[str]) -> dict[str, list]:
    """Boxes per label (normalised, overlapping duplicates merged)."""
    import torch

    text = ". ".join(labels) + "."
    inputs = dp(images=im, text=text, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = dm(**inputs)
    res = dp.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[(im.height, im.width)])[0]
    got: dict[str, list] = {l: [] for l in labels}
    names = res.get("text_labels") or res.get("labels")
    for box, label, score in zip(res["boxes"].tolist(), names, res["scores"].tolist()):
        label = str(label).strip().lower()
        key = next((l for l in labels if l in label or label in l), None)
        if key is None:
            continue
        b = [box[0] / im.width, box[1] / im.height, box[2] / im.width, box[3] / im.height]
        if any(_iou(b, o) > 0.6 for o in got[key]):
            continue
        got[key].append([round(x, 3) for x in b])
    return got


def embed(ep, em, im):
    import torch

    with torch.inference_mode():
        f = em(**ep(images=im, return_tensors="pt").to(DEVICE)).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(f, dim=-1)[0].cpu()


def main() -> int:
    from PIL import Image

    t = time.time()
    dp, dm, ep, em = load()
    print(json.dumps({"phase": "load", "seconds": round(time.time() - t, 1)}), flush=True)
    sheets: dict[str, object] = {}
    for line in sys.stdin:
        if not line.strip():
            continue
        j = json.loads(line)
        t = time.time()
        try:
            im = Image.open(j["image"]).convert("RGB")
            labels = [str(x).lower() for x in j.get("count", ["person", "face"])]
            if j.get("same") and "person" not in labels:
                labels.append("person")
            boxes = detect(dp, dm, im, labels)
            same = {}
            for name, ref in (j.get("same") or {}).items():
                if ref not in sheets:
                    sheets[ref] = embed(ep, em, Image.open(ref).convert("RGB"))
                best = 0.0
                for b in boxes.get("person", []) + boxes.get(j.get("creature_label", "dragon"), []):
                    crop = im.crop((int(b[0] * im.width), int(b[1] * im.height),
                                    int(b[2] * im.width), int(b[3] * im.height)))
                    if crop.width < 16 or crop.height < 16:
                        continue
                    best = max(best, float(embed(ep, em, crop) @ sheets[ref]))
                same[name] = round(best, 3)
            print(json.dumps({"id": j["id"], "counts": {k: len(v) for k, v in boxes.items()}, "boxes": boxes,
                              "same": same, "seconds": round(time.time() - t, 1)}), flush=True)
        except Exception as exc:
            print(json.dumps({"id": j.get("id"), "error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
