"""Post-draw VISION CHECK of a keyframe against its blockout (layer 5), CPU only (rtmlib DWPose wholebody).

    ~/tts-ab/.venv-pose/bin/python -I tools/keyframe_check.py KEYFRAME.png EXPECT.json [--json out.json]

EXPECT.json is written by tools/blockout_video.py --stills (<start|end>_expect.json): per figure its 2D
bbox and, per gripping fist, the projected fist, handle segment and working part (strike_face/tip/jaws).
Checks (generic, any shot):
  1. figure present: a detected person overlaps every blockout figure's bbox (IoU >= 0.2 or its centre
     inside), so a drawer that drops or melts a character fails;
  2. grip: the drawn hand (wholebody hand keypoints, the matching side) lies ON the handle segment (within
     1.5 fist sizes) and nearer the handle than the working part (a hand on the hammer head fails);
  3. readability of the blockout itself: the 2D gap between fist and working part >= 1.5 fist sizes.
Exit 0 = pass, 2 = fail (with reasons), 3 = a check could not be made (no hand found).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def seg_dist(p, a, b):
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
    return float(np.linalg.norm(p - (a + ab * t)))


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def detect(path, size):
    import cv2
    from rtmlib import Wholebody
    img = cv2.imread(path)
    sx, sy = size[0] / img.shape[1], size[1] / img.shape[0]
    wb = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    kps, scores = wb(img)
    people = []
    for k, sc in zip(kps, scores):
        k = np.asarray(k, float) * [sx, sy]
        body = k[:17][sc[:17] > 0.3]
        if len(body) < 4:
            continue
        bb = [body[:, 0].min(), body[:, 1].min(), body[:, 0].max(), body[:, 1].max()]
        hands = {}
        for side, sl in (("l", slice(91, 112)), ("r", slice(112, 133))):
            hk, hs = k[sl], sc[sl]
            good = hk[hs > 0.3]
            hands[side] = good.mean(0) if len(good) >= 5 else None
        people.append({"bbox": bb, "hands": hands, "n": int((sc > 0.3).sum())})
    return people


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("expect")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ex = json.load(open(a.expect))
    people = detect(a.image, ex["size"])
    out = {"image": a.image, "figures": [], "fail": [], "unknown": []}
    for fg in ex["figures"]:
        bb = fg["bbox"]
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        cand = [(iou(bb, p["bbox"]), p) for p in people
                if iou(bb, p["bbox"]) >= 0.2 or (p["bbox"][0] <= cx <= p["bbox"][2] and p["bbox"][1] <= cy <= p["bbox"][3])]
        r = {"who": fg["who"], "present": bool(cand), "grips": []}
        if not cand:
            out["fail"].append(f"{fg['who']}: no drawn person over the blockout figure")
        person = max(cand, key=lambda t: t[0])[1] if cand else None
        for g in fg["grips"]:
            fist, h0, h1, work, fpx = g["fist"], g["handle"][0], g["handle"][1], g["work"], max(g.get("fist_px", 20), 8)
            gap = float(np.linalg.norm(np.subtract(work, fist))) if work and fist else None
            rec = {"hand": g["hand"], "item": g["item"], "readability_gap_fists": None if gap is None else round(gap / fpx, 2)}
            if gap is not None and gap < 1.5 * fpx:
                out["fail"].append(f"{fg['who']} {g['hand']} {g['item']}: BLOCKOUT not readable: fist-to-"
                                   f"{g['work_name']} gap {gap / fpx:.1f} fist sizes (< 1.5)")
            hp = person["hands"].get(g["hand"]) if person else None
            if hp is None:
                rec["drawn"] = "hand not found"
                out["unknown"].append(f"{fg['who']} {g['hand']}: no hand keypoints")
            else:
                dh = seg_dist(hp, h0, h1)
                dw = float(np.linalg.norm(np.subtract(hp, work))) if work else 1e9
                rec.update(d_handle_fists=round(dh / fpx, 2), d_work_fists=round(dw / fpx, 2),
                           hand_xy=[round(float(x), 1) for x in hp])
                if dh > 1.5 * fpx or dw < dh:
                    out["fail"].append(f"{fg['who']} {g['hand']} hand not on the {g['item']} handle "
                                       f"(handle {dh / fpx:.1f}, {g['work_name']} {dw / fpx:.1f} fist sizes)")
            r["grips"].append(rec)
        out["figures"].append(r)
    out["pass"] = not out["fail"]
    s = json.dumps(out, indent=1, default=float)
    if a.json:
        open(a.json, "w").write(s)
    print(s)
    return 0 if out["pass"] and not out["unknown"] else (2 if out["fail"] else 3)


if __name__ == "__main__":
    sys.exit(main())
