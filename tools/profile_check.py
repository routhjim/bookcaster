"""Is a drawn person a true SIDE view? (10-22: Edric's "right" A-pose drawing was a three-quarter view, the vision
judge called it a profile, and projected onto the model's side it painted his face on the side of the head.)

    ~/tts-ab/.venv-pose/bin/python -I tools/profile_check.py IMG [IMG ...]   -> JSON {img: {"ratio", "facing", "profile"}}

Geometry, not a question: the whole-body keypoints' eyes and the ear/nose. In a profile the two eyes nearly line up
seen from the side (their horizontal separation is small next to the ear-to-nose distance): ratio = |eye_l.x -
eye_r.x| / |nose.x - ear.x| (the ear with the better score). Frontal ~0.8-1, three-quarter 0.2-0.6, profile < 0.15.
"facing" = "left" / "right": the side of the picture the nose points to.
"""

from __future__ import annotations

import json
import sys

PROFILE_MAX = 0.15                 # 10-22 cast: left profiles 0.07-0.12, the (three-quarter) right views 0.20-0.26


def check(path: str, model=None) -> dict:
    import cv2
    import numpy as np
    from rtmlib import Wholebody
    model = model or Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    img = cv2.imread(path)
    kps, sc = model(img)
    if len(kps) == 0:
        return {"ratio": None, "facing": None, "profile": False, "why": "no person found"}
    i = int(np.argmax([s[:5].mean() for s in sc]))
    k, s = np.asarray(kps[i]), np.asarray(sc[i])
    nose, el, er, al, ar = k[0], k[1], k[2], k[3], k[4]
    ear = al if s[3] >= s[4] else ar
    span = abs(float(nose[0] - ear[0]))
    if span < 3:
        return {"ratio": None, "facing": None, "profile": False, "why": "ear on the nose (frontal or unclear)"}
    ratio = abs(float(el[0] - er[0])) / span
    return {"ratio": round(ratio, 3), "facing": "right" if nose[0] > ear[0] else "left", "profile": ratio < PROFILE_MAX,
            "scores": {"eyes": [round(float(s[1]), 2), round(float(s[2]), 2)], "ears": [round(float(s[3]), 2), round(float(s[4]), 2)]}}


def main() -> int:
    from rtmlib import Wholebody
    m = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    sys.stdout.write("\nJSON " + json.dumps({p: check(p, m) for p in sys.argv[1:]}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
