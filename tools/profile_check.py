"""Is a drawn person a true SIDE view? (10-22: Edric's "right" A-pose drawing was a three-quarter view, the vision
judge called it a profile, and projected onto the model's side it painted his face on the side of the head.)

    ~/tts-ab/.venv-pose/bin/python -I tools/profile_check.py IMG [IMG ...]   -> JSON {img: {"ratio", "facing", "profile"}}

Geometry, not a question: the whole-body keypoints' eyes and the ear/nose. In a profile the two eyes nearly line up
seen from the side (their horizontal separation is small next to the ear-to-nose distance): ratio = |eye_l.x -
eye_r.x| / |nose.x - ear.x| (the ear with the better score). Frontal ~0.8-1, three-quarter 0.2-0.6, profile < 0.15.
"facing" = "left" / "right": the side of the picture the nose points to.

The BODY too (11-24: a "side" drawing with the head in profile on a body facing the front passed, and painted onto
the model's sides it laid a second front picture over the first - a doubled face, doubled arms): "body" = the
shoulders' horizontal separation over the shoulder-to-hip height. Front views 0.61-0.79, true side views 0.24-0.42.
A profile needs both.
"""

from __future__ import annotations

import json
import sys

BODY_MAX = 0.45                    # 11-24 cast of 17: side views 0.24-0.42, front-facing bodies 0.48-0.79
PROFILE_MAX = 0.15                 # 10-22 cast: left profiles 0.07-0.12, the (three-quarter) right views 0.20-0.26


def body_ratio(k):
    """Shoulder separation over shoulder-to-hip height (whole-body keypoints 5, 6 = shoulders, 11, 12 = hips);
    None when the trunk has no height."""
    torso = abs(float((k[5][1] + k[6][1]) / 2 - (k[11][1] + k[12][1]) / 2))
    return round(abs(float(k[5][0] - k[6][0])) / torso, 3) if torso > 3 else None


def is_profile(ratio, body) -> bool:
    """Head AND body seen from the side (a body that cannot be measured does not fail the head's verdict)."""
    return ratio is not None and ratio < PROFILE_MAX and (body is None or body < BODY_MAX)


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
    body = body_ratio(k)
    return {"ratio": round(ratio, 3), "body": body, "facing": "right" if nose[0] > ear[0] else "left",
            "profile": is_profile(ratio, body),
            "scores": {"eyes": [round(float(s[1]), 2), round(float(s[2]), 2)], "ears": [round(float(s[3]), 2), round(float(s[4]), 2)]}}


def main() -> int:
    from rtmlib import Wholebody
    m = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    sys.stdout.write("\nJSON " + json.dumps({p: check(p, m) for p in sys.argv[1:]}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
