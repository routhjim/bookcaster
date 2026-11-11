"""Face proportions from the editor's image landmark tracker (char_render.py "track" job result), as ratios to the
distance between the eye centres. Stdlib only.

    python landmarks.py track_result.json [out.json]

Measures (all divided by the eye distance; y grows downwards):
  mouth_width        lip corner to lip corner
  mouth_below_eyes   eye-centre line to the mouth's centre (lip seam), vertically
  lip_height         top of the upper lip to the bottom of the lower lip
  upper_lip / lower_lip   each lip's height at the centre
  eye_width          mean width of one eye (corner to corner)
  eye_open           mean lid opening at the eye's middle
  nasolabial_top     distance between the upper ends of the two nasolabial folds (nose-wing width proxy)
  nose_below_eyes    eye-centre line to the upper ends of the nasolabial folds
"""

import json
import math
import sys


def mean(pts):
    return [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def measures(c):
    eye = {}
    for s in ("l", "r"):
        up, lo = c[f"crv_eyelid_upper_{s}"], c[f"crv_eyelid_lower_{s}"]
        eye[s] = {"centre": mean(up + lo), "width": dist(up[0], up[-1]),
                  "open": abs(up[len(up) // 2][1] - lo[len(lo) // 2][1])}
    ed = dist(eye["l"]["centre"], eye["r"]["centre"])
    ey = (eye["l"]["centre"][1] + eye["r"]["centre"][1]) / 2
    cl, cr = c["crv_lip_upper_outer_l"][-1], c["crv_lip_upper_outer_r"][-1]
    seam = mean(c["crv_lip_upper_inner_l"] + c["crv_lip_upper_inner_r"])
    top = min(p[1] for p in c["crv_lip_upper_outer_l"] + c["crv_lip_upper_outer_r"])
    bot = max(p[1] for p in c["crv_lip_lower_outer_l"] + c["crv_lip_lower_outer_r"])
    up_c = c["crv_lip_philtrum_l"][0][1]
    lo_c = c["crv_lip_lower_outer_l"][0][1]
    in_c = c["crv_lip_lower_inner_l"][0][1]
    nl, nr = c["crv_nasolabial_l"][0], c["crv_nasolabial_r"][0]
    r = {"eye_distance_px": round(ed, 1),
         "mouth_width": dist(cl, cr) / ed,
         "mouth_below_eyes": (seam[1] - ey) / ed,
         "lip_height": (bot - top) / ed,
         "upper_lip": (in_c - up_c) / ed,
         "lower_lip": (lo_c - in_c) / ed,
         "eye_width": (eye["l"]["width"] + eye["r"]["width"]) / 2 / ed,
         "eye_open": (eye["l"]["open"] + eye["r"]["open"]) / 2 / ed,
         "nasolabial_top": dist(nl, nr) / ed,
         "nose_below_eyes": ((nl[1] + nr[1]) / 2 - ey) / ed}
    return {k: round(v, 3) for k, v in r.items()}


def main():
    res = json.load(open(sys.argv[1]))
    tr = res.get("track") or res.get("result", {}).get("track")
    out = {name: measures(v["curves"]) for name, v in tr.items() if v.get("curves")}
    txt = json.dumps(out, indent=1)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
