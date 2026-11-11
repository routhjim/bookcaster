"""2D gate for a spread-pose drawing before it is generated in 3D: are the arms out far enough and is there air
between each arm and the body up to a real armpit? (Pillow + numpy, milliseconds; a drawing on a plain background.)

    python tools/assets/img3d/pose_gate.py DRAWING.png [...] [--min-angle 35] [--min-armpit 0.70]

The figure is keyed from the background (the median of the picture's border, colour distance > --key). Per side:
the armpit is the highest row (as a fraction of the figure's height above its feet) where the outermost run of figure
pixels (the arm) is separated from the next run inward (the coat) by background; the arm angle is the angle from
vertical of a line fitted through the arm run's centres from the armpit down to the hand. Passes when both sides
have angle >= --min-angle and armpit >= --min-armpit. Prints one JSON line per drawing; exit 1 when any fails.
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image


def runs(row):
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def gate(path, key, min_angle, min_armpit):
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    border = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]])
    bg = np.median(border, 0)
    fg = np.linalg.norm(im - bg, axis=2) > key
    rows = np.nonzero(fg.any(1))[0]
    top, bot = rows.min(), rows.max()
    H = bot - top
    cx = np.nonzero(fg[top + int(0.3 * H)])[0].mean()        # the figure's middle at chest height
    out = {"file": path, "height_px": int(H)}
    ok = True
    for side, sgn in (("left_of_picture", -1), ("right_of_picture", 1)):
        arm_pts, gaps = {}, {}
        for r in range(top + int(0.15 * H), top + int(0.75 * H)):
            rs = [(a_, b_) for a_, b_ in runs(fg[r]) if (sgn > 0 and a_ > cx) or (sgn < 0 and b_ < cx)]
            allr = runs(fg[r])
            inner = [min(allr, key=lambda ab: 0 if ab[0] <= cx <= ab[1] else min(abs(ab[0] - cx), abs(ab[1] - cx)))] \
                if allr else []
            rs = [ab for ab in rs if ab not in inner]
            if not rs or not inner:
                gaps[r] = False
                continue
            outer = max(rs, key=lambda ab: sgn * (ab[0] + ab[1]))
            gap = (outer[0] - inner[0][1]) if sgn > 0 else (inner[0][0] - outer[1])
            gaps[r] = gap > 2
            if gaps[r]:
                arm_pts[r] = (outer[0] + outer[1]) / 2
        # the armpit: walk up from the lowest row with air beside the arm while the air holds
        rows_g = sorted(r for r, g in gaps.items() if g)
        armpit = None
        if rows_g:
            r = rows_g[-1]
            while gaps.get(r - 1):
                r -= 1
            armpit = r
            arm_pts = [(rr, c) for rr, c in arm_pts.items() if rr >= armpit]
        if armpit is None or len(arm_pts) < 5:
            out[side] = {"armpit": None, "angle_deg": None, "pass": False}
            ok = False
            continue
        P = np.array(arm_pts)
        k = np.polyfit(P[:, 0], P[:, 1], 1)[0]                 # columns per row
        ang = float(np.degrees(np.arctan(abs(k))))
        ap = float((bot - armpit) / H)
        p = ang >= min_angle and ap >= min_armpit
        out[side] = {"armpit": round(ap, 3), "angle_deg": round(ang, 1), "pass": p}
        ok &= p
    out["pass"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drawings", nargs="+")
    ap.add_argument("--key", type=float, default=28.0)
    ap.add_argument("--min-angle", type=float, default=35.0)
    ap.add_argument("--min-armpit", type=float, default=0.70)
    a = ap.parse_args()
    allok = True
    for d in a.drawings:
        r = gate(d, a.key, a.min_angle, a.min_armpit)
        print(json.dumps(r), flush=True)
        allok &= r["pass"]
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
