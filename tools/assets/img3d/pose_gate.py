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


def runs(row, min_len=8):
    """runs of figure pixels in a row, specks (paper texture, stray shading) under min_len dropped"""
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return [(a, b) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]) if b - a >= min_len]


def pocket(fg, top, bot, cx, sgn):
    """(armpit fraction, arm angle from vertical) by columns, or None."""
    from collections import deque
    Hh, Ww = fg.shape
    H = bot - top
    # the torso's edge on this side at chest height
    r0 = top + int(0.35 * H)
    rr = runs(fg[r0])
    body = min(rr, key=lambda ab: 0 if ab[0] <= cx <= ab[1] else min(abs(ab[0] - cx), abs(ab[1] - cx))) if rr else None
    if body is None:
        return None
    edge = body[1] if sgn > 0 else body[0]
    # background reached from the bottom row on this side of the body (a flood over background pixels)
    # the hand's reach on this side (farthest figure column above the hips): the pocket is closed there, so the flood
    # cannot pass round a hand inside the picture and over the arm
    up = fg[top:top + int(0.6 * H)]
    oc = np.nonzero(up.any(0))[0]
    reach = int(oc.max()) if sgn > 0 else int(oc.min())
    seen = np.zeros_like(fg)
    q = deque()
    cols = range(edge + 1, reach) if sgn > 0 else range(reach + 1, edge)
    for c in cols:
        if not fg[bot, c]:
            seen[bot, c] = True
            q.append((bot, c))
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y2, x2 = y + dy, x + dx
            if top <= y2 <= bot and 0 <= x2 < Ww and not seen[y2, x2] and not fg[y2, x2]:
                if (sgn > 0 and (x2 <= edge - int(0.02 * H) or x2 >= reach)) or \
                        (sgn < 0 and (x2 >= edge + int(0.02 * H) or x2 <= reach)):
                    continue            # this side of the body, inside the hand's reach
                seen[y2, x2] = True
                q.append((y2, x2))
    # the pocket's highest point within 0.12 H of the body's edge
    band = range(edge, min(Ww, edge + int(0.12 * H))) if sgn > 0 else range(max(0, edge - int(0.12 * H)), edge + 1)
    tops = [np.nonzero(seen[:, c])[0] for c in band]
    tops = [t.min() for t in tops if len(t)]
    if not tops:
        return None
    arm_row = min(tops)
    ap = float((bot - arm_row) / H)
    # the arm's centre line: per column beyond the edge, the first figure run from the top of the shoulders down
    pts = []
    c0 = edge + sgn * int(0.05 * H)
    c_end = Ww - 1 if sgn > 0 else 0
    for c in range(c0, c_end, sgn * 2):
        col = fg[top:top + int(0.7 * H), c]
        ys = np.nonzero(col)[0]
        if len(ys) == 0:
            break
        d = np.diff(np.r_[0, col.astype(np.int8), 0])
        a_, b_ = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        if not len(a_):
            break
        pts.append((c, top + (a_[0] + b_[0]) / 2))
    if len(pts) < 10:
        return None
    P = np.array(pts)
    k = np.polyfit(P[:, 0], P[:, 1], 1)[0]              # rows per column
    ang = float(90.0 - np.degrees(np.arctan(abs(k))))
    return ap, ang


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
            far = abs((outer[0] + outer[1]) / 2 - cx) > 0.10 * H     # an arm, not the other leg or a coat tail
            gaps[r] = gap > 2 and far
            if gaps[r]:
                arm_pts[r] = (outer[0] + outer[1]) / 2
        # the armpit: walk up from the lowest row with air beside the arm while the air holds
        # the armpit: the highest row from which air separates the arm from the body for most of the next 0.1 H
        rows_g = sorted(r for r, g in gaps.items() if g)
        armpit = None
        win = max(5, int(0.1 * H))
        for r in rows_g:
            if sum(1 for rr in range(r, r + win) if gaps.get(rr)) >= 0.8 * win:
                armpit = r
                break
        if armpit is not None:
            arm_pts = [(rr, c) for rr, c in arm_pts.items() if armpit <= rr < armpit + 2 * win]
        if armpit is None or len(arm_pts) < 5:
            # arms near horizontal: no row crosses the arm apart from the body (the arm joins the shoulder in every
            # row it is in), so the row walk finds nothing (11-12: good spread drawings at 70-90 degrees all read
            # "no armpit"). Measured instead by columns: the background pocket under the arm, reached from the
            # picture's bottom on that side, and its highest point next to the body is the armpit; the arm's
            # centre line across the columns gives its angle
            pk = pocket(fg, top, bot, cx, sgn)
            if pk is None:
                out[side] = {"armpit": None, "angle_deg": None, "pass": False}
                ok = False
                continue
            ap, ang = pk
            p = ang >= min_angle and ap >= min_armpit
            out[side] = {"armpit": round(ap, 3), "angle_deg": round(ang, 1), "pass": p, "by": "columns"}
            ok &= p
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
