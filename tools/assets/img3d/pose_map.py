"""OpenPose-style (18 keypoints) map of a measured pose seen straight from the front, from the scene tool's joints
(result.json subjects.<key>.measured.joints, set frame, metres), for drawing a figure in that exact pose (a seated
garment, for example). Stdlib + Pillow (pasting lines only).

    python pose_map.py RESULT.json KEY POSITION_X,Y FACING_X,Y OUT.png [--w 896] [--h 1344] [--margin 0.08]
"""
import argparse
import json
import math

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("result")
ap.add_argument("key")
ap.add_argument("position")
ap.add_argument("facing")
ap.add_argument("out")
ap.add_argument("--w", type=int, default=896)
ap.add_argument("--h", type=int, default=1344)
ap.add_argument("--margin", type=float, default=0.08)
a = ap.parse_args()
J = json.load(open(a.result))["subjects"][a.key]["measured"]["joints"]
px, py = map(float, a.position.split(","))
fx, fy = map(float, a.facing.split(","))
n = math.hypot(fx, fy)
fx, fy = fx / n, fy / n
lx, ly = -fy, fx                      # the person's left


def front(p):
    """(x across the picture, z up) as the camera in front of the person sees it: the person's left on the right."""
    d = (p[0] - px, p[1] - py)
    return d[0] * lx + d[1] * ly, p[2]


P = {k: front(v) for k, v in J.items()}
# the top of the head ~ 0.12 m above the head joint; nose between the eyes, a little lower
hx, hz = P["head"]
eyes = [P[k] for k in ("FACIAL_L_Eye", "FACIAL_R_Eye") if k in P]
nose = (sum(e[0] for e in eyes) / len(eyes), sum(e[1] for e in eyes) / len(eyes) - 0.04) if eyes else (hx, hz)
top = hz + 0.12
bot = min(v[1] for v in P.values()) - 0.03
H = top - bot
scale = a.h * (1 - 2 * a.margin) / H
cx = (max(v[0] for v in P.values()) + min(v[0] for v in P.values())) / 2


def pix(p):
    return (a.w / 2 + (p[0] - cx) * scale, a.h * a.margin + (top - p[1]) * scale)


neck = ((P["upperarm_l"][0] + P["upperarm_r"][0]) / 2, (P["upperarm_l"][1] + P["upperarm_r"][1]) / 2)
# OpenPose order: 0 nose 1 neck 2 r shoulder 3 r elbow 4 r wrist 5 l shoulder 6 l elbow 7 l wrist 8 r hip 9 r knee
# 10 r ankle 11 l hip 12 l knee 13 l ankle 14 r eye 15 l eye 16 r ear 17 l ear
kp = [nose, neck, P["upperarm_r"], P["lowerarm_r"], P["hand_r"], P["upperarm_l"], P["lowerarm_l"], P["hand_l"],
      P["thigh_r"], P["calf_r"], P["foot_r"], P["thigh_l"], P["calf_l"], P["foot_l"],
      P.get("FACIAL_R_Eye", nose), P.get("FACIAL_L_Eye", nose),
      (hx - 0.07 * (1 if lx >= 0 else 1), hz), (hx + 0.07, hz)]
limbs = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
         [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
cols = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
im = Image.new("RGB", (a.w, a.h), "black")
d = ImageDraw.Draw(im)
sw = max(4, int(a.w / 128))
K = [pix(p) for p in kp]
for i, (p, q) in enumerate(limbs):
    d.line([K[p - 1], K[q - 1]], fill=tuple(int(v * 0.6) for v in cols[i]), width=sw * 2)
for i in range(18):
    x, y = K[i]
    d.ellipse([x - sw, y - sw, x + sw, y + sw], fill=tuple(cols[i]))
im.save(a.out)
print(a.out)
