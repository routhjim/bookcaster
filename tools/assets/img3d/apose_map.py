"""OpenPose-style (18 keypoints) skeleton map of a front-facing spread rest pose, drawn from fixed proportions
(no detector): python apose_map.py OUT.png [--deg 42] [--w 768] [--h 1344] [--fig 1100] [--female]"""
import argparse, math
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--deg", type=float, default=42)
ap.add_argument("--w", type=int, default=768); ap.add_argument("--h", type=int, default=1344)
ap.add_argument("--fig", type=float, default=1100); ap.add_argument("--legs_deg", type=float, default=5)
a = ap.parse_args()
H = a.fig; cx = a.w / 2; top = (a.h - H) / 2
def P(x, z): return (cx + x * H, top + (1 - z) * H)   # x, z as fractions of height (z up from the feet)
sh, hip = 0.112, 0.055
th = math.radians(a.deg); ua, fa = 0.175, 0.16
kp = {}
kp[0] = P(0, 0.935); kp[1] = P(0, 0.82)
kp[2] = P(-sh, 0.815); kp[5] = P(sh, 0.815)
for i, s in ((3, -1), (6, 1)):
    sx = s * sh
    kp[i] = P(sx + s * ua * math.sin(th), 0.815 - ua * math.cos(th))
    kp[i + 1] = P(sx + s * (ua + fa) * math.sin(th), 0.815 - (ua + fa) * math.cos(th))
lg = math.radians(a.legs_deg)
for i, s in ((8, -1), (11, 1)):
    kp[i] = P(s * hip, 0.52)
    kp[i + 1] = P(s * (hip + 0.245 * math.sin(lg)), 0.52 - 0.245 * math.cos(lg))
    kp[i + 2] = P(s * (hip + 0.49 * math.sin(lg)), 0.52 - 0.49 * math.cos(lg))
kp[14] = P(-0.018, 0.948); kp[15] = P(0.018, 0.948); kp[16] = P(-0.045, 0.94); kp[17] = P(0.045, 0.94)
# the picture's left is the person's right: OpenPose index 2 (right shoulder) sits at the picture's left
limbs = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
         [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
cols = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
im = Image.new("RGB", (a.w, a.h), "black"); d = ImageDraw.Draw(im)
sw = max(4, int(a.w / 128))
for i, (p, q) in enumerate(limbs):
    A, B = kp[p - 1], kp[q - 1]
    c = tuple(int(v * 0.6) for v in cols[i])
    d.line([A, B], fill=c, width=sw * 2)
for i in range(18):
    x, y = kp[i]; d.ellipse([x - sw, y - sw, x + sw, y + sw], fill=tuple(cols[i]))
im.save(a.out); print(a.out)
