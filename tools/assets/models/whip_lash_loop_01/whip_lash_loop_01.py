"""Model script: short dog whip on the floor, its whipcord lash curled into a loop and knotted.

Stock: lathe along X (knob at -X). Lash: one swept cord leaving the stock tip, running out, round a loop and back
over itself to a knot near the stock, with a short tail; a knot lump and a plaited keeper."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

leather = K.fabric("leather", (0.030, 0.016, 0.009), wear=0.6, scale=3.0)
cordm = K.fabric("rope", (0.10, 0.085, 0.06))
RS = 0.013
stock = K.lathe("stock", K.keys([(0.0, 0.0001, "c"), (0.0, 0.010, "c"), (0.006, 0.016), (0.016, 0.017), (0.026, 0.012),
                                 (0.04, 0.0125), (0.30, 0.0085), (0.31, 0.0001, "c")], 0.004), seg=20, mat=leather)
stock.location, stock.rotation_euler = (-0.30, 0.0, 0.017), (0, math.pi / 2, 0)
keeper = K.lathe("keeper", K.keys([(0.0, 0.0001, "c"), (0.0, 0.0095, "c"), (0.004, 0.0105), (0.03, 0.0098),
                                   (0.034, 0.0060), (0.036, 0.0001, "c")], 0.002), seg=16, mat=cordm)
keeper.location, keeper.rotation_euler = (-0.005, 0.0, 0.017), (0, math.pi / 2, 0)
# lash path on the floor (z = cord radius, raised where it crosses itself)
RC = 0.0045
pts = [(0.025, 0.0)]
pts += [p for p in K.bezier((0.025, 0.0), (0.09, 0.0), (0.16, -0.03), (0.22, -0.055), 16)[1:]]
CX, CY, LR = 0.24, 0.0, 0.062                       # the loop
a0 = math.atan2(-0.055 - CY, 0.22 - CX)
for k in range(1, 41):
    a = a0 + 2 * math.pi * 0.86 * k / 40
    pts.append((CX + LR * math.cos(a) * (1 + 0.05 * math.sin(3 * a)), CY + LR * math.sin(a)))
end = pts[-1]
pts += [p for p in K.bezier(end, (end[0] - 0.03, end[1] - 0.02), (0.12, 0.012), (0.075, 0.004), 12)[1:]]
tail = K.bezier((0.075, 0.004), (0.06, 0.004), (0.06, 0.03), (0.08, 0.05), 8)[1:]
path = []
for i, (x, y) in enumerate(pts + tail):
    z = RC
    if 0.06 < x < 0.13 and abs(y) < 0.02 and i > 20:        # passes over the outgoing lash near the knot
        z = RC + 2 * RC * K.smooth(1 - abs(x - 0.095) / 0.035)
    path.append((x, y, z))
radii = [RC * (1.25 if i < 6 else 1.0) * (0.75 if i > len(path) - 6 else 1.0) for i in range(len(path))]
K.sweep("lash", path, [K.superellipse2d(r, r, 2.0, 0.0, 10) for r in radii], up=(0, 0, 1), mat=cordm, angle=70)
knot = K.lathe("knot", K.keys([(0.0, 0.0001, "c"), (0.0, 0.007, "c"), (0.005, 0.011), (0.012, 0.012), (0.018, 0.008),
                               (0.02, 0.0001, "c")], 0.0015), seg=16, mat=cordm)
knot.location, knot.rotation_euler = (0.085, 0.002, 0.009), (0, math.pi / 2, 0.15)
K.warp(knot, lambda c: (c.x * 1.0, c.y * 0.8, c.z))
