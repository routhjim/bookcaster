"""Model script: twisted bell-pull rope with a tassel, hanging along Z from a ceiling plate at the top.

Three helical strands (period_kit.tube) round the axis, a ring and plate at the top, a knot, a turned head and a
skirt of loose cords splaying a little at the bottom."""

import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

rope = K.fabric("rope", (0.34, 0.27, 0.17))
cordm = K.fabric("rope", (0.30, 0.235, 0.15))
brass = K.metal("brass", tarnish=0.85, polish=0.3)
parts = []
Z_TASSEL = 0.20                 # where the rope enters the tassel head
Z_TOP = 1.74
R_H, R_S, PITCH = 0.0095, 0.0105, 0.075
for s in range(3):
    pts = []
    n = int((Z_TOP - Z_TASSEL + 0.03) / PITCH * 18)
    for k in range(n + 1):
        z = K.lerp(Z_TASSEL - 0.03, Z_TOP, k / n)
        a = math.tau * (z / PITCH + s / 3)
        pts.append((R_H * math.cos(a), R_H * math.sin(a), z))
    parts.append(K.tube(f"strand{s}", pts, R_S, seg=8, mat=rope))
# top: rope loop through a ring under a plate
ring_pts = [(0.0, 0.022 * math.cos(a), Z_TOP + 0.026 + 0.022 * math.sin(a)) for a in [math.tau * k / 24 for k in range(24)]]
parts.append(K.tube("ring", ring_pts, 0.004, seg=8, closed=True, mat=brass))
parts.append(K.lathe("whip", K.keys([(Z_TOP - 0.05, 0.0001, "c"), (Z_TOP - 0.05, 0.022, "c"), (Z_TOP, 0.022, "l"),
                                     (Z_TOP + 0.01, 0.012), (Z_TOP + 0.012, 0.0001, "c")], 0.003), seg=20, mat=cordm))
plate = K.lathe("plate", K.keys([(0.0, 0.0001, "c"), (0.0, 0.045, "c"), (0.004, 0.044), (0.008, 0.035), (0.012, 0.012),
                                 (0.016, 0.006), (0.016, 0.0001, "c")], 0.0015), seg=32, mat=brass)
plate.location, plate.rotation_euler = (0, 0, Z_TOP + 0.064), (math.pi, 0, 0)
parts.append(plate)
# knot + tassel head
parts.append(K.lathe("knot", K.keys([(Z_TASSEL - 0.005, 0.0001, "c"), (Z_TASSEL - 0.005, 0.026, "c"), (Z_TASSEL + 0.02, 0.034),
                                     (Z_TASSEL + 0.05, 0.026), (Z_TASSEL + 0.065, 0.018), (Z_TASSEL + 0.07, 0.0001, "c")],
                                    0.003), seg=24, mat=rope))
parts.append(K.lathe("head", K.keys([(Z_TASSEL - 0.075, 0.0001, "c"), (Z_TASSEL - 0.075, 0.03, "c"), (Z_TASSEL - 0.04, 0.034),
                                     (Z_TASSEL - 0.01, 0.028), (Z_TASSEL, 0.0001, "c")], 0.003), seg=24, mat=cordm))
R = random.Random(5)
for k in range(44):
    a = math.tau * k / 44 + R.uniform(-0.05, 0.05)
    r0 = R.uniform(0.012, 0.028)
    L = R.uniform(0.10, 0.125)
    z0 = Z_TASSEL - 0.06
    pts = []
    for j in range(9):
        t = j / 8
        rr = r0 + 0.012 * t + R.uniform(-0.002, 0.002) * t
        pts.append((rr * math.cos(a + 0.1 * t), rr * math.sin(a + 0.1 * t), z0 - L * t))
    pts[-1] = (pts[-1][0], pts[-1][1], max(pts[-1][2], 0.0))
    parts.append(K.tube(f"cord{k}", pts, 0.0032, seg=6, mat=cordm, r_fn=lambda t: 0.0032 * (1 - 0.3 * t)))
