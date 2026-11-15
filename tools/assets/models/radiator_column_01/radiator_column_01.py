"""Model script: cast-iron column radiator (12 sections of 4 columns), front -Y."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

paintm = K.metal("cast_iron", tarnish=0.5, polish=0.35, rust=0.15)
k = K.NK("radiator_paint")
n = k.noise(k.obj, 20.0, 4.0, 0.55)
cv = k.cavity(0.03, 0.9, 0.4)
e = k.edge(0.002)
col = k.mix(k.mr(n, 0.3, 0.7), (0.10, 0.065, 0.035, 1), (0.14, 0.095, 0.05, 1))
col = k.mix(k.m("MULTIPLY", cv, 0.8), col, (0.03, 0.025, 0.02, 1))
chip = k.m("MULTIPLY", e, k.mr(k.noise(k.obj, 150.0, 3.0, 0.5), 0.55, 0.6))
col = k.mix(chip, col, (0.05, 0.05, 0.05, 1))
paint = k.out(col, k.mixf(chip, 0.55, 0.4), k.mixf(chip, 0.3, 0.8), k.bump(n, 0.0004, 0.4))
brass = K.metal("brass", tarnish=0.7, polish=0.5)
NS, SW, H, Z0 = 12, 0.065, 0.66, 0.06
x0 = -(NS - 1) * SW / 2
for i in range(NS):
    x = x0 + i * SW
    for j, y in enumerate((-0.06, -0.02, 0.02, 0.06)):
        c = K.rbox(f"col{i}{j}", (0.045, 0.030, H), r=0.014, spacing=0.03, loc=(x, y, Z0 + H / 2), mat=paint)
        # cast leaf band: a swelling ring two thirds up
        K.warp(c, lambda v: (v.x * (1 + 0.12 * math.exp(-((v.z - 0.12) / 0.03) ** 2)), v.y, v.z))
    for z in (Z0 + 0.03, Z0 + H - 0.03):
        K.box(f"hub{i}{z:.2f}", (SW, 0.15, 0.05), (x, 0.0, z), bev=0.012, seg=3, mat=paint)
for sx in (-1, 1):
    K.box(f"foot{sx}", (0.05, 0.18, Z0 + 0.01), (sx * (-x0 - SW), 0.0, (Z0 + 0.01) / 2), bev=0.006, seg=2, mat=paint)
# inlet pipe and valve on the left
px = x0 - SW / 2 - 0.05
pipe = K.tube("pipe", [(x0 - SW / 2 + 0.01, 0.0, Z0 + H - 0.03), (px, 0.0, Z0 + H - 0.03), (px, 0.0, 0.0)], 0.013, seg=12, mat=paintm)
val = K.lathe("valve", K.keys([(0.0, 0.0001, "c"), (0.0, 0.022, "c"), (0.03, 0.022, "l"), (0.035, 0.012), (0.06, 0.008),
                               (0.06, 0.0001, "c")], 0.003), seg=20, mat=brass)
val.location = (px, 0.0, Z0 + H - 0.10)
wheel = K.tube("wheel", [(px + 0.03 * math.cos(a), 0.03 * math.sin(a), Z0 + H - 0.04) for a in [math.tau * t / 20 for t in range(20)]],
               0.004, seg=8, closed=True, mat=brass)
for a in (0, math.pi / 2):
    K.box(f"spoke{a:.1f}", (0.06, 0.004, 0.004), (px, 0, Z0 + H - 0.04), (0, 0, a), bev=0.0, mat=brass)
