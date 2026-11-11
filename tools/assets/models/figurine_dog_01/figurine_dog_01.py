"""Model script: Staffordshire-style seated spaniel figurine on an oval base, built from rounded bodies."""

import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

# glaze: white with rust-brown patches (noise mask), gilt chain/collar, black eyes
k = K.NK("staffs_glaze")
n = k.noise(k.obj, 18.0, 3.0, 0.55)
patch = k.mr(n, 0.55, 0.58)
col = k.mix(patch, (0.66, 0.64, 0.58, 1), (0.22, 0.07, 0.025, 1))
cv = k.cavity(0.01)
col = k.mix(k.m("MULTIPLY", cv, 0.5), col, (0.25, 0.22, 0.18, 1))
glaze = k.out(col, 0.12, 0.0, None)
gm = K.gilt()
black = K.ceramic((0.01, 0.01, 0.01), glaze=0.95, craze=0.0, name="black_glaze")
base = K.rbox("base", (0.11, 0.08, 0.018), r=0.008, spacing=0.008, loc=(0, 0, 0.009), mat=glaze)
K.warp(base, lambda c: (c.x, c.y * (1 - 0.35 * (c.x / 0.055) ** 2), c.z))
body = K.rbox("body", (0.07, 0.06, 0.09), r=0.028, spacing=0.008, bulge=(0.006, 0.006, 0.0), loc=(0, 0.008, 0.06), mat=glaze)
K.warp(body, lambda c: (c.x * (1 - 0.25 * (c.z + 0.045) / 0.09), c.y, c.z))
chest = K.rbox("chest", (0.05, 0.04, 0.06), r=0.02, spacing=0.007, loc=(0, -0.016, 0.085), mat=glaze)
head = K.rbox("head", (0.048, 0.05, 0.045), r=0.02, spacing=0.006, bulge=(0.004, 0, 0.004), loc=(0, -0.012, 0.142), mat=glaze)
muz = K.rbox("muzzle", (0.026, 0.03, 0.022), r=0.009, spacing=0.004, loc=(0, -0.040, 0.134), mat=glaze)
nose = K.rbox("nose", (0.010, 0.006, 0.007), r=0.003, spacing=0.002, loc=(0, -0.0545, 0.139), mat=black)
for s in (-1, 1):
    ear = K.rbox(f"ear{s}", (0.012, 0.03, 0.05), r=0.005, spacing=0.004, bulge=(0.002, 0, 0), loc=(s * 0.028, -0.006, 0.124),
                 rot=(0.1, s * 0.15, 0), mat=glaze)
    eye = K.rbox(f"eye{s}", (0.008, 0.004, 0.007), r=0.0019, spacing=0.0015, loc=(s * 0.012, -0.0365, 0.152), mat=black)
    leg = K.rbox(f"leg{s}", (0.016, 0.016, 0.075), r=0.007, spacing=0.005, loc=(s * 0.016, -0.026, 0.055), mat=glaze)
    paw = K.rbox(f"paw{s}", (0.018, 0.024, 0.012), r=0.005, spacing=0.004, loc=(s * 0.016, -0.032, 0.022), mat=glaze)
    haunch = K.rbox(f"haunch{s}", (0.022, 0.05, 0.035), r=0.011, spacing=0.006, loc=(s * 0.03, 0.012, 0.035), mat=glaze)
collar = K.tube("collar", [(0.024 * math.cos(a), -0.014 + 0.022 * math.sin(a), 0.118) for a in [math.tau * i / 20 for i in range(20)]],
                0.003, seg=6, closed=True, mat=gm)
chain = K.tube("chain", [(0.0, -0.036, 0.117), (0.006, -0.040, 0.10), (0.010, -0.040, 0.075), (0.012, -0.036, 0.05),
                         (0.012, -0.030, 0.03)], 0.002, seg=6, mat=gm)
