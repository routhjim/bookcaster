"""Model script: Victorian bedside pot cupboard: gallery top, drawer, panelled door, plinth. Front -Y."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

W, D, H = 0.40, 0.36, 0.73
wv = K.wood("mahogany", "Z", wear=0.5, gloss=0.5, grime=0.6)
wx = K.wood("mahogany", "X", wear=0.6, gloss=0.55, grime=0.6, offset=0.3)
wy = K.wood("mahogany", "Y", wear=0.5, gloss=0.5, grime=0.6, offset=0.6)
brass = K.metal("brass", tarnish=0.7, polish=0.5)
FY = -D / 2
for s in (-1, 1):
    K.box(f"side{s}", (0.018, D - 0.02, H - 0.08), (s * (W / 2 - 0.009), 0.01, 0.08 + (H - 0.08) / 2), bev=0.002, mat=wv)
K.box("back", (W - 0.036, 0.01, H - 0.08), (0, D / 2 - 0.005, 0.08 + (H - 0.08) / 2), bev=0.001, mat=wv)
K.box("plinth", (W + 0.01, D + 0.005, 0.08), (0, 0.0, 0.04), bev=0.004, seg=2, mat=wx)
K.box("top", (W + 0.02, D + 0.02, 0.02), (0, 0.0, H + 0.01), bev=0.004, seg=3, mat=wx)
gal = [(0.0, 0.0), (0.0, 0.05), (0.02, 0.055), (0.05, 0.04), (0.08, 0.03), (0.08, 0.0)]
for s in (-1, 1):
    g = K.plate(f"gal{s}", [(D / 2 - u - 0.01, H + 0.02 + v) for u, v in [(0, 0), (0, 0.06), (0.03, 0.055), (0.12, 0.03), (0.16, 0.0)]],
                0.012, mat=wy)
    g.rotation_euler = (0, 0, math.pi / 2)
    g.location = (s * (W / 2 - 0.006), 0.0, 0.0)
K.plate("gal_b", [(-W / 2, H + 0.02), (W / 2, H + 0.02), (W / 2, H + 0.08), (0.12, H + 0.075), (0.0, H + 0.09),
                  (-0.12, H + 0.075), (-W / 2, H + 0.08)], 0.012, y=D / 2 - 0.006, mat=wx)
K.box("drawer", (W - 0.05, 0.02, 0.10), (0, FY + 0.004, H - 0.06), bev=0.003, seg=2, mat=wx)
K.box("rail", (W - 0.03, 0.02, 0.02), (0, FY + 0.012, H - 0.12), bev=0.002, mat=wx)
K.frame_panel("door", W - 0.05, H - 0.08 - 0.14, 0.02, (0, FY + 0.012, 0.085), wv, wx, wy, stile=0.045, rail=0.045, raise_w=0.025)
for z in (H - 0.06, 0.38):
    kn = K.lathe(f"knob{z:.2f}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.008, "c"), (0.008, 0.006), (0.014, 0.011), (0.020, 0.012),
                                         (0.024, 0.0001)], 0.0015), seg=16, mat=wv)
    kn.location, kn.rotation_euler = (0.0 if z > 0.5 else W / 2 - 0.06, FY - 0.004, z), (math.pi / 2, 0, 0)
