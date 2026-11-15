"""Model script: small hanging pine shelf with two rows of books; back flat on the wall plane."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

pz = K.wood("pine", "Z", wear=0.5, gloss=0.2, grime=0.8)
px = K.wood("pine", "X", wear=0.5, gloss=0.2, grime=0.8, offset=0.3)
W, D, H = 0.70, 0.18, 0.55
side = [(0.0, 0.0), (0.18, 0.0), (0.18, 0.05), (0.12, 0.10), (0.10, 0.20), (0.18, 0.27), (0.18, 0.33), (0.11, 0.40),
        (0.10, 0.48), (0.04, 0.55), (0.0, 0.55)]
for s in (-1, 1):
    g = K.plate(f"side{s}", [(D / 2 - u, v) for u, v in side], 0.018, mat=pz)
    g.rotation_euler, g.location = (0, 0, math.pi / 2), (s * (W / 2 - 0.009), 0.0, 0.0)
for z in (0.0, 0.27):
    K.box(f"shelf{z}", (W - 0.036, D, 0.016), (0, 0.0, z + 0.008), bev=0.002, mat=px)
K.box("rail_top", (W - 0.036, 0.016, 0.06), (0, D / 2 - 0.008, H - 0.06), bev=0.002, mat=px)
K.box("rail_back", (W - 0.036, 0.012, 0.04), (0, D / 2 - 0.006, 0.20), bev=0.002, mat=px)
for i, z in enumerate((0.016, 0.286)):
    K.book_row(f"bk{i}", -W / 2 + 0.02, W / 2 - 0.02, D / 2 - 0.002, z, D - 0.01, h_range=(0.16, 0.22), seed=21 + i,
               stacks=i)
