"""Model script: Victorian dressing stool: leather stuffed top with nails, seat rails, cabriole legs. Front -Y."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

W, D, H = 0.50, 0.36, 0.45
leather = K.fabric("leather", (0.06, 0.028, 0.013), wear=0.8)
wv = K.wood("mahogany", "Z", wear=0.6, gloss=0.6, grime=0.6)
wx = K.wood("mahogany", "X", wear=0.5, gloss=0.6, grime=0.6, offset=0.3)
wy = K.wood("mahogany", "Y", wear=0.5, gloss=0.6, grime=0.6, offset=0.5)
brass = K.metal("brass", tarnish=0.75, polish=0.5)
RZ0, RZ1 = 0.30, 0.37
parts = []
for sx in (-1, 1):
    for sy in (-1, 1):
        lg = K.cabriole(f"leg{sx}{sy}", RZ1, knee=0.026, top=0.022, ankle=0.012, foot=0.018, out=0.022, mat=wv)
        lg.location, lg.rotation_euler = (sx * (W / 2 - 0.03), sy * (D / 2 - 0.03), 0.0), (0, 0, math.atan2(sx, -sy))
        parts.append(lg)
for sy in (-1, 1):
    r = K.box(f"rail_x{sy}", (W - 0.06, 0.02, RZ1 - RZ0), (0, sy * (D / 2 - 0.03), (RZ0 + RZ1) / 2), bev=0.002, mat=wx)
    K.warp(r, lambda c: (c.x, c.y, c.z - (0.012 * math.cos(math.pi * c.x / 0.22) if c.z < 0 else 0.0)))
    parts.append(r)
for sx in (-1, 1):
    parts.append(K.box(f"rail_y{sx}", (0.02, D - 0.06, RZ1 - RZ0), (sx * (W / 2 - 0.03), 0, (RZ0 + RZ1) / 2), bev=0.002, mat=wy))
top = K.rbox("top", (W, D, H - RZ1 + 0.008), r=0.02, spacing=0.018, bulge=(0.004, 0.004, 0.014), loc=(0, 0, (H + RZ1) / 2 - 0.004),
             mat=leather)
parts.append(top)
pts, nrm = [], []
zN = RZ1 + 0.004
for k in range(17):
    x = K.lerp(-W / 2 + 0.02, W / 2 - 0.02, k / 16)
    for sy in (-1, 1):
        pts.append((x, sy * (D / 2 + 0.0035), zN)); nrm.append((0, sy, 0))
for k in range(1, 11):
    y = K.lerp(-D / 2 + 0.02, D / 2 - 0.02, k / 11)
    for sx in (-1, 1):
        pts.append((sx * (W / 2 + 0.0035), y, zN)); nrm.append((sx, 0, 0))
parts += K.nails("nail", pts, 0.0052, brass, nrm)
