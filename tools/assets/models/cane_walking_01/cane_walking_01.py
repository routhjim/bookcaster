"""Model script: crook-handled walking cane lying on the floor along X (tip +X, crook -X curling to +Y).

One sweep for shaft + crook (round section, tapering, slight node swellings), lathe collar and ferrule."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

cane = K.wood("satinwood", "X", wear=0.6, gloss=0.7, grime=0.7)
silver = K.metal("nickel", tarnish=0.7, polish=0.7)
brass = K.metal("brass", tarnish=0.6, polish=0.6)
R0, R1 = 0.0125, 0.0095
X_TIP, X_CROOK = 0.445, -0.36
pts, rad = [], []
n = 80
for k in range(n + 1):                        # shaft from the tip back to the crook start
    x = K.lerp(X_TIP - 0.03, X_CROOK, k / n)
    t = (X_TIP - x) / (X_TIP - X_CROOK)
    node = 0.0007 * math.exp(-((((x * 100) % 22) - 11) / 1.2) ** 2)
    pts.append((x, 0.0, 0.0))
    rad.append(K.lerp(R1, R0, t) + node)
CR = 0.048                                     # crook: half circle of radius CR then a short return
for k in range(1, 31):
    a = math.pi * k / 30
    pts.append((X_CROOK - CR * math.sin(a), CR - CR * math.cos(a), 0.0))
    rad.append(R0 * (1 - 0.05 * math.sin(a)))
for k in range(1, 9):
    pts.append((X_CROOK + 0.006 * k, 2 * CR, 0.0))
    rad.append(R0 * (1 - 0.15 * k / 8))
secs = [K.superellipse2d(r, r, 2.0, 0.0, 16) for r in rad]
z = R0
shaft = K.sweep("cane", [(x, y, z) for x, y, _ in pts], secs, up=(0, 0, 1), mat=cane, angle=60)
col = K.lathe("collar", K.keys([(0.0, 0.0001, "c"), (0.0, 0.0135, "c"), (0.003, 0.0142), (0.019, 0.0142, "l"),
                                (0.022, 0.0135), (0.022, 0.0001, "c")], 0.001), seg=24, mat=silver)
col.location, col.rotation_euler = (X_CROOK + 0.03, 0.0, z), (0, -math.pi / 2, 0)
fer = K.lathe("ferrule", K.keys([(0.0, 0.0001, "c"), (0.0, 0.0102, "c"), (0.035, 0.0104, "l"), (0.040, 0.0090),
                                 (0.042, 0.0001, "c")], 0.001), seg=20, mat=brass)
fer.location, fer.rotation_euler = (X_TIP - 0.042, 0.0, z), (0, math.pi / 2, 0)
