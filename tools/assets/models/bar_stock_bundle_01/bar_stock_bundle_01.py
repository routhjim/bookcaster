"""Model script: a bundle of iron bar stock lying on the floor along X, ~1.2 m: nine bars (square, round and flat)
in two layers, slightly bent and out of line, tied with twisted wire loops ~0.18 m from each end. Mill scale and
orange rust.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
rng = random.Random(12)
bars = []
# (y centre, z centre, w (Y), h (Z), roundness)
lay = [(-0.050, 0.010, 0.020, 0.020, 0.15), (-0.027, 0.010, 0.020, 0.020, 1.0), (-0.004, 0.006, 0.040, 0.012, 0.1),
       (0.020, 0.010, 0.020, 0.020, 0.15), (0.043, 0.0095, 0.019, 0.019, 1.0),
       (-0.038, 0.030, 0.020, 0.020, 0.15), (-0.015, 0.026, 0.025, 0.012, 0.1), (0.010, 0.0305, 0.021, 0.021, 1.0),
       (0.032, 0.030, 0.020, 0.020, 0.15)]
for k, (y, z, w, h, rnd) in enumerate(lay):
    L = rng.uniform(1.12, 1.2)
    x0 = rng.uniform(-0.6, -0.56)
    sag = rng.uniform(-0.004, 0.004)
    path = [(x0 + L * t, y + sag * math.sin(math.pi * t) + rng.uniform(-0.0005, 0.0005), z) for t in
            [i / 24 for i in range(25)]]
    bars.append(F.bar(f"bar{k}", path, h, w, rnd=rnd, seg=16 if rnd > 0.5 else 12))
wires = []
for xw in (-0.40, 0.40):
    pts = []
    for i in range(41):
        a = 2 * math.pi * i / 40
        pts.append((xw + 0.003 * math.sin(a * 9), 0.066 * math.cos(a), 0.0205 + 0.024 * math.sin(a)))
    w = K.tube(f"wire{xw}", pts[:-1], 0.0022, seg=6, closed=True)
    wires.append(w)
M = F.forged_iron("bar_iron", scale_amt=0.85, rust=0.6, polish=0.05)
MW = F.forged_iron("wire_iron", scale_amt=0.3, rust=0.8, polish=0.0)
for o in bars:
    K.assign(o, M)
for o in wires:
    K.assign(o, MW)
