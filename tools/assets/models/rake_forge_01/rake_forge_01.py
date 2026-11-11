"""Model script: a blacksmith's fire rake lying flat along X, ~0.86 m: a thin square iron shaft with a ring handle
at +X; at -X the shaft is drawn out flat and bent at a right angle into a short flat blade (~0.1 m long, 35 mm wide)
lying in the floor plane (pointing -Y). Sooty black iron.
"""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
SZ = 0.006
X0 = -0.42
shaft = F.bar("shaft", [(K.lerp(X0 + 0.012, 0.36, i / 50), 0.0, SZ) for i in range(51)], 0.011, 0.011, rnd=0.25)
# bend: the flattened end turns -Y; section widens (in plane) and thins (in height)
path, secs = [], []
for i in range(25):
    t = i / 24
    a = math.pi / 2 * K.smooth(min(1, t / 0.35))
    if t < 0.35:
        r = 0.02
        x = X0 + 0.012 - r * math.sin(a)
        y = -(r - r * math.cos(a))
    else:
        x = X0 + 0.012 - 0.02
        y = -0.02 - 0.09 * (t - 0.35) / 0.65
    path.append((x, y, SZ))
    w = K.lerp(0.0055, 0.0175, K.smooth(t / 0.4))
    h = K.lerp(0.0055, 0.0028, K.smooth(t / 0.4))
    secs.append(K.superellipse2d(h, w, 5.0, 0.0, 14))
path = list(reversed(path))
secs = list(reversed(secs))
blade = K.sweep("blade", path, secs, up=(0, 0, 1), angle=40)
ring = K.tube("ring", [(0.36 + 0.035 + 0.035 * math.cos(math.pi + 2 * math.pi * i / 32), 0.035 * math.sin(2 * math.pi * i / 32), SZ)
                       for i in range(32)], 0.0055, seg=10, closed=True)
M = F.forged_iron("rake_iron", scale_amt=0.7, rust=0.3, polish=0.35, soot=0.5)
for o in (shaft, blade, ring):
    K.assign(o, M)
