"""Model script: a half-forged straight sword blade lying flat along X, ~0.82 m (tip to +X): a double-edged blade
~0.66 m long, ~52 mm wide at the shoulders narrowing to ~38 mm before a rough rounded point, still thick (edges not
drawn out: ~3 mm at the edge, ~8 mm at the centre), uneven and wavy from the hammer; a short rough tang ~0.16 m
tapering from the shoulders. Dark hammer-scaled iron with blue-grey scale.
"""

import math
import os
import random
import sys

from mathutils import Vector, noise

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
rng = random.Random(21)
X_TANG0, X_SH, X_TIP = -0.41, -0.25, 0.41
N = 120
path, secs = [], []
for i in range(N + 1):
    t = i / N
    x = K.lerp(X_TANG0, X_TIP, t)
    wav = 0.0015 * noise.noise(Vector((x * 8, 0.3, 0.0)))
    if x < X_SH:                                    # tang: square-ish, tapering toward its end
        u = (x - X_TANG0) / (X_SH - X_TANG0)
        hw, ht = K.lerp(0.0065, 0.010, u), K.lerp(0.0035, 0.0045, u)
        n = 6.0
    else:
        u = (x - X_SH) / (X_TIP - X_SH)
        sh = K.smooth(min(1.0, (x - X_SH) / 0.025))         # shoulders flare out quickly
        hw = K.lerp(0.010, 0.026, sh) * K.lerp(1.0, 0.73, u)
        if u > 0.88:                                         # rough rounded point
            hw *= math.sqrt(max(0.0, 1 - ((u - 0.88) / 0.12) ** 2)) * 0.97 + 0.03
        hw += 0.0012 * noise.noise(Vector((x * 30, 1.7, 0.0)))
        ht = K.lerp(0.0045, 0.0038, u) * (0.6 + 0.4 * (hw / 0.026))
        n = 1.3                                              # lozenge section: thick centre, thick-ish edges
    path.append((x, wav, 0.0))
    secs.append(K.superellipse2d(max(ht, 0.0012), max(hw, 0.0015), n, 0.0, 20))
blade = K.sweep("blade", path, secs, up=(0, 0, 1), angle=35)
# hammer unevenness on the faces
for v in blade.data.vertices:
    v.co.z += 0.00045 * noise.noise(v.co * 90.0) * (1 if abs(v.co.z) > 0.0015 else 0.3)
blade.data.update()
lift = min(v.co.z for v in blade.data.vertices)
blade.location.z = -lift
M = F.forged_iron("blade_iron", scale_amt=0.9, rust=0.15, polish=0.12)
K.assign(blade, M)
