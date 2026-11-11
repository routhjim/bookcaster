"""Model script: a square iron bar fresh from the fire, lying flat along X, ~0.8 m x 16 mm: the -X end glows
(emissive: yellow-white at the tip through orange and cherry red to dull red ~0.2 m back), the rest black scaled
iron. Lay its hot end in a forge's fire or on an anvil face.
"""

import os
import sys

from mathutils import Vector, noise

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
L, S = 0.8, 0.016
path = [(-L / 2 + L * t / 40, 0.0, S / 2) for t in range(41)]
bar = F.bar("bar", path, S, S, rnd=0.12, seg=16,
            w_fn=lambda t: 0.9 + 0.1 * min(1.0, t / 0.03))          # the hot tip a little drawn down
for v in bar.data.vertices:                                         # hammer unevenness
    v.co.y += 0.0004 * noise.noise(Vector((v.co.x * 60, 0.3, 0.0)))
    v.co.z += 0.0004 * noise.noise(Vector((v.co.x * 60, 1.7, 0.0)))
bar.data.update()
M = F.forged_iron("hot_bar_iron", scale_amt=0.85, rust=0.1, polish=0.1, hot=("X", -L / 2, -L / 2 + 0.22))
K.assign(bar, M)
