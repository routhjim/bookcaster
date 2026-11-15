"""Model script: a small blacksmith's fire shovel lying flat along X, ~0.86 m: a flat hand-forged shield-shaped blade ~0.15 m wide
and 0.17 m long coming to a point, with a short turned-up back lip, a thin square shaft with a twisted section, ending in a ring. Blade
at -X, ring at +X. Sooty black iron, worn bright on the twist and ring.
"""

import math
import os
import sys

import bmesh

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
BX0, BX1, BW, T = -0.43, -0.26, 0.15, 0.003
SIDE_H = 0.028


def blade():
    """A sheet: flat floor, sides and back turned up (one bent plate, solidified)."""
    bm = bmesh.new()
    nx, ny = 12, 12
    grid = []
    for i in range(nx + 1):
        row = []
        for j in range(ny + 1):
            x = K.lerp(BX0, BX1, i / nx)
            y = K.lerp(-BW / 2, BW / 2, j / ny)
            u = (x - BX0) / (BX1 - BX0)                                       # 0 at the point, 1 at the back
            taper = max(0.02, math.sin(math.pi / 2 * min(1.0, u / 0.7)) ** 0.8)  # shield outline to a point
            up = SIDE_H * 0.08 * (abs(y) / (BW / 2)) ** 2                      # very slightly dished
            up = max(up, SIDE_H * 0.9 * K.smooth((x - (BX1 - 0.012)) / 0.012))  # short turned-up back lip
            row.append(bm.verts.new((x, y * taper, T / 2 + up)))
        grid.append(row)
    for i in range(nx):
        for j in range(ny):
            bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
    ob = K.from_bm("blade", bm)
    s = ob.modifiers.new("t", "SOLIDIFY")
    s.thickness, s.offset = T, 0.0
    K.shade(ob, 40)
    return ob


b = blade()
# shaft: from the back of the blade, rising to sit on the blade's back edge, then flat along the floor
SZ = 0.0065
path = [(BX1 - 0.03, 0.0, T + 0.004), (BX1 + 0.01, 0.0, SZ + 0.012), (BX1 + 0.05, 0.0, SZ)]
path += [(K.lerp(BX1 + 0.05, 0.36, i / 40), 0.0, SZ) for i in range(1, 41)]
shaft = F.bar("shaft", path, 0.011, 0.011, twist=0.0, rnd=0.2)
twist = F.bar("twist", [(K.lerp(0.18, 0.33, i / 30), 0.0, SZ) for i in range(31)], 0.0125, 0.0125, twist=2.0, rnd=0.1)
ring = K.tube("ring", [(0.36 + 0.035 + 0.035 * math.cos(math.pi + 2 * math.pi * i / 32), 0.035 * math.sin(2 * math.pi * i / 32), SZ)
                       for i in range(32)], 0.0055, seg=10, closed=True)
M = F.forged_iron("shovel_iron", scale_amt=0.7, rust=0.3, polish=0.35, soot=0.5)
for o in (b, shaft, twist, ring):
    K.assign(o, M)
