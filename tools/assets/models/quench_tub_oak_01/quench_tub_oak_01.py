"""Model script: a blacksmith's quench tub (slack tub): a low coopered oak tub ~0.8 m across the rim, 0.42 m tall,
narrowing to ~0.7 m at the foot; 30 staves with uneven tops, two riveted iron hoops, filled to ~6 cm below the rim
with dark still water (a little scale and ash floating at the edges). Staves wet and dark toward the waterline inside
and at the foot outside.
"""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
H, R_TOP, R_FOOT = 0.42, 0.40, 0.35


def r_fn(z):
    t = z / H
    return R_FOOT + (R_TOP - R_FOOT) * t + 0.006 * math.sin(math.pi * t)


staves, bottom = F.coopered("tub_", H, r_fn, staves=30, thick=0.026, nz=14, na=3, top_jitter=0.007, seed=4)
hp = F.hoops("tub_", r_fn, [0.085, 0.31], width=0.042, thick=0.005)
rivets = []
for n, z in enumerate((0.085, 0.31)):
    pts, nrm = [], []
    for i in range(7):
        a = 2 * math.pi * (i + 0.5 * n) / 7
        r = r_fn(z) + 0.0065
        pts.append((r * math.cos(a), r * math.sin(a), z))
        nrm.append((math.cos(a), math.sin(a), 0))
    rivets += K.nails(f"rv{n}_", pts, r=0.007, normal=nrm)
wat = F.water("water", r_fn(H - 0.06) - 0.026 + 0.003, H - 0.06, ripple=0.0012)

M_WOOD = F.smithy_wood("tub_oak", axis="Z", base=(0.15, 0.095, 0.05), grey=0.35, wet=0.12, soot=0.15, wear=0.3)
M_IRON = F.forged_iron("tub_hoop_iron", scale_amt=0.5, rust=0.6, polish=0.1)
M_WATER = F.dark_water("tub_water")
for o in staves + [bottom]:
    K.assign(o, M_WOOD)
for o in hp + rivets:
    K.assign(o, M_IRON)
K.assign(wat, M_WATER)
