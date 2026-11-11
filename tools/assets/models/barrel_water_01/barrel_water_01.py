"""Model script: an upright open water barrel, ~0.62 m across at the belly, 0.86 m tall; 24 bulging oak staves,
four riveted iron hoops (at the rims and a quarter in), open top with dark water ~5 cm below the
rim. Weathered grey-brown, dark wet staining at the foot and below the rim.
"""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
H, R_END, R_BELLY = 0.86, 0.255, 0.31


def r_fn(z):
    t = z / H
    return R_END + (R_BELLY - R_END) * math.sin(math.pi * t) ** 0.9


staves, bottom = F.coopered("brl_", H, r_fn, staves=24, thick=0.024, nz=20, na=3, top_jitter=0.004, bottom_z=0.06,
                            seed=9)
HZ = [0.045, 0.24, H - 0.21, H - 0.032]
hp = F.hoops("brl_", r_fn, HZ, width=0.046, thick=0.0045)
rivets = []
for n, z in enumerate(HZ):
    pts, nrm = [], []
    for i in range(8):
        a = 2 * math.pi * (i + 0.5 * (n % 2)) / 8
        r = r_fn(z) + 0.0065
        pts.append((r * math.cos(a), r * math.sin(a), z))
        nrm.append((math.cos(a), math.sin(a), 0))
    rivets += K.nails(f"rv{n}_", pts, r=0.0065, normal=nrm)
wat = F.water("water", r_fn(H - 0.055) - 0.024 + 0.003, H - 0.055, ripple=0.0012)
M_WOOD = F.smithy_wood("barrel_oak", axis="Z", base=(0.13, 0.085, 0.05), grey=0.55, wet=0.2, soot=0.1, wear=0.25)
M_IRON = F.forged_iron("barrel_hoop_iron", scale_amt=0.4, rust=0.7, polish=0.05)
M_WATER = F.dark_water("barrel_water", scum=0.2)
for o in staves + [bottom]:
    K.assign(o, M_WOOD)
for o in hp + rivets:
    K.assign(o, M_IRON)
K.assign(wat, M_WATER)
