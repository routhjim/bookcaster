"""Model script: a section of rough fieldstone wall facing for lining a room (front -Y, back flat at +Y against the
room's wall): rubble stones in rough courses over a recessed lime-mortar core, ~0.18 m deep. Width and height come
from the spec (SPEC["panel_w"], SPEC["panel_h"], metres; default 1.5 x 3.0); the stones at the side edges are cut
straight so sections butt together or into a corner. Soot darkens it upward from ~1.6 m, floor grime at the foot.
SPEC["seed"] varies the stone pattern.
"""

import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
W = float(SPEC.get("panel_w", 1.5))  # noqa: F821
H = float(SPEC.get("panel_h", 3.0))  # noqa: F821
rng = random.Random(int(SPEC.get("seed", 5)))  # noqa: F821
D, CORE = 0.18, 0.13
R = F.Rubble(rng, joint=0.02, amp=0.014, spacing=0.11)
core = K.box("core", (W, CORE, H), loc=(0, D / 2 - CORE / 2, H / 2), bev=0.0)
R.lay("y", -1, -D / 2, -W / 2, W / 2, F.Rubble.courses(0.0, H, rng, 0.17, 0.3), 0.12, lmin=0.2, lmax=0.55,
      ragged=0.05)
M_STONE = F.stone_material("rubble_stone", "stone", soot_z=(1.4, 3.0), soot=0.75)
M_MORTAR = F.stone_material("rubble_mortar", "mortar", soot_z=(1.4, 3.0), soot=0.75)
for o in R.stones:
    K.assign(o, M_STONE)
K.assign(core, M_MORTAR)
