"""Model script: Victorian four-panel door, closed, with lining, architrave, plinth blocks and brass furniture.

Wall room face at y = 0 (room toward -Y); opening x +-0.447, z 0..2.045. Leaf from period_kit.frame_panel (2 x 2
fielded panels, deep lock rail), architrave from period_kit.moulding runs turned upright."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

LW, LH, LT = 0.84, 2.03, 0.045
OX, OZ = 0.447, 2.045
paint = K.painted_wood((0.40, 0.35, 0.26), wear=0.5, gloss=0.45)
oak_v = K.wood("dark_oak", "Z", wear=0.5, gloss=0.5, grime=0.6)
oak_h = K.wood("dark_oak", "X", wear=0.5, gloss=0.5, grime=0.6, offset=0.3)
oak_p = K.wood("dark_oak", "Z", wear=0.3, gloss=0.45, grime=0.7, offset=0.6)
brass = K.metal("brass", tarnish=0.7, polish=0.7)
parts = []

# ---- leaf (front face 0.012 behind the wall face, against the stops)
LY = 0.012 + LT / 2
parts += K.frame_panel("leaf", LW, LH, LT, (0, LY, 0.006), oak_v, oak_h, oak_p, cols=2, rows=2, stile=0.11, rail=0.11,
                       rail_bottom=0.21, mid=0.19, row_h=[0.52, 1.0], raise_w=0.045)

# ---- lining (jambs and head through the wall) and stops
for s in (-1, 1):
    parts.append(K.box(f"jamb{s}", (0.022, 0.15, OZ), (s * (OX - 0.011), 0.075, OZ / 2), bev=0.001, mat=paint))
    parts.append(K.box(f"stop{s}", (0.012, 0.03, OZ - 0.012), (s * (OX - 0.028), LY + LT / 2 + 0.015, (OZ - 0.012) / 2), bev=0.002, mat=paint))
parts.append(K.box("head", (2 * OX, 0.15, 0.022), (0, 0.075, OZ - 0.011), bev=0.001, mat=paint))

# ---- architrave (room side), plinth blocks
arch = [(0.0, 0.0), (-0.010, 0.0), (-0.016, 0.02), (-0.022, 0.045), (-0.018, 0.07), (-0.012, 0.08), (-0.012, 0.09),
        (0.0, 0.09)]
PB = 0.22
for s in (-1, 1):
    a = K.moulding(f"arch{s}", arch, PB, OZ, 0, 0, paint)
    a.rotation_euler = (0, -math.pi / 2, 0)
    a.location = (OX + 0.09 if s > 0 else -OX, 0.0, 0.0)
    parts.append(a)
    parts.append(K.box(f"plinth{s}", (0.10, 0.028, PB), (s * (OX + 0.045), -0.014, PB / 2), bev=0.003, mat=paint))
top = K.moulding("arch_top", arch, -OX - 0.09, OX + 0.09, 0, 0, paint)
top.location = (0, 0, OZ)
parts.append(top)

# ---- brass: knob on a rose, escutcheon, finger plate, hinge knuckles on the hinge side (-X)
FY = LY - LT / 2                        # leaf front face
KX, KZ = LW / 2 - 0.065, 0.95
rose = K.lathe("rose", K.keys([(0.0, 0.0001, "c"), (0.0, 0.026, "c"), (0.003, 0.026), (0.008, 0.018), (0.012, 0.009),
                                (0.012, 0.0001, "c")], 0.0015), seg=28, mat=brass)
rose.location, rose.rotation_euler = (KX, FY, KZ), (math.pi / 2, 0, 0)
knob = K.lathe("knob", K.keys([(0.0, 0.0001, "c"), (0.0, 0.007, "c"), (0.035, 0.006), (0.045, 0.022), (0.060, 0.026),
                                (0.072, 0.020), (0.076, 0.0001)], 0.002), seg=32, mat=brass)
knob.location, knob.rotation_euler = (KX, FY, KZ), (math.pi / 2, 0, 0)
esc = K.lathe("escutcheon", K.keys([(0.0, 0.0001, "c"), (0.0, 0.014, "c"), (0.002, 0.014), (0.004, 0.010),
                                     (0.004, 0.0001, "c")], 0.001), seg=24, mat=brass)
esc.location, esc.rotation_euler = (KX, FY, KZ - 0.085), (math.pi / 2, 0, 0)
keyhole = K.box("keyhole", (0.005, 0.004, 0.014), (KX, FY - 0.0035, KZ - 0.087), bev=0.0, mat=K.metal("iron", 0.9, polish=0.0))
plate = K.box("fingerplate", (0.075, 0.003, 0.30), (KX - 0.0, FY - 0.0015, 1.35), bev=0.001, seg=1, mat=brass)
parts += [rose, knob, esc, keyhole, plate]
for z in (0.25, 1.80):
    kn = K.lathe(f"hinge{z}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.0065, "c"), (0.10, 0.0065, "l"), (0.10, 0.0001, "c")],
                                      0.01), seg=12, mat=brass)
    kn.location = (-LW / 2 - 0.004, LY - LT / 2 + 0.004, z)
    fin = K.lathe(f"hfin{z}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.0065, "c"), (0.006, 0.004), (0.009, 0.0001)], 0.002),
                  seg=12, mat=brass)
    fin.location = (-LW / 2 - 0.004, LY - LT / 2 + 0.004, z + 0.10)
    parts += [kn, fin]
