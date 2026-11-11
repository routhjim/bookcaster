"""Model script: a heavy blacksmith's workbench with a leg (post) vise, ~2.0 m long (X), 0.7 m deep, top at 0.86 m.

Top: two thick rough oak planks (with a gap), on four stout square legs joined by aprons and low stretchers.
Vise on the front left corner (x = -0.72, front -Y): a fixed iron leg standing on the floor beside the bench leg
and strapped to the top, a moving leg hinged low and opening a little at the jaws, square jaws just above the top,
a screw through both legs with a screw box behind and a tommy bar handle in front, a leaf spring between the legs.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
rng = random.Random(5)
L, D, TOP, TT = 2.0, 0.7, 0.86, 0.085
LEG = 0.13
wood_x, wood_z, irons = [], [], []

# top: two planks, rough-hewn (slightly uneven thickness), small gap between
for k, (y0, y1) in enumerate(((-D / 2, -0.012), (0.004, D / 2))):
    p = K.rbox(f"plank{k}", (L, y1 - y0, TT), r=0.008, spacing=0.06, bulge=(0, 0, 0.003),
               loc=(0, (y0 + y1) / 2, TOP - TT / 2))
    K.warp(p, lambda v: (v.x, v.y, v.z + 0.003 * math.sin(v.x * 3.1 + k) - 0.002 * (v.x / L) ** 2))
    wood_x.append(p)
# legs
for sx in (-1, 1):
    for sy in (-1, 1):
        x, y = sx * (L / 2 - 0.16), sy * (D / 2 - 0.09)
        wood_z.append(K.box("leg", (LEG, LEG, TOP - TT), loc=(x, y, (TOP - TT) / 2), bev=0.008))
# aprons under the top and low stretchers
for sy in (-1, 1):
    y = sy * (D / 2 - 0.09)
    wood_x.append(K.box("apron", (L - 0.32 - LEG, 0.05, 0.12), loc=(0, y, TOP - TT - 0.06), bev=0.005))
    wood_x.append(K.box("stretcher", (L - 0.32 - LEG, 0.06, 0.08), loc=(0, y, 0.16), bev=0.006))
for sx in (-1, 1):
    x = sx * (L / 2 - 0.16)
    wood_x.append(K.box("rail", (0.06, D - 0.18 - LEG, 0.08), loc=(x, 0, 0.16), bev=0.006))
    wood_x.append(K.box("rail_top", (0.05, D - 0.18 - LEG, 0.12), loc=(x, 0, TOP - TT - 0.06), bev=0.005))
# a low shelf board resting on both stretchers
wood_x.append(K.box("shelf", (L - 0.5, 0.58, 0.03), loc=(0, 0.0, 0.215), bev=0.004))

# ------------------------------------------------------------------ leg vise
VX = -0.72
YF = -D / 2                              # bench front face
JZ = TOP + 0.075                         # jaw top
SZ = TOP - 0.12                          # screw height
fixed = F.bar("vise_fixed", [(VX, YF - 0.022, 0.0), (VX, YF - 0.022, JZ - 0.07)], 0.05, 0.034, rnd=0.15)
irons.append(fixed)
foot = K.box("vise_foot", (0.09, 0.08, 0.02), loc=(VX, YF - 0.03, 0.01), bev=0.004)
irons.append(foot)
# moving leg: hinged at z 0.14, leaning forward so the jaws stand 12 mm open
hz = 0.14
open_ = 0.012
mv = []
for i in range(25):
    t = i / 24
    z = K.lerp(hz, JZ - 0.07, t)
    mv.append((VX, YF - 0.022 - 0.044 - open_ * (z - hz) / (JZ - 0.07 - hz), z))
irons.append(F.bar("vise_moving", mv, 0.05, 0.034, rnd=0.15))
irons.append(K.lathe("hinge_pin", [(0.0001, -0.05), (0.011, -0.05), (0.011, 0.05), (0.0, 0.05)], seg=16))
irons[-1].location, irons[-1].rotation_euler = (VX, YF - 0.044, hz), (0, math.pi / 2, 0)
irons.append(K.box("hinge_strap", (0.075, 0.07, 0.04), loc=(VX, YF - 0.044, hz), bev=0.004))
# jaws: square blocks, a little wider than the legs, chequered faces (in the material)
irons.append(K.box("jaw_fixed", (0.13, 0.034, 0.075), loc=(VX, YF - 0.022, JZ - 0.0375), bev=0.003))
irons.append(K.box("jaw_moving", (0.13, 0.034, 0.075), loc=(VX, YF - 0.066 - open_, JZ - 0.0375), bev=0.003))
# screw: along Y through both legs; screw box behind the fixed leg; boss and tommy bar in front
ys0, ys1 = YF + 0.06, YF - 0.13
irons.append(K.lathe("screw", [(0.0001, 0.0), (0.014, 0.0), (0.014, ys0 - ys1), (0.0, ys0 - ys1)], seg=20))
irons[-1].location, irons[-1].rotation_euler = (VX, ys1, SZ), (-math.pi / 2, 0, 0)
irons.append(K.lathe("screw_box", [(0.0001, 0.0), (0.024, 0.0), (0.026, 0.01), (0.022, 0.02), (0.022, 0.07),
                                    (0.027, 0.075), (0.0, 0.08)], seg=24))
irons[-1].location, irons[-1].rotation_euler = (VX, YF - 0.006, SZ), (-math.pi / 2, 0, 0)
irons.append(K.lathe("screw_boss", [(0.0001, 0.0), (0.03, 0.0), (0.03, 0.035), (0.0, 0.035)], seg=24))
irons[-1].location, irons[-1].rotation_euler = (VX, ys1 + 0.035, SZ), (math.pi / 2, 0, 0)
bar_y = ys1 + 0.018
irons.append(K.tube("tommy", [(VX - 0.19, bar_y, SZ - 0.02), (VX, bar_y, SZ), (VX + 0.19, bar_y, SZ + 0.02)], 0.009,
                    seg=12))
for sx in (-1, 1):
    b = K.lathe("knob", [(0.0001, -0.014), (0.016, -0.01), (0.016, 0.01), (0.0, 0.014)], seg=16)
    b.location, b.rotation_euler = (VX + sx * 0.2, bar_y, SZ + sx * 0.021), (0, math.pi / 2, 0)
    irons.append(b)
# leaf spring between the legs, below the screw
irons.append(K.tube("spring", [(VX, YF - 0.04, SZ - 0.035), (VX, YF - 0.046, SZ - 0.1), (VX, YF - 0.052, SZ - 0.16)],
                    0.006, seg=10))
# mounting strap from the fixed leg back over the top, held by a wedge
irons.append(K.box("strap_side", (0.05, 0.03, 0.02), loc=(VX, YF - 0.0, TOP - 0.02), bev=0.003))
irons.append(K.box("strap_top", (0.05, 0.18, 0.008), loc=(VX, YF + 0.09, TOP + 0.003), bev=0.002))
irons.append(K.box("wedge", (0.03, 0.05, 0.03), loc=(VX, YF + 0.17, TOP + 0.018), bev=0.003))
irons += K.nails("strap_nail", [(VX - 0.015, YF + 0.05, TOP + 0.007), (VX + 0.015, YF + 0.12, TOP + 0.007)], r=0.007,
                 normal=(0, 0, 1))

M_TOPW = F.smithy_wood("bench_top", axis="X", base=(0.14, 0.085, 0.045), grey=0.3, soot=0.45, wear=0.5, burns=0.8)
M_LEGW = F.smithy_wood("bench_legs", axis="Z", base=(0.14, 0.085, 0.045), grey=0.3, soot=0.3, wear=0.3)
M_RAILW = F.smithy_wood("bench_rails", axis="X", base=(0.14, 0.085, 0.045), grey=0.3, soot=0.3, wear=0.3)
M_IRON = F.forged_iron("vise_iron", scale_amt=0.6, rust=0.35, polish=0.4)
for o in wood_x:
    K.assign(o, M_TOPW if o.name.startswith("plank") else M_RAILW)
for o in wood_z:
    K.assign(o, M_LEGW)
for o in irons:
    K.assign(o, M_IRON)
