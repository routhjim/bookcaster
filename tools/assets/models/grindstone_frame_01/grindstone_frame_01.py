"""Model script: a grindstone: a sandstone wheel ~0.62 m across and 0.15 m thick, iron hub flanges bolted on both
faces, on a square iron axle (along X) resting on the rims of a heavy timber trough, crank handle on the right (+X);
the trough (planks two high, water inside the wheel dips into) stands on four splayed legs. ~0.9 m wide (X, with the
crank), 0.86 m long (Y), wheel top ~0.94 m.
"""

import math
import os
import random
import sys

from mathutils import Vector, noise

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
rng = random.Random(4)
R, W = 0.31, 0.15
AZ = 0.63                                # axle height (it rests on the trough's side rims)
TX, TY = 0.30, 0.38                      # trough half sizes (outer): X along the axle, Y across
TZ0, TZ1 = 0.29, 0.62                    # trough bottom and rim
timber_y, timber_z, timber_x, irons = [], [], [], []

# wheel: lathe about Z, then turned to the X axis; dressed rough faces, rim worn slightly out of round
prof = [(0.03, -W / 2), (R - 0.014, -W / 2), (R - 0.003, -W / 2 + 0.01), (R, -W / 2 + 0.025), (R, W / 2 - 0.025),
        (R - 0.003, W / 2 - 0.01), (R - 0.014, W / 2), (0.03, W / 2), (0.03, -W / 2)]
wheel = K.lathe("wheel", prof, seg=96, cap0=False, cap1=False)
for v in wheel.data.vertices:
    r = math.hypot(v.co.x, v.co.y)
    if r > 0.05:
        a = math.atan2(v.co.y, v.co.x)
        k = 1 + (0.006 * noise.noise(Vector((math.cos(a) * 2, math.sin(a) * 2, 0.5))) +
                 0.002 * noise.noise(Vector((math.cos(a) * 9, math.sin(a) * 9, v.co.z * 30)))) * (r / R)
        v.co.x *= k
        v.co.y *= k
        if abs(v.co.z) > W / 2 - 0.001:                   # pecked faces
            v.co.z += 0.002 * noise.noise(Vector((v.co.x * 40, v.co.y * 40, 0.0))) * (1 if v.co.z > 0 else -1)
wheel.data.update()
wheel.location, wheel.rotation_euler = (0, 0, AZ), (0, math.pi / 2, 0)
K.shade(wheel, 50)
# axle (square), hub flanges with bolt heads on both faces, crank
irons.append(K.box("axle", (2 * TX + 0.16, 0.032, 0.032), loc=(0.05, 0, AZ), bev=0.002))
for sx in (-1, 1):
    fl = K.lathe("flange", [(0.0001, 0.0), (0.075, 0.0), (0.075, 0.012), (0.03, 0.016), (0.0, 0.016)], seg=32)
    fl.location, fl.rotation_euler = (sx * (W / 2 - 0.001), 0, AZ), (0, sx * math.pi / 2, 0)
    irons.append(fl)
    pts = [(sx * (W / 2 + 0.013), 0.055 * math.cos(2 * math.pi * k / 6), AZ + 0.055 * math.sin(2 * math.pi * k / 6))
           for k in range(6)]
    irons += K.nails(f"bolt{sx}", pts, r=0.009, normal=(sx, 0, 0))
    irons.append(K.box("nut", (0.02, 0.05, 0.05), loc=(sx * (W / 2 + 0.024), 0, AZ), bev=0.003))
CX = TX + 0.12
irons.append(F.bar("crank_arm", [(CX, 0, AZ), (CX, 0, AZ + 0.06), (CX, 0, AZ + 0.17)], 0.028, 0.014, rnd=0.2))
h = K.lathe("crank_grip", [(0.0001, 0.0), (0.014, 0.0), (0.019, 0.04), (0.019, 0.11), (0.015, 0.15), (0.0, 0.15)],
            seg=16)
h.location, h.rotation_euler = (CX + 0.004, 0, AZ + 0.165), (0, math.pi / 2, 0)
irons.append(h)
# trough: thick planks (two high on each side) with a bottom board; the axle rests in notches on the X-side rims
for sx in (-1, 1):
    for k, (z0, z1) in enumerate(((TZ0, TZ0 + 0.17), (TZ0 + 0.17, TZ1))):
        timber_y.append(K.box("side", (0.05, 2 * TY, z1 - z0 - 0.004), loc=(sx * (TX - 0.025), 0, (z0 + z1) / 2),
                              bev=0.006))
for sy in (-1, 1):
    for k, (z0, z1) in enumerate(((TZ0, TZ0 + 0.17), (TZ0 + 0.17, TZ1))):
        timber_x.append(K.box("end", (2 * TX - 0.1, 0.05, z1 - z0 - 0.004), loc=(0, sy * (TY - 0.025), (z0 + z1) / 2),
                              bev=0.006))
timber_y.append(K.box("bottom", (2 * TX - 0.1, 2 * TY - 0.1, 0.03), loc=(0, 0, TZ0 + 0.015), bev=0.004))
for sx in (-1, 1):                         # bearing blocks on the rims under the axle
    timber_z.append(K.box("bearing", (0.07, 0.12, 0.02), loc=(sx * (TX - 0.025), 0, TZ1 + 0.002 - 0.004), bev=0.004))
# legs: splayed, from under the trough corners to the floor
for sx in (-1, 1):
    for sy in (-1, 1):
        top = Vector((sx * (TX - 0.06), sy * (TY - 0.07), TZ0 + 0.02))
        foot = Vector((sx * (TX + 0.06), sy * (TY + 0.05), 0.0))
        timber_z.append(F.bar("leg", [top.lerp(foot, t / 8) for t in range(9)], 0.08, 0.08, rnd=0.1))
# water in the trough the wheel dips into
trough = []
wat = K.box("water", (2 * TX - 0.1, 2 * TY - 0.1, 0.01), loc=(0, 0, AZ - R + 0.03), bev=0.0)

M_STONE_K = K.NK("grindstone_sandstone")
k = M_STONE_K
s = k.N.new("ShaderNodeSeparateXYZ")
k.L.new(k.geo.outputs["Position"], s.inputs[0])
fine = k.noise(k.obj, 300.0, 3.0, 0.6)
mid = k.noise(k.obj, 30.0, 4.0, 0.55)
band = k.noise(k.obj, 1.0, 3.0, 0.5, stretch=(4.0, 4.0, 60.0))         # bedding layers across the wheel
col = k.mix(k.mr(band, 0.35, 0.65), (0.42, 0.37, 0.27), (0.55, 0.49, 0.37))
col = k.mix(k.mr(mid, 0.5, 0.75, 0.0, 0.5), col, (0.30, 0.25, 0.18))
col = k.mix(k.mr(fine, 0.6, 0.8, 0.0, 0.5), col, (0.62, 0.58, 0.50))
wet = k.mr(s.outputs["Z"], AZ - R + 0.14, AZ - R + 0.03)
col = k.mix(wet, col, k.mul_col(col, (0.45, 0.43, 0.42)))
cv = k.cavity(0.02)
col = k.mix(k.m("MULTIPLY", cv, 0.5), col, (0.10, 0.09, 0.07))
rough = k.mixf(wet, k.mr(fine, 0.3, 0.7, 0.85, 0.97), 0.55)
M_STONE = k.out(col, rough, 0.0, k.bump(k.m("ADD", fine, k.m("MULTIPLY", mid, 0.4)), 0.0008, 0.6))

M_TY = F.smithy_wood("grind_timber_y", axis="Y", base=(0.15, 0.10, 0.06), grey=0.45, soot=0.15, wear=0.3)
M_TZ = F.smithy_wood("grind_timber_z", axis="Z", base=(0.15, 0.10, 0.06), grey=0.45, soot=0.15, wear=0.3)
M_TX = F.smithy_wood("grind_timber_x", axis="X", base=(0.15, 0.10, 0.06), grey=0.45, soot=0.15, wear=0.5)
M_IRON = F.forged_iron("grind_iron", scale_amt=0.5, rust=0.6, polish=0.3)
M_WATER = F.dark_water("grind_water", murk=(0.03, 0.028, 0.022), scum=0.0)
K.assign(wheel, M_STONE)
for o in timber_y:
    K.assign(o, M_TY)
for o in timber_z:
    K.assign(o, M_TZ)
for o in timber_x:
    K.assign(o, M_TX)
for o in irons:
    K.assign(o, M_IRON)
K.assign(wat, M_WATER)
