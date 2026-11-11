"""Model script: Victorian cast-iron cabinet safe, front -Y. Rounded body (period_kit.rbox), raised door border
from four bevelled bars, rivets (period_kit.nails), turned dial, handle, keyhole cover, barrel hinges, feet."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

W, D, H = 0.72, 0.58, 1.25
Z0 = 0.075                               # body bottom (feet + plinth below)
iron = K.metal("cast_iron", tarnish=0.5, polish=0.55, rust=0.35)
iron_dark = K.metal("iron", tarnish=0.6, polish=0.4, rust=0.25)
brass = K.metal("brass", tarnish=0.8, polish=0.6)
parts = []
FY = -D / 2                              # front face of the body

body = K.rbox("body", (W - 0.02, D, H - Z0 - 0.02), r=0.02, spacing=0.05, loc=(0, 0, Z0 + 0.02 + (H - Z0 - 0.02) / 2),
              mat=iron)
parts.append(body)
parts.append(K.box("plinth", (W, D + 0.02, 0.035), (0, -0.0, Z0 + 0.0125), bev=0.006, seg=3, mat=iron))
parts.append(K.box("cap", (W, D + 0.02, 0.03), (0, 0.0, H - 0.015), bev=0.008, seg=3, mat=iron))
for sx in (-1, 1):
    for sy in (-1, 1):
        f = K.box(f"foot{sx}{sy}", (0.08, 0.08, Z0 - 0.005), (sx * (W / 2 - 0.05), sy * (D / 2 - 0.05), (Z0 - 0.005) / 2),
                  bev=0.008, seg=3, mat=iron_dark)
        K.warp(f, lambda c: (c.x * (1 - 0.25 * (0.5 - c.z / (Z0 - 0.005))), c.y * (1 - 0.25 * (0.5 - c.z / (Z0 - 0.005))), c.z))
        parts.append(f)

# ---- door: slab proud of the body with a raised bevelled border, inner panel
DX0, DX1, DZ0, DZ1 = -W / 2 + 0.055, W / 2 - 0.055, Z0 + 0.07, H - 0.075
dw, dh = DX1 - DX0, DZ1 - DZ0
dcx, dcz = (DX0 + DX1) / 2, (DZ0 + DZ1) / 2
parts.append(K.box("door", (dw, 0.03, dh), (dcx, FY - 0.012, dcz), bev=0.004, seg=2, mat=iron))
bw = 0.05
for (x, z, sx, sz) in ((dcx, DZ1 - bw / 2, dw, bw), (dcx, DZ0 + bw / 2, dw, bw), (DX0 + bw / 2, dcz, bw, dh - 2 * bw),
                       (DX1 - bw / 2, dcz, bw, dh - 2 * bw)):
    parts.append(K.box(f"border{x:.2f}{z:.2f}", (sx, 0.012, sz), (x, FY - 0.032, z), bev=0.005, seg=2, mat=iron))
# rivets round the border
rv = []
for k in range(9):
    x = K.lerp(DX0 + 0.025, DX1 - 0.025, k / 8)
    rv += [(x, FY - 0.038, DZ1 - 0.025), (x, FY - 0.038, DZ0 + 0.025)]
for k in range(1, 17):
    z = K.lerp(DZ0 + 0.025, DZ1 - 0.025, k / 17)
    rv += [(DX0 + 0.025, FY - 0.038, z), (DX1 - 0.025, FY - 0.038, z)]
parts += K.nails("rivet", rv, 0.0065, iron_dark, (0, -1, 0))

# ---- dial: brass ring, iron dial with a knurled edge and a grip
DZ = DZ1 - 0.20
ring = K.lathe("dial_ring", K.keys([(0.0, 0.0001, "c"), (0.0, 0.075, "c"), (0.006, 0.075), (0.012, 0.068), (0.014, 0.056),
                                     (0.014, 0.0001, "c")], 0.0015), seg=48, mat=brass)
ring.location, ring.rotation_euler = (0, FY - 0.026, DZ), (math.pi / 2, 0, 0)
dial = K.lathe("dial", K.keys([(0.0, 0.0001, "c"), (0.0, 0.052, "c"), (0.022, 0.052, "l"), (0.026, 0.046),
                               (0.03, 0.036), (0.03, 0.0001, "c")], 0.0015), seg=60, mat=iron_dark)
dial.location, dial.rotation_euler = (0, FY - 0.026, DZ), (math.pi / 2, 0, 0)
grip = K.box("dial_grip", (0.016, 0.03, 0.06), (0, FY - 0.068, DZ), bev=0.005, seg=2, mat=iron_dark)
parts += [ring, dial, grip]
for k in range(40):                       # knurl ribs on the dial edge
    a = math.tau * k / 40
    parts.append(K.box(f"knurl{k}", (0.004, 0.018, 0.003), (0.053 * math.cos(a), FY - 0.040, DZ + 0.053 * math.sin(a)),
                       (0, -a, 0), bev=0.0008, seg=1, mat=iron_dark))

# ---- handle: boss + T bar, below and left of the dial; keyhole cover to the right
HZ = DZ - 0.24
boss = K.lathe("boss", K.keys([(0.0, 0.0001, "c"), (0.0, 0.035, "c"), (0.012, 0.03), (0.02, 0.018), (0.04, 0.014),
                               (0.045, 0.0001, "c")], 0.002), seg=32, mat=iron_dark)
boss.location, boss.rotation_euler = (-0.12, FY - 0.026, HZ), (math.pi / 2, 0, 0)
tbar = K.lathe("tbar", K.keys([(0.0, 0.0001, "c"), (0.0, 0.012, "c"), (0.16, 0.012, "l"), (0.16, 0.0001, "c")], 0.02),
               seg=16, mat=iron_dark)
tbar.location, tbar.rotation_euler = (-0.12, FY - 0.068, HZ - 0.08), (0, 0, 0)
parts += [boss, tbar]
esc = K.lathe("keyplate", K.keys([(0.0, 0.0001, "c"), (0.0, 0.032, "c"), (0.004, 0.032), (0.007, 0.026), (0.007, 0.0001, "c")],
                                 0.001), seg=32, mat=brass)
esc.location, esc.rotation_euler = (0.12, FY - 0.026, HZ), (math.pi / 2, 0, 0)
cover = K.box("keycover", (0.022, 0.004, 0.05), (0.12, FY - 0.035, HZ - 0.018), (0, 0.15, 0), bev=0.002, seg=2, mat=brass)
parts += [esc, cover]
plate = K.box("plate", (0.16, 0.004, 0.05), (0, FY - 0.0285, DZ0 + 0.13), bev=0.002, seg=2, mat=brass)
parts.append(plate)

# ---- barrel hinges on the right edge of the door
for z in (DZ0 + 0.15, DZ1 - 0.15):
    hg = K.lathe(f"hinge{z:.2f}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.016, "c"), (0.008, 0.018), (0.13, 0.018, "l"),
                                          (0.138, 0.016), (0.145, 0.0001, "c")], 0.004), seg=20, mat=iron_dark)
    hg.location = (DX1 + 0.012, FY - 0.018, z - 0.07)
    parts.append(hg)
