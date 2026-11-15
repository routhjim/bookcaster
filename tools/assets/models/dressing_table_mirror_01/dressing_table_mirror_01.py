"""Model script: Victorian mahogany dressing table with a swing mirror on a stand, front -Y."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

W, D, TZ = 1.05, 0.50, 0.76
wv = K.wood("mahogany", "Z", wear=0.5, gloss=0.55, grime=0.6)
wx = K.wood("mahogany", "X", wear=0.6, gloss=0.6, grime=0.5, offset=0.3)
wy = K.wood("mahogany", "Y", wear=0.4, gloss=0.5, grime=0.6, offset=0.6)
brass = K.metal("brass", tarnish=0.75, polish=0.6)
mirror = K.metal("nickel", tarnish=0.25, polish=0.0, name="mirror_silver")
parts = []
FY = -D / 2 + 0.02
# top
parts.append(K.box("top", (W, D, 0.025), (0, 0, TZ - 0.0125), bev=0.004, seg=3, mat=wx))
# legs: square, tapered, set in from the corners
for sx in (-1, 1):
    for sy in (-1, 1):
        lg = K.box(f"leg{sx}{sy}", (0.045, 0.045, TZ - 0.025), (sx * (W / 2 - 0.045), sy * (D / 2 - 0.045), (TZ - 0.025) / 2),
                   bev=0.003, mat=wv)
        h = TZ - 0.025
        K.warp(lg, lambda c, h=h: (c.x * K.lerp(0.62, 1.0, K.smooth((c.z + h / 2) / h * 1.3)),
                                   c.y * K.lerp(0.62, 1.0, K.smooth((c.z + h / 2) / h * 1.3)), c.z))
        parts.append(lg)
# frieze: back and sides, drawers at the front
FZ0, FZ1 = TZ - 0.025 - 0.15, TZ - 0.025
for s in (-1, 1):
    parts.append(K.box(f"apron{s}", (0.02, D - 0.13, FZ1 - FZ0), (s * (W / 2 - 0.05), 0, (FZ0 + FZ1) / 2), bev=0.002, mat=wy))
parts.append(K.box("apron_b", (W - 0.13, 0.02, FZ1 - FZ0), (0, D / 2 - 0.05, (FZ0 + FZ1) / 2), bev=0.002, mat=wx))
parts.append(K.box("rail_low", (W - 0.11, 0.02, 0.02), (0, -D / 2 + 0.05, FZ0 + 0.01), bev=0.002, mat=wx))
xs = [(-0.335, 0.24), (0.0, 0.40), (0.335, 0.24)]
for i, (cx, w) in enumerate(xs):
    parts.append(K.box(f"drawer{i}", (w, 0.022, 0.11), (cx, -D / 2 + 0.045, FZ0 + 0.02 + 0.065), bev=0.003, seg=2, mat=wx))
    kn = K.lathe(f"knob{i}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.010, "c"), (0.006, 0.006), (0.012, 0.010), (0.018, 0.011),
                                     (0.022, 0.0001)], 0.0015), seg=16, mat=brass)
    kn.location, kn.rotation_euler = (cx, -D / 2 + 0.034, FZ0 + 0.085), (math.pi / 2, 0, 0)
    parts.append(kn)
for x in (-0.205, 0.205):
    parts.append(K.box(f"div{x}", (0.03, 0.02, 0.13), (x, -D / 2 + 0.05, FZ0 + 0.085), bev=0.002, mat=wv))
# mirror stand on the top
SB = (0.62, 0.20, 0.10)
SY = 0.10
sz0 = TZ
parts.append(K.box("stand_base", SB, (0, SY, sz0 + SB[2] / 2), bev=0.004, seg=2, mat=wx))
for s in (-1, 1):
    parts.append(K.box(f"sdrawer{s}", (0.26, 0.012, 0.07), (s * 0.15, SY - SB[1] / 2 - 0.004, sz0 + 0.05), bev=0.002, mat=wx))
    kn = K.lathe(f"sknob{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.007, "c"), (0.010, 0.005), (0.014, 0.008),
                                      (0.016, 0.0001)], 0.0015), seg=12, mat=brass)
    kn.location, kn.rotation_euler = (s * 0.15, SY - SB[1] / 2 - 0.009, sz0 + 0.05), (math.pi / 2, 0, 0)
    parts.append(kn)
UZ0 = sz0 + SB[2]
UH = 0.66
for s in (-1, 1):
    up = K.lathe(f"upright{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.020, "c"), (0.03, 0.018), *K.bead(0.04, 0.022, 0.012),
                                        (0.06, 0.014), (UH - 0.08, 0.012), *K.bead(UH - 0.07, 0.017, 0.010), (UH - 0.05, 0.012),
                                        (UH - 0.02, 0.018), (UH, 0.0001)], 0.003), seg=20, mat=wv)
    up.location = (s * 0.27, SY, UZ0)
    parts.append(up)
# swing mirror: arched frame (sweep round a rounded-top rectangle) with silvered glass inside
MW, MH = 0.46, 0.56
MZ = UZ0 + 0.06 + MH / 2
frame_path = K.rrect_path(MW, MH, 0.03, 8, "xz", (0, SY, MZ), top_r=0.12)
fsec = [K.superellipse2d(0.022, 0.014, 2.6, 0.0, 12) for _ in frame_path]
fr = K.sweep("mframe", frame_path, fsec, closed=True, up=(0, 1, 0), mat=wv, angle=50)
parts.append(fr)
# glass: a thin slab in the same outline
import bmesh  # noqa: E402
gp = K.rrect_path(MW - 0.02, MH - 0.02, 0.022, 8, "xz", (0, SY, MZ), top_r=0.11)
bm = bmesh.new()
fv = [bm.verts.new((x, SY - 0.002, z)) for x, _, z in gp]
bv = [bm.verts.new((x, SY + 0.004, z)) for x, _, z in gp]
bm.faces.new(fv)
bm.faces.new(list(reversed(bv)))
n = len(gp)
for i in range(n):
    j = (i + 1) % n
    bm.faces.new((fv[j], fv[i], bv[i], bv[j]))
bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
gl = K.from_bm("mirror", bm)
K.assign(gl, mirror)
backb = K.box("mback", (MW - 0.03, 0.006, MH - 0.08), (0, SY + 0.009, MZ - 0.02), bev=0.0, mat=wv)
parts += [gl, backb]
# pivots
for s in (-1, 1):
    pv = K.lathe(f"pivot{s}", [(0.0001, 0.0), (0.010, 0.0), (0.010, 0.03), (0.0001, 0.03)], seg=12, mat=brass)
    pv.location, pv.rotation_euler = (s * 0.27, SY, MZ), (0, -s * math.pi / 2, 0)
    parts.append(pv)
