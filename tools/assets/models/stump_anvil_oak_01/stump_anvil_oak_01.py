"""Model script: an anvil stump, a sawn section of oak trunk ~0.58 m across at the top, flaring to ~0.7 m at the
foot, 0.5 m tall (a 0.3 m anvil on it puts the face at ~0.8 m, knuckle height).

Deeply furrowed bark (vertical ridges displaced into the mesh), a thin pale sapwood ring and a sawn end-grain top with
growth rings, radial checks, black scorch rings where hot work was set down and a few flakes of iron scale; one iron
band below the top.
"""

import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Vector, noise

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

rng = random.Random(3)
H, R_TOP, R_FOOT = 0.50, 0.285, 0.345
NA, NZ = 128, 44
BAND_Z, BAND_H = 0.415, 0.032


def r_core(z):
    """Trunk radius at height z (flared foot, slight belly), before the bark relief."""
    t = z / H
    return R_TOP + (R_FOOT - R_TOP) * (1 - K.smooth(t / 0.55)) ** 1.6 + 0.006 * math.sin(math.pi * t)


def bark(a, z):
    """Outward bark relief at angle a, height z: long vertical ridges that wander, split and break."""
    u = a * 34 / (2 * math.pi)
    p = Vector((math.cos(a) * 3.0, math.sin(a) * 3.0, z * 2.2))
    wander = 0.55 * noise.noise(p * 1.3 + Vector((0, 0, 7)))
    ridge = 1 - abs(math.sin(math.pi * (u + wander)))        # 1 on a ridge crest, 0 in a furrow
    ridge = ridge ** 0.45
    breaks = 0.5 + 0.5 * noise.noise(Vector((math.cos(a) * 9, math.sin(a) * 9, z * 14)))
    plates = 0.004 * noise.noise(p * 7.0)
    return 0.03 * ridge * (0.5 + 0.5 * breaks) + plates


bm = bmesh.new()
rows = []
zs = [H * (k / NZ) for k in range(NZ + 1)]
top_side = None
for z in zs:
    row = []
    for i in range(NA):
        a = 2 * math.pi * i / NA
        r = r_core(z) + bark(a, z)
        if z < 0.01:
            r -= 0.01 * (1 - z / 0.01)                        # foot edge rolls in
        row.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
    rows.append(row)
for a_, b_ in zip(rows, rows[1:]):
    for i in range(NA):
        bm.faces.new((a_[i], a_[(i + 1) % NA], b_[(i + 1) % NA], b_[i]))
bm.faces.new(list(reversed(rows[0])))
# the top: bark rim drops slightly to a sawn face, a ring of sapwood, then the end grain (radial grid)
RS = R_TOP - 0.016
rim = [bm.verts.new((RS * math.cos(2 * math.pi * i / NA), RS * math.sin(2 * math.pi * i / NA), H - 0.002))
       for i in range(NA)]
for i in range(NA):
    bm.faces.new((rows[-1][i], rows[-1][(i + 1) % NA], rim[(i + 1) % NA], rim[i]))
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
side = K.from_bm("bark", bm)
K.shade(side, 50)

bm = bmesh.new()
NR = 14
rings = []
for j in range(NR + 1):
    rr = RS * (j / NR)
    if j == 0:
        rings.append([bm.verts.new((0, 0, H - 0.0015))])
        continue
    ring = []
    for i in range(NA):
        a = 2 * math.pi * i / NA
        x, y = rr * math.cos(a), rr * math.sin(a)
        # sawn face: shallow saw undulation and slight dishing from years of blows
        z = H - 0.002 - 0.004 * (1 - (rr / RS) ** 2) + 0.0012 * math.sin(x * 40 + y * 9)
        ring.append(bm.verts.new((x, y, z)))
    rings.append(ring)
for i in range(NA):
    bm.faces.new((rings[0][0], rings[1][i], rings[1][(i + 1) % NA]))
for a_, b_ in zip(rings[1:], rings[2:]):
    for i in range(NA):
        bm.faces.new((a_[i], b_[i], b_[(i + 1) % NA], a_[(i + 1) % NA]))
# close underneath (a thin disc; the bark mesh's sawn rim meets its edge)
bot = [bm.verts.new((RS * math.cos(2 * math.pi * i / NA), RS * math.sin(2 * math.pi * i / NA), H - 0.03))
       for i in range(NA)]
for i in range(NA):
    bm.faces.new((rings[-1][(i + 1) % NA], rings[-1][i], bot[i], bot[(i + 1) % NA]))
bm.faces.new(bot)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
top = K.from_bm("endgrain", bm)
K.shade(top, 30)

# iron band, seated in the bark (radius follows the ridges' crests)
rb = r_core(BAND_Z) + 0.026
band = K.lathe("band", [(rb - 0.006, BAND_Z - BAND_H / 2), (rb + 0.004, BAND_Z - BAND_H / 2),
                        (rb + 0.0055, BAND_Z), (rb + 0.004, BAND_Z + BAND_H / 2),
                        (rb - 0.006, BAND_Z + BAND_H / 2), (rb - 0.006, BAND_Z - BAND_H / 2)], seg=128,
               cap0=False, cap1=False)
nails = K.nails("bn", [(rb + 0.005, 0, BAND_Z), (-rb - 0.005, 0, BAND_Z), (0, rb + 0.005, BAND_Z),
                       (0, -rb - 0.005, BAND_Z)], r=0.007,
                normal=[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)])

# flakes of forge scale on the top
flakes = []
for k in range(14):
    a, rr = rng.uniform(0, 6.283), rng.uniform(0.05, 0.22)
    bm = bmesh.new()
    n = 9
    s = rng.uniform(0.008, 0.018)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        q = s * rng.uniform(0.55, 1.0)
        pts.append((q * math.cos(t), q * math.sin(t)))
    lo = [bm.verts.new((x, y, 0.0)) for x, y in pts]
    hi = [bm.verts.new((x * 0.9, y * 0.9, 0.0016)) for x, y in pts]
    bm.faces.new(list(reversed(lo)))
    bm.faces.new(hi)
    for i in range(n):
        bm.faces.new((lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]))
    f = K.from_bm(f"flake{k}", bm)
    f.location = (rr * math.cos(a), rr * math.sin(a), H - 0.004 - 0.004 * (1 - (rr / RS) ** 2) + 0.0005)
    f.rotation_euler = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0, 6.28))
    flakes.append(f)


# ------------------------------------------------------------------ materials
def mat_bark():
    k = K.NK("stump_bark")
    big = k.noise(k.obj, 4.0, 3.0, 0.5)
    mid = k.noise(k.obj, 30.0, 5.0, 0.6)
    fine = k.noise(k.obj, 180.0, 3.0, 0.6)
    cv = k.cavity(0.012, 0.95, 0.5)
    col = k.mix(k.mr(mid, 0.3, 0.7), (0.05, 0.04, 0.032), (0.12, 0.10, 0.08))
    col = k.mix(k.mr(big, 0.45, 0.7, 0.0, 0.5), col, (0.10, 0.11, 0.08))      # a little grey-green weathering
    col = k.mix(k.mr(fine, 0.6, 0.8, 0.0, 0.4), col, (0.24, 0.22, 0.19))
    col = k.mix(cv, col, (0.018, 0.014, 0.011))
    # soot and scale dust high on the trunk, floor dirt at the foot
    s = k.N.new("ShaderNodeSeparateXYZ")
    k.L.new(k.geo.outputs["Position"], s.inputs[0])
    col = k.mix(k.m("MULTIPLY", k.mr(s.outputs["Z"], 0.3, H), 0.5), col, (0.035, 0.031, 0.028))
    col = k.mix(k.m("MULTIPLY", k.mr(s.outputs["Z"], 0.08, 0.0), 0.7), col, (0.05, 0.042, 0.034))
    rough = k.mr(fine, 0.3, 0.7, 0.82, 0.96)
    return k.out(col, rough, 0.0, k.bump(k.m("ADD", fine, k.m("MULTIPLY", mid, 0.6)), 0.0015, 0.6))


def mat_endgrain():
    k = K.NK("stump_endgrain")
    s = k.N.new("ShaderNodeSeparateXYZ")
    k.L.new(k.obj, s.inputs[0])
    x, y = s.outputs["X"], s.outputs["Y"]
    dist = k.m("SQRT", k.m("ADD", k.m("MULTIPLY", x, x), k.m("MULTIPLY", y, y)))
    wob = k.noise(k.obj, 6.0, 2.0, 0.5)
    d2 = k.m("ADD", dist, k.m("MULTIPLY", wob, 0.012))
    rings = k.m("SINE", k.m("MULTIPLY", d2, 2 * math.pi * 95))
    rings = k.mr(rings, -1.0, 1.0)
    late = k.mr(rings, 0.75, 0.95)                         # dark latewood bands
    ang = k.m("ARCTAN2", y, x)
    rays = k.mr(k.m("SINE", k.m("MULTIPLY", ang, 140.0)), 0.92, 1.0, 0.0, 0.35)
    fine = k.noise(k.obj, 220.0, 3.0, 0.6)
    col = k.mix(late, (0.19, 0.125, 0.07), (0.10, 0.06, 0.03))
    col = k.mix(rays, col, (0.23, 0.16, 0.09))
    col = k.mix(k.mr(dist, RS - 0.025, RS - 0.012), col, (0.26, 0.20, 0.12))   # pale sapwood ring
    # radial drying checks: dark cracks from the rim inward
    crack = k.mr(k.m("SINE", k.m("ADD", k.m("MULTIPLY", ang, 7.0), k.m("MULTIPLY", wob, 2.0))), 0.985, 1.0)
    crack = k.m("MULTIPLY", crack, k.mr(dist, 0.08, 0.2))
    col = k.mix(crack, col, (0.02, 0.012, 0.008))
    # grey weathering and black scorch rings where hot iron was set down; hammer-dulled centre
    burn_n = k.noise(k.obj, 9.0, 4.0, 0.6)
    burn = k.m("MULTIPLY", k.mr(burn_n, 0.55, 0.7), k.mr(dist, 0.27, 0.12, 0.3, 1.0))
    col = k.mix(k.mr(fine, 0.3, 0.7, 0.15, 0.4), col, (0.20, 0.18, 0.16))
    col = k.mix(burn, col, (0.018, 0.013, 0.010))
    rough = k.mixf(burn, k.mr(fine, 0.3, 0.7, 0.68, 0.85), 0.9)
    h = k.m("ADD", k.m("MULTIPLY", late, -0.4), k.m("MULTIPLY", crack, -1.5))
    h = k.m("ADD", h, k.m("MULTIPLY", fine, 0.3))
    return k.out(col, rough, 0.0, k.bump(h, 0.0008, 0.5))


M_BARK = mat_bark()
M_END = mat_endgrain()
M_IRON = K.metal("iron", tarnish=0.7, soot=0.3, rust=0.55, polish=0.2, name="stump_iron")
M_SCALE = K.metal("iron", tarnish=0.9, rust=0.2, polish=0.05, name="stump_scale")
K.assign(side, M_BARK)
K.assign(top, M_END)
K.assign(band, M_IRON)
for o in nails:
    K.assign(o, M_IRON)
for o in flakes:
    K.assign(o, M_SCALE)
