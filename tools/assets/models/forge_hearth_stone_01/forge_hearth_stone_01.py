"""Model script: a village blacksmith's forge hearth of rough fieldstone, ~1.8 m wide (X), 1.15 m deep (Y), 3.4 m tall.

Front faces -Y. A waist-high stone fire bed (top 0.74 m) with a flat-lintelled ash pit under the front, capstones
round a sunken fire pit holding a mound of charcoal; a stone back wall behind the bed up to 2.4 m; a lean-to
pyramid hood of riveted black sheet iron over the fire (its back against the wall) narrowing into a square stone
chimney to 3.4 m; an iron tuyere (air pipe) entering the fire bed through the left (-X) side at 0.62 m, its flared
mouth ~0.1 m proud of the stone for a bellows nozzle. Soot up the hood, the wall above the fire and the chimney.

SPEC["variant"] == "lit": the charcoal glows (emissive): white-yellow over the tuyere mouth, orange, then dull red,
fading to dark clinker at the pit's edges. Otherwise the fire is out: black charcoal under grey ash.
Every stone is its own rounded, noise-displaced block laid in courses over a recessed mortar core, so the joints are
real relief.
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

LIT = (SPEC.get("variant") == "lit")  # noqa: F821
rng = random.Random(7)

# ------------------------------------------------------------------ dimensions (metres)
BED_X = 0.76            # fire bed half width
BED_Y0, BED_Y1 = -0.55, 0.35
BED_TOP = 0.64          # top of the walling under the capstones
CAP_T = 0.10            # capstone thickness (top 0.74)
WALL_X = 0.84
WALL_Y0, WALL_Y1 = 0.35, 0.60
WALL_TOP = 2.40
PIT = (-0.22, 0.22, -0.25, 0.19)          # fire pit opening in the capstones x0, x1, y0, y1
POT_C, POT_R = (0.0, -0.03), 0.245         # round iron fire-pot rim on the capstones: centre, inner radius
FIRE_C = Vector((-0.06, -0.04, 0.0))      # hottest point (just off the tuyere mouth)
ASH = (-0.30, 0.30, 0.0, 0.42)            # ash pit opening in the front: x0, x1, z0, z1
HOOD_Z0, HOOD_Z1 = 1.62, 2.30
CH = (-0.30, 0.30, -0.04, 0.60)           # chimney x0, x1, y0, y1
CH_TOP = 3.40
TUY_Z, TUY_Y = 0.62, -0.03
JOINT = 0.016

stones, mortar, irons, coals = [], [], [], []


# ------------------------------------------------------------------ rough stones
def stone(cx, cy, cz, sx, sy, sz, amp=0.010):
    """A rounded fieldstone block centred at (cx, cy, cz), noise-displaced, pillowed, colour-tinted."""
    r = min(0.028, sx * 0.3, sy * 0.3, sz * 0.3)
    ob = K.rbox(f"st{len(stones)}", (sx, sy, sz), r=r, spacing=0.085,
                bulge=(rng.uniform(0.002, 0.008), rng.uniform(0.002, 0.008), rng.uniform(0.002, 0.006)))
    h = (sx / 2, sy / 2, sz / 2)
    inner = [x - r for x in h]
    off = Vector((rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 50)))
    tilt = (rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04))
    for v in ob.data.vertices:
        p = v.co.copy()
        q = Vector([max(-inner[i], min(inner[i], p[i])) for i in range(3)])
        d = p - q
        n = d.normalized() if d.length > 1e-9 else Vector((0, 0, 1))
        w = Vector((cx, cy, cz)) + p
        disp = amp * (noise.noise(w * 9.0 + off) * 0.75 + noise.noise(w * 23.0 + off) * 0.35)
        disp += amp * 0.6 * abs(noise.noise(w * 4.0 + off))            # knobbly high spots
        p += n * disp
        p.z += tilt[0] * p.x + tilt[1] * p.y                              # out-of-square faces
        v.co = p
    ob.data.update()
    ob.location = (cx, cy, cz)
    tint = rng.choice([(0.115, 0.110, 0.100), (0.140, 0.130, 0.115), (0.095, 0.095, 0.090), (0.125, 0.110, 0.090),
                       (0.165, 0.155, 0.140), (0.105, 0.095, 0.080)])
    j = rng.uniform(0.85, 1.12)
    K.set_color(ob, tuple(c * j for c in tint))
    stones.append(ob)
    return ob


def lay(axis, sign, plane, u0, u1, courses, depth, skip=(), lmin=0.22, lmax=0.48, out=0.0):
    """Lay coursed stones on an axis-aligned face. axis 'y': face normal along Y (u = X); 'x': normal along X
    (u = Y). sign: outward direction. plane: the face's outer coordinate. courses: z boundaries. skip: list of
    (u0, u1, z0, z1) rectangles left open. Stones are `depth` deep into the mass."""
    for ci, (z0, z1) in enumerate(zip(courses, courses[1:])):
        u = u0 + (rng.uniform(0, lmin) if ci % 2 else 0.0)
        segs = []
        if u > u0:
            segs.append((u0, u))
        while u < u1 - 1e-6:
            L = rng.uniform(lmin, lmax)
            if u1 - (u + L) < lmin * 0.6:
                L = u1 - u
            segs.append((u, min(u1, u + L)))
            u += L
        for a, b in segs:
            pieces = [(a, b)]
            for (s0, s1, sz0, sz1) in skip:
                if z1 > sz0 + 1e-6 and z0 < sz1 - 1e-6:
                    nxt = []
                    for (pa, pb) in pieces:
                        if pb <= s0 or pa >= s1:
                            nxt.append((pa, pb))
                        else:
                            if pa < s0:
                                nxt.append((pa, s0))
                            if pb > s1:
                                nxt.append((s1, pb))
                    pieces = nxt
            for (pa, pb) in pieces:
                if pb - pa < 0.06:
                    continue
                L = pb - pa - JOINT
                H = z1 - z0 - JOINT
                proud = out + rng.uniform(-0.004, 0.010)
                dd = depth * rng.uniform(0.85, 1.0)
                c_n = plane - sign * dd / 2 + sign * proud
                cu = (pa + pb) / 2
                cz = (z0 + z1) / 2
                if axis == "y":
                    stone(cu, c_n, cz, L, dd, H)
                else:
                    stone(c_n, cu, cz, dd, L, H)


def core(name, x0, x1, y0, y1, z0, z1):
    ob = K.box(name, (x1 - x0, y1 - y0, z1 - z0), loc=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), bev=0.0)
    mortar.append(ob)
    return ob


# ------------------------------------------------------------------ fire bed
IN = 0.014   # mortar core sits this far behind the stone faces
bed = core("bed_core", -BED_X + IN, BED_X - IN, BED_Y0 + IN, BED_Y1, 0.0, BED_TOP + 0.02)
cut = K.box("ashpit_cut", (ASH[1] - ASH[0], 0.80, ASH[3] + 0.1), loc=((ASH[0] + ASH[1]) / 2, BED_Y0 + 0.10,
                                                                       (ASH[3] - 0.1) / 2), bev=0.0)
m = bed.modifiers.new("ash", "BOOLEAN")
m.operation, m.object, m.solver = "DIFFERENCE", cut, "EXACT"
cut.hide_set(True)
pitcut = K.box("pit_cut", (PIT[1] - PIT[0] + 0.02, PIT[3] - PIT[2] + 0.02, 0.2),
               loc=((PIT[0] + PIT[1]) / 2, (PIT[2] + PIT[3]) / 2, BED_TOP + 0.02), bev=0.0)
m = bed.modifiers.new("pit", "BOOLEAN")
m.operation, m.object, m.solver = "DIFFERENCE", pitcut, "EXACT"
pitcut.hide_set(True)

C3 = [0.0, 0.215, ASH[3], BED_TOP]
lay("y", -1, BED_Y0, -BED_X, BED_X, C3, 0.22, skip=[(ASH[0], ASH[1], ASH[2], ASH[3]),
                                                    (ASH[0] - 0.12, ASH[1] + 0.12, ASH[3], BED_TOP)])
# lintel over the ash pit
stone(0.0, BED_Y0 - 0.11 + 0.006, (ASH[3] + BED_TOP) / 2, ASH[1] - ASH[0] + 0.24 - JOINT, 0.22,
      BED_TOP - ASH[3] - JOINT, amp=0.008)
lay("x", -1, -BED_X, BED_Y0 + 0.22, BED_Y1, C3, 0.22, lmin=0.2, lmax=0.4)
lay("x", 1, BED_X, BED_Y0 + 0.22, BED_Y1, C3, 0.22, lmin=0.2, lmax=0.4)
# ash pit: side walls of thin stones, ash and clinker on its floor
lay("x", 1, ASH[0], BED_Y0 + 0.2, BED_Y0 + 0.5, [0.0, ASH[3]], 0.05, lmin=0.28, lmax=0.3, out=-0.01)
lay("x", -1, ASH[1], BED_Y0 + 0.2, BED_Y0 + 0.5, [0.0, ASH[3]], 0.05, lmin=0.28, lmax=0.3, out=-0.01)


def mound(name, x0, x1, y0, y1, base, peak, nx, ny, centre=(0.0, 0.0), sharp=1.6, rough=0.012, edge_drop=0.0):
    """A closed heap: a displaced top grid over a flat bottom."""
    bm = bmesh.new()
    top, bot = [], []
    for i in range(nx + 1):
        rt, rb = [], []
        for j in range(ny + 1):
            x = K.lerp(x0, x1, i / nx)
            y = K.lerp(y0, y1, j / ny)
            u = (x - centre[0]) / max(centre[0] - x0, x1 - centre[0])
            w = (y - centre[1]) / max(centre[1] - y0, y1 - centre[1])
            d = min(1.0, math.sqrt(u * u + w * w))
            hgt = (1 - d ** sharp) * peak
            edge = i in (0, nx) or j in (0, ny)
            z = base + (0.0 if edge else max(0.003, hgt - edge_drop * d + rough * noise.noise(Vector((x * 30, y * 30, 3.0)))))
            rt.append(bm.verts.new((x, y, z)))
            rb.append(bm.verts.new((x, y, base - 0.02)))
        top.append(rt)
        bot.append(rb)
    for i in range(nx):
        for j in range(ny):
            bm.faces.new((top[i][j], top[i + 1][j], top[i + 1][j + 1], top[i][j + 1]))
            bm.faces.new((bot[i][j], bot[i][j + 1], bot[i + 1][j + 1], bot[i + 1][j]))
    ring = [(i, 0) for i in range(nx)] + [(nx, j) for j in range(ny)] + [(i, ny) for i in range(nx, 0, -1)] + \
           [(0, j) for j in range(ny, 0, -1)]
    for k in range(len(ring)):
        (a, b), (c, d) = ring[k], ring[(k + 1) % len(ring)]
        bm.faces.new((top[a][b], top[c][d], bot[c][d], bot[a][b]))
    ob = K.from_bm(name, bm)
    K.shade(ob, 80)
    return ob


def height_at(ob_heap, x, y, x0, x1, y0, y1, base, peak, centre, sharp):
    u = (x - centre[0]) / max(centre[0] - x0, x1 - centre[0])
    w = (y - centre[1]) / max(centre[1] - y0, y1 - centre[1])
    d = min(1.0, math.sqrt(u * u + w * w))
    return base + (1 - d ** sharp) * peak


ash_heap = mound("ash_heap", ASH[0] + 0.01, ASH[1] - 0.01, BED_Y0 + 0.02, BED_Y0 + 0.48, 0.0, 0.07, 18, 16,
                 centre=(0.0, BED_Y0 + 0.30), sharp=1.3, rough=0.008)
ashes = [ash_heap]

# capstones round the fire pit (top at BED_TOP + CAP_T)
zc = BED_TOP + CAP_T / 2
for (x0, x1, y0, y1, n) in [(-BED_X - 0.01, BED_X + 0.01, BED_Y0 - 0.015, PIT[2], 4),
                            (-BED_X - 0.01, BED_X + 0.01, PIT[3], BED_Y1, 3),
                            (-BED_X - 0.01, PIT[0], PIT[2], PIT[3], 2),
                            (PIT[1], BED_X + 0.01, PIT[2], PIT[3], 2)]:
    cuts = sorted([x0] + [K.lerp(x0, x1, (k + rng.uniform(-0.18, 0.18)) / n) for k in range(1, n)] + [x1])
    for a, b in zip(cuts, cuts[1:]):
        stone((a + b) / 2, (y0 + y1) / 2, zc + rng.uniform(-0.004, 0.006), b - a - JOINT, y1 - y0 - JOINT,
              CAP_T - 0.006, amp=0.007)

# ------------------------------------------------------------------ back wall
core("wall_core", -WALL_X + IN, WALL_X - IN, WALL_Y0 + IN, WALL_Y1 - 0.01, 0.0, WALL_TOP - 0.02)
CW = [0.0, 0.2, 0.42, BED_TOP + CAP_T]
z = CW[-1]
while z < WALL_TOP - 0.1:
    z = min(WALL_TOP, z + rng.uniform(0.17, 0.25))
    CW.append(z if WALL_TOP - z > 0.12 else WALL_TOP)
    z = CW[-1]
lay("y", -1, WALL_Y0, -WALL_X, WALL_X, [c for c in CW if c >= BED_TOP + CAP_T - 1e-6], 0.24)
for s in (-1, 1):   # the wall's two ends, visible beside the bed
    lay("x", s, s * WALL_X, WALL_Y0, WALL_Y1, CW, 0.08, lmin=0.25, lmax=0.26)
# top course of the wall: flat copes
for a, b in [(-WALL_X, -0.32), (0.32, WALL_X)]:
    stone((a + b) / 2, (WALL_Y0 + WALL_Y1) / 2, WALL_TOP + 0.03, b - a - JOINT, WALL_Y1 - WALL_Y0 + 0.02, 0.06,
          amp=0.006)

# ------------------------------------------------------------------ chimney stack
core("chimney_core", CH[0] + IN, CH[1] - IN, CH[2] + IN, CH[3] - IN, HOOD_Z1 - 0.06, CH_TOP - 0.02)
CC = [HOOD_Z1 - 0.06]
while CC[-1] < CH_TOP - 0.05:
    CC.append(min(CH_TOP, CC[-1] + rng.uniform(0.16, 0.22)))
lay("y", -1, CH[2], CH[0], CH[1], CC, 0.14, lmin=0.18, lmax=0.36)
lay("y", 1, CH[3], CH[0], CH[1], [c for c in CC if c >= WALL_TOP - 0.2] or CC, 0.14, lmin=0.18, lmax=0.36)
lay("x", -1, CH[0], CH[2] + 0.14, CH[3], CC, 0.14, lmin=0.18, lmax=0.3)
lay("x", 1, CH[1], CH[2] + 0.14, CH[3], CC, 0.14, lmin=0.18, lmax=0.3)
# a capping slab with the flue showing dark
stone((CH[0] + CH[1]) / 2, (CH[2] + CH[3]) / 2, CH_TOP + 0.02, CH[1] - CH[0] + 0.06, CH[3] - CH[2] + 0.06, 0.06,
      amp=0.006)

# ------------------------------------------------------------------ iron hood (lean-to pyramid, back on the wall)
HB = (-0.80, 0.80, -0.62, WALL_Y0 - 0.002)      # bottom rectangle
HT = (CH[0] + 0.03, CH[1] - 0.03, CH[2] + 0.03, WALL_Y0 - 0.002)
bm = bmesh.new()
NS = 10
rows = []
for k in range(NS + 1):
    t = k / NS
    zz = K.lerp(HOOD_Z0, HOOD_Z1, t)
    tt = t ** 0.85
    x0, x1 = K.lerp(HB[0], HT[0], tt), K.lerp(HB[1], HT[1], tt)
    y0 = K.lerp(HB[2], HT[2], tt)
    yb = HB[3]
    # left side (back -> front), front (left -> right), right side (front -> back)
    pts = [(x0, yb), (x0, y0 + (yb - y0) * 0.5), (x0, y0)]
    pts += [(K.lerp(x0, x1, q / 8), y0) for q in range(1, 8)]
    pts += [(x1, y0), (x1, y0 + (yb - y0) * 0.5), (x1, yb)]
    rows.append([bm.verts.new((px, py, zz)) for (px, py) in pts])
for a, b in zip(rows, rows[1:]):
    for i in range(len(a) - 1):
        bm.faces.new((a[i], a[i + 1], b[i + 1], b[i]))
hood = K.from_bm("hood", bm, recalc=False)     # winding gives outward normals
sol = hood.modifiers.new("thick", "SOLIDIFY")
sol.thickness, sol.offset = 0.006, 1.0
K.shade(hood, 30)
irons.append(hood)
# hood skirt band, corner straps, rivets
band_z = (HOOD_Z0 - 0.10, HOOD_Z0 + 0.01)
for (sx, sy, cx, cy) in [(HB[1] - HB[0] + 0.024, 0.012, 0.0, HB[2] - 0.006),
                         (0.012, HB[3] - HB[2], HB[0] - 0.006, (HB[2] + HB[3]) / 2),
                         (0.012, HB[3] - HB[2], HB[1] + 0.006, (HB[2] + HB[3]) / 2)]:
    irons.append(K.box("band", (sx, sy, band_z[1] - band_z[0]), loc=(cx, cy, (band_z[0] + band_z[1]) / 2),
                       bev=0.002))
for s in (-1, 1):   # sloped front corners: angle straps
    xb, xt = (HB[0], HT[0]) if s < 0 else (HB[1], HT[1])
    p0 = Vector((xb, HB[2], HOOD_Z0))
    p1 = Vector((xt, HT[2], HOOD_Z1))
    irons.append(K.tube(f"strap{s}", [p0.lerp(p1, q / 12) for q in range(13)], 0.011, seg=8))
rv = []
for q in range(15):
    rv.append((K.lerp(HB[0] + 0.05, HB[1] - 0.05, q / 14), HB[2] - 0.0125, HOOD_Z0 - 0.045))
irons += K.nails("rv", rv, r=0.008, normal=(0, -1, 0))
rvs = []
for s in (-1, 1):
    for q in range(6):
        rvs.append((s * (HB[1] + 0.0125), K.lerp(HB[2] + 0.06, HB[3] - 0.05, q / 5), HOOD_Z0 - 0.045))
irons += K.nails("rvs", rvs, r=0.008, normal=[(-1, 0, 0)] * 6 + [(1, 0, 0)] * 6)
# two wall brackets holding the hood's front corners up (iron stays from the wall to the hood rim)
for s in (-1, 1):
    a = Vector((s * 0.74, HB[3] - 0.01, HOOD_Z0 + 0.42))
    b = Vector((s * 0.76, HB[2] + 0.06, HOOD_Z0 + 0.02))
    irons.append(K.tube(f"stay{s}", [a.lerp(b, q / 10) for q in range(11)], 0.009, seg=8))

# ------------------------------------------------------------------ tuyere: iron pipe through the left side
TX0, TX1 = -BED_X - 0.12, PIT[0] + 0.04
path = [Vector((K.lerp(TX0, TX1, q / 20), TUY_Y, TUY_Z)) for q in range(21)]


def tr(t):
    if t < 0.12:
        return K.lerp(0.058, 0.036, K.smooth(t / 0.12))   # flared mouth for the bellows nozzle
    return 0.036


tuy = K.tube("tuyere", path, 0.036, seg=20, r_fn=tr)
irons.append(tuy)
bore = K.tube("tuyere_bore", [Vector((TX0 - 0.01, TUY_Y, TUY_Z)), Vector((TX0 + 0.08, TUY_Y, TUY_Z))], 0.03, seg=20,
              r_fn=lambda t: K.lerp(0.046, 0.02, t))
mb = tuy.modifiers.new("bore", "BOOLEAN")
mb.operation, mb.object, mb.solver = "DIFFERENCE", bore, "EXACT"
bore.hide_set(True)
flange = K.box("flange", (0.014, 0.2, 0.2), loc=(-BED_X - 0.004, TUY_Y, TUY_Z), bev=0.003)
irons.append(flange)
irons += K.nails("fl", [(-BED_X - 0.012, TUY_Y + dy, TUY_Z + dz) for dy in (-0.075, 0.075) for dz in (-0.075, 0.075)],
                 r=0.009, normal=(-1, 0, 0))

# ------------------------------------------------------------------ the fire: charcoal mound and lumps
PX0, PX1, PY0, PY1 = PIT[0] - 0.02, PIT[1] + 0.02, PIT[2] - 0.02, PIT[3] + 0.02
BASE, PEAK, SH = BED_TOP - 0.06, 0.17, 1.7
bedm = mound("coal_bed", PX0, PX1, PY0, PY1, BASE, PEAK, 24, 20, centre=(FIRE_C.x, FIRE_C.y), sharp=SH,
             rough=0.01)
coals.append(bedm)


def lump(x, y, z, r):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    off = Vector((rng.uniform(0, 99), rng.uniform(0, 99), rng.uniform(0, 99)))
    sc = Vector((rng.uniform(0.8, 1.35), rng.uniform(0.7, 1.2), rng.uniform(0.5, 0.85)))
    for v in bm.verts:
        p = v.co
        k = 1.0 + 0.28 * noise.noise(p * 1.6 + off) + 0.10 * noise.noise(p * 4.0 + off)
        v.co = Vector((p.x * sc.x, p.y * sc.y, p.z * sc.z)) * k * r
    ob = K.from_bm(f"lump{len(coals)}", bm)
    ob.rotation_euler = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28))
    ob.location = (x, y, z)
    for p in ob.data.polygons:
        p.use_smooth = False
    coals.append(ob)
    return ob


placed = []
tries = 0
while len(placed) < 330 and tries < 6000:
    tries += 1
    a, rr = rng.uniform(0, 6.2832), POT_R * math.sqrt(rng.random())
    x, y = POT_C[0] + rr * math.cos(a), POT_C[1] + rr * math.sin(a)
    r = rng.uniform(0.014, 0.034)
    if rr + r * 0.6 > POT_R:
        continue
    if any((x - a) ** 2 + (y - b) ** 2 < ((r + c) * 0.78) ** 2 for a, b, c in placed):
        continue
    placed.append((x, y, r))
    z = height_at(None, x, y, PX0, PX1, PY0, PY1, BASE, PEAK, (FIRE_C.x, FIRE_C.y), SH)
    if not (PIT[0] < x < PIT[1] and PIT[2] < y < PIT[3]):
        z = max(z, BED_TOP + CAP_T)                      # on the capstone inside the ring
    lump(x, y, z + r * 0.15, r)
# a few spilled on the capstones and the front of the bed
for k in range(26):
    side = rng.random()
    if side < 0.55:
        x, y = rng.uniform(-0.45, 0.45), rng.uniform(BED_Y0 + 0.04, PIT[2] - 0.02)
    elif side < 0.8:
        x, y = rng.uniform(PIT[1] + 0.03, 0.6), rng.uniform(PIT[2], PIT[3])
    else:
        x, y = rng.uniform(-0.62, PIT[0] - 0.03), rng.uniform(PIT[2] + 0.08, PIT[3])
    r = rng.uniform(0.010, 0.022)
    lump(x, y, BED_TOP + CAP_T + r * 0.35, r)
# clinker and charcoal in the ash pit
for k in range(14):
    x, y = rng.uniform(ASH[0] + 0.06, ASH[1] - 0.06), rng.uniform(BED_Y0 + 0.08, BED_Y0 + 0.45)
    r = rng.uniform(0.012, 0.026)
    z = height_at(None, x, y, ASH[0] + 0.01, ASH[1] - 0.01, BED_Y0 + 0.02, BED_Y0 + 0.48, 0.0, 0.07,
                  (0.0, BED_Y0 + 0.30), 1.3)
    ashes.append(lump(x, y, z + r * 0.1, r))
    coals.remove(ashes[-1])


# round iron rim of the fire pot, sitting on the capstones
ring = K.lathe("pot_rim", [(POT_R, 0.0), (POT_R + 0.068, 0.0), (POT_R + 0.068, 0.018), (POT_R + 0.06, 0.026),
                           (POT_R + 0.008, 0.026), (POT_R, 0.02), (POT_R, 0.0)], seg=64, cap0=False, cap1=False)
ring.location = (POT_C[0], POT_C[1], BED_TOP + CAP_T - 0.004)
K.shade(ring, 40)
irons.append(ring)

# ------------------------------------------------------------------ materials
def P(k):
    s = k.N.new("ShaderNodeSeparateXYZ")
    k.L.new(k.geo.outputs["Position"], s.inputs[0])
    return s


def mat_stone(kind):
    """kind 'stone' (pcol tint) or 'mortar'. Soot over the fire, up the wall and chimney; heat-calcined stone and
    ash round the pit; floor grime at the base."""
    k = K.NK(f"forge_{kind}")
    pos = k.geo.outputs["Position"]
    s = P(k)
    big = k.noise(pos, 3.0, 4.0, 0.55)
    mid = k.noise(pos, 14.0, 5.0, 0.6)
    fine = k.noise(pos, 90.0, 4.0, 0.6)
    vor = k.voronoi(pos, 38.0, "F1", "Distance")
    if kind == "stone":
        base = k.pcol()
        col = k.mix(k.mr(mid, 0.3, 0.7, 0.0, 0.45), base, k.mul_col(base, (0.62, 0.60, 0.58)))
        col = k.mix(k.mr(fine, 0.55, 0.75, 0.0, 0.35), col, k.mul_col(base, (1.3, 1.28, 1.22)))
        # mineral flecks and darker inclusions
        col = k.mix(k.mr(vor, 0.08, 0.0, 0.0, 0.5), col, (0.12, 0.115, 0.11))
        col = k.mix(k.mr(big, 0.55, 0.75, 0.0, 0.4), col, (0.30, 0.22, 0.14))   # iron-stain patches
    else:
        col = k.mix(k.mr(mid, 0.3, 0.7), (0.15, 0.14, 0.125), (0.22, 0.21, 0.19))
        col = k.mix(k.mr(fine, 0.5, 0.7, 0.0, 0.5), col, (0.10, 0.095, 0.085))
    e = k.edge(0.008, 0.995, 0.93)
    cv = k.cavity(0.05, 0.92, 0.45)
    # soot: wall face and chimney above the fire, strongest at the centre line, plus a ring round the pit
    zabove = k.mr(s.outputs["Z"], BED_TOP + 0.05, BED_TOP + 0.6)
    xcent = k.mr(k.m("ABSOLUTE", s.outputs["X"]), 0.95, 0.15)
    yback = k.m("MULTIPLY", k.mr(s.outputs["Y"], 0.0, 0.33), k.mr(s.outputs["Y"], 0.56, 0.45))
    soot = k.m("MULTIPLY", k.m("MULTIPLY", zabove, xcent), yback)
    chim = k.mr(s.outputs["Z"], HOOD_Z0 + 0.4, HOOD_Z1 + 0.2, 0.0, 0.8)
    soot = k.m("MAXIMUM", soot, chim)
    dx = k.m("SUBTRACT", s.outputs["X"], FIRE_C.x)
    dy = k.m("SUBTRACT", s.outputs["Y"], FIRE_C.y)
    dist = k.m("SQRT", k.m("ADD", k.m("MULTIPLY", dx, dx), k.m("MULTIPLY", dy, dy)))
    rim = k.m("MULTIPLY", k.mr(dist, 0.62, 0.30), k.mr(s.outputs["Z"], BED_TOP - 0.1, BED_TOP + 0.05))
    soot = k.m("MAXIMUM", soot, k.m("MULTIPLY", rim, 0.7))
    soot = k.m("MULTIPLY", soot, k.mr(big, 0.25, 0.65, 0.65, 1.0), clamp=True)
    soot = k.m("ADD", soot, k.m("MULTIPLY", cv, 0.25), clamp=True)
    # calcined (heat-paled, cracked) stone right at the pit edge
    calc = k.m("MULTIPLY", k.mr(dist, 0.40, 0.26), k.mr(s.outputs["Z"], BED_TOP + 0.02, BED_TOP + 0.08))
    cracks = k.mr(k.voronoi(pos, 22.0, "DISTANCE_TO_EDGE", "Distance"), 0.0, 0.03, 1.0, 0.0)
    col = k.mix(k.m("MULTIPLY", calc, 0.75), col, (0.30, 0.25, 0.20))
    col = k.mix(k.m("MULTIPLY", e, 0.35), col, k.mul_col(col, (1.35, 1.33, 1.3)))
    col = k.mix(k.m("MULTIPLY", cv, 0.6), col, k.mul_col(col, (0.45, 0.43, 0.42)))
    sootc = k.mix(k.mr(fine, 0.3, 0.7), (0.016, 0.015, 0.014), (0.045, 0.040, 0.036))
    col = k.mix(soot, col, sootc)
    # ash dust settled on up-facing surfaces near the fire, floor grime at the base
    nz = k.N.new("ShaderNodeSeparateXYZ")
    k.L.new(k.geo.outputs["Normal"], nz.inputs[0])
    up = k.mr(nz.outputs["Z"], 0.6, 0.95)
    dust = k.m("MULTIPLY", up, k.mr(mid, 0.35, 0.7, 0.2, 0.75))
    col = k.mix(k.m("MULTIPLY", dust, k.mr(dist, 0.9, 0.35, 0.25, 0.7)), col, (0.24, 0.23, 0.21))
    base_g = k.mr(s.outputs["Z"], 0.25, 0.0, 0.0, 0.55)
    col = k.mix(k.m("MULTIPLY", base_g, k.mr(big, 0.3, 0.7, 0.5, 1.0)), col, (0.07, 0.06, 0.05))
    col = k.mix(k.m("MULTIPLY", cracks, k.m("MULTIPLY", calc, 0.8)), col, (0.06, 0.05, 0.045))
    rough = k.mr(fine, 0.3, 0.7, 0.78, 0.95)
    rough = k.mixf(soot, rough, 0.9)
    h = k.m("ADD", k.m("MULTIPLY", fine, 0.6), k.m("MULTIPLY", mid, 0.5))
    h = k.m("ADD", h, k.m("MULTIPLY", k.mr(vor, 0.0, 0.1, -0.3, 0.0), 1.0))
    h = k.m("SUBTRACT", h, k.m("MULTIPLY", k.m("MULTIPLY", cracks, calc), 0.8))
    return k.out(col, rough, 0.0, k.bump(h, 0.0015, 0.55 if kind == "stone" else 0.4))


def mat_coal(kind):
    """kind 'coal' (the fire) or 'ash' (ash pit). Black, slightly glassy charcoal with a grey ash bloom; lit: an
    emissive glow from the crevices and cracks, white-yellow at the centre to dull red, out at the pit's edges."""
    k = K.NK(f"forge_{kind}{'_lit' if LIT and kind == 'coal' else ''}")
    pos = k.geo.outputs["Position"]
    s = P(k)
    mid = k.noise(pos, 25.0, 4.0, 0.6)
    fine = k.noise(pos, 160.0, 3.0, 0.6)
    cv = k.cavity(0.02, 0.95, 0.5)
    crk = k.mr(k.voronoi(pos, 70.0, "DISTANCE_TO_EDGE", "Distance"), 0.0, 0.012, 1.0, 0.0)
    nz = k.N.new("ShaderNodeSeparateXYZ")
    k.L.new(k.geo.outputs["Normal"], nz.inputs[0])
    up = k.mr(nz.outputs["Z"], 0.2, 0.9)
    dx = k.m("SUBTRACT", s.outputs["X"], FIRE_C.x)
    dy = k.m("SUBTRACT", s.outputs["Y"], FIRE_C.y)
    dist = k.m("SQRT", k.m("ADD", k.m("MULTIPLY", dx, dx), k.m("MULTIPLY", dy, dy)))
    coal = k.mix(k.mr(fine, 0.3, 0.7), (0.012, 0.011, 0.010), (0.035, 0.032, 0.030))
    if kind == "ash":
        ash = k.mix(k.mr(mid, 0.3, 0.7), (0.30, 0.29, 0.27), (0.48, 0.46, 0.43))
        col = k.mix(k.mr(mid, 0.25, 0.5, 0.0, 1.0), coal, ash)
        col = k.mix(k.m("MULTIPLY", cv, 0.6), col, (0.05, 0.045, 0.04))
        return k.out(col, 0.93, 0.0, k.bump(fine, 0.0008, 0.5))
    # ash bloom on top faces: heavy in a cold fire, light at the edges of a lit one
    ashc = k.mix(k.mr(fine, 0.3, 0.7), (0.28, 0.27, 0.25), (0.55, 0.53, 0.50))
    bloom = k.m("MULTIPLY", up, k.mr(mid, 0.35, 0.65, 0.0, 0.9 if not LIT else 0.55))
    if LIT:
        heat = k.mr(dist, 0.33, 0.0)            # 1 at the centre, 0 at ~0.33 m
        heat = k.m("MULTIPLY", heat, k.mr(mid, 0.2, 0.8, 0.75, 1.15), clamp=True)
        bloom = k.m("MULTIPLY", bloom, k.mr(heat, 0.2, 0.7, 1.0, 0.15))
        clink = k.m("MULTIPLY", k.mr(dist, 0.24, 0.40), k.mr(mid, 0.4, 0.6))
    else:
        clink = k.m("MULTIPLY", k.mr(dist, 0.20, 0.40, 0.2, 1.0), k.mr(mid, 0.45, 0.6))
    col = k.mix(bloom, coal, ashc)
    col = k.mix(k.m("MULTIPLY", clink, 0.7), col, (0.09, 0.065, 0.05))          # rusty-brown glassy clinker
    rough = k.mr(fine, 0.3, 0.7, 0.42, 0.62)
    rough = k.mixf(bloom, rough, 0.95)
    rough = k.mixf(clink, rough, 0.3)
    nrm = k.bump(k.m("ADD", fine, k.m("MULTIPLY", crk, -0.4)), 0.0006, 0.5)
    if not LIT:
        return k.out(col, rough, 0.0, nrm)
    # glow: crevices and cracks everywhere it is hot; the whole surface at the very centre
    glow = k.m("MAXIMUM", k.m("MULTIPLY", cv, 1.2), k.m("MULTIPLY", crk, 0.9), clamp=True)
    glow = k.m("MAXIMUM", glow, k.mr(heat, 0.55, 0.95))
    glow = k.m("MAXIMUM", glow, k.m("MULTIPLY", k.mr(nz.outputs["Z"], 0.2, -0.6), 0.9))   # undersides of lumps
    g = k.m("MULTIPLY", glow, heat, clamp=True)
    g = k.m("MULTIPLY", g, k.m("SUBTRACT", 1.0, k.m("MULTIPLY", bloom, 0.8)), clamp=True)
    ecol = k.ramp(g, [(0.0, (0.0, 0.0, 0.0)), (0.12, (0.35, 0.025, 0.0)), (0.4, (0.9, 0.16, 0.01)),
                      (0.75, (1.0, 0.42, 0.06)), (1.0, (1.0, 0.78, 0.38))])
    # hot surfaces read orange in the base colour too (so a still without bloom still shows heat)
    col = k.mix(k.mr(g, 0.25, 0.9, 0.0, 0.85), col, (0.55, 0.12, 0.02))
    return k.out(col, rough, 0.0, nrm, ecol, 14.0)


M_STONE = mat_stone("stone")
M_MORTAR = mat_stone("mortar")
M_IRON = K.metal("iron", tarnish=0.7, soot=0.85, rust=0.35, polish=0.15, soot_z=(HOOD_Z0 - 0.2, HOOD_Z1 + 0.3),
                 name="forge_hood_iron")
M_TUY = K.metal("iron", tarnish=0.8, soot=0.6, rust=0.5, polish=0.2, name="forge_tuyere_iron")
M_COAL = mat_coal("coal")
M_ASH = mat_coal("ash")

for o in stones:
    K.assign(o, M_STONE)
for o in mortar:
    K.assign(o, M_MORTAR)
for o in irons:
    K.assign(o, M_TUY if o.name.startswith(("tuyere", "flange", "fl", "pot_rim")) else M_IRON)
for o in coals:
    K.assign(o, M_COAL)
for o in ashes:
    K.assign(o, M_ASH)
