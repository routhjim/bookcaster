"""Model script: a heap of wood charcoal on the floor, ~0.95 m x 0.85 m and 0.33 m high: a conical irregular mound
(the heap's body) covered with ~550 pieces, about half split angular sticks showing the wood's form, the rest
broken chunks, with grey charcoal dust spreading at the foot.
"""

import math
import os
import random
import sys

import bmesh
from mathutils import Vector, noise

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
rng = random.Random(15)
AX, AY, PEAK = 0.44, 0.40, 0.30


def hgt(x, y):
    d = math.sqrt((x / AX) ** 2 + (y / AY) ** 2)
    a = math.atan2(y, x)
    d *= 1 + 0.12 * noise.noise(Vector((math.cos(a) * 1.5, math.sin(a) * 1.5, 0.3)))
    if d >= 1:
        return 0.0
    return PEAK * (1 - d ** 1.25) ** 1.2 + 0.012 * noise.noise(Vector((x * 9, y * 9, 1.0)))


bm = bmesh.new()
n = 40
top, bot = [], []
for i in range(n + 1):
    rt, rb = [], []
    for j in range(n + 1):
        x, y = K.lerp(-AX * 1.15, AX * 1.15, i / n), K.lerp(-AY * 1.15, AY * 1.15, j / n)
        edge = i in (0, n) or j in (0, n)
        rt.append(bm.verts.new((x, y, 0.002 if edge else max(0.002, hgt(x, y)))))
        rb.append(bm.verts.new((x, y, 0.0)))
    top.append(rt)
    bot.append(rb)
for i in range(n):
    for j in range(n):
        bm.faces.new((top[i][j], top[i + 1][j], top[i + 1][j + 1], top[i][j + 1]))
        bm.faces.new((bot[i][j], bot[i][j + 1], bot[i + 1][j + 1], bot[i + 1][j]))
ring = [(i, 0) for i in range(n)] + [(n, j) for j in range(n)] + [(i, n) for i in range(n, 0, -1)] + \
       [(0, j) for j in range(n, 0, -1)]
for k in range(len(ring)):
    (a, b), (c, d) = ring[k], ring[(k + 1) % len(ring)]
    bm.faces.new((top[a][b], top[c][d], bot[c][d], bot[a][b]))
body = K.from_bm("heap", bm)
K.shade(body, 80)

lumps = []


def lump(x, y, z, r, long=False):
    """A broken chunk (rough icosphere) or, when long, a charcoal stick: a split, angular billet of wood."""
    bm = bmesh.new()
    off = Vector((rng.uniform(0, 99), rng.uniform(0, 99), rng.uniform(0, 99)))
    if long:
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=2, use_grid_fill=True)
        sc = Vector((rng.uniform(2.2, 3.6), rng.uniform(0.8, 1.1), rng.uniform(0.7, 0.95)))
        amp = 0.16
    else:
        bmesh.ops.create_icosphere(bm, subdivisions=1 if r < 0.016 else 2, radius=1.0)
        sc = Vector((rng.uniform(0.9, 1.4), rng.uniform(0.7, 1.1), rng.uniform(0.55, 0.85)))
        amp = 0.22
    for v in bm.verts:
        p = v.co
        k = 1.0 + amp * noise.noise(p * 1.7 + off)
        v.co = Vector((p.x * sc.x, p.y * sc.y, p.z * sc.z)) * k * r
    ob = K.from_bm(f"lump{len(lumps)}", bm)
    ob.location = (x, y, z)
    ob.rotation_euler = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0, 6.28))
    for p in ob.data.polygons:
        p.use_smooth = False
    lumps.append(ob)


placed = []
tries = 0
while len(placed) < 560 and tries < 12000:
    tries += 1
    x, y = rng.uniform(-AX * 1.05, AX * 1.05), rng.uniform(-AY * 1.05, AY * 1.05)
    h = hgt(x, y)
    if h < 0.004 and rng.random() < 0.85:
        continue
    r = rng.uniform(0.012, 0.04) if rng.random() < 0.8 else rng.uniform(0.008, 0.014)
    if any((x - a) ** 2 + (y - b) ** 2 < ((r + c) * 0.7) ** 2 for a, b, c in placed):
        continue
    placed.append((x, y, r))
    lump(x, y, h + r * 0.1, r * (0.75 if r > 0.02 else 1.0), long=(r > 0.02 and rng.random() < 0.55))

M_COAL_K = K.NK("charcoal")
k = M_COAL_K
fine = k.noise(k.obj, 200.0, 3.0, 0.6)
mid = k.noise(k.geo.outputs["Position"], 25.0, 4.0, 0.6)
grain = k.noise(k.obj, 1.0, 3.0, 0.6, stretch=(3.0, 300.0, 300.0))      # cracked wood grain on the lumps
cv = k.cavity(0.02, 0.95, 0.5)
col = k.mix(k.mr(fine, 0.3, 0.7), (0.010, 0.0095, 0.009), (0.030, 0.028, 0.027))
col = k.mix(k.mr(grain, 0.6, 0.75, 0.0, 0.6), col, (0.004, 0.004, 0.004))
nz = k.N.new("ShaderNodeSeparateXYZ")
k.L.new(k.geo.outputs["Normal"], nz.inputs[0])
dust = k.m("MULTIPLY", k.mr(nz.outputs["Z"], 0.4, 0.95), k.mr(mid, 0.45, 0.7, 0.0, 0.55))
col = k.mix(dust, col, (0.10, 0.098, 0.095))
col = k.mix(k.m("MULTIPLY", cv, 0.7), col, (0.004, 0.004, 0.004))
rough = k.mixf(dust, k.mr(grain, 0.3, 0.7, 0.35, 0.55), 0.9)
M_COAL = k.out(col, rough, 0.0, k.bump(k.m("ADD", fine, k.m("MULTIPLY", grain, 0.6)), 0.0006, 0.6))
K.assign(body, M_COAL)
for o in lumps:
    K.assign(o, M_COAL)
