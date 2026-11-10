"""Model script: a forge bellows lying on the floor, ~0.97 m long along X (boards ~0.7 m), nozzle to +X.

Two teardrop oak boards with handle tabs at the back (-X); the top board is hinged near the nose and stands open
at the back. Pleated leather sides (a lofted tube whose rows bulge out in folds, more where the opening is wider),
an iron-strapped nose block, an iron collar and tapered nozzle, and a row of round-head tacks along the leather on
both boards. Materials: weathered oak (stripes, rays, silvered patches, sooty nose, dark handle grips), dark oiled
leather (crackle, pale scuffs on fold ridges, dark folds), black iron with rust and worn edges.
"""

import math

import bmesh
import bpy
from mathutils import Vector

# ------------------------------------------------------------------ helpers (geometry)
def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def superellipse2d(hw, hh, n, rot, seg):
    pts = []
    for k in range(seg):
        a = 2 * math.pi * (k + 0.5) / seg
        c, s = math.cos(a), math.sin(a)
        u = hw * math.copysign(abs(c) ** (2 / n), c)
        v = hh * math.copysign(abs(s) ** (2 / n), s)
        cr, sr = math.cos(rot), math.sin(rot)
        pts.append((u * cr - v * sr, u * sr + v * cr))
    return pts


def sweep(name, path, sections, cap_start=True, cap_end=True):
    """path: centreline points; sections: one 2D point list per path point (same count), placed in a
    parallel-transport frame (N, B) around the tangent."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    path = [Vector(p) for p in path]
    tang = []
    for i in range(len(path)):
        a = path[max(i - 1, 0)]
        b = path[min(i + 1, len(path) - 1)]
        tang.append((b - a).normalized())
    up = Vector((0, 0, 1)) if abs(tang[0].z) < 0.9 else Vector((1, 0, 0))
    n = (up - tang[0] * up.dot(tang[0])).normalized()
    rows = []
    for i, (p, t) in enumerate(zip(path, tang)):
        if i:
            n = (n - t * n.dot(t)).normalized()
        b = t.cross(n)
        rows.append([bm.verts.new(p + n * v + b * u) for (u, v) in sections[i]])
    k = len(rows[0])
    for r0, r1 in zip(rows, rows[1:]):
        for j in range(k):
            bm.faces.new((r0[j], r0[(j + 1) % k], r1[(j + 1) % k], r1[j]))
    if cap_start:
        bm.faces.new(list(reversed(rows[0])))
    if cap_end:
        bm.faces.new(rows[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def shade(ob, angle=40):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))


def finalize(objs):
    """Apply transforms and move the floor centre of the bounding box to the origin, so texture coordinates
    (object space) match what the harness bakes."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    lo = Vector((1e9,) * 3)
    hi = Vector((-1e9,) * 3)
    for o in objs:
        for v in o.data.vertices:
            for i in range(3):
                lo[i] = min(lo[i], v.co[i])
                hi[i] = max(hi[i], v.co[i])
    shift = Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    for o in objs:
        o.data.transform(__import__("mathutils").Matrix.Translation(shift))
    return shift


# ------------------------------------------------------------------ plan shapes
B_T = 0.022                                          # board thickness
X_BACK, X_NOSE, X_HINGE = -0.35, 0.35, 0.30
R_BACK, C_BACK, HW_NOSE = 0.18, -0.17, 0.046
X_GRIP = -0.47


def hw_body(x):
    """Half width of the teardrop board body at x (0 outside)."""
    if x < X_BACK or x > X_NOSE:
        return 0.0
    if x <= C_BACK:
        return math.sqrt(max(R_BACK ** 2 - (x - C_BACK) ** 2, 0.0))
    return lerp(R_BACK, HW_NOSE, smooth((x - C_BACK) / (X_NOSE - C_BACK)) ** 0.9)


def hw_board(x):
    """Board outline including the handle tab at the back."""
    h = 0.0
    if X_GRIP <= x <= -0.28:
        h = 0.027
        if x < X_GRIP + 0.024:
            h = 0.027 * math.sqrt(max(1 - ((X_GRIP + 0.024 - x) / 0.024) ** 2, 0.0))
        h *= 1 + 0.25 * smooth((x + 0.36) / 0.08)    # flares into the board
    return max(hw_body(x), h)


def gap(x):
    """Opening between the boards (inner faces) at x: hinged near the nose, open at the back."""
    return 0.020 + (X_HINGE - x) / (X_HINGE - X_BACK) * 0.150


def outline(hwf, x0, x1, n):
    """Closed outline (counter-clockwise from above): upper side +Y then lower side -Y."""
    xs = [x0 + (x1 - x0) * (0.5 - 0.5 * math.cos(math.pi * i / n)) for i in range(n + 1)]
    top = [(x, hwf(x)) for x in xs]
    bot = [(x, -hwf(x)) for x in reversed(xs)]
    pts = top + bot
    # dedupe zero-width ends
    out = []
    for p in pts:
        if not out or math.dist(p, out[-1]) > 0.0015:
            out.append(p)
    if math.dist(out[0], out[-1]) < 0.0015:
        out.pop()
    return out


def slab(name, pts, zb, zt):
    """Prism over a closed 2D outline: bottom z = zb(x), top z = zt(x)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    lo = [bm.verts.new((x, y, zb(x))) for x, y in pts]
    hi = [bm.verts.new((x, y, zt(x))) for x, y in pts]
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((lo[i], lo[j], hi[j], hi[i]))
    bm.faces.new(lo[::-1])
    bm.faces.new(hi)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bmesh.ops.dissolve_degenerate(bm, dist=2e-5, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def bevel(ob, w, seg=2, angle=40):
    m = ob.modifiers.new("bv", "BEVEL")
    m.width, m.segments, m.limit_method, m.angle_limit = w, seg, "ANGLE", math.radians(angle)
    m.harden_normals = False
    shade(ob, angle)


# ------------------------------------------------------------------ boards
bpts = outline(hw_board, X_GRIP, X_NOSE, 120)
bottom = slab("board_bottom", bpts, lambda x: 0.0, lambda x: B_T)
top = slab("board_top", bpts, lambda x: B_T + gap(x), lambda x: 2 * B_T + gap(x))
bevel(bottom, 0.004)
bevel(top, 0.004)

# ------------------------------------------------------------------ pleated leather between the boards
INSET, FOLDS, ROWS = 0.012, 6, 6
lx0, lx1 = X_BACK + INSET, 0.29
lpts = outline(lambda x: max(hw_body(x) - INSET, 0.0), lx0, lx1, 110)
lpts = [p for p in lpts]
nL = len(lpts)
norms = []
for i in range(nL):
    a, b = Vector(lpts[i - 1] + (0,)), Vector(lpts[(i + 1) % nL] + (0,))
    t = (b - a).normalized()
    norms.append(Vector((t.y, -t.x, 0)))
# orientation: make normals point outward (away from the centroid)
cx = sum(p[0] for p in lpts) / nL
cy = sum(p[1] for p in lpts) / nL
if sum((Vector((p[0] - cx, p[1] - cy, 0))).dot(nn) for p, nn in zip(lpts, norms)) < 0:
    norms = [-v for v in norms]
me = bpy.data.meshes.new("leather")
bm = bmesh.new()
rows = []
fold_rows = []                                       # per row: 0 in the creases, 1 on the outer ridges
nrow = FOLDS * ROWS
for r in range(nrow + 1):
    s = r / nrow
    f = (s * FOLDS) % 1.0
    tri = 1 - abs(2 * f - 1)                         # 0 at the folds' inner creases, 1 at the outer ridges
    bulge = 0.5 - 0.5 * math.cos(tri * math.pi)
    row = []
    fold_rows.append(bulge)
    for (x, y), nn in zip(lpts, norms):
        g = gap(x)
        z = lerp(B_T - 0.002, B_T + g + 0.002, s)
        amp = 0.003 + 0.06 * g
        # irregular hand-made folds: ridge height and crease depth wander along the outline and per fold
        k = int(s * FOLDS)
        wob = 0.22 * math.sin(x * 37 + k * 1.7) * math.sin(y * 29 - k * 2.3) + 0.12 * math.sin(x * 113 + y * 71 + k)
        o = amp * bulge * (1 + wob) + 0.0012 * math.sin(x * 260 + y * 190 + r * 0.9) * bulge
        row.append(bm.verts.new((x + nn.x * o, y + nn.y * o, z)))
    rows.append(row)
for r0, r1 in zip(rows, rows[1:]):
    for i in range(nL):
        j = (i + 1) % nL
        bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
bm.faces.new(rows[0][::-1])
bm.faces.new(rows[-1])
bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
bmesh.ops.dissolve_degenerate(bm, dist=2e-5, edges=bm.edges)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bm.to_mesh(me)
bm.free()
fa = me.attributes.new("fold", "FLOAT", "POINT")      # read by the leather material (ridge scuffs, crease dirt)
for k, v in enumerate(fa.data):
    v.value = fold_rows[min(k // nL, nrow)]
leather = bpy.data.objects.new("leather", me)
bpy.context.scene.collection.objects.link(leather)
shade(leather, 70)

# ------------------------------------------------------------------ nose block, iron band, nozzle
nb = slab("nose", outline(lambda x: hw_body(x) - 0.004, 0.24, X_NOSE - 0.002, 16), lambda x: B_T - 0.001,
          lambda x: B_T + gap(x) + 0.001)
bevel(nb, 0.002)
ZN = B_T + gap(X_NOSE) / 2
bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.026, depth=0.016, location=(X_NOSE + 0.004, 0, ZN),
                                    rotation=(0, math.pi / 2, 0))
collar = bpy.context.active_object
collar.name = "nozzle_collar"
bevel(collar, 0.002)
noz_path = [(X_NOSE - 0.03 + 0.17 * i / 30, 0.0, ZN) for i in range(31)]
noz_secs = []
for i in range(31):
    t = i / 30
    r = lerp(0.021, 0.0105, smooth(t) ** 0.8)
    if t > 0.93:
        r *= 1 + 0.18 * smooth((t - 0.93) / 0.07)          # rolled lip
    noz_secs.append(superellipse2d(r, r, 2.0, 0.0, 32))
nozzle = sweep("nozzle", noz_path, noz_secs)
shade(nozzle, 50)
# iron straps across the nose of each board
straps = []
for (z0f, nm) in ((lambda x: B_T, "strap_b"), (lambda x: 2 * B_T + gap(x), "strap_t")):
    s = slab(nm, outline(lambda x: hw_board(x) + 0.0015, 0.255, 0.285, 4), z0f, lambda x, f=z0f: f(x) + 0.003)
    bevel(s, 0.0008)
    straps.append(s)

# ------------------------------------------------------------------ tacks along the leather edge on both boards
tacks = []
tpts = outline(lambda x: max(hw_body(x) - INSET + 0.006, 0.0), lx0 - 0.006, lx1, 70)
acc, last = 0.0, None
for p in tpts:
    if last is not None:
        acc += math.dist(p, last)
    last = p
    if acc >= 0.034 or len(tacks) == 0:
        acc = 0.0
        for zt, sgn in ((B_T, 1), (B_T + gap(p[0]), -1)):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=5, radius=0.0048,
                                                 location=(p[0], p[1], zt - sgn * 0.0018))
            t = bpy.context.active_object
            t.scale.z = 0.55
            tacks.append(t)
for t in tacks:
    shade(t, 80)

finalize([bottom, top, leather, nb, collar, nozzle] + straps + tacks)

# ------------------------------------------------------------------ material helpers (use the current N, L)
def node(t, **kw):
    n = N.new(t)
    for k, v in kw.items():
        n.inputs[k].default_value = v
    return n


def mth(op, a, b=None, clamp=True):
    n = N.new("ShaderNodeMath")
    n.operation, n.use_clamp = op, clamp
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            L.new(x, n.inputs[i])
    return n.outputs[0]


def mr(x, a, b, c=0.0, d=1.0):
    n = node("ShaderNodeMapRange")
    n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = a, b
    n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = c, d
    L.new(x, n.inputs["Value"])
    return n.outputs["Result"]


def mix(f, a, b):
    n = N.new("ShaderNodeMix")
    n.data_type = "RGBA"
    for sock, v in ((n.inputs[6], a), (n.inputs[7], b)):
        if isinstance(v, tuple):
            sock.default_value = v
        else:
            L.new(v, sock)
    if isinstance(f, (int, float)):
        n.inputs["Factor"].default_value = f
    else:
        L.new(f, n.inputs["Factor"])
    return n.outputs[2]


def mixf(f, a, b):
    n = N.new("ShaderNodeMix")
    for sock, v in ((n.inputs[2], a), (n.inputs[3], b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        else:
            L.new(v, sock)
    L.new(f, n.inputs["Factor"])
    return n.outputs[0]


def noise(vec, scale, detail=4.0, rough=0.5, stretch=None):
    if stretch:
        mp = N.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = stretch
        L.new(vec, mp.inputs["Vector"])
        vec = mp.outputs["Vector"]
    n = node("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough)
    L.new(vec, n.inputs["Vector"])
    return n.outputs["Fac"]



# ------------------------------------------------------------------ materials
def new_mat(name):
    global N, L
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links
    tc = N.new("ShaderNodeTexCoord")
    geo = N.new("ShaderNodeNewGeometry")
    sp = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sp.inputs["Vector"])
    return m, N["Principled BSDF"], tc.outputs["Object"], geo, sp


def edge_mask(geo, radius, lo=0.995, hi=0.94):
    bev = node("ShaderNodeBevel", Radius=radius)
    bev.samples = 16
    dp = N.new("ShaderNodeVectorMath")
    dp.operation = "DOT_PRODUCT"
    L.new(bev.outputs["Normal"], dp.inputs[0])
    L.new(geo.outputs["Normal"], dp.inputs[1])
    return mr(dp.outputs["Value"], lo, hi)


def cavity(dist, lo=0.9, hi=0.5):
    ao = N.new("ShaderNodeAmbientOcclusion")
    ao.samples = 16
    ao.inputs["Distance"].default_value = dist
    return mr(ao.outputs["AO"], lo, hi)


# weathered oak, sooty toward the nozzle, worn pale on the edges and handle grips
wood, pr, obj, geo, sp = new_mat("bellows_oak")
gmap = N.new("ShaderNodeMapping")
gmap.inputs["Location"].default_value = (0.0, 0.6, 0.0)           # pith beside the board: rings -> stripes along X
gmap.inputs["Rotation"].default_value = (0.0, 0.0, 0.025)
L.new(obj, gmap.inputs["Vector"])
gw = node("ShaderNodeTexWave", Scale=22.0, Distortion=4.0, Detail=5.0, **{"Detail Scale": 0.8, "Detail Roughness": 0.6})
gw.wave_type, gw.rings_direction = "RINGS", "X"
L.new(gmap.outputs["Vector"], gw.inputs["Vector"])
gr = mr(gw.outputs["Fac"], 0.2, 0.9)
pores = noise(obj, 1.0, 4.0, 0.6, stretch=(20.0, 700.0, 700.0))
rays = noise(obj, 1.0, 3.0, 0.5, stretch=(60.0, 6.0, 400.0))         # oak medullary flecks
blot = noise(obj, 25.0, 4.0, 0.55)
big = noise(obj, 4.0, 3.0, 0.5)
fine = noise(obj, 500.0, 2.0, 0.5)
edge = edge_mask(geo, 0.004)
soot = mth("MULTIPLY", mr(sp.outputs["X"], 0.10, 0.34), mr(big, 0.3, 0.6, 0.5, 1.0))
grips = mr(sp.outputs["X"], -0.38, -0.44)
col = mix(gr, (0.20, 0.145, 0.095, 1), (0.105, 0.072, 0.045, 1))
col = mix(mth("MULTIPLY", mr(rays, 0.62, 0.72), 0.5), col, (0.36, 0.26, 0.16, 1))
col = mix(mth("MULTIPLY", pores, 0.55), col, (0.08, 0.05, 0.03, 1))
col = mix(mr(big, 0.4, 0.7, 0.0, 0.35), col, (0.20, 0.185, 0.165, 1))                 # silvered weathering
col = mix(mr(blot, 0.5, 0.75, 0.0, 0.5), col, (0.06, 0.045, 0.03, 1))                # grime
col = mix(mth("MULTIPLY", edge, mr(blot, 0.3, 0.6, 0.4, 1.0)), col, (0.38, 0.30, 0.20, 1))
col = mix(mth("MULTIPLY", grips, 0.7), col, (0.12, 0.075, 0.04, 1))
col = mix(soot, col, (0.025, 0.022, 0.02, 1))
L.new(col, pr.inputs["Base Color"])
L.new(mixf(grips, mr(fine, 0.3, 0.7, 0.68, 0.86), 0.42), pr.inputs["Roughness"])
hgt = mth("ADD", mth("MULTIPLY", gr, 0.4), mth("MULTIPLY", pores, 0.6))
hgt = mth("ADD", hgt, mth("MULTIPLY", mr(big, 0.4, 0.7), 0.3))
bump = node("ShaderNodeBump", Distance=0.0008, Strength=0.6)
L.new(hgt, bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])

# leather: dark oiled brown, creased, scuffed pale on the outer fold ridges, dark in the folds
lea, pr, obj, geo, sp = new_mat("bellows_leather")
cav = cavity(0.02, 0.92, 0.45)
cell = node("ShaderNodeTexVoronoi", Scale=180.0, Randomness=1.0)
cell.feature = "DISTANCE_TO_EDGE"
L.new(obj, cell.inputs["Vector"])
crack = mr(cell.outputs["Distance"], 0.0, 0.06, 1.0, 0.0)                 # fine crackle lines
grainL = noise(obj, 900.0, 3.0, 0.6)
blot = noise(obj, 22.0, 4.0, 0.55)
big = noise(obj, 5.0, 3.0, 0.5)
fattr = N.new("ShaderNodeAttribute")
fattr.attribute_name = "fold"
crest = mr(fattr.outputs["Fac"], 0.7, 1.0)
crease = mr(fattr.outputs["Fac"], 0.35, 0.0)
cav = mth("MAXIMUM", mth("MULTIPLY", cav, 0.6), crease)
ridge = mth("MULTIPLY", crest, mr(blot, 0.35, 0.65, 0.3, 1.0))
# ridges: where AO is open and the surface faces out sideways
sn = N.new("ShaderNodeSeparateXYZ")
L.new(geo.outputs["Normal"], sn.inputs["Vector"])
side = mr(mth("ABSOLUTE", sn.outputs["Z"]), 0.6, 0.15)
scuff = mth("MULTIPLY", mth("MULTIPLY", ridge, side), mr(grainL, 0.4, 0.6, 0.4, 0.9))
col = mix(mr(big, 0.3, 0.7), (0.040, 0.017, 0.008, 1), (0.068, 0.030, 0.013, 1))
col = mix(mth("MULTIPLY", scuff, 0.75), col, (0.16, 0.085, 0.045, 1))
col = mix(mth("MULTIPLY", crack, 0.35), col, (0.04, 0.022, 0.012, 1))
col = mix(mth("MULTIPLY", cav, 0.8), col, (0.025, 0.015, 0.01, 1))
col = mix(mth("MULTIPLY", mr(sp.outputs["X"], 0.12, 0.3), 0.6), col, (0.03, 0.025, 0.02, 1))   # soot at the nose
L.new(col, pr.inputs["Base Color"])
L.new(mixf(scuff, mr(blot, 0.3, 0.7, 0.34, 0.52), 0.7), pr.inputs["Roughness"])
pr.inputs["Sheen Weight"].default_value = 0.2
hgt = mth("ADD", mth("MULTIPLY", grainL, 0.5), mth("MULTIPLY", crack, -0.4))
bump = node("ShaderNodeBump", Distance=0.0005, Strength=0.6)
L.new(hgt, bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])

# black iron for the nozzle, collar, straps and tacks
iron, pr, obj, geo, sp = new_mat("bellows_iron")
edge = edge_mask(geo, 0.002)
blot = noise(obj, 80.0, 5.0, 0.55)
big = noise(obj, 12.0, 3.0, 0.5)
fine = noise(obj, 600.0, 3.0, 0.6)
rust = mth("MULTIPLY", mr(blot, 0.6, 0.72), mr(big, 0.35, 0.6, 0.3, 1.0))
hot = mr(sp.outputs["X"], 0.43, 0.52)                                     # heat-scaled nozzle tip
bright = mth("MULTIPLY", edge, mr(blot, 0.35, 0.6, 0.2, 0.9))
col = mix(mr(big, 0.3, 0.7), (0.02, 0.019, 0.018, 1), (0.05, 0.045, 0.04, 1))
col = mix(mth("MULTIPLY", hot, 0.7), col, (0.09, 0.06, 0.045, 1))
col = mix(rust, col, mix(fine, (0.14, 0.06, 0.025, 1), (0.23, 0.10, 0.04, 1)))
col = mix(bright, col, (0.33, 0.32, 0.31, 1))
L.new(col, pr.inputs["Base Color"])
L.new(mixf(bright, mixf(rust, mr(fine, 0.3, 0.7, 0.5, 0.72), 0.9), 0.3), pr.inputs["Roughness"])
L.new(mixf(bright, mixf(rust, 0.3, 0.0), 1.0), pr.inputs["Metallic"])
vor = node("ShaderNodeTexVoronoi", Scale=120.0, Randomness=0.9)
vor.feature = "SMOOTH_F1"
L.new(obj, vor.inputs["Vector"])
bump = node("ShaderNodeBump", Distance=0.0004, Strength=0.5)
L.new(mth("SUBTRACT", mr(vor.outputs["Distance"], 0.1, 0.9), mth("MULTIPLY", rust, 0.5)), bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])

for o in (bottom, top, nb):
    o.data.materials.append(wood)
leather.data.materials.append(lea)
for o in [collar, nozzle] + straps + tacks:
    o.data.materials.append(iron)
