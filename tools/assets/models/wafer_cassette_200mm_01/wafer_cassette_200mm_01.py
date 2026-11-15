"""Model script: a 200 mm wafer cassette (open carrier) holding 25 wafers, ~0.22 m wide (X), 0.19 m deep (Y).

Wafers stand in the XZ plane, one per slot along Y (6.35 mm pitch). Two side walls follow the wafers' curve down
to the bottom rails, with a comb tooth between every two wafers; a U-frame end plate at -Y (the wafer faces show
through it) and an H-bar end plate with a stand-off grip handle at +Y. Materials: white moulded plastic (flow
lines, yellowing in the slots, grey scuffs on edges and rails); polished silicon wafers with a die grid, scribe
lines and thin-film tints on both faces. Opaque only: the chain bakes no transmission.
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


P = {"pitch": 0.00635, "n_wafers": 25, "wafer_r": 0.100, "wafer_t": 0.00075, "wall_t": 0.0038, "y_half": 0.090,
     "tooth_t": 0.0030, "tooth_depth": 0.0075}
Z_WAFER = 0.121                                       # wafer centre height


def drop_collinear(poly, eps=1e-9):
    """Remove repeated and collinear corners (they triangulate into zero-area faces)."""
    out = list(poly)
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cross) < eps or math.dist(a, b) < 1e-6:
                out.pop(i)
                changed = True
                break
    return out


def prism_xz(name, poly, y0, y1):
    """Closed polygon in the XZ plane (counter-clockwise seen from -Y) extruded from y0 to y1."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    poly = drop_collinear(poly)
    a = [bm.verts.new((x, y0, z)) for x, z in poly]
    b = [bm.verts.new((x, y1, z)) for x, z in poly]
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((a[i], a[j], b[j], b[i]))
    bm.faces.new(a[::-1])
    bm.faces.new(b)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def wall_strip(name, outer, inner, y0, y1):
    """A thin wall between two matching polylines in XZ, extruded from y0 to y1, built from quads only."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    ring = outer + inner[::-1]
    a = [bm.verts.new((x, y0, z)) for x, z in ring]
    b = [bm.verts.new((x, y1, z)) for x, z in ring]
    n, m = len(ring), len(outer)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((a[i], a[j], b[j], b[i]))
    for i in range(m - 1):                           # end caps as quads across the wall thickness
        o0, o1, i0, i1 = i, i + 1, n - 1 - i, n - 2 - i
        bm.faces.new((a[o0], a[i0], a[i1], a[o1]))
        bm.faces.new((b[o0], b[o1], b[i1], b[i0]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def offset_line(pts, d):
    """Offset an open polyline in XZ by d along its left normal (miter at the joints)."""
    out = []
    for i, p in enumerate(pts):
        a = Vector(pts[max(i - 1, 0)])
        b = Vector(pts[min(i + 1, len(pts) - 1)])
        t = (b - a).normalized()
        nrm = Vector((-t.y, t.x))
        out.append((p[0] + nrm.x * d, p[1] + nrm.y * d))
    return out


def resample(pts, step):
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.dist(a, b) / step))
        out += [(lerp(a[0], b[0], k / n), lerp(a[1], b[1], k / n)) for k in range(1, n + 1)]
    return out


# right wall profile, top to bottom (outer face); the left wall mirrors it
WALL = resample([(0.1085, 0.205), (0.1085, 0.112), (0.1030, 0.088), (0.0870, 0.050), (0.0640, 0.022),
                 (0.0560, 0.010), (0.0540, 0.000)], 0.006)
parts = []
for side in (1, -1):
    outer = [(side * x, z) for x, z in WALL]
    inner = offset_line(outer, -side * P["wall_t"])
    w = wall_strip(f"wall_{'r' if side == 1 else 'l'}", outer, inner, -P["y_half"], P["y_half"])
    shade(w, 35)
    parts.append(w)
    # top lip: a thicker rounded rail along the wall top
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.0042, depth=2 * P["y_half"],
                                        location=(side * 0.1068, 0, 0.2035), rotation=(math.pi / 2, 0, 0))
    lip = bpy.context.active_object
    shade(lip, 40)
    parts.append(lip)
    # outer stiffening ribs: every fifth slot, following the wall profile
    rib = [p for p in outer if p[1] >= 0.012]
    rib_out = offset_line(rib, side * 0.0035)
    rpoly = rib + rib_out[::-1]
    for k in range(0, P["n_wafers"] + 1, 5):
        y = (k - P["n_wafers"] / 2) * P["pitch"]
        parts.append(prism_xz(f"rib_{side}_{k}", rpoly, y - 0.0014, y + 0.0014))
    # slot teeth: one comb tooth on the inner face between every two wafers (and at both ends)
    band = [p for p in inner if 0.026 <= p[1] <= 0.196]
    tip = offset_line(band, -side * P["tooth_depth"])
    back = offset_line(band, side * 0.0006)
    tpoly = back + tip[::-1]
    for k in range(P["n_wafers"] + 1):
        y = (k - P["n_wafers"] / 2) * P["pitch"]
        t = prism_xz(f"tooth_{side}_{k}", tpoly, y - P["tooth_t"] / 2, y + P["tooth_t"] / 2)
        parts.append(t)

# end plates: U frame at -Y; H-bar with a grip handle at +Y
# plate polygon: right wall bottom -> top, across the top to the left wall top -> bottom (counter-clockwise from -Y)
right = [(x, z) for x, z in WALL[::-1]]
left = [(-x, z) for x, z in WALL]
plate = right + left
for y0, y1, name, hbar in ((-P["y_half"] - 0.004, -P["y_half"] + 0.002, "end_front", False),
                           (P["y_half"] - 0.002, P["y_half"] + 0.004, "end_back", True)):
    pl = prism_xz(name, plate[::-1], y0, y1)
    cuts = [(0.030, 0.186)] if not hbar else [(0.030, 0.098), (0.130, 0.186)]
    for z0, z1 in cuts:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, (y0 + y1) / 2, (z0 + z1) / 2))
        c = bpy.context.active_object
        c.scale = (0.150 if not hbar else (0.150 if z0 > 0.11 else 0.110), 0.05, z1 - z0)
        bvc = c.modifiers.new("r", "BEVEL")
        bvc.width, bvc.segments = 0.012, 4
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        m = pl.modifiers.new("cut", "BOOLEAN")
        m.operation, m.object, m.solver = "DIFFERENCE", c, "EXACT"
        bpy.context.view_layer.objects.active = pl
        bpy.ops.object.select_all(action="DESELECT")
        pl.select_set(True)
        bpy.ops.object.modifier_apply(modifier="cut")
        bpy.data.objects.remove(c, do_unlink=True)
    bv = pl.modifiers.new("bv", "BEVEL")
    bv.width, bv.segments, bv.limit_method, bv.angle_limit = 0.0008, 2, "ANGLE", math.radians(50)
    shade(pl, 20)
    parts.append(pl)

# grip handle: a rounded bar standing off the H-bar
hp = []
for i in range(41):
    t = i / 40
    x = lerp(-0.055, 0.055, t)
    out = math.sin(math.pi * t) ** 0.35
    hp.append((x, P["y_half"] + 0.001 + 0.024 * out, 0.114))
handle = sweep("handle", hp, [superellipse2d(0.0055, 0.0075, 2.6, 0.0, 20) for _ in hp])
shade(handle, 50)
parts.append(handle)
# bottom rails joining the walls under the end plates
for y in (-P["y_half"] + 0.004, P["y_half"] - 0.004):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, y, 0.006))
    r = bpy.context.active_object
    r.scale = (0.112, 0.008, 0.012)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bv = r.modifiers.new("bv", "BEVEL")
    bv.width, bv.segments = 0.0012, 2
    shade(r, 35)
    parts.append(r)

# wafers: 200 mm discs standing in the slots, with a notch at the bottom
wafers = []
for k in range(P["n_wafers"]):
    y = (k - (P["n_wafers"] - 1) / 2) * P["pitch"]
    bpy.ops.mesh.primitive_cylinder_add(vertices=256, radius=P["wafer_r"], depth=P["wafer_t"],
                                        location=(0, y, Z_WAFER), rotation=(math.pi / 2, 0, 0))
    wf = bpy.context.active_object
    wf.name = f"wafer_{k:02d}"
    shade(wf, 30)
    wafers.append(wf)

for o in parts:                                       # bevels and booleans can leave slivers: dissolve them
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-5, edges=bm.edges)
    bm.to_mesh(o.data)
    bm.free()
finalize(parts + wafers)

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


# white injection-moulded carrier: flow lines, faint yellowing in the slots, grey scuffs on edges and rails
pla, pr, obj, geo, sp = new_mat("cassette_plastic")
bev = node("ShaderNodeBevel", Radius=0.0012)
bev.samples = 16
dp = N.new("ShaderNodeVectorMath")
dp.operation = "DOT_PRODUCT"
L.new(bev.outputs["Normal"], dp.inputs[0])
L.new(geo.outputs["Normal"], dp.inputs[1])
edge = mr(dp.outputs["Value"], 0.995, 0.93)
ao = N.new("ShaderNodeAmbientOcclusion")
ao.samples = 16
ao.inputs["Distance"].default_value = 0.012
cav = mr(ao.outputs["AO"], 0.9, 0.45)
flow = noise(obj, 1.0, 5.0, 0.6, stretch=(30.0, 30.0, 300.0))
blot = noise(obj, 90.0, 4.0, 0.55)
fine = noise(obj, 1200.0, 2.0, 0.5)
low = mr(sp.outputs["Z"], 0.02, 0.0)
col = mix(mr(flow, 0.3, 0.7), (0.80, 0.80, 0.78, 1), (0.74, 0.745, 0.73, 1))
col = mix(mth("MULTIPLY", cav, 0.55), col, (0.62, 0.58, 0.48, 1))
col = mix(mth("MULTIPLY", mth("MAXIMUM", edge, low), mr(blot, 0.4, 0.65, 0.2, 0.8)), col, (0.48, 0.48, 0.47, 1))
L.new(col, pr.inputs["Base Color"])
L.new(mixf(edge, mr(fine, 0.3, 0.7, 0.30, 0.42), 0.55), pr.inputs["Roughness"])
pr.inputs["Metallic"].default_value = 0.0
pr.inputs["Subsurface Weight"].default_value = 0.0
hgt = mth("ADD", mth("MULTIPLY", flow, 0.3), mth("MULTIPLY", fine, 0.2))
bump = node("ShaderNodeBump", Distance=0.0002, Strength=0.3)
L.new(hgt, bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])

# polished silicon with a die grid and thin-film tints on the patterned faces
si, pr, obj, geo, sp = new_mat("wafer_silicon")
sn = N.new("ShaderNodeSeparateXYZ")
L.new(geo.outputs["Normal"], sn.inputs["Vector"])
face = mr(mth("ABSOLUTE", sn.outputs["Y"]), 0.9, 0.99)
rel = N.new("ShaderNodeVectorMath")
rel.operation = "SUBTRACT"
L.new(obj, rel.inputs[0])
rel.inputs[1].default_value = (0.0, 0.0, Z_WAFER)
r2 = N.new("ShaderNodeVectorMath")
r2.operation = "MULTIPLY"
L.new(rel.outputs[0], r2.inputs[0])
r2.inputs[1].default_value = (1.0, 0.0, 1.0)
rl = N.new("ShaderNodeVectorMath")
rl.operation = "LENGTH"
L.new(r2.outputs[0], rl.inputs[0])
inside = mr(rl.outputs["Value"], 0.094, 0.092)                         # edge exclusion ring
gx = mth("ABSOLUTE", mth("SUBTRACT", mth("FRACT", mth("MULTIPLY", sp.outputs["X"], 1 / 0.012, clamp=False), clamp=False), 0.5, clamp=False), clamp=False)
gz = mth("ABSOLUTE", mth("SUBTRACT", mth("FRACT", mth("MULTIPLY", sp.outputs["Z"], 1 / 0.009, clamp=False), clamp=False), 0.5, clamp=False), clamp=False)
scribe = mth("MAXIMUM", mr(gx, 0.44, 0.47), mr(gz, 0.43, 0.46))
die = mth("MULTIPLY", mth("MULTIPLY", face, inside), mth("SUBTRACT", 1.0, scribe))
tint = N.new("ShaderNodeValToRGB")
cr = tint.color_ramp
cr.elements[0].color = (0.22, 0.16, 0.42, 1)
cr.elements[1].color = (0.10, 0.30, 0.38, 1)
e = cr.elements.new(0.5)
e.color = (0.35, 0.22, 0.30, 1)
film = noise(obj, 9.0, 2.0, 0.5)
L.new(film, tint.inputs["Fac"])
cell = node("ShaderNodeTexVoronoi", Scale=1.0)
cell.feature = "F1"
cm = N.new("ShaderNodeMapping")
cm.inputs["Scale"].default_value = (1 / 0.012, 1.0, 1 / 0.009)
L.new(obj, cm.inputs["Vector"])
L.new(cm.outputs["Vector"], cell.inputs["Vector"])
col = mix(die, (0.46, 0.47, 0.50, 1), tint.outputs["Color"])
col = mix(mth("MULTIPLY", die, mr(cell.outputs["Distance"], 0.2, 0.6, 0.0, 0.25)), col, (0.55, 0.52, 0.40, 1))
L.new(col, pr.inputs["Base Color"])
L.new(mixf(die, 0.04, 0.12), pr.inputs["Roughness"])
L.new(mixf(die, 1.0, 0.75), pr.inputs["Metallic"])

for o in parts:
    o.data.materials.append(pla)
for o in wafers:
    o.data.materials.append(si)
