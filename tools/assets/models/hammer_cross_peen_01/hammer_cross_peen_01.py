"""Model script: a blacksmith's cross-peen hammer lying on its side, ~0.40 m long along X.

Head at -X with its axis along Y: flat ground face to -Y (front), cross peen to +Y (edge vertical, across the
handle). The handle runs to +X: shoulder at the eye, thin flexing neck, swelling grip, rounded end; a steel wedge
across the top. Head: lofted superellipse sections (round face, chamfered octagonal body, tapered peen).
Materials: forge-black steel with a ground face (circular grinding marks), bright peen, worn edges, rust bloom;
ash handle with wandering grain, pores, a dark polished grip and overstrike scuffs below the head.
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


# ------------------------------------------------------------------ head: lofted along Y (face -Y, cross peen +Y)
XH = -0.184                                          # head centre X; the handle runs to +X
ZC = 0.0165                                          # head and handle centre height (head side on the floor)


def head_section(y):
    """(half X, half Z, exponent) of the head at y."""
    keys = [  # y, half x (handle direction), half z (width), exponent
        (-0.0625, 0.0146, 0.0146, 2.0), (-0.0618, 0.0160, 0.0160, 2.0), (-0.0600, 0.0166, 0.0166, 2.1),
        (-0.0500, 0.0160, 0.0160, 2.6), (-0.0380, 0.0150, 0.0152, 4.5), (-0.0150, 0.0150, 0.0166, 6.0),
        (0.0150, 0.0150, 0.0166, 6.0), (0.0270, 0.0146, 0.0156, 5.5), (0.0450, 0.0100, 0.0157, 4.5),
        (0.0560, 0.0060, 0.0158, 3.5), (0.0610, 0.0034, 0.0156, 2.6), (0.0625, 0.0016, 0.0150, 2.2)]
    for (y0, a0, b0, n0), (y1, a1, b1, n1) in zip(keys, keys[1:]):
        if y0 <= y <= y1:
            t = (y - y0) / (y1 - y0)
            return lerp(a0, a1, t), lerp(b0, b1, t), lerp(n0, n1, t)
    return keys[-1][1:]


ys = sorted(set(round(y, 5) for y in [-0.0625, -0.0622, -0.0618, -0.0612, -0.0606, -0.0600]
                + [-0.059 + 0.003 * i for i in range(40)] + [0.0610, 0.0618, 0.0625]))
ys = [y for y in ys if y <= 0.0625]
path = [(XH, y, ZC) for y in ys]
secs = []
for y in ys:
    hx, hz, n = head_section(y)
    # sweep frame along +Y: N = +Z (up), B = T x N = +X  -> section (u, v) = (x offset, z offset)
    secs.append(superellipse2d(hx, hz, n, 0.0, 48))
head = sweep("head", path, secs)
shade(head, 35)

# ------------------------------------------------------------------ handle: lofted along X through the eye
X_TOP, X_END = XH - 0.0170, XH + 0.3850


def handle_section(x):
    """(half Y = swing-plane width, half Z, exponent)."""
    d = x - XH
    keys = [(-0.0170, 0.0108, 0.0076, 2.4), (0.0150, 0.0118, 0.0084, 2.4), (0.0280, 0.0135, 0.0104, 2.3),
            (0.0600, 0.0126, 0.0098, 2.3), (0.1100, 0.0120, 0.0094, 2.3), (0.2200, 0.0148, 0.0114, 2.4),
            (0.3300, 0.0172, 0.0128, 2.5), (0.3700, 0.0178, 0.0131, 2.5), (0.3790, 0.0168, 0.0124, 2.4),
            (0.3835, 0.0130, 0.0098, 2.2), (0.3850, 0.0080, 0.0060, 2.0)]
    for (x0, a0, b0, n0), (x1, a1, b1, n1) in zip(keys, keys[1:]):
        if x0 <= d <= x1:
            t = smooth((d - x0) / (x1 - x0))
            return lerp(a0, a1, t), lerp(b0, b1, t), lerp(n0, n1, t)
    return keys[-1][1:]


xs = [X_TOP + i * 0.004 for i in range(int((X_END - X_TOP - 0.008) / 0.004))]
xs += [X_END - 0.007, X_END - 0.005, X_END - 0.0035, X_END - 0.002, X_END - 0.001, X_END]
hpath = [(x, 0.0, ZC) for x in xs]
# frame along +X: N = +Z, B = X x Z = -Y -> section (u, v) = (-y, z)
hsecs = [superellipse2d(*handle_section(x)[:2], handle_section(x)[2], 0.0, 32) for x in xs]
handle = sweep("handle", hpath, hsecs)
shade(handle, 50)

# steel wedge across the top of the handle (seen on the -X end), and a wooden wedge line beside it
bpy.ops.mesh.primitive_cube_add(size=1, location=(X_TOP - 0.0004, 0.0, ZC))
wedge = bpy.context.active_object
wedge.name = "wedge"
wedge.scale = (0.0016, 0.0190, 0.0022)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
bv = wedge.modifiers.new("bv", "BEVEL")
bv.width, bv.segments = 0.0004, 2
shade(wedge, 30)

finalize_shift = finalize([head, handle, wedge])

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
steel = bpy.data.materials.new("hammer_steel")
steel.use_nodes = True
N, L = steel.node_tree.nodes, steel.node_tree.links
pr = N["Principled BSDF"]
tc = N.new("ShaderNodeTexCoord")
obj = tc.outputs["Object"]
geo = N.new("ShaderNodeNewGeometry")
sp = N.new("ShaderNodeSeparateXYZ")
L.new(obj, sp.inputs["Vector"])
sn = N.new("ShaderNodeSeparateXYZ")
L.new(geo.outputs["Normal"], sn.inputs["Vector"])
bev = node("ShaderNodeBevel", Radius=0.0022)
bev.samples = 16
dp = N.new("ShaderNodeVectorMath")
dp.operation = "DOT_PRODUCT"
L.new(bev.outputs["Normal"], dp.inputs[0])
L.new(geo.outputs["Normal"], dp.inputs[1])
edge = mr(dp.outputs["Value"], 0.995, 0.94)
blot = noise(obj, 70.0, 5.0, 0.55)
big = noise(obj, 14.0, 3.0, 0.5)
fine = noise(obj, 700.0, 3.0, 0.6)
# ground face: facing -Y, at the face end; circular grinding marks around the face centre
face = mth("MULTIPLY", mr(sn.outputs["Y"], -0.80, -0.97), mr(sp.outputs["Y"], -0.057, -0.061))
rel = N.new("ShaderNodeVectorMath")
rel.operation = "SUBTRACT"
L.new(obj, rel.inputs[0])
rel.inputs[1].default_value = (XH, 0.0, ZC)
ln = N.new("ShaderNodeVectorMath")
ln.operation = "MULTIPLY"
L.new(rel.outputs[0], ln.inputs[0])
ln.inputs[1].default_value = (1.0, 0.0, 1.0)
rad = N.new("ShaderNodeVectorMath")
rad.operation = "LENGTH"
L.new(ln.outputs[0], rad.inputs[0])
wave = node("ShaderNodeTexWave", Scale=1.0, Distortion=3.0, Detail=6.0)
wave.wave_type, wave.bands_direction = "RINGS", "X"
mvec = N.new("ShaderNodeCombineXYZ")
L.new(mth("MULTIPLY", rad.outputs["Value"], 900.0, clamp=False), mvec.inputs["X"])
L.new(mth("MULTIPLY", blot, 0.4, clamp=False), mvec.inputs["Y"])
L.new(mvec.outputs[0], wave.inputs["Vector"])
rings = mr(wave.outputs["Fac"], 0.2, 0.8)
# peen edge: bright where it strikes
peen = mr(sp.outputs["Y"], 0.055, 0.0615)
# worn edges, broken up
wear = mth("MULTIPLY", edge, mr(blot, 0.35, 0.6, 0.3, 1.2))
bright = mth("MAXIMUM", mth("MAXIMUM", face, mth("MULTIPLY", peen, mr(big, 0.3, 0.6, 0.5, 1.0))), wear)
# rust bloom in pits on the cheeks, and a darker tarnish ring around the face
rust = mth("MULTIPLY", mr(blot, 0.63, 0.74), mr(big, 0.4, 0.65, 0.2, 1.0))
rust = mth("MULTIPLY", rust, mth("SUBTRACT", 1.0, bright))
black = mix(mr(big, 0.3, 0.7), (0.020, 0.020, 0.022, 1), (0.050, 0.047, 0.045, 1))
black = mix(mth("MULTIPLY", fine, 0.5), black, (0.07, 0.065, 0.06, 1))
ground = mix(rings, (0.50, 0.50, 0.50, 1), (0.36, 0.36, 0.37, 1))
col = mix(bright, black, ground)
col = mix(rust, col, mix(fine, (0.15, 0.06, 0.025, 1), (0.24, 0.11, 0.045, 1)))
L.new(col, pr.inputs["Base Color"])
rough = mixf(bright, mr(fine, 0.3, 0.7, 0.42, 0.62), mr(rings, 0.0, 1.0, 0.16, 0.30))
rough = mixf(rust, rough, 0.88)
L.new(rough, pr.inputs["Roughness"])
metal = mixf(bright, 0.75, 1.0)
metal = mixf(rust, metal, 0.0)
L.new(metal, pr.inputs["Metallic"])
vor = node("ShaderNodeTexVoronoi", Scale=160.0, Randomness=0.9)       # forging dimples on the cheeks
vor.feature = "SMOOTH_F1"
L.new(obj, vor.inputs["Vector"])
hgt = mth("ADD", mth("MULTIPLY", mr(vor.outputs["Distance"], 0.1, 0.9), mth("SUBTRACT", 1.0, bright)), mth("MULTIPLY", rings, mth("MULTIPLY", face, 0.3)))
hgt = mth("SUBTRACT", hgt, mth("MULTIPLY", rust, 0.4))
bump = node("ShaderNodeBump", Distance=0.0003, Strength=0.5)
L.new(hgt, bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])

wood = bpy.data.materials.new("hammer_ash")
wood.use_nodes = True
N, L = wood.node_tree.nodes, wood.node_tree.links
pr = N["Principled BSDF"]
tc = N.new("ShaderNodeTexCoord")
obj = tc.outputs["Object"]
sp = N.new("ShaderNodeSeparateXYZ")
L.new(obj, sp.inputs["Vector"])
X = sp.outputs["X"]
# grain: ring bands across Y/Z, long along X; distorted so the figure wanders
gmap = N.new("ShaderNodeMapping")
gmap.inputs["Location"].default_value = (0.0, 0.045, -0.02)     # pith outside the handle: rings -> long stripes
gmap.inputs["Rotation"].default_value = (0.0, 0.035, 0.03)        # slight run-out: the stripes drift and arch
L.new(obj, gmap.inputs["Vector"])
gwave = node("ShaderNodeTexWave", Scale=95.0, Distortion=1.2, Detail=3.0, **{"Detail Scale": 2.0})
gwave.wave_type, gwave.bands_direction = "RINGS", "Y"
gwave.rings_direction = "X"
L.new(gmap.outputs["Vector"], gwave.inputs["Vector"])
gr = mr(gwave.outputs["Fac"], 0.25, 0.85)
pores = noise(obj, 1.0, 4.0, 0.6, stretch=(30.0, 2500.0, 2500.0))
blot = noise(obj, 40.0, 4.0, 0.55)
big = noise(obj, 8.0, 3.0, 0.5)
fine = noise(obj, 900.0, 2.0, 0.5)
grip = mr(X, XH + 0.20, XH + 0.29)                        # hand zone: darker, greasy, polished
near_head = mr(X, XH + 0.07, XH + 0.02)                   # overstrike scuffs, soot below the head
light = (0.30, 0.19, 0.095, 1)
late = (0.165, 0.092, 0.045, 1)
col = mix(gr, light, late)
col = mix(mth("MULTIPLY", pores, 0.6), col, (0.20, 0.12, 0.06, 1))
col = mix(mr(big, 0.3, 0.7, 0.0, 0.3), col, (0.30, 0.25, 0.19, 1))                  # weathered grey patches
col = mix(mr(blot, 0.5, 0.75, 0.0, 0.6), col, (0.09, 0.065, 0.045, 1))             # grime
col = mix(mth("MULTIPLY", mr(fine, 0.62, 0.75), 0.7), col, (0.05, 0.04, 0.03, 1))     # dirt specks
col = mix(mth("MULTIPLY", grip, mr(blot, 0.3, 0.7, 0.6, 0.9)), col, (0.10, 0.055, 0.03, 1))
col = mix(mth("MULTIPLY", near_head, mr(blot, 0.45, 0.7, 0.2, 0.8)), col, (0.08, 0.06, 0.05, 1))
L.new(col, pr.inputs["Base Color"])
rough = mixf(grip, mr(fine, 0.3, 0.7, 0.72, 0.88), mr(blot, 0.3, 0.7, 0.34, 0.5))
L.new(rough, pr.inputs["Roughness"])
pr.inputs["Metallic"].default_value = 0.0
dents = node("ShaderNodeTexVoronoi", Scale=90.0, Randomness=1.0)
dents.feature = "SMOOTH_F1"
L.new(obj, dents.inputs["Vector"])
dh = mth("MULTIPLY", mr(dents.outputs["Distance"], 0.0, 0.25, 1.0, 0.0), mth("MULTIPLY", near_head, mr(blot, 0.5, 0.65)))
hgt = mth("ADD", mth("MULTIPLY", gr, 0.25), mth("MULTIPLY", pores, 0.5))
hgt = mth("SUBTRACT", hgt, mth("MULTIPLY", dh, 0.8))
bump = node("ShaderNodeBump", Distance=0.0003, Strength=0.45)
L.new(hgt, bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])

head.data.materials.append(steel)
wedge.data.materials.append(steel)
handle.data.materials.append(wood)
