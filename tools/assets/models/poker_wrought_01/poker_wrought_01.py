"""Model script: a wrought-iron fireplace poker lying on the floor, ~0.75 m long along X.

Ring handle at -X, then an upset collar, a twisted square section, a second collar, a round shaft as thick as a
thumb, a forged square spike at +X and a side hook curling back on the +Y side. One centreline sweep with
per-station sections (superellipse, rotated for the twist), a torus ring and a swept hook.
Material: blackened wrought iron with fibrous grain along the bar, hammer facets, mill scale, rust in pits,
hand-polished bright edges on the handle and twist, heat colours and soot toward the tip.
"""

import math

import bmesh
import bpy
from mathutils import Vector

P = {
    "length": 0.75, "shaft_r": 0.0080, "shaft_r_tip": 0.0068, "twist_half": 0.0076, "twist_turns": 2.5,
    "collar_r": 0.0095, "ring_R": 0.0265, "ring_r": 0.0056, "hook_r": 0.0060, "seg": 24,
}
HALF = P["length"] / 2
Z0 = 0.0099                                        # centreline height: the twist corners touch the floor


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


# ------------------------------------------------------------------ bar: ring neck -> collar -> twist -> collar -> shaft -> spike
X_RING_C = -HALF + P["ring_R"] + P["ring_r"]
X0 = X_RING_C + P["ring_R"] - 0.0045                # bar starts inside the ring's tube
TW0, TW1 = -0.285, -0.165                            # twisted square
SP0 = 0.282                                          # spike starts
X1 = HALF


def bar_section(x):
    """(half width, half height, exponent, rotation) at x."""
    s, cr = P["shaft_r"], P["collar_r"]
    if x < X0 + 0.008:                               # neck: the bar end wrapped into the ring
        r = P["ring_r"] * 1.04
        return r, r * 0.94, 2.0, 0.0
    if x < TW0:                                      # first collar (upset), easing into the square
        t = (x - X0 - 0.008) / (TW0 - X0 - 0.008)
        bump = math.exp(-((t - 0.45) / 0.2) ** 2)
        r = lerp(lerp(P["ring_r"] * 1.04, P["twist_half"] * 1.03, smooth(t)), cr, bump)
        return r, r * lerp(0.94, 1.0, t), lerp(2.0, 7.0, smooth((t - 0.55) / 0.45)), 0.0
    if x < TW1:                                      # twisted square bar
        t = (x - TW0) / (TW1 - TW0)
        rot = t * P["twist_turns"] * 2 * math.pi
        h = P["twist_half"] * (1 + 0.04 * math.sin(t * 9.0))      # slight hand-forged irregularity
        return h, h, 7.0, rot
    if x < TW1 + 0.016:                              # second collar
        t = (x - TW1) / 0.016
        bump = math.sin(t * math.pi) ** 0.6
        r = lerp(lerp(P["twist_half"] * 1.05, s, t), cr, bump)
        return r, r, lerp(6.0, 2.0, smooth(t)), P["twist_turns"] * 2 * math.pi
    if x < SP0:                                      # round shaft, slight taper, slightly oval from forging
        t = (x - TW1 - 0.016) / (SP0 - TW1 - 0.016)
        r = lerp(s, P["shaft_r_tip"], t) * (1 + 0.025 * math.sin(t * 23.0) * math.sin(t * 7.0) * (1 - smooth((t - 0.9) / 0.1)))
        return r, r * 0.97, 2.0, P["twist_turns"] * 2 * math.pi
    t = (x - SP0) / (X1 - SP0)                       # forged spike: round -> square taper to a blunt point
    r = P["shaft_r_tip"] * (1 - t) ** 0.85 + 0.0006 * t
    return r, r * lerp(0.97, 1.0, smooth(t * 3)), lerp(2.0, 4.5, smooth(t * 3)), P["twist_turns"] * 2 * math.pi + math.pi / 4 * smooth(t * 3)


xs = []
x = X0
while x < X1:
    xs.append(x)
    step = 0.0016 if (TW0 - 0.02 < x < TW1 + 0.02) else (0.0012 if x < X0 + 0.012 else 0.006)
    if x > SP0 - 0.01:
        step = 0.004
    x += step
xs.append(X1)
path = [(x, 0.0, Z0) for x in xs]
secs = [superellipse2d(*bar_section(x), P["seg"]) for x in xs]
bar = sweep("bar", path, secs)
shade(bar, 50)

# ------------------------------------------------------------------ ring handle, lying flat
bpy.ops.mesh.primitive_torus_add(major_radius=P["ring_R"], minor_radius=P["ring_r"], major_segments=64,
                                 minor_segments=18, location=(X_RING_C, 0, Z0))
ring = bpy.context.active_object
ring.name = "ring"
ring.scale = (1.0, 1.06, 0.92)                      # hand-bent: a touch oval, flattened by the hammer
shade(ring, 60)

# ------------------------------------------------------------------ hook on the +Y side, curling back to -X
def bez(p0, p1, p2, p3, t):
    u = 1 - t
    return tuple(u ** 3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t ** 3 * d for a, b, c, d in zip(p0, p1, p2, p3))


hp = [bez((0.250, 0.000), (0.290, 0.004), (0.312, 0.044), (0.270, 0.061), i / 40) for i in range(41)]
hpath = [(x, y, Z0) for x, y in hp]
hsecs = []
for i in range(41):
    t = i / 40
    r = P["hook_r"] * (1 - 0.82 * t ** 1.2)
    hsecs.append(superellipse2d(r, r * 0.95, lerp(2.0, 3.2, t), 0.0, 20))
hook = sweep("hook", hpath, hsecs)
shade(hook, 50)

objs = [bar, ring, hook]
finalize(objs)

# ------------------------------------------------------------------ material: blackened wrought iron
mat = bpy.data.materials.new("wrought_iron")
mat.use_nodes = True
nt = mat.node_tree
N, L = nt.nodes, nt.links
pr = N["Principled BSDF"]


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


tc = N.new("ShaderNodeTexCoord")
obj = tc.outputs["Object"]
geo = N.new("ShaderNodeNewGeometry")
sx = N.new("ShaderNodeSeparateXYZ")
L.new(obj, sx.inputs["Vector"])
X = sx.outputs["X"]
# edge mask: bevelled normal v true normal (twist corners, collars, ring outer edge)
bev = node("ShaderNodeBevel", Radius=0.0025)
bev.samples = 16
dp = N.new("ShaderNodeVectorMath")
dp.operation = "DOT_PRODUCT"
L.new(bev.outputs["Normal"], dp.inputs[0])
L.new(geo.outputs["Normal"], dp.inputs[1])
edge = mr(dp.outputs["Value"], 0.995, 0.95)
# cavity from AO (between twist flutes, collar shoulders, hook root)
ao = N.new("ShaderNodeAmbientOcclusion")
ao.samples = 16
ao.inputs["Distance"].default_value = 0.006
cav = mr(ao.outputs["AO"], 0.85, 0.45)
# noises: fibrous grain along X (wrought iron slag stringers), mid blotches, big patches
grain = noise(obj, 1.0, 6.0, 0.6, stretch=(25.0, 900.0, 900.0))
blot = noise(obj, 60.0, 5.0, 0.55)
big = noise(obj, 9.0, 3.0, 0.5)
fine = noise(obj, 600.0, 3.0, 0.6)
vor = node("ShaderNodeTexVoronoi", Scale=110.0, Randomness=0.9)   # hammer dimples (smooth F1 cells)
vor.feature = "SMOOTH_F1"
vor.inputs["Smoothness"].default_value = 0.6
L.new(obj, vor.inputs["Vector"])
facet = mr(vor.outputs["Distance"], 0.1, 0.9)
# zones along the bar
handle = mr(X, -0.16, -0.20)                                   # 1 on ring + twist (hand contact)
tipz = mr(X, 0.16, 0.30)                                       # 1 near the working end (fire)
tip_soot = mr(X, 0.30, 0.36)
# polish: edges of the handle zone, broken up; a little on every edge
polish = mth("MULTIPLY", mth("ADD", mth("MULTIPLY", handle, 0.85), 0.25), edge)
polish = mth("MULTIPLY", polish, mr(blot, 0.35, 0.6, 0.4, 1.2))
polish = mth("MAXIMUM", polish, mth("MULTIPLY", handle, mr(big, 0.55, 0.7, 0.0, 0.35)))   # rubbed patches
# rust: in cavities and pits, more on the shaft, not where polished
rust = mth("MULTIPLY", mth("ADD", mth("MULTIPLY", cav, 0.9), mr(blot, 0.62, 0.75, 0.0, 0.7)), mr(big, 0.35, 0.6, 0.3, 1.0))
rust = mth("MULTIPLY", rust, mth("SUBTRACT", 1.0, polish))
# colours
scale_col = mix(grain, (0.012, 0.0115, 0.011, 1), (0.032, 0.029, 0.027, 1))          # black mill scale, streaky
scale_col = mix(mr(big, 0.3, 0.7), scale_col, (0.040, 0.034, 0.030, 1))
temper = N.new("ShaderNodeValToRGB")                                              # heat colours toward the tip
cr = temper.color_ramp
cr.elements[0].position, cr.elements[0].color = 0.0, (0.050, 0.045, 0.042, 1)
cr.elements[1].position, cr.elements[1].color = 0.35, (0.16, 0.085, 0.040, 1)
e = cr.elements.new(0.6)
e.color = (0.10, 0.045, 0.085, 1)
e = cr.elements.new(0.8)
e.color = (0.05, 0.075, 0.16, 1)
e = cr.elements.new(1.0)
e.color = (0.07, 0.075, 0.08, 1)
L.new(mth("ADD", tipz, mth("MULTIPLY", mth("SUBTRACT", big, 0.5), 0.35)), temper.inputs["Fac"])
base = mix(mth("MULTIPLY", tipz, 0.75), scale_col, temper.outputs["Color"])
base = mix(mth("MULTIPLY", tip_soot, mr(blot, 0.3, 0.6, 0.5, 1.0)), base, (0.012, 0.011, 0.010, 1))
rust_col = mix(fine, (0.13, 0.055, 0.022, 1), (0.22, 0.10, 0.04, 1))
base = mix(rust, base, rust_col)
bright = mix(grain, (0.42, 0.41, 0.40, 1), (0.30, 0.295, 0.29, 1))
base = mix(polish, base, bright)
L.new(base, pr.inputs["Base Color"])
# roughness / metal: scale is semi-metallic and rough, rust rough dielectric, polish smooth metal
rough = mixf(polish, mr(fine, 0.3, 0.7, 0.50, 0.74), mr(grain, 0.3, 0.7, 0.20, 0.34))
rough = mixf(rust, rough, 0.9)
L.new(rough, pr.inputs["Roughness"])
metal = mixf(polish, mr(grain, 0.3, 0.7, 0.15, 0.35), 1.0)
metal = mixf(rust, metal, 0.0)
L.new(metal, pr.inputs["Metallic"])
# relief: hammer facets + grain stringers + rust pits
h = mth("ADD", mth("MULTIPLY", facet, 0.5), mth("MULTIPLY", grain, 0.35))
h = mth("SUBTRACT", h, mth("MULTIPLY", mr(blot, 0.66, 0.78), 0.35))
h = mth("ADD", h, mth("MULTIPLY", fine, 0.15))
bump = node("ShaderNodeBump", Distance=0.0004, Strength=0.55)
L.new(h, bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])

for o in objs:
    o.data.materials.append(mat)
