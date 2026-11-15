"""Model script: a cleanroom process-tool bay, ~1.2 m wide (X), 1.0 m deep (Y), 2.0 m tall plus a signal tower.

A dark frame clad in white powder-coated panels with 4 mm seams and button-head screws; front (-Y): two lower doors
with pulls and hinges, a louvered intake, a glass load-port window in a gasket over a stainless load-port stage
(kinematic pins, presence sensor, bracket), a status screen in a bezel with indicator lamps, a red EMO button, a top
band with a louvered vent and a label plate. Sides: six panels each and a rear louvered vent; back panels; roof with a
fan filter unit and a three-lamp signal tower; four levelling feet (pad, threaded stem, hex nut, plate).
The screen and green lamp are emissive in Blender; the chain bakes base colour only (see the report).
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


W, D, H = 1.20, 0.80, 2.00                           # cabinet body (the load-port stage adds ~0.2 m in front)
XH, YH = W / 2 - 0.010, D / 2 - 0.010                 # core box half sizes (panels sit 10 mm proud of it)
Z_CORE0, Z_PAN0, ZTOP = 0.055, 0.125, H - 0.010
GAP, PT = 0.004, 0.010                               # seam gap, panel thickness
groups = {k: [] for k in ("panel", "core", "glass", "steel", "screen", "emo_red", "emo_yel", "lamp_r", "lamp_a",
                          "lamp_g", "rubber", "grille")}


def box(group, lo, hi, bevel=0.0, seg=2, name="box"):
    lo, hi = Vector(lo), Vector(hi)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(lo + hi) / 2)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = hi - lo
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        m = ob.modifiers.new("bv", "BEVEL")
        m.width, m.segments = bevel, seg
    shade(ob, 35)
    groups[group].append(ob)
    return ob


def cyl(group, loc, r, depth, axis="Y", verts=24, bevel=0.0):
    rot = {"X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0), "Z": (0, 0, 0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=depth, location=loc, rotation=rot)
    ob = bpy.context.active_object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if bevel:
        m = ob.modifiers.new("bv", "BEVEL")
        m.width, m.segments, m.limit_method = bevel, 2, "ANGLE"
    shade(ob, 40)
    groups[group].append(ob)
    return ob


def screws(face, u0, u1, v0, v1, inset=0.022):
    """Four button-head screws near the corners of a panel; face = (axis, coordinate of the panel's outer plane)."""
    ax, c = face
    for u in (u0 + inset, u1 - inset):
        for v in (v0 + inset, v1 - inset):
            if ax == "-Y":
                cyl("steel", (u, c - 0.0008, v), 0.0045, 0.0028, "Y", 12, 0.0008)
            elif ax == "+X":
                cyl("steel", (c + 0.0008, u, v), 0.0045, 0.0028, "X", 12, 0.0008)
            elif ax == "-X":
                cyl("steel", (c - 0.0008, u, v), 0.0045, 0.0028, "X", 12, 0.0008)
            elif ax == "+Y":
                cyl("steel", (u, c + 0.0008, v), 0.0045, 0.0028, "Y", 12, 0.0008)


def front_panel(x0, x1, z0, z1, scr=True):
    yf = -YH - 0.010
    box("panel", (x0 + GAP / 2, yf - PT, z0 + GAP / 2), (x1 - GAP / 2, yf + 0.001, z1 - GAP / 2), 0.0025)
    if scr:
        screws(("-Y", yf - PT), x0, x1, z0, z1)


def side_panel(sgn, y0, y1, z0, z1, scr=True):
    xs = sgn * (XH + 0.010)
    a, b = sorted((xs - 0.001 * sgn, xs + PT * sgn))
    box("panel", (a, y0 + GAP / 2, z0 + GAP / 2), (b, y1 - GAP / 2, z1 - GAP / 2), 0.0025)
    if scr:
        screws(("+X" if sgn > 0 else "-X", xs + PT * sgn), y0, y1, z0, z1)


def back_panel(x0, x1, z0, z1):
    yb = YH + 0.010
    box("panel", (x0 + GAP / 2, yb - 0.001, z0 + GAP / 2), (x1 - GAP / 2, yb + PT, z1 - GAP / 2), 0.0025)
    screws(("+Y", yb + PT), x0, x1, z0, z1)


def louvers(group_frame, plane, u0, u1, v0, v1, n):
    """A recessed vent: dark frame pocket plus n tilted slats. plane: ('-Y', y) or ('+X'/'-X', x)."""
    ax, c = plane
    out = -1 if ax in ("-Y", "-X") else 1
    pitch = (v1 - v0) / n
    for k in range(n):
        vc = v0 + pitch * (k + 0.5)
        if ax == "-Y":
            s = box("panel", (u0, c - 0.006, vc - pitch * 0.32), (u1, c + 0.002, vc + pitch * 0.32), 0.0008, 1, "slat")
            s.rotation_euler.x = math.radians(-35)
        else:
            a, b = sorted((c + 0.002 * out, c - 0.006 * out))
            s = box("panel", (a, u0, vc - pitch * 0.32), (b, u1, vc + pitch * 0.32), 0.0008, 1, "slat")
            s.rotation_euler.y = math.radians(35 * out)
    if ax == "-Y":
        box("core", (u0 - 0.006, c - 0.001, v0 - 0.006), (u1 + 0.006, c + 0.012, v1 + 0.006), name="vent_pocket")
    else:
        a, b = sorted((c + 0.001 * out, c - 0.012 * out))
        box("core", (a, u0 - 0.006, v0 - 0.006), (b, u1 + 0.006, v1 + 0.006), name="vent_pocket")


# ------------------------------------------------------------------ core frame (dark, shows in seams and the plinth)
core = box("core", (-XH, -YH, Z_CORE0), (XH, YH, ZTOP), 0.004, 2, "core")

# ------------------------------------------------------------------ levelling feet
for sx in (-1, 1):
    for sy in (-1, 1):
        x, y = sx * (XH - 0.07), sy * (YH - 0.07)
        cyl("rubber", (x, y, 0.006), 0.034, 0.012, "Z", 32, 0.002)
        cyl("steel", (x, y, 0.032), 0.0095, 0.046, "Z", 16)
        cyl("steel", (x, y, 0.024), 0.019, 0.011, "Z", 6, 0.0012)        # hex nut
        cyl("steel", (x, y, Z_CORE0 - 0.002), 0.03, 0.006, "Z", 24, 0.001)  # mounting plate

# ------------------------------------------------------------------ front: doors, load-port band, top vent band
ZD1, ZB1 = 0.840, 1.400
front_panel(-XH, 0.0, Z_PAN0, ZD1, False)
front_panel(0.0, XH, Z_PAN0, ZD1, False)
yf = -YH - 0.010 - PT
for x in (-0.10, 0.10):                              # recessed door pulls (dark pocket + steel lip)
    box("core", (x - 0.018, yf - 0.002, 0.46), (x + 0.018, yf + 0.004, 0.58), 0.0015, 2, "pull_pocket")
    box("steel", (x - 0.016, yf - 0.006, 0.555), (x + 0.016, yf + 0.001, 0.566), 0.0015, 2, "pull_lip")
for x in (-XH + 0.03, XH - 0.03):                    # hinges
    for z in (0.22, 0.74):
        cyl("steel", (x, yf - 0.006, z), 0.007, 0.06, "Z", 12, 0.001)
louvers("panel", ("-Y", yf), -0.45, -0.15, 0.17, 0.32, 7)
front_panel(-XH, -0.30, ZD1, ZB1)
front_panel(0.30, XH, ZD1, ZB1)
# window frame around the load-port glass
front_panel(-0.30, 0.30, ZD1, 0.93, False)
front_panel(-0.30, 0.30, 1.32, ZB1, False)
box("rubber", (-0.262, yf - 0.002, 0.926), (0.262, yf + PT + 0.002, 1.324), 0.003, 2, "gasket")
box("glass", (-0.250, yf + 0.003, 0.935), (0.250, yf + 0.008, 1.315), 0.001, 1, "glass")
box("core", (-0.30 + GAP / 2, yf + 0.008, 0.93), (0.30 - GAP / 2, yf + PT + 0.02, 1.32), name="port_back")
# load-port stage: a stainless shelf on a bracket, with three kinematic pins and a presence sensor
box("steel", (-0.23, yf - 0.20, 0.866), (0.23, yf + 0.001, 0.884), 0.003, 2, "stage")
box("panel", (-0.21, yf - 0.19, 0.80), (0.21, yf + 0.001, 0.866), 0.004, 2, "stage_body")
box("core", (-0.035, yf - 0.12, 0.66), (0.035, yf + 0.001, 0.80), 0.003, 2, "stage_bracket")
for (px, py) in ((-0.10, -0.06), (0.10, -0.06), (0.0, -0.16)):
    cyl("steel", (px, yf + py, 0.889), 0.006, 0.010, "Z", 16, 0.002)
box("rubber", (0.15, yf - 0.17, 0.884), (0.19, yf - 0.15, 0.89), 0.001, 1, "sensor")
# status screen in a bezel, EMO button below it
box("rubber", (0.345, yf - 0.012, 1.165), (0.555, yf + 0.001, 1.315), 0.004, 2, "bezel")
box("screen", (0.355, yf - 0.0128, 1.175), (0.545, yf - 0.011, 1.305), 0.0, 1, "screen")
cyl("emo_yel", (0.45, yf - 0.002, 1.02), 0.045, 0.004, "Y", 32, 0.001)
cyl("emo_red", (0.45, yf - 0.018, 1.02), 0.028, 0.026, "Y", 32, 0.006)
for i in range(3):                                   # small indicator lamps beside the screen
    cyl(("lamp_g", "lamp_a", "lamp_r")[i], (0.37 + i * 0.03, yf - 0.004, 1.12), 0.007, 0.008, "Y", 16, 0.002)
# top band: one wide panel with a louvered intake and a label plate
front_panel(-XH, XH, ZB1, ZTOP)
louvers("panel", ("-Y", yf), -0.45, 0.05, 1.56, 1.80, 9)
box("rubber", (0.22, yf - 0.0015, 1.70), (0.42, yf + 0.001, 1.76), 0.001, 1, "label")

# ------------------------------------------------------------------ sides: 2 columns x 3 rows, vent low at the rear
for sgn in (-1, 1):
    for (y0, y1) in ((-YH - 0.01, 0.0), (0.0, YH + 0.01)):
        for (z0, z1) in ((Z_PAN0, 0.84), (0.84, 1.40), (1.40, ZTOP)):
            side_panel(sgn, y0, y1, z0, z1)
    xs = sgn * (XH + 0.010 + PT)
    louvers("panel", ("+X" if sgn > 0 else "-X", xs), 0.08, 0.34, 0.20, 0.42, 8)

# ------------------------------------------------------------------ back and top
for (x0, x1) in ((-XH, 0.0), (0.0, XH)):
    back_panel(x0, x1, Z_PAN0, 1.05)
    back_panel(x0, x1, 1.05, ZTOP)
box("panel", (-XH - 0.01, -YH - 0.02, ZTOP - 0.001), (XH + 0.01, YH + 0.01, ZTOP + 0.008), 0.003, 2, "top")
box("core", (-0.42, -0.30, ZTOP + 0.008), (0.30, 0.25, ZTOP + 0.030), 0.004, 2, "ffu")          # fan filter unit
box("grille", (-0.40, -0.28, ZTOP + 0.030), (0.28, 0.23, ZTOP + 0.032), 0.0, 1, "ffu_grille")
# signal tower at the rear right of the roof
TX, TY = 0.46, 0.26
cyl("steel", (TX, TY, ZTOP + 0.014), 0.03, 0.012, "Z", 24, 0.002)
cyl("steel", (TX, TY, ZTOP + 0.060), 0.008, 0.085, "Z", 16)
for i, g in enumerate(("lamp_g", "lamp_a", "lamp_r")):
    cyl(g, (TX, TY, ZTOP + 0.122 + i * 0.044), 0.030, 0.040, "Z", 32, 0.002)
    cyl("rubber", (TX, TY, ZTOP + 0.143 + i * 0.044), 0.031, 0.004, "Z", 32)
cyl("rubber", (TX, TY, ZTOP + 0.232), 0.031, 0.012, "Z", 32, 0.003)

finalize([o for g in groups.values() for o in g])      # also applies the slat rotations

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


def edge_mask(geo, radius, lo=0.995, hi=0.93):
    bev = node("ShaderNodeBevel", Radius=radius)
    bev.samples = 16
    dp = N.new("ShaderNodeVectorMath")
    dp.operation = "DOT_PRODUCT"
    L.new(bev.outputs["Normal"], dp.inputs[0])
    L.new(geo.outputs["Normal"], dp.inputs[1])
    return mr(dp.outputs["Value"], lo, hi)


def cavity(dist, lo=0.92, hi=0.5):
    ao = N.new("ShaderNodeAmbientOcclusion")
    ao.samples = 16
    ao.inputs["Distance"].default_value = dist
    return mr(ao.outputs["AO"], lo, hi)


mats = {}
# white powder coat: orange peel, faint panel-to-panel tint shifts, grime low and in the seams, scuffs at kick height
m, pr, obj, geo, sp = new_mat("bay_powdercoat")
peel = noise(obj, 900.0, 2.0, 0.5)
blot = noise(obj, 30.0, 4.0, 0.55)
big = noise(obj, 3.0, 2.0, 0.5)
edge = edge_mask(geo, 0.004)
cav = cavity(0.03)
low = mr(sp.outputs["Z"], 0.40, 0.10)
scuffz = mth("MULTIPLY", mr(sp.outputs["Z"], 0.32, 0.16), mr(noise(obj, 1.0, 3.0, 0.6, stretch=(140.0, 140.0, 20.0)), 0.6, 0.72))
col = mix(mr(big, 0.3, 0.7), (0.80, 0.81, 0.80, 1), (0.76, 0.765, 0.75, 1))
col = mix(mth("MULTIPLY", cav, 0.7), col, (0.42, 0.42, 0.40, 1))
col = mix(mth("MULTIPLY", low, mr(blot, 0.4, 0.7, 0.1, 0.5)), col, (0.55, 0.53, 0.49, 1))
col = mix(mth("MULTIPLY", edge, mr(blot, 0.45, 0.65, 0.0, 0.5)), col, (0.60, 0.60, 0.58, 1))
col = mix(mth("MULTIPLY", scuffz, 0.7), col, (0.35, 0.35, 0.36, 1))
L.new(col, pr.inputs["Base Color"])
L.new(mr(peel, 0.3, 0.7, 0.34, 0.46), pr.inputs["Roughness"])
bump = node("ShaderNodeBump", Distance=0.0002, Strength=0.25)
L.new(peel, bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])
mats["panel"] = m

m, pr, obj, geo, sp = new_mat("bay_frame")                 # dark grey anodised frame
fine = noise(obj, 400.0, 3.0, 0.5)
L.new(mix(fine, (0.045, 0.047, 0.05, 1), (0.065, 0.067, 0.07, 1)), pr.inputs["Base Color"])
L.new(mr(fine, 0.3, 0.7, 0.45, 0.6), pr.inputs["Roughness"])
pr.inputs["Metallic"].default_value = 0.3
mats["core"] = m

m, pr, obj, geo, sp = new_mat("bay_glass")                 # tinted glass (opaque bake: dark and glossy)
smudge = noise(obj, 25.0, 4.0, 0.6)
L.new(mix(mr(smudge, 0.5, 0.7, 0.0, 0.3), (0.012, 0.016, 0.018, 1), (0.05, 0.055, 0.055, 1)), pr.inputs["Base Color"])
L.new(mr(smudge, 0.45, 0.75, 0.03, 0.18), pr.inputs["Roughness"])
mats["glass"] = m

m, pr, obj, geo, sp = new_mat("bay_steel")                 # brushed stainless
br = noise(obj, 1.0, 6.0, 0.6, stretch=(2.0, 2.0, 600.0))
br2 = noise(obj, 1.0, 6.0, 0.6, stretch=(600.0, 2.0, 2.0))
sn = N.new("ShaderNodeSeparateXYZ")
L.new(geo.outputs["Normal"], sn.inputs["Vector"])
up = mr(mth("ABSOLUTE", sn.outputs["Z"]), 0.5, 0.9)
brush = mixf(up, br, br2)
L.new(mix(mr(brush, 0.3, 0.7), (0.52, 0.53, 0.54, 1), (0.66, 0.67, 0.68, 1)), pr.inputs["Base Color"])
L.new(mr(brush, 0.3, 0.7, 0.22, 0.36), pr.inputs["Roughness"])
pr.inputs["Metallic"].default_value = 1.0
mats["steel"] = m

m, pr, obj, geo, sp = new_mat("bay_grille")                # perforated filter face: hole grid in brushed steel
hx = mth("ABSOLUTE", mth("SUBTRACT", mth("FRACT", mth("MULTIPLY", sp.outputs["X"], 1 / 0.008, clamp=False), clamp=False), 0.5, clamp=False), clamp=False)
hy = mth("ABSOLUTE", mth("SUBTRACT", mth("FRACT", mth("MULTIPLY", sp.outputs["Y"], 1 / 0.008, clamp=False), clamp=False), 0.5, clamp=False), clamp=False)
hole = mr(mth("ADD", mth("MULTIPLY", hx, hx, clamp=False), mth("MULTIPLY", hy, hy, clamp=False), clamp=False), 0.06, 0.045)
L.new(mix(hole, (0.55, 0.56, 0.57, 1), (0.02, 0.02, 0.022, 1)), pr.inputs["Base Color"])
L.new(mixf(hole, 0.35, 0.9), pr.inputs["Roughness"])
L.new(mixf(hole, 1.0, 0.0), pr.inputs["Metallic"])
bump = node("ShaderNodeBump", Distance=0.001, Strength=0.6)
L.new(mth("SUBTRACT", 1.0, hole), bump.inputs["Height"])
L.new(bump.outputs["Normal"], pr.inputs["Normal"])
mats["grille"] = m

m, pr, obj, geo, sp = new_mat("bay_screen")                # status UI: bars and a header strip; also emissive
u = mr(sp.outputs["X"], 0.355, 0.545)
v = mr(sp.outputs["Z"], 1.175, 1.305)
header = mr(v, 0.80, 0.82)
rows = mth("FRACT", mth("MULTIPLY", v, 6.0, clamp=False), clamp=False)
bars = mth("MULTIPLY", mr(rows, 0.15, 0.2), mr(rows, 0.65, 0.6))
barlen = mth("LESS_THAN", u, mr(noise(obj, 40.0, 0.0, 0.0), 0.3, 0.7, 0.3, 0.95), clamp=False)
bars = mth("MULTIPLY", mth("MULTIPLY", bars, barlen), mth("SUBTRACT", 1.0, header))
ui = mix(header, (0.02, 0.06, 0.10, 1), (0.10, 0.45, 0.65, 1))
ui = mix(bars, ui, (0.25, 0.85, 0.45, 1))
L.new(ui, pr.inputs["Base Color"])
L.new(ui, pr.inputs["Emission Color"])
pr.inputs["Emission Strength"].default_value = 3.0
pr.inputs["Roughness"].default_value = 0.08
mats["screen"] = m

for key, c, rough, emit in (("emo_red", (0.55, 0.02, 0.015, 1), 0.3, 0), ("emo_yel", (0.85, 0.62, 0.02, 1), 0.4, 0),
                            ("lamp_r", (0.7, 0.04, 0.03, 1), 0.15, 0), ("lamp_a", (0.85, 0.42, 0.02, 1), 0.15, 0),
                            ("lamp_g", (0.10, 0.75, 0.15, 1), 0.15, 2.0), ("rubber", (0.025, 0.025, 0.027, 1), 0.7, 0)):
    m, pr, obj, geo, sp = new_mat(f"bay_{key}")
    fine = noise(obj, 300.0, 2.0, 0.5)
    L.new(mix(mr(fine, 0.3, 0.7, 0.0, 0.25), c, tuple(x * 0.7 for x in c[:3]) + (1,)), pr.inputs["Base Color"])
    pr.inputs["Roughness"].default_value = rough
    if emit:
        pr.inputs["Emission Color"].default_value = c
        pr.inputs["Emission Strength"].default_value = emit
    mats[key] = m

for key, obs in groups.items():
    for o in obs:
        o.data.materials.append(mats[key])
