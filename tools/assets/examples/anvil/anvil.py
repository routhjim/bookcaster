"""Model script: a blacksmith's anvil (London pattern), ~0.6 m long (X), 0.2 m wide (Y), 0.3 m tall (Z).

Front faces -Y, horn points +X, heel -X. Body and horn are one loft of superellipse sections along X (boxy at the
face, round at the horn); waist and base are one loft along Z; feet are cut by two arches. Material: dark forged
steel, rust in crevices, worn bright upper edges, a polished striking face with fine lengthwise scratches.
"""

import math

import bmesh
import bpy

SEG = 48


def superellipse(cx, cy, cz, w, h, n, seg=SEG):
    """Points of a superellipse section in a plane: centre, width (y), height (z), exponent n (2 round, big boxy)."""
    pts = []
    for k in range(seg):
        a = 2 * math.pi * (k + 0.5) / seg
        c, s = math.cos(a), math.sin(a)
        y = w / 2 * math.copysign(abs(c) ** (2 / n), c)
        z = h / 2 * math.copysign(abs(s) ** (2 / n), s)
        pts.append((cx + y, cy + z, cz))
    return pts


def loft(name, rings, axis):
    """rings: list of point lists; axis 'x' maps (u, v, t) -> (t, u, v); 'z' maps (u, v, t) -> (u, v, t)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = []
    for r in rings:
        row = []
        for (u, v, t) in r:
            row.append(bm.verts.new((t, u, v) if axis == "x" else (u, v, t)))
        vs.append(row)
    n = len(rings[0])
    for a, b in zip(vs, vs[1:]):
        for i in range(n):
            bm.faces.new((a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]))
    bm.faces.new(list(reversed(vs[0])))
    bm.faces.new(vs[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def cut(ob, cutter):
    m = ob.modifiers.new(cutter.name, "BOOLEAN")
    m.operation, m.object, m.solver = "DIFFERENCE", cutter, "EXACT"


def finish(ob, angle=30):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))


# ------------------------------------------------------------------ body + step + horn, lofted along X
FACE_TOP, STEP_TOP, BODY_BOT, W = 0.300, 0.284, 0.200, 0.122


def section(x):
    """(width, top z, bottom z, exponent) of the body/horn at x."""
    if x <= 0.105:                                   # face and heel
        top = FACE_TOP
        heel = smooth((-0.12 - x) / 0.18)            # heel underside rises toward its end
        bot = lerp(BODY_BOT, 0.262, heel ** 0.8)
        w = W * (1 - 0.06 * heel)
        return w, top, bot, 10
    if x <= 0.165:                                   # step (table), slightly lower than the face
        t = smooth((x - 0.105) / 0.004)
        top = lerp(FACE_TOP, STEP_TOP, t)
        u = (x - 0.105) / 0.06
        return W * lerp(1.0, 0.92, u), top, lerp(BODY_BOT, 0.205, u), 10
    t = (x - 0.165) / (0.300 - 0.165)                # horn: round cone, top drops a little, underside sweeps up
    top = lerp(STEP_TOP, 0.270, t ** 1.1)
    bot = lerp(0.196, 0.262, t ** 1.3)
    h = max(top - bot, 0.004)
    w = max(0.112 * (1 - t) ** 0.75, 0.004)
    w = min(w, h * 1.1) if t > 0.05 else w
    n = lerp(10, 2.2, smooth(t / 0.3))
    return w, top, bot, n


xs = [-0.300, -0.298, -0.294]
x = -0.288
while x < 0.105:
    xs.append(x)
    x += 0.012
xs += [0.105, 0.107, 0.109, 0.112, 0.120, 0.135, 0.150, 0.160, 0.165]
xs += [0.165 + (0.135) * (i / 20) ** 0.9 for i in range(1, 20)] + [0.2975]
rings = []
for x in xs:
    w, top, bot, n = section(x)
    # end sections shrink slightly so the heel end has rounded edges
    k = 1.0
    if x < -0.293:
        k = 0.96 + 0.04 * (x + 0.300) / 0.007
    rings.append([(u, v, x) for (u, v, _z) in superellipse(0, (top + bot) / 2, 0, w * k, (top - bot) * (k if x < -0.29 else 1), n)])
body = loft("body", rings, "x")
bpy.ops.mesh.primitive_cube_add(size=0.026, location=(-0.255, 0, 0.28))
hardy = bpy.context.active_object
hardy.name = "hardy"
hardy.scale.z = 6
bpy.ops.mesh.primitive_cylinder_add(radius=0.0085, depth=0.2, vertices=24, location=(-0.212, 0, 0.28))
pritchel = bpy.context.active_object
pritchel.name = "pritchel"
cut(body, hardy)
cut(body, pritchel)
finish(body, 28)
for o in (hardy, pritchel):
    bpy.data.objects.remove(o, do_unlink=True)

# ------------------------------------------------------------------ waist + base, lofted along Z
zs = [0.0, 0.004, 0.012] + [0.02 + 0.195 * i / 24 for i in range(25)]
rings = []
for z in zs:
    t = z / 0.215
    # hourglass profile keyed by height fraction: (t, length, width)
    keys = [(0.0, 0.36, 0.200), (0.12, 0.345, 0.190), (0.30, 0.26, 0.135), (0.55, 0.19, 0.088),
            (0.75, 0.198, 0.090), (0.90, 0.225, 0.100), (1.0, 0.24, 0.108)]
    for (t0, l0, w0), (t1, l1, w1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = smooth((t - t0) / (t1 - t0))
            L, Wd = lerp(l0, l1, u), lerp(w0, w1, u)
            break
    if z < 0.005:
        L, Wd = L - 0.006 + z, Wd - 0.006 + z
    cxo = -0.012
    rings.append([(u, v, z) for (u, v, _) in superellipse(cxo, 0, 0, L, Wd, 8)])
waist = loft("waist", rings, "z")
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.085, depth=0.4, location=(-0.012, 0, -0.045),
                                    rotation=(math.pi / 2, 0, 0))
ax = bpy.context.active_object
ax.name = "arch_x"
ax.scale.y = 0.75
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.07, depth=0.6, location=(-0.012, 0, -0.04),
                                    rotation=(0, math.pi / 2, 0))
ay = bpy.context.active_object
ay.name = "arch_y"
ay.scale.x = 0.7
cut(waist, ax)
cut(waist, ay)
bv = waist.modifiers.new("bevel", "BEVEL")
bv.width, bv.segments, bv.limit_method, bv.angle_limit = 0.004, 3, "ANGLE", math.radians(50)
finish(waist, 35)
for o in (ax, ay):
    bpy.data.objects.remove(o, do_unlink=True)

# ------------------------------------------------------------------ material
mat = bpy.data.materials.new("anvil_steel")
mat.use_nodes = True
nt = mat.node_tree
N, L_ = nt.nodes, nt.links
pr = N["Principled BSDF"]


def node(t, **inputs):
    n = N.new(t)
    for k, v in inputs.items():
        n.inputs[k].default_value = v
    return n


def math_node(op, a, b=None, clamp=False):
    n = N.new("ShaderNodeMath")
    n.operation, n.use_clamp = op, clamp
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            L_.new(x, n.inputs[i])
    return n.outputs[0]


def maprange(x, fmin, fmax, tmin=0.0, tmax=1.0):
    n = node("ShaderNodeMapRange")
    n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = fmin, fmax
    n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = tmin, tmax
    L_.new(x, n.inputs["Value"])
    return n.outputs["Result"]


geo = N.new("ShaderNodeNewGeometry")
tc = N.new("ShaderNodeTexCoord")
sepP = N.new("ShaderNodeSeparateXYZ")
L_.new(geo.outputs["Position"], sepP.inputs["Vector"])
sepN = N.new("ShaderNodeSeparateXYZ")
L_.new(geo.outputs["Normal"], sepN.inputs["Vector"])
# edge mask from the bevel node (rounded normal v true normal)
bev = node("ShaderNodeBevel", Radius=0.009)
bev.samples = 16
dot = N.new("ShaderNodeVectorMath")
dot.operation = "DOT_PRODUCT"
L_.new(bev.outputs["Normal"], dot.inputs[0])
L_.new(geo.outputs["Normal"], dot.inputs[1])
edge = maprange(dot.outputs["Value"], 0.99, 0.90)
# breakup noises
n_fine = node("ShaderNodeTexNoise", Scale=90.0, Detail=10.0, Roughness=0.6)
L_.new(tc.outputs["Object"], n_fine.inputs["Vector"])
n_mid = node("ShaderNodeTexNoise", Scale=18.0, Detail=6.0)
L_.new(tc.outputs["Object"], n_mid.inputs["Vector"])
n_big = node("ShaderNodeTexNoise", Scale=5.0, Detail=3.0)
L_.new(tc.outputs["Object"], n_big.inputs["Vector"])
# wear: edges, broken up, stronger high on the anvil (hands and work touch the top)
height = maprange(sepP.outputs["Z"], 0.10, 0.29, 0.25, 1.0)
wear = math_node("MULTIPLY", edge, maprange(n_mid.outputs["Fac"], 0.38, 0.62), clamp=True)
wear = math_node("MULTIPLY", wear, height, clamp=True)
# polished striking face: flat, up, at the top; plus the step and horn top get light polish
up = maprange(sepN.outputs["Z"], 0.96, 0.995)
top = maprange(sepP.outputs["Z"], 0.293, 0.298)
face = math_node("MULTIPLY", up, top, clamp=True)
bright = math_node("MAXIMUM", wear, face, clamp=True)
# lengthwise polish scratches on the face (noise stretched along X)
mp = N.new("ShaderNodeMapping")
mp.inputs["Scale"].default_value = (3.0, 300.0, 300.0)
L_.new(tc.outputs["Object"], mp.inputs["Vector"])
scr = node("ShaderNodeTexNoise", Scale=1.0, Detail=4.0)
L_.new(mp.outputs["Vector"], scr.inputs["Vector"])
scratch = maprange(scr.outputs["Fac"], 0.45, 0.7)
# rust/scale in crevices from AO
ao = N.new("ShaderNodeAmbientOcclusion")
ao.samples = 16
ao.inputs["Distance"].default_value = 0.03
crev = math_node("MULTIPLY", maprange(ao.outputs["AO"], 0.75, 0.35), maprange(n_big.outputs["Fac"], 0.4, 0.6),
                 clamp=True)
# colours
dark = N.new("ShaderNodeValToRGB")
dark.color_ramp.elements[0].color = (0.026, 0.026, 0.028, 1)
dark.color_ramp.elements[1].color = (0.060, 0.056, 0.052, 1)
L_.new(n_big.outputs["Fac"], dark.inputs["Fac"])
rustc = N.new("ShaderNodeMix")
rustc.data_type = "RGBA"
L_.new(crev, rustc.inputs["Factor"])
L_.new(dark.outputs["Color"], rustc.inputs[6])
rustc.inputs[7].default_value = (0.085, 0.040, 0.020, 1)
steel = N.new("ShaderNodeMix")
steel.data_type = "RGBA"
L_.new(scratch, steel.inputs["Factor"])
steel.inputs[6].default_value = (0.40, 0.40, 0.42, 1)
steel.inputs[7].default_value = (0.29, 0.29, 0.30, 1)
col = N.new("ShaderNodeMix")
col.data_type = "RGBA"
L_.new(bright, col.inputs["Factor"])
L_.new(rustc.outputs[2], col.inputs[6])
L_.new(steel.outputs[2], col.inputs[7])
L_.new(col.outputs[2], pr.inputs["Base Color"])
# roughness: forged 0.55-0.7, rust 0.85, polished 0.22-0.35
r_body = math_node("ADD", maprange(n_fine.outputs["Fac"], 0.3, 0.7, 0.52, 0.70),
                   math_node("MULTIPLY", crev, 0.2))
r_pol = maprange(scratch, 0.0, 1.0, 0.26, 0.42)
rmix = N.new("ShaderNodeMix")
L_.new(bright, rmix.inputs["Factor"])
L_.new(r_body, rmix.inputs[2])
L_.new(r_pol, rmix.inputs[3])
L_.new(rmix.outputs[0], pr.inputs["Roughness"])
met = math_node("MULTIPLY", maprange(bright, 0.0, 1.0, 0.70, 1.0), math_node("SUBTRACT", 1.0, math_node(
    "MULTIPLY", crev, 0.7)), clamp=True)
L_.new(met, pr.inputs["Metallic"])
# hammer pitting and forging texture; weaker on the face
bump = node("ShaderNodeBump", Distance=0.001)
L_.new(maprange(face, 0, 1, 0.45, 0.06), bump.inputs["Strength"])
hmix = math_node("ADD", math_node("MULTIPLY", n_fine.outputs["Fac"], 0.6), math_node("MULTIPLY", n_mid.outputs["Fac"], 0.4))
L_.new(hmix, bump.inputs["Height"])
L_.new(bump.outputs["Normal"], pr.inputs["Normal"])

for ob in (body, waist):
    ob.data.materials.append(mat)
