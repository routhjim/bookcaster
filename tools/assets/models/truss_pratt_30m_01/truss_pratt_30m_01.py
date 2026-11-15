"""Model script: 30 m flat Pratt truss (set piece), same layout as the set builder's UE-built truss so the two can
be compared: chord centres at z = 0.175 and 1.775 (I 350 x 250), 12 panels of 2.5 m, verticals at every station,
diagonals falling toward mid-span, built from back-to-back angles (two L 100 x 100 x 10 with a 12 mm gap), a
gusset plate at each node with bolts, end bearing plates. Materials: painted steel (dark grey, worn edges, dust
on up-facing surfaces, rust bleeding below bolts), galvanised bolts."""

import math

import bpy
from mathutils import Vector

parts = {"steel": [], "bolt": []}
SPAN, NP, CD, CW, TF, TW = 30.0, 12, 0.35, 0.25, 0.022, 0.012
ZB, ZT = CD / 2, CD / 2 + 1.6
GAP, AL, AT = 0.012, 0.10, 0.010


def finish(ob, g, bevel=0.0):
    if bevel:
        m = ob.modifiers.new("bv", "BEVEL")
        m.width, m.segments, m.limit_method = bevel, 1, "ANGLE"
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    parts[g].append(ob)
    return ob


def box(c, size, g="steel", rot=None, bevel=0.003):
    bpy.ops.mesh.primitive_cube_add(size=1, location=c)
    ob = bpy.context.active_object
    ob.scale = size
    if rot is not None:
        ob.rotation_euler = rot
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(ob, g, bevel)


def bar(a, b, w, h, off=(0, 0, 0), bevel=0.003):
    """Box from a to b in the XZ plane (width w along Y, thickness h in-plane), shifted by off."""
    a, b = Vector(a), Vector(b)
    d = b - a
    ang = math.atan2(d.z, d.x)
    c = (a + b) / 2 + Vector(off)
    return box(c, (d.length, w, h), rot=(0, -ang, 0), bevel=bevel)


def ibeam(z):
    box((0, 0, z + CD / 2 - TF / 2), (SPAN, CW, TF))
    box((0, 0, z - CD / 2 + TF / 2), (SPAN, CW, TF))
    box((0, 0, z), (SPAN, TW, CD - 2 * TF + 0.002))


def angles(a, b):
    """Two back-to-back L angles from a to b on either side of the gusset plane (Y = 0)."""
    for s in (-1, 1):
        y0 = s * (GAP / 2 + AT / 2)
        bar(a, b, AT, AL, (0, y0, 0))                         # leg in the truss plane
        y1 = s * (GAP / 2 + AL / 2)
        d = (Vector(b) - Vector(a)).normalized()
        n = Vector((-d.z, 0, d.x)) * (AL / 2 - AT / 2)       # outstanding leg along one edge of the first
        bar(a, b, AL, AT, (n.x, y1, n.z))


ibeam(ZB)
ibeam(ZT)
xs = [-SPAN / 2 + SPAN * i / NP for i in range(NP + 1)]
inner_b, inner_t = ZB + CD / 2, ZT - CD / 2
for i, x in enumerate(xs):
    xx = min(max(x, -SPAN / 2 + 0.15), SPAN / 2 - 0.15)
    angles((xx, 0, inner_b), (xx, 0, inner_t))
    for z, sgn in ((inner_b, 1), (inner_t, -1)):             # gusset plate against the chord web
        box((xx, 0, z + sgn * 0.17), (0.5, GAP, 0.34), bevel=0.002)
        for bx in (-0.15, 0.15):
            for bz in (0.09, 0.25):
                for s in (-1, 1):
                    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.014, depth=0.012,
                                                        location=(xx + bx, s * (GAP / 2 + AT + 0.006), z + sgn * bz),
                                                        rotation=(math.pi / 2, 0, 0))
                    finish(bpy.context.active_object, "bolt")
    if i < NP:
        x2 = xs[i + 1]
        if x2 <= 1e-6:
            a, b = (x + 0.15, 0, inner_t - 0.1), (x2 - 0.15, 0, inner_b + 0.1)
        else:
            a, b = (x + 0.15, 0, inner_b + 0.1), (x2 - 0.15, 0, inner_t - 0.1)
        angles(a, b)
for s in (-1, 1):                                             # end bearing plates
    box((s * (SPAN / 2 - 0.01), 0, (ZB + ZT) / 2), (0.02, CW, ZT - ZB + CD), bevel=0.002)

# ------------------------------------------------------------------ materials
def mat_paint():
    m = bpy.data.materials.new("truss_paint")
    m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links
    pr = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord")
    geo = N.new("ShaderNodeNewGeometry")

    def noise(scale, detail=4.0, stretch=None):
        n = N.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value, n.inputs["Detail"].default_value = scale, detail
        if stretch:
            mp = N.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = stretch
            L.new(tc.outputs["Object"], mp.inputs["Vector"])
            L.new(mp.outputs["Vector"], n.inputs["Vector"])
        else:
            L.new(tc.outputs["Object"], n.inputs["Vector"])
        return n.outputs["Fac"]

    def mr(x, a, b, c=0.0, d=1.0):
        n = N.new("ShaderNodeMapRange")
        n.inputs[1].default_value, n.inputs[2].default_value = a, b
        n.inputs[3].default_value, n.inputs[4].default_value = c, d
        L.new(x, n.inputs[0])
        return n.outputs[0]

    def mix(f, a, b):
        n = N.new("ShaderNodeMix")
        n.data_type = "RGBA"
        for sock, v in ((n.inputs[6], a), (n.inputs[7], b)):
            if isinstance(v, tuple):
                sock.default_value = v
            else:
                L.new(v, sock)
        L.new(f, n.inputs["Factor"])
        return n.outputs[2]

    def mul(a, b):
        n = N.new("ShaderNodeMath")
        n.operation = "MULTIPLY"
        n.use_clamp = True
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    bev = N.new("ShaderNodeBevel")
    bev.inputs["Radius"].default_value = 0.006
    dp = N.new("ShaderNodeVectorMath")
    dp.operation = "DOT_PRODUCT"
    L.new(bev.outputs["Normal"], dp.inputs[0])
    L.new(geo.outputs["Normal"], dp.inputs[1])
    edge = mr(dp.outputs["Value"], 0.995, 0.9)
    up = N.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Normal"], up.inputs[0])
    dust = mul(mr(up.outputs["Z"], 0.6, 0.95), mr(noise(3.0), 0.35, 0.65))
    rust = mul(mr(noise(1.0, 6.0, (40.0, 40.0, 3.0)), 0.6, 0.75), mr(noise(8.0), 0.4, 0.7))
    blot = noise(0.4)
    col = mix(mr(blot, 0.3, 0.7), (0.035, 0.037, 0.036, 1), (0.055, 0.056, 0.053, 1))
    col = mix(mul(edge, mr(noise(20.0), 0.4, 0.6)), col, (0.22, 0.21, 0.2, 1))
    col = mix(mul(rust, 0.8), col, (0.12, 0.05, 0.02, 1))
    col = mix(mul(dust, 0.9), col, (0.16, 0.15, 0.13, 1))
    L.new(col, pr.inputs["Base Color"])
    L.new(mr(dust, 0, 1, 0.5, 0.9), pr.inputs["Roughness"])
    return m


paint = mat_paint()
bolt = bpy.data.materials.new("truss_bolt")
bolt.use_nodes = True
b = bolt.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0.3, 0.3, 0.29, 1)
b.inputs["Metallic"].default_value = 1.0
b.inputs["Roughness"].default_value = 0.55
for g, mt in (("steel", paint), ("bolt", bolt)):
    for ob in parts[g]:
        ob.data.materials.clear()
        ob.data.materials.append(mt)
