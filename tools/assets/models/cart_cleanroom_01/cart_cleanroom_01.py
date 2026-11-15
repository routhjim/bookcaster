"""Model script: cleanroom stainless utility cart, 0.9 x 0.6 m shelves, top shelf surface at 0.90 m.

Two shelves (1.5 mm sheet look: 25 mm thick boxes with bevelled edges and a 15 mm rim on the top shelf), four
25 mm round posts, push handle (two uprights + grip bar) at +X, four swivel casters (plate, fork, axle, 100 mm
grey polyurethane wheel). Materials: brushed stainless with faint fingerprints and edge polish; wheel PU.
"""

import math

import bpy
from mathutils import Vector

parts = {"steel": [], "wheel": []}


def shade(ob, angle=40):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))


def box(g, lo, hi, bevel=0.002):
    lo, hi = Vector(lo), Vector(hi)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(lo + hi) / 2)
    ob = bpy.context.active_object
    ob.scale = hi - lo
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        m = ob.modifiers.new("bv", "BEVEL")
        m.width, m.segments = bevel, 2
    shade(ob, 35)
    parts[g].append(ob)
    return ob


def cyl(g, a, b, r, verts=20, bevel=0.0):
    a, b = Vector(a), Vector(b)
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=d.length, location=(a + b) / 2)
    ob = bpy.context.active_object
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if bevel:
        m = ob.modifiers.new("bv", "BEVEL")
        m.width, m.segments, m.limit_method = bevel, 2, "ANGLE"
    shade(ob, 40)
    parts[g].append(ob)
    return ob


HX, HY = 0.45, 0.30           # shelf half sizes
TOP, LOW, T = 0.90, 0.27, 0.025
PX, PY, PR = HX - 0.025, HY - 0.025, 0.0125
# shelves and rim
box("steel", (-HX, -HY, TOP - T), (HX, HY, TOP), 0.004)
box("steel", (-HX, -HY, LOW - T), (HX, HY, LOW), 0.004)
for (lo, hi) in (((-HX, -HY, TOP - 0.002), (HX, -HY + 0.012, TOP + 0.015)),
                 ((-HX, HY - 0.012, TOP - 0.002), (HX, HY, TOP + 0.015)),
                 ((-HX, -HY, TOP - 0.002), (-HX + 0.012, HY, TOP + 0.015)),
                 ((HX - 0.012, -HY, TOP - 0.002), (HX, HY, TOP + 0.015))):
    box("steel", lo, hi, 0.003)
# posts
for sx in (-1, 1):
    for sy in (-1, 1):
        cyl("steel", (sx * PX, sy * PY, 0.125), (sx * PX, sy * PY, TOP - 0.004), PR, 20, 0.002)
# handle at +X
for sy in (-1, 1):
    cyl("steel", (HX - 0.02, sy * 0.22, TOP + 0.01), (HX + 0.035, sy * 0.22, 1.0), 0.011, 16)
cyl("steel", (HX + 0.035, -0.235, 1.0), (HX + 0.035, 0.235, 1.0), 0.013, 20, 0.002)
# casters
WR, WW = 0.05, 0.034
for sx in (-1, 1):
    for sy in (-1, 1):
        x, y = sx * PX, sy * PY
        box("steel", (x - 0.035, y - 0.035, 0.112), (x + 0.035, y + 0.035, 0.125), 0.002)
        for e in (-1, 1):
            box("steel", (x - 0.02, y + e * (WW / 2) - (0.004 if e < 0 else 0), WR - 0.012),
                (x + 0.02, y + e * (WW / 2) + (0.004 if e > 0 else 0), 0.113), 0.001)
        cyl("steel", (x, y - WW / 2 - 0.004, WR), (x, y + WW / 2 + 0.004, WR), 0.006, 12)
        cyl("wheel", (x, y - WW / 2 + 0.001, WR), (x, y + WW / 2 - 0.001, WR), WR, 32, 0.006)

# ------------------------------------------------------------------ materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree.nodes, m.node_tree.links, m.node_tree.nodes["Principled BSDF"]


m, N, L, pr = new_mat("cart_steel")
tc = N.new("ShaderNodeTexCoord")
mp = N.new("ShaderNodeMapping")
mp.inputs["Scale"].default_value = (2.0, 300.0, 2.0)          # brushed along X
L.new(tc.outputs["Object"], mp.inputs["Vector"])
br = N.new("ShaderNodeTexNoise")
br.inputs["Scale"].default_value, br.inputs["Detail"].default_value = 1.0, 6.0
L.new(mp.outputs["Vector"], br.inputs["Vector"])
fp = N.new("ShaderNodeTexNoise")                             # smudges / fingerprints
fp.inputs["Scale"].default_value, fp.inputs["Detail"].default_value = 18.0, 4.0
L.new(tc.outputs["Object"], fp.inputs["Vector"])
rr = N.new("ShaderNodeMapRange")
rr.inputs["From Min"].default_value, rr.inputs["From Max"].default_value = 0.35, 0.7
rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = 0.24, 0.42
L.new(fp.outputs["Fac"], rr.inputs["Value"])
L.new(rr.outputs["Result"], pr.inputs["Roughness"])
cr = N.new("ShaderNodeMapRange")
cr.inputs["To Min"].default_value, cr.inputs["To Max"].default_value = 0.55, 0.68
L.new(br.outputs["Fac"], cr.inputs["Value"])
cc = N.new("ShaderNodeCombineColor")
for k in ("Red", "Green", "Blue"):
    L.new(cr.outputs["Result"], cc.inputs[k])
L.new(cc.outputs["Color"], pr.inputs["Base Color"])
pr.inputs["Metallic"].default_value = 1.0
steel = m

m, N, L, pr = new_mat("cart_wheel")
pr.inputs["Base Color"].default_value = (0.32, 0.33, 0.34, 1)
pr.inputs["Roughness"].default_value = 0.6
wheel = m

for g, mat in (("steel", steel), ("wheel", wheel)):
    for ob in parts[g]:
        ob.data.materials.clear()
        ob.data.materials.append(mat)
