"""Model script (method B): an asset from its reference drawings: the image-to-3D shape (tools/blockout_shape.py
--views, a .glb) cleaned up to the contract and textured by projecting the drawings onto it.

Spec block "from_views":
    "mesh":   the shape (.glb, Y-up as the image-to-3D model writes it; front +Z in glTF = -Y here)
    "views":  {"front" | "back" | "left" | "right" | "top": picture}; "left" is the view with the object's front to
              the picture's left (camera on +X), "right" the other side, "top" has the front at the bottom
    "yaw_deg": turn about Z after import (if the shape's front is not -Y), default 0
    "keep_aspect": false (default) scales each axis to size_m; true scales by height only
    "rough", "metal": the surface's roughness / metallic (the drawings give colour only), defaults 0.6 / 0.0
    "power": sharpness of the blend between views (default 4)
    "regions": [{"name", "z": [z0, z1] (0..1 of the height), "r": [r0, r1] (0..1 of the half-width, radial from the
                vertical axis), "hsv": [[h0, h1], [s0, s1], [v0, v1]] (the projected colour, all 0..1, optional),
                "box": [x0, x1, y0, y1, z0, z1] (0..1 of the bounds), "facing": [nx, ny, nz, min dot],
                "material": "glass" | "emit" | "solid" | "delete", "color", "rough", "ior", "strength", "glow"}]
              faces matching a region get that material instead of the projection: real glass (transmissive; with
              "strength", a glow of colour "glow"), an emissive surface (flames, coals), a solid colour, or removed
              (where method C puts an exact part)
    "parts":  {"script": another model script, "spec": {...}} adds that script's parts (method C: exact glass, flame,
              dial or legs from the scripted model)

Every step runs in Blender (the import, a vectorised face classification, node set-up); the projection itself is
a node material, baked on the GPU by the harness like any other.
"""

import colorsys
import math
import os
import runpy

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

FV = SPEC["from_views"]  # noqa: F821
P = float(FV.get("power", 4.0))
here = os.path.dirname(os.path.abspath(__file__))


def ex(p):
    return os.path.expandvars(os.path.expanduser(p))


# ---------------------------------------------------------------- the shape
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=ex(FV["mesh"]))
new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
for o in bpy.data.objects:
    if o not in before and o.type != "MESH":
        bpy.data.objects.remove(o, do_unlink=True)
bpy.context.view_layer.update()
for o in new:                                         # bake the importer's Y-up -> Z-up rotation into the mesh
    o.data.transform(o.matrix_world)
    o.matrix_world = Matrix.Identity(4)
if len(new) > 1:
    with bpy.context.temp_override(active_object=new[0], selected_editable_objects=new):
        bpy.ops.object.join()
ob = new[0]
ob.name = "shape"
me = ob.data
if FV.get("yaw_deg"):
    me.transform(Matrix.Rotation(math.radians(FV["yaw_deg"]), 4, "Z"))
co = np.empty(len(me.vertices) * 3)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
lo, hi = co.min(0), co.max(0)
size = np.array(SPEC["size_m"], dtype=float)  # noqa: F821
ext = np.maximum(hi - lo, 1e-6)
sc = size / ext if not FV.get("keep_aspect") else np.full(3, size[2] / ext[2])
co = (co - [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]) * sc
me.vertices.foreach_set("co", co.ravel())
me.update()
for p in me.polygons:
    p.use_smooth = True
bm = bmesh.new()                                      # sensible normals: outward, consistent
bm.from_mesh(me)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bm.to_mesh(me)
bm.free()
me.update()
half = np.array([size[0] / 2, size[1] / 2]) if not FV.get("keep_aspect") else ext[:2] * sc[:2] / 2
H = co[:, 2].max()


# ---------------------------------------------------------------- the drawings
def object_box(img):
    """The object's box in the picture (0..1, origin bottom left) by its difference from the corner colour."""
    w, h = img.size
    a = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(h, w, 4)[..., :3]
    bg = np.concatenate([a[:6, :6].reshape(-1, 3), a[-6:, -6:].reshape(-1, 3), a[:6, -6:].reshape(-1, 3),
                         a[-6:, :6].reshape(-1, 3)]).mean(0)
    m = np.abs(a - bg).sum(-1) > 0.07
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return (0.0, 1.0, 0.0, 1.0), a
    x0, x1 = np.percentile(xs, [0.2, 99.8])
    y0, y1 = np.percentile(ys, [0.2, 99.8])
    return (x0 / w, (x1 + 1) / w, y0 / h, (y1 + 1) / h), a


# u, v from the generated (0..1 box) coordinates per view, and the normal axis that faces it
VIEWDEF = {"front": (("x", False), ("z", False), ("y", -1)), "back": (("x", True), ("z", False), ("y", 1)),
           "left": (("y", False), ("z", False), ("x", 1)), "right": (("y", True), ("z", False), ("x", -1)),
           "top": (("x", False), ("y", False), ("z", 1))}
views = {}
for name, path in FV["views"].items():
    img = bpy.data.images.load(ex(path), check_existing=True)
    img.colorspace_settings.name = "sRGB"
    box, arr = object_box(img)
    views[name] = (img, box, arr)

mat = bpy.data.materials.new("projected")
mat.use_nodes = True
nt = mat.node_tree
N, L = nt.nodes, nt.links
pr = N["Principled BSDF"]
tc = N.new("ShaderNodeTexCoord")
gs = N.new("ShaderNodeSeparateXYZ")
L.new(tc.outputs["Generated"], gs.inputs[0])
ns = N.new("ShaderNodeSeparateXYZ")
L.new(tc.outputs["Normal"], ns.inputs[0])


def math_node(op, a, b=None):
    n = N.new("ShaderNodeMath")
    n.operation = op
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            L.new(x, n.inputs[i])
    return n.outputs[0]


acc_c, acc_w = None, None
for name, (img, (u0, u1, v0, v1), _) in views.items():
    (ua, uf), (va, vf), (na, ns_sign) = VIEWDEF[name]
    gu = gs.outputs[ua.upper()]
    gv = gs.outputs[va.upper()]
    if uf:
        gu = math_node("SUBTRACT", 1.0, gu)
    if vf:
        gv = math_node("SUBTRACT", 1.0, gv)
    cx = N.new("ShaderNodeCombineXYZ")
    L.new(math_node("ADD", math_node("MULTIPLY", gu, u1 - u0), u0), cx.inputs[0])
    L.new(math_node("ADD", math_node("MULTIPLY", gv, v1 - v0), v0), cx.inputs[1])
    tx = N.new("ShaderNodeTexImage")
    tx.image, tx.extension = img, "EXTEND"
    L.new(cx.outputs[0], tx.inputs[0])
    w = math_node("POWER", math_node("MAXIMUM", math_node("MULTIPLY", ns.outputs[na.upper()], float(ns_sign)), 0.0), P)
    if name in ("left", "right") and "back" not in views:
        w = math_node("ADD", w, 0.02)                  # the back gets the sides when there is no back view
    if name == "front":
        w = math_node("ADD", w, 0.002)                 # undersides
    c = N.new("ShaderNodeVectorMath")
    c.operation = "SCALE"
    L.new(tx.outputs["Color"], c.inputs[0])
    L.new(w, c.inputs["Scale"])
    if acc_c is None:
        acc_c, acc_w = c.outputs[0], w
    else:
        a = N.new("ShaderNodeVectorMath")
        a.operation = "ADD"
        L.new(acc_c, a.inputs[0])
        L.new(c.outputs[0], a.inputs[1])
        acc_c, acc_w = a.outputs[0], math_node("ADD", acc_w, w)
dv = N.new("ShaderNodeVectorMath")
dv.operation = "SCALE"
L.new(acc_c, dv.inputs[0])
L.new(math_node("DIVIDE", 1.0, math_node("MAXIMUM", acc_w, 1e-4)), dv.inputs["Scale"])
L.new(dv.outputs[0], pr.inputs["Base Color"])
pr.inputs["Roughness"].default_value = float(FV.get("rough", 0.6))
pr.inputs["Metallic"].default_value = float(FV.get("metal", 0.0))
me.materials.append(mat)


# ---------------------------------------------------------------- regions (glass, glow) by place and colour
def projected_colour(cent, nrm):
    """The drawing's colour at each face centre (the view the face looks at most), vectorised."""
    g = (cent - [-half[0], -half[1], 0.0]) / [2 * half[0], 2 * half[1], H]
    out = np.zeros((len(cent), 3))
    best = np.full(len(cent), -1.0)
    for name, (img, (u0, u1, v0, v1), arr) in views.items():
        (ua, uf), (va, vf), (na, s) = VIEWDEF[name]
        ax = "xyz"
        gu = g[:, ax.index(ua)]
        gv = g[:, ax.index(va)]
        gu = 1 - gu if uf else gu
        gv = 1 - gv if vf else gv
        u = np.clip(u0 + gu * (u1 - u0), 0, 0.9999)
        v = np.clip(v0 + gv * (v1 - v0), 0, 0.9999)
        h_, w_ = arr.shape[:2]
        col = arr[(v * h_).astype(int), (u * w_).astype(int)]
        f = nrm[:, ax.index(na)] * s
        take = f > best
        out[take], best[take] = col[take], f[take]
    return out


regs = FV.get("regions") or []
if regs:
    nf = len(me.polygons)
    cent = np.empty(nf * 3)
    me.polygons.foreach_get("center", cent)
    cent = cent.reshape(-1, 3)
    nrm = np.empty(nf * 3)
    me.polygons.foreach_get("normal", nrm)
    nrm = nrm.reshape(-1, 3)
    col = projected_colour(cent, nrm)
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in np.clip(col, 0, 1)]) if any("hsv" in r for r in regs) else None
    zf = cent[:, 2] / H
    rf = np.hypot(cent[:, 0] / half[0], cent[:, 1] / half[1])
    idx = np.zeros(nf, dtype=np.int32)
    for k, r in enumerate(regs):
        sel = np.ones(nf, dtype=bool)
        if "z" in r:
            sel &= (zf >= r["z"][0]) & (zf <= r["z"][1])
        if "r" in r:
            sel &= (rf >= r["r"][0]) & (rf <= r["r"][1])
        if "hsv" in r:
            for i, (a, b) in enumerate(r["hsv"]):
                sel &= (hsv[:, i] >= a) & (hsv[:, i] <= b)
        if "box" in r:                                 # [x0, x1, y0, y1, z0, z1], 0..1 of the bounds
            b = r["box"]
            gx = (cent[:, 0] + half[0]) / (2 * half[0])
            gy = (cent[:, 1] + half[1]) / (2 * half[1])
            sel &= (gx >= b[0]) & (gx <= b[1]) & (gy >= b[2]) & (gy <= b[3]) & (zf >= b[4]) & (zf <= b[5])
        if "facing" in r:                              # [nx, ny, nz, min dot]
            f = r["facing"]
            sel &= nrm @ np.array(f[:3]) >= f[3]
        if r.get("material") == "delete":              # an exact part takes this place (method C)
            idx[sel & (idx == 0)] = -1
            continue
        m = bpy.data.materials.new(r.get("name", f"region{k}"))
        m.use_nodes = True
        q = m.node_tree.nodes["Principled BSDF"]
        kind = r.get("material", "solid")
        if kind == "glass":
            q.inputs["Base Color"].default_value = (*r.get("color", (0.95, 0.95, 0.95)), 1)
            q.inputs["Roughness"].default_value = r.get("rough", 0.05)
            q.inputs["IOR"].default_value = r.get("ior", 1.5)
            q.inputs["Transmission Weight"].default_value = 1.0
            if r.get("strength"):
                q.inputs["Emission Color"].default_value = (*r.get("glow", (1.0, 0.7, 0.4)), 1)
                q.inputs["Emission Strength"].default_value = r["strength"]
            m["asset_glass"] = True
        elif kind == "emit":                           # a glowing surface (flame, coals) of one colour
            q.inputs["Emission Color"].default_value = (*r.get("color", (1.0, 0.55, 0.2)), 1)
            q.inputs["Emission Strength"].default_value = r.get("strength", 8.0)
            q.inputs["Base Color"].default_value = (*r.get("color", (1.0, 0.55, 0.2)), 1)
        else:
            q.inputs["Base Color"].default_value = (*r.get("color", (0.5, 0.5, 0.5)), 1)
            q.inputs["Roughness"].default_value = r.get("rough", 0.5)
            q.inputs["Metallic"].default_value = r.get("metal", 0.0)
        me.materials.append(m)
        idx[sel & (idx == 0)] = len(me.materials) - 1
    gone = np.nonzero(idx < 0)[0]
    idx[idx < 0] = 0
    me.polygons.foreach_set("material_index", idx)
    me.update()
    if len(gone):
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in gone], context="FACES")
        bm.to_mesh(me)
        bm.free()
        me.update()
    print("from_views regions:", {r.get("name", k): int((idx == k + 1).sum()) for k, r in enumerate(regs)})

# ---------------------------------------------------------------- method C: exact parts from the scripted model
if FV.get("parts"):
    pp = FV["parts"]
    runpy.run_path(ex(pp["script"]) if os.path.isabs(ex(pp["script"])) else os.path.join(here, pp["script"]),
                   init_globals={"SPEC": dict(SPEC, **pp.get("spec", {}))})  # noqa: F821
