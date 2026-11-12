"""An outer layer (cloak, cape, mantle, shawl) as its own garment piece over a fitted character.

    blender -b --factory-startup --python bl_layer.py -- JOB.json

The layer is generated drawn on a plain neutral mannequin in the character's spread pose (the same pose map as its
outfit), made 3D (TRELLIS.2), then here:
  1. the figure is scaled to the body's height (feet on the floor), centred on the pelvis and the chest;
  2. the mannequin is removed: vertices near the body's surface (MANNEQUIN_NEAR) whose colour is the mannequin's
     (the median colour of the vertices nearest the body, low saturation), and everything above the neck and beyond
     the wrists;
  3. what is left is pushed outside the body and the character's fitted outfit (outfit FBX) by MARGIN;
  4. skinned from the body's nearest weights, then held to the upper body: thigh, calf and foot weights go to the
     pelvis and the lower arm and hand weights to the upper arm, so the hem hangs free of the legs and the layer
     follows the shoulders, spine and upper arms; no cloth relaxation;
  5. written as the character tools' outfit FBX (centimetres) with its baked colour texture.

JOB: {"body_fbx", "head_fbx", "figure_glb", "outfit_fbx" (the character's fitted outfit), "out",
      "margin_m": 0.015, "mannequin_near_m": 0.06, "mannequin_colour_tol": 0.14}
"""
import json
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = JOB["out"]
os.makedirs(os.path.join(OUT, "textures"), exist_ok=True)
REP = {"times": {}}
T0 = time.time()


def import_fbx(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path, use_anim=False, ignore_leaf_bones=False, automatic_bone_orientation=False)
    return [o for o in bpy.data.objects if o not in before]


def flatten_empties():
    for o in list(bpy.data.objects):
        if o.type == "EMPTY":
            for ch in list(o.children):
                mw = ch.matrix_world.copy()
                ch.parent = None
                ch.matrix_world = mw
            bpy.data.objects.remove(o)


bpy.ops.wm.read_factory_settings(use_empty=True)
objs = import_fbx(JOB["body_fbx"])
arm = next(o for o in objs if o.type == "ARMATURE")
body = max((o for o in objs if o.type == "MESH"), key=lambda o: len(o.data.vertices))
for o in objs:
    if o.type == "MESH" and o is not body:
        bpy.data.objects.remove(o)
hn = [o.name for o in import_fbx(JOB["head_fbx"])]
flatten_empties()
hobjs = [bpy.data.objects[n] for n in hn if n in bpy.data.objects]
hm = max((o for o in hobjs if o.type == "MESH"), key=lambda o: len(o.data.vertices))
top_z = max((hm.matrix_world @ v.co).z for v in hm.data.vertices)
for o in hobjs:
    bpy.data.objects.remove(o)
flatten_empties()
bpy.ops.object.select_all(action="DESELECT")
for o in (arm, body):
    o.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
B = arm.data.bones


def J(n):
    return np.array(B[n].head_local)


# the body posed to the drawing's arm angle (the layer was drawn on a figure standing so), bound in that pose
DRAW_DEG = JOB.get("draw_deg")
if DRAW_DEG is not None:
    import math
    from mathutils import Matrix
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    th = math.radians(float(DRAW_DEG))
    for side, sg in (("l", 1), ("r", -1)):
        pb = arm.pose.bones[f"upperarm_{side}"]
        hd = pb.head.copy()
        frm = arm.pose.bones[f"hand_{side}"].head - hd
        to = Vector((sg * math.sin(th), 0.0, -math.cos(th)))
        q = frm.normalized().rotation_difference(to)
        pb.matrix = Matrix.Translation(hd) @ q.to_matrix().to_4x4() @ Matrix.Translation(-hd) @ pb.matrix
        bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")
    REP["posed_deg"] = float(DRAW_DEG)


# the character's fitted outfit (collider): centimetres in its FBX
on = import_fbx(JOB["outfit_fbx"])
flatten_empties()
oc = [o for o in on if o.type == "MESH" and o.name in bpy.data.objects]
for o in on:
    if o.name in bpy.data.objects and o not in oc:
        bpy.data.objects.remove(o)
for o in oc:
    o.parent = None
    for m in list(o.modifiers):
        o.modifiers.remove(m)
bpy.ops.object.select_all(action="DESELECT")
for o in oc:
    o.select_set(True)
bpy.context.view_layer.objects.active = oc[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
outfit_v = np.array([v.co[:] for o in oc for v in o.data.vertices])
if outfit_v[:, 2].max() > 10:          # centimetres
    for o in oc:
        o.data.transform(__import__("mathutils").Matrix.Scale(0.01, 4))
        o.data.update()

# ------------------------------------------------------------------ the figure and its colours
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=JOB["figure_glb"])
fm = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in fm:
    o.select_set(True)
bpy.context.view_layer.objects.active = fm[0]
if len(fm) > 1:
    bpy.ops.object.join()
fig = bpy.context.view_layer.objects.active
fig.parent = None
for o in list(bpy.data.objects):
    if o not in (arm, body, fig) and o not in oc:
        bpy.data.objects.remove(o)
bpy.ops.object.select_all(action="DESELECT")
fig.select_set(True)
bpy.context.view_layer.objects.active = fig
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
img = None
for m in fig.data.materials:
    for n in (m.node_tree.nodes if m and m.use_nodes else []):
        if n.type == "TEX_IMAGE" and n.image is not None:
            img = n.image
            break
    if img:
        break
W_, H_ = img.size
px = np.empty(W_ * H_ * 4, np.float32)
img.pixels.foreach_get(px)
px = px.reshape(H_, W_, 4)
me = fig.data
nl = len(me.loops)
uv = np.empty(nl * 2, np.float32)
me.uv_layers.active.data.foreach_get("uv", uv)
uv = uv.reshape(-1, 2)
lv = np.empty(nl, np.int64)
me.loops.foreach_get("vertex_index", lv)
lc = px[np.clip((uv[:, 1] * H_).astype(int), 0, H_ - 1), np.clip((uv[:, 0] * W_).astype(int), 0, W_ - 1), :3]
nv = len(me.vertices)
col = np.zeros((nv, 3))
cnt = np.zeros(nv)
np.add.at(col, lv, lc)
np.add.at(cnt, lv, 1)
col /= np.maximum(cnt, 1)[:, None]

# ------------------------------------------------------------------ 1. scale and place
co = np.array([v.co[:] for v in me.vertices])
zmin, zmax = co[:, 2].min(), co[:, 2].max()
s = top_z / (zmax - zmin)
co = (co - [0, 0, zmin]) * s
bv = np.array([v.co[:] for v in body.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
# placed by the neutral figure itself (light, unsaturated), not the whole drawing: an open-fronted layer around it
# put the whole figure's middle off the body (the cloak landed at the right front, 11-12). Its neck and shoulders go
# on this body's, its chest centred front to back on the body's chest.
_sat = (col.max(1) - col.min(1)) / np.maximum(col.max(1), 1e-3)
man0 = (_sat < 0.15) & (col.mean(1) > 0.45)
REP["figure_light_verts"] = int(man0.sum())
if man0.sum() > 200:
    mz = co[man0]
    mtop = float(mz[:, 2].max())
    s2 = top_z / mtop                                  # the figure's head top on the body's
    co *= s2
    mz = co[man0]
    # the neck: the narrowest slice of the figure between 0.78 and 0.9 of its height
    zs_ = np.linspace(0.78 * top_z, 0.9 * top_z, 25)
    wid = [np.ptp(mz[np.abs(mz[:, 2] - z) < 0.01, 0]) if (np.abs(mz[:, 2] - z) < 0.01).sum() > 10 else 9 for z in zs_]
    zn = float(zs_[int(np.argmin(wid))])
    sl = mz[np.abs(mz[:, 2] - zn) < 0.01]
    nk = J("neck_01")
    co[:, 0] += nk[0] - float(np.median(sl[:, 0]))
    co[:, 2] += 0.5 * (J("neck_01")[2] + J("neck_02")[2]) - zn
    mz = co[man0]
    cz = (mz[:, 2] > J("spine_03")[2]) & (mz[:, 2] < J("spine_05")[2])
    bz = (bv[:, 2] > J("spine_03")[2]) & (bv[:, 2] < J("spine_05")[2]) & (np.abs(bv[:, 0] - J("spine_05")[0]) < 0.12)
    if cz.sum() > 20:
        co[:, 1] += float(np.median(bv[bz, 1])) - float(np.median(mz[cz, 1]))
    REP["aligned_on"] = {"figure_verts": int(man0.sum()), "neck_z": round(zn, 3)}
    s = s * s2
else:
    band = (co[:, 2] > 0.52 * top_z) & (co[:, 2] < 0.58 * top_z)
    co[:, 0] += J("pelvis")[0] - float(np.median(co[band, 0]))
    cz = (co[:, 2] > J("spine_03")[2]) & (co[:, 2] < J("spine_05")[2]) & (np.abs(co[:, 0] - J("spine_05")[0]) < 0.08)
    bz = (bv[:, 2] > J("spine_03")[2]) & (bv[:, 2] < J("spine_05")[2]) & (np.abs(bv[:, 0] - J("spine_05")[0]) < 0.08)
    co[:, 1] += float(np.median(bv[bz, 1])) - float(np.median(co[cz, 1]))
REP["scale"] = round(float(s), 4)

# ------------------------------------------------------------------ 2. the mannequin out
dg = bpy.context.evaluated_depsgraph_get()
bvb = BVHTree.FromObject(body, dg)
near = float(JOB.get("mannequin_near_m", 0.06))
dist = np.full(nv, 9.0)
for i in range(nv):
    loc, nrm, _, d = bvb.find_nearest(Vector(co[i]), 0.5)
    if loc is not None:
        dist[i] = d
sat = (col.max(1) - col.min(1)) / np.maximum(col.max(1), 1e-3)
lum = col.mean(1)
# the mannequin is drawn plain light grey: its colour is the median of the light, unsaturated vertices near the body
# (the layer itself also lies near the body at the shoulders and may be dark)
mk = (dist < 0.04) & (sat < 0.15) & (lum > 0.45)
man_col = np.median(col[mk], axis=0) if mk.sum() > 50 else np.array([0.75, 0.75, 0.75])
tol = float(JOB.get("mannequin_colour_tol", 0.14))
# the layer's own colour: the vertices far from the body (the mannequin is under it, near the body)
far = dist > 0.12
lay_col = np.median(col[far], axis=0) if far.sum() > 50 else None
apart = lay_col is None or np.linalg.norm(lay_col - man_col) > 2 * tol
# the mannequin by its colour anywhere when the layer's colour is clearly another (its arms stand far from the body
# in the drawn pose); only near the body otherwise
is_man = (np.linalg.norm(col - man_col, axis=1) < tol) & ((dist < near) | apart)
above = co[:, 2] > J("neck_01")[2] + 0.01
beyond = np.zeros(nv, bool)
drop = is_man | above
REP["mannequin"] = {"colour": [round(float(c), 3) for c in man_col], "by_colour": int(is_man.sum()),
                    "above_neck": int(above.sum()), "beyond_wrists": int(beyond.sum()), "kept": int((~drop).sum())}
if JOB.get("debug_npz"):
    np.savez_compressed(JOB["debug_npz"], co=co, col=col, drop=drop, dist=dist)
me.vertices.foreach_set("co", co.ravel())
me.update()
bm = bmesh.new()
bm.from_mesh(me)
bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(drop)[0]], context="VERTS")
# small loose pieces left by the cut
bm.verts.ensure_lookup_table()
comp = {}
for v0 in bm.verts:
    if v0.index in comp:
        continue
    stack, cid = [v0], len(set(comp.values()))
    comp[v0.index] = cid
    while stack:
        v = stack.pop()
        for e in v.link_edges:
            w = e.other_vert(v)
            if w.index not in comp:
                comp[w.index] = cid
                stack.append(w)
sizes = np.bincount(np.array([comp[v.index] for v in bm.verts]))
# only crumbs (the colour cut leaves the layer in several large pieces where the figure showed through it)
small = [v for v in bm.verts if sizes[comp[v.index]] < int(JOB.get("crumb_verts", 300))]
REP["pieces"] = int((sizes >= int(JOB.get("crumb_verts", 300))).sum())
bmesh.ops.delete(bm, geom=small, context="VERTS")
bm.to_mesh(me)
bm.free()
me.update()
REP["layer_verts"] = len(me.vertices)

# ------------------------------------------------------------------ 3. outside the body and the outfit
margin = float(JOB.get("margin_m", 0.015))
# the outfit's torso only: its sleeves are in the outfit's own bind pose (the drawing's spread), not this one
shx = abs(float(J("upperarm_l")[0]))
torso_cols = []
for o in oc:
    bm_ = bmesh.new()
    bm_.from_mesh(o.data)
    bm_.verts.ensure_lookup_table()
    bmesh.ops.delete(bm_, geom=[v for v in bm_.verts if abs(v.co.x) > shx + 0.02], context="VERTS")
    torso_cols.append(BVHTree.FromBMesh(bm_))
    bm_.free()
cols = [bvb] + torso_cols
P = np.array([v.co[:] for v in me.vertices])
moved = 0
for i in range(len(P)):
    for bvh in cols:
        loc, nrm, _, d = bvh.find_nearest(Vector(P[i]), 0.08)
        if loc is None:
            continue
        sd = (Vector(P[i]) - loc).dot(nrm)
        if sd < margin:
            P[i] = np.array(loc + nrm * margin)
            moved += 1
me.vertices.foreach_set("co", P.ravel())
me.update()
REP["pushed_out"] = moved

# ------------------------------------------------------------------ 4. weights: the body's nearest, held to the upper body
for o in oc:
    bpy.data.objects.remove(o)
dt = fig.modifiers.new("dt", "DATA_TRANSFER")
dt.object = body
dt.use_vert_data = True
dt.data_types_verts = {"VGROUP_WEIGHTS"}
dt.vert_mapping = "POLYINTERP_NEAREST"
dt.layers_vgroup_select_src = "ALL"
dt.layers_vgroup_select_dst = "NAME"
bpy.context.view_layer.objects.active = fig
bpy.ops.object.datalayout_transfer(modifier="dt")
bpy.ops.object.modifier_apply(modifier="dt")
names = {g.index: g.name for g in fig.vertex_groups}


def target(nm):
    """the upper body keeps its weights; arm below the shoulder goes to the upper arm, everything else (legs, feet,
    toes) to the pelvis, so the hem hangs free of the legs"""
    low = nm.lower()
    if low.startswith(("spine", "clavicle", "neck", "head", "pelvis", "upperarm")):
        return nm
    for side in ("_l", "_r"):
        if low.endswith(side) and low.startswith(("lowerarm", "hand", "index", "middle", "ring", "pinky", "thumb")):
            return "upperarm" + side
    return "pelvis"


W = {}
for v in fig.data.vertices:
    w = {}
    for g in v.groups:
        t = target(names[g.group])
        w[t] = w.get(t, 0.0) + g.weight
    tot = sum(w.values())
    W[v.index] = {k: x / tot for k, x in w.items() if tot > 0 and x / tot > 0.01}
for g in list(fig.vertex_groups):
    fig.vertex_groups.remove(g)
groups = {}
for vi, w in W.items():
    for k, x in w.items():
        if k not in groups:
            groups[k] = fig.vertex_groups.new(name=k)
        groups[k].add([vi], x, "REPLACE")
REP["groups"] = sorted(groups)

# ------------------------------------------------------------------ 5. export as an outfit
if DRAW_DEG is not None:
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
bpy.data.objects.remove(body)
fig.data.polygons.foreach_set("use_smooth", np.ones(len(fig.data.polygons), bool))
for i, u in enumerate(fig.data.uv_layers):
    u.name = "UVMap" if i == 0 else f"UV{i}"
while len(fig.data.uv_layers) > 1:
    fig.data.uv_layers.remove(fig.data.uv_layers[-1])
mo = bpy.data.materials.new("M_outfit")
mo.use_nodes = True
tn = mo.node_tree.nodes.new("ShaderNodeTexImage")
tn.image = img
mo.node_tree.links.new(tn.outputs["Color"], mo.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
fig.data.materials.clear()
fig.data.materials.append(mo)
for o in (arm, fig):
    o.scale = (100.0, 100.0, 100.0)
bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
fig.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
fig.parent = arm
am = fig.modifiers.new("Armature", "ARMATURE")
am.object = arm
if JOB.get("drawing_png"):
    # the layer's colour from its drawing: the 3D model's texture came out grey for a dark blue cloak (11-12); the
    # texture is scaled per channel so the layer's median colour is the drawing's (the layer's pixels: not the
    # background, not the light unsaturated figure)
    _di = bpy.data.images.load(JOB["drawing_png"])
    _w, _h = _di.size
    _px = np.empty(_w * _h * 4, np.float32)
    _di.pixels.foreach_get(_px)
    d_ = _px.reshape(_h, _w, 4)[..., :3]
    bgc = np.median(np.concatenate([d_[0], d_[-1], d_[:, 0], d_[:, -1]]), 0)
    fgm = np.linalg.norm(d_ - bgc, axis=2) > 0.12
    sat_ = (d_.max(2) - d_.min(2)) / np.maximum(d_.max(2), 1e-3)
    lay = fgm & ~((sat_ < 0.15) & (d_.mean(2) > 0.45))
    want = np.median(d_[lay], 0) if lay.sum() > 500 else None
    kept_col = np.median(col[~drop], 0)
    if want is not None:
        k_ = np.clip(want / np.maximum(kept_col, 1e-3), 0.2, 5.0)
        pxs = np.empty(len(img.pixels), np.float32)
        img.pixels.foreach_get(pxs)
        pxs = pxs.reshape(-1, 4)
        pxs[:, :3] = np.clip(pxs[:, :3] * k_, 0, 1)
        img.pixels.foreach_set(pxs.ravel())
        REP["colour_from_drawing"] = {"drawing": [round(float(x), 3) for x in want],
                                      "mesh": [round(float(x), 3) for x in kept_col], "gain": [round(float(x), 3) for x in k_]}
img.filepath_raw = os.path.join(OUT, "textures", "T_outfit_BaseColor.png")
img.file_format = "PNG"
img.save()
for name, rgb in (("Normal", (0.5, 0.5, 1.0)), ("RMA", (0.95, 0.0, 1.0))):
    im = bpy.data.images.new(f"T_outfit_{name}", 64, 64, alpha=False)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(np.tile(np.array(list(rgb) + [1.0], np.float32), 64 * 64))
    im.filepath_raw = os.path.join(OUT, "textures", f"T_outfit_{name}.png")
    im.file_format = "PNG"
    im.save()
bpy.ops.object.select_all(action="DESELECT")
fig.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, "outfit.fbx"), use_selection=True,
                         object_types={"ARMATURE", "MESH"}, apply_unit_scale=False, global_scale=1.0,
                         apply_scale_options="FBX_SCALE_NONE", axis_forward="-Z", axis_up="Y", add_leaf_bones=False,
                         use_armature_deform_only=False, primary_bone_axis="Y", secondary_bone_axis="X",
                         armature_nodetype="NULL", bake_anim=False, mesh_smooth_type="FACE", use_tspace=True,
                         path_mode="STRIP")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "layer.blend"), compress=True)
REP["total_s"] = round(time.time() - T0, 1)
json.dump(REP, open(os.path.join(OUT, "report.json"), "w"), indent=1)
print("LAYER", json.dumps(REP))
