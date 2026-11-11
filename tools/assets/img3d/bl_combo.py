"""Blender 4.5 (background): dress a MetaHuman body in a picture-to-3D figure's clothes. The figure (one fused mesh:
body, clothes, head, hands) is fitted over the MetaHuman body, its head and hands are cut away, the body's own hands
are carried in the outfit (a skin slot, as the character tools' outfits do), weights are copied from the body, and
the result is exported as an outfit FBX on the body's skeleton for the character tools' import (char_render.py
import_outfit: hide the body, the outfit follows it by leader pose; the MetaHuman head, neck and hair stay).

    blender -b --factory-startup -t 0 --python bl_combo.py -- JOB.json

JOB {"body_fbx": the MetaHuman body FBX exported from UE (centimetres, as the character tools export it),
     "head_fbx": the head FBX (its top sets the height) | null, "figure_glb": img3d.py's .glb, "out": folder,
     "neck_cut_m": cut height above the body's neck_02 joint [0.03], "keep_figure_hands": false,
     "skin_rgb": colour test for exposed skin [see is_skin], "preview_device": "HIP" substring | null}

Fit (the figure stands in a spread rest pose, the body in its A-pose bind pose):
1. scale the figure to the body's height (feet on the floor, pelvis over pelvis);
2. landmarks on the figure from its texture colour: hands (skin below the shoulders, out to the sides), shoes
   (lowest dark parts per side), the neck (skin ring under the chin);
3. pose the body's skeleton to the figure: each upper arm turned so the body's hand points at the figure's wrist, each
   thigh so the body's ankle points at the figure's shoe; the figure's arms are stretched along the arm so the sleeve
   end meets the body's wrist; the upper torso is shifted so the collar centres on the body's neck;
4. cut: everything above the neck cut, exposed skin above the collarbones, and (default) the figure's hands;
5. weights: the posed body's vertex groups copied to the figure (nearest face, interpolated), four per vertex;
6. that pose becomes the skeleton's rest pose (the outfit's bind pose; in the engine the outfit follows the body's
   bones by leader pose, so the bind pose is per mesh), and the body's hands (posed) are joined in as slot M_skin.
Outputs in out/: outfit.fbx, textures/T_outfit_{BaseColor,Normal,RMA}.png (the figure's colour; flat normal;
roughness 0.8, metallic 0, occlusion 1), combo.blend, report.json (landmarks, pose angles, stretch, cut counts,
times).
"""
import json
import math
import os
import shutil
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = JOB["out"]
os.makedirs(os.path.join(OUT, "textures"), exist_ok=True)
REP = {"times": {}}
T0 = time.time()


def mark(name, t):
    REP["times"][name] = round(time.time() - t, 2)


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


# ------------------------------------------------------------------ load
t = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
objs = import_fbx(JOB["body_fbx"])
arm = next(o for o in objs if o.type == "ARMATURE")
body = max((o for o in objs if o.type == "MESH"), key=lambda o: len(o.data.vertices))
for o in objs:
    if o.type == "MESH" and o is not body:
        bpy.data.objects.remove(o)
top_z = None
if JOB.get("head_fbx"):
    hnames = [o.name for o in import_fbx(JOB["head_fbx"])]
    flatten_empties()
    hobjs = [bpy.data.objects[n] for n in hnames if n in bpy.data.objects]
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
body.name = "body"
B = arm.data.bones


def J(n):
    return B[n].head_local.copy()


if top_z is None:
    top_z = J("head").z + 0.135
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
    if o not in (arm, body, fig):
        bpy.data.objects.remove(o)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
fig.name = "outfit"
mark("load", t)

# ------------------------------------------------------------------ colour per vertex (from the texture)
t = time.time()
img = None
for m in fig.data.materials:
    for n in (m.node_tree.nodes if m and m.use_nodes else []):
        if n.type == "TEX_IMAGE" and n.image is not None:
            img = n.image
            break
    if img:
        break
W, H = img.size
px = np.empty(W * H * 4, np.float32)
img.pixels.foreach_get(px)
px = px.reshape(H, W, 4)
me = fig.data
nl = len(me.loops)
uv = np.empty(nl * 2, np.float32)
me.uv_layers.active.data.foreach_get("uv", uv)
uv = uv.reshape(-1, 2)
lv = np.empty(nl, np.int64)
me.loops.foreach_get("vertex_index", lv)
xi = np.clip((uv[:, 0] * W).astype(int), 0, W - 1)
yi = np.clip((uv[:, 1] * H).astype(int), 0, H - 1)
lc = px[yi, xi, :3]
nv = len(me.vertices)
col = np.zeros((nv, 3))
cnt = np.zeros(nv)
np.add.at(col, lv, lc)
np.add.at(cnt, lv, 1)
col /= np.maximum(cnt, 1)[:, None]
co = np.empty(nv * 3)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
mark("colour", t)


def is_skin(c):
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    return (r > g) & (g > b) & (r - b > 0.10) & (r > 0.25)


skin = is_skin(col)

# ------------------------------------------------------------------ 1. scale and place
t = time.time()
zmin, zmax = co[:, 2].min(), co[:, 2].max()
s = top_z / (zmax - zmin)
co = (co - [0, 0, zmin]) * s
band = (co[:, 2] > 0.52 * top_z) & (co[:, 2] < 0.58 * top_z) & (np.abs(co[:, 0]) < 0.12)
pel = co[band].mean(0) if band.any() else co.mean(0)
co[:, 0] += J("pelvis").x - pel[0]
co[:, 1] += J("pelvis").y - pel[1]
REP["scale"] = round(float(s), 4)
REP["height_m"] = round(float(top_z), 3)

# ------------------------------------------------------------------ 2. landmarks
sh = {"l": J("upperarm_l"), "r": J("upperarm_r")}
LM = {}
for side, sg in (("l", 1), ("r", -1)):
    shx = sh[side].x
    hand = skin & (sg * co[:, 0] > abs(shx) + 0.12) & (co[:, 2] < sh[side].z - 0.05)
    if hand.sum() < 20:
        raise RuntimeError(f"no hand found on side {side}")
    hp = co[hand]
    d = np.linalg.norm(hp - np.array(sh[side]), axis=1)
    near = hp[d <= np.quantile(d, 0.08)]
    LM[f"wrist_{side}"] = near.mean(0)
    LM[f"palm_{side}"] = hp.mean(0)
    feet = (co[:, 2] < 0.06 * top_z) & (sg * co[:, 0] > 0)
    fp = co[feet]
    LM[f"ankle_{side}"] = np.array([fp[:, 0].mean(), np.quantile(fp[:, 1], 0.7), J(f"foot_{side}").z])
neck = skin & (co[:, 2] > J("clavicle_l").z) & (co[:, 2] < J("head").z) & (np.abs(co[:, 0] - J("neck_01").x) < 0.09)
LM["neck"] = co[neck].mean(0) if neck.sum() > 10 else np.array(J("neck_01"))
REP["landmarks"] = {k: [round(float(x), 3) for x in v] for k, v in LM.items()}
mark("landmarks", t)

# ------------------------------------------------------------------ 3. pose the body to the figure
t = time.time()
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")


def turn(bone, frm, to):
    """Rotate pose bone `bone` about its head (world space) so that direction frm becomes to."""
    pb = arm.pose.bones[bone]
    q = Vector(frm).normalized().rotation_difference(Vector(to).normalized())
    hd = pb.head.copy()
    M = Matrix.Translation(hd) @ q.to_matrix().to_4x4() @ Matrix.Translation(-hd)
    pb.matrix = M @ pb.matrix
    bpy.context.view_layer.update()
    return math.degrees(q.angle)


def pose_head(n):
    return arm.pose.bones[n].head.copy()


REP["pose_deg"] = {}
for side in ("l", "r"):
    a = turn(f"upperarm_{side}", pose_head(f"hand_{side}") - pose_head(f"upperarm_{side}"),
             Vector(LM[f"wrist_{side}"]) - pose_head(f"upperarm_{side}"))
    REP["pose_deg"][f"upperarm_{side}"] = round(a, 1)
    a = turn(f"thigh_{side}", pose_head(f"foot_{side}") - pose_head(f"thigh_{side}"),
             Vector(LM[f"ankle_{side}"]) - pose_head(f"thigh_{side}"))
    REP["pose_deg"][f"thigh_{side}"] = round(a, 1)
bpy.ops.object.mode_set(mode="OBJECT")
# stretch the figure's arms along the arm so the sleeve end meets the body's (posed) wrist
REP["arm_stretch"] = {}
for side, sg in (("l", 1), ("r", -1)):
    S0 = np.array(pose_head(f"upperarm_{side}"))
    wr = np.array(LM[f"wrist_{side}"])
    axis = (wr - S0) / np.linalg.norm(wr - S0)
    k = np.linalg.norm(np.array(pose_head(f"hand_{side}")) - S0) / np.linalg.norm(wr - S0)
    proj = (co - S0) @ axis
    L = np.linalg.norm(wr - S0)
    sel = (sg * co[:, 0] > sg * S0[0]) & (proj > 0)
    ramp = np.clip((proj - 0.15 * L) / (0.45 * L), 0, 1) * sel
    co += ((k - 1) * proj * ramp)[:, None] * axis[None, :]
    REP["arm_stretch"][side] = round(float(k), 3)
# centre the collar on the body's neck
dn = np.array(J("neck_01")) - LM["neck"]
dn[2] = 0
w = np.clip((co[:, 2] - 0.62 * top_z) / (0.16 * top_z), 0, 1)
co += w[:, None] * dn[None, :]
REP["neck_shift_m"] = [round(float(x), 3) for x in dn]
fig.data.vertices.foreach_set("co", co.ravel())
fig.data.update()
mark("fit", t)

# ------------------------------------------------------------------ 4. cut head and hands
t = time.time()
cut_z = J("neck_02").z + float(JOB.get("neck_cut_m", 0.03))
gone = co[:, 2] > cut_z
gone |= skin & (co[:, 2] > J("clavicle_l").z - 0.02) & (np.abs(co[:, 0] - J("neck_01").x) < 0.12)
if not JOB.get("keep_figure_hands"):
    for side, sg in (("l", 1), ("r", -1)):
        S0 = np.array(pose_head(f"upperarm_{side}"))
        hw = np.array(pose_head(f"hand_{side}"))
        axis = (hw - S0) / np.linalg.norm(hw - S0)
        p = (co - S0) @ axis
        radial = np.linalg.norm((co - S0) - p[:, None] * axis[None, :], axis=1)
        gone |= (p > np.linalg.norm(hw - S0) - 0.01) & (radial < 0.16) & (sg * co[:, 0] > 0)
bm = bmesh.new()
bm.from_mesh(fig.data)
bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(gone)[0]], context="VERTS")
# crumbs the cut leaves (finger tips, hair ends): islands under 300 vertices (the .glb splits vertices at UV seams:
# weld them first, the UVs live on the loops)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
bm.verts.index_update()
seen, crumbs = set(), []
for v0 in bm.verts:
    if v0.index in seen:
        continue
    isl, stack = [], [v0]
    seen.add(v0.index)
    while stack:
        v = stack.pop()
        isl.append(v)
        for e in v.link_edges:
            w = e.other_vert(v)
            if w.index not in seen:
                seen.add(w.index)
                stack.append(w)
    if len(isl) < 300:
        crumbs += isl
bmesh.ops.delete(bm, geom=crumbs, context="VERTS")
bm.to_mesh(fig.data)
bm.free()
REP["cut_verts"] = int(gone.sum())
REP["crumb_verts"] = len(crumbs)
mark("cut", t)

# ------------------------------------------------------------------ 5. weights from the posed body
t = time.time()
for g in body.vertex_groups:
    fig.vertex_groups.new(name=g.name)
dt = fig.modifiers.new("dt", "DATA_TRANSFER")
dt.object = body
dt.use_vert_data = True
dt.data_types_verts = {"VGROUP_WEIGHTS"}
dt.vert_mapping = "POLYINTERP_NEAREST"
dt.layers_vgroup_select_src = "ALL"
dt.layers_vgroup_select_dst = "NAME"
bpy.context.view_layer.objects.active = fig
bpy.ops.object.modifier_apply(modifier=dt.name)
bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
bpy.ops.object.mode_set(mode="OBJECT")
mark("weights", t)

# the body's hands, posed, as the outfit's skin slot
t = time.time()
hands = body.copy()
hands.data = body.data.copy()
bpy.context.scene.collection.objects.link(hands)
bpy.context.view_layer.objects.active = hands
for m in list(hands.modifiers):
    if m.type == "ARMATURE":
        bpy.ops.object.modifier_apply(modifier=m.name)
    else:
        hands.modifiers.remove(m)
HANDB = ("hand", "thumb", "index", "middle", "ring", "pinky")
gi = {g.index: g.name for g in hands.vertex_groups}
keep = np.zeros(len(hands.data.vertices), bool)
for v in hands.data.vertices:
    best = max(v.groups, key=lambda g: g.weight, default=None)
    if best is not None and gi[best.group].startswith(HANDB):
        keep[v.index] = True
bm = bmesh.new()
bm.from_mesh(hands.data)
bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(~keep)[0]], context="VERTS")
bm.to_mesh(hands.data)
bm.free()
hands.data.materials.clear()
ms = bpy.data.materials.new("M_skin")
hands.data.materials.append(ms)
REP["hand_verts"] = len(hands.data.vertices)
mark("hands", t)

# ------------------------------------------------------------------ 6. this pose becomes the rest pose; join; export
t = time.time()
bpy.data.objects.remove(body)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.pose.armature_apply(selected=False)
bpy.ops.object.mode_set(mode="OBJECT")
mo = bpy.data.materials.new("M_outfit")
mo.use_nodes = True
tn = mo.node_tree.nodes.new("ShaderNodeTexImage")             # for previews; the engine gets its own material
tn.image = img
mo.node_tree.links.new(tn.outputs["Color"], mo.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
fig.data.materials.clear()
fig.data.materials.append(mo)
for o in (fig, hands):
    for i, u in enumerate(o.data.uv_layers):
        u.name = "UVMap" if i == 0 else f"UV{i}"
    while len(o.data.uv_layers) > 1:
        o.data.uv_layers.remove(o.data.uv_layers[-1])
bpy.ops.object.select_all(action="DESELECT")
fig.select_set(True)
hands.select_set(True)
bpy.context.view_layer.objects.active = fig
bpy.ops.object.join()
# the character tools' outfit FBX carries centimetre values (no unit conversion on export or import): scale both
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
REP["outfit_verts"] = len(fig.data.vertices)
REP["outfit_tris"] = sum(len(p.vertices) - 2 for p in fig.data.polygons)
# textures for the character tools' baked material
img.filepath_raw = os.path.join(OUT, "textures", "T_outfit_BaseColor.png")
img.file_format = "PNG"
img.save()
for name, rgb in (("Normal", (0.5, 0.5, 1.0)), ("RMA", (0.8, 0.0, 1.0))):
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
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "combo.blend"), compress=True)
mark("export", t)
REP["total_s"] = round(time.time() - T0, 1)
json.dump(REP, open(os.path.join(OUT, "report.json"), "w"), indent=1)
print("COMBO", json.dumps(REP))
