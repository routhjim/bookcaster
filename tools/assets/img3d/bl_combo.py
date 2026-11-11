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
# mode "statue": the whole figure (its own head and hands) rigged to the body's skeleton, the body and head hidden
# under it; mode "outfit" (default): its clothes only, on the MetaHuman head and hands
STATUE = JOB.get("mode") == "statue"
if STATUE:
    JOB.setdefault("keep_figure_hands", True)
    JOB.setdefault("hair_cap", False)
    JOB.setdefault("head_ease_m", 0.025)
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
    mw = hm.matrix_world.copy()
    hm.parent = None
    hm.matrix_world = mw
    for o in hobjs:
        if o is not hm:
            bpy.data.objects.remove(o)
    for m in list(hm.modifiers):
        hm.modifiers.remove(m)
    hm.name = "head_ref"                # kept for the push-out: its neck and chest skin must stay under the outfit
flatten_empties()
bpy.ops.object.select_all(action="DESELECT")
HEAD = bpy.data.objects.get("head_ref")
for o in (arm, body, HEAD):
    if o is not None:
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
    if o not in (arm, body, fig, HEAD):
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
at = me.attributes.new("fcol", "FLOAT_COLOR", "POINT")     # the colour survives the cuts and welds below
at.data.foreach_set("color", np.c_[col, np.ones(nv)].astype(np.float32).ravel())
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
# a drawn figure's head is often large for its body: match the shirt collar's top (the highest white near the
# middle) to just under the body's neck_02 joint instead of the head tops, when the collar is found
mx = np.median(co[:, 0])
white = (col.min(1) > 0.55) & (col.max(1) - col.min(1) < 0.10) & (np.abs(co[:, 0] - mx) < 0.06 * (zmax - zmin))
white &= co[:, 2] > zmin + 0.7 * (zmax - zmin)
if white.sum() > 20 and JOB.get("match", "collar") == "collar":
    cz = np.quantile(co[white, 2], 0.98)
    s = (J("neck_02").z + float(JOB.get("collar_above_neck02_m", 0.0))) / (cz - zmin)
    REP["collar_match"] = {"figure_collar_frac": round(float((cz - zmin) / (zmax - zmin)), 3)}
co = (co - [0, 0, zmin]) * s
band = (co[:, 2] > 0.52 * top_z) & (co[:, 2] < 0.58 * top_z) & (np.abs(co[:, 0]) < 0.12)
pel = co[band].mean(0) if band.any() else co.mean(0)
co[:, 0] += J("pelvis").x - pel[0]
# front-back: centre the body's chest between the figure's chest front and back (the coat's tails make any centroid
# sit behind the body)
bco0 = np.array([v.co[:] for v in body.data.vertices])
cx = J("spine_05").x


def chest(P):
    m = (P[:, 2] > J("spine_03").z) & (P[:, 2] < J("spine_05").z) & (np.abs(P[:, 0] - cx) < 0.08)
    return np.quantile(P[m, 1], 0.02), np.quantile(P[m, 1], 0.98)


bf, bb = chest(bco0)
ff, fb = chest(co)
co[:, 1] += (bf + bb) / 2 - (ff + fb) / 2
REP["chest_depth_m"] = {"body": round(float(bb - bf), 3), "figure": round(float(fb - ff), 3)}
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
    radial = np.linalg.norm((co - S0) - proj[:, None] * axis[None, :], axis=1)
    sel = (sg * co[:, 0] > sg * S0[0]) & (proj > 0) & (radial < 0.16)     # the arm only, not the legs below it
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
cut_z = J("neck_02").z + float(JOB.get("neck_cut_m", 0.01))
# the head goes by one clean tilted plane (below); here only the figure's own neck skin inside the collar (a tube
# round the neck axis) and, below, the hands
nr = np.linalg.norm(co[:, :2] - np.array(J("neck_01"))[:2], axis=1)
gone = skin & (co[:, 2] > J("clavicle_l").z - 0.02) & (nr < float(JOB.get("neck_tube_m", 0.085)))
if STATUE:
    gone[:] = False
if not JOB.get("keep_figure_hands"):
    for side, sg in (("l", 1), ("r", -1)):
        S0 = np.array(pose_head(f"upperarm_{side}"))
        hw = np.array(pose_head(f"hand_{side}"))
        axis = (hw - S0) / np.linalg.norm(hw - S0)
        p = (co - S0) @ axis
        radial = np.linalg.norm((co - S0) - p[:, None] * axis[None, :], axis=1)
        gone |= (p > np.linalg.norm(hw - S0) - 0.01) & (radial < 0.16) & (sg * co[:, 0] > 0)
tilt = math.radians(float(JOB.get("collar_tilt_deg", 20)))
above = (co[:, 2] - cut_z) * math.cos(tilt) - (co[:, 1] - J("neck_01").y) * math.sin(tilt) > 0
HAIR = None
if JOB.get("hair_cap", True):
    # the figure's own hair, kept for the back and sides of the head (dark, not skin, above the collar line)
    lum = col @ np.array([0.3, 0.59, 0.11])
    hair = above & ~skin & (lum < float(JOB.get("hair_lum_max", 0.22)))
    if hair.sum() > 500:
        HAIR = fig.copy()
        HAIR.data = fig.data.copy()
        bpy.context.scene.collection.objects.link(HAIR)
        hb = bmesh.new()
        hb.from_mesh(HAIR.data)
        hb.verts.ensure_lookup_table()
        bmesh.ops.delete(hb, geom=[hb.verts[i] for i in np.nonzero(~hair)[0]], context="VERTS")
        hb.to_mesh(HAIR.data)
        hb.free()
        HAIR["src_head_lo"] = co[above].min(0).tolist()
        HAIR["src_head_hi"] = co[above].max(0).tolist()
bm = bmesh.new()
bm.from_mesh(fig.data)
bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(gone)[0]], context="VERTS")
# the collar line: a plane through the neck at cut_z, tilted so it runs lower at the front (as a collar does)
pc = Vector((J("neck_01").x, J("neck_01").y, cut_z))
pn = Vector((0.0, -math.sin(tilt), math.cos(tilt)))
if not STATUE:
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5, plane_co=pc, plane_no=pn,
                           clear_outer=True)
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

# weights by region: only the sleeves follow the arms. A vertex is sleeve when it lies within a sleeve's reach of
# the arm's centre line, past the armhole, and nearer that line than the torso's surface; everything else (coat
# body, lapels, skirt, waistcoat) takes the weights of the body without its arms (nearest torso surface), so the coat
# hangs from the shoulders and spine and the armhole is the one place where the two meet (smoothed below).
from mathutils.bvhtree import BVHTree  # noqa: E402
ARMB = ("clavicle", "upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky")


def is_arm(n):
    return n.startswith(ARMB) and "toe" not in n     # the feet have index/middle/ring toes


fco = np.array([v.co[:] for v in fig.data.vertices])
torso = body.copy()
torso.data = body.data.copy()
bpy.context.scene.collection.objects.link(torso)
tg = {g.index: g.name for g in torso.vertex_groups}
bm = bmesh.new()
bm.from_mesh(torso.data)
dl = bm.verts.layers.deform.active
bm.verts.ensure_lookup_table()
drop = [v for v in bm.verts if v[dl] and is_arm(tg[max(v[dl].items(), key=lambda kv: kv[1])[0]])]
bmesh.ops.delete(bm, geom=drop, context="VERTS")
bm.to_mesh(torso.data)
bm.free()
dg = bpy.context.evaluated_depsgraph_get()
tbvh = BVHTree.FromObject(torso, dg)
d_torso = np.array([(tbvh.find_nearest(Vector(p))[3] or 9.0) for p in fco])
gname = {g.index: g.name for g in fig.vertex_groups}
armw = np.array([sum(g.weight for g in v.groups if is_arm(gname[g.group]) and not gname[g.group].startswith("clavicle"))
                 for v in fig.data.vertices])
near_sh = np.zeros(len(fco), bool)
for side in ("l", "r"):
    S0 = np.array(pose_head(f"upperarm_{side}"))
    hw = np.array(pose_head(f"hand_{side}"))
    L = np.linalg.norm(hw - S0)
    p = (fco - S0) @ ((hw - S0) / L)
    near_sh |= ((p < float(JOB.get("armhole_frac", 0.25)) * L) & (np.linalg.norm(fco - S0, axis=1) < 0.10)
                & (fco[:, 2] > S0[2] - 0.04))
OFF = float(JOB.get("sleeve_off_torso_m", 0.035))
armo = body.copy()
armo.data = body.data.copy()
bpy.context.scene.collection.objects.link(armo)
bm = bmesh.new()
bm.from_mesh(armo.data)
dl = bm.verts.layers.deform.active
bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not (v[dl] and is_arm(tg[max(v[dl].items(), key=lambda kv: kv[1])[0]])
                                                       and not tg[max(v[dl].items(), key=lambda kv: kv[1])[0]]
                                                       .startswith("clavicle"))], context="VERTS")
bm.to_mesh(armo.data)
bm.free()
abvh = BVHTree.FromObject(armo, bpy.context.evaluated_depsgraph_get())
d_arm = np.array([(abvh.find_nearest(Vector(p))[3] or 9.0) for p in fco])
bpy.data.objects.remove(armo)
# a sleeve vertex is nearer the arm's surface than the torso's (the sleeve's underside too), or clear of the torso
sleeve = (armw >= 0.3) & ((d_arm < d_torso) | (d_torso > OFF) | near_sh)
# below the armpit only the sleeve tube itself: within SLEEVE_R of the arm's centre line (the coat's side and front
# panels hang there in the spread pose and would otherwise ride up with a raised arm like a cape)
SLEEVE_R = float(JOB.get("sleeve_radius_m", 0.10))
tube = np.zeros(len(fco), bool)
for side in ("l", "r"):
    S0 = np.array(pose_head(f"upperarm_{side}"))
    hw = np.array(pose_head(f"hand_{side}"))
    L = np.linalg.norm(hw - S0)
    ax = (hw - S0) / L
    p = (fco - S0) @ ax
    rad = np.linalg.norm((fco - S0) - np.clip(p, 0, L)[:, None] * ax[None, :], axis=1)
    tube |= (rad < SLEEVE_R) & (p > 0)
below = fco[:, 2] < min(J("upperarm_l").z, J("upperarm_r").z) - 0.04
sleeve &= ~below | tube
REP["sleeve_verts"] = int(sleeve.sum())
mask = fig.vertex_groups.new(name="_coat")
mask.add([int(i) for i in np.nonzero(~sleeve)[0]], 1.0, "REPLACE")
d2 = fig.modifiers.new("dt2", "DATA_TRANSFER")
d2.object = torso
d2.use_vert_data = True
d2.data_types_verts = {"VGROUP_WEIGHTS"}
d2.vert_mapping = "POLYINTERP_NEAREST"
d2.layers_vgroup_select_src = "ALL"
d2.layers_vgroup_select_dst = "NAME"
d2.mix_mode = "REPLACE"
d2.vertex_group = "_coat"
bpy.context.view_layer.objects.active = fig
bpy.ops.object.modifier_apply(modifier=d2.name)
fig.vertex_groups.remove(fig.vertex_groups["_coat"])
bpy.data.objects.remove(torso)

# skirt rule: cloth below the waist that is off the legs (a coat's skirt, a long dress) does not split with the
# legs: it follows the pelvis by SKIRT_PELVIS and the nearer thigh by the rest (1.0 = hangs from the pelvis only)
SKIRT_PELVIS = float(JOB.get("skirt_pelvis", 0.6))
LEG_OFF = float(JOB.get("skirt_off_leg_m", 0.05))
legs = body.copy()
legs.data = body.data.copy()
bpy.context.scene.collection.objects.link(legs)
lgn = {g.index: g.name for g in legs.vertex_groups}
bm = bmesh.new()
bm.from_mesh(legs.data)
dl = bm.verts.layers.deform.active
bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not (v[dl] and lgn[max(v[dl].items(), key=lambda kv: kv[1])[0]]
                                                       .startswith(("thigh", "calf")))], context="VERTS")
bm.to_mesh(legs.data)
bm.free()
for m in list(legs.modifiers):
    if m.type != "ARMATURE":
        legs.modifiers.remove(m)
lbvh = BVHTree.FromObject(legs, bpy.context.evaluated_depsgraph_get())
d_leg = np.array([(lbvh.find_nearest(Vector(p))[3] or 9.0) for p in fco])
bpy.data.objects.remove(legs)
legd = []
for side in ("l", "r"):
    a_ = np.array(arm.pose.bones[f"thigh_{side}"].head)
    b_ = np.array(arm.pose.bones[f"foot_{side}"].head)
    ab = b_ - a_
    tt = np.clip(((fco - a_) @ ab) / (ab @ ab), 0, 1)
    legd.append(np.linalg.norm(fco - (a_ + tt[:, None] * ab[None, :]), axis=1))
legd = np.array(legd)
skirt = (fco[:, 2] < J("pelvis").z - 0.05) & (fco[:, 2] > 0.14) & (d_leg > LEG_OFF) & ~sleeve
for gname_ in [g.name for g in fig.vertex_groups]:
    fig.vertex_groups[gname_].remove([int(i) for i in np.nonzero(skirt)[0]])
for i in np.nonzero(skirt)[0]:
    (fig.vertex_groups.get("pelvis") or fig.vertex_groups.new(name="pelvis")).add([int(i)], SKIRT_PELVIS, "ADD")
    if SKIRT_PELVIS < 1:
        side = "l" if legd[0, i] < legd[1, i] else "r"
        tgn = f"thigh_{side}"
        (fig.vertex_groups.get(tgn) or fig.vertex_groups.new(name=tgn)).add([int(i)], 1 - SKIRT_PELVIS, "ADD")
REP["skirt_verts"] = int(skirt.sum())

# shoes are rigid: everything under the ankle follows its foot bone only
zank = float(JOB.get("shoe_top_m", 0.11))
fco = np.array([v.co[:] for v in fig.data.vertices])
low = np.nonzero(fco[:, 2] < zank)[0]
if len(low):
    names = [g.name for g in fig.vertex_groups]
    for gname_ in names:
        fig.vertex_groups[gname_].remove([int(i) for i in low])
    for side, sg in (("l", 1), ("r", -1)):
        ids = [int(i) for i in low if sg * (fco[i, 0] - J("pelvis").x) > 0]
        fig.vertex_groups[f"foot_{side}"].add(ids, 1.0, "REPLACE")
    REP["rigid_shoe_verts"] = len(low)

# nothing of the outfit inside the body (the head mesh's neck and chest skin follow the body's surface and would
# show through): every vertex nearer than EASE to the body, or inside it, goes out along the body's normal
from mathutils.bvhtree import BVHTree  # noqa: E402
if STATUE and HEAD is not None and JOB.get("statue_head_fit", False):
    # the statue's head must enclose the hidden head (whose face would show through): map the statue's head box
    # (above the collar line) onto the head mesh's box grown by the head ease, fading in over 6 cm below the line
    P = np.array([v.co[:] for v in fig.data.vertices])
    hv = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
    hv = hv[hv[:, 2] > J("neck_02").z + 0.02]
    he = float(JOB.get("head_ease_m", 0.025))
    mlo, mhi = hv.min(0) - he, hv.max(0) + he
    sd = (P[:, 2] - cut_z) * math.cos(tilt) - (P[:, 1] - J("neck_01").y) * math.sin(tilt)
    hm_ = sd > 0
    slo, shi = P[hm_].min(0), P[hm_].max(0)
    slo[2] = mlo[2] = cut_z
    if JOB.get("statue_head_uniform", True):
        # one scale for all axes (the drawn head keeps its shape), large enough to hold the hidden head's box,
        # centred on it in x and y, its chin at the collar line
        k = float(np.max((mhi - mlo) / np.maximum(shi - slo, 1e-6)))
        c_s, c_m = (slo + shi) / 2, (mlo + mhi) / 2
        tgt = (P - c_s) * k + c_m
        tgt[:, 2] = (P[:, 2] - slo[2]) * k + mlo[2]
    else:
        tgt = mlo + (P - slo) * (mhi - mlo) / np.maximum(shi - slo, 1e-6)
    w = np.clip((sd + 0.06) / 0.06, 0, 1)[:, None]
    P = P * (1 - w) + tgt * w
    fig.data.vertices.foreach_set("co", P.ravel())
    fig.data.update()
    REP["head_fit_scale"] = [round(float(x), 3) for x in (mhi - mlo) / np.maximum(shi - slo, 1e-6)]
dg = bpy.context.evaluated_depsgraph_get()
bvhs = [] if STATUE else [BVHTree.FromObject(o, dg) for o in (body, HEAD) if o is not None]   # statue: nothing under it
fco = np.array([v.co[:] for v in fig.data.vertices])
EASE_ALL = float(JOB.get("body_ease_m", 0.014))
HEAD_EASE = float(JOB.get("head_ease_m", 0.014))      # the statue's head must cover the hidden head and its hair
disp = np.zeros_like(fco)
zfeet = float(JOB.get("push_above_m", 0.14))          # shoes stay as drawn (the body's feet are bigger)
for i, p in enumerate(fco):
    if p[2] < zfeet:
        continue
    for bvh in bvhs:
        q = Vector(p) + Vector(disp[i])
        loc, nrm, _, dist = bvh.find_nearest(q, 0.08)
        if loc is None:
            continue
        s_ = (q - loc).dot(nrm)
        ease = HEAD_EASE if (STATUE and p[2] > J("neck_02").z) else EASE_ALL
        if s_ < ease:
            disp[i] = disp[i] + np.array(nrm) * (ease - s_)
fig.data.vertices.foreach_set("co", (fco + disp).ravel())
fig.data.update()
# shoulders: the drawn coat's shoulders stand off the body's narrower ones; pull what stands more than SH_GAP off the
# body back toward it (half the excess), fading out 22 cm from the shoulder joint
SH_GAP = float(JOB.get("shoulder_gap_m", 0.025))
fco2 = fco + disp
for side in (() if STATUE else ("l", "r")):
    S0 = np.array(pose_head(f"upperarm_{side}"))
    dd = np.linalg.norm(fco2 - S0, axis=1)
    zone = np.nonzero((dd < 0.22) & (fco2[:, 2] > S0[2] - 0.12))[0]
    for i in zone:
        q = Vector(fco2[i])
        loc, nrm, _, dist = bvhs[0].find_nearest(q, 0.3)
        if loc is None or dist <= SH_GAP:
            continue
        f = (1 - dd[i] / 0.22) * 0.5
        fco2[i] = fco2[i] + (np.array(loc) - fco2[i]) * f * (dist - SH_GAP) / dist
# the collar: the drawn coat's collar stands off a thinner neck and shows as a raised tab from behind; pull what
# stands more than COL_GAP off the body or head back to it, fading out 13 cm from the neck's axis
COL_GAP = float(JOB.get("collar_gap_m", 0.012))
if not STATUE and bvhs:
    ncx = np.array(J("neck_01"))[:2]
    rr = np.linalg.norm(fco2[:, :2] - ncx, axis=1)
    zone = np.nonzero((rr < 0.13) & (fco2[:, 2] > J("clavicle_l").z - 0.02))[0]
    for i in zone:
        q = Vector(fco2[i])
        best = None
        for bvh in bvhs:
            loc, nrm, _, dist = bvh.find_nearest(q, 0.15)
            if loc is not None and (best is None or dist < best[1]):
                best = (loc, dist)
        if best is None or best[1] <= COL_GAP:
            continue
        f = 0.8 * (1 - rr[i] / 0.13)
        fco2[i] = fco2[i] + (np.array(best[0]) - fco2[i]) * f * (best[1] - COL_GAP) / best[1]
disp = fco2 - fco
fig.data.vertices.foreach_set("co", (fco + disp).ravel())
fig.data.update()
# the cravat: the drawn one is crumpled; smooth its surface (light, unsaturated cloth in front of the neck)
fc = np.empty(len(fig.data.vertices) * 4, np.float32)
fig.data.attributes["fcol"].data.foreach_get("color", fc)
fc = fc.reshape(-1, 4)[:, :3]
nc2 = np.array(J("neck_01"))
P = fco + disp
cm = ((fc.min(1) > 0.30) & (fc.max(1) - fc.min(1) < 0.12) & (P[:, 2] > J("spine_04").z) & (P[:, 1] < nc2[1])
      & (np.abs(P[:, 0] - nc2[0]) < 0.10))
if cm.sum() > 50:
    ed = np.array([e.vertices[:] for e in fig.data.edges])
    deg = np.bincount(ed.ravel(), minlength=len(P)).astype(np.float64)
    for _ in range(int(JOB.get("cravat_smooth", 15))):
        acc = np.zeros_like(P)
        np.add.at(acc, ed[:, 0], P[ed[:, 1]])
        np.add.at(acc, ed[:, 1], P[ed[:, 0]])
        avg = acc / np.maximum(deg, 1)[:, None]
        P[cm] = 0.5 * P[cm] + 0.5 * avg[cm]
    disp = P - fco
    fig.data.vertices.foreach_set("co", P.ravel())
    fig.data.update()
REP["cravat_verts_smoothed"] = int(cm.sum())
REP["pushed_out_verts"] = int((np.linalg.norm(disp, axis=1) > 1e-4).sum())
REP["push_max_m"] = round(float(np.linalg.norm(disp, axis=1).max()), 3)

# soften the weight seams (sleeve / coat side panel, collar / shoulder) so cloth bends instead of tearing: a few
# rounds of averaging over mesh edges, in numpy over the groups in use (Blender's own smoothing op walks every
# group and took 150 s here)
t_s = time.time()
nvx = len(fig.data.vertices)
used = sorted({g.group for v in fig.data.vertices for g in v.groups})
col_of = {gi: k for k, gi in enumerate(used)}
Wm = np.zeros((nvx, len(used)), np.float32)
for v in fig.data.vertices:
    for g in v.groups:
        Wm[v.index, col_of[g.group]] = g.weight
ed = np.array([e.vertices[:] for e in fig.data.edges])
deg = np.bincount(ed.ravel(), minlength=nvx).astype(np.float32)
for _ in range(int(JOB.get("weight_smooth", 8))):
    acc = np.zeros_like(Wm)
    np.add.at(acc, ed[:, 0], Wm[ed[:, 1]])
    np.add.at(acc, ed[:, 1], Wm[ed[:, 0]])
    Wm = 0.5 * Wm + 0.5 * acc / np.maximum(deg, 1)[:, None]
# the armholes take extra rounds, so the coat's shoulder bends round the joint instead of folding into a bump
fP = np.array([v.co[:] for v in fig.data.vertices])
shm = np.zeros(nvx, bool)
for side in ("l", "r"):
    shm |= np.linalg.norm(fP - np.array(pose_head(f"upperarm_{side}")), axis=1) < float(JOB.get("armhole_smooth_m", 0.20))
for _ in range(int(JOB.get("armhole_smooth", 25))):
    acc = np.zeros_like(Wm)
    np.add.at(acc, ed[:, 0], Wm[ed[:, 1]])
    np.add.at(acc, ed[:, 1], Wm[ed[:, 0]])
    Wm[shm] = 0.5 * Wm[shm] + 0.5 * (acc / np.maximum(deg, 1)[:, None])[shm]
fz = np.array([v.co.z for v in fig.data.vertices])
for k, gi in enumerate(used):
    grp = fig.vertex_groups[gi]
    nz = np.nonzero((Wm[:, k] > 1e-3) & (fz >= zank))[0]          # rigid shoes keep their foot weight
    grp.remove([int(i) for i in np.nonzero((Wm[:, k] <= 1e-3) & (fz >= zank))[0]])
    for i in nz:
        grp.add([int(i)], float(Wm[i, k]), "REPLACE")
REP["weight_smooth_s"] = round(time.time() - t_s, 2)
bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
bpy.ops.object.mode_set(mode="OBJECT")
if STATUE:
    # the statue's head is rigid on the head bone above the collar line; its hands rigid on the hand bones past
    # the wrists (fused fingers, no finger bones)
    P = np.array([v.co[:] for v in fig.data.vertices])
    rig = {}
    hd = (P[:, 2] - cut_z) * math.cos(tilt) - (P[:, 1] - J("neck_01").y) * math.sin(tilt) > 0
    rig["head"] = np.nonzero(hd)[0]
    for side in ("l", "r"):
        S0 = np.array(pose_head(f"upperarm_{side}"))
        hw = np.array(pose_head(f"hand_{side}"))
        L = np.linalg.norm(hw - S0)
        ax = (hw - S0) / L
        p = (P - S0) @ ax
        rad = np.linalg.norm((P - S0) - p[:, None] * ax[None, :], axis=1)
        rig[f"hand_{side}"] = np.nonzero((p > L - 0.01) & (rad < 0.16) & ~hd)[0]
    names = [g.name for g in fig.vertex_groups]
    for bone, ids in rig.items():
        ids = [int(i) for i in ids]
        for gn in names:
            fig.vertex_groups[gn].remove(ids)
        (fig.vertex_groups.get(bone) or fig.vertex_groups.new(name=bone)).add(ids, 1.0, "REPLACE")
    REP["rigid_verts"] = {k: len(v) for k, v in rig.items()}
mark("weights", t)

# the body's hands, posed, as the outfit's skin slot
t = time.time()
hands = None
if STATUE:
    body_hands = False
else:
    body_hands = True
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
    if best is not None and gi[best.group].startswith(HANDB) and "toe" not in gi[best.group]:
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
if HAIR is not None and HEAD is not None:
    # fit the figure's hair cap to the head: its head box (above the collar line) onto the head mesh's box above
    # the neck, then out of the head by HAIR_EASE; the front (forehead and face) is dropped so the head's own
    # hairline shows; it follows the head bone rigidly
    hv = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
    hv = hv[hv[:, 2] > J("neck_02").z + 0.03]
    mlo, mhi = hv.min(0), hv.max(0)
    slo, shi = np.array(HAIR["src_head_lo"]), np.array(HAIR["src_head_hi"])
    slo[2] = max(slo[2], cut_z)
    P = np.array([v.co[:] for v in HAIR.data.vertices])
    P = mlo + (P - slo) * (mhi - mlo) / np.maximum(shi - slo, 1e-6)
    keep = (P[:, 1] > mlo[1] + float(JOB.get("hair_front_frac", 0.35)) * (mhi[1] - mlo[1])) | \
        (P[:, 2] > mhi[2] - float(JOB.get("hair_top_m", 0.035)))
    hbvh = BVHTree.FromObject(HEAD, bpy.context.evaluated_depsgraph_get())
    HE = float(JOB.get("hair_ease_m", 0.004))
    HMAX = float(JOB.get("hair_max_m", 0.012))
    for i, p in enumerate(P):
        loc, nrm, _, dist = hbvh.find_nearest(Vector(p), 0.1)
        if loc is not None:
            sd = (Vector(p) - loc).dot(nrm)
            if sd < HE:
                P[i] = P[i] + np.array(nrm) * (HE - sd)
            elif sd > HMAX:                     # seat it on the skull: no more than HMAX of hair off the head
                P[i] = np.array(loc) + np.array(nrm) * HMAX
    HAIR.data.vertices.foreach_set("co", P.ravel())
    # its colour: the drawn hair's dark blue-grey reads blue under warm light; point every hair face at the
    # darkest texel of the atlas (near black)
    sm = px[::8, ::8, :3].mean(2)
    yy, xx = np.unravel_index(np.argmin(sm + (px[::8, ::8, 3] < 0.5) * 9), sm.shape)
    duv = ((xx * 8 + 0.5) / W, (yy * 8 + 0.5) / H)
    for luv in HAIR.data.uv_layers.active.data:
        luv.uv = duv
    hb = bmesh.new()
    hb.from_mesh(HAIR.data)
    hb.verts.ensure_lookup_table()
    bmesh.ops.delete(hb, geom=[hb.verts[i] for i in np.nonzero(~keep)[0]], context="VERTS")
    hb.to_mesh(HAIR.data)
    hb.free()
    HAIR.vertex_groups.clear()
    hg = HAIR.vertex_groups.new(name="head")
    hg.add(list(range(len(HAIR.data.vertices))), 1.0, "REPLACE")
    REP["hair_cap_verts"] = len(HAIR.data.vertices)
    bpy.ops.object.select_all(action="DESELECT")
    fig.select_set(True)
    HAIR.select_set(True)
    bpy.context.view_layer.objects.active = fig
    bpy.ops.object.join()
elif HAIR is not None:
    bpy.data.objects.remove(HAIR)
bpy.data.objects.remove(body)
if HEAD is not None:
    bpy.data.objects.remove(HEAD)
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
if not body_hands:
    bpy.data.objects.remove(hands)
    hands = None
for o in [x for x in (fig, hands) if x is not None]:
    for i, u in enumerate(o.data.uv_layers):
        u.name = "UVMap" if i == 0 else f"UV{i}"
    while len(o.data.uv_layers) > 1:
        o.data.uv_layers.remove(o.data.uv_layers[-1])
if hands is not None:
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
