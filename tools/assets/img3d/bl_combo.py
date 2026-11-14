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


def stage_save(name):
    """JOB "stage_dir": a copy of the scene at this step (the clothes audit renders each one the same way)."""
    if JOB.get("stage_dir"):
        os.makedirs(JOB["stage_dir"], exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(JOB["stage_dir"], f"{name}.blend"), copy=True, compress=True)


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


FIT_POSE = JOB.get("fit_pose")
if FIT_POSE:
    # a garment generated in a pose (seated): the body is first posed to that pose's measured joints (the scene
    # tool's result: set frame, metres, with the mark and facing), and every joint the fit reads is the posed one;
    # the posed body becomes the outfit's bind pose at the end, as a stance variant
    _f = np.array(FIT_POSE["facing"], float)
    _f /= np.linalg.norm(_f)
    _left = np.array([-_f[1], _f[0]])
    _pos = np.array(FIT_POSE["position"], float)

    def _local(p):
        d = np.array(p[:2]) - _pos
        return Vector((float(d @ _left), float(-(d @ _f)), float(p[2])))
    _JT = {k: _local(v) for k, v in FIT_POSE["joints"].items()}
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    _PB = arm.pose.bones

    def _hd(n):
        return arm.matrix_world @ _PB[n].head

    def _turn(bone, frm, to):
        q = Vector(frm).normalized().rotation_difference(Vector(to).normalized())
        h = _hd(bone)
        _PB[bone].matrix = (Matrix.Translation(h) @ q.to_matrix().to_4x4() @ Matrix.Translation(-h)) @ _PB[bone].matrix
        bpy.context.view_layer.update()
    _mt = (_JT["thigh_l"] + _JT["thigh_r"]) / 2
    _mr = (_hd("thigh_l") + _hd("thigh_r")) / 2
    _PB["pelvis"].matrix = Matrix.Translation(_mt - _mr) @ _PB["pelvis"].matrix
    bpy.context.view_layer.update()
    _turn("pelvis", _hd("thigh_l") - _hd("thigh_r"), _JT["thigh_l"] - _JT["thigh_r"])
    for b_ in ("spine_01", "spine_02", "spine_03", "spine_04", "spine_05"):
        v0 = _hd("neck_01") - _hd(b_)
        q = v0.normalized().rotation_difference((_JT["neck_01"] - _hd(b_)).normalized())
        if q.angle / 3 > 1e-4:
            _turn(b_, v0, Matrix.Rotation(q.angle / 3, 4, q.axis) @ v0)
    for a_, b_ in (("neck_01", "head"), ("upperarm_l", "lowerarm_l"), ("lowerarm_l", "hand_l"),
                   ("upperarm_r", "lowerarm_r"), ("lowerarm_r", "hand_r"), ("thigh_l", "calf_l"), ("calf_l", "foot_l"),
                   ("thigh_r", "calf_r"), ("calf_r", "foot_r")):
        if a_ in _JT and b_ in _JT:
            _turn(a_, _hd(b_) - _hd(a_), _JT[b_] - _hd(a_))
    bpy.ops.object.mode_set(mode="OBJECT")
    REP["fit_pose_err_cm"] = {k: round((_hd(k) - _JT[k]).length * 100, 2) for k in _JT if k in _PB}

    def J(n):                                     # noqa: F811  the posed joints from here on
        return (arm.matrix_world @ arm.pose.bones[n].head).copy()
    if HEAD is not None:
        # the head mesh moves rigidly with the posed neck (its face bones are not on the body's skeleton); the
        # collar's measures and the push-out read it there
        _pb = arm.pose.bones["neck_01"]
        _M = (arm.matrix_world @ _pb.matrix) @ (arm.matrix_world @ _pb.bone.matrix_local).inverted()
        HEAD.data.transform(HEAD.matrix_world.inverted() @ _M @ HEAD.matrix_world)
        HEAD.data.update()
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
# what each vertex IS (hair, face, bare skin, a garment), from the trained parser's labels of this same figure
# (parse_figure.py: the drawing and twelve views of the mesh, voted): read by nearest point, the glb's y-up
# coordinates turned to these. Told by colour alone, brown hair passed as skin, a ponytail and a beard counted as a
# collar (and the collar was lifted to the jaw for it), and a tan garment was eaten away as skin.
LAB_GROUP = None           # per vertex: 0 unknown, 1 hair (and a hat), 2 face, 3 bare skin, 4 garment
LAB_RAW = None             # per vertex: the parser's own class (2 head hair, 1 face, 18 facial hair, ...)
if JOB.get("labels") and os.path.exists(JOB["labels"]):
    from mathutils.kdtree import KDTree as _KD
    _lz = np.load(JOB["labels"], allow_pickle=False)
    _lp = _lz["points"]
    _lp = np.stack([_lp[:, 0], -_lp[:, 2], _lp[:, 1]], 1)
    _kd = _KD(len(_lp))
    for _i, _q in enumerate(_lp):
        _kd.insert(_q, _i)
    _kd.balance()
    _near = np.array([_kd.find(_q)[1:] for _q in co])
    LAB_GROUP = _lz["groups"][_near[:, 0].astype(int)]
    LAB_RAW = _lz["labels"][_near[:, 0].astype(int)]
    REP["labels"] = {"from": JOB["labels"], "max_gap_m": round(float(_near[:, 1].max()), 4),
                     "hair": int((LAB_GROUP == 1).sum()), "face": int((LAB_GROUP == 2).sum()),
                     "skin": int((LAB_GROUP == 3).sum()), "garment": int((LAB_GROUP == 4).sum())}
mark("colour", t)


def is_skin(c):
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    return (r > g) & (g > b) & (r - b > 0.10) & (r > 0.25)


skin = is_skin(col)
if LAB_GROUP is not None:
    # skin by colour, but never what the labels say is a garment or hair; and the face is skin whatever its colour
    REP["labels"]["skin_by_colour_on_garment_or_hair"] = int((skin & np.isin(LAB_GROUP, (1, 4))).sum())
    skin = (skin & ~np.isin(LAB_GROUP, (1, 4))) | (LAB_GROUP == 2)
HAIRY = (LAB_GROUP == 1) if LAB_GROUP is not None else np.zeros(len(co), bool)

# ------------------------------------------------------------------ 1. scale and place
t = time.time()
zmin, zmax = co[:, 2].min(), co[:, 2].max()
s = top_z / (zmax - zmin)
# a drawn figure's head is often large for its body: match the shirt collar's top (the highest white near the
# middle) to just under the body's neck_02 joint instead of the head tops, when the collar is found
mx = np.median(co[:, 0])
white = (col.min(1) > 0.55) & (col.max(1) - col.min(1) < 0.10) & (np.abs(co[:, 0] - mx) < 0.06 * (zmax - zmin))
white &= co[:, 2] > zmin + 0.7 * (zmax - zmin)
match = JOB.get("match", "neck")
if match == "neck":
    # the neck: the narrowest horizontal slice of the upper figure (head above is wider, collar and shoulders
    # below are wider; spread arms hang lower), matched to the body's neck_01 joint
    Hf = zmax - zmin
    # searched only near where the body's own neck sits as a fraction of its height (the drawing is made from a pose
    # map of this body's joints): long loose hair framing a face made the narrowest slice the top of the head
    # (0.95 of the height), the figure was scaled to put its head at the body's neck and its face stayed on the
    # garment (11-11)
    exp_ = 0.5 * (J("neck_01").z + J("neck_02").z) / top_z
    lo_, hi_ = max(0.74, exp_ - float(JOB.get("neck_search_below", 0.05))), min(0.95, exp_ + float(JOB.get("neck_search_above", 0.03)))
    zs_ = np.linspace(zmin + lo_ * Hf, zmin + hi_ * Hf, 31)
    wid = []
    for z_ in zs_:
        sl = co[np.abs(co[:, 2] - z_) < 0.004 * Hf]
        wid.append(np.ptp(sl[:, 0]) if len(sl) > 10 else 9.0)
    zn = zs_[int(np.argmin(wid))]
    s = 0.5 * (J("neck_01").z + J("neck_02").z) / (zn - zmin)        # the narrowest slice is mid-neck
    REP["neck_match"] = {"figure_neck_frac": round(float((zn - zmin) / Hf), 3), "width_frac": round(float(min(wid) / Hf), 3),
                         "expected_frac": round(float(exp_), 3), "window": [round(lo_, 3), round(hi_, 3)]}
elif white.sum() > 20 and match == "collar":
    cz = np.quantile(co[white, 2], 0.98)
    s = (J("neck_02").z + float(JOB.get("collar_above_neck02_m", 0.0))) / (cz - zmin)
    REP["collar_match"] = {"figure_collar_frac": round(float((cz - zmin) / (zmax - zmin)), 3)}
co = (co - [0, 0, zmin]) * s
band = (co[:, 2] > 0.52 * top_z) & (co[:, 2] < 0.58 * top_z) & (np.abs(co[:, 0]) < 0.12)
pel = co[band].mean(0) if band.any() else co.mean(0)
co[:, 0] += J("pelvis").x - pel[0]
# front-back: centre the body's chest between the figure's chest front and back (the coat's tails make any centroid
# sit behind the body)
bco0 = (np.array([v.co[:] for v in body.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
        if FIT_POSE else np.array([v.co[:] for v in body.data.vertices]))
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
    side_pts = (sg * co[:, 0] > abs(shx) + 0.12) & (co[:, 2] < sh[side].z - 0.05)
    # the hand is skin at the far end of the arm (a brown cloth passes the colour test too): within HAND_LEN of the
    # farthest point of that side from the shoulder
    dsh = np.linalg.norm(co - np.array(sh[side]), axis=1)
    cand_ = skin & side_pts
    far_ = np.quantile(dsh[cand_], 0.98) if cand_.sum() > 20 else 9.0
    hand = cand_ & (dsh > far_ - float(JOB.get("hand_len_m", 0.22)))
    if JOB.get("draw_deg") is not None and not FIT_POSE:
        # where the drawing put the hand: the pose map it was drawn from is this body's arm at the drawing's angle
        # from vertical, so the hand lies along that line from the shoulder (tan or brown cloth passes the skin
        # colour test: the far end of the "skin" was a wrist at the feet, 11-12); skin colour only narrows it
        th_ = math.radians(float(JOB["draw_deg"]))
        S_ = np.array(sh[side])
        L_ = float(np.linalg.norm(np.array(J(f"hand_{side}")) - np.array(J(f"upperarm_{side}"))))
        exp_ = S_ + np.array([sg * math.sin(th_), 0.0, -math.cos(th_)]) * L_
        dxz = np.linalg.norm((co - exp_)[:, [0, 2]], axis=1)
        nearh = dxz < float(JOB.get("hand_search_r_m", 0.12))
        hand = (nearh & skin) if (nearh & skin).sum() >= 20 else nearh
        REP.setdefault("hand_search", {})[side + "_expected"] = [round(float(v), 3) for v in exp_]
        if hand.sum() < 20:
            # nothing where the pose map put the hand: the drawing hung the arm at another angle (a child's drawn
            # 30 degrees under the map, 11-13). The arm's end is then the point of that side, out beyond the
            # shoulder and above mid-thigh, farthest from the shoulder in the front view; the hand is round it
            arm_ = (sg * co[:, 0] > abs(shx) + 0.12) & (co[:, 2] > J("pelvis").z - 0.25)
            if arm_.sum() >= 20:
                dxz_s = np.linalg.norm((co - S_)[:, [0, 2]], axis=1)
                end_ = co[arm_][dxz_s[arm_] >= np.quantile(dxz_s[arm_], 0.98)].mean(0)
                nearh = np.linalg.norm((co - end_)[:, [0, 2]], axis=1) < float(JOB.get("hand_search_r_m", 0.12))
                hand = (nearh & skin) if (nearh & skin).sum() >= 20 else nearh
                REP["hand_search"][side + "_found_deg"] = round(math.degrees(math.atan2(
                    abs(end_[0] - S_[0]), S_[2] - end_[2])), 1)
    if FIT_POSE:
        # a posed figure (hands in the lap): the hand is the skin round the posed body's hand, not the far end of
        # a spread arm
        hj = np.array(J(f"hand_{side}"))
        hand = skin & (np.linalg.norm(co - hj, axis=1) < float(JOB.get("hand_len_m", 0.22)) * 0.6)
    REP.setdefault("hand_search", {})[side] = {"skin_side": int(cand_.sum()), "far_m": round(float(far_), 3),
                                               "hand": int(hand.sum())}
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
for side in (() if FIT_POSE else ("l", "r")):
    a = turn(f"upperarm_{side}", pose_head(f"hand_{side}") - pose_head(f"upperarm_{side}"),
             Vector(LM[f"wrist_{side}"]) - pose_head(f"upperarm_{side}"))
    REP["pose_deg"][f"upperarm_{side}"] = round(a, 1)
    if JOB.get("turn_legs", True):       # a floor-length skirt hides the shoes: its hem is no ankle
        a = turn(f"thigh_{side}", pose_head(f"foot_{side}") - pose_head(f"thigh_{side}"),
                 Vector(LM[f"ankle_{side}"]) - pose_head(f"thigh_{side}"))
        REP["pose_deg"][f"thigh_{side}"] = round(a, 1)
bpy.ops.object.mode_set(mode="OBJECT")
# stretch the figure's arms along the arm so the sleeve end meets the body's (posed) wrist
REP["arm_stretch"] = {}
for side, sg in (() if FIT_POSE else (("l", 1), ("r", -1))):
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
cap = float(JOB.get("neck_shift_max_m", 0.03))         # a figure leaning its head forward must not shear its coat
if np.linalg.norm(dn) > cap:
    dn = dn * cap / np.linalg.norm(dn)
w = np.clip((co[:, 2] - 0.62 * top_z) / (0.16 * top_z), 0, 1)
co += w[:, None] * dn[None, :]
REP["neck_shift_m"] = [round(float(x), 3) for x in dn]
fig.data.vertices.foreach_set("co", co.ravel())
fig.data.update()
mark("fit", t)
stage_save("s1_scaled_placed")

# the neckline, measured on the generated figure before any cut or push (a beard hides a neckline from a picture,
# not from the mesh; after the fit the cloth hugs every neck): a collar is cloth all round the neck (a beard is only
# in front of it, long hair only behind and beside it), so at mid-neck height the ring just outside the neck's radius
# is read sector by sector: cloth in most sectors = up the neck to the chin; otherwise, just under the shoulder line,
# the figure's own skin in front = a low open neckline, cloth there = a neckline at the base of the neck
if not STATUE and HEAD is not None:
    try:
        _nk0 = np.array(J("neck_01"))
        _hv0 = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
        _ring0 = np.abs(_hv0[:, 2] - _nk0[2]) < 0.012
        _nr0 = float(np.median(np.linalg.norm(_hv0[_ring0, :2] - _nk0[:2], axis=1))) if _ring0.sum() > 20 else 0.06
        _zsh0 = float(J("clavicle_l").z)
        _chin0 = float(J("head").z) - 0.04
        _zmid = _zsh0 + 0.5 * (_chin0 - _zsh0)
        _rr0 = np.linalg.norm(co[:, :2] - _nk0[:2], axis=1)
        _an0 = np.arctan2(co[:, 1] - _nk0[1], co[:, 0] - _nk0[0])
        _sec0 = ((_an0 + math.pi) / (2 * math.pi) * 16).astype(int) % 16
        _band0 = (np.abs(co[:, 2] - _zmid) < 0.012) & (_rr0 > _nr0 - 0.01) & (_rr0 < _nr0 + 0.05)
        # (cloth: what the labels call a garment; without labels, what is not skin-coloured)
        _is_cloth0 = (LAB_GROUP == 4) if LAB_GROUP is not None else ~skin
        _cloth_secs = sorted({int(k) for k in _sec0[_band0 & _is_cloth0]})
        _low = (np.abs(co[:, 2] - (_zsh0 - 0.03)) < 0.012) & (co[:, 1] < _nk0[1]) & (np.abs(co[:, 0] - _nk0[0]) < _nr0 + 0.02)
        _skin_front = float((skin[_low]).mean()) if _low.sum() > 10 else 0.0
        # skin on the chest below the shoulder line rules out a collar up to the chin (a beard and long hair also
        # make a ring round the neck: the open-necked tunic read 15 of 16 sectors)
        if _skin_front > 0.4:
            _cls0 = "open"
        elif len(_cloth_secs) >= 10:
            _cls0 = "chin"
        else:
            _cls0 = "base"
        REP["neckline_figure"] = {"class": _cls0, "cloth_sectors_at_mid_neck": len(_cloth_secs),
                                  "skin_share_in_front_below_shoulders": round(_skin_front, 3)}
    except Exception as _e:
        REP["neckline_figure"] = {"error": str(_e)}

# ------------------------------------------------------------------ 4. cut head and hands
SKL_SURE = None
CUFF = {}          # side -> (shoulder, arm axis, cuff station): the body's own forearm shows beyond a short sleeve


def _skin_like():
    global SKL_SURE
    hm_ = np.zeros(len(co), bool)
    for s_ in ("l", "r"):
        S_ = np.array(pose_head(f"upperarm_{s_}"))
        E_ = np.array(pose_head(f"hand_{s_}"))
        a_ = (E_ - S_) / np.linalg.norm(E_ - S_)
        pp = (co - S_) @ a_
        rr = np.linalg.norm((co - S_) - pp[:, None] * a_[None, :], axis=1)
        hm_ |= (pp > np.linalg.norm(E_ - S_) + 0.02) & (rr < 0.1)
    cm_ = ((co[:, 2] > J("spine_03").z) & (co[:, 2] < J("spine_05").z)
           & (np.abs(co[:, 0] - J("spine_05").x) < 0.12) & (np.abs(co[:, 0] - J("spine_05").x) > 0.04))
    if hm_.sum() < 30 or cm_.sum() < 30:
        SKL_SURE = skin
        return skin, None
    sk_ref, cl_ref = np.median(col[hm_], 0), np.median(col[cm_], 0)
    ds_, dc_ = np.linalg.norm(col - sk_ref, axis=1), np.linalg.norm(col - cl_ref, axis=1)
    like = ds_ < dc_
    SKL_SURE = ds_ < float(JOB.get("skin_sure_ratio", 0.5)) * dc_      # clearly the skin's colour, not between
    return like, {"skin": [round(float(x), 3) for x in sk_ref], "cloth": [round(float(x), 3) for x in cl_ref],
                  "apart": round(float(np.linalg.norm(sk_ref - cl_ref)), 3)}



t = time.time()
cut_z = J("neck_02").z + float(JOB.get("neck_cut_m", 0.01))
# what counts as the figure's own skin round the neck: skin-coloured AND nearer its own hands' colour than its own
# cloth's (a tan or brown garment passes any fixed skin test: a man's coveralls were eaten away at the collar, the
# shoulders and the nape as "neck skin", and what was left of his collar hung in strips)
_like_n, _ref_n = _skin_like()
if _ref_n is not None:
    REP["neck_skin_ref"] = _ref_n
    skin_n = skin & _like_n
else:
    skin_n = skin
# the head goes by one clean tilted plane (below); here only the figure's own neck skin inside the collar (a tube
# round the neck axis) and, below, the hands
nr = np.linalg.norm(co[:, :2] - np.array(J("neck_01"))[:2], axis=1)
if JOB.get("neck_cone", True):
    # the figure's own neck and head go by shape, not by colour: everything inside a surface round the body's own
    # neck (its radius there, measured on the MetaHuman's head mesh, plus a finger's width) that opens out above
    # the neck's base to take in the jaw, the beard and the head. Whatever colour it is, what lies inside is the
    # generated head and neck; whatever colour it is, what lies outside is cloth and stays (by colour, a tan
    # garment lost its collar, shoulders and nape as "skin", and a beard and a ponytail stayed as "cloth")
    _n1, _hd_ = np.array(J("neck_01")), np.array(J("head"))
    _ax = (_hd_ - _n1) / np.linalg.norm(_hd_ - _n1)
    _t = (co - _n1) @ _ax
    _rr = np.linalg.norm((co - _n1) - _t[:, None] * _ax[None, :], axis=1)
    _rn = 0.065
    if HEAD is not None:
        _hp = np.array([(HEAD.matrix_world @ v_.co)[:] for v_ in HEAD.data.vertices])
        _ht = (_hp - _n1) @ _ax
        _hr = np.linalg.norm((_hp - _n1) - _ht[:, None] * _ax[None, :], axis=1)
        # (mid-neck, between the neck's two joints: at its base the head mesh already spreads over the shoulders
        # and read 16 cm)
        _mid = float((np.array(J("neck_02")) - _n1) @ _ax) * 0.75
        _band = np.abs(_ht - _mid) < 0.012
        if _band.sum() > 20:
            _rn = float(min(0.085, np.quantile(_hr[_band], 0.75)))
    _R = _rn + float(JOB.get("neck_clear_m", 0.02)) + np.maximum(_t, 0.0) * float(JOB.get("neck_cone_slope", 0.9))
    gone = (_t > (J("clavicle_l").z - 0.02 - _n1[2])) & (_rr < _R)
    # and what lies on the body's own head and neck: the generated figure's flesh (a chin, a beard, the skin of an
    # open neckline) sits where the MetaHuman's does, inside its head mesh or on its surface; cloth stands off it
    _flesh = 0
    if HEAD is not None and JOB.get("neck_flesh", False):   # (off: on a thick-necked man it took the collar's back too)
        from mathutils.bvhtree import BVHTree as _BVn  # noqa: E402
        _hbn = _BVn.FromObject(HEAD, bpy.context.evaluated_depsgraph_get())
        _off = float(JOB.get("neck_flesh_off_m", 0.008))
        # (only the neck's own column, from just under its base up: over the shoulders and chest the head mesh
        # runs on under the cloth, and by this test alone the whole collar and yoke went)
        _cand = np.nonzero((_t > -0.03) & (_rr < _rn + float(JOB.get("neck_flesh_reach_m", 0.06))) & ~gone)[0]
        for i in _cand:
            loc_, nrm_, _, d_ = _hbn.find_nearest(Vector(co[i]), 0.06)
            if loc_ is None:
                continue
            if (Vector(co[i]) - loc_).dot(nrm_) < _off:          # inside the head mesh, or within the offset of it
                gone[i] = True
                _flesh += 1
    # the figure's own neck and hair where they stand off the body's (a head carried forward, a ponytail down the
    # back): by colour, but only where the figure's skin is clearly not its cloth's colour, and its hair clearly
    # darker than its cloth
    _col_n = 0
    if _ref_n is not None and _ref_n["apart"] >= float(JOB.get("neck_colour_apart", 0.12)) and SKL_SURE is not None:
        _own = (co[:, 2] > J("clavicle_l").z - 0.02) & (nr < 0.20) & SKL_SURE & ~gone
        gone |= _own
        _col_n += int(_own.sum())
    if _ref_n is not None:
        _cl_l = float(np.dot(_ref_n["cloth"], [0.3, 0.59, 0.11]))
        _lum = col @ np.array([0.3, 0.59, 0.11])
        if _cl_l > 0.2:                       # (a dark garment's back is not hair)
            _hair = ((co[:, 2] > J("spine_03").z) & (nr < 0.22) & (co[:, 1] > _n1[1] - 0.02) & (_lum < 0.08) & ~gone)
            gone |= _hair
            _col_n += int(_hair.sum())
    # the figure's own hair wherever it hangs round the neck, over the shoulders or down the back (the parser's
    # labels): the head and its hair are the MetaHuman's
    _hairy = HAIRY & (co[:, 2] > J("spine_01").z) & ~gone
    if LAB_GROUP is not None:                # and its face (a beard, a chin carried forward of the cone)
        _hairy |= (LAB_GROUP == 2) & ~gone
    gone |= _hairy
    REP["neck_cone"] = {"neck_r_m": round(_rn, 3), "verts": int(gone.sum()), "on_the_body": _flesh, "by_colour": _col_n,
                        "hair": int(_hairy.sum())}
else:
    gone = skin_n & (co[:, 2] > J("clavicle_l").z - 0.02) & (nr < float(JOB.get("neck_tube_m", 0.12)))
    # the figure's own hair and head left under the collar line (a long face, hair at the nape): dark or skin above
    # the body's upper neck, near the neck's axis
    lum_ = col @ np.array([0.3, 0.59, 0.11])
    gone |= (co[:, 2] > J("neck_02").z - 0.01) & (nr < 0.13) & (skin_n | (lum_ < float(JOB.get("hair_lum_max", 0.22))))
    if HEAD is not None and gone.any():
        # only what lies on or inside the MetaHuman's own neck and head: a brown cloth, a dark collar or a tie near the
        # neck passes the colour tests too
        from mathutils.bvhtree import BVHTree as _BV  # noqa: E402
        _hb = _BV.FromObject(HEAD, bpy.context.evaluated_depsgraph_get())
        _near = float(JOB.get("neck_skin_near_m", 0.035))
        for i in np.nonzero(gone)[0]:
            loc_, nrm_, _, d_ = _hb.find_nearest(Vector(co[i]), 0.5)
            if loc_ is not None and d_ > _near and (Vector(co[i]) - loc_).dot(nrm_) > 0:
                gone[i] = False
# the web: generated figures often join the sleeve's underside to the garment's side well below a real armpit (one
# sheet of cloth from the elbow up); it stretches into a wing when the arm rises and hides the arm's outline when it
# falls. Between the height where the arm really comes free (sliced in the front plane, as mesh_gate.py does) and
# the body's armpit, cloth that lies outside the torso's own side (+ WEB_EASE) and outside the arm's tube (below
# the arm) is cut away; the sleeve and the garment then meet only at the armhole.
if JOB.get("web_cut", False) and not STATUE:   # experimental: cut 0 faces on the tested figures (11-11)
    WEB_EASE = float(JOB.get("web_ease_m", 0.03))
    TUBE = float(JOB.get("web_tube_m", 0.065))
    cxw = J("pelvis").x
    beval0 = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    BPw = np.array([v.co[:] for v in beval0.data.vertices])
    bgw = {g.index: g.name for g in body.vertex_groups}
    barmw = np.array([bool(v.groups) and bgw[max(v.groups, key=lambda g: g.weight).group].startswith(
        ("clavicle", "upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky")) for v in body.data.vertices])
    BTw = BPw[~barmw]
    REP["web_cut"] = {}
    for side, sg in (("l", 1), ("r", -1)):
        S0 = np.array(pose_head(f"upperarm_{side}"))
        wr = np.array(LM[f"wrist_{side}"])
        ax = (wr - S0) / np.linalg.norm(wr - S0)
        z_pit = S0[2] - float(JOB.get("armpit_below_shoulder_m", 0.07))
        # free height: walking down from the armpit, the first 1 cm slice where the side's outermost cloth run is
        # apart from the run holding the body's centre line
        cell = 0.006
        z_free = None
        for z_ in np.arange(z_pit, S0[2] - 0.45, -0.01):
            sl = co[np.abs(co[:, 2] - z_) < 0.005]
            if len(sl) < 20:
                continue
            xs = np.unique(np.floor((sl[:, 0] - cxw) / cell).astype(int))
            xs = xs[sg * xs > 0] if sg > 0 else xs[xs < 0]
            if len(xs) == 0:
                continue
            o = np.sort(sg * xs)
            if (np.diff(o) > 2).any() and o[0] <= 1:
                z_free = float(z_)
                break
        if z_free is None or z_free >= z_pit - 0.01:
            REP["web_cut"][side] = {"free_z": z_free, "armpit_z": round(float(z_pit), 3), "cut": 0}
            continue
        zz = np.arange(z_free - 0.02, z_pit + 0.03, 0.01)
        ext = np.array([np.quantile(sg * (BTw[np.abs(BTw[:, 2] - z_) < 0.015, 0] - cxw), 0.99)
                        if (np.abs(BTw[:, 2] - z_) < 0.015).sum() > 10 else 0.15 for z_ in zz])
        # the arm's inner edge in each horizontal slice: where its axis crosses that height, less the sleeve's
        # half width in the slice (its radius TUBE over the axis' vertical component)
        az_ = max(abs(float(ax[2])), 0.3)
        x_axis = sg * (S0[0] - cxw) + (S0[2] - co[:, 2]) / az_ * np.hypot(ax[0], ax[1])
        x_inner = x_axis - TUBE / az_
        lat = sg * (co[:, 0] - cxw)
        web = ((co[:, 2] > z_free) & (co[:, 2] < z_pit) & (lat > np.interp(co[:, 2], zz, ext) + WEB_EASE)
               & (lat < x_inner) & ~skin)
        gone |= web
        REP["web_cut"][side] = {"free_z": round(z_free, 3), "armpit_z": round(float(z_pit), 3), "cut": int(web.sum())}
COLLAR_RGB = None
if JOB.get("collar_build") and not STATUE:
    # the generated shirt front, cravat and shirt collar go (bl_collar.py builds clean ones): cloth clearly lighter
    # than the coat (the coat's colour: the median over the shoulders) in front of the upper chest and round the
    # neck. Their colours, light and dark halves of what goes, colour the built pieces.
    lum_c = col @ np.array([0.3, 0.59, 0.11])
    shz_ = np.zeros(len(co), bool)
    for side in ("l", "r"):
        shz_ |= np.linalg.norm(co - np.array(J(f"upperarm_{side}")), axis=1) < 0.15
    coat_l = float(np.median(lum_c[shz_])) if shz_.sum() > 20 else 0.15
    nc_ = np.array(J("neck_01"))
    rr_ = np.linalg.norm(co[:, :2] - nc_[:2], axis=1)
    reg = (((co[:, 2] > J("spine_03").z) & (co[:, 1] < nc_[1]) & (np.abs(co[:, 0] - nc_[0]) < 0.11))
           | ((co[:, 2] > J("clavicle_l").z - 0.01) & (rr_ < 0.10)))
    # in the throat (the cravat, the collar) anything clearly lighter than the coat goes; lower down only what is
    # nearer the shirt's white than the coat (a mid-grey waistcoat between the lapels stays)
    lm_ = float(JOB["collar_build"].get("light_margin", 0.12))
    q90 = float(np.quantile(lum_c[reg], 0.9)) if reg.sum() > 20 else coat_l + 2 * lm_
    throat = (co[:, 2] > J("spine_05").z - 0.04) & (np.abs(co[:, 0] - nc_[0]) < 0.06)
    thr = np.where(throat, coat_l + lm_, max(coat_l + lm_, 0.5 * (coat_l + q90)))
    light = reg & (lum_c > thr) & ~skin
    REP["collar_region"] = {"coat_lum": round(coat_l, 3), "light_verts": int(light.sum())}
    COAT_RGB = np.median(col[shz_], 0).tolist() if shz_.sum() > 20 else [0.05, 0.05, 0.05]
    if light.sum() > 50:
        ll = np.sort(lum_c[light])
        cl = col[light][np.argsort(lum_c[light])]
        n_ = len(ll)
        COLLAR_RGB = {"collar": np.median(cl[int(0.7 * n_):], 0).tolist(),
                      "cravat": np.median(cl[:int(0.2 * n_)], 0).tolist()}
        gone |= light
        if JOB["collar_build"].get("neckwear") == "tie":
            # a tie is darker than the shirt: its colour is the cloth down the shirt front's centre line that is
            # not the light shirt (nor the coat's colour at the shoulders)
            strip = ((np.abs(co[:, 0] - nc_[0]) < 0.02) & (co[:, 1] < nc_[1]) & (co[:, 2] > J("spine_04").z)
                     & (co[:, 2] < J("clavicle_l").z) & ~light & ~skin & (np.abs(lum_c - coat_l) > 0.03))
            if strip.sum() > 20:
                COLLAR_RGB["cravat"] = np.median(col[strip], 0).tolist()
                gone |= strip
    COLLAR_RGB = COLLAR_RGB or {}
    COLLAR_RGB.setdefault("dress", COAT_RGB)
    for k_ in ("collar", "cravat"):
        if JOB["collar_build"].get(f"{k_}_rgb"):
            COLLAR_RGB = COLLAR_RGB or {}
            COLLAR_RGB[k_] = list(JOB["collar_build"][f"{k_}_rgb"])
    REP["collar_rgb"] = {k: [round(float(x), 3) for x in v] for k, v in (COLLAR_RGB or {}).items()}
# the garment ends where the cloth ends: the generated figure's own bare limbs below a short sleeve or a hem were kept
# as garment and worn as cloth (flat skin-coloured forearms, 11-12; long sleeves had hidden the case). Along each limb
# (shoulder to the body's wrist, hip to the ankle) the outer radius of the figure round the limb's axis is read
# station by station; the cuff is the station where it steps down from a sleeve standing off the limb to a bare limb
# that stays thin to the end, skin colour only a second vote (more skin below the step than above). Beyond the cuff
# the limb goes, hands included; on the legs from the hem to just above the foot (footwear is the garment's).
# the second vote's colours: a skin-coloured tunic passes any fixed skin test (11-12), so a vertex is skin-like when
# it is nearer the figure's own skin (its hands: the far end of each arm, past the body's wrist) than its cloth (the
# chest and back between the armpits, off the neck)
def _limb_edge(S0, E0, sg, tube, step_min, ratio, lo_frac, hi_frac):
    ax_ = E0 - S0
    L_ = float(np.linalg.norm(ax_))
    ax_ = ax_ / L_
    p_ = (co - S0) @ ax_
    rad_ = np.linalg.norm((co - S0) - p_[:, None] * ax_[None, :], axis=1)
    side_ = (sg * (co[:, 0] - J("pelvis").x) > 0)
    st_ = np.arange(lo_frac * L_, hi_frac * L_, 0.01)
    r_, sk_ = [], []
    for c_ in st_:
        m_ = side_ & (np.abs(p_ - c_) < 0.006) & (rad_ < tube)
        r_.append(float(np.quantile(rad_[m_], 0.9)) if m_.sum() >= 8 else tube)    # nothing near the axis: cloth
        sk_.append(float(SKL[m_].mean()) if m_.sum() >= 8 else 0.0)                # standing off it
    r_, sk_ = np.array(r_), np.array(sk_)
    cands = []
    for i in range(4, len(st_) - 3):
        above, below = np.median(r_[max(0, i - 8):i]), np.median(r_[i:i + 8])
        rest = r_[i:]
        if (above - below >= step_min and above >= ratio * below and np.quantile(rest, 0.9) <= 1.3 * below
                and sk_[i:].mean() >= 0.5):
            cands.append((i, above - below))
    best = None
    if cands:
        # the windowed step is flat over several stations round the cuff (a plateau of medians): among those within
        # 2 mm of the largest, the station with the steepest single drop is the cuff, not the first one (noise
        # decided among ties and the cut landed up to 3.5 cm inside the sleeve)
        top = max(c[1] for c in cands)
        tied = [c for c in cands if c[1] >= top - 0.002]
        best = max(tied, key=lambda c: (r_[c[0] - 1] - r_[c[0]], -c[0]))
    rep_ = {"radius_cm": [round(100 * x, 1) for x in r_[::2]], "skin": [round(x, 2) for x in sk_[::2]]}
    if best is None:
        return None, p_, rad_, side_, rep_
    i = best[0]
    rep_.update({"at_frac": round(float(st_[i] / L_), 3), "step_cm": round(100 * best[1], 1),
                 "above_cm": round(100 * float(np.median(r_[max(0, i - 8):i])), 1),
                 "below_cm": round(100 * float(np.median(r_[i:i + 8])), 1)})
    return float(st_[i]), p_, rad_, side_, rep_


def _hull_perimeter(Q):
    """the convex hull's perimeter of 2-D points (a slice's girth, offset from the axis does not matter)"""
    Q = np.unique(np.round(Q, 5), axis=0)
    if len(Q) < 3:
        return None
    Q = Q[np.lexsort((Q[:, 1], Q[:, 0]))]

    def half(P_):
        h = []
        for q in P_:
            # the 2-D cross written out (np.cross on 2-D vectors is gone in numpy 2)
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    H_ = half(Q)[:-1] + half(Q[::-1])[:-1]
    H_ = np.array(H_)
    return float(np.linalg.norm(np.roll(H_, -1, axis=0) - H_, axis=1).sum())


def _girth_at(P_, S_, ax_, st, sg, r_max=0.11):
    """girth (cm) of points P_ round the limb axis at station st (m along it from S_), that side only"""
    p_ = (P_ - S_) @ ax_
    rr = (P_ - S_) - p_[:, None] * ax_[None, :]
    m_ = (np.abs(p_ - st) < 0.008) & (np.linalg.norm(rr, axis=1) < r_max) & (sg * (P_[:, 0] - J("pelvis").x) > 0)
    if m_.sum() < 12:
        return None
    u_ = np.cross(ax_, [0.0, 1.0, 0.0])
    u_ = u_ / np.linalg.norm(u_) if np.linalg.norm(u_) > 1e-6 else np.array([1.0, 0.0, 0.0])
    v_ = np.cross(ax_, u_)
    g_ = _hull_perimeter(np.c_[rr[m_] @ u_, rr[m_] @ v_])
    return None if g_ is None else round(100 * g_, 1)


# where a limb is bare its thickness comes from the generated figure (the MetaHuman's own forearms were thin against
# the drawn ones and the sleeves, 11-12): the girth of the figure's bare limb and of the posed body's at the same
# stations below the cuff or hem, read against the MetaHuman's body constraints by name (relative changes, set by
# cast_char.py as for the neck)
LIMB_GIRTH = {}


def _limb_girths(kind, side, sg, S_, E_, c_):
    BVp = np.array([v.co[:] for v in body.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
    ax_ = (E_ - S_) / np.linalg.norm(E_ - S_)
    L_ = float(np.linalg.norm(E_ - S_))
    out = {}
    if kind == "arm":
        el = float(np.linalg.norm(np.array(pose_head(f"lowerarm_{side}")) - S_))
        named = [("Forearm", list(np.arange(max(c_ + 0.02, el + 0.03), min(max(c_ + 0.02, el + 0.03) + 0.07, L_ - 0.05),
                                            0.01))),
                 ("Wrist", [L_ - 0.04])]
        if el > c_ + 0.02:
            named.append(("Elbow", [el]))
    else:
        kn = float(np.linalg.norm(np.array(pose_head(f"calf_{side}")) - S_))
        named = [("Calf", list(np.arange(max(c_ + 0.02, kn + 0.06), max(c_ + 0.02, kn + 0.06) + 0.12, 0.01)))]
        if kn > c_ + 0.02:
            named.append(("Knee", [kn]))
    prof = []
    for st in np.arange(c_ + 0.01, L_ + 0.06, 0.01):
        prof.append((round(st / L_, 3), _girth_at(co, S_, ax_, st, sg), _girth_at(BVp, S_, ax_, st, sg)))
    out["_profile"] = prof
    # the limb never wider than the hole it comes out of: its girth is capped a little under the sleeve's or hem's
    # opening, measured on the figure just above the cuff (a thicker limb poked through the cloth otherwise)
    opening = _girth_at(co, S_, ax_, c_ - 0.01, sg, r_max=0.16)
    cap = None if opening is None else round(float(JOB.get("limb_opening_frac", 0.9)) * opening, 1)
    out["_opening_cm"] = opening
    out["_cap_cm"] = cap
    for name, sts in named:
        rows = [(gf, gb) for gf, gb in ((_girth_at(co, S_, ax_, st, sg), _girth_at(BVp, S_, ax_, st, sg)) for st in sts)
                if gf is not None and gb is not None]
        if rows:
            r_ = np.array(rows)
            want = r_[:, 0] if cap is None else np.minimum(r_[:, 0], cap)
            out[name] = {"figure_cm": round(float(np.median(r_[:, 0])), 1), "body_cm": round(float(np.median(r_[:, 1])), 1),
                         "opening_cm": opening, "capped_cm": round(float(np.median(want)), 1),
                         "delta_cm": round(float(np.median(want - r_[:, 1])), 1), "stations": len(rows)}
    LIMB_GIRTH[f"{kind}_{side}"] = out
    return out


if not STATUE and not FIT_POSE and JOB.get("cut_bare_limbs", True):
    SKL, _refs = _skin_like()
    CUFF = {}
    REP["bare_limbs"] = {"colours": _refs}
    for side, sg in (("l", 1), ("r", -1)):
        S0 = np.array(pose_head(f"upperarm_{side}"))
        E0 = np.array(pose_head(f"hand_{side}"))
        c_, p_, rad_, side_, rep_ = _limb_edge(S0, E0, sg, 0.14, 0.012, 1.3, 0.3, 1.0)
        if c_ is not None:
            CUFF[side] = (S0, (E0 - S0) / np.linalg.norm(E0 - S0), c_)
            # a cuff is not square to the arm (its underside hangs lower): the bare limb inside the cuff's reach,
            # as thin as the bare limb and skin-like, goes too
            cut_ = side_ & (p_ > c_) & (rad_ < 0.16)
            cut_ |= (side_ & (p_ > c_ - 0.10) & (p_ <= c_) & SKL & (rad_ < 1.3 * rep_["below_cm"] / 100))
            rep_["cut_inside_cuff"] = int((cut_ & (p_ <= c_)).sum())
            gone |= cut_
            rep_["cut"] = int(cut_.sum())
            rep_["girths"] = _limb_girths("arm", side, sg, S0, E0, c_)
        REP["bare_limbs"][f"arm_{side}"] = rep_
        S1 = np.array(pose_head(f"thigh_{side}"))
        E1 = np.array(pose_head(f"foot_{side}"))
        c_, p_, rad_, side_, rep_ = _limb_edge(S1, E1, sg, 0.14, 0.015, 1.3, 0.35, 0.92)
        if c_ is not None:
            cut_ = side_ & (p_ > c_) & (rad_ < 0.14) & (co[:, 2] > E1[2] + 0.04)
            gone |= cut_
            rep_["cut"] = int(cut_.sum())
            rep_["girths"] = _limb_girths("leg", side, sg, S1, E1, c_)
        REP["bare_limbs"][f"leg_{side}"] = rep_
    REP["limb_girths"] = LIMB_GIRTH
    # inside the neckline of an open or base neckline: the figure's own chest skin joined to the neck's (the skin
    # already cut round the neck), within the shoulders and above the chest's middle
    if (JOB.get("neckline") or (REP.get("neckline_figure") or {}).get("class")) in ("open", "base"):
        _ev = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get("vertices", _ev)
        _ev = _ev.reshape(-1, 2)
        _shx = abs(float(J("upperarm_l").x - J("neck_01").x))
        _box = SKL_SURE & (co[:, 2] > J("spine_04").z) & (np.abs(co[:, 0] - J("neck_01").x) < _shx) & (co[:, 1] < J("neck_01").y + 0.02)
        _in = gone & SKL_SURE
        _n0 = int(_in.sum())
        for _ in range(200):
            m_ = _box[_ev[:, 0]] & _box[_ev[:, 1]] & (_in[_ev[:, 0]] != _in[_ev[:, 1]])
            if not m_.any():
                break
            _in[_ev[m_].ravel()] = True
        REP["bare_limbs"]["neckline_skin"] = int(_in.sum()) - _n0
        gone |= _in
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
HAIR_KEPT = False
if LAB_RAW is not None and JOB.get("hair_keep", True) and HEAD is not None and (LAB_RAW == 2).sum() > 500 \
        and (LAB_RAW == 1).sum() > 100:
    # the figure's own hair, kept (the parser's labels say which of it is head hair): the MetaHuman's grooms have
    # no long hair worn back, no plait, no child's mop, and a woman drawn with long dark hair stood in the engine
    # with a crop (11-14). Cut out here as its own piece, before the head goes; set on the MetaHuman's head below.
    _hm = LAB_RAW == 2
    _nk_k = np.array(J("neck_01"))
    HAIR = fig.copy()
    HAIR.data = fig.data.copy()
    bpy.context.scene.collection.objects.link(HAIR)
    hb = bmesh.new()
    hb.from_mesh(HAIR.data)
    hb.verts.ensure_lookup_table()
    bmesh.ops.delete(hb, geom=[hb.verts[i] for i in np.nonzero(~_hm)[0]], context="VERTS")
    # only hair that hangs from the head: each joined piece of it must reach above the neck's base (a dark patch
    # on a chest pocket labelled hair is not the head's)
    hb.verts.ensure_lookup_table()
    _seen, _drop = set(), []
    for v0 in hb.verts:
        if v0.index in _seen:
            continue
        _isl, _st = [], [v0]
        _seen.add(v0.index)
        while _st:
            v_ = _st.pop()
            _isl.append(v_)
            for e_ in v_.link_edges:
                w_ = e_.other_vert(v_)
                if w_.index not in _seen:
                    _seen.add(w_.index)
                    _st.append(w_)
        if max(v_.co.z for v_ in _isl) < _nk_k[2] + 0.02:
            _drop += _isl
    if _drop:
        bmesh.ops.delete(hb, geom=_drop, context="VERTS")
    hb.to_mesh(HAIR.data)
    hb.free()
    # the drawn face: where the drawn head is, and how wide (the face labels on the head itself: above the neck's
    # base, within a head's width of its axis)
    _fm = (LAB_RAW == 1) & (co[:, 2] > _nk_k[2]) & (np.abs(co[:, 0] - _nk_k[0]) < 0.14)
    _fc = co[_fm] if _fm.sum() > 100 else co[LAB_RAW == 1]
    HAIR["face_x"] = [float(np.quantile(_fc[:, 0], 0.02)), float(np.quantile(_fc[:, 0], 0.98))]
    HAIR["face_front_y"] = float(np.quantile(_fc[:, 1], 0.03))
    HAIR["chin_z"] = float(np.quantile(_fc[:, 2], 0.02))
    HAIR["face_top_z"] = float(np.quantile(_fc[:, 2], 0.98))
    # kept when it is hair a groom cannot give: hair with a body of its own. Either it hangs below the jaw (worn
    # long and down), or it stands well out from the head (long hair tied back, a plait, a mop): wider than the
    # face by a quarter, or reaching back behind the face by more than a head is deep. Short hair stays the
    # MetaHuman's own groom (kept, a crop was a thin shell of dots on the skull)
    _hv_ = np.array([v.co[:] for v in HAIR.data.vertices]) if len(HAIR.data.vertices) else np.zeros((1, 3))
    _fw_ = max(0.08, HAIR["face_x"][1] - HAIR["face_x"][0])
    _hang = HAIR["chin_z"] - float(np.quantile(_hv_[:, 2], 0.02))
    _wide = float(np.quantile(_hv_[:, 0], 0.98) - np.quantile(_hv_[:, 0], 0.02)) / _fw_
    _deep = (float(np.quantile(_hv_[:, 1], 0.98)) - HAIR["face_front_y"]) / _fw_
    REP["hair_body"] = {"hangs_below_chin_m": round(_hang, 3), "width_over_face": round(_wide, 2),
                        "depth_over_face": round(_deep, 2)}
    if _hang >= float(JOB.get("hair_keep_hang_m", 0.05)) or _wide >= float(JOB.get("hair_keep_wide", 1.25)) \
            or _deep >= float(JOB.get("hair_keep_deep", 1.7)):
        HAIR_KEPT = True
    else:
        bpy.data.objects.remove(HAIR)
        HAIR = None
elif JOB.get("hair_cap", True):
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
seen, crumbs, islands = set(), [], []
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
    else:
        islands.append(isl)
# pieces left floating round the head by the collar cut (a bun, a hat brim, hair at the nape): any piece other than
# the largest that lies wholly above the collarbones, off the neck's axis (a collar or frill ring is centred on it)
if len(islands) > 1:
    islands.sort(key=len, reverse=True)
    zc_ = J("clavicle_l").z + float(JOB.get("float_above_clav_m", 0.04))
    for isl in islands[1:]:
        cxy = np.mean([v.co[:2] for v in isl], axis=0)
        if min(v.co.z for v in isl) > zc_ and np.linalg.norm(cxy - np.array(J("neck_01"))[:2]) > 0.06:  # not a collar ring
            crumbs += isl
            REP["floating_dropped"] = REP.get("floating_dropped", 0) + len(isl)
bmesh.ops.delete(bm, geom=crumbs, context="VERTS")
# holes the generator left in the cloth (a dark hole low on a jacket front shows its inside): every open boundary
# loop shorter than HOLE_MAX is filled; the new faces take the UVs their vertices have in the faces round them
HOLE_MAX = float(JOB.get("fill_holes_m", 0.30))
if HOLE_MAX > 0:
    bnd = [e for e in bm.edges if e.is_boundary]
    seen_e, loops = set(), []
    adj = {}
    for e in bnd:
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    for e0 in bnd:
        if e0 in seen_e:
            continue
        loop, stack = [], [e0]
        seen_e.add(e0)
        while stack:
            e = stack.pop()
            loop.append(e)
            for v in e.verts:
                for e2 in adj[v]:
                    if e2 not in seen_e:
                        seen_e.add(e2)
                        stack.append(e2)
        loops.append(loop)
    uvl = bm.loops.layers.uv.active
    # never the garment's own openings: the neck (round the neck's axis, above the collarbones) and the cuffs and
    # hems are openings, not holes (a filled neck opening became a disc that the head push blew into a shroud)
    _nk = np.array(J("neck_01"))
    _wr = [np.array(LM[f"wrist_{s_}"]) for s_ in ("l", "r") if f"wrist_{s_}" in LM]

    def _opening(lp):
        c_ = np.mean([np.array(v.co[:]) for e in lp for v in e.verts], axis=0)
        if c_[2] > J("clavicle_l").z - 0.06 and np.linalg.norm(c_[:2] - _nk[:2]) < 0.14:
            return True
        return any(np.linalg.norm(c_ - w_) < 0.15 for w_ in _wr) or c_[2] < 0.15
    # the opening itself is the longest loop of each kind (the neck's, each cuff's, the hem's); smaller loops in the
    # same place are tears beside it (a frayed neckline's slits showed the body through, 11-12) and are holes
    _lens = [sum(e.calc_length() for e in lp) for lp in loops]
    _open_ids = set()
    for _kind in ("neck", "cuff_l", "cuff_r", "hem"):
        def _is_kind(lp, kind=_kind):
            c_ = np.mean([np.array(v.co[:]) for e in lp for v in e.verts], axis=0)
            if kind == "neck":
                return c_[2] > J("clavicle_l").z - 0.06 and np.linalg.norm(c_[:2] - _nk[:2]) < 0.14
            if kind == "hem":
                return c_[2] < 0.15
            w_ = _wr[0 if kind == "cuff_l" else -1] if _wr else None
            return w_ is not None and np.linalg.norm(c_ - w_) < 0.15
        ks = [i for i, lp in enumerate(loops) if _is_kind(lp)]
        if ks:
            _open_ids.add(max(ks, key=lambda i: _lens[i]))
            if _kind == "neck":
                # the neck opening's width (the skin probe leaves the opening out: skin in it is the neck and
                # chest by design), as a share of the shoulder span between the upper arms' heads
                _nl = max(ks, key=lambda i: _lens[i])
                _nx = np.array([v.co.x for e in loops[_nl] for v in e.verts])
                _span = abs(float(J("upperarm_l").x - J("upperarm_r").x))
                _hw = float(np.max(np.abs(_nx - _nk[0])))
                REP["neck_opening"] = {"half_width_m": round(_hw, 3), "shoulder_span_m": round(_span, 3),
                                       "half_width_of_span": round(_hw / max(_span, 1e-3), 3)}
    small = [lp for i, lp in enumerate(loops) if len(lp) >= 3 and _lens[i] < HOLE_MAX and i not in _open_ids
             and (not _opening(lp) or _lens[i] < float(JOB.get("tear_max_m", 0.25)))]
    REP["open_loops_left"] = sorted([round(_lens[i], 2) for i, lp in enumerate(loops)
                                     if len(lp) >= 3 and lp not in small and i not in _open_ids], reverse=True)[:12]
    REP["open_loops_left_near_neck"] = sorted([round(_lens[i], 2) for i, lp in enumerate(loops)
                                               if len(lp) >= 3 and lp not in small and i not in _open_ids
                                               and _opening(lp)], reverse=True)[:12]
    vuv = {}
    for lp in small:
        for e in lp:
            for v in e.verts:
                if v not in vuv and v.link_loops:
                    vuv[v] = v.link_loops[0][uvl].uv.copy()
    edges_ = [e for lp in small for e in lp]
    # one call for every small hole (a call per hole cost 157 s on a dress with 2,600 of them)
    res = bmesh.ops.holes_fill(bm, edges=edges_, sides=0) if edges_ else {"faces": []}
    for f in res["faces"]:
        for lp_ in f.loops:
            if lp_.vert in vuv:
                lp_[uvl].uv = vuv[lp_.vert]
    nfill = len(small)
    REP["holes_filled"] = nfill
bm.to_mesh(fig.data)
bm.free()
REP["cut_verts"] = int(gone.sum())
REP["crumb_verts"] = len(crumbs)
mark("cut", t)
stage_save("s2_cut")

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
    if JOB.get("sleeve_radius_m") is None or JOB.get("sleeve_tube_measured", True):
        # the tube measured from the sleeve itself: per 2 cm station along the arm, the outer radius of the
        # arm-weighted cloth there (95th percentile) plus a margin, capped short of the torso's surface in that
        # slice (so a puffed sleeve gets its wide tube at the puff, a coat sleeve a narrow one, never the coat's side)
        st = np.arange(0.0, L + 0.02, 0.02)
        rprof = np.full(len(st), SLEEVE_R)
        cap = np.full(len(st), 0.5)
        tv = np.array([v.co[:] for v in torso.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
        tp = (tv - S0) @ ax
        trad = np.linalg.norm((tv - S0) - np.clip(tp, 0, L)[:, None] * ax[None, :], axis=1)
        for k, t0 in enumerate(st):
            sl = (np.abs(p - t0) < 0.012) & (armw >= 0.3)
            if sl.sum() > 15:
                rprof[k] = float(np.quantile(rad[sl], 0.95)) + float(JOB.get("sleeve_tube_margin_m", 0.02))
            ts = np.abs(tp - t0) < 0.012
            if ts.any():
                cap[k] = max(float(trad[ts].min()) - 0.01, 0.04)
        # the torso cap matters near the shoulder only; along the forearm the slab meets the hip and the coat skirt,
        # so there the tube is the sleeve's own radius, never below the job's floor
        rlim = np.where(st > 0.4 * L, np.maximum(rprof, SLEEVE_R), np.minimum(rprof, cap))
        thr = np.interp(np.clip(p, 0, L), st, rlim)
        REP.setdefault("sleeve_tube_m", {})[side] = {"max": round(float(rlim.max()), 3), "min": round(float(rlim.min()), 3),
                                                     "median": round(float(np.median(rlim)), 3)}
        tube |= (rad < thr) & (p > 0)
    else:
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
skirt = (fco[:, 2] < J("pelvis").z - 0.05) & (fco[:, 2] > float(JOB.get("skirt_min_z_m", 0.14))) & (d_leg > LEG_OFF) & ~sleeve
if JOB.get("long_skirt"):
    # a floor-length skirt: all the cloth below the waist hangs from the pelvis (none of it is trouser legs; what
    # lies close to the legs is the skirt's back and inner side, or the generator's legs inside, dropped below)
    skirt = (fco[:, 2] < J("pelvis").z - 0.05) & ~sleeve
for gname_ in [g.name for g in fig.vertex_groups]:
    fig.vertex_groups[gname_].remove([int(i) for i in np.nonzero(skirt)[0]])
for i in np.nonzero(skirt)[0]:
    (fig.vertex_groups.get("pelvis") or fig.vertex_groups.new(name="pelvis")).add([int(i)], SKIRT_PELVIS, "ADD")
    if SKIRT_PELVIS < 1:
        side = "l" if legd[0, i] < legd[1, i] else "r"
        tgn = f"thigh_{side}"
        (fig.vertex_groups.get(tgn) or fig.vertex_groups.new(name=tgn)).add([int(i)], 1 - SKIRT_PELVIS, "ADD")
REP["skirt_verts"] = int(skirt.sum())
# the band above the skirt line blends from the skirt's pelvis toward the body's own weights over SKIRT_BLEND, so a
# bending spine or a stepping thigh does not tear the cloth along that line
SKIRT_BLEND = float(JOB.get("skirt_blend_m", 0.12))
if SKIRT_BLEND > 0 and skirt.any():
    z0_ = J("pelvis").z - 0.05
    band_ = np.nonzero(~skirt & ~sleeve & (fco[:, 2] >= z0_) & (fco[:, 2] < z0_ + SKIRT_BLEND))[0]
    pg_ = fig.vertex_groups.get("pelvis") or fig.vertex_groups.new(name="pelvis")
    gl_ = {g.index: g for g in fig.vertex_groups}
    for i in band_:
        t_ = float((fco[i, 2] - z0_) / SKIRT_BLEND)
        v_ = fig.data.vertices[int(i)]
        cur = {g.group: g.weight for g in v_.groups}
        tot = sum(cur.values()) or 1.0
        for gi_, w_ in cur.items():
            gl_[gi_].add([int(i)], t_ * w_ / tot, "REPLACE")
        pg_.add([int(i)], (1 - t_) + t_ * cur.get(pg_.index, 0.0) / tot, "REPLACE")
    REP["skirt_blend_verts"] = int(len(band_))
# the loose cloth's vertex ids (the outfit's first vertices keep their ids through the joins below), for the
# per-stance settle (bl_settle.py)
if JOB.get("long_skirt"):
    # a floor-length skirt: whatever the generator built inside it below the waist (legs, a lining, the skirt's
    # inner face) is dropped;
    # it is never seen standing and pokes through the cloth once the legs move
    # inner = below the waist, not skirt, and enclosed: a ray outward from the body's axis meets more cloth
    # (the skirt's own inner face and the hem's bottom cap are enclosed too: one surface per column for the settle)
    # the skirt's own inner face (normal toward the body's axis) and the hem's bottom cap (normal down) go by their
    # normals (a ray would also catch the valleys of the skirt's folds)
    cand = np.nonzero((fco[:, 2] < J("pelvis").z - 0.05) & (d_leg <= LEG_OFF) & ~sleeve)[0]
    fn_ = np.array([v.normal[:] for v in fig.data.vertices])
    rad_ = fco[:, :2] - np.array(J("pelvis"))[:2]
    rad_ /= np.maximum(np.linalg.norm(rad_, axis=1), 1e-6)[:, None]
    skin_in = skirt & (fco[:, 2] < J("pelvis").z - float(JOB.get("inner_face_below_m", 0.15))) & ((np.sum(fn_[:, :2] * rad_, 1) < -0.3) | (fn_[:, 2] < -0.7))
    fbvh = BVHTree.FromObject(fig, bpy.context.evaluated_depsgraph_get())
    inner = np.zeros(len(fco), bool)
    pc_ = np.array(J("pelvis"))
    for i in cand:
        dv = fco[i] - pc_
        dv[2] = 0
        n_ = np.linalg.norm(dv)
        if n_ < 1e-6:
            continue
        hit = fbvh.ray_cast(Vector(fco[i] + dv / n_ * 0.01), Vector(dv / n_), 0.6)
        inner[i] = hit[0] is not None
    inner |= skin_in
    sg_ = fig.vertex_groups.new(name="_skirt")
    sg_.add([int(i) for i in np.nonzero(skirt)[0]], 1.0, "REPLACE")
    bm = bmesh.new()
    bm.from_mesh(fig.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(inner)[0]], context="VERTS")
    bm.to_mesh(fig.data)
    bm.free()
    gi_ = sg_.index
    skirt = np.array([any(g.group == gi_ for g in v.groups) for v in fig.data.vertices])
    fig.vertex_groups.remove(sg_)
    REP["inner_dropped_verts"] = int(inner.sum())
# a foot drawn bare or in open footwear (a sandal): the body's own foot shows there (its skin shows wherever cloth
# does not cover it), so the figure's drawn foot goes and what is left of its footwear is centred on the body's foot
# (11-12: an empty sandal with a drawn foot in it stood beside the body's bare foot)
if not STATUE and JOB.get("body_skin_shows", True) and JOB.get("open_footwear", True) and "fcol" in fig.data.attributes:
    _zs = float(JOB.get("open_footwear_top_m", 0.11))    # (its own height: a long garment's job has no shoe top)
    _fc = np.zeros(len(fig.data.vertices) * 4, np.float32)
    fig.data.attributes["fcol"].data.foreach_get("color", _fc)
    _sk = is_skin(_fc.reshape(-1, 4)[:, :3])
    _co = np.array([v.co[:] for v in fig.data.vertices])
    _gone = np.zeros(len(_co), bool)
    REP["open_footwear"] = {"lowest_m": round(float(_co[:, 2].min()), 3), "under_shoe_top": int((_co[:, 2] < _zs).sum())}
    for side, sg in (("l", 1), ("r", -1)):
        _low = (_co[:, 2] < _zs) & (sg * (_co[:, 0] - J("pelvis").x) > 0)
        if _low.sum() < 20:
            continue
        frac = float(_sk[_low].mean())
        REP["open_footwear"][side] = {"verts": int(_low.sum()), "skin_frac": round(frac, 3)}
        if frac < float(JOB.get("open_footwear_skin", 0.25)):
            continue                                     # a closed shoe: as drawn, the body's foot stays hidden in it
        _gone |= _low & _sk
        _wear = _low & ~_sk
        if _wear.sum() < float(JOB.get("open_footwear_min_share", 0.3)) * _low.sum():
            _gone |= _low                                # footwear the colour cannot tell from the foot (tan leather):
            REP["open_footwear"][side]["barefoot"] = True    # a clean bare foot, not crumbs of a sandal beside it
            continue
        if _wear.sum() >= 20 and f"ball_{side}" in arm.pose.bones:
            tgt = (np.array(pose_head(f"foot_{side}")[:2]) + np.array(pose_head(f"ball_{side}")[:2])) / 2
            sh_ = tgt - _co[_wear, :2].mean(0)
            for i in np.nonzero(_wear)[0]:
                fig.data.vertices[int(i)].co.x += float(sh_[0])
                fig.data.vertices[int(i)].co.y += float(sh_[1])
            REP["open_footwear"][side]["moved_m"] = [round(float(x), 3) for x in sh_]
            # and it is rigid on that foot, not loose cloth
            ids_ = [int(i) for i in np.nonzero(_wear)[0]]
            for g_ in fig.vertex_groups:
                g_.remove(ids_)
            (fig.vertex_groups.get(f"foot_{side}") or fig.vertex_groups.new(name=f"foot_{side}")).add(ids_, 1.0, "REPLACE")
            skirt[_wear] = False
    if _gone.any():
        sg_ = fig.vertex_groups.new(name="_skirt2")
        sg_.add([int(i) for i in np.nonzero(skirt)[0]], 1.0, "REPLACE")
        bm = bmesh.new()
        bm.from_mesh(fig.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(_gone)[0]], context="VERTS")
        bm.to_mesh(fig.data)
        bm.free()
        gi_ = sg_.index
        skirt = np.array([any(g.group == gi_ for g in v.groups) for v in fig.data.vertices])
        fig.vertex_groups.remove(sg_)
        REP["open_footwear"]["drawn_foot_verts_removed"] = int(_gone.sum())
np.save(os.path.join(OUT, "skirt_idx.npy"), np.nonzero(skirt)[0].astype(np.int32))

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
# the body is hidden in the engine: cloth inside it is never seen, and pushing every vertex out of it printed the
# body's own shape (a bust, a belly) into the garment. Only the visible head mesh (neck, shoulder skin) pushes, unless
# the job asks for the body too.
_push_objs = (body, HEAD) if JOB.get("push_body", False) else (HEAD,)
bvhs = [] if STATUE else [BVHTree.FromObject(o, dg) for o in _push_objs if o is not None]   # statue: nothing under it
fco = np.array([v.co[:] for v in fig.data.vertices])
EASE_ALL = float(JOB.get("body_ease_m", 0.014))
HEAD_EASE = float(JOB.get("head_ease_m", 0.014))      # the statue's head must cover the hidden head and its hair
disp = np.zeros_like(fco)
zfeet = float(JOB.get("push_above_m", 0.14))          # shoes stay as drawn (the body's feet are bigger)
# HEAD_PUSH_R: only cloth within this distance of the neck's axis is pushed out of the head mesh (the head mesh's
# shoulder skin reaches under a puffed shoulder; pushing there raised the shoulders into points)
# the collar's girth: where the MetaHuman's neck is thicker than the generated collar, pushing each vertex out of the
# head mesh blew the collar and cravat into a shroud. Instead, per 1 cm of height between the shoulder line and the
# chin, the neck's own radius (head mesh points round the axis, largest per direction) against the innermost cloth's
# radius there; all cloth round the neck moves out by that difference as one radial offset (every layer together,
# full up to 8 cm outside the inner layer, fading over the next 6 cm), so the collar keeps its shape. The per-vertex
# head push then acts only away from the neck.
NECK_RADIAL = bool(JOB.get("neck_radial_fit", False)) and not STATUE and HEAD is not None
if not STATUE and HEAD is not None:
    _P = fco + disp
    _nk = np.array(J("neck_01"))
    _hv = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
    _rr = np.linalg.norm(_P[:, :2] - _nk[:2], axis=1)
    _ang = np.arctan2(_P[:, 1] - _nk[1], _P[:, 0] - _nk[0])
    _sec = ((_ang + math.pi) / (2 * math.pi) * 16).astype(int) % 16
    _ha = np.arctan2(_hv[:, 1] - _nk[1], _hv[:, 0] - _nk[0])
    _hs = ((_ha + math.pi) / (2 * math.pi) * 16).astype(int) % 16
    _hr = np.linalg.norm(_hv[:, :2] - _nk[:2], axis=1)
    z0n, z1n = float(J("clavicle_l").z) + 0.01, float(J("neck_02").z) + 0.02
    zgrid = np.arange(z0n, z1n + 0.01, 0.01)
    off = np.zeros((len(zgrid), 16))
    rin_g = np.full((len(zgrid), 16), np.nan)
    for zi, z in enumerate(zgrid):
        cm_ = (np.abs(_P[:, 2] - z) < 0.006) & (_rr < 0.15)
        hm_ = (np.abs(_hv[:, 2] - z) < 0.006) & (_hr < 0.11)
        for k_ in range(16):
            c_ = cm_ & (_sec == k_)
            h_ = hm_ & (_hs == k_)
            if c_.sum() > 3 and h_.sum() > 3:
                rin = float(np.quantile(_rr[c_], 0.03))
                rneed = float(_hr[h_].max()) + float(JOB.get("neck_ease_m", 0.006))
                rin_g[zi, k_] = rin
                off[zi, k_] = max(rneed - rin, 0.0)
    # smooth the offsets round the neck and up it (one smooth field)
    for _ in range(3):
        off = 0.5 * off + 0.25 * (np.roll(off, 1, 1) + np.roll(off, -1, 1))
    zi_ = np.clip(np.round((_P[:, 2] - z0n) / 0.01).astype(int), 0, len(zgrid) - 1)
    in_band = (_P[:, 2] > z0n - 0.04) & (_P[:, 2] < z1n + 0.03) & (_rr < 0.22)
    o_ = off[zi_, _sec]
    rin_v = np.nan_to_num(rin_g[zi_, _sec], nan=0.06)
    wr_ = np.clip((rin_v + 0.14 - _rr) / 0.06, 0, 1)
    wz_ = np.clip((_P[:, 2] - (z0n - 0.04)) / 0.04, 0, 1)
    dr = o_ * wr_ * wz_ * in_band
    radial = np.zeros_like(_P)
    radial[:, :2] = (_P[:, :2] - _nk[:2]) / np.maximum(_rr, 1e-6)[:, None]
    if NECK_RADIAL:
        disp = disp + radial * dr[:, None]
    # measured for the character build: the collar's inner girth at the neck's upper joint against the neck's own
    _zi = int(np.argmin(np.abs(zgrid - float(J("neck_02").z) + 0.01)))
    _ri = rin_g[_zi][~np.isnan(rin_g[_zi])]
    REP["collar_inner_girth_cm"] = round(float(2 * math.pi * np.median(_ri)) * 100, 1) if len(_ri) else None
    _hn = (np.abs(_hv[:, 2] - zgrid[_zi]) < 0.006) & (_hr < 0.11)
    REP["neck_girth_cm"] = round(float(2 * math.pi * np.median(_hr[_hn])) * 100, 1) if _hn.sum() > 10 else None
    REP["neck_radial_fit"] = {"max_offset_cm": round(float(off.max()) * 100, 1),
                              "median_offset_cm": round(float(np.median(off[off > 0])) * 100, 1) if (off > 0).any() else 0.0,
                              "verts": int((dr > 1e-4).sum())}
HEAD_PUSH_R = float(JOB.get("head_push_r_m", 9.0))
_nax = np.array(J("neck_01"))[:2]
for i, p in enumerate(fco):
    if p[2] < zfeet:
        continue
    # full push within HEAD_PUSH_R of the neck's axis, fading to none over the next 5 cm (a hard edge left a ridge)
    _wr = float(np.clip((HEAD_PUSH_R + 0.05 - np.linalg.norm(p[:2] - _nax)) / 0.05, 0, 1))
    if NECK_RADIAL and p[2] > float(J("clavicle_l").z) - 0.03 and np.linalg.norm(p[:2] - _nax) < 0.16:
        continue                       # the neck's cloth was fitted by the radial offset above
    if _wr <= 0:
        continue
    for bvh in bvhs:
        q = Vector(p) + Vector(disp[i])
        loc, nrm, _, dist = bvh.find_nearest(q, 0.08)
        if loc is None:
            continue
        s_ = (q - loc).dot(nrm)
        ease = HEAD_EASE if (STATUE and p[2] > J("neck_02").z) else EASE_ALL
        if s_ < ease:
            disp[i] = disp[i] + np.array(nrm) * (ease - s_) * _wr
fig.data.vertices.foreach_set("co", (fco + disp).ravel())
fig.data.update()
# shoulders: the drawn coat's shoulders stand off the body's narrower ones; pull what stands more than SH_GAP off the
# body back toward it (half the excess), fading out 22 cm from the shoulder joint
SH_GAP = float(JOB.get("shoulder_gap_m", 9.0))      # pull-in off by default (it roughened the shoulders)
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
COL_GAP = float(JOB.get("collar_gap_m", 9.0))       # collar pull off by default (jagged collar edges)
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
# the coat's sides: a drawn coat flares well past the body from the waist down (24-29 cm from the centre line on a
# 17 cm hip), so arms hanging at the sides sink into it and the still and its depth show one block with no arm gap.
# Between SLIM_LO and the armpit, cloth that stands more than SLIM_GAP outside the body's own side (arms excluded) is
# pulled in to SLIM_GAP + SLIM_K of the excess; monotonic across the layers (a waistcoat inside stays inside), fading
# out over 8 cm under the armpit; the sleeves are left alone.
SLIM_K = float(JOB.get("side_slim_k", 1.0))     # 1 = off; for a figure that fails mesh_gate
if SLIM_K < 1:   # statue (pure figure) too
    SLIM_GAP = float(JOB.get("side_slim_gap_m", 0.015))
    beval = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    BP = np.array([v.co[:] for v in beval.data.vertices])
    bgn = {g.index: g.name for g in body.vertex_groups}
    barm = np.array([bool(v.groups) and is_arm(bgn[max(v.groups, key=lambda g: g.weight).group]) for v in body.data.vertices])
    BT = BP[~barm]
    cx_ = J("pelvis").x
    z_hi = min(J("upperarm_l").z, J("upperarm_r").z) - 0.10
    z_lo = (J("pelvis").z - 0.05) if JOB.get("long_skirt") else (J("pelvis").z - float(JOB.get("side_slim_below_hip_m", 0.32)))
    zg = np.arange(z_lo - 0.03, z_hi + 0.11, 0.01)
    ext = {}
    for sg in (1, -1):
        e_ = []
        for z_ in zg:
            b_ = BT[np.abs(BT[:, 2] - z_) < 0.03]
            e_.append(np.quantile(sg * (b_[:, 0] - cx_), 0.99) if len(b_) > 10 else 0.15)
        ext[sg] = np.array(e_)
    gnf = {g.index: g.name for g in fig.vertex_groups}
    aw_ = np.array([sum(g.weight for g in v.groups if is_arm(gnf[g.group]) and not gnf[g.group].startswith("clavicle"))
                    for v in fig.data.vertices])
    # never the arms: anything within the sleeve tube round shoulder -> wrist (hands and fingers outside the tube
    # radius of a spread figure carried coat weights and were pulled into the body)
    in_tube = np.zeros(len(fco2), bool)
    for side in ("l", "r"):
        S0_ = np.array(pose_head(f"upperarm_{side}"))
        wr_ = np.array(LM[f"wrist_{side}"])
        L_ = np.linalg.norm(wr_ - S0_)
        ax_ = (wr_ - S0_) / L_
        t_ = (fco2 - S0_) @ ax_
        rr_ = np.linalg.norm((fco2 - S0_) - np.clip(t_, 0, None)[:, None] * ax_[None, :], axis=1)
        in_tube |= (t_ > 0.15 * L_) & (rr_ < float(JOB.get("sleeve_radius_m", 0.10)) + 0.05)
    nslim, worst = 0, 0.0
    for i in np.nonzero((aw_ < 0.5) & ~in_tube & (fco2[:, 2] > z_lo) & (fco2[:, 2] < z_hi + 0.08))[0]:
        sg = 1 if fco2[i, 0] >= cx_ else -1
        lat = sg * (fco2[i, 0] - cx_)
        e = float(np.interp(fco2[i, 2], zg, ext[sg])) + SLIM_GAP
        if lat <= e:
            continue
        w_ = float(np.clip((z_hi + 0.08 - fco2[i, 2]) / 0.08, 0, 1))
        new = e + (lat - e) * SLIM_K
        fco2[i, 0] = cx_ + sg * (lat + (new - lat) * w_)
        worst = max(worst, lat - e)
        nslim += 1
    REP["side_slim"] = {"verts": nslim, "max_excess_cm": round(worst * 100, 1), "k": SLIM_K,
                        "z": [round(z_lo, 2), round(z_hi, 2)]}
# the sleeves: a drawn sleeve is often a wide tube (a puffed dress sleeve, a loose coat sleeve) that merges with the
# body once the arm is down; sleeve cloth standing more than SLEEVE_GAP outside the body's own arm (its radius
# measured per 2 cm along each arm bone) is pulled in to SLEEVE_GAP + SLEEVE_K of the excess, radially about the bone
SLEEVE_K = float(JOB.get("sleeve_slim_k", 1.0))
if not STATUE and SLEEVE_K < 1:
    SLEEVE_GAP = float(JOB.get("sleeve_slim_gap_m", 0.012))
    beval = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    BP2 = np.array([v.co[:] for v in beval.data.vertices])
    bgn2 = {g.index: g.name for g in body.vertex_groups}
    bdom = [bgn2[max(v.groups, key=lambda g: g.weight).group] if v.groups else "" for v in body.data.vertices]
    gnf2 = {g.index: g.name for g in fig.vertex_groups}
    nsl = 0
    for side in ("l", "r"):
        for a_b, b_b in (("upperarm", "lowerarm"), ("lowerarm", "hand")):
            A_ = np.array(pose_head(f"{a_b}_{side}"))
            B_ = np.array(pose_head(f"{b_b}_{side}"))
            L_ = np.linalg.norm(B_ - A_)
            ax_ = (B_ - A_) / L_
            mine = np.array([n.startswith(a_b) and n.endswith(f"_{side}") for n in bdom])
            tb = (BP2[mine] - A_) @ ax_
            rb = np.linalg.norm((BP2[mine] - A_) - tb[:, None] * ax_[None, :], axis=1)
            bins = np.arange(0, L_ + 0.02, 0.02)
            rmax = np.array([np.quantile(rb[np.abs(tb - t_) < 0.015], 0.95) if (np.abs(tb - t_) < 0.015).sum() > 8
                             else np.nan for t_ in bins])
            if np.isnan(rmax).all():
                continue
            rmax = np.where(np.isnan(rmax), np.nanmedian(rmax), rmax)
            w_bone = np.array([sum(g.weight for g in v.groups if gnf2[g.group].startswith(a_b)
                                   and gnf2[g.group].endswith(f"_{side}")) for v in fig.data.vertices])
            t_ = (fco2 - A_) @ ax_
            sel = (w_bone > 0.5) & (t_ > 0.04 * L_) & (t_ < L_)
            for i in np.nonzero(sel)[0]:
                c_ = A_ + t_[i] * ax_
                rv = fco2[i] - c_
                r_ = np.linalg.norm(rv)
                e_ = float(np.interp(t_[i], bins, rmax)) + SLEEVE_GAP
                if r_ <= e_ or r_ < 1e-6:
                    continue
                # fade in from the armhole over the bone's first quarter (the shoulder's own shape stays)
                f_ = float(np.clip(t_[i] / (0.25 * L_), 0, 1)) if a_b == "upperarm" else 1.0
                new_r = e_ + (r_ - e_) * SLEEVE_K
                fco2[i] = c_ + rv / r_ * (r_ + (new_r - r_) * f_)
                nsl += 1
    REP["sleeve_slim"] = {"verts": nsl, "k": SLEEVE_K}
disp = fco2 - fco
DSM = int(JOB.get("disp_smooth", 12))
if DSM > 0 and not STATUE:
    # the pushes, pull-ins and slims above move each vertex on its own, which shears triangles (faceting) and
    # irons folds; smoothing the DISPLACEMENT (not the positions) keeps the garment's own surface and moves it as
    # a whole. Afterwards, anything left inside the visible head mesh goes out once more.
    _ed = np.array([e.vertices[:] for e in fig.data.edges])
    _deg = np.bincount(_ed.ravel(), minlength=len(fco)).astype(np.float64)
    for _ in range(DSM):
        acc = np.zeros_like(disp)
        np.add.at(acc, _ed[:, 0], disp[_ed[:, 1]])
        np.add.at(acc, _ed[:, 1], disp[_ed[:, 0]])
        disp = 0.5 * disp + 0.5 * acc / np.maximum(_deg, 1)[:, None]
    if HEAD is not None and bvhs:
        hb_ = bvhs[-1]
        nfix = 0
        for i in range(len(fco)):
            if np.linalg.norm(fco[i, :2] - _nax) > HEAD_PUSH_R + 0.05:
                continue
            if NECK_RADIAL and fco[i, 2] > float(J("clavicle_l").z) - 0.03 and np.linalg.norm(fco[i, :2] - _nax) < 0.16:
                continue
            q = Vector(fco[i] + disp[i])
            loc, nrm, _, dist = hb_.find_nearest(q, 0.05)
            if loc is None:
                continue
            s_ = (q - loc).dot(nrm)
            if s_ < EASE_ALL:
                disp[i] = disp[i] + np.array(nrm) * (EASE_ALL - s_)
                nfix += 1
        REP["disp_smooth"] = {"rounds": DSM, "head_refix_verts": nfix}
# the collar's height: a generated figure's collar and cravat reach its chin, but its neck is shorter than the
# MetaHuman's, so after the fit they can end centimetres under the jaw (a bare neck). Measured: the front top of the
# cloth round the neck against the head mesh's chin; the whole garment is lifted by one smooth ramp from the hip line
# (nothing) to the collar (full), sleeves following their shoulder end, so its top sits COLLAR_UNDER_CHIN under the
# chin and the collar keeps its generated shape and texture. REP["bare_neck_cm"] is what remains.
COLLAR_LIFT = bool(JOB.get("collar_lift", False))
# the garment's neckline (chin / base / open), the neck check's limit: the text's fact when it states one, else the
# class measured on the generated figure above
REP["neckline"] = JOB.get("neckline") or (REP.get("neckline_figure") or {}).get("class")
REP["neckline_source"] = "text" if JOB.get("neckline") else ("figure" if REP["neckline"] else None)
if not STATUE and HEAD is not None:
    P_ = fco + disp
    nc_ = np.array(J("neck_01"))
    hv_ = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
    prof_ = hv_[np.abs(hv_[:, 0] - nc_[0]) < 0.012]
    zs_ = np.arange(J("neck_02").z - 0.03, J("head").z + 0.02, 0.003)
    fy_ = np.array([prof_[np.abs(prof_[:, 2] - z) < 0.0015, 1].min() if (np.abs(prof_[:, 2] - z) < 0.0015).any()
                    else np.nan for z in zs_])
    ok_ = ~np.isnan(fy_)
    chin_ = float(zs_[ok_][int(np.argmin(np.diff(fy_[ok_])))]) if ok_.sum() > 3 else float(J("head").z - 0.07)
    rr_ = np.linalg.norm(P_[:, :2] - nc_[:2], axis=1)
    front_ = (np.abs(P_[:, 0] - nc_[0]) < 0.03) & (P_[:, 1] < nc_[1] - 0.02) & (rr_ < 0.13)
    z_sh = float(J("clavicle_l").z) - 0.02
    top0 = float(P_[front_ & (P_[:, 2] < chin_), 2].max()) if (front_ & (P_[:, 2] < chin_)).any() else None
    REP["collar_height"] = {"chin_z": round(chin_, 3), "collar_top_front_z": round(top0, 3) if top0 else None}
    # the innermost cloth round the neck (a shirt collar, a cravat): per 22.5-degree sector round the neck's axis,
    # the cloth within 2.5 cm of that sector's smallest radius between the shoulder line and the chin. Only this
    # layer is lifted or laid: an outer coat collar and the lapels are never moved.
    ang_ = np.arctan2(P_[:, 1] - nc_[1], P_[:, 0] - nc_[0])
    sec_ = ((ang_ + math.pi) / (2 * math.pi) * 16).astype(int) % 16
    band_ = (P_[:, 2] > z_sh) & (P_[:, 2] < chin_) & (rr_ < 0.15)
    rmin_ = np.full(16, 9.0)
    for k_ in range(16):
        m_ = band_ & (sec_ == k_)
        if m_.any():
            rmin_[k_] = float(np.quantile(rr_[m_], 0.03))
    inner_w = np.clip((rmin_[sec_] + 0.035 - rr_) / 0.02, 0, 1) * (rr_ < 0.15)
    # the neckline measured on the garment itself (a beard hides a neckline from a picture, not from the mesh): the
    # innermost cloth round the neck, its front top between the shoulder line and the chin, and its front opening's
    # radius against the neck's: up the neck to the chin / at the base of the neck / lower and open
    try:
        _hn = hv_[np.abs(hv_[:, 2] - float(J("neck_01").z)) < 0.012]
        _neck_r = float(np.median(np.linalg.norm(_hn[:, :2] - nc_[:2], axis=1))) if len(_hn) > 20 else 0.06
        _fr = (ang_ > -math.pi * 0.75) & (ang_ < -math.pi * 0.25)          # in front of the neck (-y)
        _front_inner = band_ & _fr & (inner_w > 0.5)
        _top_f = float(P_[_front_inner, 2].max()) if _front_inner.any() else None
        _r_f = float(np.median(rmin_[[k for k in range(16) if -0.75 * math.pi < (k + 0.5) / 16 * 2 * math.pi - math.pi
                                      < -0.25 * math.pi]]))
        _frac = (_top_f - z_sh) / max(chin_ - z_sh, 1e-3) if _top_f is not None else 0.0
        if _frac >= 0.55 and _r_f < _neck_r * 1.6:
            _cls = "chin"
        elif _top_f is not None and _top_f >= z_sh - 0.02 and _r_f < _neck_r * 2.0:
            _cls = "base"
        else:
            _cls = "open"
        REP["neckline_measured"] = {"class": _cls, "front_top_frac": round(_frac, 3),
                                    "front_radius_cm": round(_r_f * 100, 1), "neck_radius_cm": round(_neck_r * 100, 1)}
    except Exception as _e:          # a measure, never a failure of the fit
        REP["neckline_measured"] = {"error": str(_e)}
    REP["collar_height"]["inner_layer_verts"] = int((inner_w > 0.5).sum())
    BARE_MAX = float(JOB.get("bare_neck_max_m", 0.02))
    REP["bare_neck_measured_cm"] = round((chin_ - top0) * 100, 1) if top0 is not None else None
    if top0 is not None and ((chin_ - top0) <= BARE_MAX or not COLLAR_LIFT):
        REP["bare_neck_cm"] = round((chin_ - top0) * 100, 1)       # already under the limit: nothing is moved
        top0 = None
    if top0 is not None:
        lift = (chin_ - float(JOB.get("collar_under_chin_m", 0.012))) - top0
        if lift > 0.003 and top0 > z_sh + 0.02:
            # the whole garment rises, one smooth ramp from the hip line (nothing) to the collar (the full lift): a
            # 3 cm rise over 50-60 cm is a 5 % stretch nobody sees, so the collar keeps its generated shape and texture
            # and only sits higher (a lift over the chest or the inner layer alone stretched and smeared the collar
            # into a tall band or a bib). A sleeve follows its shoulder end, the rise fading to nothing at the cuff.
            z_lo_ = float(J("pelvis").z)
            t_ = np.clip((P_[:, 2] - z_lo_) / (top0 - z_lo_), 0, 1)
            t_ = t_ * t_ * (3 - 2 * t_)
            gidx = {g.index: g.name for g in fig.vertex_groups}
            armw_ = np.zeros(len(P_))
            for v in fig.data.vertices:
                for g in v.groups:
                    nm = gidx.get(g.group, "")
                    if nm.startswith(("upperarm", "lowerarm", "hand", "clavicle")) or "_finger" in nm:
                        armw_[v.index] += g.weight if not nm.startswith("clavicle") else 0.0
            armw_ = np.clip(armw_, 0, 1)
            dz_arm = np.zeros(len(P_))
            for side, sg in (("l", 1), ("r", -1)):
                S0 = np.array(J(f"upperarm_{side}"))
                Hd = np.array(J(f"hand_{side}"))
                L_ = max(np.linalg.norm(Hd - S0), 0.1)
                ax_ = (Hd - S0) / L_
                ts0 = float(np.clip((S0[2] - z_lo_) / (top0 - z_lo_), 0, 1))
                ts0 = ts0 * ts0 * (3 - 2 * ts0)
                on = sg * (P_[:, 0] - float(J("pelvis").x)) > 0
                along = np.clip(((P_ - S0) @ ax_) / L_, 0, 1)
                dz_arm[on] = lift * ts0 * (1 - along[on])
            dz = (1 - armw_) * lift * t_ + armw_ * dz_arm
            disp[:, 2] += dz
            REP.setdefault("collar_height", {})["lift_shape"] = "torso"
        if lift > 0.003 and top0 > z_sh + 0.02:
            # anything the lift took into the head mesh goes out again
            hb2 = BVHTree.FromObject(HEAD, bpy.context.evaluated_depsgraph_get())
            for i in np.nonzero(dz > 1e-4)[0]:
                q = Vector(fco[i] + disp[i])
                loc, nrm, _, dist = hb2.find_nearest(q, 0.05)
                if loc is not None and (q - loc).dot(nrm) < EASE_ALL:
                    disp[i] = disp[i] + np.array(nrm) * (EASE_ALL - (q - loc).dot(nrm))
            REP["collar_height"]["lift_cm"] = round(lift * 100, 1)
            top0 = top0 + lift
        REP["bare_neck_cm"] = round((chin_ - top0) * 100, 1)
        # a clean top edge: the cut at the neck leaves a torn line; everything round the neck that stands above a
        # smooth tilted surface just under the jaw (front at the collar top, rising toward the nape) is laid down
        # onto it (no vertex removed: the stance settle keeps its vertex ids), so the cloth ends in one even loop
        if JOB.get("collar_clean_edge", True):
            P2 = fco + disp
            rr2 = np.linalg.norm(P2[:, :2] - nc_[:2], axis=1)
            tilt_c = math.radians(float(JOB.get("collar_edge_tilt_deg", 22)))
            z_front = min(top0, chin_ - 0.004)
            # height of the edge surface at each vertex: front at z_front, rising backward (+y) by tan(tilt)
            zedge = z_front + np.maximum(P2[:, 1] - (nc_[1] - 0.06), 0) * math.tan(tilt_c)
            sel = (rr2 < float(JOB.get("collar_edge_r_m", 0.11))) & (P2[:, 2] > zedge) & (inner_w > 0.5)
            disp[sel, 2] -= (P2[sel, 2] - zedge[sel])
            REP["collar_height"]["edge_laid_verts"] = int(sel.sum())
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
    for _ in range(int(JOB.get("cravat_smooth", 0))):
        acc = np.zeros_like(P)
        np.add.at(acc, ed[:, 0], P[ed[:, 1]])
        np.add.at(acc, ed[:, 1], P[ed[:, 0]])
        avg = acc / np.maximum(deg, 1)[:, None]
        P[cm] = 0.5 * P[cm] + 0.5 * avg[cm]
    disp = P - fco
    fig.data.vertices.foreach_set("co", P.ravel())
    fig.data.update()
REP["cravat_verts_smoothed"] = int(cm.sum())
ED = np.array([e.vertices[:] for e in fig.data.edges])
DEG = np.bincount(ED.ravel(), minlength=len(fco)).astype(np.float64)
if JOB.get("collar_build") and not STATUE:
    # the edges left round the removed shirt front (lapels, the coat's collar) are ragged: smooth each open
    # boundary along itself (boundary neighbours only) and let the next ring follow, near the neck and chest
    P = fco + disp
    fe = np.array([(p.vertices[k], p.vertices[(k + 1) % len(p.vertices)]) for p in fig.data.polygons
                   for k in range(len(p.vertices))])
    fe_s = np.sort(fe, 1)
    u_, cnt_ = np.unique(fe_s, axis=0, return_counts=True)
    bed = u_[cnt_ == 1]
    nc3 = np.array(J("neck_01"))
    near = (P[:, 2] > J("spine_03").z) & ((P[:, 1] < nc3[1] + 0.02) | (np.linalg.norm(P[:, :2] - nc3[:2], axis=1) < 0.12))
    bed = bed[near[bed[:, 0]] & near[bed[:, 1]]]
    bv = np.zeros(len(P), bool)
    bv[bed.ravel()] = True
    bdeg = np.bincount(bed.ravel(), minlength=len(P)).astype(np.float64)
    ring1 = np.zeros(len(P), bool)
    m1 = bv[ED[:, 0]] | bv[ED[:, 1]]
    ring1[ED[m1].ravel()] = True
    ring1 &= ~bv
    for _ in range(int(JOB["collar_build"].get("edge_smooth", 12))):
        acc = np.zeros_like(P)
        np.add.at(acc, bed[:, 0], P[bed[:, 1]])
        np.add.at(acc, bed[:, 1], P[bed[:, 0]])
        ok_ = bv & (bdeg >= 2)
        P[ok_] = 0.5 * P[ok_] + 0.5 * acc[ok_] / bdeg[ok_, None]
        acc = np.zeros_like(P)
        np.add.at(acc, ED[:, 0], P[ED[:, 1]])
        np.add.at(acc, ED[:, 1], P[ED[:, 0]])
        P[ring1] = 0.7 * P[ring1] + 0.3 * acc[ring1] / np.maximum(DEG[ring1], 1)[:, None]
    disp = P - fco
    fig.data.vertices.foreach_set("co", P.ravel())
    fig.data.update()
    REP["hole_edge_smoothed"] = int(bv.sum())


def smooth_pos(P, w, rounds, lam=0.5, mu=-0.53):
    """Taubin smoothing (shrink, then inflate) of P over the mesh edges, by per-vertex weight w (0 = fixed)"""
    for _ in range(rounds):
        for f_ in (lam, mu):
            acc = np.zeros_like(P)
            np.add.at(acc, ED[:, 0], P[ED[:, 1]])
            np.add.at(acc, ED[:, 1], P[ED[:, 0]])
            avg = acc / np.maximum(DEG, 1)[:, None]
            P = P + (f_ * w)[:, None] * (avg - P)
    return P


# the coat's shoulders: the drawn shoulder seam and the pull-in above leave a flat top with a hard corner over the
# arm; round it (Taubin smoothing, no shrink) over the shoulder cap, fading out SH_ROUND from the shoulder joint
SH_ROUND = float(JOB.get("shoulder_round_m", 0.0))      # smoothing flattens the garment: off
P = fco + disp
if not STATUE and SH_ROUND > 0:
    wsh = np.zeros(len(P))
    for side in ("l", "r"):
        S0 = np.array(pose_head(f"upperarm_{side}"))
        dd = np.linalg.norm(P - S0, axis=1)
        wsh = np.maximum(wsh, np.clip(1 - dd / SH_ROUND, 0, 1) * (P[:, 2] > S0[2] - 0.08))
    P = smooth_pos(P, np.clip(wsh * 1.6, 0, 1), int(JOB.get("shoulder_round", 12)))
    REP["shoulder_round_verts"] = int((wsh > 0).sum())
# what the outfit must cover: the MetaHuman head mesh reaches down over the trapezius and collarbones (its own
# shoulder flaps, weighted to spine_04, the clavicles and the arms' corrective joints). Every outfit vertex over
# that skin, below the collar line, ends at least FLAP_EASE outside it (after the pull-ins above, which can sink
# the coat into it), and later takes the skin's own weights so the two move together in every pose.
FLAP_EASE = float(JOB.get("flap_ease_m", 0.0))     # > 0: also push the cloth that far out of the skin
FLAP_NEAR = float(JOB.get("flap_follow_m", 0.05))
flap_w = np.zeros(len(P))
flap_src = np.full(len(P), -1)
if not STATUE and HEAD is not None:
    hbvh_ = BVHTree.FromObject(HEAD, bpy.context.evaluated_depsgraph_get())
    hpoly = HEAD.data.polygons
    below = (P[:, 2] - cut_z) * math.cos(tilt) - (P[:, 1] - J("neck_01").y) * math.sin(tilt) < 0
    npush = 0
    for i in np.nonzero(below & (P[:, 2] > J("spine_03").z))[0]:
        loc, nrm, fi, dist = hbvh_.find_nearest(Vector(P[i]), FLAP_NEAR)
        if loc is None:
            continue
        sd = (Vector(P[i]) - loc).dot(nrm)
        if FLAP_EASE > 0 and sd < FLAP_EASE:
            P[i] = P[i] + np.array(nrm) * (FLAP_EASE - sd)
            npush += 1
        flap_w[i] = float(np.clip(1 - (dist - 0.02) / (FLAP_NEAR - 0.02), 0, 1))
        vv = hpoly[fi].vertices
        hv_ = np.array([HEAD.data.vertices[k].co[:] for k in vv])
        flap_src[i] = vv[int(np.argmin(np.linalg.norm(hv_ - P[i], axis=1)))]
    REP["flap_pushed_verts"] = npush
    REP["flap_follow_verts"] = int((flap_w > 0).sum())
disp = P - fco
fig.data.vertices.foreach_set("co", P.ravel())
fig.data.update()
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
if (flap_w > 0).any():
    # over the head mesh's shoulder skin the outfit takes that skin's weights (bones the outfit's skeleton has),
    # blended in by nearness, so the skin cannot rise through the cloth when a clavicle or corrective joint moves
    hgn = {g.index: g.name for g in HEAD.vertex_groups}
    have = {g.name for g in fig.vertex_groups}
    bnames = {b.name for b in arm.data.bones}
    for i in np.nonzero(flap_w > 0)[0]:
        hv_ = HEAD.data.vertices[int(flap_src[i])]
        src = {hgn[g.group]: g.weight for g in hv_.groups if hgn[g.group] in bnames and g.weight > 1e-3}
        tot = sum(src.values())
        if tot <= 0:
            continue
        a_ = float(flap_w[i])
        v_ = fig.data.vertices[int(i)]
        cur = {fig.vertex_groups[g.group].name: g.weight for g in v_.groups}
        ct = sum(cur.values()) or 1.0
        new = {n: (1 - a_) * w / ct for n, w in cur.items()}
        for n, w in src.items():
            new[n] = new.get(n, 0.0) + a_ * w / tot
        for n, w in new.items():
            if n not in have:
                fig.vertex_groups.new(name=n)
                have.add(n)
            fig.vertex_groups[n].add([int(i)], float(w), "REPLACE")
if not STATUE and HEAD is not None:
    # coverage check: head-mesh skin below the collar line whose outward ray meets no outfit within 8 cm shows
    obvh = BVHTree.FromObject(fig, bpy.context.evaluated_depsgraph_get())
    nshow = nlow = 0
    for v in HEAD.data.vertices:
        p = v.co
        if (p.z - cut_z) * math.cos(tilt) - (p.y - J("neck_01").y) * math.sin(tilt) > -0.02:
            continue
        nlow += 1
        hit = obvh.ray_cast(p + v.normal * 0.001, v.normal, 0.08)
        if hit[0] is None:
            nshow += 1
    REP["head_skin_below_collar"] = {"verts": nlow, "uncovered": nshow}
bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=int(JOB.get("max_influences", 4)))
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
stage_save("s3_fitted_weighted")

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
YOKE = ("clavicle", "spine_04", "spine_05", "neck")
gi = {g.index: g.name for g in hands.vertex_groups}
keep = np.zeros(len(hands.data.vertices), bool)
for v in hands.data.vertices:
    best = max(v.groups, key=lambda g: g.weight, default=None)
    if best is not None and gi[best.group].startswith(HANDB) and "toe" not in gi[best.group]:
        keep[v.index] = True
# the body's skin is hidden only where cloth covers it (the body itself is hidden under the outfit, so a bare
# forearm below a short sleeve, a shin below a hem or a leg the seated drape leaves open showed as nothing, 11-12):
# every body vertex off the head mesh that has no cloth within BODY_NEAR of it and none along its outward normal
# within BODY_REACH is kept with the hands in the skin slot (the head mesh keeps the neck)
if not STATUE and JOB.get("body_skin_shows", True):
    _dg = bpy.context.evaluated_depsgraph_get()
    _gb = BVHTree.FromObject(fig, _dg)
    _near = float(JOB.get("body_near_m", 0.03))
    _reach = float(JOB.get("body_reach_m", 0.15))
    # the head mesh keeps the neck: body vertices on or inside it (within HEAD_NEAR of its surface) stay hidden
    _hbv = BVHTree.FromObject(HEAD, _dg) if HEAD is not None else None
    _hnear = float(JOB.get("body_head_near_m", 0.015))
    hands.data.update()
    # every body vertex off the head goes into the skin slot, marked "body_skin", and the ones cloth covers in this
    # stance "covered_standing": they are cut just before this outfit's export, and a stance's settle (bl_settle.py)
    # tests them again against its own settled cloth (a leg a seated drape leaves open must show, 11-12)
    _shown = np.zeros(len(hands.data.vertices), bool)
    _elig = np.zeros(len(hands.data.vertices), bool)
    for v in hands.data.vertices:
        if keep[v.index]:
            continue
        if _hbv is not None and _hbv.find_nearest(v.co, _hnear)[0] is not None:
            continue
        if v.co.z > float(J("head").z):
            continue
        # the shoulders and the top of the trunk are never bare skin of the body's own: round the neck the head
        # mesh is the skin that shows, and under a collar's edge the body's slope looked out past the cloth (its
        # normal met none) and stood out of every garment as skin-coloured horns at the shoulders
        _b = max(v.groups, key=lambda g: g.weight, default=None)
        if _b is not None and gi[_b.group].startswith(YOKE):
            continue
        # an arm's skin shows only beyond its sleeve's cuff (CUFF: found where a short sleeve ends); above it, and on
        # a long-sleeved arm anywhere, a gap in the cloth is a gap, not bare skin (it showed as patches of skin on
        # the shoulders of a man in a flight suit)
        if _b is not None and gi[_b.group].startswith(("upperarm", "lowerarm")):
            _cf = CUFF.get(gi[_b.group][-1])
            if _cf is None or float((np.array(v.co) - np.array(_cf[0])) @ np.array(_cf[1])) < float(_cf[2]) - 0.02:
                continue
        _elig[v.index] = True
        loc_, _, _, d_ = _gb.find_nearest(v.co, _near)
        if loc_ is not None:
            continue
        if _gb.ray_cast(v.co + v.normal * 0.002, v.normal, _reach)[0] is not None:
            continue
        _shown[v.index] = True
    keep |= _elig
    # a point attribute, not a vertex group (a group would count as a bone influence in the weight limits):
    # 0 not body skin (cloth, hands), 1 body skin shown, 2 body skin cloth covers in this stance
    _st = np.where(_elig, np.where(_shown, 1, 2), 0).astype(np.int32)
    _at = hands.data.attributes.new("body_skin", "INT", "POINT")
    _at.data.foreach_set("value", _st)
    REP["body_skin_shown_verts"] = int(_shown.sum())
# a bare limb's thickness from the generated figure: the MetaHuman's limb constraints (Forearm, Wrist, Hand) changed
# the solved body not at all (11-12: Forearm 22.6 -> 35.7 left every measured girth as it was, as Neck Length did), so
# the body's shown limb is thickened here, round its own axis, station by station: to the figure's forearm girth just
# below the cuff, tapering to the figure's wrist at the wrist; never wider than a little under the sleeve's opening;
# the wrist held in the body's own proportion to the hand (the figure's hand against the body's: a wrist wider than
# its hand is the 3D model's blobbing, not the drawing); the hand as it is; never thinner than the body's own
REP["limb_thicken"] = {}
for side, (S0_, ax_, c_) in CUFF.items():
    lg_ = LIMB_GIRTH.get(f"arm_{side}") or {}
    if "Forearm" not in lg_ or "Wrist" not in lg_:
        continue
    sg_ = 1 if side == "l" else -1
    L_ = float(np.linalg.norm(np.array(pose_head(f"hand_{side}")) - S0_))
    BVp_ = np.array([v.co[:] for v in hands.data.vertices])
    fh, bh = _girth_at(co, S0_, ax_, L_ + 0.05, sg_), _girth_at(BVp_, S0_, ax_, L_ + 0.05, sg_)
    cap_ = lg_.get("_cap_cm") or 1e9
    t_f = min(lg_["Forearm"]["figure_cm"], cap_)
    t_w = min(lg_["Wrist"]["figure_cm"], cap_)
    if fh and bh:
        t_w = min(t_w, lg_["Wrist"]["body_cm"] * fh / bh)
    p_ = (BVp_ - S0_) @ ax_
    rr_ = (BVp_ - S0_) - p_[:, None] * ax_[None, :]
    rad_ = np.linalg.norm(rr_, axis=1)
    sel_ = keep & (sg_ * (BVp_[:, 0] - J("pelvis").x) > 0) & (p_ > c_ - 0.06) & (p_ < L_ + 0.02) & (rad_ < 0.12)
    ks, k_w = [], 1.0
    for st in np.arange(c_ - 0.06, L_ + 0.02, 0.01):
        m_ = sel_ & (np.abs(p_ - st) < 0.005)
        if m_.sum() < 6:
            continue
        b_ = _girth_at(BVp_, S0_, ax_, st, sg_)
        if not b_:
            continue
        if st <= L_ - 0.04:
            w_ = float(np.clip((st - c_) / max(L_ - 0.04 - c_, 0.02), 0, 1))
            k_ = max(1.0, ((1 - w_) * t_f + w_ * t_w) / b_)
            k_w = k_
        else:                                     # wrist to hand: back to the body's own by the hand
            k_ = 1.0 + (k_w - 1.0) * max(0.0, 1 - (st - (L_ - 0.04)) / 0.06)
        ctr = BVp_[m_].mean(0)
        dv = BVp_[m_] - ctr
        dv -= (dv @ ax_)[:, None] * ax_[None, :]
        BVp_[m_] = BVp_[m_] + (k_ - 1) * dv
        ks.append((round(st / L_, 3), round(k_, 2)))
    hands.data.vertices.foreach_set("co", BVp_.ravel())
    hands.data.update()
    REP["limb_thicken"][f"arm_{side}"] = {"forearm_target_cm": round(t_f, 1), "wrist_target_cm": round(t_w, 1),
                                          "opening_cm": lg_.get("_opening_cm"), "cap_cm": lg_.get("_cap_cm"),
                                          "figure_hand_cm": fh, "body_hand_cm": bh, "scale_by_station": ks}
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
if HAIR is not None and HEAD is not None and HAIR_KEPT:
    # the kept hair set on the MetaHuman's head: the drawn head's place and size (from the drawn face: its middle,
    # its front, its chin, its width) onto the MetaHuman head's own, one scale and one shift for the whole piece;
    # then whatever of it lies inside the skull or the neck is moved out to just off the skin. It keeps its own
    # volume and its own colours. On the head it follows the head; hair hanging below the neck's base follows the
    # upper back more the lower it hangs.
    hv = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
    _hz = float(J("head").z)
    _chin_m = (REP.get("collar_height") or {}).get("chin_z") or _hz - 0.03
    _fz0, _fz1 = HAIR["chin_z"], HAIR["face_top_z"]
    # the MetaHuman's face over the same share of its head: its width and front at cheek height
    _cheek = hv[(hv[:, 2] > _chin_m + 0.05) & (hv[:, 2] < _chin_m + 0.11)]
    _mw = float(np.quantile(_cheek[:, 0], 0.98) - np.quantile(_cheek[:, 0], 0.02)) if len(_cheek) > 50 else 0.15
    _dw = max(0.08, HAIR["face_x"][1] - HAIR["face_x"][0])
    _s = float(min(1.2, max(0.85, _mw / _dw)))
    _cd = np.array([sum(HAIR["face_x"]) / 2, HAIR["face_front_y"], _fz0])
    _cm = np.array([float(np.median(_cheek[:, 0])) if len(_cheek) > 50 else float(J("head").x),
                    float(np.quantile(_cheek[:, 1], 0.03)) if len(_cheek) > 50 else float(J("head").y) - 0.09, _chin_m])
    P = np.array([v.co[:] for v in HAIR.data.vertices])
    P = _cm + (P - _cd) * _s
    hbvh = BVHTree.FromObject(HEAD, bpy.context.evaluated_depsgraph_get())
    HE = float(JOB.get("hair_ease_m", 0.004))
    _moved = 0
    for i, p_ in enumerate(P):
        loc, nrm, _, dist = hbvh.find_nearest(Vector(p_), 0.08)
        if loc is not None and (Vector(p_) - loc).dot(nrm) < HE:
            P[i] = np.array(loc) + np.array(nrm) * HE
            _moved += 1
    HAIR.data.vertices.foreach_set("co", P.ravel())
    HAIR.vertex_groups.clear()
    hg, sg_ = HAIR.vertex_groups.new(name="head"), HAIR.vertex_groups.new(name="spine_05")
    _nz = float(J("neck_01").z)
    for i, p_ in enumerate(P):
        w_ = float(min(1.0, max(0.0, (_nz - p_[2]) / 0.12)))            # 0 at the neck's base, 1 a hand lower
        hg.add([i], 1.0 - w_, "REPLACE")
        if w_ > 0:
            sg_.add([i], w_, "REPLACE")
    REP["hair_kept"] = {"verts": int(len(P)), "scale": round(_s, 3), "shift_m": [round(float(v), 3) for v in (_cm - _cd)],
                        "moved_off_the_head": _moved, "lowest_z_m": round(float(P[:, 2].min()), 3),
                        "face_width_m": {"drawn": round(_dw, 3), "metahuman": round(_mw, 3)}}
    if JOB.get("hair_debug_npz"):
        np.savez_compressed(JOB["hair_debug_npz"], hair=P, head=hv)
    bpy.ops.object.select_all(action="DESELECT")
    fig.select_set(True)
    HAIR.select_set(True)
    bpy.context.view_layer.objects.active = fig
    bpy.ops.object.join()
elif HAIR is not None and HEAD is not None:
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
# constructed neckwear (bl_collar.py) joins before the liner, so the liner sees it as cover
PIECES = []
if JOB.get("collar_build") and not STATUE and HEAD is not None:
    t_c = time.time()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bl_collar  # noqa: E402
    cr = dict(COLLAR_RGB or {})
    # white linen indoors: an albedo over ~0.72 blows out in window light (the engine's exposure is set for the room)
    cmax = float(JOB["collar_build"].get("albedo_max", 0.72))
    # a cravat reads as dark silk: at most CRAV_MAX (the sheet's dark grey; a mid grey read as a white neck brace)
    if "cravat" in cr and max(cr["cravat"]) > float(JOB["collar_build"].get("cravat_albedo_max", 0.22)):
        m_ = max(cr["cravat"])
        cr["cravat"] = [c_ * float(JOB["collar_build"].get("cravat_albedo_max", 0.22)) / m_ for c_ in cr["cravat"]]
    for k_ in list(cr):
        m_ = max(cr[k_])
        if m_ > cmax:
            cr[k_] = [c_ * cmax / m_ for c_ in cr[k_]]
    sw = bl_collar.free_swatches(fig, img, [cr.get("collar", [0.72, 0.72, 0.70]), cr.get("cravat", [0.3, 0.3, 0.32]),
                                            cr.get("dress", [0.05, 0.05, 0.05])])
    PIECES = bl_collar.build(J, HEAD, body, arm, img, JOB, REP,
                             {"_uv": {"collar": sw[0], "cravat": sw[1], "dress": sw[2]},
                              "_uv_name": fig.data.uv_layers.active.name}, PH=pose_head)
    nfig = len(fig.data.vertices)
    for o in PIECES:
        o.data.materials.clear()
    bpy.ops.object.select_all(action="DESELECT")
    fig.select_set(True)
    for o in PIECES:
        o.select_set(True)
    bpy.context.view_layer.objects.active = fig
    bpy.ops.object.join()
    REP["collar_build"]["s"] = round(time.time() - t_c, 2)
    JOB.setdefault("liner_hole_m", 0.08)
LINER = None
if not STATUE and HEAD is not None and JOB.get("liner", True):
    # hide the head mesh's skin under the outfit: the part of it the outfit covers (its outward ray meets the
    # outfit within LINER_REACH), grown by LINER_RINGS edge rings below the collar line, is copied as a thin liner
    # LINER_OFF outside the skin, painted with the outfit's own colour at that spot and carrying the skin's own
    # weights; a gap or a thin spot in the generated cloth then shows cloth, never skin, in any pose
    t_l = time.time()
    REACH = float(JOB.get("liner_reach_m", 0.10))
    obvh = BVHTree.FromObject(fig, bpy.context.evaluated_depsgraph_get())
    hv = HEAD.data.vertices
    nH = len(hv)
    Hco = np.array([v.co[:] for v in hv])
    Hn = np.array([v.normal[:] for v in hv])
    low_ = ((Hco[:, 2] - cut_z) * math.cos(tilt) - (Hco[:, 1] - J("neck_01").y) * math.sin(tilt)
            < -float(JOB.get("liner_below_collar_m", 0.02)))
    hr = np.linalg.norm(Hco[:, :2] - np.array(J("neck_01"))[:2], axis=1)
    # liner_min_r_m: only the head mesh's skin this far from the neck's axis (its shoulder flaps) is lined; near the
    # neck the liner showed as dark spikes above the collar
    low_ &= hr > float(JOB.get("liner_min_r_m", 0.0))
    HOLE_R = float(JOB.get("liner_hole_min_r_m", 0.08))
    cov = np.zeros(nH, bool)
    hitf = np.full(nH, -1)
    hitp = np.zeros((nH, 3))
    for i in np.nonzero(low_)[0]:
        loc, nrm, fi, d = obvh.ray_cast(Vector(Hco[i] + Hn[i] * 0.001), Vector(Hn[i]), REACH)
        if loc is None and hr[i] > HOLE_R:  # a hole in the cloth over the shoulders (not the neck, which shows
            # above the collar): covered when cloth lies all round it
            loc, nrm, fi, d = obvh.find_nearest(Vector(Hco[i]), float(JOB.get("liner_hole_m", 0.05)))
        if loc is not None:
            cov[i] = True
            hitf[i] = fi
            hitp[i] = loc[:]
    hed = np.array([e.vertices[:] for e in HEAD.data.edges])
    grown = cov.copy()
    for _ in range(int(JOB.get("liner_rings", 2))):
        g2 = grown.copy()
        g2[hed[:, 0]] |= grown[hed[:, 1]]
        g2[hed[:, 1]] |= grown[hed[:, 0]]
        grown = g2 & low_
    # the ring-grown vertices take the cloth point of their nearest covered neighbour
    for _ in range(int(JOB.get("liner_rings", 2)) + 1):
        miss = grown & (hitf < 0)
        if not miss.any():
            break
        for a_, b_ in ((0, 1), (1, 0)):
            m_ = miss[hed[:, a_]] & (hitf[hed[:, b_]] >= 0)
            hitf[hed[m_, a_]] = hitf[hed[m_, b_]]
            hitp[hed[m_, a_]] = hitp[hed[m_, b_]]
    keep_l = grown & (hitf >= 0)
    if keep_l.sum() > 50:
        LINER = HEAD.copy()
        LINER.data = HEAD.data.copy()
        bpy.context.scene.collection.objects.link(LINER)
        LO = float(JOB.get("liner_off_m", 0.004))
        LINER.data.vertices.foreach_set("co", (Hco + Hn * LO).ravel())
        # colour: one texel of the atlas, the one nearest the cloth's median colour over the shoulders (per-hit
        # texels pick up the drawn skin and lining at the collar)
        fc_ = np.empty(len(fig.data.vertices) * 4, np.float32)
        fig.data.attributes["fcol"].data.foreach_get("color", fc_)
        fc_ = fc_.reshape(-1, 4)[:, :3]
        FP = np.array([v.co[:] for v in fig.data.vertices])
        shz = np.zeros(len(FP), bool)
        for side in ("l", "r"):
            shz |= np.linalg.norm(FP - np.array(pose_head(f"upperarm_{side}")), axis=1) < 0.15
        want_c = np.median(fc_[shz], 0) if shz.sum() > 20 else np.median(fc_, 0)
        sub = px[::4, ::4]
        # the texel must sit in an even patch of that colour (bilinear filtering blends its neighbours)
        dc = np.linalg.norm(sub[..., :3] - want_c, axis=2) + (sub[..., 3] < 0.5) * 9
        dmax = dc.copy()
        for dy_ in (-1, 0, 1):
            for dx_ in (-1, 0, 1):
                dmax = np.maximum(dmax, np.roll(np.roll(dc, dy_, 0), dx_, 1))
        dc = dmax
        yy, xx = np.unravel_index(np.argmin(dc), dc.shape)
        vuv = np.tile([(xx * 4 + 0.5) / W, (yy * 4 + 0.5) / H], (nH, 1))
        REP["liner_rgb"] = [round(float(x), 3) for x in want_c]
        me_l = LINER.data
        while len(me_l.uv_layers) > 1:
            me_l.uv_layers.remove(me_l.uv_layers[-1])
        if not me_l.uv_layers:
            me_l.uv_layers.new(name="UVMap")
        # the join matches UV layers by name: under another name the liner's UVs would land in a second layer and
        # read (0, 0) in the outfit's own
        me_l.uv_layers[0].name = fig.data.uv_layers.active.name
        lvi = np.empty(len(me_l.loops), np.int64)
        me_l.loops.foreach_get("vertex_index", lvi)
        me_l.uv_layers[0].data.foreach_set("uv", vuv[lvi].astype(np.float32).ravel())
        # weights: the skin's, on bones the outfit's skeleton has
        bn = {b.name for b in arm.data.bones}
        for g in list(LINER.vertex_groups):
            if g.name not in bn:
                LINER.vertex_groups.remove(g)
        ng = LINER.vertex_groups.get("neck_01") or LINER.vertex_groups.new(name="neck_01")
        ng.add([v.index for v in me_l.vertices if not v.groups], 1.0, "REPLACE")
        for p_ in me_l.polygons:
            p_.material_index = 0
        lb = bmesh.new()
        lb.from_mesh(me_l)
        lb.verts.ensure_lookup_table()
        bmesh.ops.delete(lb, geom=[lb.verts[i] for i in np.nonzero(~keep_l)[0]], context="VERTS")
        lb.to_mesh(me_l)
        lb.free()
        for a_ in list(me_l.attributes):
            if a_.name == "fcol":
                me_l.attributes.remove(a_)
        me_l.materials.clear()
        REP["liner"] = {"covered": int(cov.sum()), "verts": len(me_l.vertices), "s": round(time.time() - t_l, 2)}
if not STATUE and HEAD is not None and JOB.get("strip_hair", True) and LAB_GROUP is None:      # (with labels: done by them)
    # the generated figure's own hair hanging below the neck line (long loose hair over the shoulders) is attached to
    # its head, not to the garment, and stays after the cut. Removed by geometry (colour fails for black hair on a
    # black dress): above the armpit line, (a) islands of cloth not joined to the garment's main piece, and (b) cloth
    # within a head's width of the neck's axis standing off the body's and head's surface by more than STAND_OFF (a
    # collar lies on the neck, a puffed sleeve's top is further out to the side than a head's width)
    _nk = np.array(J("neck_01"))
    _z_arm = min(float(J("upperarm_l").z), float(J("upperarm_r").z)) - 0.04
    _fv0 = np.array([fig.matrix_world @ v.co for v in fig.data.vertices])
    _bm = bmesh.new()
    _bm.from_mesh(fig.data)
    _bm.verts.ensure_lookup_table()
    _comp = np.full(len(_bm.verts), -1)
    _nc = 0
    for v0 in _bm.verts:
        if _comp[v0.index] >= 0:
            continue
        stack = [v0]
        _comp[v0.index] = _nc
        while stack:
            v = stack.pop()
            for e in v.link_edges:
                w = e.other_vert(v)
                if _comp[w.index] < 0:
                    _comp[w.index] = _nc
                    stack.append(w)
        _nc += 1
    _sizes = np.bincount(_comp)
    _main = int(np.argmax(_sizes))
    _island = np.zeros(len(_fv0), bool)
    for k in range(_nc):
        if k == _main:
            continue
        m = _comp == k
        if np.median(_fv0[m, 2]) > _z_arm and _sizes[k] < 0.05 * len(_fv0):
            _island |= m
    _bvb = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    _bvh = BVHTree.FromObject(HEAD, bpy.context.evaluated_depsgraph_get())
    STAND_OFF = float(JOB.get("hair_stand_off_m", 0.05))
    # (b) only when the figure's hair hangs below the neck (the hair facts: long, or worn down): a coat's collar
    # rolled behind the neck stands off the body too (11-11 control on a short-haired man: 447 collar vertices)
    _cand = np.nonzero((_fv0[:, 2] > _z_arm) & (np.abs(_fv0[:, 0] - _nk[0]) < float(JOB.get("hair_half_width_m", 0.12)))
                       & ~_island)[0] if JOB.get("hair_hangs") else np.array([], int)
    _off = np.zeros(len(_fv0), bool)
    for i in _cand:
        q = Vector(_fv0[i])
        d1 = _bvb.find_nearest(q, STAND_OFF)[0]
        d2 = _bvh.find_nearest(q, STAND_OFF)[0]
        _off[i] = d1 is None and d2 is None
    _hair = _island | _off
    if _hair.any():
        bmesh.ops.delete(_bm, geom=[_bm.verts[i] for i in np.nonzero(_hair)[0]], context="VERTS")
        _bm.to_mesh(fig.data)
        fig.data.update()
    _bm.free()
    REP["hair_stripped_verts"] = {"islands": int(_island.sum()), "standing_off": int(_off.sum())}
    if JOB.get("hair_debug"):
        np.savez_compressed(JOB["hair_debug"], fig=_fv0, hair=_hair, neck=_nk, head=np.array(J("head")))
if not STATUE and HEAD is not None:
    # nothing of the garment in front of the face or inside the head above the chin: a generated head left on the
    # garment by a wrong neck line is thousands of vertices there (11-11); cast_char fails the build on it
    _hv = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
    _hx = np.array(J("head"))
    _chin = (REP.get("collar_height") or {}).get("chin_z") or float(J("head").z) - 0.03
    _fv = np.array([fig.matrix_world @ v.co for v in fig.data.vertices])
    _face_y = float(np.quantile(_hv[_hv[:, 2] > _chin, 1], 0.5)) if (_hv[:, 2] > _chin).any() else float(_hx[1])
    _sel = ((_fv[:, 2] > _chin + 0.01) & (_fv[:, 2] < float(_hv[:, 2].max()) + 0.05)
            & (np.abs(_fv[:, 0] - _hx[0]) < 0.10) & (_fv[:, 1] < _face_y))
    REP["head_volume_verts"] = int(_sel.sum())
    # all round the head above the chin (hair left at the back or the sides), away from a collar's stand
    _rr = np.linalg.norm(_fv[:, :2] - _hx[:2], axis=1)
    # above the collar's clean-edge surface (front at the chin, rising toward the nape at the edge's tilt)
    _zedge = _chin + 0.01 + np.maximum(_fv[:, 1] - (_hx[1] - 0.06), 0) * math.tan(math.radians(
        float(JOB.get("collar_edge_tilt_deg", 22)))) + 0.01
    _round = ((_fv[:, 2] > _zedge) & (_fv[:, 2] < float(_hv[:, 2].max()) + 0.05) & (_rr < 0.16) & ~_sel
              & (_rr > float(np.median(np.linalg.norm(_hv[np.abs(_hv[:, 2] - _chin) < 0.01, :2] - _hx[:2], axis=1)))
                 + 0.035 if (np.abs(_hv[:, 2] - _chin) < 0.01).any() else _rr > 0.12))
    REP["head_round_verts"] = int(_round.sum())
    if JOB.get("hair_debug"):
        np.savez_compressed(JOB["hair_debug"].replace(".npz", "_round.npz"), fig=_fv, round=_round, front=_sel,
                            chin=_chin, head=_hx)
if JOB.get("skin_mask") is not None and not STATUE and HEAD is not None:
    # the head mesh's skin under the cloth, as an opacity mask in its UVs (the engine hides it: char_skinmask.py)
    t = time.time()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bl_skinmask  # noqa: E402
    os.makedirs(os.path.join(OUT, "textures"), exist_ok=True)
    bl_skinmask.build(HEAD, fig, J, JOB, REP, os.path.join(OUT, "textures", "T_head_mask.png"))
    mark("skin_mask", t)
bpy.data.objects.remove(body)
if HEAD is not None:
    bpy.data.objects.remove(HEAD)
if LINER is not None:
    bpy.context.view_layer.objects.active = LINER
    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=int(JOB.get("max_influences", 4)))
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    fig.select_set(True)
    LINER.select_set(True)
    bpy.context.view_layer.objects.active = fig
    bpy.ops.object.join()
bpy.context.view_layer.objects.active = arm
stage_save("s4_with_liner_pieces")
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
# smooth shading: the generated surface is a fine mesh with smooth normals; flat faces (the glTF's custom normals
# are lost at the first weld) made every garment look faceted in the engine (export smoothing "FACE" = per face)
fig.data.polygons.foreach_set("use_smooth", np.ones(len(fig.data.polygons), bool))
fig.data.update()
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
for name, rgb in (("Normal", (0.5, 0.5, 1.0)), ("RMA", (float(JOB.get("roughness", 0.95)), 0.0, 1.0))):   # wool / matte cloth
    im = bpy.data.images.new(f"T_outfit_{name}", 64, 64, alpha=False)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(np.tile(np.array(list(rgb) + [1.0], np.float32), 64 * 64))
    im.filepath_raw = os.path.join(OUT, "textures", f"T_outfit_{name}.png")
    im.file_format = "PNG"
    im.save()
# ------------------------------------------------------------------ the neck opening, finished
# the head and neck of the generated figure were taken out vertex by vertex, by colour: the opening's rim is a
# sawtooth, scraps of the figure's own head (skin flakes, a beard, a ponytail) float round it, and the cloth is one
# skin thick, so from behind one looks through it at the neck (seen on every figure, from every side but the front).
# Here, by shape alone: the scraps round the neck go, the rim is drawn in to an even line, and its edge is turned in
# and down so the opening has a lip
if JOB.get("neck_finish", True) and not STATUE:
    t = time.time()
    _bm = bmesh.new()
    _bm.from_mesh(fig.data)
    _bm.verts.ensure_lookup_table()
    _sk = None
    if "body_skin" in fig.data.attributes:
        _sk = np.zeros(len(fig.data.vertices), np.int32)
        fig.data.attributes["body_skin"].data.foreach_get("value", _sk)
    # (the mesh is in the export's units by now, the joints in metres)
    _u = 100.0 if max(v_.co.z for v_ in _bm.verts) > 20.0 else 1.0
    _ju = _u if float(J("neck_01").z) < 20.0 else 1.0          # (the joints as they are read here, in either)
    _nk = np.array(J("neck_01")) * _ju
    _zc = float(J("clavicle_l").z) * _ju - 0.04 * _u

    def _round_neck(v_):
        return v_.co.z > _zc and math.hypot(v_.co.x - _nk[0], v_.co.y - _nk[1]) < 0.22 * _u

    _cloth = [v_ for v_ in _bm.verts if _sk is None or _sk[v_.index] == 0]
    _is_cloth = {v_.index for v_ in _cloth}
    for v_ in _bm.verts:                     # (a vert's tag stays with it when others are deleted; its index does not)
        v_.tag = v_.index in _is_cloth
    # 1. scraps: small pieces of cloth, joined to nothing else, round the neck
    _seen, _scr = set(), []
    for v0 in _cloth:
        if v0.index in _seen:
            continue
        comp, stack = [], [v0]
        _seen.add(v0.index)
        while stack:
            v_ = stack.pop()
            comp.append(v_)
            for e_ in v_.link_edges:
                o_ = e_.other_vert(v_)
                if o_.index not in _seen and o_.index in _is_cloth:
                    _seen.add(o_.index)
                    stack.append(o_)
        if len(comp) < int(JOB.get("neck_scrap_verts", 400)) and any(_round_neck(v_) for v_ in comp):
            _scr += comp
    if _scr:
        bmesh.ops.delete(_bm, geom=_scr, context="VERTS")
        _bm.verts.ensure_lookup_table()
    # 2. the rim: the opening's boundary verts drawn toward their neighbours along the rim, then the cloth just
    # behind them relaxed
    _rim_e = [e_ for e_ in _bm.edges if len(e_.link_faces) == 1 and all(_round_neck(v_) and v_.tag for v_ in e_.verts)]
    _rim_s = set(_rim_e)
    _rim_v = list({v_ for e_ in _rim_e for v_ in e_.verts})
    _nb = {v_: [e_.other_vert(v_) for e_ in v_.link_edges if e_ in _rim_s] for v_ in _rim_v}
    for _ in range(int(JOB.get("neck_rim_iters", 30))):
        new_ = {}
        for v_, ns in _nb.items():
            if len(ns) >= 2:
                new_[v_] = v_.co * 0.5 + sum((n_.co for n_ in ns), Vector()) / len(ns) * 0.5
        for v_, c_ in new_.items():
            v_.co = c_
    _ring = {o_ for v_ in _rim_v for e_ in v_.link_edges for o_ in (e_.other_vert(v_),)} - set(_rim_v)
    for _ in range(6):
        new_ = {v_: v_.co * 0.5 + sum((e_.other_vert(v_).co for e_ in v_.link_edges), Vector()) / len(v_.link_edges) * 0.5
                for v_ in _ring if v_.link_edges}
        for v_, c_ in new_.items():
            v_.co = c_
    # 3. the lip: the rim turned in toward the neck and down
    _lip = 0
    if _rim_e:
        ex = bmesh.ops.extrude_edge_only(_bm, edges=_rim_e)
        for v_ in (g_ for g_ in ex["geom"] if isinstance(g_, bmesh.types.BMVert)):
            d_ = Vector((_nk[0] - v_.co.x, _nk[1] - v_.co.y, 0.0))
            if d_.length > 1e-4:
                v_.co += d_.normalized() * min(float(JOB.get("neck_lip_in_m", 0.012)) * _u, d_.length * 0.5)
            v_.co.z -= float(JOB.get("neck_lip_down_m", 0.02)) * _u
            _lip += 1
    bmesh.ops.recalc_face_normals(_bm, faces=_bm.faces)
    _bm.to_mesh(fig.data)
    _bm.free()
    fig.data.update()
    REP["neck_finish"] = {"scrap_verts": len(_scr), "rim_verts": len(_rim_v), "lip_verts": _lip}
    mark("neck_finish", t)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "combo.blend"), compress=True)
# this stance's outfit: the body skin that cloth covers here is cut (combo.blend keeps it all for the stances)
if "body_skin" in fig.data.attributes:
    _sv = np.zeros(len(fig.data.vertices), np.int32)
    fig.data.attributes["body_skin"].data.foreach_get("value", _sv)
    _cut = [int(i) for i in np.nonzero(_sv == 2)[0]]
    _bm = bmesh.new()
    _bm.from_mesh(fig.data)
    _bm.verts.ensure_lookup_table()
    bmesh.ops.delete(_bm, geom=[_bm.verts[i] for i in _cut], context="VERTS")
    _bm.to_mesh(fig.data)
    _bm.free()
    fig.data.update()
    REP["body_skin_cut_standing"] = len(_cut)
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

mark("export", t)
REP["total_s"] = round(time.time() - T0, 1)
json.dump(REP, open(os.path.join(OUT, "report.json"), "w"), indent=1)
print("COMBO", json.dumps(REP))
