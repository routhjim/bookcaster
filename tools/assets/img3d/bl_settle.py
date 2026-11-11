"""Blender 4.5 (background): settle a combo outfit's loose cloth (a long skirt, a coat's skirt) for one stance and
export the outfit with that stance as its bind pose. With pelvis-only weights a skirt hangs straight whatever the
legs do: seated, the thighs and the chair pass through it; walking, the forward leg does. Here the body is posed to
the stance, the loose panels are laid over the leg volumes, the seat and the floor and relaxed (a short
position-based settle in numpy: edge lengths kept from the standing outfit, gravity, collisions, the top edge
pinned to its skinned place), and the posed shape is baked. In the engine the stance's outfit follows the body by
leader pose like the standing one; it is right for that stance (small changes of arms or head are fine, the legs
must keep the stance).

    blender -b --factory-startup -t 0 --python bl_settle.py -- JOB.json

JOB {"combo": bl_combo.py's out folder (combo.blend, skirt_idx.npy, textures/),
     "joints": {bone: [x, y, z]} posed joint positions in the set frame, metres (the scene tool's result.json:
               subjects.<key>.measured.joints), "position": [x, y] the character's mark, "facing": [dx, dy],
     "mode": "drape" (seated: cloth laid over the thighs, then falling) | "push" (standing, walking: cloth pushed
             out of the legs along its own outward direction),
     "seat": {"z": m, "front": m ahead of the hips, "back": m behind, "half_width": m} | null,
     "radii": {"thigh": 0.085, "knee": 0.06, "calf": 0.055, "hips": 0.15}, "clear_m": 0.012,
     "iters": 60, "out": folder}
Writes out/outfit.fbx (+ textures copied) and out/settle.json (times, counts, worst leg/seat penetration after).
"""
import json
import math
import os
import shutil
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = JOB["out"]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
REP = {"times": {}}
bpy.ops.wm.open_mainfile(filepath=os.path.join(JOB["combo"], "combo.blend"))
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
fig = next(o for o in bpy.data.objects if o.type == "MESH" and o.parent is arm)
CM = 100.0                                   # the combo's export scale: centimetres in the file

# ------------------------------------------------------------------ joints into the character's own frame (cm)
f = np.array(JOB.get("facing", [0, -1]), float)
f /= np.linalg.norm(f)
left = np.array([-f[1], f[0]])
pos = np.array(JOB.get("position", [0, 0]), float)


def local(p):
    d = np.array(p[:2]) - pos
    return Vector((float(d @ left) * CM, float(-(d @ f)) * CM, float(p[2]) * CM))


JT = {k: local(v) for k, v in JOB["joints"].items()}
# helper joints (correctives, twists) are driven by the engine's pose logic, which Blender does not have: posed here
# they would tear the cloth they hold, and the engine would move them differently from this bind. The stance's
# outfit rides on the main joints only: each helper's weight goes to its nearest main ancestor.
import re  # noqa: E402
HELP = re.compile(JOB.get("helper_re", r"(fwd|bck|_in(_|$)|_out|Cor|twist|lwr|latissimus|pec|scap|bulge)"))


def main_of(n):
    b = arm.data.bones.get(n)
    while b is not None and HELP.search(b.name):
        b = b.parent
    return b.name if b is not None else n


t_f = time.time()
gname = {g.index: g.name for g in fig.vertex_groups}
fold = {gi: main_of(n) for gi, n in gname.items() if HELP.search(n) and n in arm.data.bones}
if fold:
    for tgt in set(fold.values()):
        if tgt not in fig.vertex_groups:
            fig.vertex_groups.new(name=tgt)
    tg_i = {n: fig.vertex_groups[n].index for n in set(fold.values())}
    add = {}
    for v in fig.data.vertices:
        acc = {}
        for g in v.groups:
            if g.group in fold and g.weight > 0:
                acc[tg_i[fold[g.group]]] = acc.get(tg_i[fold[g.group]], 0.0) + g.weight
        if acc:
            cur = {g.group: g.weight for g in v.groups}
            for gi_, w_ in acc.items():
                add.setdefault(gi_, []).append((v.index, cur.get(gi_, 0.0) + w_))
    for gi_, lst in add.items():
        grp = fig.vertex_groups[gi_]
        for vi, w_ in lst:
            grp.add([vi], w_, "REPLACE")
    for gi in sorted(fold, reverse=True):
        fig.vertex_groups.remove(fig.vertex_groups[gname[gi]])
REP["helpers_folded"] = {"groups": len(fold), "s": round(time.time() - t_f, 2)}
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
PB = arm.pose.bones


def head(n):
    return arm.matrix_world @ PB[n].head


def turn(bone, frm, to):
    pb = PB[bone]
    q = Vector(frm).normalized().rotation_difference(Vector(to).normalized())
    hd = head(bone)
    pb.matrix = (Matrix.Translation(hd) @ q.to_matrix().to_4x4() @ Matrix.Translation(-hd)) @ pb.matrix
    bpy.context.view_layer.update()


# pelvis: moved so the hip joints' midpoint lands on the posed one, turned so the hip line matches
mid_t = (JT["thigh_l"] + JT["thigh_r"]) / 2
mid_r = (head("thigh_l") + head("thigh_r")) / 2
pb = PB["pelvis"]
pb.matrix = Matrix.Translation(mid_t - mid_r) @ pb.matrix
bpy.context.view_layer.update()
turn("pelvis", head("thigh_l") - head("thigh_r"), JT["thigh_l"] - JT["thigh_r"])
# spine: the chain from the pelvis bends toward the posed neck (spread over spine_01..05)
if "neck_01" in JT:
    for b in ("spine_01", "spine_02", "spine_03", "spine_04", "spine_05"):
        if b in PB:
            v0 = head("neck_01") - head(b)
            v1 = JT["neck_01"] - head(b)
            q = v0.normalized().rotation_difference(v1.normalized())
            ang = q.angle / 3.0
            if ang > 1e-4:
                turn(b, v0, (Matrix.Rotation(ang, 4, q.axis) @ v0))
for a_, b_ in (("neck_01", "head"), ("upperarm_l", "lowerarm_l"), ("lowerarm_l", "hand_l"),
               ("upperarm_r", "lowerarm_r"), ("lowerarm_r", "hand_r"), ("thigh_l", "calf_l"), ("calf_l", "foot_l"),
               ("thigh_r", "calf_r"), ("calf_r", "foot_r")):
    if a_ in JT and b_ in JT:
        turn(a_, head(b_) - head(a_), JT[b_] - head(a_))
REP["pose_err_cm"] = {k: round((head(k) - JT[k]).length, 2) for k in JT if k in PB}
bpy.ops.object.mode_set(mode="OBJECT")

# ------------------------------------------------------------------ the posed outfit
dg = bpy.context.evaluated_depsgraph_get()
ev = fig.evaluated_get(dg)
P = np.array([v.co[:] for v in ev.data.vertices]) / CM          # metres, character frame
R = np.array([v.co[:] for v in fig.data.vertices]) / CM          # the standing (bind) shape
_E0 = np.array([e.vertices[:] for e in fig.data.edges])
_le = np.linalg.norm(P[_E0[:, 0]] - P[_E0[:, 1]], axis=1)
_far = np.unique(_E0[_le > float(JOB.get("dbg_edge_m", 0.25))].ravel())
if len(_far):
    gn_ = {g.index: g.name for g in fig.vertex_groups}
    REP["far_verts"] = {"n": int(len(_far)), "z": [round(float(P[_far, 2].min()), 2), round(float(P[_far, 2].max()), 2)],
                        "groups": sorted({gn_[g.group] for i in _far[:200] for g in fig.data.vertices[int(i)].groups if g.weight > 0.05})}
nv = len(P)
sk = np.load(os.path.join(JOB["combo"], "skirt_idx.npy"))
sk = sk[sk < nv]
free = np.zeros(nv, bool)
free[sk] = True
ED = np.array([e.vertices[:] for e in fig.data.edges])
# pinned: free vertices touching a non-free one, and the top 3 cm of the free cloth
touch = np.zeros(nv, bool)
m_ = free[ED[:, 0]] != free[ED[:, 1]]
touch[ED[m_].ravel()] = True
ztop = R[free, 2].max() if free.any() else 0
pin = free & (touch | (R[:, 2] > ztop - float(JOB.get("pin_top_m", 0.03))))
mov = free & ~pin
REP["free_verts"] = int(free.sum())
REP["pinned"] = int(pin.sum())
REP["times"]["pose"] = round(time.time() - T0, 2)

J = {k: np.array(v[:]) / CM for k, v in JT.items()}
for k in ("thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r", "pelvis", "spine_03"):
    if k not in J and k in PB:
        J[k] = np.array(head(k)[:]) / CM
RAD = {"thigh": 0.085, "knee": 0.06, "calf": 0.055, "hips": 0.15}
RAD.update(JOB.get("radii") or {})
CLR = float(JOB.get("clear_m", 0.012))
caps = []                                    # (a, b, ra, rb)
for s in ("l", "r"):
    # standing or walking the thigh's top quarter is inside the hips, which the skirt already clears: start lower
    f0 = 0.0 if JOB.get("mode") == "drape" else float(JOB.get("thigh_from", 0.3))
    a0 = J[f"thigh_{s}"] + f0 * (J[f"calf_{s}"] - J[f"thigh_{s}"])
    caps.append((a0, J[f"calf_{s}"], RAD["thigh"] + f0 * (RAD["knee"] - RAD["thigh"]), RAD["knee"]))
    caps.append((J[f"calf_{s}"], J[f"foot_{s}"], RAD["knee"], RAD["calf"]))
if JOB.get("mode") == "drape":                # seated: the hips and seat of the body under the skirt's back
    caps.append((J["thigh_l"] + np.array([0, 0.02, 0.03]), J["thigh_r"] + np.array([0, 0.02, 0.03]), RAD["hips"],
                 RAD["hips"]))
if JOB.get("mode") == "drape":
    # the lap: cloth spans from thigh to thigh as a sheet, not down between them; horizontal rungs between
    # matching points of the two thighs (the drape rule lays cloth over their tops)
    for fr in (0.2, 0.4, 0.6, 0.8, 1.0):
        a_ = J["thigh_l"] + fr * (J["calf_l"] - J["thigh_l"])
        b_ = J["thigh_r"] + fr * (J["calf_r"] - J["thigh_r"])
        rr = RAD["thigh"] + (RAD["knee"] - RAD["thigh"]) * fr
        caps.append((a_, b_, rr, rr))
    # and from knee to knee down the shins: the skirt falls from the knees as one sheet in front of the shins
    for fr in (0.25, 0.5, 0.75, 0.95):
        a_ = J["calf_l"] + fr * (J["foot_l"] - J["calf_l"])
        b_ = J["calf_r"] + fr * (J["foot_r"] - J["calf_r"])
        caps.append((a_, b_, RAD["calf"], RAD["calf"]))
SEAT = JOB.get("seat")
hip_y = (J["thigh_l"][1] + J["thigh_r"][1]) / 2
hip_x = (J["thigh_l"][0] + J["thigh_r"][0]) / 2


def cap_dist(X, a, b, ra, rb):
    ab = b - a
    t = np.clip(((X - a) @ ab) / (ab @ ab), 0, 1)
    c = a + t[:, None] * ab
    d = X - c
    n = np.linalg.norm(d, axis=1)
    r = ra + (rb - ra) * t
    return n - r, d / np.maximum(n, 1e-9)[:, None], c, r, t


def collide(X, outward=None):
    """push X out of every capsule (+CLR), the seat box and the floor; outward: per-vertex exit direction
    (horizontal) for 'push' mode, else radial, or straight up for capsules lying near horizontal"""
    for a, b, ra, rb in caps:
        sd, nr, c, r, t = cap_dist(X, a, b, ra, rb)
        hit = sd < CLR
        if not hit.any():
            continue
        ab = (b - a) / np.linalg.norm(b - a)
        if abs(ab[2]) < 0.6 and JOB.get("mode") == "drape":
            # a thigh lying along the seat: cloth over it goes to its top
            dz = np.sqrt(np.maximum((r + CLR) ** 2 - np.sum((X - c)[:, :2] ** 2, 1), 0))
            X[hit, 2] = np.maximum(X[hit, 2], (c[hit, 2] + dz[hit]))
        elif outward is not None:
            # out along the cloth's own outward direction until clear (solve |X + s u - axis| = r + CLR)
            u = outward[hit]
            q = X[hit] - c[hit]
            q -= (q @ ab)[:, None] * ab
            u2 = u - (u @ ab)[:, None] * ab
            A = np.sum(u2 * u2, 1)
            B = 2 * np.sum(q * u2, 1)
            Cc = np.sum(q * q, 1) - (r[hit] + CLR) ** 2
            disc = np.maximum(B * B - 4 * A * Cc, 0)
            s_ = np.clip((-B + np.sqrt(disc)) / np.maximum(2 * A, 1e-9), 0, 0.3)
            par = A < 0.1                    # the direction runs along the capsule: out radially instead
            mv = s_[:, None] * u
            mv[par] = (nr[hit] * (CLR - sd[hit])[:, None])[par]
            X[hit] += mv
        else:
            X[hit] += nr[hit] * (CLR - sd[hit])[:, None]
    if SEAT:
        z0 = float(SEAT["z"])
        inb = ((np.abs(X[:, 0] - hip_x) < SEAT.get("half_width", 0.32)) & (X[:, 1] > hip_y - SEAT.get("front", 0.42))
               & (X[:, 1] < hip_y + SEAT.get("back", 0.25)) & (X[:, 2] < z0 + CLR) & (X[:, 2] > z0 - 0.25))
        X[inb, 2] = z0 + CLR
    X[:, 2] = np.maximum(X[:, 2], 0.004)
    return X


t = time.time()
X = P.copy()
mode = JOB.get("mode", "push")
if mode == "drape" and mov.any():
    # lay each free vertex along its column's path: down from its pinned top, over what lies under it, then down
    # to the floor; the path length to it is its distance below the top in the standing outfit
    # columns by angle round the waist: each bin's path is marched once from its pinned top (down while nothing is
    # under it, outward over what is: thighs, hips, seat; on the floor or on the seat behind the hips it stays), and
    # every free vertex is placed on its bin's path (interpolated between neighbouring bins) at its distance below
    # the top in the standing outfit: smooth across columns by construction
    ids = np.nonzero(mov)[0]
    top_ids = np.nonzero(pin)[0]
    cx_, cy_ = R[top_ids, 0].mean(), R[top_ids, 1].mean()
    th_r = np.arctan2(R[:, 0] - cx_, -(R[:, 1] - cy_))
    NB = int(JOB.get("lay_bins", 144))
    fb = (th_r + np.pi) / (2 * np.pi) * NB
    bt = np.floor(fb[top_ids]).astype(int) % NB
    topP = np.zeros((NB, 3))
    topR = np.zeros(NB)
    have = np.zeros(NB, bool)
    for bb in range(NB):
        m_ = bt == bb
        if m_.any():
            topP[bb] = P[top_ids[m_]].mean(0)
            topR[bb] = R[top_ids[m_], 2].mean()
            have[bb] = True
    hv_ = np.nonzero(have)[0]
    for bb in np.nonzero(~have)[0]:                 # empty bins (an open coat front): the nearest filled bin
        k_ = hv_[np.argmin(np.abs(((hv_ - bb + NB // 2) % NB) - NB // 2))]
        topP[bb], topR[bb] = topP[k_], topR[k_]
    c0 = J.get("pelvis", np.array([hip_x, hip_y, 0.9]))
    ang = (np.arange(NB) + 0.5) / NB * 2 * np.pi - np.pi
    d = np.c_[np.sin(ang), -np.cos(ang), np.zeros(NB)]     # th = atan2(x, -y): x = sin, y = -cos
    step = float(JOB.get("lay_step_m", 0.01))
    smax = float(max(topR.max() - R[ids, 2].min(), 0.05))
    nst = int(smax / step) + 2
    path = np.zeros((NB, nst, 3))
    p = topP.copy()
    path[:, 0] = p
    for k_ in range(1, nst):
        down = p.copy()
        down[:, 2] -= step
        q = collide(down.copy(), d)
        sup = (q[:, 2] > down[:, 2] + 1e-6) | (q[:, 2] <= 0.0041)
        nxt = collide(p + d * step, d)
        stay = q[:, 2] <= 0.0041
        if SEAT:
            on_seat = sup & (np.abs(q[:, 2] - (float(SEAT["z"]) + CLR)) < 0.003)
            stay |= on_seat & (d[:, 1] > float(JOB.get("sat_on_dir", 0.3)))
        p = np.where(sup[:, None], np.where(stay[:, None], q, nxt), q)
        path[:, k_] = p
    sv = np.maximum(topR[np.floor(fb[ids]).astype(int) % NB] - R[ids, 2], 0) / step
    b0 = np.floor(fb[ids] - 0.5).astype(int) % NB
    b1 = (b0 + 1) % NB
    wb = (fb[ids] - 0.5) - np.floor(fb[ids] - 0.5)
    k0 = np.clip(np.floor(sv).astype(int), 0, nst - 2)
    ws = np.clip(sv - k0, 0, 1)

    def at(bi):
        return path[bi, k0] * (1 - ws)[:, None] + path[bi, k0 + 1] * ws[:, None]
    X[ids] = at(b0) * (1 - wb)[:, None] + at(b1) * wb[:, None]
    REP["laid"] = int(len(ids))
    # the columns were laid one by one: smooth their displacement over the cloth so neighbours agree (no comb
    # of strips at the hem), then clear the colliders again
    Dp = X - P
    Ea = ED[mov[ED[:, 0]] | mov[ED[:, 1]]]
    dga = np.bincount(Ea.ravel(), minlength=nv).astype(float)
    for _ in range(int(JOB.get("lay_smooth", 10))):
        acc = np.zeros_like(Dp)
        np.add.at(acc, Ea[:, 0], Dp[Ea[:, 1]])
        np.add.at(acc, Ea[:, 1], Dp[Ea[:, 0]])
        Dp[mov] = 0.5 * Dp[mov] + 0.5 * acc[mov] / np.maximum(dga[mov], 1)[:, None]
    Y = P[mov] + Dp[mov]
    rd_ = np.c_[Y[:, 0] - c0[0], Y[:, 1] - c0[1], np.zeros(len(Y))]
    X[mov] = collide(Y, rd_ / np.maximum(np.linalg.norm(rd_, axis=1), 1e-6)[:, None])
    REP["laid"] = int(len(ids))
REP["times"]["lay"] = round(time.time() - t, 2)

# relax: edge lengths of the standing outfit (Jacobi averaging), gravity, collisions; pins hold their skinned place
t = time.time()
Ef = ED[free[ED[:, 0]] | free[ED[:, 1]]]
L0 = np.linalg.norm(R[Ef[:, 0]] - R[Ef[:, 1]], axis=1)
outw = None
if mode == "drape":                          # out of a slanted shin: forward, away from the body's axis
    c0_ = J.get("pelvis", np.array([hip_x, hip_y, 0.9]))
    outw = np.c_[X[:, 0] - c0_[0], X[:, 1] - c0_[1], np.zeros(nv)]
    outw /= np.maximum(np.linalg.norm(outw, axis=1), 1e-6)[:, None]
if mode == "push":
    nrm = np.array([v.normal[:] for v in fig.data.vertices])
    outw = nrm.copy()
    outw[:, 2] = 0
    outw /= np.maximum(np.linalg.norm(outw, axis=1), 1e-6)[:, None]
deg = np.bincount(Ef.ravel(), minlength=nv).astype(float)
for it in range(int(JOB.get("iters", 40))):
    X[mov, 2] -= float(JOB.get("gravity_step_m", 0.0 if mode == "push" else 0.0005))
    for _ in range(int(JOB.get("solve", 8))):
        d = X[Ef[:, 1]] - X[Ef[:, 0]]
        ln = np.linalg.norm(d, axis=1)
        corr = ((ln - L0) / np.maximum(ln, 1e-9))[:, None] * d * 0.5
        acc = np.zeros_like(X)
        np.add.at(acc, Ef[:, 0], corr)
        np.add.at(acc, Ef[:, 1], -corr)
        X[mov] += (acc[mov] / np.maximum(deg[mov], 1)[:, None]) * 1.6
    X[pin] = P[pin]
    X[~free] = P[~free]
    X[mov] = collide(X[mov], None if outw is None else outw[mov])
REP["times"]["relax"] = round(time.time() - t, 2)
# check: worst penetration of the free cloth into legs and seat
pen = 0.0
for a, b, ra, rb in caps[:4]:
    sd = cap_dist(X[mov], a, b, ra, rb)[0]
    pen = max(pen, float(-sd.min()) if len(sd) else 0.0)
REP["worst_leg_penetration_m"] = round(pen, 4)
ln = np.linalg.norm(X[Ef[:, 1]] - X[Ef[:, 0]], axis=1)
REP["edge_stretch_p95"] = round(float(np.quantile(ln / np.maximum(L0, 1e-6), 0.95)), 3)
REP["hem_min_z_m"] = round(float(X[mov, 2].min()), 3) if mov.any() else None

# ------------------------------------------------------------------ bake: this pose is the bind pose
# (by hand: Blender's apply-pose left some helper bones of this skeleton off their posed place)
bpy.context.view_layer.objects.active = arm
posed = {pb.name: pb.matrix.copy() for pb in arm.pose.bones}
bpy.ops.object.mode_set(mode="EDIT")
for eb in arm.data.edit_bones:
    eb.use_connect = False
for eb in arm.data.edit_bones:
    eb.matrix = posed[eb.name]
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.location = (0, 0, 0)
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.rotation_euler = (0, 0, 0)
    pb.scale = (1, 1, 1)
bpy.ops.object.mode_set(mode="OBJECT")
fig.data.vertices.foreach_set("co", (X * CM).ravel())
fig.data.update()
# the bound outfit in its new bind pose must read back as the settled shape
_ev = fig.evaluated_get(bpy.context.evaluated_depsgraph_get())
_E = np.array([v.co[:] for v in _ev.data.vertices]) / CM
_dd = np.linalg.norm(_E - X, axis=1)
REP["rebind_err_m"] = {"max": round(float(_dd.max()), 4), "n_over_1cm": int((_dd > 0.01).sum())}
if (_dd > 0.01).any():
    gn_ = {g.index: g.name for g in fig.vertex_groups}
    bad_ = np.nonzero(_dd > 0.01)[0][:200]
    REP["rebind_err_m"]["groups"] = sorted({gn_[g.group] for i in bad_ for g in fig.data.vertices[int(i)].groups
                                            if g.weight > 0.05})[:30]
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
td = os.path.join(OUT, "textures")
if os.path.isdir(td):
    shutil.rmtree(td)
shutil.copytree(os.path.join(JOB["combo"], "textures"), td)
if JOB.get("save_blend", True):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "settled.blend"), compress=True)
REP["total_s"] = round(time.time() - T0, 1)
json.dump(REP, open(os.path.join(OUT, "settle.json"), "w"), indent=1)
print("SETTLE", json.dumps(REP))
