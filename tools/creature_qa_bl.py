"""Blender side: QA renders of a rigged CREATURE (tools/blockout_creature_bl.py output), any template creature.

    blender -b --factory-startup -t 8 --python tools/creature_qa_bl.py -- RIGGED.blend RIG.json OUT_PREFIX \
        [--carry PERSON_rigged.blend] [--person-height 1.65]

Poses (template bone names; rotations are about WORLD axes through each bone's head, so they mean the same on
any mesh rigged to the template): rest, jaw open, neck up, wings spread, fore-claw strike, and CARRY (wings
spread, fore limbs forward and down so the claws meet in front of the belly; with --carry, that person's model
is hung from the claw anchors: pelvis at the claws' midpoint). Two views each (side, front), painted
(vertex colours, Workbench). Prints QA_ANCHOR lines: the claw anchors' world positions in the carry pose and
the gap between the claws.
"""

import json
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
blend, rigj, out = argv[0], argv[1], argv[2]
carry = argv[argv.index("--carry") + 1] if "--carry" in argv else None
ph = float(argv[argv.index("--person-height") + 1]) if "--person-height" in argv else 1.65

CYCLES = "--clayqa" in argv
if "--checks-only" in argv:                         # the per-pose checks without the renders (seconds)
    class _NoRender:
        @staticmethod
        def render(**kw):
            return {"FINISHED"}
    bpy.ops.render = _NoRender
bpy.ops.wm.open_mainfile(filepath=blend)
arm = bpy.data.objects["rig"]
model = bpy.data.objects["model"]
RJ = json.load(open(rigj))
H = float(RJ.get("height", 6.0))
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sc.render.resolution_x, sc.render.resolution_y = 640, 640
sh = sc.display.shading
sh.light = "STUDIO"
sh.color_type = "TEXTURE" if model.data.uv_layers else ("VERTEX" if model.data.color_attributes else "SINGLE")
sh.show_shadows = False
sc.world = bpy.data.worlds.new("w")
sc.world.color = (0.12, 0.12, 0.13)
cd = bpy.data.cameras.new("c")
cd.type = "ORTHO"
cd.ortho_scale = H * 1.45
cam = bpy.data.objects.new("c", cd)
sc.collection.objects.link(cam)
sc.camera = cam
if CYCLES:
    # SHAPE readability: grey clay (material override), 3-point lights riding on the camera, mid-grey background
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "Standard"
    clay = bpy.data.materials.new("clay")
    clay.use_nodes = True
    clay.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1)
    clay.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.65
    bpy.context.view_layer.material_override = clay
    w_ = bpy.data.worlds.new("grey")
    w_.use_nodes = True
    w_.node_tree.nodes["Background"].inputs[0].default_value = (0.32, 0.32, 0.33, 1)
    w_.node_tree.nodes["Background"].inputs[1].default_value = 0.25
    sc.world = w_
    for nm_, loc_, en_ in (("key", (-7, 7, 9), 6000), ("fill", (8, 3, 3), 1500), ("rim", (3, -9, 7), 3500)):
        ld_ = bpy.data.lights.new(nm_, "AREA")
        ld_.energy = en_ * (H / 6.0) ** 2
        ld_.size = 4.0 * H / 6.0
        lo_ = bpy.data.objects.new(nm_, ld_)
        sc.collection.objects.link(lo_)
        lo_.parent = cam
        lo_.location = tuple(x * H / 6.0 for x in loc_)
        tq = bpy.data.objects.new(nm_ + "_t", None)
        sc.collection.objects.link(tq)
        tq.parent = cam
        tq.location = (0, 0, -30)
        c_ = lo_.constraints.new("TRACK_TO")
        c_.target = tq
        c_.track_axis = "TRACK_NEGATIVE_Z"
        c_.up_axis = "UP_Y"
pb = arm.pose.bones


def reset():
    for b in pb:
        b.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def wrot(name, axis, deg):
    """Rotate a pose bone about a WORLD axis through its head (keeps children attached)."""
    b = pb.get(name)
    if b is None:
        return
    bpy.context.view_layer.update()
    M = arm.matrix_world @ b.matrix
    head = M.translation.copy()
    R = Matrix.Rotation(math.radians(deg), 4, Vector(axis))
    Mn = Matrix.Translation(head) @ R @ Matrix.Translation(-head) @ M
    b.matrix = arm.matrix_world.inverted() @ Mn
    bpy.context.view_layer.update()


def aim(name, d):
    """Turn a pose bone (head fixed) by the smallest rotation that points it along WORLD direction d."""
    b = pb.get(name)
    if b is None:
        return
    bpy.context.view_layer.update()
    M = arm.matrix_world @ b.matrix
    head = M.translation.copy()
    tail = arm.matrix_world @ b.tail
    cur = (tail - head).normalized()
    tgt = Vector(d).normalized()
    q = cur.rotation_difference(tgt)
    Mn = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ M
    b.matrix = arm.matrix_world.inverted() @ Mn
    bpy.context.view_layer.update()


def wing(side, upper, fore, tip, f1, f2, f3):
    """Aims for one wing (template bone = its tail joint): upper arm, forearm, leading finger, fingers 1-3.
    Directions are given for the LEFT wing (+Y); the right mirrors y."""
    sg = 1 if side == "l" else -1
    m = lambda v: (v[0], v[1] * sg, v[2])
    return [("aim", f"wing_{side}_elbow", m(upper)), ("aim", f"wing_{side}_wrist", m(fore)),
            ("aim", f"wing_{side}_tip", m(tip)), ("aim", f"wing_{side}_finger1", m(f1)),
            ("aim", f"wing_{side}_finger2", m(f2)), ("aim", f"wing_{side}_finger3", m(f3))]


def both(*a):
    return wing("l", *a) + wing("r", *a)


X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)        # the model faces +X, left = +Y, up = +Z
side_l = [n for n in ("l",) ]


def chain(prefix):
    return [b.name for b in arm.data.bones if b.name.startswith(prefix)]


POSES = {
    "rest": [],
    "jaw_open": [("jaw", Y, 38)],                 # +Y axis, positive: the snout end drops (opens)
    "wings_spread": both((-0.1, 1, 0.45), (0.05, 1, 0.25), (0.05, 0.7, 0.7), (-0.15, 1, 0.05),
                         (-0.35, 0.9, -0.35), (-0.45, 0.55, -0.7)),
    # folded against the back: the elbow up behind the shoulder, the forearm and the fingers hanging down
    "wings_folded": both((-0.45, 0.3, 0.85), (-0.3, 0.12, -0.95), (-0.2, 0.15, -0.97), (-0.28, 0.12, -0.95),
                         (-0.34, 0.1, -0.93), (-0.4, 0.08, -0.91)),
    "flap_up": both((-0.15, 0.45, 0.88), (-0.1, 0.25, 0.96), (-0.1, 0.2, 0.97), (-0.25, 0.3, 0.92),
                    (-0.4, 0.35, 0.85), (-0.55, 0.35, 0.75)),
    # downstroke: the wings sweep OUT and DOWN (and a little back), never in over the head
    "flap_down": both((-0.15, 1, -0.45), (-0.1, 1, -0.35), (-0.05, 0.8, -0.6), (-0.25, 0.75, -0.6),
                      (-0.4, 0.55, -0.73), (-0.45, 0.35, -0.82)),
    # template fore limb: *_hip bone = clavicle (chest->shoulder), *_knee = upper arm, *_ankle = forearm
    "strike": [("fleg_r_knee", Y, -95), ("fleg_r_ankle", Y, 25), ("jaw", Y, 25)],
    "carry": [("wing_l_shoulder", X, -30), ("wing_r_shoulder", X, 30), ("fleg_l_knee", Y, -50),
              ("fleg_r_knee", Y, -50), ("fleg_l_knee", X, -15), ("fleg_r_knee", X, 15), ("fleg_l_ankle", Y, 20),
              ("fleg_r_ankle", Y, 20)],
}


for _k, _ops in (RJ.get("qa_poses") or {}).items():   # a template's own QA poses (its bone names)
    POSES[_k] = [(o[0], tuple(o[1]), float(o[2])) for o in _ops]


def anchor(name):
    a = RJ["anchors"].get(name)
    if not a:
        return None
    b = pb.get(a["joint"])
    if b is None:
        return None
    M = arm.matrix_world @ b.matrix
    tail = arm.matrix_world @ b.tail
    blen = float(np.linalg.norm(np.asarray(RJ["joints"]["chest"]) - np.asarray(RJ["joints"]["pelvis"])))
    off = Vector(a["offset"]) * blen
    return np.array(tail if a["joint"].endswith("_toe") else M.translation) + np.array(off)


person = None
if carry:
    with bpy.data.libraries.load(carry) as (src, dst):
        dst.objects = [n for n in src.objects if n in ("model", "rig")]
    objs = [o for o in dst.objects if o is not None]
    for o in objs:
        o.name = "person_" + o.name
        sc.collection.objects.link(o)
    person = next((o for o in objs if o.type == "MESH"), None)
    prig = next((o for o in objs if o.type == "ARMATURE"), None)
    root = prig or person
    for o in objs:
        if o is not root and o.parent is None:
            o.parent = root

views = {"side": ((0, -40, H * 0.5), (math.pi / 2, 0, 0)), "front": ((40, 0, H * 0.5), (math.pi / 2, 0, math.pi / 2)),
         "3q": ((28.3, -28.3, H * 0.75), (math.radians(89), 0, math.radians(45)))}
# a WINGS mask for the head close-ups (a wing in front of the head is not the head's QA) + rest edge lengths for
# the per-pose TEAR check (generic: an edge stretched > 2.5x its rest length outside the wings = torn skin)
_gi = {g.index: g.name for g in model.vertex_groups}
_wing_v = [v.index for v in model.data.vertices
           if v.groups and _gi[max(v.groups, key=lambda g: g.weight).group].startswith("wing_")]
_wg = model.vertex_groups.get("_qa_wings") or model.vertex_groups.new(name="_qa_wings")
_wg.add(_wing_v, 1.0, "REPLACE")
_mask = model.modifiers.new("qa_wing_mask", "MASK")
_mask.vertex_group = "_qa_wings"
_mask.invert_vertex_group = True
_mask.show_render = _mask.show_viewport = False
_E = np.array([e.vertices[:] for e in model.data.edges])
_ws = set(_wing_v)
_E = _E[[not (a in _ws or b in _ws) for a, b in _E]]
_R = np.array([v.co[:] for v in model.data.vertices])
_L0 = np.linalg.norm(_R[_E[:, 0]] - _R[_E[:, 1]], axis=1)
CHECKS = {}


def tear_check(name):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = model.evaluated_get(dg)
    me_ = ev.to_mesh()
    if len(me_.vertices) != len(_R):
        ev.to_mesh_clear()
        return
    P = np.array([v.co[:] for v in me_.vertices])
    ev.to_mesh_clear()
    r = np.linalg.norm(P[_E[:, 0]] - P[_E[:, 1]], axis=1) / np.maximum(_L0, 1e-6)
    grow = np.linalg.norm(P[_E[:, 0]] - P[_E[:, 1]], axis=1) - _L0
    bad = np.where((r > 2.5) & (grow > 0.01 * H))[0]           # stretched 2.5x AND by > 1 % of the height
    groups = {}
    for k in bad[:2000]:
        v = model.data.vertices[int(_E[k, 0])]
        g = _gi[max(v.groups, key=lambda g: g.weight).group] if v.groups else "-"
        groups[g] = groups.get(g, 0) + 1
    where_ = []
    if len(bad):
        Mb = (_R[_E[bad, 0]] + _R[_E[bad, 1]]) / 2
        for q in np.percentile(Mb, [10, 50, 90], axis=0):
            where_.append(np.round(q, 2).tolist())
    CHECKS[name] = {"torn_edges": int(len(bad)), "max_stretch": round(float(r[bad].max()) if len(bad) else 1.0, 2),
                    "rest_at_pct10_50_90": where_,
                    "by_bone": dict(sorted(groups.items(), key=lambda kv: -kv[1])[:8]), "ok": int(len(bad)) == 0}
    print("QA_TEAR", name, json.dumps(CHECKS[name]))


for name, ops in POSES.items():
    reset()
    for o in ops:
        if o[0] == "aim":
            aim(o[1], o[2])
        else:
            wrot(*o)
    if name == "carry":
        cl, cr = anchor("claw_l"), anchor("claw_r")
        if cl is not None and cr is not None:
            mid = (cl + cr) / 2
            print("QA_ANCHOR", json.dumps({"claw_l": cl.round(3).tolist(), "claw_r": cr.round(3).tolist(),
                                           "gap_m": round(float(np.linalg.norm(cl - cr)), 3)}))
            if person is not None:
                root.location = Vector((mid[0] + 0.05, mid[1], mid[2] - ph * 0.55))
                root.rotation_euler = (0, 0, math.pi / 2)
                person.hide_render = False
                if prig is not None:                  # held at the waist: arms up over the claws, legs dangling
                    bpy.context.view_layer.update()
                    for bn, ax_, deg in (("r_uarm", (1, 0, 0), -120), ("l_uarm", (1, 0, 0), 120),
                                         ("r_thigh", (0, 1, 0), -10), ("l_thigh", (0, 1, 0), 10)):
                        pbn = prig.pose.bones.get(bn)
                        if pbn is None:
                            continue
                        Mb = prig.matrix_world @ pbn.matrix
                        hd = Mb.translation.copy()
                        Rr = Matrix.Rotation(math.radians(deg), 4, Vector(ax_))
                        pbn.matrix = prig.matrix_world.inverted() @ (Matrix.Translation(hd) @ Rr @ Matrix.Translation(-hd) @ Mb)
                        bpy.context.view_layer.update()
                bpy.context.view_layer.update()
                vs = [person.matrix_world @ person.data.vertices[i_].co for i_ in range(0, len(person.data.vertices), 500)]
                print("QA_PERSON", person.name, root.name, person.parent.name if person.parent else None,
                      [round(min(v[i] for v in vs), 2) for i in range(3)], [round(max(v[i] for v in vs), 2) for i in range(3)],
                      "hidden", person.hide_get(), person.hide_render)
    elif person is not None:
        person.hide_render = True
    bpy.context.view_layer.update()
    tear_check(name)
    for vn, (loc, rot) in views.items():
        cam.location = loc
        cam.rotation_euler = rot
        if vn == "3q":                               # aim at the body centre (the whole body in frame)
            c0 = Vector((float(RJ["joints"]["pelvis"][0]), 0.0, H * 0.5))
            dv = Vector((0.7, -0.7, 0.25)).normalized()
            cam.location = c0 + dv * 40
            cam.rotation_euler = (-dv).to_track_quat("-Z", "Y").to_euler()
        cd.ortho_scale = H * 1.45
        sc.render.filepath = f"{out}_{name}_{vn}.png"
        bpy.ops.render.render(write_still=True)
    if name in ("strike", "carry"):                   # larger: the upper body from 3 angles
        ctr = np.asarray(RJ["joints"]["chest"], float)
        if name == "carry" and person is not None:    # frame the claws and the person held in them too
            cl_, cr_ = anchor("claw_l"), anchor("claw_r")
            if cl_ is not None:
                ctr = (ctr + (cl_ + cr_) / 2) / 2 - np.array([0, 0, 0.4])
        for vn, d in (("bigside", (0, -1, 0.05)), ("bigfront", (1, 0, 0.05)), ("big3q", (0.7, -0.7, 0.2))):
            dv = Vector(d).normalized()
            cd.ortho_scale = H * 0.75
            cam.location = Vector(ctr) + dv * 30
            cam.rotation_euler = (-dv).to_track_quat("-Z", "Y").to_euler()
            sc.render.filepath = f"{out}_{name}_{vn}.png"
            bpy.ops.render.render(write_still=True)
        cd.ortho_scale = H * 1.45
    if name in ("rest", "jaw_open") or CYCLES:      # head at HEAD scale (every pose in the clay QA)
        # frame the HEAD's own vertices (owned by the head / skull / jaw bones), in this pose
        dg = bpy.context.evaluated_depsgraph_get()
        ev = model.evaluated_get(dg)
        mev = ev.to_mesh()
        gi_ = {g.index: g.name for g in model.vertex_groups}
        hv = [ev.matrix_world @ mev.vertices[i].co for i, v in enumerate(model.data.vertices)
              if v.groups and gi_[max(v.groups, key=lambda g: g.weight).group] in ("head", "head_end", "jaw")]
        if hv:
            P_ = np.array([tuple(q) for q in hv])
            lo_, hi_ = np.percentile(P_, 2, axis=0), np.percentile(P_, 98, axis=0)
            hc = (lo_ + hi_) / 2
            hs = float(max(hi_ - lo_)) * 1.35
        else:
            hc, hs = np.asarray(RJ["joints"]["head"], float), H * 0.2
        ev.to_mesh_clear()
        for vn, d in (("head", (0, -1, 0.0)), ("head3q", (0.75, -0.6, 0.15))):
            dv = Vector(d).normalized()
            cd.ortho_scale = hs
            cam.location = Vector(hc) + dv * 30
            cam.rotation_euler = (-dv).to_track_quat("-Z", "Y").to_euler()
            # the wings are MASKED out (a clip plane cut horn tips open and showed the wing end caps)
            _mask.show_render = True
            if person is not None:
                _ph = person.hide_render
                person.hide_render = True
            cd.clip_start = 0.1
            sc.render.filepath = f"{out}_{name}_{vn}.png"
            bpy.ops.render.render(write_still=True)
            _mask.show_render = False
            if person is not None:
                person.hide_render = _ph
        cd.clip_start = 0.1
        cd.ortho_scale = H * 1.45
json.dump({"poses": CHECKS, "ok": all(c["ok"] for c in CHECKS.values())}, open(f"{out}_checks.json", "w"), indent=1)
print("QA_DONE", list(POSES))
