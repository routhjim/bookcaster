"""Blender side of the creature rig: an armature from rig_fit.py's bones, weights onto the asset's own mesh, test
poses as one action (frame 1 rest, then one pose per frame), Cycles stills on a GPU and an FBX for UE.

    blender -b --factory-startup -t 4 --python bl_rig.py -- JOB.json
    JOB {"blend": the asset's source .blend, "id": asset id, "rig": rig_fit.py OUT.json, "out": folder,
         "hip_bus": "05", "poses": [{"name": "...", "rot": {"bone": [axis "x|y|z" (world), degrees] or a list of those}}, ...],
         "stills": true, "px": [768, 512]}

Bone rotations are given about world axes (asset frame: X to the creature's left, -Y front, Z up) and applied in
each bone's rest frame, so a chain bends the same way whatever the bone rolls are. Writes out/<id>_rig.fbx
(armature + mesh + action), out/<id>_rig.blend, out/pose_<name>.png, out/rig.json (times).
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.kdtree import KDTree

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
AID = JOB["id"]
OUT = JOB["out"]
os.makedirs(OUT, exist_ok=True)
T = {}
t0 = time.time()
bpy.ops.wm.open_mainfile(filepath=JOB["blend"])
sc = bpy.context.scene
ob = bpy.data.objects[f"SM_{AID}"]
for o in list(bpy.data.objects):
    if o is not ob:
        bpy.data.objects.remove(o, do_unlink=True)
R = json.load(open(JOB["rig"]))
arm = bpy.data.armatures.new(f"SK_{AID}")
rig = bpy.data.objects.new(f"SK_{AID}", arm)
sc.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for b in R["bones"]:
    e = arm.edit_bones.new(b["name"])
    e.head, e.tail = Vector(b["head"]), Vector(b["tail"])
    if (e.tail - e.head).length < 1e-3:
        e.tail = e.head + Vector((0, 0, 0.05))
    e.roll = 0
    if b["parent"]:
        e.parent = eb[b["parent"]]
    eb[b["name"]] = e
bpy.ops.object.mode_set(mode="OBJECT")
T["armature"] = round(time.time() - t0, 2)
# weights: nearest fitted vertex
t = time.time()
xyz = R["verts_xyz"]
kd = KDTree(len(xyz))
for i, p in enumerate(xyz):
    kd.insert(p, i)
kd.balance()
names = [b["name"] for b in R["bones"]]
groups = [ob.vertex_groups.new(name=nm) for nm in names]
wi, ww = R["weights_idx"], R["weights_w"]
per = {}
mw = ob.matrix_world
for v in ob.data.vertices:
    _, j, _ = kd.find(mw @ v.co)
    for bi, w in zip(wi[j], ww[j]):
        if w > 1e-3:
            per.setdefault(bi, ([], []))
            per[bi][0].append(v.index)
            per[bi][1].append(w)
for bi, (vs, ws) in per.items():
    g = groups[bi]
    for v_, w_ in zip(vs, ws):
        g.add([v_], w_, "REPLACE")
ob.parent = rig
md = ob.modifiers.new("Armature", "ARMATURE")
md.object = rig
T["weights"] = round(time.time() - t, 2)
T["verts"] = len(ob.data.vertices)
# poses
t = time.time()
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
act = bpy.data.actions.new(f"A_{AID}_poses")
rig.animation_data_create()
rig.animation_data.action = act
poses = [{"name": "rest", "rot": {}}] + JOB.get("poses", [])
AX = {"x": Vector((1, 0, 0)), "y": Vector((0, 1, 0)), "z": Vector((0, 0, 1))}
for f, pz in enumerate(poses, start=1):
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion()
        pb.location = Vector()
    for bn, spec in pz["rot"].items():
        if bn not in rig.pose.bones:
            continue
        pb = rig.pose.bones[bn]
        rest = pb.bone.matrix_local.to_3x3()
        q = Quaternion()
        for ax, deg in ([spec] if isinstance(spec[0], str) else spec):
            q = Quaternion((rest.inverted() @ AX[ax]).normalized(), math.radians(deg)) @ q
        pb.rotation_quaternion = q
    for pb in rig.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=f)
        pb.keyframe_insert("location", frame=f)
for fc in act.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "CONSTANT"
sc.frame_start, sc.frame_end = 1, len(poses)
bpy.ops.object.mode_set(mode="OBJECT")
T["poses"] = round(time.time() - t, 2)


def stretch(frame):
    """edge length change against rest (deformed mesh), for the report"""
    sc.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    me = ob.evaluated_get(dg).to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ed = np.empty(len(me.edges) * 2, int)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    ob.evaluated_get(dg).to_mesh_clear()
    return np.linalg.norm(co[ed[:, 0]] - co[ed[:, 1]], axis=1)


t = time.time()
base = stretch(1)
rep_p = {}
for f, pz in enumerate(poses, start=1):
    if f == 1:
        continue
    r = stretch(f) / np.maximum(base, 1e-6)
    rep_p[pz["name"]] = {"edges_stretched_over_50pct": int((r > 1.5).sum()), "edges_squashed_under_50pct":
                         int((r < 0.5).sum()), "max_ratio": round(float(r.max()), 2)}
sc.frame_set(1)
T["stretch_check"] = round(time.time() - t, 2)
# FBX for UE (cm, armature + mesh + the action)
t = time.time()
bpy.ops.object.select_all(action="DESELECT")
ob.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
fbx = os.path.join(OUT, f"{AID}_rig.fbx")
bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, apply_unit_scale=True, apply_scale_options="FBX_SCALE_UNITS",
                         object_types={"ARMATURE", "MESH"}, add_leaf_bones=False, bake_anim=True,
                         bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0,
                         bake_anim_step=1.0, path_mode="COPY", embed_textures=True, axis_forward="-Y", axis_up="Z",
                         mesh_smooth_type="FACE", armature_nodetype="NULL", primary_bone_axis="Y",
                         secondary_bone_axis="X")
T["fbx"] = round(time.time() - t, 2)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f"{AID}_rig.blend"), compress=True)
# stills (Cycles, GPU)
if JOB.get("stills", True):
    t = time.time()
    p = bpy.context.preferences.addons["cycles"].preferences
    p.compute_device_type = "HIP"
    p.get_devices()
    want = f"0000:{JOB.get('hip_bus', '05')}:00"
    hit = False
    for d in p.devices:
        d.use = d.type == "HIP" and d.id.endswith(want)
        hit = hit or d.use
    if not hit:
        raise RuntimeError(f"no HIP device {want}: refusing to render on the CPU")
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y = JOB.get("px", [768, 512])
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.62, 0.65, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(-35))
    sc.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    lo = np.array([min(v[i] for v in xyz) for i in range(3)])
    hi = np.array([max(v[i] for v in xyz) for i in range(3)])
    c = (lo + hi) / 2
    cam.data.lens = 40
    # the rest pose's bounding box, grown by "frame_margin" for limbs that leave it, inside both the width and the
    # height of the picture: per box corner, the distance at which it sits on the picture's edge
    px_ = JOB.get("px", [768, 512])
    tw = 18.0 / cam.data.lens
    th = tw * px_[1] / px_[0]
    d = Vector(JOB.get("view_dir", (0.9, -0.8, 0.35))).normalized()   # from the creature's front-left
    right = d.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(d).normalized()
    mg = float(JOB.get("frame_margin", 1.1))
    dist = 0.0
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                q = (Vector((cx, cy, cz)) - Vector(c)) * mg
                dist = max(dist, q.dot(d) + abs(q.dot(right)) / tw, q.dot(d) + abs(q.dot(up)) / th)
    cam.location = Vector(c) + d * dist
    cam.rotation_euler = (Vector(c) - cam.location).to_track_quat("-Z", "Y").to_euler()
    for f, pz in enumerate(poses, start=1):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(OUT, f"pose_{f - 1}_{pz['name']}.png")
        bpy.ops.render.render(write_still=True)
    T["stills"] = round(time.time() - t, 2)
T["total"] = round(time.time() - t0, 2)
json.dump({"times_s": T, "poses": [p["name"] for p in poses], "stretch": rep_p, "fbx": fbx,
           "fps": sc.render.fps}, open(os.path.join(OUT, "rig.json"), "w"), indent=1)
print("RIG", json.dumps(T), json.dumps(rep_p))
