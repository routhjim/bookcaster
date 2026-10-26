"""Blender side: pose a rigged model (tools/blockout_model*.py) with a RigMotion and render / bake it.

    ~/opt/blender/current/blender -b -t 16 --factory-startup --python tools/motion_render_bl.py -- JOB.json

JOB {"blend": ".../rigged.blend", "motion": "<RigMotion base>", "frames": [0, 10, ...] | "all",
     "out_dir": dir, "res": 512, "views": ["34", "side", "front"], "clay": true,
     "bake": "/abs/out.blend" (optional: keyframe every frame, save the animated rig),
     "markers": {"r_hand": [r, g, b]} (optional: small spheres on joints)}
Posing: every bone gets pose_bone.matrix = R_bone @ rest (R from the RigMotion's bone_rot, which carries
twist: wrist roll, forearm, pelvis/chest turn) with its head at the motion's head joint, parents first.
Writes out_dir/<view>_<frame>.png (Workbench, studio light, ortho camera fitted to the whole clip).
"""

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
job = json.load(open(argv[0]))
TOOLS = os.path.dirname(os.path.abspath(__file__))

base = job["motion"][:-4] if job["motion"].endswith(".npz") else job["motion"]
Z = np.load(base + ".npz")
H = json.load(open(base + ".json"))
POS, ROT = Z["pos"], Z["bone_rot"]
NAMES, BONES = H["joints"], H["bones"]
JI = {n: i for i, n in enumerate(NAMES)}
HEAD_OF = {"pelvis": "pelvis", "spine": "spine_mid", "neck": "neck", "head": "head", "r_clav": "neck",
           "l_clav": "neck", "r_uarm": "r_shoulder", "l_uarm": "l_shoulder", "r_farm": "r_elbow",
           "l_farm": "l_elbow", "r_hand": "r_wrist", "l_hand": "l_wrist", "r_hipb": "pelvis", "l_hipb": "pelvis",
           "r_thigh": "r_hip", "l_thigh": "l_hip", "r_shin": "r_knee", "l_shin": "l_knee", "r_foot": "r_ankle",
           "l_foot": "l_ankle"}

bpy.ops.wm.open_mainfile(filepath=job["blend"])
scene = bpy.context.scene
arm = bpy.data.objects["rig"]
ob = bpy.data.objects["model"]
for o in list(scene.objects):
    if o not in (arm, ob):
        bpy.data.objects.remove(o, do_unlink=True)
arm.matrix_world = Matrix.Identity(4)
order = [b.name for b in arm.data.bones if b.parent is None]
i = 0
while i < len(order):
    order += [c.name for c in arm.data.bones[order[i]].children]
    i += 1
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"


def pose(f):
    for name in order:
        b = arm.data.bones[name]
        pb = arm.pose.bones[name]
        R = Matrix(ROT[f, BONES.index(name)].tolist())
        M = (R @ b.matrix_local.to_3x3()).to_4x4()
        M.translation = Vector(POS[f, JI[HEAD_OF[name]]].tolist())
        pb.matrix = M
        bpy.context.view_layer.update()


F = len(POS)
frames = list(range(F)) if job.get("frames", "all") == "all" else [int(f) for f in job["frames"]]

if job.get("bake"):
    scene.frame_start, scene.frame_end = 0, F - 1
    scene.render.fps = int(round(float(Z["fps"])))
    for f in range(F):
        scene.frame_set(f)
        pose(f)
        for pb in arm.pose.bones:
            pb.keyframe_insert("rotation_quaternion", frame=f)
            pb.keyframe_insert("location", frame=f)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.save_as_mainfile(filepath=job["bake"])
    bpy.ops.object.mode_set(mode="POSE")
    print("BAKED", job["bake"], F)

if job.get("out_dir"):
    os.makedirs(job["out_dir"], exist_ok=True)
    if job.get("clay", True):
        mat = bpy.data.materials.new("clay")
        mat.diffuse_color = (0.75, 0.72, 0.68, 1)
        ob.data.materials.clear()
        ob.data.materials.append(mat)
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    gp = bpy.context.active_object
    gm = bpy.data.materials.new("ground")
    gm.diffuse_color = (0.35, 0.37, 0.4, 1)
    gp.data.materials.append(gm)
    bpy.context.view_layer.objects.active = arm
    markers = []
    for jn, rgb in job.get("markers", {}).items():
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035)
        s = bpy.context.active_object
        m = bpy.data.materials.new("m_" + jn)
        m.diffuse_color = (*rgb, 1)
        s.data.materials.append(m)
        markers.append((jn, s))
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    sh = scene.display.shading
    scene.render.engine = "BLENDER_WORKBENCH"
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_shadows = True
    sh.show_cavity = True
    scene.render.resolution_x = scene.render.resolution_y = int(job.get("res", 512))
    scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new("w")
    scene.world.color = (0.9, 0.9, 0.92)
    lo, hi = POS.reshape(-1, 3).min(0), POS.reshape(-1, 3).max(0)
    c = (lo + hi) / 2
    ext = float(max(hi - lo)) * 1.15 + 0.2
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    cam_d.ortho_scale = ext
    cam = bpy.data.objects.new("cam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    yaw0 = math.radians(float(H.get("facing_deg", 0.0)))
    VIEWS = {"34": -50.0, "side": -90.0, "front": 0.0, "back": 180.0}
    for f in frames:
        if job.get("bake"):          # baked keyframes drive the pose now: evaluate them at this frame
            scene.frame_set(f)
        else:
            pose(f)
        for jn, s in markers:
            s.location = Vector(POS[f, JI[jn]].tolist())
        for v in job.get("views", ["34", "side"]):
            a = yaw0 + math.radians(VIEWS[v])
            d = Vector((math.cos(a), math.sin(a), 0.18)).normalized()
            cam.location = Vector(c.tolist()) + d * 10
            cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = os.path.join(job["out_dir"], f"{v}_{f:04d}.png")
            bpy.ops.render.render(write_still=True)
    print("RENDERED", len(frames), "frames")
