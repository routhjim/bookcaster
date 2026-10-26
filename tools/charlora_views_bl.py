"""Blender side of tools/charlora.py (any book, any character model): render a character model from many
cameras, optionally posed by RigMotion frames, as colour + depth pairs for training-data structure.

    ~/opt/blender/current/blender -b -t 16 --factory-startup --python tools/charlora_views_bl.py -- JOB.json

JOB {"blend": ".../rigged.blend" (objects "model" [+ armature "rig"]), "out_dir": dir, "size": [w, h],
     "views": [{"id": "v000", "yaw": deg (0 = in front of the model, which faces +X), "elev": deg,
                "frame": "head" | "half" | "full", "lens": 50, "light_rot": deg,
                "motion": "<RigMotion base>" (optional), "motion_frame": int}]}
Writes <out_dir>/<id>_color.png (Workbench, the model's vertex colours) and <id>_depth.npy (camera Z,
inf where empty). The posing is tools/motion_render_bl.py's (pose_bone.matrix = R_bone @ rest, heads at
the motion's joints, parents first).
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
os.makedirs(job["out_dir"], exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=job["blend"])
sc = bpy.context.scene
arm = bpy.data.objects.get("rig")
ob = bpy.data.objects.get("model")
if ob is None:          # a composite model (creature parts): every visible mesh, joined into one "model"
    meshes = [o for o in sc.objects if o.type == "MESH" and not o.hide_render]
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = "model"
    if arm is None:
        arm = next((o for o in sc.objects if o.type == "ARMATURE"), None)
for o in list(sc.objects):
    if o not in (ob, arm):
        bpy.data.objects.remove(o, do_unlink=True)

HEAD_OF = {"pelvis": "pelvis", "spine": "spine_mid", "neck": "neck", "head": "head", "r_clav": "neck",
           "l_clav": "neck", "r_uarm": "r_shoulder", "l_uarm": "l_shoulder", "r_farm": "r_elbow",
           "l_farm": "l_elbow", "r_hand": "r_wrist", "l_hand": "l_wrist", "r_hipb": "pelvis", "l_hipb": "pelvis",
           "r_thigh": "r_hip", "l_thigh": "l_hip", "r_shin": "r_knee", "l_shin": "l_knee", "r_foot": "r_ankle",
           "l_foot": "l_ankle"}
_motions = {}


def rest():
    if arm is None:
        return
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def pose(base, f):
    if arm is None:
        return
    if base not in _motions:
        Z = np.load(base + ".npz")
        H = json.load(open(base + ".json"))
        _motions[base] = (Z["pos"], Z["bone_rot"], H["joints"], H["bones"])
    POS, ROT, NAMES, BONES = _motions[base]
    JI = {n: i for i, n in enumerate(NAMES)}
    f = max(0, min(int(f), len(POS) - 1))
    order = [b.name for b in arm.data.bones if b.parent is None]
    i = 0
    while i < len(order):
        order += [c.name for c in arm.data.bones[order[i]].children]
        i += 1
    for name in order:
        if name not in BONES or name not in HEAD_OF:
            continue
        b = arm.data.bones[name]
        pb = arm.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        R = Matrix(ROT[f, BONES.index(name)].tolist())
        M = (R @ b.matrix_local.to_3x3()).to_4x4()
        M.translation = Vector(POS[f, JI[HEAD_OF[name]]].tolist())
        pb.matrix = M
        bpy.context.view_layer.update()


def bounds():
    dg = bpy.context.evaluated_depsgraph_get()
    e = ob.evaluated_get(dg)
    me = e.to_mesh()
    v = np.array([(e.matrix_world @ x.co)[:] for x in me.vertices], np.float32)
    e.to_mesh_clear()
    return v


if arm is not None:
    arm.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")

W, H = job.get("size", [768, 1024])
sc.render.engine = "BLENDER_WORKBENCH"
sh = sc.display.shading
sh.light = "STUDIO"
sh.color_type = "VERTEX" if ob.data.color_attributes else "MATERIAL"
sh.show_shadows = True
sh.show_cavity = True
sc.display.render_aa = "8"
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.view_settings.view_transform = "Standard"
sc.view_layers[0].use_pass_z = True
sc.use_nodes = True
nt = sc.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new("CompositorNodeRLayers")
comp = nt.nodes.new("CompositorNodeComposite")
fo = nt.nodes.new("CompositorNodeOutputFile")
fo.format.file_format = "OPEN_EXR"
fo.format.color_depth = "32"
fo.file_slots[0].path = "depth_"
nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
nt.links.new(rl.outputs["Depth"], fo.inputs[0])
sc.render.use_compositing = True

cam_d = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam

FRAMES = {"head": (0.86, 1.0, 0.19), "half": (0.5, 1.0, 0.55), "full": (0.0, 1.0, 1.08)}
for v in job["views"]:
    if v.get("motion"):
        pose(v["motion"], v.get("motion_frame", 0))
    else:
        rest()
    P = bounds()
    lo, hi = P.min(0), P.max(0)
    hgt = float(hi[2] - lo[2])
    a, b, cover = FRAMES[v.get("frame", "full")]
    z0, z1 = lo[2] + a * hgt, lo[2] + b * hgt
    sel = P[(P[:, 2] >= z0 - 0.02) & (P[:, 2] <= z1 + 0.02)]
    c = (sel.min(0) + sel.max(0)) / 2
    span = max(cover * hgt, float(np.ptp(sel[:, 0])), float(np.ptp(sel[:, 1])), (z1 - z0) * 1.15)
    lens = float(v.get("lens", 50))
    cam_d.lens = lens
    cam_d.sensor_fit = "VERTICAL"
    fov = 2 * math.atan(cam_d.sensor_height / 2 / lens)          # vertical field of view
    span = max(span, max(float(np.ptp(sel[:, 0])), float(np.ptp(sel[:, 1]))) * H / W)
    dist = (span * 1.12 / 2) / math.tan(fov / 2) + 0.15
    yaw, el = math.radians(v.get("yaw", 0)), math.radians(v.get("elev", 5))
    d = Vector((math.cos(yaw) * math.cos(el), math.sin(yaw) * math.cos(el), math.sin(el)))
    target = Vector(c.tolist())
    cam.location = target + d * dist
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sh.studiolight_rotate_z = math.radians(v.get("light_rot", 0))
    tmp = os.path.join(job["out_dir"], f".{v['id']}_d")
    os.makedirs(tmp, exist_ok=True)
    fo.base_path = tmp
    sc.frame_set(1)
    sc.render.filepath = os.path.join(job["out_dir"], f"{v['id']}_color.png")
    bpy.ops.render.render(write_still=True)
    exr = os.path.join(tmp, "depth_0001.exr")
    img = bpy.data.images.load(exr)
    img.colorspace_settings.name = "Non-Color"
    px = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], -1)[::-1, :, 0]
    np.save(os.path.join(job["out_dir"], f"{v['id']}_depth.npy"), px)
    bpy.data.images.remove(img)
    os.remove(exr)
    os.rmdir(tmp)
    print("VIEW", v["id"], flush=True)
print("VIEWS_DONE", len(job["views"]))
