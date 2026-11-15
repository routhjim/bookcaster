"""Blender side of the orbit-skin stage, part 2: for every fitted orbit camera, the world position seen at each pixel
(Cycles Position pass, 1 sample, no pixel filter, so a pixel is one exact surface point) and the coverage (alpha).
A camera whose fy differs from fx (fy_over_fx) renders with square pixels over the same field, then is
resampled to the frame's rows. GPU only.

    blender -b --factory-startup -t 4 --python bl_views.py -- JOB.json
    JOB {"blend": <id>_uv.blend, "id": asset id, "cams": cams_asset.json, "out": folder, "hip_bus": "05",
         "names": [optional subset]}

Writes out/<view>_pos.npy (H x W x 4 float32: x, y, z, alpha; rows bottom-up as Blender stores images).
"""
import json
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
AID = JOB["id"]
OUT = JOB["out"]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
bpy.ops.wm.open_mainfile(filepath=JOB["blend"])
sc = bpy.context.scene
for o in list(bpy.data.objects):
    if o.name != f"SM_{AID}":
        bpy.data.objects.remove(o, do_unlink=True)
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
p.use_hiprt = False
sc.render.engine = "CYCLES"
sc.cycles.device = "GPU"
sc.cycles.samples = 1
sc.cycles.use_denoising = False
sc.cycles.max_bounces = 0
sc.render.filter_size = 0.0
sc.render.film_transparent = True
sc.render.resolution_percentage = 100
vl = sc.view_layers[0]
vl.use_pass_position = True
sc.use_nodes = True
tree = sc.node_tree
tree.nodes.clear()
rl = tree.nodes.new("CompositorNodeRLayers")
comp = tree.nodes.new("CompositorNodeComposite")
tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
fo = tree.nodes.new("CompositorNodeOutputFile")
fo.format.file_format = "OPEN_EXR"
fo.format.color_depth = "32"
fo.file_slots.clear()
for s in ("Image", "Position"):
    fo.file_slots.new(s)
    tree.links.new(rl.outputs[s], fo.inputs[s])
cd = bpy.data.cameras.new("orbit_cam")
cam = bpy.data.objects.new("orbit_cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
V = json.load(open(JOB["cams"]))["views"]
names = set(JOB.get("names") or [v["name"] for v in V])
n = 0
for v in V:
    if v["name"] not in names:
        continue
    W, H0 = v["size"]
    asp = float(v.get("fy_over_fx", 1.0))
    H = int(round(H0 / asp))                   # square-pixel render covering the frame's vertical field
    sc.render.resolution_x, sc.render.resolution_y = W, H
    cam.matrix_world = Matrix(v["matrix_world"])
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = v["sensor_width"]
    cd.lens = v["lens"]
    cd.shift_x, cd.shift_y = v.get("shift_x", 0.0), v.get("shift_y", 0.0)
    cd.clip_start, cd.clip_end = 0.05, 1000.0
    tmp = os.path.join(OUT, f"_tmp_{v['name']}")
    fo.base_path = tmp + os.sep
    sc.render.filepath = os.path.join(OUT, "_composite.png")
    bpy.ops.render.render(write_still=False)
    ims = {}
    for s in ("Image", "Position"):
        f = os.path.join(tmp, f"{s}0001.exr")
        im = bpy.data.images.load(f)
        a = np.empty(W * H * 4, np.float32)
        im.pixels.foreach_get(a)
        ims[s] = a.reshape(H, W, 4)
        bpy.data.images.remove(im)
        os.remove(f)
    os.rmdir(tmp)
    arr = np.concatenate([ims["Position"][..., :3], ims["Image"][..., 3:4]], -1).astype(np.float32)
    if H != H0:                                # back onto the frame's rows (nearest: positions stay exact)
        rows = np.clip(np.floor((np.arange(H0) + 0.5 - H0 / 2) / asp + H / 2), 0, H - 1).astype(int)
        arr = arr[rows]
    np.save(os.path.join(OUT, f"{v['name']}_pos.npy"), arr)
    n += 1
print("VIEWS", n, round(time.time() - t0, 1))
