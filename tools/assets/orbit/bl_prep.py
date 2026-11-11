"""Blender side of the orbit-skin stage, part 1: the mesh in the asset frame -> an object with UVs, and every texel's
world position and normal baked to arrays (Cycles on a GPU; refuses the CPU).

    blender -b --factory-startup -t 4 --python bl_prep.py -- JOB.json
    JOB {"mesh": mesh_asset.ply, "id": asset id, "res": 4096, "out": folder, "hip_bus": "05", "decimate": 0 (faces),
         "smooth": true}

Writes out/<id>_uv.blend (one object SM_<id>, a Principled material named M_<id>, UV map "UVMap"),
out/texel_pos.npy, texel_nrm.npy (res x res x 3, float16, rows bottom-up as Blender stores images; -1e4 where no
face covers the texel), out/prep.json (times, faces).
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
AID = JOB["id"]
OUT = JOB["out"]
os.makedirs(OUT, exist_ok=True)
T = {}
t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
p = bpy.context.preferences.addons["cycles"].preferences
p.compute_device_type = "HIP"
p.get_devices()
want = f"0000:{JOB.get('hip_bus', '05')}:00"
hit = False
for d in p.devices:
    d.use = d.type == "HIP" and d.id.endswith(want)
    hit = hit or d.use
if not hit:
    raise RuntimeError(f"no HIP device {want}: refusing to bake on the CPU")
p.use_hiprt = False
sc.render.engine = "CYCLES"
sc.cycles.device = "GPU"

bpy.ops.wm.ply_import(filepath=JOB["mesh"])
ob = bpy.context.selected_objects[0]
ob.name = ob.data.name = f"SM_{AID}"
bpy.context.view_layer.objects.active = ob
T["import"] = round(time.time() - t0, 2)
t = time.time()
if JOB.get("decimate"):
    n = len(ob.data.polygons)
    if n > JOB["decimate"]:
        md = ob.modifiers.new("dec", "DECIMATE")
        md.ratio = JOB["decimate"] / n
        bpy.ops.object.modifier_apply(modifier="dec")
T["decimate"] = round(time.time() - t, 2)
t = time.time()
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.002)
bpy.ops.object.mode_set(mode="OBJECT")
if JOB.get("smooth", True):
    bpy.ops.object.shade_smooth()
T["uv"] = round(time.time() - t, 2)
m = bpy.data.materials.new(f"M_{AID}")
m.use_nodes = True
ob.data.materials.clear()
ob.data.materials.append(m)
res = int(JOB.get("res", 4096))

# texel position / normal: emission bakes into float images
t = time.time()
nt = m.node_tree
pr = nt.nodes["Principled BSDF"]
outn = nt.nodes["Material Output"]
em = nt.nodes.new("ShaderNodeEmission")
geo = nt.nodes.new("ShaderNodeNewGeometry")
tn = nt.nodes.new("ShaderNodeTexImage")
nt.nodes.active = tn
nt.links.new(em.outputs[0], outn.inputs["Surface"])
sc.render.bake.margin = 0
sc.cycles.samples = 1
arrs = {}
for key, sock in (("pos", "Position"), ("nrm", "Normal")):
    im = bpy.data.images.new(f"texel_{key}", res, res, alpha=True, float_buffer=True)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(np.full(res * res * 4, -1e4, dtype=np.float32))
    tn.image = im
    nt.links.new(geo.outputs[sock], em.inputs["Color"])
    bpy.ops.object.bake(type="EMIT", use_clear=False)
    a = np.empty(res * res * 4, np.float32)
    im.pixels.foreach_get(a)
    arrs[key] = a.reshape(res, res, 4)[..., :3]
    bpy.data.images.remove(im)
T["bake"] = round(time.time() - t, 2)
# Emission takes the colour as given, but the bake writes linear values; positions are exact, normals unit
cov = arrs["pos"][..., 0] > -1e3
np.save(os.path.join(OUT, "texel_pos.npy"), np.where(cov[..., None], arrs["pos"], -1e4).astype(np.float32))
np.save(os.path.join(OUT, "texel_nrm.npy"), arrs["nrm"].astype(np.float16))
nt.nodes.remove(em)
nt.nodes.remove(geo)
nt.nodes.remove(tn)
nt.links.new(pr.outputs[0], outn.inputs["Surface"])
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f"{AID}_uv.blend"), compress=True)
T["total"] = round(time.time() - t0, 2)
json.dump({"times_s": T, "faces": len(ob.data.polygons), "verts": len(ob.data.vertices),
           "coverage": round(float(cov.mean()), 4), "device": want}, open(os.path.join(OUT, "prep.json"), "w"),
          indent=1)
print("PREP", json.dumps(T))
