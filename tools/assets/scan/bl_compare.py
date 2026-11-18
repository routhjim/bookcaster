"""Blender 4.5 (Cycles on a HIP card only): render several meshes of the same subject from the same sides for a side-by-
side look. Each mesh is levelled the same way: long axis on X (principal axis of its ground-plane footprint), the
taller end (the head) towards +X, centred, scaled to the same length.

    blender -b -P bl_compare.py -- OUT_DIR LEN_M NAME=MESH[:grey|col] ...

MESH: .ply (vertex colours) or .glb. Writes OUT_DIR/<NAME>_<view>.png (views: front, left, back, right, top3q).
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out, length = Path(argv[0]), float(argv[1])
items = [x.split("=", 1) for x in argv[2:]]
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "HIP"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "HIP"
if not any(d.use for d in prefs.devices):
    raise SystemExit("no HIP device: refusing to render on the CPU")
sc.cycles.device = "GPU"
sc.cycles.samples = 48
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 900, 600
sc.render.film_transparent = True
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
sc.world = world
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(40), 0, math.radians(30))
sc.collection.objects.link(sun)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"


def grey_mat():
    m = bpy.data.materials.new("grey")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1)
    b.inputs["Roughness"].default_value = 0.7
    return m


def vcol_mat(attr):
    m = bpy.data.materials.new("vcol")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.8
    a = nt.nodes.new("ShaderNodeVertexColor")
    a.layer_name = attr
    nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    return m


for name, spec in items:
    path, _, look = spec.partition(":")
    look = look or "col"
    for o in list(sc.objects):
        if o.type == "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    if path.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    else:
        bpy.ops.import_scene.gltf(filepath=path)
    obs = [o for o in sc.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    me = ob.data
    def verts():
        buf = np.zeros(len(me.vertices) * 3)
        me.vertices.foreach_get("co", buf)
        return buf.reshape(-1, 3)

    V = verts()
    c = V[:, :2].mean(0)
    ev, evec = np.linalg.eigh(np.cov((V[:, :2] - c).T))
    ax = evec[:, 1]
    ang = -math.atan2(ax[1], ax[0])
    Rz = Matrix.Rotation(ang, 4, "Z")
    X = (V[:, :2] - c) @ np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
    xs = X[:, 0]
    lo, hi = np.quantile(xs, 0.15), np.quantile(xs, 0.85)
    flip = V[xs < lo, 2].max() > V[xs > hi, 2].max()
    Rf = Matrix.Rotation(math.pi if flip else 0, 4, "Z")
    ob.matrix_world = Rf @ Rz @ Matrix.Translation(Vector((-c[0], -c[1], -V[:, 2].min())))
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    V = verts()
    s = length / (V[:, 0].max() - V[:, 0].min())
    xc = (V[:, 0].max() + V[:, 0].min()) / 2
    ob.matrix_world = Matrix.Scale(s, 4) @ Matrix.Translation(Vector((-xc, -(V[:, 1].max() + V[:, 1].min()) / 2, 0)))
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for p in me.polygons:
        p.use_smooth = True
    if look == "grey" or not (path.endswith(".ply") and me.color_attributes):
        if path.endswith(".ply") or look == "grey":
            me.materials.clear()
            me.materials.append(grey_mat())
    else:
        me.materials.clear()
        me.materials.append(vcol_mat(me.color_attributes[0].name))
    V = verts()
    ext = V.max(0) - V.min(0)
    zc = ext[2] / 2
    views = {"front": (0, 0), "left": (90, 0), "back": (180, 0), "right": (270, 0), "top3q": (35, 35)}
    for vn, (azd, eld) in views.items():
        az_, el_ = math.radians(azd), math.radians(eld)
        dvec = Vector((math.cos(az_) * math.cos(el_), math.sin(az_) * math.cos(el_), math.sin(el_)))
        cam.location = Vector((0, 0, zc)) + dvec * 40
        cam.rotation_euler = (-dvec).to_track_quat("-Z", "Y").to_euler()
        cam.data.ortho_scale = max(ext) * 1.08
        cam.data.clip_end = 200
        sc.render.filepath = str(out / f"{name}_{vn}.png")
        bpy.ops.render.render(write_still=True)
print("COMPARE_DONE", flush=True)
