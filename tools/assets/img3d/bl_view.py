"""Look at a generated .glb in Blender Cycles on a GPU (never the CPU): an overview ring, orthographic axis views for
locating parts (bounds.json gives their scale), and aimed close-ups.

    blender -b --factory-startup -t 0 --python bl_view.py -- MODEL.glb OUT_DIR DEVICE [SHOTS.json]

DEVICE is a substring of the HIP device name ("7900" for a card, "8060" for the iGPU); the caller sets
HIP_VISIBLE_DEVICES to the one device it holds the lock for. SHOTS.json (optional) lists aimed shots:
[{"name", "target": [x, y, z] (Blender metres, after import), "az": deg (0 = from -Y, 90 = from +X), "el": deg,
  "dist": m, "lens": mm}]. Without it: ring (front, three-quarters, side, back) and ortho front/side/top.
Prints BOUNDS and writes OUT_DIR/bounds.json.
"""

import json
import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
glb, out, want = args[:3]
shots = json.load(open(args[3])) if len(args) > 3 else None
bpy.ops.wm.read_factory_settings(use_empty=True)
if glb.endswith(".blend"):
    bpy.ops.wm.open_mainfile(filepath=glb)
else:
    bpy.ops.import_scene.gltf(filepath=glb)
sc = bpy.context.scene
for o in list(sc.objects):
    if o.type in ("CAMERA", "LIGHT"):
        bpy.data.objects.remove(o)
objs = [o for o in sc.objects if o.type == "MESH" and o.visible_get()]
lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
for o in objs:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        lo, hi = Vector(map(min, lo, w)), Vector(map(max, hi, w))
ctr, size = (lo + hi) / 2, max(hi - lo)
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "HIP"
prefs.get_devices()
used = []
for d in prefs.devices:
    d.use = d.type == "HIP" and want.lower() in d.name.lower()
    if d.use:
        used.append(d.name)
if not used:
    raise SystemExit("no GPU device matching " + want + ": " + str([d.name for d in prefs.devices]))
print("DEVICES", used)
sc.cycles.device = "GPU"
sc.cycles.samples = 64
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 1152, 640
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.55, 0.57, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.8
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50), 0, math.radians(35))
sc.collection.objects.link(sun)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam


def shot(name, az, el, dist, target, lens=50, ortho=None):
    cam.data.type = "ORTHO" if ortho else "PERSP"
    if ortho:
        cam.data.ortho_scale = ortho
    cam.data.lens = lens
    cam.data.clip_start = 0.002
    a, e = math.radians(az), math.radians(el)
    cam.location = target + Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * dist
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{out}/{name}.png"
    bpy.ops.render.render(write_still=True)


if shots:
    for s in shots:
        shot(s["name"], s.get("az", 0), s.get("el", 10), s.get("dist", size * 0.4), Vector(s["target"]),
             s.get("lens", 50))
else:
    d = size * 1.9
    for nm, az, el in (("front", 0, 8), ("front34", 35, 12), ("side", 90, 8), ("back34", 215, 15), ("back", 180, 8),
                       ("side_l", 270, 8)):
        shot(nm, az, el, d, ctr)
    # ortho views centred on the bounds' centre; bounds.json gives the scale to read positions off them
    for nm, az, el in (("ortho_front", 0, 0), ("ortho_side", 90, 0), ("ortho_top", 0, 89.9)):
        shot(nm, az, el, size * 3, ctr, ortho=size * 1.15)
b = {"lo": list(lo), "hi": list(hi), "ctr": list(ctr), "size": size, "ortho_scale": size * 1.15,
     "res": [sc.render.resolution_x, sc.render.resolution_y]}
json.dump(b, open(f"{out}/bounds.json", "w"), indent=1)
print("BOUNDS", tuple(lo), tuple(hi))
