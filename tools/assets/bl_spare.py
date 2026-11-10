"""A warm Blender for the scheduler: started ahead of work on a card, it loads Cycles on that card with a tiny bake,
then waits for ONE harness job and runs bl_harness.py for it in this process, then exits (no state carries over
from one asset to the next). It saves the process start and the HIP device/kernel set-up of a cold Blender
(measured 11-10: 1.4-5 s cold to the first sample v 0.3 s warm).

    blender -b --factory-startup -t N --python tools/assets/bl_spare.py -- SPARE_DIR BUS HIPRT

SPARE_DIR/ready is written when warm; the scheduler then renames a file {"job": harness job JSON, "log": log path}
to SPARE_DIR/job.json. SPARE_DIR/quit (or the parent going away) ends an idle spare.
"""

import json
import os
import runpy
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
SPARE, BUS, HIPRT = sys.argv[sys.argv.index("--") + 1:][:3]
PARENT = os.getppid()


def warm():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    ob = bpy.context.active_object
    me = ob.data
    me.uv_layers.new(name="UVMap")
    m = bpy.data.materials.new("warm")
    m.use_nodes = True
    me.materials.append(m)
    tn = m.node_tree.nodes.new("ShaderNodeTexImage")
    tn.image = bpy.data.images.new("warm", 32, 32)
    m.node_tree.nodes.active = tn
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    p = bpy.context.preferences.addons["cycles"].preferences
    p.compute_device_type = "HIP"
    p.get_devices()
    hit = False
    for d in p.devices:
        d.use = d.type == "HIP" and d.id.endswith(f"0000:{BUS}:00")
        hit = hit or d.use
    if not hit:
        raise SystemExit(f"no HIP device {BUS}")
    p.use_hiprt = HIPRT == "1"
    sc.cycles.device = "GPU"
    sc.cycles.samples = 1
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.bake(type="EMIT")


t0 = time.time()
warm()
with open(os.path.join(SPARE, "ready.tmp"), "w") as f:
    json.dump({"pid": os.getpid(), "warm_s": round(time.time() - t0, 2)}, f)
os.replace(os.path.join(SPARE, "ready.tmp"), os.path.join(SPARE, "ready"))
jp = os.path.join(SPARE, "job.json")
while not os.path.exists(jp):
    if os.path.exists(os.path.join(SPARE, "quit")) or os.getppid() != PARENT:
        sys.exit(0)
    time.sleep(0.02)
job = json.load(open(jp))
fd = os.open(job["log"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
sys.stdout.flush()
sys.stderr.flush()
os.dup2(fd, 1)
os.dup2(fd, 2)
sys.argv = [sys.argv[0], "--", job["job"]]
runpy.run_path(os.path.join(HERE, "bl_harness.py"), run_name="__main__")
sys.stdout.flush()
os._exit(0)                     # the harness exits 2 itself on an error
