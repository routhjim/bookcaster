"""Blender (background): the HANDLE of each jointed part of a prop model = the part's vertex farthest from its hinge
axis among the part's upper 40 % (where a hand pushes: the bellows board's free end). Prints one line per joint:
HANDLE <name> [x, y, z] (prop-model metres, the same frame as pivot_m).

    blender -b <model.blend> --python tools/props_handle_bl.py -- '<joints json>'
"""

import json
import sys

import bpy
import numpy as np

joints = json.loads(sys.argv[sys.argv.index("--") + 1])
for jd in joints:
    ob = bpy.data.objects.get(jd.get("object") or f"joint_{jd['name']}")
    if ob is None or ob.type != "MESH" or not len(ob.data.vertices):
        continue
    co = np.array([v.co[:] for v in ob.data.vertices])           # relative to the pivot (the part's origin)
    ax = {"x": 0, "y": 1, "z": 2}[jd.get("axis", "y")]
    o = [a for a in (0, 1, 2) if a != ax]
    r = np.hypot(co[:, o[0]], co[:, o[1]])
    top = co[:, 2] > np.percentile(co[:, 2], 60)
    i = int(np.argmax(np.where(top, r, -1.0)))
    h = np.asarray(ob.location[:]) + co[i]
    print("HANDLE " + jd["name"] + " " + json.dumps([round(float(x), 4) for x in h]), flush=True)
