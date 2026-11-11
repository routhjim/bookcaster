"""Blender 4.5, CPU bookkeeping only (a few dozen ray casts): put the front frame's tracked 2D face landmarks onto the
skinned head mesh (bl_orbit_skin.py output) as 3D points.

    blender -b -t 0 --factory-startup --python bl_head_landmarks.py -- head.blend skin_report.json track.json \
        track_key out.json

The front view (azimuth 0) maps frame pixels to the mesh exactly as bl_orbit_skin.py projected it (orthographic,
matched in height and centre); each landmark is cast along +Y from in front of the head onto the mesh. Writes
{"units": "m", "frame": "front -Y, +Z up, +X = the head's left", "points": {name: [x, y, z]},
 "distances_m": {...}, "ratios_to_eye_distance": {...}}.
"""

import json
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
blend, rep_p, track_p, key, out_p = argv[:5]
EYE_M = float(argv[5]) if len(argv) > 5 else 0.0      # rescale so the eye centres are this far apart (and export)
bpy.ops.wm.open_mainfile(filepath=blend)
ob = next(o for o in bpy.data.objects if o.type == "MESH")
rep = json.load(open(rep_p))
tr = json.load(open(track_p))
tr = tr.get("track") or tr["result"]["track"]
curves = tr[key]["curves"]
front = next(v for v in rep["views"] if abs(v["az"]) < 1e-6)
bb, s = front["bbox"], front["px_per_m"]
co = [v.co for v in ob.data.vertices]
xs = [c.x for c in co]
zmax = max(c.z for c in co)
cx_m = (min(xs) + max(xs)) / 2
cx_img = (bb[0] + bb[2]) / 2


def to_mesh(px, py):
    x = (px - cx_img) / s + cx_m
    z = zmax - (py - bb[1]) / s
    ok, loc, nrm, idx = ob.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)))
    return [round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)] if ok else None


def mean(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


P = {}
for side in ("l", "r"):
    up, lo = curves[f"crv_eyelid_upper_{side}"], curves[f"crv_eyelid_lower_{side}"]
    P[f"eye_centre_{side}"] = to_mesh(*mean(up + lo))
    P[f"eye_corner_a_{side}"] = to_mesh(*up[0])
    P[f"eye_corner_b_{side}"] = to_mesh(*up[-1])
    P[f"mouth_corner_{side}"] = to_mesh(*curves[f"crv_lip_upper_outer_{side}"][-1])
    P[f"nasolabial_top_{side}"] = to_mesh(*curves[f"crv_nasolabial_{side}"][0])
P["upper_lip_top"] = to_mesh(*curves["crv_lip_philtrum_l"][-1])
P["lip_seam_centre"] = to_mesh(*curves["crv_lip_upper_inner_l"][0])
P["lower_lip_bottom"] = to_mesh(*curves["crv_lip_lower_outer_l"][0])
# the nose tip and the chin: the mesh's front-most point on the centre line above / below the mouth
mid = [c for c in co if abs(c.x - cx_m) < 0.004]
mouth_z = P["lip_seam_centre"][2] if P["lip_seam_centre"] else zmax - 0.15
eye_z = P["eye_centre_l"][2] if P["eye_centre_l"] else zmax - 0.1
nose = min((c for c in mid if mouth_z < c.z < eye_z), key=lambda c: c.y, default=None)
chin = min((c for c in mid if c.z < mouth_z - 0.02), key=lambda c: c.y, default=None)
P["nose_tip"] = [round(nose.x, 4), round(nose.y, 4), round(nose.z, 4)] if nose else None
P["chin_front"] = [round(chin.x, 4), round(chin.y, 4), round(chin.z, 4)] if chin else None
P["crown"] = [round(cx_m, 4), 0.0, round(zmax, 4)]


def d(a, b):
    return math.dist(P[a], P[b]) if P.get(a) and P.get(b) else None


ed = d("eye_centre_l", "eye_centre_r")
D = {"eye_distance": ed, "mouth_width": d("mouth_corner_l", "mouth_corner_r"),
     "eyes_to_mouth": (eye_z - mouth_z) if P["lip_seam_centre"] else None,
     "eyes_to_nose_tip": (eye_z - P["nose_tip"][2]) if P["nose_tip"] else None,
     "eyes_to_chin_front": (eye_z - P["chin_front"][2]) if P["chin_front"] else None,
     "crown_to_eyes": zmax - eye_z,
     "nose_projection_from_eyes": (P["eye_centre_l"][1] - P["nose_tip"][1]) if P["nose_tip"] else None,
     "head_width": max(xs) - min(xs),
     "head_depth": max(c.y for c in co) - min(c.y for c in co)}
scale = 1.0
if EYE_M and ed:
    scale = EYE_M / ed
    ob.data.transform(__import__("mathutils").Matrix.Scale(scale, 4))
    ob.data.update()
    P = {k: [round(c * scale, 4) for c in v] if v else None for k, v in P.items()}
    D = {k: (v * scale if v is not None else None) for k, v in D.items()}
    ed = EYE_M
    glb = out_p.rsplit(".", 1)[0] + ".glb"
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=glb, use_selection=True, export_format="GLB")
out = {"units": "m", "scale_applied": round(scale, 4), "frame": "front -Y, +Z up, +X = the viewer's right seen from the front (the head's left)",
       "source": {"blend": blend, "front_view": front}, "points": P,
       "distances_m": {k: round(v, 4) if v is not None else None for k, v in D.items()},
       "ratios_to_eye_distance": {k: round(v / ed, 3) for k, v in D.items() if v is not None and ed}}
json.dump(out, open(out_p, "w"), indent=1)
print("LANDMARKS " + json.dumps(out["ratios_to_eye_distance"]))
