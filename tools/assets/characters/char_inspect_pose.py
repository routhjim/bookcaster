"""Editor-side (scheduler --ue-job): reference-pose transforms (world/component space) of chosen bones in several
skeletons, to check that garments imported from Blender share the character body's bind pose.

JOB = {"skeletons": {"label": "/Game/.../Skeleton"}, "bones": [...], "result": path}
"""

import json
import traceback

import unreal


def run(job):
    r = {}
    AP = unreal.AnimPoseExtensions
    for label, path in job["skeletons"].items():
        try:
            sk = unreal.EditorAssetLibrary.load_asset(path)
            pose = AP.get_reference_pose(sk)
            names = [str(n) for n in AP.get_bone_names(pose)]
            row = {"n": len(names), "first": names[:4]}
            for b in job["bones"]:
                t = AP.get_bone_pose(pose, b, unreal.AnimPoseSpaces.WORLD)
                q = t.rotation
                row[b] = [round(t.translation.x, 2), round(t.translation.y, 2), round(t.translation.z, 2),
                          round(q.x, 3), round(q.y, 3), round(q.z, 3), round(q.w, 3),
                          round(t.scale3d.x, 3)]
            r[label] = row
        except Exception:
            r[label] = {"error": traceback.format_exc()}
    json.dump(r, open(job["result"], "w"), indent=1)
    return {"ok": True}
