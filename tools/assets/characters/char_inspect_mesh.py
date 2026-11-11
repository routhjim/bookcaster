"""Editor-side (scheduler --ue-job): compare imported garment skeletal meshes with a character body mesh: bounds,
bone counts, root bones, bones missing on either side, and the reference-pose position of a few bones.

JOB = {"meshes": ["/Game/...", ...], "body": "/Game/.../SKM_Body", "result": path}
"""

import json
import traceback

import unreal

EAL = unreal.EditorAssetLibrary
BONES = ("root", "pelvis", "spine_05", "head", "hand_l", "index_03_l", "foot_l", "thigh_l")


def info(path):
    m = EAL.load_asset(path)
    r = {"path": path}
    b = m.get_bounds()
    r["bounds"] = [[round(b.origin.x, 1), round(b.origin.y, 1), round(b.origin.z, 1)],
                   [round(b.box_extent.x, 1), round(b.box_extent.y, 1), round(b.box_extent.z, 1)]]
    sk = m.skeleton
    r["skeleton"] = sk.get_path_name() if sk else None
    names = []
    try:
        n = sk.get_reference_pose() if False else None
    except Exception:
        pass
    try:
        lib = unreal.SkeletalMeshEditorSubsystem if hasattr(unreal, "SkeletalMeshEditorSubsystem") else None
    except Exception:
        lib = None
    try:
        # bone names through the skeleton's reference
        i = 0
        while True:
            nm = sk.get_bone_name(i) if hasattr(sk, "get_bone_name") else None
            if not nm or str(nm) == "None":
                break
            names.append(str(nm))
            i += 1
            if i > 2000:
                break
    except Exception as e:
        r["names_err"] = str(e)
    r["n_bones"] = len(names)
    r["first"] = names[:5]
    r["names"] = names
    try:
        ref = unreal.AnimationLibrary  # noqa
        pose = {}
        for bn in BONES:
            try:
                t = m.get_ref_pose_component_transform(bn) if hasattr(m, "get_ref_pose_component_transform") else None
                if t:
                    pose[bn] = [round(t.translation.x, 1), round(t.translation.y, 1), round(t.translation.z, 1)]
            except Exception:
                pass
        r["pose"] = pose
    except Exception as e:
        r["pose_err"] = str(e)
    return r


def run(job):
    out = {}
    try:
        for p in job["meshes"] + [job["body"]]:
            out[p] = info(p)
        bn = set(out[job["body"]]["names"])
        for p in job["meshes"]:
            mine = set(out[p]["names"])
            out[p]["missing_in_body"] = sorted(mine - bn)[:40]
            out[p]["n_missing_in_body"] = len(mine - bn)
        for v in out.values():
            v.pop("names", None)
    except Exception:
        out["error"] = traceback.format_exc()
    json.dump(out, open(job["result"], "w"), indent=1, default=str)
    return {"ok": "error" not in out}
