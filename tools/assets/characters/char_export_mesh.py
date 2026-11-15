"""Editor-side (scheduler --ue-job): export skeletal mesh assets to FBX files.

JOB = {"meshes": {"/Game/...": "/disk/file.fbx"}, "result": path}
"""

import json
import os
import traceback

import unreal

EAL = unreal.EditorAssetLibrary


def run(job):
    r = {}
    for asset, out in job["meshes"].items():
        try:
            o = EAL.load_asset(asset)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            t = unreal.AssetExportTask()
            t.object, t.filename, t.automated, t.replace_identical, t.prompt = o, out, True, True, False
            t.exporter = unreal.SkeletalMeshExporterFBX()
            opt = unreal.FbxExportOption()
            opt.collision = False
            opt.level_of_detail = False
            opt.export_morph_targets = False
            t.options = opt
            r[asset] = {"ok": unreal.Exporter.run_asset_export_task(t), "bytes": os.path.getsize(out) if os.path.exists(out) else 0}
        except Exception:
            r[asset] = {"error": traceback.format_exc()}
    json.dump(r, open(job["result"], "w"), indent=1)
    return r
