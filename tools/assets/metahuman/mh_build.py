"""UE side: make a MetaHuman Character and assemble it, including the two cloud steps (needs a signed-in Epic account).

    python tools/assets/metahuman/mh_run.py cmdlet tools/assets/metahuman/mh_build.py --job job.json
    job = {"result": path, "folder": "/Game/MH", "name": "MH_A",
           "preset": "<preset name>" | null,          (copy a preset shipped in the plugin; else defaults)
           "rig_type": "JOINTS_AND_BLEND_SHAPES" | "JOINTS_ONLY",
           "pipeline": "CINEMATIC" | "OPTIMIZED", "quality": "HIGH" | "MEDIUM" | ...,
           "build_path": "/Game/MetaHumans", "cloud": true}

Steps: copy preset (or create from defaults) -> try_add_object_to_edit -> request_auto_rigging (cloud, blocking)
-> request_texture_sources (cloud, blocking) -> can_build_meta_human -> build_meta_human. Each step's state is in the
result JSON, so a run that stops at the cloud shows exactly where. With "cloud": false the two requests are skipped
(this is how it was run on the unsigned box: the build gate then reports the missing step).

Unattended use: the cloud requests reuse the editor's stored Epic login (EOS persistent auth). Without one, the
service answers 401 and the plugin starts Epic's account-portal sign-in in a browser; pass
-NoMetaHumanAccountPortalLoginFallback (env MH_EXTRA_ARGS) to make an unattended run fail instead of waiting.
"""

import json
import os
import time
import traceback

import unreal

job = json.load(open(os.environ["MH_JOB"]))
EAL = unreal.EditorAssetLibrary
mhs = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
at = unreal.AssetToolsHelpers.get_asset_tools()
folder, name = job.get("folder", "/Game/MH"), job["name"]
path = f"{folder}/{name}"
r = {"ok": False, "steps": []}


def step(n, **kw):
    kw["step"], kw["t_s"] = n, round(time.time() - T0, 1)
    r["steps"].append(kw)
    unreal.log_warning(f"MH build: {json.dumps(kw, default=str)}")


T0 = time.time()
c = None
try:
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    if job.get("preset"):
        unreal.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous(
            ["/MetaHumanCharacter/Optional/Presets"], True)
        c = EAL.duplicate_asset(f"/MetaHumanCharacter/Optional/Presets/{job['preset']}", path)
    else:
        c = at.create_asset(asset_name=name, package_path=folder, asset_class=unreal.MetaHumanCharacter,
                            factory=unreal.new_object(type=unreal.MetaHumanCharacterFactoryNew))
    step("create", ok=c is not None)
    if not mhs.try_add_object_to_edit(c):
        raise RuntimeError("try_add_object_to_edit failed")
    if job.get("cloud", True):
        p = unreal.MetaHumanCharacterAutoRiggingRequestParams()
        p.blocking, p.report_progress = True, False
        p.rig_type = getattr(unreal.MetaHumanRigType, job.get("rig_type", "JOINTS_AND_BLEND_SHAPES"))
        mhs.request_auto_rigging(c, p)
        step("auto_rig", can_build=mhs.can_build_meta_human(c, True))
        t = unreal.MetaHumanCharacterTextureRequestParams()
        t.blocking, t.report_progress = True, False
        mhs.request_texture_sources(c, t)
        step("textures", high_res=c.get_editor_property("has_high_resolution_textures"))
    ok = mhs.can_build_meta_human(c, True)
    step("gate", can_build=ok)
    if ok:
        b = unreal.MetaHumanCharacterEditorBuildParameters()
        b.pipeline_type = getattr(unreal.MetaHumanDefaultPipelineType, job.get("pipeline", "CINEMATIC"))
        b.pipeline_quality = getattr(unreal.MetaHumanQualityLevel, job.get("quality", "HIGH"))
        b.absolute_build_path = job.get("build_path", "/Game/MetaHumans")
        b.common_folder_path = job.get("build_path", "/Game/MetaHumans") + "/Common"
        b.enable_wardrobe_item_validation = False
        mhs.build_meta_human(c, b)
        out = EAL.list_assets(f"{b.absolute_build_path}/{name}", recursive=True)
        step("build", assets=len(out), blueprint=[a for a in out if "/BP_" in a][:3])
        r["ok"] = len(out) > 0
    mhs.remove_object_to_edit(c)
    EAL.save_loaded_asset(c, False)
except Exception:
    r["error"] = traceback.format_exc()
    unreal.log_error(r["error"])
json.dump(r, open(job["result"], "w"), indent=1, default=str)
