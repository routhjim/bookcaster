"""UE side: import one exported asset into the project and write what landed where.

    UnrealEditor-Cmd <project> -run=pythonscript -script=tools/assets/ue/ue_import.py -nullrhi
    (env ASSETS_UE_JOB = job JSON: {"file": .glb, "dest": "/Game/Assets/<class>/<id>", "id", "nanite": true,
                                    "result": result JSON path})

Interchange glTF import (glTF metres -> UE centimetres; Blender -Y (asset front) -> UE +Y). The folder is cleared
first so a re-import leaves nothing stale. The static mesh is renamed SM_<id>; Nanite set per the job;
texture settings are checked (BaseColor sRGB, Normal as a normal map with the green flip glTF needs, ORM linear).
Run alone as a commandlet (import-only) or called by ue_render.py inside the render editor (one start per asset).
"""

import json
import os
import time
import traceback

import unreal


def import_asset(job):
    """Import job["file"] to job["dest"]; returns the result dict (also written to job["result"] if given)."""
    res = {"ok": False, "dest": job["dest"], "assets": {}, "problems": []}
    t0 = time.time()
    EAL = unreal.EditorAssetLibrary
    try:
        if EAL.does_directory_exist(job["dest"]):
            EAL.delete_directory(job["dest"])
        task = unreal.AssetImportTask()
        task.filename = job["file"]
        task.destination_path = job["dest"].rsplit("/", 1)[0]     # Interchange puts a <file stem>/ folder inside
        task.automated = True
        task.replace_existing = True
        task.save = True
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        res["imported"] = [str(p) for p in task.imported_object_paths]
        sm = None
        for p in EAL.list_assets(job["dest"], recursive=True):
            a = EAL.load_asset(p)
            if isinstance(a, unreal.StaticMesh):
                sm = a
        if sm is None:
            raise RuntimeError("no static mesh imported")
        want = f"SM_{job['id']}"
        if sm.get_name() != want:
            newp = sm.get_path_name().rsplit("/", 1)[0] + "/" + want
            if EAL.rename_asset(sm.get_path_name(), newp):
                sm = EAL.load_asset(newp)
        # the editor subsystem is None in a commandlet; the property route rebuilds through post_edit_change
        ns = sm.get_editor_property("nanite_settings")
        res["nanite_as_imported"] = bool(ns.enabled)
        if bool(ns.enabled) != bool(job.get("nanite", True)):
            ses = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
            ns.enabled = bool(job.get("nanite", True))
            if ses is not None:
                ses.set_nanite_settings(sm, ns, apply_changes=True)
            else:
                sm.modify()
                sm.set_editor_property("nanite_settings", ns)
                EAL.save_loaded_asset(sm, only_if_is_dirty=False)
        ns = sm.get_editor_property("nanite_settings")
        bb = sm.get_bounding_box()
        res["static_mesh"] = sm.get_path_name()
        res["bounds_cm"] = {"min": [bb.min.x, bb.min.y, bb.min.z], "max": [bb.max.x, bb.max.y, bb.max.z]}
        res["nanite"] = bool(ns.enabled)
        res["lod0_tris_reported"] = sm.get_num_triangles(0)
        res["materials"] = []
        for m in sm.static_materials:
            mi = m.material_interface
            d = {"slot": str(m.material_slot_name), "path": mi.get_path_name() if mi else None}
            if isinstance(mi, unreal.MaterialInstanceConstant):
                d["parent"] = mi.parent.get_path_name() if mi.parent else None
                d["textures"] = {str(t.parameter_info.name): t.parameter_value.get_path_name()
                                 for t in mi.texture_parameter_values if t.parameter_value}
            res["materials"].append(d)
        for p in EAL.list_assets(job["dest"], recursive=True):
            a = EAL.load_asset(p)
            if isinstance(a, unreal.Texture2D):
                n = a.get_name()
                info = {"path": a.get_path_name(), "srgb": bool(a.srgb), "compression": str(a.compression_settings),
                        "flip_green": bool(a.flip_green_channel)}
                res["assets"][n] = info
                if (n.endswith("_BaseColor") or n.endswith("_Emissive")) and not a.srgb:
                    res["problems"].append(f"{n} is not sRGB")
                if n.endswith("_ORM") and a.srgb:
                    a.srgb = False
                    EAL.save_loaded_asset(a)
                    res["problems"].append(f"{n} was sRGB; set linear")
                if n.endswith("_Normal") and (a.compression_settings != unreal.TextureCompressionSettings.TC_NORMALMAP
                                              or not a.flip_green_channel):
                    res["problems"].append(f"{n}: compression {a.compression_settings}, flip_green {a.flip_green_channel}")
        EAL.save_directory(job["dest"], only_if_is_dirty=False, recursive=True)
        res["ok"] = True
    except Exception:
        res["error"] = traceback.format_exc()
        unreal.log_error(res["error"])
    res["time_s"] = round(time.time() - t0, 2)
    if job.get("result"):
        json.dump(res, open(job["result"], "w"), indent=1, default=str)
    unreal.log_warning(f"ASSETS import ok={res['ok']} in {res['time_s']} s")
    return res


if __name__ != "assets_ue_import":           # run as the commandlet script; imported by ue_render.py otherwise
    import_asset(json.load(open(os.environ["ASSETS_UE_JOB"])))
