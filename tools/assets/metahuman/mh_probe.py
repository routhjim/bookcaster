"""UE side, no GPU: make MetaHuman Character assets with only local data and record how far each gets.

    UnrealEditor-Cmd <project> -run=pythonscript -script=tools/assets/metahuman/mh_probe.py -nullrhi
    (env MH_JOB = {"result": path, "folder": "/Game/MH", "presets": ["<preset name>", ...]})

Characters made (all saved under job["folder"]):
  MH_Default      new asset from the factory, defaults only
  MH_DefaultRig   new asset + the archetype face DNA imported as a whole rig (local, no cloud)
  MH_<preset>     a copy of a preset shipped in the MetaHumanCharacter plugin
For each: can_build_meta_human (the build gate) and the reason it gives, plus the preview collection's slots and
items. The cloud calls (request_auto_rigging, request_texture_sources) are NOT made here: unauthenticated they
start Epic's account-portal sign-in (see README).
"""

import json
import os
import traceback

import unreal

job = json.load(open(os.environ["MH_JOB"]))
FOLDER = job.get("folder", "/Game/MH")
EAL = unreal.EditorAssetLibrary
mhs = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
at = unreal.AssetToolsHelpers.get_asset_tools()
res = {"characters": {}, "ok": False}


def new_character(name):
    path = f"{FOLDER}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    return at.create_asset(asset_name=name, package_path=FOLDER, asset_class=unreal.MetaHumanCharacter,
                           factory=unreal.new_object(type=unreal.MetaHumanCharacterFactoryNew))


def describe(c):
    d = {"has_high_resolution_textures": c.get_editor_property("has_high_resolution_textures"),
         "preview_material_type": str(c.get_editor_property("preview_material_type")),
         "synth_face_textures": [str(k) for k in c.get_editor_property("synthesized_face_textures_info").keys()],
         "high_res_body_textures": [str(k) for k in c.get_editor_property("high_res_body_textures_info").keys()],
         "can_build": mhs.can_build_meta_human(c, True)}
    try:
        col = mhs.get_preview_collection(c)
        d["slots"] = [str(s) for s in col.get_slot_names()]
        d["items"] = {str(col.get_item_slot_name(k)): [] for k in col.get_all_item_keys()}
        for k in col.get_all_item_keys():
            d["items"][str(col.get_item_slot_name(k))].append(str(col.get_item_display_name(k)))
    except Exception as e:
        d["collection_error"] = str(e)
    return d


def probe(name, make):
    r = {}
    try:
        c = make()
        if not mhs.try_add_object_to_edit(c):
            raise RuntimeError("try_add_object_to_edit failed")
        try:
            extra = job.get("_after", {}).get(name)
            r.update(describe(c))
            if name == "MH_DefaultRig":
                p = unreal.ImportFromDNAParams()
                p.import_whole_rig = True
                dna = unreal.Paths.combine([unreal.PluginBlueprintLibrary.get_plugin_content_dir("MetaHumanCoreTech"),
                                            "ArchetypeDNA/SKM_Face.dna"])
                r["import_face_dna"] = str(mhs.import_from_face_dna(c, dna, p))
                r["after_rig"] = describe(c)
        finally:
            mhs.remove_object_to_edit(c)
        r["saved"] = EAL.save_loaded_asset(c, False)
        r["path"] = c.get_path_name()
    except Exception:
        r["error"] = traceback.format_exc()
    res["characters"][name] = r
    unreal.log_warning(f"MH probe {name}: {json.dumps(r, default=str)[:600]}")


try:
    unreal.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous(
        ["/MetaHumanCharacter/Optional/Presets"], True)
    probe("MH_Default", lambda: new_character("MH_Default"))
    probe("MH_DefaultRig", lambda: new_character("MH_DefaultRig"))
    for pre in job.get("presets", []):
        def dup(pre=pre):
            dst = f"{FOLDER}/MH_{pre}"
            if EAL.does_asset_exist(dst):
                EAL.delete_asset(dst)
            return EAL.duplicate_asset(f"/MetaHumanCharacter/Optional/Presets/{pre}", dst)
        probe(f"MH_{pre}", dup)
    res["ok"] = True
except Exception:
    res["error"] = traceback.format_exc()
json.dump(res, open(job["result"], "w"), indent=1, default=str)
unreal.log_warning(f"MH probe done ok={res['ok']}")
