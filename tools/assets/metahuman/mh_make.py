"""UE side, no GPU: make a MetaHuman Character asset from a small spec using only local data (no Epic cloud).

    python tools/assets/metahuman/mh_run.py cmdlet tools/assets/metahuman/mh_make.py --job job.json
    job = {"result": path, "folder": "/Game/MH",
           "characters": [{"name": "MH_A",
                           "base": "default" | "<preset name>",        (preset = copy of a shipped preset)
                           "rig": "archetype" | null,                  (import the shipped archetype face DNA as a
                                                                         whole rig: the face then animates, but its
                                                                         shape becomes the archetype face)
                           "skin_from": "<preset name>" | null,        (copy skin, eyes, makeup settings)
                           "skin": {"u": 0.5, "v": 0.5, "roughness": 1.0, "face_texture_index": 0, ...} | null,
                           "grooms": {"Hair": "WI_Hair_S_Messy", "Eyebrows": "WI_Eyebrows_M_Dense", ...},
                           "outfit": true}]}                            (the shipped default garment)

Every step is local: face/body state, synthesized 1k skin textures (the shipped texture-synthesis model), grooms and
the default garment from the plugin's Optional content, and the archetype rig. What it cannot make offline is a rig
fitted to a non-archetype face (request_auto_rigging) or the high-resolution textures (request_texture_sources),
so build_meta_human (assembly into a Blueprint) stays closed; render through spawn_meta_human_actor instead.
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
PRESETS = "/MetaHumanCharacter/Optional/Presets"
GROOMS = "/MetaHumanCharacter/Optional/Grooms/Bindings"
res = {"characters": {}, "ok": False}


def make(spec):
    name = spec["name"]
    path = f"{FOLDER}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    base = spec.get("base", "default")
    if base == "default":
        c = at.create_asset(asset_name=name, package_path=FOLDER, asset_class=unreal.MetaHumanCharacter,
                            factory=unreal.new_object(type=unreal.MetaHumanCharacterFactoryNew))
    else:
        c = EAL.duplicate_asset(f"{PRESETS}/{base}", path)
    if c is None:
        raise RuntimeError(f"could not create {path} from {base}")
    r = {"path": c.get_path_name()}
    col = c.get_editor_property("internal_collection")
    inst = col.get_editor_property("default_instance")
    for slot, wi in (spec.get("grooms") or {}).items():
        d = {"Beard": "Beards", "Mustache": "Mustaches"}.get(slot, slot)
        item = EAL.load_asset(f"{GROOMS}/{d}/{wi}")
        if item is None:
            raise RuntimeError(f"no wardrobe item {GROOMS}/{slot}/{wi}")
        key = col.try_add_item_from_wardrobe_item(slot, item)
        inst.set_single_slot_selection(slot, key)
    if spec.get("outfit"):
        key = col.try_add_item_from_wardrobe_item("Outfits", EAL.load_asset(
            "/MetaHumanCharacter/Optional/Clothing/WI_DefaultGarment"))
        inst.set_single_slot_selection("Outfits", key)
    if not mhs.try_add_object_to_edit(c):
        raise RuntimeError("try_add_object_to_edit failed")
    try:
        if spec.get("rig") == "archetype":
            p = unreal.ImportFromDNAParams()
            p.import_whole_rig = True
            dna = unreal.Paths.combine([unreal.PluginBlueprintLibrary.get_plugin_content_dir("MetaHumanCoreTech"),
                                        "ArchetypeDNA/SKM_Face.dna"])
            r["rig"] = str(mhs.import_from_face_dna(c, dna, p))
        src = EAL.load_asset(f"{PRESETS}/{spec['skin_from']}") if spec.get("skin_from") else None
        skin = src.get_editor_property("skin_settings") if src else c.get_editor_property("skin_settings")
        for k, v in (spec.get("skin") or {}).items():
            skin.skin.set_editor_property(k, v)
        c.set_editor_property("preview_material_type", unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE)
        mhs.commit_skin_settings(c, skin)
        if src:
            mhs.commit_eyes_settings(c, src.get_editor_property("eyes_settings"))
            mhs.commit_makeup_settings(c, src.get_editor_property("makeup_settings"))
        mhs.assemble_for_preview(c)
        r["can_build"] = mhs.can_build_meta_human(c, True)
        r["slots"] = sorted({str(col.get_item_slot_name(k)) for k in col.get_all_item_keys()})
    finally:
        mhs.remove_object_to_edit(c)
    r["saved"] = EAL.save_loaded_asset(c, False)
    return r


unreal.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous([PRESETS, GROOMS, "/MetaHumanCharacter/Optional/Clothing"], True)
for spec in job["characters"]:
    try:
        res["characters"][spec["name"]] = make(spec)
    except Exception:
        res["characters"][spec["name"]] = {"error": traceback.format_exc()}
    unreal.log_warning(f"MH make {spec['name']}: {json.dumps(res['characters'][spec['name']], default=str)[:500]}")
res["ok"] = all("error" not in v for v in res["characters"].values())
json.dump(res, open(job["result"], "w"), indent=1, default=str)
