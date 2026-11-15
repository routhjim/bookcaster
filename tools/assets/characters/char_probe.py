"""UE side, no GPU: what a MetaHuman Character exposes offline for likeness work, and whether its body exports.

    python tools/assets/metahuman/mh_run.py cmdlet tools/assets/characters/char_probe.py --job probe.json
    job = {"result": path, "presets": ["<preset>", ...], "export": {"preset": "<preset>", "folder": "/Game/Char/Geo",
           "fbx_dir": "<disk folder>"}}

Per preset: skin / eyes / makeup / head-model settings as dicts, face-model coefficient count, landmark count, body
constraints (name, value, range), grooms in its collection. Export test: MetaHumanCharacterExportBlueprintLibrary.
export_geometry into project assets, then an AssetExportTask of each skeletal mesh to FBX on disk.
"""

import json
import os
import time
import traceback

import unreal

job = json.load(open(os.environ["MH_JOB"]))
EAL = unreal.EditorAssetLibrary
mhs = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
PRESETS = "/MetaHumanCharacter/Optional/Presets"
GROOMS = "/MetaHumanCharacter/Optional/Grooms/Bindings"
res = {"presets": {}, "grooms": {}, "export": {}}


def sdict(s):
    try:
        return {k: str(v) for k, v in s.to_dict().items()}
    except Exception as e:
        return f"ERR {e}"


ar = unreal.AssetRegistryHelpers.get_asset_registry()
ar.scan_paths_synchronous([PRESETS, GROOMS, "/MetaHumanCharacter/Optional/Clothing"], True)
for d in EAL.list_assets(GROOMS, recursive=True, include_folder=False):
    slot = d.split("/")[-2]
    res["grooms"].setdefault(slot, []).append(d.split("/")[-1].split(".")[0])

for name in job.get("presets", []):
    r = {}
    try:
        c = EAL.load_asset(f"{PRESETS}/{name}")
        r["skin"] = sdict(c.get_editor_property("skin_settings").skin)
        r["eyes_iris"] = sdict(c.get_editor_property("eyes_settings").eye_left.iris)
        r["face_eval"] = sdict(c.get_editor_property("face_evaluation_settings"))
        r["fixed_body_type"] = str(c.get_editor_property("fixed_body_type"))
        col = c.get_editor_property("internal_collection")
        r["items"] = sorted(f"{col.get_item_slot_name(k)}:{col.get_item_display_name(k)}" for k in col.get_all_item_keys())
        if mhs.try_add_object_to_edit(c):
            try:
                r["n_coeff"] = len(mhs.get_face_model_coefficients(c))
                r["n_landmarks"] = len(mhs.get_face_landmarks(c))
                r["body"] = [(str(b.name), round(b.target_measurement, 1), round(b.min_measurement, 1),
                              round(b.max_measurement, 1), b.is_active) for b in mhs.get_body_constraints(c)]
            finally:
                mhs.remove_object_to_edit(c)
    except Exception:
        r["error"] = traceback.format_exc()
    res["presets"][name] = r

json.dump(res, open(job["result"], "w"), indent=1, default=str)
ex = job.get("export")
if ex:
    t0 = time.time()
    e = res["export"]
    try:
        folder = ex.get("folder", "/Game/Char/Geo")
        dst = f"{folder}/MHC_Probe"
        if EAL.does_asset_exist(dst):
            EAL.delete_asset(dst)
        c = EAL.duplicate_asset(f"{PRESETS}/{ex['preset']}", dst)
        if not mhs.try_add_object_to_edit(c):
            raise RuntimeError("try_add_object_to_edit failed")
        p = unreal.MetaHumanGeometryExportParams()
        p.project_path = folder
        p.overwrite_existing_assets = True
        p.body_skeletal_mesh = True
        p.head_skeletal_mesh = True
        p.full_body_skeletal_mesh = True
        try:
            unreal.MetaHumanCharacterExportBlueprintLibrary.export_geometry(c, p)
            e["export_geometry"] = "called"
        except Exception:
            e["export_geometry"] = traceback.format_exc()
        mhs.remove_object_to_edit(c)
        made = EAL.list_assets(folder, recursive=True, include_folder=False)
        e["assets"] = made
        os.makedirs(ex["fbx_dir"], exist_ok=True)
        e["fbx"] = {}
        for a in made:
            o = EAL.load_asset(a)
            if not isinstance(o, unreal.SkeletalMesh):
                continue
            out = os.path.join(ex["fbx_dir"], o.get_name() + ".fbx")
            t = unreal.AssetExportTask()
            t.object, t.filename, t.automated, t.replace_identical, t.prompt = o, out, True, True, False
            t.exporter = unreal.SkeletalMeshExporterFBX()
            opt = unreal.FbxExportOption()
            opt.collision = False
            opt.level_of_detail = False
            opt.export_morph_targets = False
            t.options = opt
            ok = unreal.Exporter.run_asset_export_task(t)
            e["fbx"][a] = {"ok": ok, "file": out, "bytes": os.path.getsize(out) if os.path.exists(out) else 0,
                           "skeleton": o.skeleton.get_path_name() if o.skeleton else None,
                           "materials": [m.material_interface.get_path_name() if m.material_interface else None
                                         for m in o.materials]}
    except Exception:
        e["error"] = traceback.format_exc()
    e["time_s"] = round(time.time() - t0, 1)

json.dump(res, open(job["result"], "w"), indent=1, default=str)
unreal.log_warning("char probe done")
