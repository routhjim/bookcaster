"""Editor side (run_asset.py --ue-job): import a skinned figure FBX (bl_combo.py mode statue) as a skeletal mesh on an
existing skeleton (the MetaHuman body's), so the pose AnimSequences made for that body play on it directly, and give
every slot one material.

Job {"fbx", "dest": "/Game/...", "skeleton_of": a SkeletalMesh whose skeleton to use, "material": a material path
     (e.g. the one the character tools' import made from the same textures) | null, "result": path}
run(job) returns {"ok", "mesh", "skeleton", "slots"}.
"""
import json
import traceback

import unreal

EAL = unreal.EditorAssetLibrary
AT = unreal.AssetToolsHelpers.get_asset_tools()


def run(job):
    r = {"ok": False}
    try:
        src = EAL.load_asset(job["skeleton_of"])
        skel = src.get_editor_property("skeleton")
        if EAL.does_directory_exist(job["dest"]):
            EAL.delete_directory(job["dest"])
        t = unreal.AssetImportTask()
        t.filename, t.destination_path = job["fbx"], job["dest"]
        t.automated, t.replace_existing, t.save = True, True, True
        ui = unreal.FbxImportUI()
        ui.import_mesh = True
        ui.import_as_skeletal = True
        ui.import_animations = False
        ui.import_materials = False
        ui.import_textures = False
        ui.mesh_type_to_import = unreal.FBXImportType.FBXIT_SKELETAL_MESH
        ui.skeleton = skel
        t.options = ui
        AT.import_asset_tasks([t])
        mesh = None
        for a in EAL.list_assets(job["dest"], recursive=True, include_folder=False):
            o = EAL.load_asset(a.split(".")[0])
            if isinstance(o, unreal.SkeletalMesh):
                mesh = o
        if mesh is None:
            raise RuntimeError(f"no skeletal mesh from {job['fbx']}")
        if job.get("material"):
            mat = EAL.load_asset(job["material"])
            mats = []
            for sm in mesh.get_editor_property("materials"):
                s2 = unreal.SkeletalMaterial()
                s2.set_editor_property("material_interface", mat)
                s2.set_editor_property("material_slot_name", sm.get_editor_property("material_slot_name"))
                mats.append(s2)
            mesh.set_editor_property("materials", mats)
            EAL.save_loaded_asset(mesh, False)
        r.update(ok=True, mesh=mesh.get_path_name(), skeleton=skel.get_path_name(),
                 slots=[str(m.get_editor_property("material_slot_name")) for m in mesh.get_editor_property("materials")])
    except Exception:
        r["error"] = traceback.format_exc()
    json.dump(r, open(job["result"], "w"), indent=1)
    if r.get("error"):
        raise RuntimeError(r["error"])
    return r
