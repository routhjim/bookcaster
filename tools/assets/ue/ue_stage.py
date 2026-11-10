"""UE side, no GPU: build the standard validation stage level once (run_asset.py calls it when the map is missing).

    UnrealEditor-Cmd <project> -run=pythonscript -script=tools/assets/ue/ue_stage.py -nullrhi
    (env ASSETS_UE_JOB = {"map": "/Game/AssetStage/L_AssetStage", "result": path})

Stage: 200 m floor (18 % grey, rough), physical sky (SkyAtmosphere + sun + real-time SkyLight), light height fog,
a key sun from front-left above (asset front = UE +Y), an unbound post-process volume. Two neutral materials are
made for the floor and for the size-reference box. Asset, reference and cameras are spawned per render, not saved.
"""

import json
import os
import traceback

import unreal

job = json.load(open(os.environ["ASSETS_UE_JOB"]))
MAP = job["map"]
FOLDER = MAP.rsplit("/", 1)[0]
res = {"ok": False, "map": MAP}
EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary
at = unreal.AssetToolsHelpers.get_asset_tools()


def flat_material(name, rgb, rough):
    path = f"{FOLDER}/{name}"
    if EAL.does_asset_exist(path):
        return EAL.load_asset(path)
    m = at.create_asset(name, FOLDER, unreal.Material, unreal.MaterialFactoryNew())
    c = MEL.create_material_expression(m, unreal.MaterialExpressionConstant3Vector, -400, 0)
    c.constant = unreal.LinearColor(*rgb, 1.0)
    MEL.connect_material_property(c, "", unreal.MaterialProperty.MP_BASE_COLOR)
    r = MEL.create_material_expression(m, unreal.MaterialExpressionConstant, -400, 200)
    r.r = rough
    MEL.connect_material_property(r, "", unreal.MaterialProperty.MP_ROUGHNESS)
    MEL.recompile_material(m)
    EAL.save_loaded_asset(m)
    return m


try:
    floor_mat = flat_material("M_StageFloor", (0.18, 0.18, 0.18), 0.85)
    flat_material("M_StageRef", (0.45, 0.45, 0.45), 0.6)
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not les.new_level(MAP):
        raise RuntimeError(f"new_level failed for {MAP}")
    act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    def spawn(cls, loc=(0, 0, 0), rot=(0, 0, 0), label=None):
        a = act.spawn_actor_from_class(cls, unreal.Vector(*loc), unreal.Rotator(roll=rot[0], pitch=rot[1], yaw=rot[2]))
        if label:
            a.set_actor_label(label)
        return a

    fl = spawn(unreal.StaticMeshActor, label="StageFloor")
    fl.static_mesh_component.set_static_mesh(EAL.load_asset("/Engine/BasicShapes/Plane"))
    fl.static_mesh_component.set_material(0, floor_mat)
    fl.set_actor_scale3d(unreal.Vector(200, 200, 1))
    sun = spawn(unreal.DirectionalLight, (0, 0, 500), (0, -40, -55), "Sun")
    lc = sun.get_component_by_class(unreal.DirectionalLightComponent)
    lc.set_editor_property("atmosphere_sun_light", True)
    lc.set_editor_property("intensity", 8.0)
    lc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    spawn(unreal.SkyAtmosphere, label="Sky")
    sky = spawn(unreal.SkyLight, (0, 0, 300), label="SkyLight")
    sc = sky.get_component_by_class(unreal.SkyLightComponent)
    sc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sc.set_editor_property("real_time_capture", True)
    fog = spawn(unreal.ExponentialHeightFog, label="Fog")
    fog.get_component_by_class(unreal.ExponentialHeightFogComponent).set_editor_property("fog_density", 0.005)
    ppv = spawn(unreal.PostProcessVolume, label="PostProcess")
    ppv.set_editor_property("unbound", True)
    # fixed exposure: auto exposure brightened dark metal when it filled the frame (close-ups read as silver)
    pp = ppv.get_editor_property("settings")
    pp.set_editor_property("override_auto_exposure_method", True)
    pp.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    pp.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
    pp.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
    pp.set_editor_property("override_auto_exposure_bias", True)
    pp.set_editor_property("auto_exposure_bias", float(job.get("exposure_ev", 0.0)))
    ppv.set_editor_property("settings", pp)
    les.save_current_level()
    res["ok"] = True
except Exception:
    res["error"] = traceback.format_exc()
    unreal.log_error(res["error"])
json.dump(res, open(job["result"], "w"), indent=1)
unreal.log_warning(f"ASSETS stage ok={res['ok']}")
