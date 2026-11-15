"""UE side (run_asset.py --ue-job, inside the resident editor): import a rigged creature FBX (skeletal mesh + its
pose action), give it the static asset's material, and shoot each pose of the action side by side on the stage.

Job {"fbx": path, "dest": "/Game/Assets/Creatures/<id>_rig", "material": material path or null,
     "poses": n (frames of the action: 0 = rest), "fps": 24, "spacing_m": 14.0, "out": folder, "id": name,
     "res": [1152, 640], "shots": [{"name", "az", "el", "focal", "fill"}] (around the row of poses),
     "per_pose": {"az", "el", "focal"} (also one shot of every pose alone), "warm_frames": 90, "settle_frames": 30}
run(job) is a generator (one step per editor tick) and returns {"ok", "mesh", "anim", "shots": [...]}.
"""
import math
import os
import time

import unreal

EAL = unreal.EditorAssetLibrary
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
at = unreal.AssetToolsHelpers.get_asset_tools()


def import_fbx(fbx, dest):
    if EAL.does_directory_exist(dest):
        EAL.delete_directory(dest)
    t = unreal.AssetImportTask()
    t.filename, t.destination_path = fbx, dest
    t.automated, t.replace_existing, t.save = True, True, True
    ui = unreal.FbxImportUI()
    ui.import_mesh = True
    ui.import_as_skeletal = True
    ui.import_animations = True
    ui.import_materials = False
    ui.import_textures = False
    ui.mesh_type_to_import = unreal.FBXImportType.FBXIT_SKELETAL_MESH
    ui.skeletal_mesh_import_data.import_morph_targets = False
    ui.anim_sequence_import_data.animation_length = unreal.FBXAnimationLengthImportType.FBXALIT_EXPORTED_TIME
    t.options = ui
    at.import_asset_tasks([t])
    sk, an = None, None
    for a in EAL.list_assets(dest, recursive=True, include_folder=False):
        o = EAL.load_asset(a.split(".")[0])
        if isinstance(o, unreal.SkeletalMesh):
            sk = o
        elif isinstance(o, unreal.AnimSequence):
            an = o
    if sk is None:
        raise RuntimeError(f"no skeletal mesh from {fbx}")
    return sk, an


def pose(comp, anim, t):
    comp.set_update_animation_in_editor(True)
    comp.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    try:
        comp.override_animation_data(anim, False, False, float(t), 0.0)
    except Exception:
        pass
    comp.set_animation(anim)
    comp.play(False)
    comp.set_position(float(t), False)
    comp.set_play_rate(0.0)
    comp.stop()
    comp.set_position(float(t), False)
    for fn in ("tick_animation", "refresh_bone_transforms"):
        try:
            getattr(comp, fn)(*((0.0, False) if fn == "tick_animation" else ()))
        except Exception:
            pass


def camera(res, loc, tgt, focal):
    cam = act.spawn_actor_from_class(unreal.CineCameraActor, loc, unreal.MathLibrary.find_look_at_rotation(loc, tgt))
    cc = cam.get_cine_camera_component()
    fb = cc.get_editor_property("filmback")
    fb.sensor_width, fb.sensor_height = 36.0, 36.0 * res[1] / res[0]
    cc.set_editor_property("filmback", fb)
    cc.set_editor_property("current_focal_length", float(focal))
    fs = cc.get_editor_property("focus_settings")
    fs.focus_method = unreal.CameraFocusMethod.DISABLE
    cc.set_editor_property("focus_settings", fs)
    return cam


def run(job):
    t0 = time.time()
    res = job.get("res", [1152, 640])
    out = job["out"]
    os.makedirs(out, exist_ok=True)
    r = {"ok": False, "shots": []}
    sk, anim = import_fbx(job["fbx"], job["dest"])
    r["mesh"], r["anim"] = sk.get_path_name(), anim.get_path_name() if anim else None
    if anim is None:
        raise RuntimeError("no animation imported")
    if job.get("material") and EAL.does_asset_exist(job["material"]):
        mat = EAL.load_asset(job["material"])
        mats = sk.get_editor_property("materials")
        for m in mats:
            m.set_editor_property("material_interface", mat)
        sk.set_editor_property("materials", mats)
    n = int(job["poses"])
    fps = float(job.get("fps", 24))
    sp = float(job.get("spacing_m", 14.0)) * 100
    actors = []
    xs = [(i - (n - 1) / 2) * sp for i in range(n)]
    for i in range(n):
        a = act.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(xs[i], 0, 0), unreal.Rotator(0, 0, 0))
        a.set_actor_label(f"pose_{i}")
        c = a.skeletal_mesh_component
        c.set_skinned_asset_and_update(sk)
        actors.append(a)
    bb = sk.get_bounds()
    ext = bb.box_extent
    H = ext.z * 2
    cams = []
    width = sp * n
    for s in job.get("shots", [{"name": "row", "az": 20, "el": 12, "focal": 35}]):
        a_, e_ = math.radians(s["az"]), math.radians(s["el"])
        tgt = unreal.Vector(0, 0, H * 0.45)
        hf = 2 * math.atan(18.0 / s["focal"])
        dist = width / 2 / math.tan(hf / 2) * float(s.get("fill", 1.05))
        loc = tgt + unreal.Vector(math.sin(a_) * math.cos(e_), math.cos(a_) * math.cos(e_), math.sin(e_)) * dist
        cams.append((s["name"], camera(res, loc, tgt, s["focal"])))
    pp = job.get("per_pose")
    if pp:
        for i in range(n):
            a_, e_ = math.radians(pp["az"]), math.radians(pp["el"])
            tgt = unreal.Vector(xs[i], 0, H * 0.45)
            hf = 2 * math.atan(18.0 / pp["focal"])
            dist = max(ext.x, ext.y, ext.z) * 2 / 2 / math.tan(hf / 2) * float(pp.get("fill", 1.1))
            loc = tgt + unreal.Vector(math.sin(a_) * math.cos(e_), math.cos(a_) * math.cos(e_), math.sin(e_)) * dist
            cams.append((f"pose{i}", camera(res, loc, tgt, pp["focal"])))
    # a paused single-node pose shows from the second application: apply, wait, apply again
    for k in range(2):
        for i, a in enumerate(actors):
            pose(a.skeletal_mesh_component, anim, i / fps)
        for _ in range(10):
            yield
    for _ in range(int(job.get("warm_frames", 90))):
        yield
    unreal.AutomationLibrary.finish_loading_before_screenshot()
    for name, cam in cams:
        for i, a in enumerate(actors):
            pose(a.skeletal_mesh_component, anim, i / fps)
        les.pilot_level_actor(cam)
        for _ in range(int(job.get("settle_frames", 30))):
            yield
        path = os.path.join(out, f"{job.get('id', 'rig')}_{name}.png")
        if os.path.exists(path):
            os.remove(path)
        unreal.AutomationLibrary.take_high_res_screenshot(res[0], res[1], path, camera=cam, force_game_view=True)
        t1 = time.time()
        while not (os.path.exists(path) and os.path.getsize(path) > 0):
            if time.time() - t1 > 60:
                raise RuntimeError(f"screenshot not written: {path}")
            yield
        les.eject_pilot_level_actor()
        r["shots"].append(path)
    for a in actors + [c for _, c in cams]:
        act.destroy_actor(a)
    r["ok"] = True
    r["seconds"] = round(time.time() - t0, 1)
    return r
