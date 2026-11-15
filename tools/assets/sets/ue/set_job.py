"""UE side: one set job inside the asset scheduler's resident editor (run_asset.py --ue-job; see the chain README).

The scheduler runs this file with JOB in its globals and calls run(JOB); run is a generator that the editor's tick
advances one step per frame, so every wait is counted in rendered frames. JOB:
  {"id", "scene": expanded scene JSON (layout.py), "level": "/Game/Sets/<id>/L_<id>", "build": bool, "render": bool,
   "cameras": [names] | null, "out": stills folder, "res": [1152, 640], "warm_frames", "settle_frames",
   "mannequin": skeletal mesh | null, "restore_map": level to load at the end | null}
Returns {"ok", "shots", "times_s", "build"}. The editor's own job loop guards against re-entry while a screenshot
pumps Slate.
"""

import importlib.util
import json
import os
import time

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
EAL = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def log(msg):
    unreal.log_warning(f"SETS: {msg}")


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cine(name, c, t, focal, res, temp):
    loc = unreal.Vector(*c)
    cam = act.spawn_actor_from_class(unreal.CineCameraActor, loc,
                                     unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*t)))
    cam.set_actor_label(f"cam_{name}")
    cc = cam.get_cine_camera_component()
    fb = cc.get_editor_property("filmback")
    fb.sensor_width, fb.sensor_height = 36.0, 36.0 * res[1] / res[0]
    cc.set_editor_property("filmback", fb)
    cc.set_editor_property("current_focal_length", float(focal))
    cc.set_editor_property("current_aperture", 22.0)
    fs = cc.get_editor_property("focus_settings")
    fs.focus_method = unreal.CameraFocusMethod.DISABLE
    cc.set_editor_property("focus_settings", fs)
    temp.append(cam)
    return cam


def _current_world():
    w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    return w.get_path_name() if w else ""


def import_missing(assets):
    """Library assets this project lacks (a set built outside the chain's project): import each from its exported
    glTF with the chain's own importer, to the same /Game path the manifest names."""
    done = {}
    for a in assets:
        if a["path"] in done or EAL.does_asset_exist(a["path"]):
            continue
        if not a.get("glb"):
            raise RuntimeError(f"{a['id']}: {a['path']} is not in this project and the library has no glb for it")
        cut = a["path"].find(f"/{a['id']}/")
        dest = a["path"][:cut + len(a["id"]) + 1] if cut > 0 else a["path"].rsplit("/", 2)[0]
        spec = importlib.util.spec_from_file_location("assets_ue_import",
                                                      os.path.join(HERE, "..", "..", "ue", "ue_import.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        res = m.import_asset({"file": a["glb"], "dest": dest, "id": a["id"], "nanite": True})
        if not res.get("ok") or not EAL.does_asset_exist(a["path"]):
            raise RuntimeError(f"import of {a['id']} failed: {res.get('error') or res.get('static_mesh')}")
        done[a["path"]] = res["time_s"]
        log(f"imported {a['id']} into this project in {res['time_s']} s")
    return done


def run(job):
    t0 = time.time()
    r = {"ok": False, "id": job["id"], "shots": [], "times_s": {}}
    scene = json.load(open(job["scene"]))
    res = job.get("res", [1152, 640])
    t = time.time()
    r["imported"] = import_missing(scene.get("assets", []))
    if r["imported"]:
        r["times_s"]["import_assets"] = round(time.time() - t, 2)
    if job.get("build"):
        t = time.time()
        r["build"] = _mod("set_build").build(job["id"], job["level"], scene)
        r["times_s"]["build"] = round(time.time() - t, 2)
    elif not _current_world().startswith(job["level"]):
        t = time.time()
        les.load_level(job["level"])
        r["times_s"]["load_level"] = round(time.time() - t, 2)
    yield
    temp = []
    old_cv = {}
    try:
        for k, v in (scene["env"].get("post") or {}).get("cvars", {}).items():   # restored in finally
            try:
                old_cv[k] = unreal.SystemLibrary.get_console_variable_float_value(k)
            except Exception:
                old_cv[k] = None
            unreal.SystemLibrary.execute_console_command(None, f"{k} {v}")
        if job.get("render"):
            les.editor_set_game_view(True)
            if job.get("mannequin") and EAL.does_asset_exist(job["mannequin"]):
                mq = EAL.load_asset(job["mannequin"])
                for k, f in enumerate(scene.get("figures", [])):
                    a = act.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(*f["c"]),
                                                   unreal.Rotator(0, 0, f["yaw"]))
                    a.set_actor_label(f"ref_figure_1p8m_{k}")
                    a.skeletal_mesh_component.set_skinned_asset_and_update(mq)
                    temp.append(a)
            want = job.get("cameras")
            shots = [(c["name"], _cine(c["name"], c["c"], c["t"], c["focal"], res, temp))
                     for c in scene["cameras"] if not want or c["name"] in want]
            os.makedirs(job["out"], exist_ok=True)
            for _ in range(job.get("warm_frames", 240)):
                yield
            t = time.time()
            unreal.AutomationLibrary.finish_loading_before_screenshot()
            r["times_s"]["finish_loading"] = round(time.time() - t, 2)
            t_r = time.time()
            for name, cam in shots:
                les.pilot_level_actor(cam)
                for _ in range(job.get("settle_frames", 180)):
                    yield
                path = os.path.join(job["out"], f"{job['id']}_{name}.png")
                if os.path.exists(path):
                    os.remove(path)
                unreal.AutomationLibrary.take_high_res_screenshot(res[0], res[1], path, camera=cam,
                                                                 force_game_view=True)
                t_s = time.time()
                while not (os.path.exists(path) and os.path.getsize(path) > 0):
                    if time.time() - t_s > 60:
                        raise RuntimeError(f"screenshot not written: {path}")
                    yield
                les.eject_pilot_level_actor()
                r["shots"].append({"name": name, "path": path, "t_s": round(time.time() - t0, 2)})
                log(f"{job['id']}: shot {name}")
            r["times_s"]["stills"] = round(time.time() - t_r, 2)
        r["ok"] = True
    finally:
        for k, v in old_cv.items():
            if v is not None:
                unreal.SystemLibrary.execute_console_command(None, f"{k} {v:g}")
        try:
            les.eject_pilot_level_actor()
        except Exception:
            pass
        for a in temp:
            try:
                act.destroy_actor(a)
            except Exception:
                pass
        if job.get("render"):
            les.save_current_level()
        if job.get("restore_map") and not _current_world().startswith(job["restore_map"]):
            les.load_level(job["restore_map"])
    r["times_s"]["total"] = round(time.time() - t0, 2)
    return r
