"""UE side, GPU: build a set level from an expanded scene (layout.py output) and render its validation stills.

Started by the asset chain's start-up shim (Content/Python/init_unreal.py runs the job JSON's "script") inside a
full editor (-RenderOffscreen) that build_set.py launched on one XTX. After the first job it keeps serving
<serve_dir>/*.job.json until <serve_dir>/quit appears or serve_idle_s passes without work, so a re-render after a
spec change costs no editor start. set_build.py / set_materials.py are reloaded for every job: edits to them take
effect on the next job without restarting the editor.

Job: {"script", "id", "scene": path (expanded scene JSON), "level": "/Game/Sets/<id>/L_<id>", "build": bool,
      "render": bool, "cameras": [names] | null (all), "out": folder, "result": path, "res": [1152, 640],
      "warm_frames", "settle_frames", "mannequin": skeletal mesh path | null, "serve_dir", "serve_idle_s"}
"""

import glob
import importlib.util
import json
import os
import time
import traceback

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
EAL = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
FIRST = json.load(open(os.environ["ASSETS_UE_JOB"]))
S = {"t0": time.time(), "busy": False, "n": 0, "job": None, "first": True, "idle_since": time.time()}


def log(msg):
    unreal.log_warning(f"SETS: {msg}")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Job:
    def __init__(self, job):
        self.j = job
        self.res = job.get("res", [1152, 640])
        self.r = {"ok": False, "id": job["id"], "shots": [], "times_s": {}}
        self.t0 = time.time()
        self.phase, self.t_phase, self.f_phase = "start", time.time(), S["n"]
        self.temp, self.shots, self.i = [], [], 0
        os.makedirs(job["out"], exist_ok=True)

    def waited(self, frames):
        return S["n"] - self.f_phase >= frames

    def goto(self, ph):
        log(f"{self.j['id']}: {ph} at {time.time() - self.t0:.1f} s")
        self.phase, self.t_phase, self.f_phase = ph, time.time(), S["n"]

    def _cine(self, name, c, t, focal):
        loc = unreal.Vector(*c)
        cam = act.spawn_actor_from_class(unreal.CineCameraActor, loc,
                                         unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*t)))
        cam.set_actor_label(f"cam_{name}")
        cam.set_folder_path("Validation")
        cc = cam.get_cine_camera_component()
        fb = cc.get_editor_property("filmback")
        fb.sensor_width, fb.sensor_height = 36.0, 36.0 * self.res[1] / self.res[0]
        cc.set_editor_property("filmback", fb)
        cc.set_editor_property("current_focal_length", float(focal))
        cc.set_editor_property("current_aperture", 22.0)
        fs = cc.get_editor_property("focus_settings")
        fs.focus_method = unreal.CameraFocusMethod.DISABLE
        cc.set_editor_property("focus_settings", fs)
        self.temp.append(cam)
        return cam

    def setup(self):
        j = self.j
        scene = json.load(open(j["scene"]))
        if j.get("build"):
            t = time.time()
            sb = load_module("set_build")
            self.r["build"] = sb.build(j["id"], j["level"], scene)
            self.r["times_s"]["build"] = round(time.time() - t, 2)
        else:
            w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
            if w is None or not w.get_path_name().startswith(j["level"]):
                t = time.time()
                les.load_level(j["level"])
                self.r["times_s"]["load_level"] = round(time.time() - t, 2)
        if not j.get("render"):
            return
        les.editor_set_game_view(True)
        if j.get("mannequin") and EAL.does_asset_exist(j["mannequin"]):
            mq = EAL.load_asset(j["mannequin"])
            for k, f in enumerate(scene.get("figures", [])):
                a = act.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(*f["c"]),
                                               unreal.Rotator(0, 0, f["yaw"]))
                a.set_actor_label(f"ref_figure_1p8m_{k}")
                a.set_folder_path("Validation")
                a.skeletal_mesh_component.set_skinned_asset_and_update(mq)
                self.temp.append(a)
        want = j.get("cameras")
        for c in scene["cameras"]:
            if want and c["name"] not in want:
                continue
            self.shots.append((c["name"], self._cine(c["name"], c["c"], c["t"], c["focal"])))

    def teardown(self):
        try:
            les.eject_pilot_level_actor()
        except Exception:
            pass
        for a in self.temp:
            try:
                act.destroy_actor(a)
            except Exception:
                pass
        if self.j.get("render"):
            les.save_current_level()            # the level is clean again; saving clears its dirty flag

    def step(self):
        j, ph = self.j, self.phase
        if ph == "start":
            self.setup()
            self.r["times_s"]["setup"] = round(time.time() - self.t0, 2)
            if not j.get("render") or not self.shots:
                self.r["ok"] = True
                return True
            self.goto("warm")
        elif ph == "warm" and self.waited(j.get("warm_frames", 60)):
            t = time.time()
            unreal.AutomationLibrary.finish_loading_before_screenshot()
            self.r["times_s"]["finish_loading"] = round(time.time() - t, 2)
            self.goto("aim")
        elif ph == "aim":
            les.pilot_level_actor(self.shots[self.i][1])
            self.goto("settle")
        elif ph == "settle" and self.waited(j.get("settle_frames", 45)):
            name, cam = self.shots[self.i]
            path = os.path.join(j["out"], f"{j['id']}_{name}.png")
            if os.path.exists(path):
                os.remove(path)
            self.pending = path
            self.goto("shoot")
            unreal.AutomationLibrary.take_high_res_screenshot(self.res[0], self.res[1], path, camera=cam,
                                                             force_game_view=True)
        elif ph == "shoot":
            if os.path.exists(self.pending) and os.path.getsize(self.pending) > 0:
                self.r["shots"].append({"name": self.shots[self.i][0], "path": self.pending,
                                        "t_s": round(time.time() - self.t0, 2)})
                les.eject_pilot_level_actor()
                self.i += 1
                if self.i >= len(self.shots):
                    self.r["ok"] = True
                    return True
                self.goto("aim")
            elif time.time() - self.t_phase > 60:
                raise RuntimeError(f"screenshot not written: {self.pending}")
        return False

    def finish(self):
        try:
            self.teardown()
        except Exception:
            self.r.setdefault("teardown_error", traceback.format_exc())
        self.r["elapsed_s"] = round(time.time() - self.t0, 2)
        self.r["editor_uptime_s"] = round(time.time() - S["t0"], 2)
        cv = {}
        for k in ("r.RayTracing", "r.Lumen.HardwareRayTracing", "r.DynamicGlobalIlluminationMethod"):
            try:
                cv[k] = unreal.SystemLibrary.get_console_variable_int_value(k)
            except Exception:
                cv[k] = None
        self.r["cvars"] = cv
        tmp = self.j["result"] + ".tmp"
        json.dump(self.r, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, self.j["result"])
        log(f"{self.j['id']}: done ok={self.r['ok']} in {self.r['elapsed_s']} s")


def next_job():
    d = FIRST.get("serve_dir")
    if not d:
        return None
    for p in sorted(glob.glob(os.path.join(d, "*.job.json"))):
        os.rename(p, p + ".taken")
        job = json.load(open(p + ".taken"))
        for k in ("mannequin", "res", "warm_frames", "settle_frames"):
            job.setdefault(k, FIRST.get(k))
        return job
    return None


def quit_editor():
    unreal.unregister_slate_post_tick_callback(HANDLE)
    log(f"quitting after {time.time() - S['t0']:.1f} s")
    unreal.SystemLibrary.quit_editor()


def tick(dt):
    if S["busy"]:
        return
    S["busy"] = True
    try:
        _tick()
    except Exception:
        err = traceback.format_exc()
        unreal.log_error(err)
        if S["job"]:
            S["job"].r["error"] = err
            S["job"].finish()
            S["job"] = None
            S["idle_since"] = time.time()
        if not FIRST.get("serve_dir"):
            quit_editor()
    finally:
        S["busy"] = False


def _tick():
    S["n"] += 1
    if S["first"]:
        S["first"] = False
        unreal.AutomationLibrary.set_scalability_quality_to_epic(
            unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world())
        S["job"] = Job(FIRST)
        log(f"editor ready after {time.time() - S['t0']:.1f} s of Python time")
        return
    if S["job"] is not None:
        if S["job"].step():
            S["job"].finish()
            S["job"] = None
            S["idle_since"] = time.time()
            if not FIRST.get("serve_dir"):
                quit_editor()
        return
    d = FIRST.get("serve_dir")
    if os.path.exists(os.path.join(d, "quit")) or time.time() - S["idle_since"] > FIRST.get("serve_idle_s", 300):
        quit_editor()
        return
    if S["n"] % 10 == 0:
        nj = next_job()
        if nj:
            S["job"] = Job(nj)


HANDLE = unreal.register_slate_post_tick_callback(tick)
log("registered")
