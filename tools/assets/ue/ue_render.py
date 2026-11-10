"""UE side, GPU: import (optional) and validation stills of one asset on the stage level, in one editor process.
Run by the start-up shim (ue/init_unreal.py) inside a full editor started with -RenderOffscreen on the stage map.

Job (env ASSETS_UE_JOB): {"script": this file, "map", "id", "out": folder, "result": path,
     "import": {ue_import job} or null (then "mesh" names an imported static mesh),
     "res": [1152, 640], "warm_frames": 90, "settle_frames": 30, "warm_s": 0, "settle_s": 0,
     "ref_box_material", "mannequin": skeletal mesh path or null,
     "serve_dir": null | folder}  (serve: after this job, keep the editor and run every <serve_dir>/*.job.json
                                   the same way; "<serve_dir>/quit" or serve_idle_s without work ends it)

Shots (asset front = UE +Y): a preset per class (job "cameras": {"preset": class, "custom": [{"name", "position",
"target" (metres, the asset's Blender frame), "focal"}]}); see PRESETS (prop: wide, medium, close, back, detail;
vehicle: low medium, close, low rear 3/4 back, side). Waits are counted in rendered frames (Lumen converges per frame, not per second); before the
first shot, finish_loading_before_screenshot() blocks until shaders and streaming are done. The tick callback
guards against re-entry: take_high_res_screenshot pumps Slate, which calls the tick again inside the call (the
"logged ten times" fault of the first GPU test).
"""

import glob
import importlib.util
import json
import math
import os
import time
import traceback

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
EAL = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
_spec = importlib.util.spec_from_file_location("assets_ue_import", os.path.join(HERE, "ue_import.py"))
ue_import = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ue_import)

FIRST = json.load(open(os.environ["ASSETS_UE_JOB"]))
S = {"t0": time.time(), "busy": False, "n": 0, "job": None, "first_level": True, "idle_since": None}


# Camera presets per asset class: (name, kind, azimuth, elevation, focal mm, margin or frame scale). kind "group" =
# asset + 1.8 m box + mannequin, "asset" = the asset's bounding sphere, "detail" = a frame k x 0.45 x the largest
# side wide on the top front edge. Azimuth 0 = the asset's front (UE +Y), positive toward +X.
PRESETS = {
    "prop": [("wide", "group", 15, 8, 35, 1.05), ("medium", "asset", -35, 22, 50, 1.12),
             ("close", "detail", 25, 30, 85, 1.0), ("back", "asset", 150, 30, 50, 1.15),
             ("detail", "detail", 20, 45, 85, 0.4)],
    "vehicle": [("wide", "group", 15, 8, 35, 1.05), ("medium", "asset", -35, 12, 50, 1.1),
                ("close", "detail", 25, 20, 85, 1.0), ("back", "asset", 155, 8, 50, 1.12),
                ("side", "asset", 90, 4, 50, 1.1)],
    "set": [("wide", "group", 15, 8, 35, 1.05), ("medium", "asset", -35, 18, 35, 1.05),
            ("close", "detail", 25, 20, 50, 1.0), ("back", "asset", 150, 18, 35, 1.05)],
    "creature": [("wide", "group", 15, 8, 35, 1.05), ("medium", "asset", -35, 10, 50, 1.1),
                 ("close", "detail", 25, 15, 85, 1.0), ("back", "asset", 150, 10, 50, 1.12),
                 ("detail", "detail", 25, 15, 85, 0.4)],
}
PRESETS["character"] = PRESETS["creature"]
PRESETS["outfit"] = PRESETS["creature"]


def log(msg):
    unreal.log_warning(f"ASSETS render: {msg}")


def cvar(name):
    try:
        return unreal.SystemLibrary.get_console_variable_int_value(name)
    except Exception:
        return None


class Job:
    def __init__(self, job):
        self.j = job
        self.res_px = job.get("res", [1152, 640])
        self.r = {"ok": False, "shots": [], "id": job["id"]}
        self.t0 = time.time()
        self.phase, self.t_phase, self.f_phase = "start", time.time(), S["n"]
        self.actors, self.shots, self.i, self.refs, self.hide = [], [], 0, [], set()
        os.makedirs(job["out"], exist_ok=True)

    # -------------------------------------------------------------- scene
    def camera(self, name, target, az, el, dist, focal):
        a, e = math.radians(az), math.radians(el)
        loc = target + unreal.Vector(math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)) * dist
        cam = act.spawn_actor_from_class(unreal.CineCameraActor, loc,
                                         unreal.MathLibrary.find_look_at_rotation(loc, target))
        cam.set_actor_label(f"cam_{name}")
        cc = cam.get_cine_camera_component()
        fb = cc.get_editor_property("filmback")
        fb.sensor_width, fb.sensor_height = 36.0, 36.0 * self.res_px[1] / self.res_px[0]
        cc.set_editor_property("filmback", fb)
        cc.set_editor_property("current_focal_length", focal)
        fs = cc.get_editor_property("focus_settings")
        fs.focus_method = unreal.CameraFocusMethod.DISABLE
        cc.set_editor_property("focus_settings", fs)
        self.actors.append(cam)
        return cam

    def fit(self, radius, focal, margin=1.1):
        hf = 2 * math.atan(18.0 / focal)
        vf = 2 * math.atan(18.0 * self.res_px[1] / self.res_px[0] / focal)
        return radius / math.sin(min(hf, vf) / 2) * margin

    def setup(self):
        j = self.j
        if j.get("import"):
            ir = ue_import.import_asset(j["import"])
            self.r["import"] = {k: ir.get(k) for k in ("ok", "static_mesh", "problems", "time_s", "error")}
            if not ir.get("ok"):
                raise RuntimeError(f"import failed: {ir.get('error')}")
            j["mesh"] = ir["static_mesh"]
        sm = EAL.load_asset(j["mesh"])
        a = act.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        a.set_actor_label(f"asset_{j['id']}")
        a.static_mesh_component.set_static_mesh(sm)
        self.actors.append(a)
        bb = sm.get_bounding_box()
        mn, mx = bb.min, bb.max
        L, W, H = mx.x - mn.x, mx.y - mn.y, mx.z - mn.z
        self.r["bounds_cm"] = [L, W, H]
        box = act.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(mn.x - 70, 0, 90),
                                         unreal.Rotator(0, 0, 0))
        box.set_actor_label("ref_box_1p8m")
        box.static_mesh_component.set_static_mesh(EAL.load_asset("/Engine/BasicShapes/Cube"))
        box.set_actor_scale3d(unreal.Vector(0.4, 0.4, 1.8))
        if j.get("ref_box_material") and EAL.does_asset_exist(j["ref_box_material"]):
            box.static_mesh_component.set_material(0, EAL.load_asset(j["ref_box_material"]))
        self.actors.append(box)
        self.refs = [box]
        right = mx.x
        if j.get("mannequin") and EAL.does_asset_exist(j["mannequin"]):
            mq = act.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(mx.x + 70, 0, 0),
                                            unreal.Rotator(0, 0, 0))
            mq.set_actor_label("ref_mannequin")
            mq.skeletal_mesh_component.set_skinned_asset_and_update(EAL.load_asset(j["mannequin"]))
            self.actors.append(mq)
            self.refs.append(mq)
            right = mx.x + 110
            self.r["mannequin"] = True
        c = unreal.Vector(0, 0, H / 2)
        r_asset = math.sqrt(L * L + W * W + H * H) / 2
        gl, gr = mn.x - 100, right
        gc = unreal.Vector((gl + gr) / 2, 0, 95)
        r_group = math.sqrt(((gr - gl) / 2) ** 2 + 95 ** 2)
        close_t = unreal.Vector(mx.x * 0.35, mx.y * 0.6, H * 0.85)
        close_w = max(L, W, H) * 0.45
        cams = j.get("cameras") or {}
        ct = cams.get("close_target")      # fractions of the bounding box in the Blender frame (x right, y back, z up)
        if ct:
            close_t = unreal.Vector(mn.x + L * ct[0], mx.y - W * ct[1], mn.z + H * ct[2])
            close_w = max(L, W, H) * float(cams.get("close_width", 0.45))
        preset = PRESETS.get(cams.get("preset") or "prop", PRESETS["prop"])
        self.shots = []
        self.hide = set()          # shots of the asset alone: the reference box and mannequin are hidden
        for s in preset:
            name, kind, az, el, focal, k = s
            if kind == "group":
                cam = self.camera(name, gc, az, el, self.fit(r_group, focal, k), focal)
            elif kind == "asset":
                cam = self.camera(name, c, az, el, self.fit(r_asset, focal, k), focal)
            else:                      # detail: a frame k x close_w wide on the top front edge
                cam = self.camera(name, close_t, az, el, close_w * k / 2 * focal / 18, focal)
            self.shots.append((name, cam))
            if kind != "group":
                self.hide.add(name)
        for s in cams.get("custom", []):
            # spec cameras in the asset's Blender frame (metres, front -Y): UE x = 100 x, y = -100 y, z = 100 z
            P_, T_ = s["position"], s["target"]
            loc = unreal.Vector(P_[0] * 100, -P_[1] * 100, P_[2] * 100)
            tgt = unreal.Vector(T_[0] * 100, -T_[1] * 100, T_[2] * 100)
            cam = act.spawn_actor_from_class(unreal.CineCameraActor, loc,
                                             unreal.MathLibrary.find_look_at_rotation(loc, tgt))
            cam.set_actor_label(f"cam_{s['name']}")
            cc = cam.get_cine_camera_component()
            fb = cc.get_editor_property("filmback")
            fb.sensor_width, fb.sensor_height = 36.0, 36.0 * self.res_px[1] / self.res_px[0]
            cc.set_editor_property("filmback", fb)
            cc.set_editor_property("current_focal_length", float(s.get("focal", 50)))
            fs = cc.get_editor_property("focus_settings")
            fs.focus_method = unreal.CameraFocusMethod.DISABLE
            cc.set_editor_property("focus_settings", fs)
            self.actors.append(cam)
            self.shots.append((s["name"], cam))
            if not s.get("show_refs"):
                self.hide.add(s["name"])

    def teardown(self):
        try:
            les.eject_pilot_level_actor()
        except Exception:
            pass
        for a in self.actors:
            try:
                act.destroy_actor(a)
            except Exception:
                pass

    # -------------------------------------------------------------- state machine, one call per tick
    def waited(self, frames, secs):
        return S["n"] - self.f_phase >= frames and time.time() - self.t_phase >= secs

    def goto(self, phase):
        self.phase, self.t_phase, self.f_phase = phase, time.time(), S["n"]

    def step(self):
        j, ph = self.j, self.phase
        if ph == "start":
            self.setup()
            self.r["setup_s"] = round(time.time() - self.t0, 2)
            log(f"{j['id']}: set up in {self.r['setup_s']} s; warming")
            self.goto("warm")
        elif ph == "warm" and self.waited(j.get("warm_frames", 90), j.get("warm_s", 0)):
            t = time.time()
            unreal.AutomationLibrary.finish_loading_before_screenshot()   # blocks on shaders + streaming
            self.r["finish_loading_s"] = round(time.time() - t, 2)
            self.r["warm_done_s"] = round(time.time() - self.t0, 2)
            self.goto("aim")
        elif ph == "aim":
            for a in self.refs:
                a.set_actor_hidden_in_game(self.shots[self.i][0] in self.hide)
            les.pilot_level_actor(self.shots[self.i][1])
            self.goto("settle")
        elif ph == "settle" and self.waited(j.get("settle_frames", 30), j.get("settle_s", 0)):
            name, cam = self.shots[self.i]
            path = os.path.join(j["out"], f"{j['id']}_{name}.png")
            if os.path.exists(path):
                os.remove(path)
            self.pending = path
            self.goto("shoot")                       # before the call: it re-enters the tick
            unreal.AutomationLibrary.take_high_res_screenshot(self.res_px[0], self.res_px[1], path, camera=cam,
                                                             force_game_view=True)
        elif ph == "shoot":
            if os.path.exists(self.pending) and os.path.getsize(self.pending) > 0:
                name = self.shots[self.i][0]
                self.r["shots"].append({"name": name, "path": self.pending, "t_s": round(time.time() - self.t0, 2),
                                        "frames_since_start": S["n"] - self.f0})
                log(f"shot {name}")
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
        self.teardown()
        r = self.r
        r["elapsed_s"] = round(time.time() - self.t0, 2)
        r["editor_uptime_s"] = round(time.time() - S["t0"], 2)
        r["cvars"] = {k: cvar(k) for k in ("r.RayTracing", "r.Lumen.HardwareRayTracing",
                                           "r.DynamicGlobalIlluminationMethod", "r.ReflectionMethod", "r.Nanite",
                                           "r.Shadow.Virtual.Enable")}
        json.dump(r, open(self.j["result"], "w"), indent=1, default=str)
        log(f"{self.j['id']}: done ok={r['ok']} in {r['elapsed_s']} s")


def next_job():
    d = FIRST.get("serve_dir")
    if not d:
        return None
    for p in sorted(glob.glob(os.path.join(d, "*.job.json"))):
        job = json.load(open(p))
        os.rename(p, p + ".taken")
        job.setdefault("result", p[:-9] + ".result.json")
        for k in ("map", "ref_box_material", "mannequin", "res", "warm_frames", "settle_frames"):
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
        if not FIRST.get("serve_dir"):
            quit_editor()
    finally:
        S["busy"] = False


def _tick():
    S["n"] += 1
    if S["first_level"]:
        S["first_level"] = False
        w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        if w is None or not w.get_path_name().startswith(FIRST["map"]):
            log(f"loading {FIRST['map']}")
            les.load_level(FIRST["map"])
        les.editor_set_game_view(True)
        unreal.AutomationLibrary.set_scalability_quality_to_epic(
            unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world())
        S["job"] = Job(FIRST)
        S["job"].f0 = S["n"]
        log(f"editor ready after {time.time() - S['t0']:.1f} s of Python time")
        return
    job = S["job"]
    if job is not None:
        if job.step():
            job.finish()
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
            S["job"].f0 = S["n"]


HANDLE = unreal.register_slate_post_tick_callback(tick)
log("registered")
