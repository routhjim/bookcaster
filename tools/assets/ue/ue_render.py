"""UE side, GPU: validation stills of one imported asset on the stage level. Run by the start-up shim
(ue/init_unreal.py) inside a full editor started with -RenderOffscreen on the stage map; ticks until done, then quits.

Job (env ASSETS_UE_JOB): {"script": this file, "map", "mesh": static mesh path, "id", "out": folder,
     "res": [1152, 640], "warm_s": 20, "settle_s": 3, "ref_box": "/Game/AssetStage/M_StageRef",
     "mannequin": skeletal mesh path or null, "result": path}

Shots (asset front = UE +Y): wide (asset + 1.8 m box + mannequin), medium 3/4 front, close on the top front edge,
back 3/4 high. Each shot: pilot the viewport through a CineCamera (36 x 20 mm back = 1152:640, no depth of field),
let Lumen settle for settle_s, take a high-res screenshot. The tick callback guards against re-entry:
take_high_res_screenshot pumps Slate, which calls the tick again inside the call (the "logged ten times" fault).
"""

import json
import math
import os
import time
import traceback

import unreal

JOB = json.load(open(os.environ["ASSETS_UE_JOB"]))
RES = JOB.get("res", [1152, 640])
OUT = JOB["out"]
os.makedirs(OUT, exist_ok=True)
EAL = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
S = {"t0": time.time(), "busy": False, "phase": "start", "n": 0, "shots": [], "i": 0, "t_phase": time.time(),
     "result": {"ok": False, "shots": []}}


def log(msg):
    unreal.log_warning(f"ASSETS render: {msg}")


def cvar(name):
    try:
        return unreal.SystemLibrary.get_console_variable_int_value(name)
    except Exception:
        return None


def look(cam_loc, target):
    return unreal.MathLibrary.find_look_at_rotation(cam_loc, target)


def place_camera(name, target, az, el, dist, focal):
    a, e = math.radians(az), math.radians(el)
    loc = target + unreal.Vector(math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)) * dist
    cam = act.spawn_actor_from_class(unreal.CineCameraActor, loc, look(loc, target))
    cam.set_actor_label(f"cam_{name}")
    cc = cam.get_cine_camera_component()
    fb = cc.get_editor_property("filmback")
    fb.sensor_width, fb.sensor_height = 36.0, 36.0 * RES[1] / RES[0]
    cc.set_editor_property("filmback", fb)
    cc.set_editor_property("current_focal_length", focal)
    fs = cc.get_editor_property("focus_settings")
    fs.focus_method = unreal.CameraFocusMethod.DISABLE
    cc.set_editor_property("focus_settings", fs)
    return cam


def fit_dist(radius, focal, margin=1.1):
    hf = 2 * math.atan(18.0 / focal)
    vf = 2 * math.atan(18.0 * RES[1] / RES[0] / focal)
    return radius / math.sin(min(hf, vf) / 2) * margin


def setup():
    sm = EAL.load_asset(JOB["mesh"])
    a = act.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    a.set_actor_label(f"asset_{JOB['id']}")
    a.static_mesh_component.set_static_mesh(sm)
    bb = sm.get_bounding_box()
    mn, mx = bb.min, bb.max
    L, W, H = mx.x - mn.x, mx.y - mn.y, mx.z - mn.z
    S["result"]["bounds_cm"] = [L, W, H]
    # size references: 1.8 m box at the asset's left (-X), mannequin at its right (+X), both facing +Y
    box = act.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(mn.x - 70, 0, 0), unreal.Rotator(0, 0, 0))
    box.set_actor_label("ref_box_1p8m")
    box.static_mesh_component.set_static_mesh(EAL.load_asset("/Engine/BasicShapes/Cube"))
    box.set_actor_scale3d(unreal.Vector(0.4, 0.4, 1.8))
    box.set_actor_location(unreal.Vector(mn.x - 70, 0, 90), False, False)
    if JOB.get("ref_box_material") and EAL.does_asset_exist(JOB["ref_box_material"]):
        box.static_mesh_component.set_material(0, EAL.load_asset(JOB["ref_box_material"]))
    right = mx.x
    if JOB.get("mannequin") and EAL.does_asset_exist(JOB["mannequin"]):
        mq = act.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(mx.x + 70, 0, 0),
                                        unreal.Rotator(0, 0, 0))
        mq.set_actor_label("ref_mannequin")
        mq.skeletal_mesh_component.set_skinned_asset_and_update(EAL.load_asset(JOB["mannequin"]))
        right = mx.x + 110
        S["result"]["mannequin"] = True
    c = unreal.Vector(0, 0, H / 2)
    r_asset = math.sqrt(L * L + W * W + H * H) / 2
    gl, gr = mn.x - 100, right
    gc = unreal.Vector((gl + gr) / 2, 0, 95)
    r_group = math.sqrt(((gr - gl) / 2) ** 2 + 95 ** 2)
    close_t = unreal.Vector(mx.x * 0.35, mx.y * 0.6, H * 0.85)
    close_w = max(L, W, H) * 0.45
    S["shots"] = [
        ("wide", place_camera("wide", gc, 15, 8, fit_dist(r_group, 35, 1.05), 35)),
        ("medium", place_camera("medium", c, -35, 22, fit_dist(r_asset, 50, 1.12), 50)),
        ("close", place_camera("close", close_t, 25, 30, close_w / 2 / math.tan(math.atan(18 / 85.0)), 85)),
        ("back", place_camera("back", c, 150, 30, fit_dist(r_asset, 50, 1.15), 50)),
    ]


def tick(dt):
    if S["busy"]:
        return
    S["busy"] = True
    try:
        _tick()
    except Exception:
        S["result"]["error"] = traceback.format_exc()
        unreal.log_error(S["result"]["error"])
        finish()
    finally:
        S["busy"] = False


def finish():
    r = S["result"]
    r["frames"] = S["n"]
    r["elapsed_s"] = round(time.time() - S["t0"], 1)
    r["cvars"] = {k: cvar(k) for k in ("r.RayTracing", "r.Lumen.HardwareRayTracing", "r.DynamicGlobalIlluminationMethod",
                                       "r.ReflectionMethod", "r.Nanite", "r.Shadow.Virtual.Enable")}
    json.dump(r, open(JOB["result"], "w"), indent=1, default=str)
    log(f"done ok={r['ok']} in {r['elapsed_s']} s")
    unreal.unregister_slate_post_tick_callback(HANDLE)
    unreal.SystemLibrary.quit_editor()


def _tick():
    S["n"] += 1
    now = time.time()
    el = now - S["t_phase"]
    ph = S["phase"]
    if ph == "start":
        w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        if w is None or not w.get_path_name().startswith(JOB["map"]):
            log(f"loading {JOB['map']}")
            les.load_level(JOB["map"])
        les.editor_set_game_view(True)
        # (editor_set_viewport_realtime trips a handled ensure in 5.8.1; the offscreen viewport ticks anyway)
        w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        unreal.AutomationLibrary.set_scalability_quality_to_epic(w)
        setup()
        S["phase"], S["t_phase"] = "warm", now
        log("set up; warming")
    elif ph == "warm" and el > JOB.get("warm_s", 20):
        unreal.AutomationLibrary.finish_loading_before_screenshot()   # shaders + streaming
        S["result"]["warm_s"] = round(now - S["t0"], 1)
        S["phase"], S["t_phase"] = "aim", now
    elif ph == "aim":
        name, cam = S["shots"][S["i"]]
        les.pilot_level_actor(cam)
        S["phase"], S["t_phase"] = "settle", now
    elif ph == "settle" and el > JOB.get("settle_s", 3):
        name, cam = S["shots"][S["i"]]
        path = os.path.join(OUT, f"{JOB['id']}_{name}.png")
        if os.path.exists(path):
            os.remove(path)
        S["phase"], S["t_phase"] = "shoot", now     # set before the call: it re-enters the tick
        unreal.AutomationLibrary.take_high_res_screenshot(RES[0], RES[1], path, camera=cam, force_game_view=True)
        S["pending"] = path
    elif ph == "shoot":
        path = S["pending"]
        if os.path.exists(path) and os.path.getsize(path) > 0 and el > 0.5:
            name = S["shots"][S["i"]][0]
            S["result"]["shots"].append({"name": name, "path": path, "t_s": round(now - S["t0"], 1)})
            log(f"shot {name} -> {path}")
            les.eject_pilot_level_actor()
            S["i"] += 1
            if S["i"] >= len(S["shots"]):
                S["result"]["ok"] = True
                finish()
            else:
                S["phase"], S["t_phase"] = "aim", now
        elif el > 60:
            raise RuntimeError(f"screenshot not written: {path}")


HANDLE = unreal.register_slate_post_tick_callback(tick)
log("registered")
