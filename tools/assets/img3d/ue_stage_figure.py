"""Editor side of stage.py (run_asset.py --ue-job): put library figures that are plain meshes (a static picture-to-3D
statue, or its rigged skeletal copy holding a pose) into a library set and photograph them with the scene tool's
cameras (scenes/framing.py), plus the depth guides an image model is given (the scene's depth and each figure's
own depth, float EXR in UE centimetres, as the scene tool's guide pass writes them).

Job {"id", "level", "res", "out", "result", "framing": path of scenes/framing.py, "warm_frames", "settle_frames",
     "exposure_ev100": null | EV, "guides": bool,
     "figures": {key: {"mesh": UE path (StaticMesh or SkeletalMesh), "anim": AnimSequence path | null, "t": s,
                       "loc_cm": [x, y, z] (UE), "yaw": deg, "facing": [dx, dy] (set frame), "position": [x, y, z],
                       "height_m": the figure's standing height (for eye / chin / waist when the mesh has no head
                       joint)}},
     "cameras": the scene tool's camera list (wide with position/target; close / medium / full computed)}
Set frame: metres, x = UE X / 100, y = -UE Y / 100, z = UE Z / 100 (scenes/SETS.md).
The body measures come from the posed bounds: feet = bottom, head top = top, and the head's eye line, chin and the
waist from standard proportions of the standing height (eye 0.065 H under the top, chin 0.13 H, waist 0.40 H above
the feet). A rigged figure with a "head" socket uses it for the eye and chin height and the head's plan position.
"""
import importlib.util
import json
import math
import os
import time
import traceback

import unreal

EAL = unreal.EditorAssetLibrary
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def to_set(v):
    return [v.x / 100.0, -v.y / 100.0, v.z / 100.0]


def to_ue(p):
    return unreal.Vector(p[0] * 100.0, -p[1] * 100.0, p[2] * 100.0)


def frames(n):
    for _ in range(int(n)):
        yield


def cine(name, pos, target, focal, aperture, focus_m, res):
    loc, tgt = to_ue(pos), to_ue(target)
    cam = act.spawn_actor_from_class(unreal.CineCameraActor, loc, unreal.MathLibrary.find_look_at_rotation(loc, tgt))
    cam.set_actor_label(f"stage_cam_{name}")
    cc = cam.get_cine_camera_component()
    fb = cc.get_editor_property("filmback")
    fb.sensor_width, fb.sensor_height = 36.0, 36.0 * res[1] / res[0]
    cc.set_editor_property("filmback", fb)
    cc.set_editor_property("current_focal_length", float(focal))
    cc.set_editor_property("current_aperture", float(aperture))
    fs = cc.get_editor_property("focus_settings")
    fs.focus_method = unreal.CameraFocusMethod.MANUAL
    fs.manual_focus_distance = float(focus_m) * 100.0
    cc.set_editor_property("focus_settings", fs)
    return cam


def guide_capture(cs, res, path, show_only=None):
    """Linear scene depth (UE cm, float RGBA) through a SceneCapture2D with the still's camera, to `path` (.exr)."""
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    loc, tgt = to_ue(cs["position"]), to_ue(cs["target"])
    a = act.spawn_actor_from_class(unreal.SceneCapture2D, loc, unreal.MathLibrary.find_look_at_rotation(loc, tgt))
    try:
        comp = a.capture_component2d
        rt = unreal.RenderingLibrary.create_render_target2d(world, res[0], res[1],
                                                            unreal.TextureRenderTargetFormat.RTF_RGBA32F)
        comp.set_editor_property("texture_target", rt)
        comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_DEPTH)
        comp.set_editor_property("fov_angle", math.degrees(2 * math.atan(18.0 / cs["focal_mm"])))
        comp.set_editor_property("capture_every_frame", False)
        comp.set_editor_property("capture_on_movement", False)
        if show_only is not None:
            comp.set_editor_property("primitive_render_mode", unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST)
            for x in show_only:
                comp.show_only_actor_components(x, True)
        comp.capture_scene()
        yield
        yield
        opt = unreal.ImageWriteOptions()
        opt.set_editor_property("format", unreal.DesiredImageFormat.EXR)
        opt.set_editor_property("overwrite_file", True)
        opt.set_editor_property("async_", False)
        opt.set_editor_property("compression_quality", 1)
        unreal.ImageWriteBlueprintLibrary.export_to_disk(rt, path, opt)
    finally:
        act.destroy_actor(a)


def find_post():
    for a in act.get_all_level_actors():
        if isinstance(a, unreal.PostProcessVolume) and a.get_actor_label() == "SetPost":
            return a
    return None


def spawn(key, f):
    mesh = EAL.load_asset(f["mesh"])
    if mesh is None:
        raise RuntimeError(f"{key}: no asset {f['mesh']}")
    rot = unreal.Rotator(0, 0, f["yaw"])
    loc = unreal.Vector(*f["loc_cm"])
    if isinstance(mesh, unreal.SkeletalMesh):
        a = act.spawn_actor_from_class(unreal.SkeletalMeshActor, loc, rot)
        c = a.skeletal_mesh_component
        c.set_skinned_asset_and_update(mesh)
        c.set_editor_property("visibility_based_anim_tick_option",
                              unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
    else:
        a = act.spawn_actor_from_object(mesh, loc, rot)
    a.set_actor_label(f"stage_fig_{key}")
    return a


def pose(a, f):
    if not f.get("anim") or not isinstance(a, unreal.SkeletalMeshActor):
        return
    c = a.skeletal_mesh_component
    anim = EAL.load_asset(f["anim"])
    c.set_update_animation_in_editor(True)
    c.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    c.set_animation(anim)
    c.play(False)
    c.set_position(float(f.get("t", 0.0)), False)
    c.set_play_rate(0.0)
    c.stop()
    c.set_position(float(f.get("t", 0.0)), False)


def measure(a, f):
    o, e = a.get_actor_bounds(False)
    lo, hi = to_set(o - e), to_set(o + e)
    feet, top = min(lo[2], hi[2]), max(lo[2], hi[2])
    H = top - feet
    base = f["position"]
    fx, fy = f["facing"]
    head_xy = [base[0] + fx * 0.04, base[1] + fy * 0.04]
    eye_z, chin_z = top - 0.065 * H, top - 0.13 * H
    if isinstance(a, unreal.SkeletalMeshActor):
        c = a.skeletal_mesh_component
        if c.does_socket_exist("head"):
            hp = to_set(c.get_socket_location("head"))
            head_xy = [hp[0] + fx * 0.08, hp[1] + fy * 0.08]
    out = {"feet_z": feet, "head_top_z": top, "eye_z": eye_z, "chin_z": chin_z, "waist_z": feet + 0.60 * H,
           "eye_xy": head_xy, "chest_xy": base[:2], "pelvis_xy": base[:2], "facing": [fx, fy],
           "head_fwd_xy": [fx, fy], "height_m": H, "bounds_set": [lo, hi]}
    return {k: ([round(x, 4) for x in v] if isinstance(v, list) and v and not isinstance(v[0], list) else
                (round(v, 4) if isinstance(v, float) else v)) for k, v in out.items()}


def plan_angle(fwd, frm, to):
    d = [to[0] - frm[0], to[1] - frm[1]]
    a = math.degrees(math.atan2(d[1], d[0]) - math.atan2(fwd[1], fwd[0]))
    return round(abs((a + 180) % 360 - 180), 1)


def run(job):
    t0 = time.time()
    r = {"ok": False, "id": job["id"], "shots": [], "subjects": {}, "times_s": {}}
    res = job.get("res", [1152, 640])
    temp, post_old = [], None
    framing = _load("stage_framing", job["framing"])
    try:
        os.makedirs(job["out"], exist_ok=True)
        les.load_level(job["level"])
        yield from frames(5)
        les.editor_set_game_view(True)
        unreal.AutomationLibrary.set_scalability_quality_to_epic(
            unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world())
        ppv = find_post()
        if ppv is not None and job.get("exposure_ev100") is not None:
            pp = ppv.get_editor_property("settings")
            post_old = (ppv, pp.get_editor_property("auto_exposure_bias"))
            pp.set_editor_property("override_auto_exposure_bias", True)
            pp.set_editor_property("auto_exposure_bias", -float(job["exposure_ev100"]))
            ppv.set_editor_property("settings", pp)
        figs = {}
        for key, f in job["figures"].items():
            figs[key] = spawn(key, f)
            temp.append(figs[key])
        for _ in range(2):
            for key, f in job["figures"].items():
                pose(figs[key], f)
            yield from frames(10)
        yield from frames(job.get("warm_frames", 150))
        unreal.AutomationLibrary.finish_loading_before_screenshot()
        for key, f in job["figures"].items():
            pose(figs[key], f)
        yield from frames(3)
        for key, f in job["figures"].items():
            r["subjects"][key] = {"measured": measure(figs[key], f), "mesh": f["mesh"], "anim": f.get("anim")}
        cams = []
        for c in job["cameras"]:
            subj = r["subjects"][c["subject"]]["measured"] if c.get("subject") else None
            if c["kind"] in ("close", "medium", "full") and not c.get("position"):
                cs = framing.shot(subj, c["kind"], c.get("focal_mm", 50), res, c.get("az", 0.0), c.get("el", 0.0),
                                  c.get("screen_x", 0.5))
            else:
                tgt = c["target"]
                if tgt == "subject":
                    tgt = [subj["pelvis_xy"][0], subj["pelvis_xy"][1], (subj["feet_z"] + subj["head_top_z"]) / 2]
                cs = {"position": c["position"], "target": tgt, "focal_mm": c.get("focal_mm", 24)}
            chk = framing.check(cs, subj, c["kind"], res) if subj else None
            eyes = [subj["eye_xy"][0], subj["eye_xy"][1], subj["eye_z"]] if subj else cs["target"]
            ap = c.get("aperture", {"close": 2.8, "medium": 4.0}.get(c["kind"], 8.0))
            cam = cine(c["name"], cs["position"], cs["target"], cs["focal_mm"], ap, math.dist(cs["position"], eyes), res)
            temp.append(cam)
            cams.append((c, cs, chk, cam))
        for c, cs, chk, cam in cams:
            ts = time.time()
            les.pilot_level_actor(cam)
            yield from frames(10)
            for key, f in job["figures"].items():
                pose(figs[key], f)
            yield from frames(job.get("settle_frames", 60))
            path = os.path.join(job["out"], f"{c['name']}.png")
            if os.path.exists(path):
                os.remove(path)
            unreal.AutomationLibrary.take_high_res_screenshot(res[0], res[1], path, camera=cam, force_game_view=True)
            tw = time.time()
            while not (os.path.exists(path) and os.path.getsize(path) > 0):
                if time.time() - tw > 90:
                    raise RuntimeError(f"screenshot not written: {path}")
                yield
            les.eject_pilot_level_actor()
            guides = {}
            if job.get("guides"):
                g = os.path.join(job["out"], "guides")
                os.makedirs(g, exist_ok=True)
                guides["depth"] = os.path.join(g, f"{c['name']}_depth.exr")
                yield from guide_capture(cs, res, guides["depth"])
                for key in job["figures"]:
                    guides[f"depth_{key}"] = os.path.join(g, f"{c['name']}_depth_{key}.exr")
                    yield from guide_capture(cs, res, guides[f"depth_{key}"], [figs[key]])
            truth = {}
            for key, v in r["subjects"].items():
                m = v["measured"]
                truth[key] = {"head_to_camera_deg": plan_angle(m["head_fwd_xy"], m["eye_xy"], cs["position"]),
                              "body_to_camera_deg": plan_angle(m["facing"], m["pelvis_xy"], cs["position"]),
                              "distance_m": round(math.dist(m["eye_xy"] + [m["eye_z"]], cs["position"]), 3)}
            r["shots"].append({"name": c["name"], "path": path, "guides": guides, "truth": truth, "kind": c["kind"],
                               "camera": cs, "frame_check": chk, "still_s": round(time.time() - ts, 2)})
        r["ok"] = True
    except Exception:
        r["error"] = traceback.format_exc()
        unreal.log_error(r["error"])
    finally:
        try:
            les.eject_pilot_level_actor()
        except Exception:
            pass
        for a in temp:
            try:
                act.destroy_actor(a)
            except Exception:
                pass
        if post_old:
            ppv, bias = post_old
            pp = ppv.get_editor_property("settings")
            pp.set_editor_property("auto_exposure_bias", bias)
            ppv.set_editor_property("settings", pp)
        r["times_s"]["total"] = round(time.time() - t0, 2)
        json.dump(r, open(job["result"], "w"), indent=1, default=str)
    if r.get("error"):
        raise RuntimeError(r["error"])
    return r
