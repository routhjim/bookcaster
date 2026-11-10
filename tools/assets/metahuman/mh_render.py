"""UE side, GPU: spawn MetaHuman Character preview actors in a lit level and render stills, optionally posed.

Run by the start-up shim (init_unreal.py) inside a full editor started by mh_run.py with -RenderOffscreen.

Job (env MH_JOB):
  {"script": this file, "map": level to load, "out": folder, "result": path, "res": [1152, 640],
   "warm_frames": 120, "settle_frames": 40, "lod": 0,
   "shots": [{"name", "character": "/Game/MH/MH_X", "frame": "close" | "full" | "close_side",
              "face_anim": anim path | null, "face_t": s, "face_curves": {curve: value} | null,
              "body_anim": anim path | null, "body_t": s, "body_scan": null | "arms_up" | "head_turn"}]}

The character is opened for editing in the MetaHumanCharacterEditorSubsystem and spawned with
spawn_meta_human_actor (the editor preview actor: body + face skeletal meshes, grooms, outfit), which needs no
assembly and no cloud step. Its face only animates when the character has face DNA (a rig); the preview actor then
runs the face post-process AnimBP (RigLogic). Animations are applied with override_animation_data (single node,
paused at a time). face_curves builds a one-frame AnimSequence on the face skeleton holding those curve values.
"""

import json
import math
import os
import time
import traceback

import unreal

EAL = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
mhs = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
at = unreal.AssetToolsHelpers.get_asset_tools()
JOB = json.load(open(os.environ["MH_JOB"]))
RES = JOB.get("res", [1152, 640])
S = {"n": 0, "busy": False, "t0": time.time(), "phase": "boot", "f_phase": 0, "i": 0, "actors": {},
     "scene": [], "r": {"ok": False, "shots": [], "notes": []}, "pending": None, "edit": []}


def log(m):
    unreal.log_warning(f"MH render: {m}")


def note(m):
    S["r"]["notes"].append(m)
    log(m)


def spawn(cls, loc, rot=(0, 0, 0), label=None):
    a = act.spawn_actor_from_class(cls, unreal.Vector(*loc), unreal.Rotator(roll=rot[0], pitch=rot[1], yaw=rot[2]))
    if label:
        a.set_actor_label(label)
    S["scene"].append(a)
    return a


def light_scene():
    """Portrait lights on top of the template level (it already has floor, sun, sky light, atmosphere, fog)."""
    key = spawn(unreal.RectLight, (120, 160, 190), (0, -15, -125), "mh_key")      # front-left, a little above
    kc = key.get_component_by_class(unreal.RectLightComponent)
    kc.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
    kc.set_editor_property("intensity", float(JOB.get("key_cd", 40)))
    kc.set_editor_property("source_width", 120.0)
    kc.set_editor_property("source_height", 120.0)
    kc.set_editor_property("attenuation_radius", 1500.0)
    fill = spawn(unreal.RectLight, (-180, 200, 160), (0, -5, -50), "mh_fill")      # front-right, softer
    fc = fill.get_component_by_class(unreal.RectLightComponent)
    fc.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
    fc.set_editor_property("intensity", float(JOB.get("fill_cd", 12)))
    fc.set_editor_property("source_width", 200.0)
    fc.set_editor_property("source_height", 200.0)
    rim = spawn(unreal.RectLight, (-60, -200, 220), (0, -25, 75), "mh_rim")        # behind, for hair edge
    rc = rim.get_component_by_class(unreal.RectLightComponent)
    rc.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
    rc.set_editor_property("intensity", float(JOB.get("rim_cd", 25)))
    ppv = spawn(unreal.PostProcessVolume, (0, 0, 0), label="mh_pp")
    ppv.set_editor_property("unbound", True)
    pp = ppv.get_editor_property("settings")
    if JOB.get("exposure_bias") is not None:
        pp.set_editor_property("override_auto_exposure_bias", True)
        pp.set_editor_property("auto_exposure_bias", float(JOB["exposure_bias"]))
    ppv.set_editor_property("settings", pp)


def character_actor(path):
    if path in S["actors"]:
        return S["actors"][path]
    c = EAL.load_asset(path)
    if c is None:
        raise RuntimeError(f"no character asset {path}")
    if not mhs.is_object_added_for_editing(c):
        if not mhs.try_add_object_to_edit(c):
            raise RuntimeError(f"try_add_object_to_edit failed for {path}")
        S["edit"].append(c)
    a = mhs.spawn_meta_human_actor(c, True)
    if a is None:
        raise RuntimeError(f"spawn_meta_human_actor returned None for {path}")
    a.set_actor_location(unreal.Vector(0, 0, 0), False, False)
    try:
        lod = a.get_editor_property("lod_sync_component")
        lod.set_editor_property("forced_lod", int(JOB.get("lod", 0)))
    except Exception as e:
        note(f"forced LOD not set: {e}")
    for comp in (a.get_editor_property("body_component"), a.get_editor_property("face_component")):
        try:
            comp.set_update_animation_in_editor(True)
        except Exception as e:
            note(f"set_update_animation_in_editor: {e}")
    if JOB.get("assemble_after_spawn", True):
        mhs.assemble_for_preview(c)
    if JOB.get("no_animated_maps"):
        info_m = []
        face = a.get_editor_property("face_component")
        for i, m in enumerate(face.get_materials()):
            if m is None:
                continue
            row = {"slot": i, "class": m.get_class().get_name(), "name": m.get_name()}
            if not isinstance(m, unreal.MaterialInstanceDynamic):
                m = face.create_dynamic_material_instance(i, m)
            if i != 0:
                continue
            MEL = unreal.MaterialEditingLibrary
            try:
                row["scalars"] = [str(x) for x in MEL.get_scalar_parameter_names(m) if "nimat" in str(x)]
                row["switches"] = [str(x) for x in MEL.get_static_switch_parameter_names(m) if "nimat" in str(x)]
                row["textures"] = [str(x) for x in MEL.get_texture_parameter_names(m) if "nimat" in str(x)]
                names = [str(x) for x in MEL.get_texture_parameter_names(m)]
                vals = {}
                for nm in names:
                    tex = m.get_texture_parameter_value(nm)
                    vals[nm] = tex.get_path_name() if tex else None
                row["texture_values"] = vals
                # The switch is static. Offline the full animated (wrinkle/blood-flow) maps are white/flat
                # placeholders, which bleach the face as soon as an expression drives them; point them at the
                # neutral synthesized maps and keep the synthesized deltas (or force one: "animated_delta_texture").
                base_c = m.get_texture_parameter_value(JOB.get("base_color_param", "Basecolor"))
                base_n = m.get_texture_parameter_value(JOB.get("base_normal_param", "Normal"))
                delta = EAL.load_asset(JOB.get("animated_delta_texture", "/Engine/EngineResources/Black"))
                for nm in names:
                    if "Animated" not in nm or nm.endswith("VT"):
                        continue
                    if "Delta" in nm:
                        if JOB.get("animated_delta_texture"):
                            m.set_texture_parameter_value(nm, delta)
                    elif nm.startswith("Basecolor") and base_c:
                        m.set_texture_parameter_value(nm, base_c)
                    elif nm.startswith("Normal") and base_n:
                        m.set_texture_parameter_value(nm, base_n)
                row["base"] = [base_c.get_path_name() if base_c else None, base_n.get_path_name() if base_n else None]
            except Exception as e:
                row["err"] = str(e)
            info_m.append(row)
        S["r"].setdefault("animated_maps", {})[path] = info_m
    S["actors"][path] = a
    info = {"class": a.get_class().get_name(),
            "face_mesh": str(a.face_component.get_skinned_asset().get_path_name()),
            "body_mesh": str(a.body_component.get_skinned_asset().get_path_name()),
            "face_anim_class": str(a.face_component.get_anim_instance().get_class().get_name()
                                   if a.face_component.get_anim_instance() else None),
            "face_post_process": str(a.face_component.get_post_process_instance().get_class().get_name()
                                     if a.face_component.get_post_process_instance() else None),
            "components": sorted({c.get_class().get_name() for c in a.get_components_by_class(unreal.ActorComponent)})}
    info["grooms"] = []
    for g in a.get_components_by_class(unreal.GroomComponent):
        gi = {"name": g.get_name(), "visible": g.is_visible()}
        for prop in ("groom_asset", "binding_asset"):
            try:
                v = g.get_editor_property(prop)
                gi[prop] = v.get_path_name() if v else None
            except Exception as e:
                gi[prop] = f"ERR {e}"
        info["grooms"].append(gi)
    S["r"].setdefault("actors", {})[path] = info
    log(f"spawned {path}: {json.dumps(info)}")
    return a


def show_only(path):
    for p, a in S["actors"].items():
        a.set_is_temporarily_hidden_in_editor(p != path)
        a.set_actor_hidden_in_game(p != path)


def curve_anim(skel, curves, name):
    """One-frame AnimSequence on the face skeleton with constant curve values."""
    folder = "/Game/MH/Anim"
    path = f"{folder}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    f = unreal.AnimSequenceFactory()
    f.set_editor_property("target_skeleton", skel)
    anim = at.create_asset(name, folder, unreal.AnimSequence, f)
    ctrl = anim.get_editor_property("controller")
    ctrl.open_bracket("mh curves", False)
    ctrl.set_frame_rate(unreal.FrameRate(30, 1), False)
    ctrl.set_number_of_frames(unreal.FrameNumber(2), False)
    for cname, val in curves.items():
        unreal.AnimationLibrary.add_curve(anim, cname, unreal.RawCurveTrackTypes.RCT_FLOAT, False)
        for t in (0.0, 1.0 / 30, 2.0 / 30):
            unreal.AnimationLibrary.add_float_curve_key(anim, cname, t, float(val))
    ctrl.close_bracket(False)
    return anim


def bone_world(anim, chain, t):
    """Component-space transform of the last bone of a root-first chain, from the animation's local poses."""
    poses = unreal.AnimationLibrary.get_bone_poses_for_time(anim, chain, t, False)
    x = unreal.Transform()
    for p in poses:
        x = unreal.MathLibrary.compose_transforms(p, x)
    return x


ARM_R = ["root", "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "clavicle_r", "upperarm_r",
         "lowerarm_r", "hand_r"]
ARM_L = [b.replace("_r", "_l") for b in ARM_R]
HEAD = ["root", "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head"]


def scan_body(anim, kind):
    """Pick the time in a body animation that best matches a wanted pose (hands highest / head yaw near 45)."""
    n = 60
    length = unreal.AnimationLibrary.get_sequence_length(anim)
    best, best_t = None, 0.0
    neutral = bone_world(anim, HEAD, 0.0)
    for i in range(n + 1):
        t = length * i / n
        if kind == "arms_up":
            z = min(bone_world(anim, ARM_R, t).translation.z, bone_world(anim, ARM_L, t).translation.z)
            score = z
        else:
            h = bone_world(anim, HEAD, t)
            fwd = h.rotation.rotate_vector(unreal.Vector(0, 1, 0))
            yaw = math.degrees(math.atan2(fwd.x, fwd.y))
            score = -abs(abs(yaw) - 45)
        if best is None or score > best:
            best, best_t = score, t
    return best_t, best, length


def apply_anim(comp, anim, t):
    """Single-node animation paused at time t. override_animation_data alone only took on the first call per
    component (later shots kept the first pose), so the node is re-set each time."""
    comp.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    comp.set_animation(anim)
    comp.play(False)
    comp.set_position(float(t), False)
    comp.set_play_rate(0.0)
    comp.stop()


def clear_anim(comp):
    comp.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    comp.set_animation(None)


def camera(name, target, az, el, dist, focal):
    a, e = math.radians(az), math.radians(el)
    loc = target + unreal.Vector(math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)) * dist
    cam = spawn(unreal.CineCameraActor, (loc.x, loc.y, loc.z), label=f"cam_{name}")
    cam.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(loc, target), False)
    cc = cam.get_cine_camera_component()
    fb = cc.get_editor_property("filmback")
    fb.sensor_width, fb.sensor_height = 36.0, 36.0 * RES[1] / RES[0]
    cc.set_editor_property("filmback", fb)
    cc.set_editor_property("current_focal_length", focal)
    fs = cc.get_editor_property("focus_settings")
    fs.focus_method = unreal.CameraFocusMethod.MANUAL
    fs.manual_focus_distance = dist
    cc.set_editor_property("focus_settings", fs)
    cc.set_editor_property("current_aperture", 4.0)
    return cam


def setup_shot(sh):
    a = character_actor(sh["character"])
    show_only(sh["character"])
    face, body = a.face_component, a.body_component
    info = {"name": sh["name"]}
    if sh.get("body_anim"):
        banim = EAL.load_asset(sh["body_anim"])
        t = sh.get("body_t", 0.0)
        if sh.get("body_scan"):
            t, score, length = scan_body(banim, sh["body_scan"])
            info["body_scan"] = {"t": round(t, 3), "score": round(score, 2), "length": round(length, 2)}
        apply_anim(body, banim, t)
        info["body"] = [sh["body_anim"], round(t, 3)]
    else:
        clear_anim(body)
    if sh.get("face_curves"):
        anim = curve_anim(face.get_skinned_asset().get_editor_property("skeleton"), sh["face_curves"],
                          f"AS_{sh['name']}")
        apply_anim(face, anim, 1.0 / 30)
        info["face"] = sh["face_curves"]
    elif sh.get("face_anim"):
        apply_anim(face, EAL.load_asset(sh["face_anim"]), sh.get("face_t", 0.0))
        info["face"] = [sh["face_anim"], sh.get("face_t", 0.0)]
    else:
        clear_anim(face)
    # aim from the head bone as posed now (falls back to a standing height)
    try:
        head = face.get_socket_location("head")
        if head.z < 50:
            raise ValueError
    except Exception:
        head = unreal.Vector(0, 0, 160)
    info["head"] = [round(head.x, 1), round(head.y, 1), round(head.z, 1)]
    fr = sh.get("frame", "close")
    az = sh.get("az", 0)
    if fr == "close":
        cam = camera(sh["name"], head + unreal.Vector(0, 0, -8), az, 3, 230, 85)
    elif fr == "full":
        cam = camera(sh["name"], unreal.Vector(head.x, head.y, head.z * 0.53), az, 5, 340, 35)
    else:
        cam = camera(sh["name"], head + unreal.Vector(0, 0, -8), az, 3, 230, 85)
    return cam, info


def finish():
    for c in S["edit"]:
        try:
            mhs.remove_object_to_edit(c)
        except Exception:
            pass
    S["r"]["elapsed_s"] = round(time.time() - S["t0"], 1)
    json.dump(S["r"], open(JOB["result"], "w"), indent=1, default=str)
    log(f"done ok={S['r']['ok']} in {S['r']['elapsed_s']} s")
    unreal.unregister_slate_post_tick_callback(HANDLE)
    unreal.SystemLibrary.quit_editor()


def goto(p):
    S["phase"], S["f_phase"], S["t_phase"] = p, S["n"], time.time()


def waited(frames):
    return S["n"] - S["f_phase"] >= frames


def _tick():
    S["n"] += 1
    ph = S["phase"]
    shots = JOB["shots"]
    if ph == "boot":
        w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        if JOB.get("map") and (w is None or not w.get_path_name().startswith(JOB["map"])):
            les.load_level(JOB["map"])
        les.editor_set_game_view(True)
        unreal.AutomationLibrary.set_scalability_quality_to_epic(
            unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world())
        light_scene()
        for sh in shots:
            character_actor(sh["character"])
        S["r"]["setup_s"] = round(time.time() - S["t0"], 1)
        goto("warm")
    elif ph == "warm" and waited(JOB.get("warm_frames", 120)):
        t = time.time()
        unreal.AutomationLibrary.finish_loading_before_screenshot()
        S["r"]["finish_loading_s"] = round(time.time() - t, 1)
        goto("pose")
    elif ph == "pose":
        S["cam"], S["info"] = setup_shot(shots[S["i"]])
        les.pilot_level_actor(S["cam"])
        goto("settle")
    elif ph == "settle" and waited(JOB.get("settle_frames", 40)):
        path = os.path.join(JOB["out"], f"{shots[S['i']]['name']}.png")
        if os.path.exists(path):
            os.remove(path)
        S["pending"] = path
        goto("shoot")
        unreal.AutomationLibrary.take_high_res_screenshot(RES[0], RES[1], path, camera=S["cam"],
                                                         force_game_view=True)
    elif ph == "shoot":
        if os.path.exists(S["pending"]) and os.path.getsize(S["pending"]) > 0:
            S["info"]["path"] = S["pending"]
            S["info"]["t_s"] = round(time.time() - S["t0"], 1)
            S["r"]["shots"].append(S["info"])
            log(f"shot {json.dumps(S['info'])}")
            les.eject_pilot_level_actor()
            S["i"] += 1
            if S["i"] >= len(shots):
                S["r"]["ok"] = True
                finish()
                return
            goto("pose")
        elif time.time() - S["t_phase"] > 90:
            raise RuntimeError(f"screenshot not written: {S['pending']}")


def tick(dt):
    if S["busy"] or S["phase"] == "over":
        return
    S["busy"] = True
    try:
        _tick()
    except Exception:
        S["r"]["error"] = traceback.format_exc()
        unreal.log_error(S["r"]["error"])
        S["phase"] = "over"
        finish()
    finally:
        S["busy"] = False


os.makedirs(JOB["out"], exist_ok=True)
HANDLE = unreal.register_slate_post_tick_callback(tick)
log("registered")
