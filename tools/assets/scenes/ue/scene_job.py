"""UE side: one scene job in the asset scheduler's resident editor (run_asset.py --ue-job; see ../README.md).

The scheduler runs this file with JOB in its globals and calls run(JOB), a generator advanced once per editor tick.
JOB (written by scene.py; set-frame metres converted to UE centimetres there):
  {"id", "level": set level, "out": stills folder, "result": result JSON, "res": [w, h],
   "char_tools": folder of the character tools (char_render.py is loaded from there and called),
   "set_tools": folder of the set builder's UE scripts (set_build.spawn_light is called for extra lights),
   "bp_deps": [assets to load before any character Blueprint],
   "characters": {key: {"path", "garments", "body_swap", "hide_body", "rebuild_from", "garment_material",
                        "loc_cm": [x, y, z], "yaw": deg, "facing": [dx, dy] (set frame),
                        "body_pose": {"anim", "t" | "scan"} | null, "face_pose": {"curves": {...}} | {"anim", "t"} | null}},
   "cameras": [{"name", "kind": wide|full|medium|close, "subject": key,
                "position"/"target" (set frame, metres; wide) | "az", "el", "screen_x" (computed shots),
                "focal_mm", "aperture"}],
   "lights": [expanded lights (layout.light_items)], "exposure_ev100": float | null,
   "warm_frames", "settle_frames", "attach_tol_cm"}
Returns {"ok", "shots": [{name, path, t_s, camera, frame_check, attach}], "subjects", "times_s", "notes"}.
Nothing is saved: the set level is loaded, dressed for the stills and left as it was (spawned actors destroyed,
post settings restored).
"""

import importlib.util
import json
import math
import os
import sys
import time
import traceback

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
EAL = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# bones read on the posed character (MetaHuman body and face skeletons)
EYES = ("FACIAL_L_Eye", "FACIAL_R_Eye")
HEAD = "head"
FEET = ("foot_l", "foot_r", "ball_l", "ball_r")
POSE_JOINTS = ("head", "neck_01", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "hand_l", "hand_r",
               "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r")


def log(m):
    unreal.log_warning(f"SCENE: {m}")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def V(v):
    return unreal.Vector(float(v[0]), float(v[1]), float(v[2]))


def to_set(v):
    """UE centimetres -> set-frame metres (x = X, y = -Y, z = Z)."""
    return [v.x / 100.0, -v.y / 100.0, v.z / 100.0]


def to_ue(p):
    return unreal.Vector(p[0] * 100.0, -p[1] * 100.0, p[2] * 100.0)


def frames(n):
    for _ in range(int(n)):
        yield


def sock(comp, name):
    v = comp.get_socket_location(name)
    return v


def measure(key, body, face, spec, head_top_off):
    """Subject (framing.py) from the posed bones: eye line from the eye joints, head top from the face mesh's
    reference bounds carried by the head joint, chin from the jaw tip, waist from the pelvis/spine."""
    eyes = [sock(face, n) for n in EYES]
    eye = unreal.Vector((eyes[0].x + eyes[1].x) / 2, (eyes[0].y + eyes[1].y) / 2, (eyes[0].z + eyes[1].z) / 2)
    if not all(face.does_socket_exist(n) for n in EYES):
        raise RuntimeError(f"{key}: the face skeleton has no {EYES} joints")
    head = sock(body, HEAD)
    feet_z = min(sock(body, n).z for n in FEET)
    pelvis = sock(body, "pelvis")
    spine = sock(body, "spine_05")
    chin = sock(face, "FACIAL_C_Jaw")
    s_eye = to_set(eye)
    # where the head faces, in plan: from the head joint (over the neck) to the eyes' midpoint
    s_head = to_set(head)
    hf = [s_eye[0] - s_head[0], s_eye[1] - s_head[1]]
    hn = math.hypot(*hf) or 1.0
    out = {"eye_z": s_eye[2], "eye_xy": s_eye[:2], "head_fwd_xy": [hf[0] / hn, hf[1] / hn],
           "head_top_z": to_set(head)[2] + head_top_off["top_above_head_m"],
           "chin_z": s_eye[2] - head_top_off["chin_below_eye_m"],
           "waist_z": (to_set(pelvis)[2] * 0.4 + to_set(sock(body, "spine_02"))[2] * 0.6),
           "feet_z": to_set(unreal.Vector(0, 0, feet_z))[2] - head_top_off.get("sole_below_foot_m", 0.0),
           "chest_xy": to_set(spine)[:2], "pelvis_xy": to_set(pelvis)[:2], "facing": spec["facing"],
           "jaw_z": to_set(chin)[2],
           "feet_above_floor_m": to_set(unreal.Vector(0, 0, feet_z))[2] - head_top_off.get("sole_below_foot_m", 0.0)
           - spec["loc_cm"][2] / 100.0}
    out = {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)) for k, v in out.items()}
    # joints for a skeleton picture (guide pass): body joints, the head joint and the eyes, set-frame metres
    J = {n: sock(body, n) for n in POSE_JOINTS if body.does_socket_exist(n)}
    J.update({n: sock(face, n) for n in EYES})
    out["joints"] = {n: [round(c, 4) for c in to_set(v)] for n, v in J.items()}
    return out


def attach_check(body, face, tol_cm):
    """The head must sit on the neck: the head joint seen by the face skeleton and by the body skeleton agree."""
    d = {}
    for n in ("head", "neck_02", "neck_01"):
        if not (body.does_socket_exist(n) and face.does_socket_exist(n)):
            continue
        try:
            a, b = sock(body, n), sock(face, n)
            d[n] = round(math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2), 3)
        except Exception as e:
            d[n] = str(e)
    worst = max(v for v in d.values() if isinstance(v, float)) if any(isinstance(v, float) for v in d.values()) else None
    return {"dist_cm": d, "worst_cm": worst, "ok": worst is not None and worst <= tol_cm}


def cine(name, pos, target, focal, aperture, focus_m, res):
    loc, tgt = to_ue(pos), to_ue(target)
    cam = act.spawn_actor_from_class(unreal.CineCameraActor, loc, unreal.MathLibrary.find_look_at_rotation(loc, tgt))
    cam.set_actor_label(f"scene_cam_{name}")
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


def _face_from_body_anim(body, face, curves, name):
    """Face-skeleton AnimSequence holding the body's current local pose for every shared bone plus expression
    curves (mirrors the character tools' function of the same name until it is merged here)."""
    path = f"/Game/Scene/Anim/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    f = unreal.AnimSequenceFactory()
    sk = face.get_skinned_asset()
    if sk is None:
        raise RuntimeError("face component has no mesh (replaced while posing?)")
    f.set_editor_property("target_skeleton", sk.get_editor_property("skeleton"))
    anim = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, "/Game/Scene/Anim", unreal.AnimSequence, f)
    ctrl = anim.get_editor_property("controller")
    ctrl.open_bracket("face_from_body", False)
    ctrl.set_frame_rate(unreal.FrameRate(30, 1), False)
    ctrl.set_number_of_frames(unreal.FrameNumber(2), False)
    n = 0
    for i in range(face.get_num_bones()):
        b = face.get_bone_name(i)
        if body.get_bone_index(b) < 0:
            continue
        t = body.get_socket_transform(b, unreal.RelativeTransformSpace.RTS_PARENT_BONE_SPACE)
        try:
            ctrl.add_bone_curve(b, False)
        except Exception:
            ctrl.add_bone_track(b, False)
        ctrl.set_bone_track_keys(b, [t.translation] * 3, [t.rotation] * 3, [t.scale3d] * 3, False)
        n += 1
    for cname, val in (curves or {}).items():
        unreal.AnimationLibrary.add_curve(anim, cname, unreal.RawCurveTrackTypes.RCT_FLOAT, False)
        for tt in (0.0, 1.0 / 30, 2.0 / 30):
            unreal.AnimationLibrary.add_float_curve_key(anim, cname, tt, float(val))
    ctrl.close_bracket(False)
    return anim, n


def settle_attach(job, comps, limit):
    """Tick until every character's face skeleton has caught up with its body (the face follows the body a few
    frames late after a pose change: the head floats off the neck until then). Returns the ticks it took."""
    n = 0
    while n < limit:
        if all(attach_check(*comps(k)[1:], job.get("attach_tol_cm", 0.1))["ok"] for k in job["characters"]):
            return n
        n += 1
        yield
    return n


class _SpawnAt:
    """The editor actor subsystem, except that a Blueprint spawn lands at a given place and yaw."""

    def __init__(self, real, loc, rot):
        self._real, self._loc, self._rot = real, loc, rot

    def spawn_actor_from_object(self, obj, loc, rot=None):
        return self._real.spawn_actor_from_object(obj, self._loc, self._rot)

    def __getattr__(self, name):
        return getattr(self._real, name)


def plan_angle(fwd, frm, to):
    """Degrees between a plan direction and the direction from `frm` to `to` (0 = facing it, 180 = facing away)."""
    d = [to[0] - frm[0], to[1] - frm[1]]
    a = math.degrees(math.atan2(d[1], d[0]) - math.atan2(fwd[1], fwd[0]))
    return round(abs((a + 180) % 360 - 180), 1)


def truth(cs, subjects):
    """Per person, from the scene itself: the angle between where the head (and the body) faces and the camera."""
    out = {}
    for k, v in subjects.items():
        m = v["measured"]
        j = m.get("joints") or {}
        near = None
        if "upperarm_l" in j and "upperarm_r" in j:
            near = "left" if math.dist(j["upperarm_l"], cs["position"]) < math.dist(j["upperarm_r"], cs["position"]) \
                else "right"
        out[k] = {"head_to_camera_deg": plan_angle(m["head_fwd_xy"], m["eye_xy"], cs["position"]),
                  "nearer_shoulder": near,
                  "body_to_camera_deg": plan_angle(m["facing"], m["pelvis_xy"], cs["position"]),
                  "distance_m": round(math.dist(m["eye_xy"] + [m["eye_z"]], cs["position"]), 3)}
    return out


def guide_capture(cs, res, path, show_only=None):
    """The camera's linear scene depth (UE centimetres, float RGBA) through a SceneCapture2D with the still's own
    position, aim and horizontal field of view, written to `path` (.exr); show_only: only these actors drawn."""
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
            comp.set_editor_property("primitive_render_mode",
                                     unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST)
            for x in show_only:
                comp.show_only_actor_components(x, True)
        comp.capture_scene()
        yield
        yield
        # EXR keeps the float depth (the render-target export's .hdr is RGBE: 8-bit steps, banded normals)
        opt = unreal.ImageWriteOptions()
        opt.set_editor_property("format", unreal.DesiredImageFormat.EXR)
        opt.set_editor_property("overwrite_file", True)
        opt.set_editor_property("async_", False)
        opt.set_editor_property("compression_quality", 1)        # 1 = uncompressed EXR (read without OpenEXR)
        unreal.ImageWriteBlueprintLibrary.export_to_disk(rt, path, opt)
    finally:
        act.destroy_actor(a)


def find_post():
    for a in act.get_all_level_actors():
        if isinstance(a, unreal.PostProcessVolume) and a.get_actor_label() == "SetPost":
            return a
    return None


def run(job):
    t0 = time.time()
    r = {"ok": False, "id": job["id"], "shots": [], "subjects": {}, "times_s": {}, "notes": []}
    res = job.get("res", [1152, 640])
    temp = []
    post_old = None
    CR = None
    framing = _load("scene_framing", os.path.join(os.path.dirname(HERE), "framing.py"))
    try:
        os.makedirs(job["out"], exist_ok=True)
        # the character tools' render module, loaded fresh (its own globals), called for spawn / dress / pose
        CR = _load("scene_char_render", os.path.join(job["char_tools"], "char_render.py"))
        CR.JOB = {"characters": {k: dict(v) for k, v in job["characters"].items()},
                  "result": os.path.join(os.path.dirname(job["result"]), "char_render_side.json")}
        CR.RES = res
        CR.reset()
        t = time.time()
        les.load_level(job["level"])
        yield from frames(5)
        for dep in job.get("bp_deps", []):
            EAL.load_asset(dep)
        yield from frames(10)            # let the Live Link AnimBP settle before a character Blueprint loads
        r["times_s"]["load_level"] = round(time.time() - t, 2)
        les.editor_set_game_view(True)
        unreal.AutomationLibrary.set_scalability_quality_to_epic(
            unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world())
        # exposure for this scene (restored at the end: the level stays loaded in the resident editor)
        ppv = find_post()
        if ppv is not None and job.get("exposure_ev100") is not None:
            pp = ppv.get_editor_property("settings")
            post_old = (ppv, pp.get_editor_property("auto_exposure_bias"))
            pp.set_editor_property("override_auto_exposure_bias", True)
            pp.set_editor_property("auto_exposure_bias", -float(job["exposure_ev100"]))
            ppv.set_editor_property("settings", pp)
        if job.get("lights"):
            SB = _load("scene_set_build", os.path.join(job["set_tools"], "set_build.py"))
            for l in job["lights"]:
                temp.append(SB.spawn_light(dict(l, f="Scene/lights")))
        # characters: spawn (dressed by the character tools), move to their marks, pose
        t = time.time()
        real_act = CR.act
        for key, spec in job["characters"].items():
            CR.JOB["garment_material"] = spec.get("garment_material")
            # spawned on its mark: the character tools spawn at the origin, and moving a MetaHuman Blueprint
            # afterwards re-runs its construction and leaves Body/Face without meshes
            CR.act = _SpawnAt(real_act, V(spec["loc_cm"]), unreal.Rotator(0, 0, spec["yaw"]))
            try:
                CR.character_actor(key)
            finally:
                CR.act = real_act
        FG = _load("scene_figure", job["figure_tools"]) if job.get("figures") else None
        figs = {}
        for key, f in (job.get("figures") or {}).items():
            figs[key] = FG.spawn(key, f)
            temp.append(figs[key])

        def comps(key):
            a = CR.S["actors"][key][0]
            body, face = CR.skel_comps(a)
            return a, body, face

        def rewire(key):
            a, body, face = comps(key)
            spec = job["characters"][key]
            for fa in CR.S.get("followers", {}).get(id(a), []):
                fa.skeletal_mesh_component.set_leader_pose_component(body, True, False)
                fa.skeletal_mesh_component.set_forced_lod(1)
            if spec.get("body_swap") or spec.get("hide_body"):
                body.set_visibility(False, False)
                body.set_cast_shadow(False)
            for c in (body, face):
                c.set_forced_lod(1)
                c.set_update_animation_in_editor(True)
            CR.S["actors"][key] = (a, body, face)

        def meshes_ready(tag):
            for key in job["characters"]:
                a, body, face = CR.skel_comps(CR.S["actors"][key][0]) and (CR.S["actors"][key][0],) + CR.skel_comps(
                    CR.S["actors"][key][0])
                if body is None or face is None or body.get_skinned_asset() is None or face.get_skinned_asset() is None:
                    return False
            return True
        n = 0
        while not meshes_ready("spawn") and n < 600:      # the Blueprint assigns its meshes a few ticks after spawn
            n += 1
            yield
        r["times_s"]["mesh_ticks_spawn"] = n
        # the character tools' wiring (leader pose, hidden body, LOD 0); it sets editor properties on the
        # Blueprint's components, which re-runs its construction and replaces Body/Face: let that happen once,
        # then re-point the garments at the live body without touching editor properties again
        for key in job["characters"]:
            CR.refresh(key)
            # the face follows the body through its own Animation Blueprint, which by default only updates while
            # the face is rendered: off-view (or before the first camera) it keeps the last pose and the head
            # floats off the neck after a body pose change. Always tick it.
            # the same for the body: an assembled Blueprint's Body only refreshes its bones while drawn, so a pose
            # applied before a camera sees it (or on a hidden body) never reaches the bones, the face or the garments
            for i in (1, 2):             # fetched again each time: the property set can replace the components
                comps(key)[i].set_editor_property("visibility_based_anim_tick_option",
                                                  unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
        yield from frames(3)
        for key in job["characters"]:
            rewire(key)
        head_off = {}
        r["times_s"]["spawn_characters"] = round(time.time() - t, 2)
        t = time.time()
        yield from frames(job.get("warm_frames", 150))
        unreal.AutomationLibrary.finish_loading_before_screenshot()
        r["times_s"]["warm"] = round(time.time() - t, 2)
        # reference measurements on the unposed character (the Blueprint's own reference pose): head top above
        # the head joint (the face mesh's reference bounds), chin under the eye line, sole under the foot joints
        for key, spec in job["characters"].items():
            a, body, face = comps(key)
            fm = face.get_skinned_asset() if face else None
            if fm is None or body.get_skinned_asset() is None:
                dbg = [(c.get_name(), str(c.get_skinned_asset().get_name()) if c.get_skinned_asset() else None)
                       for c in a.get_components_by_class(unreal.SkeletalMeshComponent)]
                raise RuntimeError(f"{key}: the character has no body/face mesh after warm-up: {dbg}; "
                                   f"body={body.get_name() if body else None} face={face.get_name() if face else None}")
            bb = fm.get_bounds()
            head_off[key] = {"face_ref_top_cm": round(bb.origin.z + bb.box_extent.z, 2)}
            base_z = a.get_actor_location().z
            eyes = [sock(face, n) for n in EYES]
            eye_z = (eyes[0].z + eyes[1].z) / 2
            head = sock(body, HEAD)
            feet = min(sock(body, n).z for n in FEET)
            ho = head_off[key]
            ho["top_above_head_m"] = round((base_z + ho["face_ref_top_cm"] - head.z) / 100.0, 4)
            ho["chin_below_eye_m"] = round(job.get("chin_below_eye_m", 0.125), 4)
            ho["sole_below_foot_m"] = round((feet - base_z) / 100.0, 4)   # joints above the floor when standing
            ho["ref_eye_above_floor_m"] = round((eye_z - base_z) / 100.0, 4)
            r["subjects"].setdefault(key, {})["reference"] = ho
            r["subjects"][key]["reference_attach"] = attach_check(body, face, job.get("attach_tol_cm", 0.1))
        # poses built from the skeleton (pose.py), made while the body still shows its reference pose
        t = time.time()
        PO = _load("scene_pose", os.path.join(HERE, "pose.py"))
        for key, spec in job["characters"].items():
            bp = spec.get("body_pose")
            if bp and bp.get("pose"):
                a, body, face = comps(key)
                anim, info = PO.build(a, body, bp["pose"], bp.get("params"), f"/Game/Scene/Poses/AS_{job['id']}_{key}")
                bp["_anim"] = anim
                r["subjects"][key]["pose_build"] = info

        def pose_all():
            for key, spec in job["characters"].items():
                a, body, face = comps(key)
                bp = spec.get("body_pose")
                if bp and bp.get("_anim"):
                    CR.apply_anim(body, bp["_anim"], 1.0 / 30)
                elif bp:
                    anim = EAL.load_asset(bp["anim"])
                    tt = bp.get("t", 0.0)
                    if bp.get("scan"):
                        tt = CR.scan_body(anim, bp["scan"])[0]
                    bp["t"] = tt
                    bp.pop("scan", None)
                    CR.apply_anim(body, anim, tt)

        def pose_faces():
            """The face carries the body's pose: a leader-pose follower of the Body (exact; the assembled
            Blueprint's Face is only attached to it in an editor world, so it otherwise keeps its reference pose and
            the head floats off the neck). A follower ignores the face's own expression curves, so a face pose
            plays a face anim holding the body's current pose for the shared bones plus the curves (the character
            tools' face_from_body_anim when they have it)."""
            for key, spec in job["characters"].items():
                a, body, face = comps(key)
                fp = spec.get("face_pose") or {}
                if fp.get("curves"):
                    face.set_leader_pose_component(None, True, False)
                    a, body, face = comps(key)              # the call can replace the Blueprint's components
                    mk = getattr(CR, "face_from_body_anim", None) or _face_from_body_anim
                    anim, n = mk(body, face, fp["curves"], f"AS_scene_{job['id']}_{key}_face")
                    CR.apply_anim(face, anim, 1.0 / 30)
                    r["subjects"].setdefault(key, {})["face_follow"] = f"anim ({n} bones from the body)"
                elif fp.get("anim"):
                    face.set_leader_pose_component(None, True, False)
                    a, body, face = comps(key)
                    CR.apply_anim(face, EAL.load_asset(fp["anim"]), fp.get("t", 0.0))
                    r["subjects"].setdefault(key, {})["face_follow"] = "clip (head pose from the clip)"
                else:
                    CR.clear_anim(face)
                    a, body, face = comps(key)
                    face.set_leader_pose_component(body, True, False)
                    r["subjects"].setdefault(key, {})["face_follow"] = "leader"
        pose_all()
        yield from frames(10)
        pose_all()                       # a paused single-node pose shows from the second application
        yield from frames(5)
        pose_faces()
        yield from frames(3)
        # the face skeleton only follows a new body pose while it is drawn: look at each character through a
        # helper camera until the face has caught up, then measure (else the eye line is the pre-pose one)
        for key, spec in job["characters"].items():
            a, body, face = comps(key)
            pv = sock(body, "spine_05")
            fwd = [spec["facing"][0], spec["facing"][1]]
            p0 = to_set(pv)
            mc = cine("measure", [p0[0] + fwd[0] * 3, p0[1] + fwd[1] * 3, p0[2]], p0, 35, 8.0, 3.0, res)
            les.pilot_level_actor(mc)
            yield from frames(5)
            pose_all()
            yield from frames(3)
            pose_faces()
            n = yield from settle_attach(job, comps, 240)
            les.eject_pilot_level_actor()
            act.destroy_actor(mc)
            r["times_s"][f"attach_ticks_pose_{key}"] = n
        for key, spec in job["characters"].items():
            a, body, face = comps(key)
            subj = measure(key, body, face, spec, head_off[key])
            r["subjects"][key]["measured"] = subj
            r["subjects"][key]["body_pose"] = {k: v for k, v in (spec.get("body_pose") or {}).items() if k != "_anim"}
        for key, f in (job.get("figures") or {}).items():
            r["subjects"][key] = {"measured": FG.measure(figs[key], f), "figure": f.get("id") or f["mesh"]}
        r["times_s"]["pose"] = round(time.time() - t, 2)
        # cameras
        cams = []
        for c in job["cameras"]:
            subj = r["subjects"][c["subject"]]["measured"] if c.get("subject") else None
            if c["kind"] in ("close", "medium", "full") and not c.get("position"):
                cs = framing.shot(subj, c["kind"], c.get("focal_mm", 50), res, c.get("az", 0.0), c.get("el", 0.0),
                                  c.get("screen_x", 0.5))
            else:
                tgt = c["target"]
                if tgt == "subject":
                    tgt = [subj["pelvis_xy"][0], subj["pelvis_xy"][1],
                           (subj["feet_z"] + subj["head_top_z"]) / 2 + c.get("target_dz", 0.0)]
                cs = {"position": c["position"], "target": tgt, "focal_mm": c.get("focal_mm", 24)}
            chk = framing.check(cs, subj, c["kind"], res) if subj else None
            everyone = {k: framing.check(cs, v["measured"], "wide" if c["kind"] in ("wide", "full") else c["kind"], res)
                        for k, v in r["subjects"].items() if k != c.get("subject")}
            eyes = [subj["eye_xy"][0], subj["eye_xy"][1], subj["eye_z"]] if subj else cs["target"]
            focus = math.dist(cs["position"], eyes)
            ap = c.get("aperture", {"close": 2.8, "medium": 4.0}.get(c["kind"], 8.0))
            cam = cine(c["name"], cs["position"], cs["target"], cs["focal_mm"], ap, focus, res)
            temp.append(cam)
            cams.append((c, cs, chk, cam, everyone))
        # stills
        t_r = time.time()
        for c, cs, chk, cam, everyone in cams:
            ts = time.time()
            les.pilot_level_actor(cam)
            yield from frames(10)
            pose_all()
            yield from frames(3)
            pose_faces()
            yield from frames(job.get("settle_frames", 60))
            path = os.path.join(job["out"], f"{c['name']}.png")
            if os.path.exists(path):
                os.remove(path)
            for key in job["characters"]:
                rewire(key)
            shot_ticks = yield from settle_attach(job, comps, 240)
            att = {}
            for key in job["characters"]:
                a, body, face = comps(key)
                att[key] = attach_check(body, face, job.get("attach_tol_cm", 0.1))
            unreal.AutomationLibrary.take_high_res_screenshot(res[0], res[1], path, camera=cam, force_game_view=True)
            tw = time.time()
            while not (os.path.exists(path) and os.path.getsize(path) > 0):
                if time.time() - tw > 90:
                    raise RuntimeError(f"screenshot not written: {path}")
                yield
            les.eject_pilot_level_actor()
            guides = {}
            if job.get("guides"):
                # guide passes for an image model: the scene's depth, and each person's own depth (their mask is
                # where it meets the scene's: the visible part)
                g = os.path.join(job["out"], "guides")
                os.makedirs(g, exist_ok=True)
                guides["depth"] = os.path.join(g, f"{c['name']}_depth.exr")
                yield from guide_capture(cs, res, guides["depth"])
                for key in job["characters"]:
                    a_ = CR.S["actors"][key][0]
                    who = [a_] + list(CR.S.get("followers", {}).get(id(a_), []))
                    guides[f"depth_{key}"] = os.path.join(g, f"{c['name']}_depth_{key}.exr")
                    yield from guide_capture(cs, res, guides[f"depth_{key}"], who)
                for key, fa in figs.items():
                    guides[f"depth_{key}"] = os.path.join(g, f"{c['name']}_depth_{key}.exr")
                    yield from guide_capture(cs, res, guides[f"depth_{key}"], [fa])
            shot = {"name": c["name"], "path": path, "guides": guides, "truth": truth(cs, r["subjects"]), "attach_ticks": shot_ticks, "kind": c["kind"], "camera": cs, "frame_check": chk, "others_check": everyone,
                    "attach": att, "still_s": round(time.time() - ts, 2), "t_s": round(time.time() - t0, 2)}
            if not all(v["ok"] for v in att.values()):
                shot["reject"] = "head detached from the body at capture"
            r["shots"].append(shot)
            log(f"{job['id']}: still {c['name']} in {shot['still_s']} s")
        r["times_s"]["stills"] = round(time.time() - t_r, 2)
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
        if CR is not None:
            try:
                r["notes"] += CR.S["r"].get("notes", [])
                CR.cleanup()
            except Exception:
                r["notes"].append("character cleanup: " + traceback.format_exc()[-300:])
        r["times_s"]["total"] = round(time.time() - t0, 2)
        json.dump(r, open(job["result"], "w"), indent=1, default=str)
    if r.get("error"):
        raise RuntimeError(r["error"])
    return r
