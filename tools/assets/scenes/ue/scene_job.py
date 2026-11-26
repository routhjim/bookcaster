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
   "warm_frames", "settle_frames", "attach_tol_cm",
   "interior": the set is a room (hair strands then take no sky light),
   "hide": [set item names hidden for these stills (a figure that brings its own chair takes the set's chair's place)]}
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


GLOW_LIGHT = {"lumens": 250.0, "radius_m": 2.5, "over_m": 0.06}   # the light of a worked thing that glows of itself
GLOW_COLOR = "#ff6a1a"      # ... at its own top, in the colour of hot metal and embers
HEAD_THING = 1.25           # a thing worn on the head stands this many head heights (crown to chin) tall
HEAD_OVER = 0.10            # ... its crown this share of a head's height over the head's own


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
    # the trunk's own bones as the engine turned them (what cloth bound to one bone alone rides on)
    out["bone_rot"] = {}
    for n in ("pelvis", "spine_03"):
        r_ = body.get_socket_rotation(n)
        out["bone_rot"][n] = [round(r_.roll, 2), round(r_.pitch, 2), round(r_.yaw, 2)]
    return out


def creature_measure(a, f):
    """A rigged creature as a subject (framing.py): its feet at its posed bounds' bottom, its head read as a face from
    the head joint (scenes/creature.py head_measure), the frame's chest at the neck's base, its middle the bounds'."""
    CRE = _load("scene_creature", os.path.join(os.path.dirname(HERE), "creature.py"))
    c = a.skeletal_mesh_component
    o, e = a.get_actor_bounds(False)
    lo, hi = to_set(o - e), to_set(o + e)
    feet, top = min(lo[2], hi[2]), max(lo[2], hi[2])
    s_ = float(f.get("scale") or 1.0)
    jt = lambda n: to_set(c.get_socket_location(n)) if c.does_socket_exist(n) else None      # noqa: E731
    head = jt("head")
    hm = CRE.head_measure(f["position"], f["facing"], top - feet, head, jt("jaw"), float(f.get("head_len_m") or 0.5) * s_)
    mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]
    neck = jt("neck_0") or head
    out = dict(hm, feet_z=round(feet, 4), waist_z=round(feet + 0.5 * (hm["eye_z"] - feet), 4),
               chest_xy=[round(x, 4) for x in (neck or mid)[:2]], pelvis_xy=[round(x, 4) for x in mid],
               facing=list(f["facing"]), height_m=round(top - feet, 4), bounds_set=[lo, hi], creature=True,
               feet_above_floor_m=round(feet - f["position"][2], 4))
    out["joints"] = {}
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


FILL_OF_KEY = 0.5         # the camera's light on what a still is of, as a share of a full exposure's light
FILL_EV100 = 5.4          # ... for a job that names no exposure
FILL_UP_M = 0.4           # ... from just above the lens
FILL_REACH_M = (1.5, 25.0)


def camera_fill(pos, target, ev100=None):
    """The light from the camera for one still -> (where it stands, m; its lumens). A point light just above the
    lens, without shadows, sized to the distance to what the still is of so that it gets FILL_OF_KEY of the light a
    mid grey needs at the job's exposure, near or far. (Lit only by the place, a face turned from the window and a
    creature against the sky were drawn as black shapes.)"""
    d = min(max(math.dist(pos, target), FILL_REACH_M[0]), FILL_REACH_M[1])
    ev = FILL_EV100 if ev100 is None else float(ev100)
    lux = FILL_OF_KEY * math.pi * (0.125 * 2.0 ** ev) / 0.18
    return [pos[0], pos[1], pos[2] + FILL_UP_M], round(4 * math.pi * lux * d * d, 1)


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


FIX_READ_CM = 160.0          # a worked set piece's surface is read this far round the person's chest (reach + lean)
FIX_READ_MAX = 40000         # ... and thinned to at most this many points


def fixture_points(name, loc_cm, PO):
    """The surface (UE world cm) of the set piece `name` (the set builder labels each piece by its spec name; a
    repeated piece name_NN: the nearest) within FIX_READ_CM of a person standing at loc_cm: its LOD 0 sections
    filled to pose.FIX_STEP_CM (pose.surface_points) in the mesh's own frame, near the person only, then carried
    by the actor's transform. -> (points | None, what was read)."""
    sm = [a for a in act.get_all_level_actors() if isinstance(a, unreal.StaticMeshActor)]
    # (the piece's own label first: "x" must not take "x_2", another piece)
    cands = [a for a in sm if a.get_actor_label() == name] or \
        [a for a in sm if a.get_actor_label().startswith(name + "_") and a.get_actor_label()[len(name) + 1:].isdigit()
         and len(a.get_actor_label()) - len(name) - 1 == 2]
    if not cands:
        return None, f"no set piece labelled {name!r}"
    lc = V([loc_cm[0], loc_cm[1], loc_cm[2] + 100.0])
    a = min(cands, key=lambda x: (x.get_actor_location() - lc).length())
    mesh = a.static_mesh_component.get_editor_property("static_mesh")
    if mesh is None:
        return None, f"{a.get_actor_label()} has no mesh"
    tf = a.get_actor_transform()
    sc = tf.scale3d
    big, small = max(abs(sc.x), abs(sc.y), abs(sc.z)) or 1.0, min(abs(sc.x), abs(sc.y), abs(sc.z)) or 1.0
    c = tf.inverse_transform_location(lc)
    verts, tris = [], []
    for sec in range(mesh.get_num_sections(0)):
        got = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(mesh, 0, sec)
        o = len(verts)
        verts += [[v.x, v.y, v.z] for v in got[0]]
        tris += [o + int(i) for i in got[1]]
    loc = PO.surface_points(verts, tris, PO.FIX_STEP_CM / big, centre=[c.x, c.y, c.z], radius=FIX_READ_CM / small)
    loc = loc[::max(1, len(loc) // FIX_READ_MAX)]
    pts = []
    for p in loc:
        w = tf.transform_location(V(p))
        pts.append([round(w.x, 2), round(w.y, 2), round(w.z, 2)])
    return pts, f"{a.get_actor_label()}: {len(verts)} vertices, {len(pts)} surface points near"


def measured_palm(body, side, PO):
    """The palm point (pose.palm_from_joints) on the posed body's own hand joints, UE world cm; None unmeasured."""
    bones = [f"hand_{side}", f"middle_01_{side}", f"index_01_{side}", f"pinky_01_{side}"]
    if not all(body.does_socket_exist(b) for b in bones):
        return None
    jv = [sock(body, b) for b in bones]
    return PO.palm_from_joints(*[[v.x, v.y, v.z] for v in jv], side)


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
    hidden = []
    post_old = None
    sun_old = None
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
        if hasattr(CR, "interior_hair"):     # a room: hair strands not lit by the open sky (restored in cleanup)
            CR.interior_hair(job.get("interior"))
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
        for a in act.get_all_level_actors() if job.get("hide") else []:
            if a.get_actor_label() in job["hide"] and not a.is_hidden_ed():
                a.set_is_temporarily_hidden_in_editor(True)
                a.set_actor_hidden_in_game(True)
                hidden.append(a)
        r["hidden"] = [a.get_actor_label() for a in hidden]
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

        def live_face(key):
            """The face component as the actor holds it NOW: clearing its leader pose can rebuild the Blueprint's
            components, and the one fetched before is then a dead object ("ObjectInstance is null" on the next
            call, 11-11, a figure with a face pose of curves)."""
            a, body, face = comps(key)
            if face is not None and unreal.SystemLibrary.is_valid(face):
                return a, body, face
            for c in a.get_components_by_class(unreal.SkeletalMeshComponent):
                if "face" in c.get_name().lower() and unreal.SystemLibrary.is_valid(c):
                    return a, body, c
            raise RuntimeError(f"{key}: no live face component after clearing its leader pose")

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
        # every held thing is gripped where its own model says (pose.grip_from_points): its handle, which end is the
        # working end, a head or a tip (11-18: only a work action's tool was measured; a second thing used the size's
        # guess)

        def model_grip(asset_):
            cm_ = (job.get("carried") or {}).get(asset_)
            if cm_ and "grip" not in cm_:
                cm_["grip"] = None
                try:
                    mesh_ = EAL.load_asset(cm_["mesh"])
                    pts_ = []
                    for sec_ in range(mesh_.get_num_sections(0)):
                        vs_ = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(mesh_, 0, sec_)[0]
                        st_ = max(1, len(vs_) // 6000)
                        pts_ += [[vs_[n_].x / 100.0, vs_[n_].y / 100.0, vs_[n_].z / 100.0] for n_ in range(0, len(vs_), st_)]
                    cm_["grip"] = PO.grip_from_points(pts_)
                except Exception as e_:                        # the size's own convention stands (pose.grip_frame)
                    r.setdefault("notes", []).append(f"grip of {asset_} not measured: {e_}")
            return (cm_ or {}).get("grip")
        for key, spec in job["characters"].items():
            pr_ = (spec.get("body_pose") or {}).get("params") or {}
            if model_grip(pr_.get("tool_asset")):
                pr_["tool_grip"] = model_grip(pr_.get("tool_asset"))
            for h_ in pr_.get("held") or []:
                if model_grip(h_.get("asset")):
                    h_["grip"] = model_grip(h_["asset"])
        # a blow is aimed at the worked thing's top as this job lays it (below: its box's thinnest side upright, its
        # bottom on the work top), read from its mesh's own box. 11-18: the aim was reckoned from the recorded size's
        # thinnest side (27 cm) while the laid model was 19 cm thick, and the face hung 7.75 cm over the work
        for key, spec in job["characters"].items():
            pr_ = (spec.get("body_pose") or {}).get("params") or {}
            wo_ = pr_.get("work_on") or {}
            cm_ = (job.get("carried") or {}).get(wo_.get("asset"))
            if not (cm_ and wo_.get("base_ue") and pr_.get("work_target_ue")):
                continue
            try:
                bb_ = EAL.load_asset(cm_["mesh"]).get_bounds()
                top_ = PO.lie_top_ue(wo_["base_ue"], [bb_.box_extent.x, bb_.box_extent.y, bb_.box_extent.z])
                r["subjects"][key]["work_top"] = {"from_size_ue": pr_["work_target_ue"], "laid_ue": top_}
                pr_["work_target_ue"] = top_
            except Exception as e_:                        # the size's reckoning stands; the re-solve below closes it
                r["notes"].append(f"{key}: the worked thing's laid top not read ({e_})")
        # a piece of the set worked with the hands (params "fixture", scene.py): its surface round the person, for the
        # pose to find a part a hand closes round (pose.grip_parts); kept here for a re-solve, not in the result
        fix_pts = {}
        for key, spec in job["characters"].items():
            fx_ = ((spec.get("body_pose") or {}).get("params") or {}).get("fixture")
            if not fx_:
                continue
            try:
                pts_, how_ = fixture_points(fx_["name"], spec["loc_cm"], PO)
            except Exception as e_:
                pts_, how_ = None, f"surface not read: {e_}"
            r["subjects"][key]["fixture_read"] = how_
            if pts_:
                fix_pts[key] = pts_
                fx_["pts_ue"] = pts_
            else:                                      # (a failure, not silence: the hands stay off the piece)
                r["subjects"][key].setdefault("contact_missing", []).append(
                    f"the hands were not put on the {fx_['name']}: {how_}")
        for key, spec in job["characters"].items():
            bp = spec.get("body_pose")
            if bp and bp.get("pose"):
                a, body, face = comps(key)
                keep_ = {}
                if spec.get("near_to") in job["characters"] and spec["near_to"] != key:
                    # a pose that rests on the one they are with aims at that person's own body as it stands
                    bp.setdefault("params", {})["partner"] = PO.partner_points(a, comps(spec["near_to"])[1])
                anim, info = PO.build(a, body, bp["pose"], bp.get("params"), f"/Game/Scene/Poses/AS_{job['id']}_{key}",
                                      keep=keep_)
                bp["_anim"] = anim
                bp["_ref"] = keep_.get("ref")          # the reference skeleton, for a re-solve on the posed body
                r["subjects"][key]["pose_build"] = info
                if (info.get("hold") or {}).get("partner") == "like self":
                    # (a failure, not silence: a pose held on another person was aimed with nobody there)
                    r["subjects"][key].setdefault("contact_missing", []).append(
                        f"the {bp['pose']} was aimed at nobody: the job names no one this person is with")
                fx_ = (bp.get("params") or {}).get("fixture")
                if fx_ and "pts_ue" in fx_:
                    fx_["points_read"] = len(fx_.pop("pts_ue"))

        def pose_all():
            for key, f in (job.get("figures") or {}).items():     # a rigged figure holds its clip's frame
                if f.get("anim"):
                    FG.pose(figs[key], f)
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
                    # the face anim is made ONCE, before the leader is cleared, and reused by the later calls: made
                    # again it deleted the AnimSequence the face was still playing (same asset path), and the next
                    # call on the face failed with "ObjectInstance is null" (11-11, every character)
                    if spec.get("_face_anim") is None:
                        mk = getattr(CR, "face_from_body_anim", None) or _face_from_body_anim
                        spec["_face_anim"] = mk(body, face, fp["curves"],
                                                f"AS_scene_{job['id']}_{key}_face{spec.get('_face_tag', '')}")
                    anim, n = spec["_face_anim"]
                    face.set_leader_pose_component(None, True, False)
                    a, body, face = live_face(key)          # the call can replace the Blueprint's components
                    CR.apply_anim(face, anim, 1.0 / 30)
                    r["subjects"].setdefault(key, {})["face_follow"] = f"anim ({n} bones from the body)"
                elif fp.get("anim"):
                    face.set_leader_pose_component(None, True, False)
                    a, body, face = live_face(key)
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
        # a child: the whole posed figure scaled about its floor point, then lifted by what the scaling took from
        # the height of the seat or bed under it (a seated child's feet hang; they do not sink into the seat)
        for key, spec in job["characters"].items():
            k_ = float(spec.get("scale") or 1.0)
            if abs(k_ - 1.0) > 0.01:
                a = comps(key)[0]
                under = float((spec.get("_lie") or {}).get("top_z_m")
                              or ((spec.get("body_pose") or {}).get("params") or {}).get("seat_z_m") or 0.0)
                a.set_actor_scale3d(unreal.Vector(k_, k_, k_))
                loc = a.get_actor_location()
                a.set_actor_location(unreal.Vector(loc.x, loc.y, loc.z + under * (1.0 - k_) * 100.0), False, False)
        yield from frames(5)
        for key, spec in job["characters"].items():
            a, body, face = comps(key)
            subj = measure(key, body, face, spec, head_off[key])
            r["subjects"][key]["measured"] = subj
            fb = {}                                  # where each piece that follows the body really is (set frame)
            for f_ in [a] + list(CR.S.get("followers", {}).get(id(a), [])):
                for c_ in f_.get_components_by_class(unreal.SkeletalMeshComponent):
                    m_ = c_.get_skinned_asset()
                    if m_ is None:
                        continue
                    o_, e_ = unreal.SystemLibrary.get_component_bounds(c_)[:2]
                    fb[m_.get_name()] = {"centre": [round(v, 3) for v in to_set(o_)],
                                         "half_m": [round(e_.x / 100, 3), round(e_.y / 100, 3), round(e_.z / 100, 3)]}
            r["subjects"][key]["piece_bounds"] = fb
            r["subjects"][key]["body_pose"] = {k: v for k, v in (spec.get("body_pose") or {}).items()
                                               if not k.startswith("_")}
        # a lying person (the spec's "_lie": the bed's head and foot points) is read by the cameras from the bed's
        # side nearer whoever is with them
        for key, spec in job["characters"].items():
            if spec.get("_lie"):
                oth = [v["measured"]["pelvis_xy"] for k, v in r["subjects"].items()
                       if k != key and k in job["characters"] and not job["characters"][k].get("_lie")]
                r["subjects"][key]["measured"] = framing.lay(r["subjects"][key]["measured"], spec["_lie"],
                                                             oth[0] if oth else None)
        for key, spec in job["characters"].items():
            if spec.get("_inside"):
                r["subjects"][key]["measured"]["inside"] = True
                r["subjects"][key]["measured"]["open_walls"] = spec["_inside"] == "open"

        def repose(key, why_):
            """A measured miss re-solved: the person's pose built again from their reference skeleton with the
            corrected aims now in their params, played, settled (the face caught up through a helper camera, as for
            the first pose) and measured again. The cameras keep the framing of the first measure."""
            spec = job["characters"][key]
            bp = spec["body_pose"]
            n_ = len(bp.setdefault("_resolves", [])) + 1
            bp["_resolves"].append(list(why_ or []))
            pr_ = dict(bp.get("params") or {})
            if key in fix_pts and pr_.get("fixture"):
                pr_["fixture"] = dict(pr_["fixture"], pts_ue=fix_pts[key])
            a, body, face = comps(key)
            anim, info = PO.build(a, body, bp["pose"], pr_, f"/Game/Scene/Poses/AS_{job['id']}_{key}_re{n_}",
                                  ref=bp.get("_ref"))
            bp["_anim"] = anim
            r["subjects"][key]["pose_build"] = info
            if spec.get("_face_anim") is not None:   # the face anim holds the body's old pose: made again, a new name
                spec["_face_anim"] = None
                spec["_face_tag"] = f"_re{n_}"
            pose_all()
            yield from frames(10)
            pose_all()
            yield from frames(5)
            pose_faces()
            yield from frames(3)
            pv = sock(comps(key)[1], "spine_05")
            p0 = to_set(pv)
            mc = cine("measure", [p0[0] + spec["facing"][0] * 3, p0[1] + spec["facing"][1] * 3, p0[2]], p0, 35, 8.0,
                      3.0, res)
            les.pilot_level_actor(mc)
            yield from frames(5)
            pose_all()
            yield from frames(3)
            pose_faces()
            n = yield from settle_attach(job, comps, 240)
            les.eject_pilot_level_actor()
            act.destroy_actor(mc)
            a, body, face = comps(key)
            old_ = r["subjects"][key].get("measured") or {}
            new_ = measure(key, body, face, spec, head_off[key])
            for k_ in ("inside", "open_walls"):
                if k_ in old_:
                    new_[k_] = old_[k_]
            r["subjects"][key]["measured"] = new_
            r["subjects"][key].setdefault("resolves", []).append({"why": list(why_ or []), "attach_ticks": n})

        def fixture_measure(key):
            """Each hand on a piece's part as the engine posed it: the palm on the body's own hand joints against the
            part's line (pose.fixture_contact), stored beside the held things' contacts; None with no piece."""
            fp_ = (r["subjects"][key].get("pose_build") or {}).get("fixture")
            if not fp_:
                return None
            body_ = comps(key)[1]
            out_ = {"name": fp_.get("name"), "points": fp_.get("points"), "parts": fp_.get("parts"), "hands": {},
                    "why": []}
            if not fp_.get("hands"):
                out_["note"] = "no part of it a free hand can take hold of within reach"
            for x_, h_ in (fp_.get("hands") or {}).items():
                palm_ = measured_palm(body_, x_, PO) or h_.get("palm_ue")
                c_ = PO.fixture_contact(palm_, h_["a_ue"], h_["b_ue"], fp_.get("name"), x_)
                c_.update(palm_ue=[round(v, 2) for v in palm_], a_ue=h_["a_ue"], b_ue=h_["b_ue"], on_ue=h_.get("on_ue"))
                out_["hands"][x_] = c_
                out_["why"] += c_["why"]
            r["subjects"][key]["fixture"] = out_
            return out_
        # each hand on a piece's part measured; a palm off its part is solved again by the measured miss across the
        # part (pose.fixture_fix), at most RESOLVE_TRIES times, before it is a failure
        for key, spec in job["characters"].items():
            for n_ in range(PO.RESOLVE_TRIES + 1):
                fc_ = fixture_measure(key)
                if not fc_ or not fc_["why"] or n_ == PO.RESOLVE_TRIES:
                    break
                fu_ = spec["body_pose"]["params"]["fixture"].setdefault("fix_ue", {})
                for x_, hc_ in fc_["hands"].items():
                    if hc_["why"]:
                        fu_[x_] = PO.fixture_fix(fu_.get(x_), hc_["palm_ue"], hc_["a_ue"], hc_["b_ue"])
                yield from repose(key, fc_["why"])
        for key, f in (job.get("figures") or {}).items():
            m_ = creature_measure(figs[key], f) if f.get("creature") else FG.measure(figs[key], f)
            r["subjects"][key] = {"measured": m_, "figure": f.get("id") or f["mesh"]}
        r["times_s"]["pose"] = round(time.time() - t, 2)
        # cameras
        cams = []
        for c in job["cameras"]:
            subj = r["subjects"][c["subject"]]["measured"] if c.get("subject") else None
            if c["kind"] in ("close", "medium", "full") and not c.get("position"):
                # inside a cabin an action is at the controls: it is seen from behind the shoulder (the cabin's
                # "full" view), the quiet shots from in front of the face. In a cabin whose walls a camera sees
                # through, an action the director framed close is a close on the face (a man in tears was shown
                # as the back of a head, 11-15)
                k_ = "full" if (subj or {}).get("inside") and c.get("shot_kind") == "action" and not \
                    ((subj or {}).get("open_walls") and c["kind"] == "close") else c["kind"]
                cs = framing.shot(subj, k_, c.get("focal_mm", 50), res, c.get("az", 0.0), c.get("el", 0.0),
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
            # (a view outside is part of the picture: stopped down, it stays readable behind the face)
            ap = c.get("aperture", 11.0 if c.get("view") else {"close": 2.8, "medium": 4.0}.get(c["kind"], 8.0))
            cam = cine(c["name"], cs["position"], cs["target"], cs["focal_mm"], ap, focus, res)
            temp.append(cam)
            cams.append((c, cs, chk, cam, everyone))
        # a place in the open (scene.py "key_light"): its sun turned for each still to come from the camera's side
        # (framing.key_toward), at the job's strength; put back as it was at the end
        kl, sun_a = job.get("key_light"), None
        if kl:
            sun_a = next((a for a in act.get_all_level_actors() if isinstance(a, unreal.DirectionalLight)
                          and a.get_actor_label() == "Sun"), None)
            if sun_a is not None:
                sc_ = sun_a.get_component_by_class(unreal.DirectionalLightComponent)
                sun_old = (sun_a, sun_a.get_actor_rotation(), sc_.get_editor_property("intensity"))
                sc_.set_editor_property("intensity", float(kl["lux"]))
            else:
                r["notes"].append("key_light: the level has no Sun")
        # stills
        fill_a = None                    # the light from the camera (one, moved for each still)
        glow_lights = {}                 # (person, library prop) -> the light of a worked thing that glows
        worn_actors = {}                 # (person, library prop) -> its actor: what people wear on the head or carry
        on_top_, resolves_ = {}, {}      # this still's worked things' tops by person; re-solves so far by person

        def hold_in_hand(key, t_, a_):
            """A thing in a hand (pose.py reports every one it gripped: info["held"], by asset; the older info["tool"]
            for a work tool): held in the palm of its hand, the handle along the direction the pose gives, its face
            where the pose turned it, its grip point in the palm. The palm is the pose's own palm point carried by the
            measured wrist joint. (11-18: anything but a work tool hung below the wrist; with a grip reported that
            fallback is not reached.) -> (placement, the pose's report) when gripped, () when hung under the wrist,
            None when the hand is not measured."""
            cm_ = job["carried"][t_["asset"]]
            sz_ = cm_["size_m"]
            m_ = r["subjects"][key]["measured"]
            pb_ = r["subjects"][key].get("pose_build") or {}
            tl_ = (pb_.get("held") or {}).get(t_["asset"])
            if not tl_ and not pb_.get("held"):
                tl_ = pb_.get("tool")
            j_ = (m_.get("joints") or {}).get(f"hand_{(tl_ or {}).get('hand') or t_.get('hand') or 'r'}")
            if not j_:
                return None
            a_.set_actor_scale3d(unreal.Vector(1.0, 1.0, 1.0))
            if not (tl_ and tl_.get("palm_ue")):
                f_ = m_["facing"]
                a_.set_actor_location(to_ue([j_[0], j_[1], j_[2] - sz_[2] - 0.02]), False, False)
                a_.set_actor_rotation(unreal.Rotator(0, 0, -(math.degrees(math.atan2(f_[1], f_[0])) + 90.0)), False)
                return ()
            w_ = to_ue(j_)
            palm_ = [w_.x + tl_["palm_ue"][0] - tl_["hand_ue"][0], w_.y + tl_["palm_ue"][1] - tl_["hand_ue"][1],
                     w_.z + tl_["palm_ue"][2] - tl_["hand_ue"][2]]
            grip_ = PO.prop_in_hand(sz_, palm_, tl_["dir_ue"], tl_["face_ue"], grip=tl_.get("grip") or cm_.get("grip"))
            a_.set_actor_rotation(unreal.MathLibrary.make_rot_from_xz(V(grip_["x"]), V(grip_["z"])), False)
            a_.set_actor_location(V(grip_["origin_cm"]), False, False)
            r["subjects"][key].setdefault("held", {})[t_["asset"]] = {
                "grip": grip_["grip"], "origin_ue": [round(v, 2) for v in grip_["origin_cm"]],
                "palm_ue": [round(v, 2) for v in palm_], "hand": tl_.get("hand"), "how": tl_.get("how"),
                "name": t_.get("name") or tl_.get("name")}
            return grip_, tl_

        def held_measure(key, t_, grip_, tl_):
            """The staged check of a held thing (pose.held_contact, the old blockout check's numbers): the palm
            measured on the posed body's own hand joints against the placed handle's line, and for a blow the working
            point against the worked thing's placed top, the face against straight down (a raised thing: how high
            over that top). Over the limits its words reach the staging's facts (speakers.staged_facts)."""
            hs_ = tl_.get("hand") or "r"
            hm_ = r["subjects"][key]["held"][t_["asset"]]
            palm_m_ = measured_palm(comps(key)[1], hs_, PO)
            if palm_m_:
                hm_["palm_measured_ue"] = [round(v, 2) for v in palm_m_]
            else:
                palm_m_ = hm_["palm_ue"]
            tgt_ = (on_top_.get(key) or tl_["target_ue"]) if tl_.get("target_ue") else None
            hm_["contact"] = PO.held_contact(grip_, palm_m_, tgt_, [0.0, 0.0, 1.0] if tgt_ else None, hm_.get("name"),
                                             raised=bool(tl_.get("raised")))
            return hm_["contact"]
        t_r = time.time()
        for c, cs, chk, cam, everyone in cams:
            ts = time.time()
            # a camera beyond a wall (a small room's wild wall): that wall is hidden for this still and its guides
            wild = []
            if job.get("room"):
                W_, D_ = job["room"][0], job["room"][1]
                px, py = cs["position"][0], cs["position"][1]
                sides = [s_ for s_, out_ in (("east", px > W_ / 2), ("west", px < -W_ / 2), ("north", py > D_ / 2),
                                             ("south", py < -D_ / 2)) if out_]
                for a_ in act.get_all_level_actors() if sides else []:
                    fp = str(a_.get_folder_path())
                    if any(fp == f"Walls/{s_}" or fp.startswith(f"Walls/{s_}/") for s_ in sides) and not a_.is_hidden_ed():
                        a_.set_is_temporarily_hidden_in_editor(True)
                        a_.set_actor_hidden_in_game(True)
                        wild.append(a_)
            # a cabin's view outside: this shot's backdrop shown, the other views' hidden for this still
            vw_ = c.get("view") or {}
            moved = []
            for a_ in act.get_all_level_actors() if vw_.get("all") else []:
                lb = a_.get_actor_label()
                if lb in vw_["all"] and lb != vw_.get("name") and not a_.is_hidden_ed():
                    a_.set_is_temporarily_hidden_in_editor(True)
                    a_.set_actor_hidden_in_game(True)
                    wild.append(a_)
                elif lb == vw_.get("name") and vw_.get("ahead"):
                    # the shot's view stands where this camera looks, beyond whoever it frames, as a painted
                    # backdrop is hung behind the actors for each set-up: whichever pane is behind them shows it
                    # (standing only ahead of the nose, every camera looking back or across saw the far dark)
                    d_ = [cs["target"][0] - cs["position"][0], cs["target"][1] - cs["position"][1]]
                    n_ = math.hypot(*d_)
                    if n_ > 1e-6:
                        d_ = [d_[0] / n_, d_[1] / n_]
                        l0, r0 = a_.get_actor_location(), a_.get_actor_rotation()
                        moved.append((a_, l0, r0))
                        # ... as far off as lets the lens take in most of the picture's width (near, a long lens
                        # saw a quarter of it, blown up to a smear the redraw painted as a wall)
                        hf_ = framing.hfov(cs["focal_mm"])
                        far_ = max(float(vw_.get("beyond_m") or 14.0),
                                   float(vw_.get("width_m") or 48.0) / 2 / math.tan(min(1.4, hf_ * 0.65)))
                        at_ = [cs["target"][0] + d_[0] * far_, cs["target"][1] + d_[1] * far_]
                        turn = math.degrees(math.atan2(d_[1], d_[0]) - math.atan2(vw_["ahead"][1], vw_["ahead"][0]))
                        a_.set_actor_location(unreal.Vector(at_[0] * 100.0, -at_[1] * 100.0, l0.z), False, False)
                        a_.set_actor_rotation(unreal.Rotator(r0.roll, r0.pitch, r0.yaw - turn), False)
            # what the people of this shot wear on the head or carry in a hand (from_plan's "carried"): each made
            # object set on the head, sized to it, or gripped in its hand; those of other shots hidden
            shown, glow_on = set(), set()

            def glow_at(tag, at):
                """The light of a thing that glows of itself in this shot (the plan's "glow"), at `at` (UE cm)."""
                try:
                    if tag not in glow_lights:
                        SBg_ = _load("scene_set_build", os.path.join(job["set_tools"], "set_build.py"))
                        glow_lights[tag] = SBg_.spawn_light({
                            "type": "point", "c": [0, 0, 0], "lumens": GLOW_LIGHT["lumens"],
                            "radius_m": GLOW_LIGHT["radius_m"], "color": GLOW_COLOR, "shadows": False,
                            "f": "Scene/lights"})
                        temp.append(glow_lights[tag])
                    glow_lights[tag].set_actor_location(at, False, False)
                    glow_on.add(tag)
                except Exception as e_g:
                    r["notes"].append(f"glow light: {e_g}")
            on_top_, in_hand_ = {}, []
            # (a worked thing lies first, so a blow's check finds its top)
            for key, things_ in (c.get("carried") or {}).items():
                things_ = sorted(things_, key=lambda t_: t_["at"] != "on")
                m_ = (r["subjects"].get(key) or {}).get("measured")
                if not m_ or m_.get("lying"):
                    continue
                for t_ in things_:
                    cm_ = (job.get("carried") or {}).get(t_["asset"])
                    tag = (key, t_["asset"])
                    if cm_ and tag not in worn_actors:
                        mesh_ = EAL.load_asset(cm_["mesh"])
                        if mesh_ is not None:
                            worn_actors[tag] = act.spawn_actor_from_object(mesh_, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
                            worn_actors[tag].set_actor_label(f"carried_{key}_{len(worn_actors)}")
                            temp.append(worn_actors[tag])
                    if tag not in worn_actors:
                        continue
                    sz_ = cm_["size_m"]
                    if t_["at"] == "on":         # the thing being worked on, lying on the top of what they work at
                        f_, k_ = m_["facing"], 1.0
                        xy_, z_ = t_["xyz"][:2], t_["xyz"][2]
                    elif t_["at"] == "head":
                        f_ = m_.get("head_fwd_xy") or m_["facing"]
                        n_ = math.hypot(f_[0], f_[1]) or 1.0
                        f_ = [f_[0] / n_, f_[1] / n_]
                        hh_ = m_["head_top_z"] - m_["chin_z"]
                        k_ = HEAD_THING * hh_ / max(sz_[2], 1e-3)
                        # (the eyes stand at the face's front: the head's middle is half a head's depth behind them)
                        xy_ = [m_["eye_xy"][0] - f_[0] * 0.42 * hh_, m_["eye_xy"][1] - f_[1] * 0.42 * hh_]
                        z_ = m_["head_top_z"] + HEAD_OVER * hh_ - sz_[2] * k_
                    else:                        # a thing in a hand: gripped as the pose holds it
                        got_ = hold_in_hand(key, t_, worn_actors[tag])
                        if got_ is None:
                            continue
                        if got_:
                            in_hand_.append((key, t_) + got_)
                        shown.add(tag)
                        if t_.get("glow"):       # (a held thing hot from a fire: its light at its own middle)
                            glow_at(tag, worn_actors[tag].get_actor_bounds(False)[0])
                        continue
                    a_ = worn_actors[tag]
                    a_.set_actor_scale3d(unreal.Vector(k_, k_, k_))
                    a_.set_actor_location(to_ue([xy_[0], xy_[1], z_]), False, False)
                    yaw_ = -(math.degrees(math.atan2(f_[1], f_[0])) + 90.0)
                    if t_["at"] == "on":
                        # a thing on a work top lies on its broadest face, not on end (a bar's model came out
                        # upright: a slab standing on the anvil taller than the anvil), its middle over the top's
                        # (the model's own measured box says which way is thin: the job's size is only its longest side)
                        a_.set_actor_rotation(unreal.Rotator(0, 0, 0), False)
                        _, e0_ = a_.get_actor_bounds(False)
                        thin_ = min(range(3), key=lambda i_: (e0_.x, e0_.y, e0_.z)[i_])
                        a_.set_actor_rotation(unreal.Rotator(90.0 if thin_ == 1 else 0.0, 90.0 if thin_ == 0 else 0.0, yaw_), False)
                        o_, e_ = a_.get_actor_bounds(False)
                        want_ = to_ue([xy_[0], xy_[1], z_])
                        l_ = a_.get_actor_location()
                        a_.set_actor_location(unreal.Vector(l_.x + want_.x - o_.x, l_.y + want_.y - o_.y,
                                                            l_.z + want_.z - (o_.z - e_.z)), False, False)
                        o_, e_ = a_.get_actor_bounds(False)
                        on_top_[key] = [o_.x, o_.y, o_.z + e_.z]       # the worked thing's top: the blow's point
                        # a worked thing that glows of itself in this shot (the plan's "glow": metal hot from the
                        # fire) lights what is round it: unlit, it was a dark lump and the key drew cold iron (11-18)
                        if t_.get("glow"):
                            glow_at(tag, unreal.Vector(o_.x, o_.y, o_.z + e_.z + GLOW_LIGHT["over_m"] * 100.0))
                    else:
                        a_.set_actor_rotation(unreal.Rotator(0, 0, yaw_), False)
                    shown.add(tag)
            # the staged check of each held thing (held_measure); a miss a re-solve can close (pose.needs_resolve:
            # the working point off the measured target, the face turned off it) is not kept: the arm is solved again
            # to the measured target, aimed by the miss (pose.aim_fix), the things placed and measured again, at most
            # RESOLVE_TRIES times per person; only a miss that outlasts them is the shot's failure (11-18: a face
            # 7.75 cm over the work was flagged in words and the frame kept)
            for key, t_, grip_, tl_ in in_hand_:
                held_measure(key, t_, grip_, tl_)
            for key in sorted({x_[0] for x_ in in_hand_}):
                while resolves_.get(key, 0) < PO.RESOLVE_TRIES:
                    bad_ = [r["subjects"][key]["held"][x_[1]["asset"]].get("contact") for x_ in in_hand_
                            if x_[0] == key and PO.needs_resolve(r["subjects"][key]["held"][x_[1]["asset"]].get("contact"))]
                    if not bad_ or not (bad_[0].get("target") and bad_[0].get("work_pt")):
                        break
                    pr_ = job["characters"][key]["body_pose"]["params"]
                    aim0_ = pr_.get("work_target_ue") or bad_[0]["target"]
                    aim_ = PO.aim_fix(aim0_, bad_[0]["target"], bad_[0]["work_pt"])
                    if math.dist(aim_, aim0_) < 0.5:         # (a face turned off with the point on: the same solve)
                        break
                    pr_["work_target_ue"] = aim_
                    resolves_[key] = resolves_.get(key, 0) + 1
                    yield from repose(key, bad_[0]["why"])
                    r["subjects"][key]["resolves"][-1]["aim_ue"] = aim_
                    again_ = []
                    for x_ in in_hand_:
                        if x_[0] != key:
                            again_.append(x_)
                            continue
                        got_ = hold_in_hand(key, x_[1], worn_actors[(key, x_[1]["asset"])])
                        if got_:
                            again_.append((key, x_[1]) + got_)
                            held_measure(key, x_[1], *got_)
                    in_hand_ = again_
            for tag, a_ in worn_actors.items():
                a_.set_is_temporarily_hidden_in_editor(tag not in shown)
                a_.set_actor_hidden_in_game(tag not in shown)
            for tag, l_ in glow_lights.items():
                on_ = tag in shown and tag in glow_on
                l_.set_is_temporarily_hidden_in_editor(not on_)
                l_.set_actor_hidden_in_game(not on_)
                lc_ = l_.get_component_by_class(unreal.PointLightComponent)
                if lc_:
                    lc_.set_visibility(on_)
            if sun_a is not None:
                sm_ = r["subjects"][c["subject"]]["measured"] if c.get("subject") else None
                eye_ = [sm_["eye_xy"][0], sm_["eye_xy"][1], sm_["eye_z"]] if sm_ else cs["target"]
                k_ = framing.key_toward(cs["position"], eye_, kl["toward_sun"], flip=bool(c.get("key_flip")))
                # (UE: y flipped; a directional light points along its x, away from the light)
                sun_a.set_actor_rotation(unreal.MathLibrary.make_rot_from_x(V([-k_[0], k_[1], -k_[2]])), False)
            # a light from the camera on every still (camera_fill): faces and things read whatever the place's own
            # light does (a beast against the sky and a man in a doorway were black shapes in the keys, 11-20)
            try:
                if fill_a is None:
                    SBf_ = _load("scene_set_build", os.path.join(job["set_tools"], "set_build.py"))
                    fill_a = SBf_.spawn_light({"type": "point", "c": [0, 0, 0], "lumens": 1.0, "radius_m": 200,
                                               "kelvin": 5000, "shadows": False, "source_radius_m": 0.3,
                                               "f": "Scene/lights"})
                    temp.append(fill_a)
                fp_, lm_ = camera_fill(cs["position"], cs["target"], job.get("exposure_ev100"))
                fill_a.set_actor_location(to_ue(fp_), False, False)
                fill_a.get_component_by_class(unreal.PointLightComponent).set_editor_property("intensity", lm_)
            except Exception as e_f:
                r["notes"].append(f"camera fill: {e_f}")
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
            for a_ in wild:
                a_.set_is_temporarily_hidden_in_editor(False)
                a_.set_actor_hidden_in_game(False)
            for a_, l0, r0 in moved:
                a_.set_actor_location(l0, False, False)
                a_.set_actor_rotation(r0, False)
            shot = {"name": c["name"], "path": path, "guides": guides, "truth": truth(cs, r["subjects"]), "attach_ticks": shot_ticks, "kind": c["kind"], "camera": cs, "frame_check": chk, "others_check": everyone,
                    "attach": att, "still_s": round(time.time() - ts, 2), "t_s": round(time.time() - t0, 2)}
            if not all(v["ok"] for v in att.values()):
                shot["reject"] = "head detached from the body at capture"
            r["shots"].append(shot)
            log(f"{job['id']}: still {c['name']} in {shot['still_s']} s")
        r["times_s"]["stills"] = round(time.time() - t_r, 2)
        # a blow at a worked thing that ended with nothing measured against the work is a failure, not silence
        # (pose.blow_contact_why; 11-18: the raised blows reported no target and nothing said so)
        for key, spec in job["characters"].items():
            pr_ = (spec.get("body_pose") or {}).get("params") or {}
            hh_ = [h_ for h_ in ((r["subjects"].get(key) or {}).get("held") or {}).values() if h_.get("how") == "work"]
            why_ = PO.blow_contact_why(pr_.get("action"), bool(pr_.get("work_target_ue")),
                                       [h_.get("contact") for h_ in hh_], hh_[0].get("name") if hh_ else None)
            if why_:
                r["subjects"].setdefault(key, {}).setdefault("contact_missing", []).extend(why_)
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
        for a in hidden:
            try:
                a.set_is_temporarily_hidden_in_editor(False)
                a.set_actor_hidden_in_game(False)
            except Exception:
                pass
        if sun_old:
            try:
                sun_old[0].set_actor_rotation(sun_old[1], False)
                sun_old[0].get_component_by_class(unreal.DirectionalLightComponent).set_editor_property(
                    "intensity", sun_old[2])
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
