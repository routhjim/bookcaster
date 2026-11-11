"""UE side, GPU: render a MetaHuman character, optionally dressed in skinned garments, posed, at fixed framings.

Run by the MetaHuman lab's start-up shim inside a full offscreen editor started by
    python tools/assets/metahuman/mh_run.py render job.json        (job["script"] = this file)

Job (env MH_JOB):
  {"script", "map", "out", "result", "res": [1152, 640], "warm_frames": 150, "settle_frames": 40,
   "backdrop": true,                              (grey cyclorama instead of the template sky)
   "characters": {"<key>": {"path": "/Game/.../Character" or "/Game/.../BP_X" (assembled Blueprint),
                            "garments": ["/Game/.../SK_Garment", ...],   (follow the body via leader pose)
                            "body_swap": "/Game/.../SK_BodyTrim" | null,   (replaces the visible body: a copy
                                                                            of it with the covered parts cut)
                            "hide_body": false}},
   "export_geometry": {"key": "<character key>", "folder": "/Game/Char/Geo", "fbx_dir": dir} | null,
   "shots": [{"name", "char": key, "frame": "close" | "medium" | "full", "az": deg, "el": deg,
              "body_anim": path | null, "body_t": s, "body_scan": null | "arms_up",
              "face_curves": {curve: value} | null, "face_anim": path | null, "face_t": s}]}

A Character asset renders through the editor preview actor (spawn_meta_human_actor; no assembly needed); an
assembled Blueprint renders as an ordinary actor. Garments are SkeletalMeshComponents added to the actor with
set_leader_pose_component(body), so they follow every body pose; body_swap hides the body component and shows the
trimmed copy (with the body's own materials) the same way. export_geometry (needs Slate, so the full editor, not a
commandlet) writes the character's body/head skeletal meshes as assets and FBX files.
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
S = {"n": 0, "busy": False, "t0": time.time(), "phase": "boot", "f_phase": 0, "i": 0, "actors": {}, "scene": [],
     "r": {"ok": False, "shots": [], "notes": []}, "edit": []}


def log(m):
    unreal.log_warning(f"CHAR render: {m}")


def note(m):
    S["r"]["notes"].append(m)
    log(m)


def spawn(cls, loc, rot=(0, 0, 0), label=None):
    a = act.spawn_actor_from_class(cls, unreal.Vector(*loc), unreal.Rotator(roll=rot[0], pitch=rot[1], yaw=rot[2]))
    if label:
        a.set_actor_label(label)
    S["scene"].append(a)
    return a


def rect(label, loc, rot, cd, size):
    a = spawn(unreal.RectLight, loc, rot, label)
    c = a.get_component_by_class(unreal.RectLightComponent)
    c.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
    c.set_editor_property("intensity", float(cd))
    c.set_editor_property("source_width", float(size))
    c.set_editor_property("source_height", float(size))
    c.set_editor_property("attenuation_radius", 2000.0)
    return a


def light_scene():
    """Neutral three-point portrait light (front-left key, front-right fill, back rim) on the template level."""
    rect("ch_key", (120, 170, 200), (0, -15, -125), JOB.get("key_cd", 40), 140)
    rect("ch_fill", (-190, 210, 160), (0, -5, -50), JOB.get("fill_cd", 14), 220)
    rect("ch_rim", (-60, -210, 230), (0, -25, 75), JOB.get("rim_cd", 25), 120)
    if JOB.get("backdrop", True):
        # a large matte grey card behind the figure (the camera looks along -Y from +Y), so stills read as a studio
        card = spawn(unreal.StaticMeshActor, (0, -260, 0), label="ch_backdrop")
        smc = card.static_mesh_component
        smc.set_static_mesh(EAL.load_asset("/Engine/BasicShapes/Plane"))
        card.set_actor_rotation(unreal.Rotator(roll=90, pitch=0, yaw=0), False)
        card.set_actor_scale3d(unreal.Vector(12, 6, 1))
        card.set_actor_location(unreal.Vector(0, -260, 250), False, False)
        mat = EAL.load_asset("/Engine/BasicShapes/BasicShapeMaterial")
        if mat:
            mi = smc.create_dynamic_material_instance(0, mat)
            mi.set_vector_parameter_value("Color", unreal.LinearColor(0.32, 0.32, 0.31, 1))
    ppv = spawn(unreal.PostProcessVolume, (0, 0, 0), label="ch_pp")
    ppv.set_editor_property("unbound", True)
    pp = ppv.get_editor_property("settings")
    if JOB.get("exposure_bias") is not None:
        pp.set_editor_property("override_auto_exposure_bias", True)
        pp.set_editor_property("auto_exposure_bias", float(JOB["exposure_bias"]))
    ppv.set_editor_property("settings", pp)


def skel_comps(a):
    """(body, face) skeletal mesh components of a preview actor or an assembled Blueprint actor."""
    try:
        return a.get_editor_property("body_component"), a.get_editor_property("face_component")
    except Exception:
        pass
    body = face = None
    for c in a.get_components_by_class(unreal.SkeletalMeshComponent):
        n = c.get_name().lower()
        if n == "body" or (body is None and "body" in n):
            body = c
        if n == "face" or (face is None and "face" in n):
            face = c
    return body, face


def add_follower(a, body, mesh_path, label, materials_from=None):
    mesh = EAL.load_asset(mesh_path)
    if mesh is None:
        raise RuntimeError(f"no skeletal mesh {mesh_path}")
    try:
        comp = a.add_component_by_class(unreal.SkeletalMeshComponent, True, unreal.Transform(), False)
    except Exception as e:
        # fall back to the subobject system (registers the component on the actor)
        note(f"add_component_by_class: {e}")
        sub = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
        root = sub.k2_gather_subobject_data_for_instance(a)[0]
        h, fail = sub.add_new_subobject(unreal.AddNewSubobjectParams(parent_handle=root,
                                                                       new_class=unreal.SkeletalMeshComponent))
        comp = unreal.SubobjectDataBlueprintFunctionLibrary.get_object(
            unreal.SubobjectDataBlueprintFunctionLibrary.get_data(h))
    comp.set_skeletal_mesh_asset(mesh)
    comp.attach_to_component(body, "", unreal.AttachmentRule.SNAP_TO_TARGET, unreal.AttachmentRule.SNAP_TO_TARGET,
                             unreal.AttachmentRule.SNAP_TO_TARGET, False)
    comp.set_leader_pose_component(body, True, False)
    comp.set_editor_property("visibility_based_anim_tick_option",
                             unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
    if materials_from is not None:
        for i, m in enumerate(materials_from.get_materials()):
            if i < comp.get_num_materials():
                comp.set_material(i, m)
    return comp


def character_actor(key):
    if key in S["actors"]:
        return S["actors"][key]
    spec = JOB["characters"][key]
    path = spec["path"]
    o = EAL.load_asset(path)
    if o is None:
        raise RuntimeError(f"no asset {path}")
    if isinstance(o, unreal.Blueprint):
        a = act.spawn_actor_from_class(o.generated_class(), unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    else:
        if not mhs.is_object_added_for_editing(o):
            if not mhs.try_add_object_to_edit(o):
                raise RuntimeError(f"try_add_object_to_edit failed for {path}")
            S["edit"].append(o)
        a = mhs.spawn_meta_human_actor(o, True)
        if a is None:
            raise RuntimeError(f"spawn_meta_human_actor returned None for {path}")
        a.set_actor_location(unreal.Vector(0, 0, 0), False, False)
        mhs.assemble_for_preview(o)
        try:
            a.get_editor_property("lod_sync_component").set_editor_property("forced_lod", 0)
        except Exception as e:
            note(f"forced LOD not set: {e}")
    body, face = skel_comps(a)
    for comp in (body, face):
        if comp is not None:
            try:
                comp.set_update_animation_in_editor(True)
            except Exception as e:
                note(f"set_update_animation_in_editor: {e}")
    followers = []
    for i, g in enumerate(spec.get("garments", [])):
        followers.append(add_follower(a, body, g, f"garment_{i}"))
    if spec.get("body_swap"):
        followers.append(add_follower(a, body, spec["body_swap"], "body_swap", materials_from=body))
    if spec.get("body_swap") or spec.get("hide_body"):
        # keep it animating (followers read its pose) but do not draw it
        body.set_editor_property("visibility_based_anim_tick_option",
                                 unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
        body.set_render_in_main_pass(False)
        body.set_cast_shadow(False)
    info = {"class": a.get_class().get_name(), "body": body.get_skinned_asset().get_path_name() if body else None,
            "face": face.get_skinned_asset().get_path_name() if face else None,
            "followers": [f.get_skinned_asset().get_path_name() for f in followers]}
    S["r"].setdefault("actors", {})[key] = info
    log(f"spawned {key}: {json.dumps(info)}")
    S["actors"][key] = (a, body, face)
    return S["actors"][key]


def export_geometry(ex):
    """Character asset -> body/head skeletal mesh assets (export_geometry) -> FBX files on disk."""
    t0 = time.time()
    r = {}
    c = EAL.load_asset(JOB["characters"][ex["key"]]["path"])
    if not mhs.is_object_added_for_editing(c):
        mhs.try_add_object_to_edit(c)
        S["edit"].append(c)
    p = unreal.MetaHumanGeometryExportParams()
    p.project_path = ex.get("folder", "/Game/Char/Geo")
    p.overwrite_existing_assets = True
    p.body_skeletal_mesh = True
    p.head_skeletal_mesh = True
    p.full_body_skeletal_mesh = bool(ex.get("full_body", False))
    unreal.MetaHumanCharacterExportBlueprintLibrary.export_geometry(c, p)
    r["assets"] = EAL.list_assets(p.project_path, recursive=True, include_folder=False)
    os.makedirs(ex["fbx_dir"], exist_ok=True)
    r["fbx"] = {}
    for ap in r["assets"]:
        o = EAL.load_asset(ap)
        if not isinstance(o, unreal.SkeletalMesh):
            continue
        out = os.path.join(ex["fbx_dir"], o.get_name() + ".fbx")
        t = unreal.AssetExportTask()
        t.object, t.filename, t.automated, t.replace_identical, t.prompt = o, out, True, True, False
        t.exporter = unreal.SkeletalMeshExporterFBX()
        opt = unreal.FbxExportOption()
        opt.collision = False
        opt.level_of_detail = False
        opt.export_morph_targets = False
        t.options = opt
        ok = unreal.Exporter.run_asset_export_task(t)
        r["fbx"][ap] = {"ok": ok, "file": out, "bytes": os.path.getsize(out) if os.path.exists(out) else 0,
                        "skeleton": o.skeleton.get_path_name() if o.skeleton else None}
    r["time_s"] = round(time.time() - t0, 1)
    S["r"]["export_geometry"] = r
    log(f"export_geometry {json.dumps(r, default=str)[:800]}")


def cloud_build(cb):
    """Cloud face rig + high-resolution textures (signed-in editor), then assembly into a Blueprint, in this editor
    (a real RHI: the -nullrhi commandlet assembly crashed). Times every step."""
    r = {"steps": []}
    S["r"]["cloud"] = r
    t0 = time.time()

    def step(n, **kw):
        kw["step"], kw["t_s"] = n, round(time.time() - t0, 1)
        r["steps"].append(kw)
        log(f"cloud {json.dumps(kw, default=str)}")
        json.dump(S["r"], open(JOB["result"], "w"), indent=1, default=str)

    c = EAL.load_asset(cb["path"])
    if not mhs.is_object_added_for_editing(c):
        if not mhs.try_add_object_to_edit(c):
            raise RuntimeError("try_add_object_to_edit failed")
        S["edit"].append(c)
    if cb.get("rig", True):
        p = unreal.MetaHumanCharacterAutoRiggingRequestParams()
        p.blocking, p.report_progress = True, False
        p.rig_type = getattr(unreal.MetaHumanRigType, cb.get("rig_type", "JOINTS_AND_BLEND_SHAPES"))
        mhs.request_auto_rigging(c, p)
        step("auto_rig", can_build=mhs.can_build_meta_human(c, False))
    if cb.get("textures", True):
        t = unreal.MetaHumanCharacterTextureRequestParams()
        t.blocking, t.report_progress = True, False
        mhs.request_texture_sources(c, t)
        step("textures", high_res=c.get_editor_property("has_high_resolution_textures"))
    ok = mhs.can_build_meta_human(c, True)
    step("gate", can_build=ok)
    EAL.save_loaded_asset(c, False)
    if ok and cb.get("build", True):
        b = unreal.MetaHumanCharacterEditorBuildParameters()
        b.pipeline_type = getattr(unreal.MetaHumanDefaultPipelineType, cb.get("pipeline", "CINEMATIC"))
        b.pipeline_quality = getattr(unreal.MetaHumanQualityLevel, cb.get("quality", "HIGH"))
        b.absolute_build_path = cb.get("build_path", "/Game/MetaHumans")
        b.common_folder_path = cb.get("build_path", "/Game/MetaHumans") + "/Common"
        b.enable_wardrobe_item_validation = False
        mhs.build_meta_human(c, b)
        name = cb["path"].split("/")[-1].split(".")[0]
        out = EAL.list_assets(f"{b.absolute_build_path}/{name}", recursive=True, include_folder=False)
        bps = [a for a in out if "/BP_" in a]
        step("build", assets=len(out), blueprint=bps[:3])
        r["blueprint"] = bps[0].split(".")[0] if bps else None
        for a in out:
            EAL.save_asset(a.split(".")[0], True)
        step("saved")
    r["ok"] = ok


def import_fbx(fbx, dest):
    """Skeletal FBX -> assets under dest (cleared first); returns the SkeletalMesh path."""
    if EAL.does_directory_exist(dest):
        EAL.delete_directory(dest)
    task = unreal.AssetImportTask()
    task.filename = fbx
    task.destination_path = dest
    task.automated = True
    task.replace_existing = True
    task.save = True
    at.import_asset_tasks([task])
    meshes = [a.split(".")[0] for a in EAL.list_assets(dest, recursive=True, include_folder=False)
              if isinstance(EAL.load_asset(a), unreal.SkeletalMesh)]
    if not meshes:
        raise RuntimeError(f"no skeletal mesh from {fbx}: {list(task.imported_object_paths)}")
    return meshes[0]


def import_texture(png, dest, kind):
    task = unreal.AssetImportTask()
    task.filename = png
    task.destination_path = dest
    task.automated = True
    task.replace_existing = True
    task.save = False
    at.import_asset_tasks([task])
    name = os.path.splitext(os.path.basename(png))[0]
    tex = EAL.load_asset(f"{dest}/{name}")
    if kind == "normal":
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
        tex.set_editor_property("srgb", False)
    elif kind == "mask":
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
        tex.set_editor_property("srgb", False)
    EAL.save_loaded_asset(tex, False)
    return tex


def baked_master(path="/Game/Char/Materials/M_BakedRMA"):
    """Opaque lit master: BaseColor, Normal and RMA (R roughness, G metallic, B occlusion) texture parameters."""
    if EAL.does_asset_exist(path):
        return EAL.load_asset(path)
    MEL = unreal.MaterialEditingLibrary
    folder, name = path.rsplit("/", 1)
    m = at.create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())

    def tex_param(pname, y, sampler):
        n = MEL.create_material_expression(m, unreal.MaterialExpressionTextureSampleParameter2D, -500, y)
        n.set_editor_property("parameter_name", pname)
        n.set_editor_property("sampler_type", sampler)
        return n
    bc = tex_param("BaseColor", -200, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    nm = tex_param("Normal", 100, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
    rma = tex_param("RMA", 400, unreal.MaterialSamplerType.SAMPLERTYPE_MASKS)
    MEL.connect_material_property(bc, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    MEL.connect_material_property(nm, "RGB", unreal.MaterialProperty.MP_NORMAL)
    MEL.connect_material_property(rma, "R", unreal.MaterialProperty.MP_ROUGHNESS)
    MEL.connect_material_property(rma, "G", unreal.MaterialProperty.MP_METALLIC)
    MEL.connect_material_property(rma, "B", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
    m.set_editor_property("used_with_skeletal_mesh", True)
    MEL.recompile_material(m)
    EAL.save_loaded_asset(m, False)
    return m


def import_outfit(spec):
    """Outfit + trimmed body FBX from the Blender step, baked textures, a material instance on the outfit."""
    t0 = time.time()
    r = {}
    dest = spec.get("dest", "/Game/Char/Outfit")
    r["outfit"] = import_fbx(spec["outfit_fbx"], dest + "/Outfit")
    if spec.get("trim_fbx"):
        r["trim"] = import_fbx(spec["trim_fbx"], dest + "/BodyTrim")
    tdir = spec["textures"]
    texs = {k: import_texture(os.path.join(tdir, f"T_outfit_{k}.png"), dest + "/Textures", kind)
            for k, kind in (("BaseColor", "color"), ("Normal", "normal"), ("RMA", "mask"))}
    mpath = dest + "/MI_Outfit"
    if EAL.does_asset_exist(mpath):
        EAL.delete_asset(mpath)
    mi = at.create_asset("MI_Outfit", dest, unreal.MaterialInstanceConstant,
                         unreal.MaterialInstanceConstantFactoryNew())
    MEL = unreal.MaterialEditingLibrary
    MEL.set_material_instance_parent(mi, baked_master())
    for k, t in texs.items():
        MEL.set_material_instance_texture_parameter_value(mi, k, t)
    EAL.save_loaded_asset(mi, False)
    sk = EAL.load_asset(r["outfit"])
    mats = sk.get_editor_property("materials")
    for sm in mats:
        sm.set_editor_property("material_interface", mi)
    sk.set_editor_property("materials", mats)
    EAL.save_loaded_asset(sk, False)
    r["time_s"] = round(time.time() - t0, 1)
    S["r"]["import_outfit"] = r
    log(f"import_outfit {json.dumps(r)}")
    for spec_c in JOB.get("characters", {}).values():
        spec_c["garments"] = [r["outfit"] if g == "@outfit" else g for g in spec_c.get("garments", [])]
        if spec_c.get("body_swap") == "@trim":
            spec_c["body_swap"] = r.get("trim")


def curve_anim(skel, curves, name):
    folder = "/Game/Char/Anim"
    path = f"{folder}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    f = unreal.AnimSequenceFactory()
    f.set_editor_property("target_skeleton", skel)
    anim = at.create_asset(name, folder, unreal.AnimSequence, f)
    ctrl = anim.get_editor_property("controller")
    ctrl.open_bracket("curves", False)
    ctrl.set_frame_rate(unreal.FrameRate(30, 1), False)
    ctrl.set_number_of_frames(unreal.FrameNumber(2), False)
    for cname, val in curves.items():
        unreal.AnimationLibrary.add_curve(anim, cname, unreal.RawCurveTrackTypes.RCT_FLOAT, False)
        for t in (0.0, 1.0 / 30, 2.0 / 30):
            unreal.AnimationLibrary.add_float_curve_key(anim, cname, t, float(val))
    ctrl.close_bracket(False)
    return anim


def bone_world(anim, chain, t):
    poses = unreal.AnimationLibrary.get_bone_poses_for_time(anim, chain, t, False)
    x = unreal.Transform()
    for p in poses:
        x = unreal.MathLibrary.compose_transforms(p, x)
    return x


ARM_R = ["root", "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "clavicle_r", "upperarm_r",
         "lowerarm_r", "hand_r"]


def scan_body(anim, kind):
    """Time in a body clip where the right hand is highest (arms_up)."""
    n, length = 80, unreal.AnimationLibrary.get_sequence_length(anim)
    best, best_t = None, 0.0
    for i in range(n + 1):
        t = length * i / n
        z = bone_world(anim, ARM_R, t).translation.z
        if best is None or z > best:
            best, best_t = z, t
    return best_t, best, length


def apply_anim(comp, anim, t):
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
    cc.set_editor_property("current_aperture", 5.6)
    return cam


def show_only(key):
    for k, (a, body, face) in S["actors"].items():
        a.set_is_temporarily_hidden_in_editor(k != key)
        a.set_actor_hidden_in_game(k != key)


def setup_shot(sh):
    a, body, face = character_actor(sh["char"])
    show_only(sh["char"])
    info = {"name": sh["name"]}
    if sh.get("body_anim"):
        banim = EAL.load_asset(sh["body_anim"])
        t = sh.get("body_t", 0.0)
        if sh.get("body_scan"):
            t, score, length = scan_body(banim, sh["body_scan"])
            info["body_scan"] = {"t": round(t, 3), "score": round(score, 2), "length": round(length, 2)}
        apply_anim(body, banim, t)
        info["body"] = [sh["body_anim"], round(t, 3)]
    elif body is not None:
        clear_anim(body)
    if face is not None:
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
    try:
        head = face.get_socket_location("head")
        if head.z < 50:
            raise ValueError
    except Exception:
        head = unreal.Vector(0, 0, 165)
    info["head"] = [round(head.x, 1), round(head.y, 1), round(head.z, 1)]
    fr, az, el = sh.get("frame", "close"), sh.get("az", 0), sh.get("el", 3)
    if fr == "close":
        cam = camera(sh["name"], head + unreal.Vector(0, 0, -6), az, el, 200, 85)
    elif fr == "medium":
        cam = camera(sh["name"], head + unreal.Vector(0, 0, -38), az, el, 260, 50)
    else:
        cam = camera(sh["name"], unreal.Vector(head.x, head.y, head.z * 0.52), az, sh.get("el", 4), 420, 35)
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
    ph, shots = S["phase"], JOB.get("shots", [])
    if ph == "boot":
        w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        if JOB.get("map") and (w is None or not w.get_path_name().startswith(JOB["map"])):
            les.load_level(JOB["map"])
        if JOB.get("make"):
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(JOB["script"])))
            import char_make
            char_make.scan()
            S["r"]["make"] = {}
            for spec in JOB["make"]:
                S["r"]["make"][spec["name"]] = char_make.make(spec)
                log(f"made {spec['name']}: {json.dumps(S['r']['make'][spec['name']], default=str)[:300]}")
        if JOB.get("cloud"):
            cloud_build(JOB["cloud"])
            bp = S["r"]["cloud"].get("blueprint")
            for k, spec in JOB.get("characters", {}).items():
                if spec.get("path") == "@built":
                    if not bp:
                        raise RuntimeError("no assembled blueprint to render")
                    spec["path"] = bp
        if JOB.get("import_outfit"):
            import_outfit(JOB["import_outfit"])
        if JOB.get("export_geometry"):
            try:
                export_geometry(JOB["export_geometry"])
            except Exception:
                S["r"]["export_geometry"] = {"error": traceback.format_exc()}
                note(S["r"]["export_geometry"]["error"])
        if not shots:
            S["r"]["ok"] = True
            finish()
            return
        les.editor_set_game_view(True)
        unreal.AutomationLibrary.set_scalability_quality_to_epic(
            unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world())
        light_scene()
        for sh in shots:
            character_actor(sh["char"])
        S["r"]["setup_s"] = round(time.time() - S["t0"], 1)
        goto("warm")
    elif ph == "warm" and waited(JOB.get("warm_frames", 150)):
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
