"""UE side, GPU: render a MetaHuman character, optionally dressed in skinned garments, posed, at fixed framings.

Two ways to run it:
  - through the asset scheduler's resident editor (preferred; no private editor, no card lock of our own):
        python tools/assets/characters/char_submit.py job.json
    the scheduler calls run(job), a generator advanced once per editor tick;
  - in a private editor (fallback) by the lab shim: python tools/assets/characters/char_job.py job.json
    (job["script"] = this file; the module then drives run() from a Slate tick callback and quits the editor).

Job:
  {"script", "map", "out", "result", "res": [1152, 640], "warm_frames": 150, "settle_frames": 40,
   "backdrop": true,                              (grey cyclorama instead of the template sky)
   "interior": null | true | false,               (true: groom strands get no sky light; see interior_hair)
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
JOB = {}
RES = [1152, 640]
S = {}


def reset():
    S.clear()
    S.update({"t0": time.time(), "actors": {}, "scene": [], "edit": [],
              "r": {"ok": False, "shots": [], "notes": []}})


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
    """(body, face) skeletal mesh components of an assembled Blueprint actor ("Body", "Face") or of the editor
    preview actor (its body_component / face_component properties)."""
    body = face = None
    for c in a.get_components_by_class(unreal.SkeletalMeshComponent):
        n = c.get_name().lower()
        if n == "body":
            body = c
        elif n == "face":
            face = c
    if body is None:
        try:
            body = a.get_editor_property("body_component")
            face = a.get_editor_property("face_component")
        except Exception:
            pass
    return body, face


def add_follower(a, body, mesh_path, label, materials_from=None):
    """A garment (or the trimmed body) as its own SkeletalMeshActor attached to the character, its component driven
    by the character's body through set_leader_pose_component (bones matched by name), so it follows every pose."""
    mesh = EAL.load_asset(mesh_path)
    if mesh is None:
        raise RuntimeError(f"no skeletal mesh {mesh_path}")
    fa = spawn(unreal.SkeletalMeshActor, (0, 0, 0), label=f"{a.get_actor_label()}_{label}")
    comp = fa.skeletal_mesh_component
    comp.set_skeletal_mesh_asset(mesh)
    # not attached: attaching re-runs the Blueprint's construction and replaces its components; the follower sits
    # where the character sits and copies the body's component-space pose
    fa.set_actor_transform(a.get_actor_transform(), False, False)
    comp.set_leader_pose_component(body, True, False)
    comp.set_editor_property("visibility_based_anim_tick_option",
                             unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
    comp.set_editor_property("bounds_scale", 2.0)
    if JOB.get("garment_material") and materials_from is None:
        gm = EAL.load_asset(JOB["garment_material"])
        for i in range(comp.get_num_materials()):
            comp.set_material(i, gm)
    if materials_from is not None:
        for i, m in enumerate(materials_from.get_materials()):
            if i < comp.get_num_materials():
                comp.set_material(i, m)
    elif body is not None and body.get_num_materials():
        # an outfit's hands (slot M_skin: the body's own hands, its UVs) wear the body's skin material; the import
        # gives every slot the cloth's baked atlas, which painted the hands with cloth texels (dark, white specks)
        try:
            names = [str(n) for n in comp.get_material_slot_names()]
        except Exception:
            names = []
        if "M_skin" in names:
            comp.set_material(names.index("M_skin"), body.get_material(0))
    S.setdefault("followers", {}).setdefault(id(a), []).append(fa)
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
        unreal.BlueprintEditorLibrary.compile_blueprint(o)    # a stale class spawns without its components
        a = act.spawn_actor_from_object(o, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        if skel_comps(a)[0] is None and spec.get("rebuild_from"):
            # loaded broken in this editor (its Live Link AnimBP cast did not resolve): reassemble it here
            note(f"{path} spawned without components; reassembling from {spec['rebuild_from']}")
            act.destroy_actor(a)
            cloud_build({"path": spec["rebuild_from"], "rig": False, "textures": False,
                         "build_path": spec.get("build_path", "/Game/MetaHumans")})
            o = EAL.load_asset(path)
            a = act.spawn_actor_from_object(o, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
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
    S["actors"][key] = (a, None, None)          # registered at once so cleanup removes it even on an error
    body, face = skel_comps(a)
    for comp in (body, face):
        if comp is not None:
            try:
                comp.set_update_animation_in_editor(True)
            except Exception as e:
                note(f"set_update_animation_in_editor: {e}")
    followers = []
    hide_face = bool(spec.get("hide_face")) or "@hide_face" in spec.get("garments", [])
    for i, g in enumerate(g_ for g_ in spec.get("garments", []) if g_ != "@hide_face"):
        followers.append(add_follower(a, body, g, f"garment_{i}"))
    if hide_face:
        # a pure generated figure (its own head, hair and hands): the MetaHuman's face and grooms are not drawn
        # (they keep animating, so the face skeleton still measures eyes and head)
        for c in a.get_components_by_class(unreal.SkeletalMeshComponent):
            if c != body and c.get_name().lower() != "body":
                c.set_render_in_main_pass(False)
                c.set_cast_shadow(False)
                try:
                    c.set_editor_property("visibility_based_anim_tick_option",
                                          unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
                except Exception:
                    pass
        for gc in a.get_components_by_class(unreal.GroomComponent):
            gc.set_visibility(False, True)
        note(f"{key}: MetaHuman face and grooms hidden (pure figure)")
    if spec.get("body_swap"):
        followers.append(add_follower(a, body, spec["body_swap"], "body_swap", materials_from=body))
    if spec.get("body_swap") or spec.get("hide_body"):
        # keep it animating (followers read its pose) but do not draw it
        body.set_editor_property("visibility_based_anim_tick_option",
                                 unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
        body.set_render_in_main_pass(False)
        body.set_cast_shadow(False)
    if body is None:
        raise RuntimeError(f"{path}: no Body component on the spawned actor (Blueprint failed to compile?)")
    info = {"class": a.get_class().get_name(),
            "body": body.get_skinned_asset().get_path_name() if body.get_skinned_asset() else None,
            "face": face.get_skinned_asset().get_path_name() if face and face.get_skinned_asset() else None,
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
    tex.set_editor_property("never_stream", True)    # a 4k map otherwise renders from a low mip (flat colour)
    EAL.save_loaded_asset(tex, False)
    return tex


def baked_material(path, texs, opacity=None):
    """An opaque, two-sided lit material sampling the baked maps directly (BaseColor, tangent Normal, RMA = R
    roughness, G metallic, B occlusion). Direct samplers rather than an instance of a parameter master: the
    instance route rendered the master's default textures here."""
    MEL = unreal.MaterialEditingLibrary
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    folder, name = path.rsplit("/", 1)
    m = at.create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())

    def sampler(tex, y, st):
        n = MEL.create_material_expression(m, unreal.MaterialExpressionTextureSample, -500, y)
        n.set_editor_property("texture", tex)
        n.set_editor_property("sampler_type", st)
        return n
    bc = sampler(texs["BaseColor"], -200, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    nm = sampler(texs["Normal"], 100, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
    rma = sampler(texs["RMA"], 400, unreal.MaterialSamplerType.SAMPLERTYPE_MASKS)
    # cloth sheen: dark wool and crape catch light at grazing angles (fibres), which is what shows their folds in a
    # dim room; the base colour is lifted toward the rim by a Fresnel term (x1 face-on, up to x2.2 at the rim)
    # instead of raising the albedo everywhere
    fr = MEL.create_material_expression(m, unreal.MaterialExpressionFresnel, -700, -350)
    fr.set_editor_property("exponent", 2.5)
    k = MEL.create_material_expression(m, unreal.MaterialExpressionConstant, -700, -260)
    k.set_editor_property("r", 1.2)
    one = MEL.create_material_expression(m, unreal.MaterialExpressionConstant, -700, -200)
    one.set_editor_property("r", 1.0)
    mul = MEL.create_material_expression(m, unreal.MaterialExpressionMultiply, -550, -330)
    MEL.connect_material_expressions(fr, "", mul, "A")
    MEL.connect_material_expressions(k, "", mul, "B")
    addn = MEL.create_material_expression(m, unreal.MaterialExpressionAdd, -420, -300)
    MEL.connect_material_expressions(mul, "", addn, "A")
    MEL.connect_material_expressions(one, "", addn, "B")
    bcm = MEL.create_material_expression(m, unreal.MaterialExpressionMultiply, -300, -220)
    MEL.connect_material_expressions(bc, "RGB", bcm, "A")
    MEL.connect_material_expressions(addn, "", bcm, "B")
    MEL.connect_material_property(bcm, "", unreal.MaterialProperty.MP_BASE_COLOR)
    MEL.connect_material_property(nm, "RGB", unreal.MaterialProperty.MP_NORMAL)
    MEL.connect_material_property(rma, "R", unreal.MaterialProperty.MP_ROUGHNESS)
    MEL.connect_material_property(rma, "G", unreal.MaterialProperty.MP_METALLIC)
    MEL.connect_material_property(rma, "B", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
    # cloth, not leather: a low specular level (no broad glossy lobe on black wool or a matte dress)
    sp = MEL.create_material_expression(m, unreal.MaterialExpressionConstant, -200, 600)
    sp.set_editor_property("r", 0.15)
    MEL.connect_material_property(sp, "", unreal.MaterialProperty.MP_SPECULAR)
    if opacity is not None:
        # a net (a veil): masked, the opacity map dithered over frames so the holes read as sheer cloth
        op = sampler(opacity, 700, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE)
        fn = EAL.load_asset("/Engine/Functions/Engine_MaterialFunctions02/Utility/DitherTemporalAA")
        if fn is not None:
            dt = MEL.create_material_expression(m, unreal.MaterialExpressionMaterialFunctionCall, -200, 700)
            dt.set_editor_property("material_function", fn)
            ins = [str(x) for x in MEL.get_material_expression_input_names(dt)]
            MEL.connect_material_expressions(op, "R", dt, ins[0] if ins else "Alpha Threshold")
            MEL.connect_material_property(dt, "", unreal.MaterialProperty.MP_OPACITY_MASK)
        else:
            MEL.connect_material_property(op, "R", unreal.MaterialProperty.MP_OPACITY_MASK)
        m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
        m.set_editor_property("opacity_mask_clip_value", 0.5)
    m.set_editor_property("two_sided", True)
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
    opacity = None
    if os.path.exists(os.path.join(tdir, "T_outfit_Opacity.png")):
        opacity = import_texture(os.path.join(tdir, "T_outfit_Opacity.png"), dest + "/Textures", "mask")
        opacity.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_GRAYSCALE)
        EAL.save_loaded_asset(opacity, False)
    mi = baked_material(dest + "/M_OutfitBaked", texs, opacity)
    r["material"] = mi.get_path_name()
    sk = EAL.load_asset(r["outfit"])
    mats = []
    for sm in sk.get_editor_property("materials"):       # the array holds copies: rebuild it
        sm2 = unreal.SkeletalMaterial()
        sm2.set_editor_property("material_interface", mi)
        sm2.set_editor_property("material_slot_name", sm.get_editor_property("material_slot_name"))
        mats.append(sm2)
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


LEG_L = ["root", "pelvis", "thigh_l", "calf_l"]
HEAD = ["root", "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head"]


def scan_body(anim, kind):
    """Time in a body clip that best shows a wanted pose: arms_up (right hand highest), knee_up (left knee furthest
    forward and up, the nearest the range-of-motion clip comes to sitting), head_turn (head yaw nearest 35 deg)."""
    n, length = 120, unreal.AnimationLibrary.get_sequence_length(anim)
    best, best_t = None, 0.0
    for i in range(n + 1):
        t = length * i / n
        if kind == "arms_up":
            score = bone_world(anim, ARM_R, t).translation.z
        elif kind == "knee_up":
            k = bone_world(anim, LEG_L, t).translation
            score = k.z + abs(k.y)
        else:                         # head turned ~30 deg, kept level (the clip also tilts the head back)
            h = bone_world(anim, HEAD, t)
            fwd = h.rotation.rotate_vector(unreal.Vector(0, 1, 0))
            yaw = math.degrees(math.atan2(fwd.x, fwd.y))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd.z))))
            score = -abs(abs(yaw) - 30) - 2 * abs(pitch)
        if best is None or score > best:
            best, best_t = score, t
    return best_t, best, length


def settle_pose(comp):
    """Evaluate the paused pose now (an editor-world component may not tick its animation on its own)."""
    for fn in ("tick_animation", "refresh_bone_transforms", "tick_pose"):
        try:
            if fn == "tick_animation":
                comp.tick_animation(0.0, False)
            elif fn == "refresh_bone_transforms":
                comp.refresh_bone_transforms()
        except Exception:
            pass


def apply_anim(comp, anim, t):
    """Single-node animation paused at time t (override data carries the start position; set again afterwards
    because a re-used single-node instance can keep the previous position)."""
    comp.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    try:
        comp.override_animation_data(anim, False, False, float(t), 0.0)
    except Exception:
        comp.set_animation(anim)
    comp.set_animation(anim)
    comp.play(False)
    comp.set_position(float(t), False)
    comp.set_play_rate(0.0)
    comp.stop()
    comp.set_position(float(t), False)
    settle_pose(comp)


def clear_anim(comp):
    """Back to the component's own Animation Blueprint (an assembled character's face follows the body's head
    through it), or to the reference pose when it has none."""
    try:
        if comp.get_editor_property("anim_class"):
            comp.set_animation_mode(unreal.AnimationMode.ANIMATION_BLUEPRINT)
            return
    except Exception:
        pass
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
        for x in [a] + S.get("followers", {}).get(id(a), []):
            x.set_is_temporarily_hidden_in_editor(k != key)
            x.set_actor_hidden_in_game(k != key)


def refresh(key):
    """Current body/face components (a Blueprint may have rebuilt them) re-wired to the followers, body hidden when
    a trimmed copy replaces it."""
    a = S["actors"][key][0]
    body, face = skel_comps(a)
    spec = JOB["characters"][key]
    for fa in S.get("followers", {}).get(id(a), []):
        if spec.get("leader", True):
            fa.skeletal_mesh_component.set_leader_pose_component(body, True, False)
        else:
            fa.skeletal_mesh_component.set_leader_pose_component(None, True, False)
    if spec.get("body_swap") or spec.get("hide_body"):
        body.set_editor_property("visibility_based_anim_tick_option",
                                 unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
        body.set_visibility(False, False)          # still posed (always tick), so followers keep following
        body.set_cast_shadow(False)
    for comp in (body, face):
        if comp is not None:
            try:
                comp.set_update_animation_in_editor(True)
            except Exception:
                pass
    # LOD 0 everywhere: lower LODs of an assembled body drop helper and finger bones, and a garment skinned to
    # those bones collapses to the origin through the leader pose
    try:
        lod = a.get_component_by_class(unreal.LODSyncComponent)
        if lod:
            lod.set_editor_property("forced_lod", 0)
    except Exception:
        pass
    for comp in [body, face] + [fa.skeletal_mesh_component for fa in S.get("followers", {}).get(id(a), [])]:
        if comp is not None:
            try:
                comp.set_forced_lod(1)
            except Exception:
                pass
    S["actors"][key] = (a, body, face)
    return a, body, face


def setup_shot(sh):
    character_actor(sh["char"])
    a, body, face = refresh(sh["char"])
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
        if sh.get("face_curves") or sh.get("face_anim"):
            S.setdefault("face_touched", {})[sh["char"]] = True
        if sh.get("face_curves"):
            anim = curve_anim(face.get_skinned_asset().get_editor_property("skeleton"), sh["face_curves"],
                              f"AS_{sh['name']}")
            apply_anim(face, anim, 1.0 / 30)
            S["face_anim_last"] = anim
            info["face"] = sh["face_curves"]
        elif sh.get("face_anim"):
            apply_anim(face, EAL.load_asset(sh["face_anim"]), sh.get("face_t", 0.0))
            info["face"] = [sh["face_anim"], sh.get("face_t", 0.0)]
        elif not S.get("face_touched", {}).get(sh["char"]):
            pass                              # keep the face's own AnimBP (it follows the body's head)
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
    if fr == "close":                 # head and shoulders: aim near the eyes (the head joint sits low in the skull)
        cam = camera(sh["name"], head + unreal.Vector(0, 0, sh.get("dz", 1)), az, el, sh.get("dist", 230), 85)
    elif fr == "medium":
        cam = camera(sh["name"], head + unreal.Vector(0, 0, -38), az, el, 260, 50)
    else:
        cam = camera(sh["name"], unreal.Vector(head.x, head.y, head.z * 0.52), az, sh.get("el", 4), 420, 35)
    return cam, info


def track_images(spec):
    """Face contour tracking (the conform tool's NNE tracker) on raw BGRA8 images: {name: [path, w, h]} ->
    {name: {curve: [[x, y], ...]}} in image pixels."""
    out = {}
    for name, (path, w, h) in spec.items():
        t0 = time.time()
        b = open(path, "rb").read()
        px = [unreal.Color(b[i + 2], b[i + 1], b[i], 255) for i in range(0, len(b), 4)]
        res = mhs.track_face_landmarks_from_image(px, int(w), int(h))
        curves = {}
        if res:
            for k, v in res.items():
                pts = None
                for attr in ("points", "Points", "key_points", "tracking_points"):
                    try:
                        pts = [[round(p.x, 2), round(p.y, 2)] for p in getattr(v, attr)]
                        break
                    except Exception:
                        pass
                curves[str(k)] = pts if pts is not None else v.export_text()
        out[name] = {"curves": curves, "time_s": round(time.time() - t0, 2)}
        log(f"tracked {name}: {len(curves)} curves in {out[name]['time_s']} s")
    S["r"]["track"] = out


HAIR_SKY = "r.HairStrands.SkyLighting"


def interior_hair(interior):
    """Groom strands take the sky light's cubemap directly, with no occlusion by walls: in a room, black hair then
    shows the outdoor sky's blue in its specular (measured 11-11 in a dressed sitting room: navy hair; with the
    strands' sky lighting off it reads black). An interior job turns it off for its stills (restored in cleanup);
    outdoors the sky does reach the hair and the default stays."""
    if interior is None:
        return
    try:
        S["hair_sky_old"] = unreal.SystemLibrary.get_console_variable_int_value(HAIR_SKY)
    except Exception:
        S["hair_sky_old"] = 1
    unreal.SystemLibrary.execute_console_command(None, f"{HAIR_SKY} {0 if interior else 1}")
    note(f"{HAIR_SKY} {0 if interior else 1} (interior={bool(interior)})")


def cleanup():
    """Leave the (possibly resident) editor as it was: our actors gone, characters released."""
    if "hair_sky_old" in S:
        unreal.SystemLibrary.execute_console_command(None, f"{HAIR_SKY} {S.pop('hair_sky_old')}")
    try:
        les.eject_pilot_level_actor()
    except Exception:
        pass
    for a in S.get("scene", []) + [t[0] for t in S.get("actors", {}).values()]:
        try:
            act.destroy_actor(a)
        except Exception:
            pass
    for c in S.get("edit", []):
        try:
            mhs.remove_object_to_edit(c)
        except Exception:
            pass
    S["r"]["elapsed_s"] = round(time.time() - S["t0"], 1)
    if JOB.get("result"):
        json.dump(S["r"], open(JOB["result"], "w"), indent=1, default=str)
    log(f"done ok={S['r']['ok']} in {S['r']['elapsed_s']} s")


def frames(n):
    for _ in range(int(n)):
        yield


def _run():
    shots = JOB.get("shots", [])
    if JOB.get("map"):
        # always a fresh copy of the level: a resident editor keeps the last job's level and any actor it leaked
        les.load_level(JOB["map"])
        yield from frames(5)
    # an assembled Blueprint casts to the plugin's Live Link AnimBP: load it before any Blueprint so the cast
    # resolves when the Blueprint is loaded and compiled
    for dep in JOB.get("bp_deps", ["/MetaHumanCharacter/Animation/ABP_MH_LiveLink"]):
        EAL.load_asset(dep)
    interior_hair(JOB.get("interior"))     # job["interior"]: true for a room (the set's own property), false outdoors
    if JOB.get("track"):
        track_images(JOB["track"])
        yield
    if JOB.get("make"):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import importlib
        import char_make
        importlib.reload(char_make)          # the resident editor keeps modules between jobs
        if JOB.get("coeff_cache"):
            char_make.COEFF_CACHE = JOB["coeff_cache"]
        char_make.scan()
        S["r"]["make"] = {}
        for spec in JOB["make"]:
            S["r"]["make"][spec["name"]] = char_make.make(spec)
            log(f"made {spec['name']}: {json.dumps(S['r']['make'][spec['name']], default=str)[:300]}")
            yield
    if JOB.get("set_params"):
        # groom / outfit instance parameters (hair melanin, redness, roughness ...) on existing Character assets;
        # a "cloud" step with rig and textures off then rebuilds the assembled Blueprint with them
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import importlib
        import char_make
        importlib.reload(char_make)
        S["r"]["set_params"] = {}
        for sp in JOB["set_params"]:
            c = EAL.load_asset(sp["path"])
            added = not mhs.is_object_added_for_editing(c) and mhs.try_add_object_to_edit(c)
            rr = {}
            try:
                insts = [c.get_editor_property("internal_collection").get_editor_property("default_instance")]
                try:
                    insts.insert(0, mhs.get_preview_collection(c).get_editor_property("default_instance"))
                except Exception as e:
                    rr["preview_instance_error"] = str(e)
                for k_, ins in enumerate(insts):
                    sub = {}
                    char_make.set_params(ins, sp["params"], sub)
                    rr[f"instance_params_{k_}"] = sub.get("instance_params")
            finally:
                if added:
                    mhs.remove_object_to_edit(c)
            rr["saved"] = EAL.save_loaded_asset(c, False)
            S["r"]["set_params"][sp["path"]] = rr
            yield
    if JOB.get("cloud"):
        try:
            cloud_build(JOB["cloud"])
        except Exception:
            S["r"].setdefault("cloud", {})["error"] = traceback.format_exc()
            note(S["r"]["cloud"]["error"])
        bp = S["r"]["cloud"].get("blueprint")
        for k, spec in JOB.get("characters", {}).items():
            if spec.get("path") == "@built":
                if not bp:
                    note("no assembled blueprint: rendering the editor preview actor instead")
                    spec["path"] = JOB["cloud"]["path"]
                else:
                    spec["path"] = bp
        yield
    if JOB.get("import_outfit"):
        import_outfit(JOB["import_outfit"])
        yield
    if JOB.get("import_fbx"):              # {"name": [fbx, dest folder]} -> garments may say "@imp:name"
        imp = {k: import_fbx(f, d) for k, (f, d) in JOB["import_fbx"].items()}
        S["r"]["import_fbx"] = imp
        for spec_c in JOB.get("characters", {}).values():
            spec_c["garments"] = [imp[g[5:]] if g.startswith("@imp:") else g for g in spec_c.get("garments", [])]
        yield
    if JOB.get("export_geometry"):
        try:
            export_geometry(JOB["export_geometry"])
        except Exception:
            S["r"]["export_geometry"] = {"error": traceback.format_exc()}
            note(S["r"]["export_geometry"]["error"])
        yield
    if not shots:
        S["r"]["ok"] = True
        return
    les.editor_set_game_view(True)
    unreal.AutomationLibrary.set_scalability_quality_to_epic(
        unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world())
    try:
        unreal.SystemLibrary.execute_console_command(None, "t.MaxFPS 60")
    except Exception:
        pass
    light_scene()
    for sh in shots:
        character_actor(sh["char"])
    for key in list(S["actors"]):
        refresh(key)
    S["r"]["setup_s"] = round(time.time() - S["t0"], 1)
    yield from frames(JOB.get("warm_frames", 150))
    t = time.time()
    try:
        unreal.AssetCompilingManager  # noqa
    except Exception:
        pass
    unreal.AutomationLibrary.finish_loading_before_screenshot()
    S["r"]["finish_loading_s"] = round(time.time() - t, 1)
    for sh in shots:
        cam, info = setup_shot(sh)
        les.pilot_level_actor(cam)
        yield from frames(10)
        if sh.get("body_anim") or sh.get("face_curves") or sh.get("face_anim"):
            # a paused single-node pose shows only from the second application (measured: one shot late)
            a, body, face = S["actors"][sh["char"]]
            if sh.get("body_anim") and "body" in info:
                apply_anim(body, EAL.load_asset(sh["body_anim"]), info["body"][1])
            if sh.get("face_curves") and S.get("face_anim_last"):
                apply_anim(face, S["face_anim_last"], 1.0 / 30)
            yield from frames(10)
        yield from frames(JOB.get("settle_frames", 40))
        path = os.path.join(JOB["out"], f"{sh['name']}.png")
        if os.path.exists(path):
            os.remove(path)
        unreal.AutomationLibrary.take_high_res_screenshot(RES[0], RES[1], path, camera=cam, force_game_view=True)
        t1 = time.time()
        while not (os.path.exists(path) and os.path.getsize(path) > 0):
            if time.time() - t1 > 90:
                raise RuntimeError(f"screenshot not written: {path}")
            yield
        info["path"], info["t_s"] = path, round(time.time() - S["t0"], 1)
        S["r"]["shots"].append(info)
        log(f"shot {json.dumps(info)}")
        les.eject_pilot_level_actor()
    S["r"]["ok"] = True


def run(job):
    """Entry point (scheduler or private editor): a generator, one step per editor tick; returns the result."""
    global JOB, RES
    JOB = job
    RES = job.get("res", [1152, 640])
    reset()
    if job.get("out"):
        os.makedirs(job["out"], exist_ok=True)
    try:
        yield from _run()
    except Exception:
        S["r"]["error"] = traceback.format_exc()
        unreal.log_error(S["r"]["error"])
        cleanup()
        raise                       # the scheduler then reports the job failed instead of waiting for outputs
    cleanup()
    return S["r"]


if __name__ == "__mh_job__":          # private editor started by char_job.py: drive run() from the Slate tick
    _GEN = run(json.load(open(os.environ["MH_JOB"])))

    _BUSY = [False]

    def _tick(dt):
        if _BUSY[0]:                  # Slate ticks re-enter while a long step (asset build, import) pumps messages
            return
        _BUSY[0] = True
        try:
            next(_GEN)
        except StopIteration:
            unreal.unregister_slate_post_tick_callback(_HANDLE)
            unreal.SystemLibrary.quit_editor()
        except Exception:
            unreal.unregister_slate_post_tick_callback(_HANDLE)
            unreal.SystemLibrary.quit_editor()
        finally:
            _BUSY[0] = False

    _HANDLE = unreal.register_slate_post_tick_callback(_tick)
