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
              "body_pose": {"name", "rots": [[bone, parent, deg], ...]} | null,   (built pose, e.g. seated)
              "turn": {"neck_02": deg, "head": deg} | null,   (extra yaw on the body pose, e.g. head turned)
              "face_curves": {curve: value} | null, "face_anim": path | null, "face_t": s}]}

A Character asset renders through the editor preview actor (spawn_meta_human_actor; no assembly needed); an
assembled Blueprint renders as an ordinary actor. Garments are SkeletalMeshComponents added to the actor with
set_leader_pose_component(body), so they follow every body pose; body_swap hides the body component and shows the
trimmed copy (with the body's own materials) the same way. Before each still: head on the neck (face and body
agree within 1 mm) and both hands visible and attached (hands_check), or the still is refused. export_geometry (needs Slate, so the full editor, not a
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
CUR_RES = []          # the shot being set up may have its own resolution (portrait full-length stills)
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
    """Neutral three-point portrait light (front-left key, front-right fill, back rim) on the template level.
    "studio": false keeps the level's own light (a set: judge the character in the room's light)."""
    if not JOB.get("studio", True):
        if JOB.get("exposure_bias") is not None:
            ppv = spawn(unreal.PostProcessVolume, (0, 0, 0), label="ch_pp")
            ppv.set_editor_property("unbound", True)
            pp = ppv.get_editor_property("settings")
            pp.set_editor_property("override_auto_exposure_bias", True)
            pp.set_editor_property("auto_exposure_bias", float(JOB["exposure_bias"]))
            ppv.set_editor_property("settings", pp)
        return
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


def add_follower(a, body, mesh_path, label, materials_from=None, skin=False):
    """A garment (or, skin=True, the trimmed body) as its own SkeletalMeshActor attached to the character, its
    component driven by the character's body through set_leader_pose_component (bones matched by name), so it follows
    every pose. The trimmed body never takes the garment material: it gets the body's own materials in refresh()
    (an assembled Blueprint has no Body component until its first construction, so they are not known here)."""
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
    if JOB.get("garment_material") and not skin:
        gm = EAL.load_asset(JOB["garment_material"])
        slots = skin_slots(comp)
        for i in range(comp.get_num_materials()):
            if i not in slots:
                comp.set_material(i, gm)
    if materials_from is not None:
        for i, m in enumerate(materials_from.get_materials()):
            if i < comp.get_num_materials():
                comp.set_material(i, m)
    elif not skin and body is not None and body.get_num_materials():
        # an outfit's hands (a skin slot: the body's own hands, its UVs) wear the body's skin material; the import
        # gives every slot the cloth's baked atlas, which painted the hands with cloth texels (dark, white specks).
        # refresh() sets the same slots again (set_skin_slots): a Blueprint has no Body materials until its first
        # construction, and rebuilds its components after spawning
        for i in skin_slots(comp):
            comp.set_material(i, body.get_material(0))
    S.setdefault("followers", {}).setdefault(id(a), []).append(fa)
    if skin:
        S.setdefault("swap", {})[id(a)] = fa
        tweak = JOB.get("swap_bounds", "leader")
        if tweak == "leader":
            comp.set_editor_property("use_bounds_from_leader_pose_component", True)
        elif tweak == "scale":
            comp.set_editor_property("bounds_scale", 20.0)
    return comp


def face_material(key, face):
    """Garment token "@face_material=<path>": the character's own masked copy of its face skin (char_skinmask.py),
    so the skin under its clothes is hidden; every face skin slot wears it (the face is held at LOD 0). Applied on
    every refresh: a Blueprint rebuilds its components (and their dynamic skin instances) after spawning."""
    spec = JOB["characters"][key]
    fm = [g[len("@face_material="):] for g in spec.get("garments", []) if g.startswith("@face_material=")]
    if not fm or face is None:
        return
    # the character's own masked copy of its face skin (char_skinmask.py): the skin under its clothes is hidden;
    # every face skin slot wears it (the face is held at LOD 0), the asset and the shared material stay as they are
    fmat = EAL.load_asset(fm[0])
    if fmat is None:
        raise RuntimeError(f"no face material {fm[0]}")
    k, seen = 0, []
    try:
        slots = [str(n) for n in face.get_material_slot_names()]
    except Exception:
        slots = []
    for j in range(face.get_num_materials()):
        m = face.get_material(j)
        chain = []
        while m is not None and len(chain) < 4:      # a Blueprint may wear dynamic instances of the skin
            chain.append(m.get_name())
            m = m.get_editor_property("parent") if isinstance(m, unreal.MaterialInstance) else None
        slot = slots[j] if j < len(slots) else ""
        if any(c.startswith("MI_Face_Skin") for c in chain) or slot.startswith("head_"):
            if chain[:1] != [fmat.get_name()]:
                face.set_material(j, fmat)
                seen.append(slot)
            k += 1
    if seen:
        note(f"{key}: face skin slots now wearing {fm[0]}: {seen} ({k} skin slots)")


def skin_slots(comp):
    """Indices of material slots that carry skin (hands in an outfit mesh): slot name contains "skin"."""
    sk = comp.get_skinned_asset()
    if sk is None:
        return []
    return [i for i, m in enumerate(sk.get_editor_property("materials"))
            if "skin" in str(m.get_editor_property("material_slot_name")).lower()]


def skin_materials(body):
    """The body's skin material(s) for the trimmed copy, slot by slot name when the names match, else the body's
    first slot (the trim keeps one skin slot; the body's UVs, so the body's skin maps fit it)."""
    out = {}
    bm = body.get_skinned_asset().get_editor_property("materials") if body.get_skinned_asset() else []
    names = [str(m.get_editor_property("material_slot_name")).lower() for m in bm]
    mats = body.get_materials()
    for i, n in enumerate(names):
        if i < len(mats) and mats[i] is not None:
            out[n] = mats[i]
    first = next((m for m in mats if m is not None), None)
    return out, first


def set_skin_slots(a, body):
    """Skin slots of every follower (hands carried in the outfit) take the body's own skin material."""
    if body is None:
        return
    _, first = skin_materials(body)
    for fa in S.get("followers", {}).get(id(a), []):
        comp = fa.skeletal_mesh_component
        for i in skin_slots(comp):
            if first is not None:
                comp.set_material(i, first)


def set_swap_materials(a, body):
    fa = S.get("swap", {}).get(id(a))
    if fa is None or body is None:
        return
    by_name, first = skin_materials(body)
    comp = fa.skeletal_mesh_component
    for spec in JOB.get("characters", {}).values():
        if spec.get("swap_material") and S["actors"].get(next((k for k, v in JOB["characters"].items()
                                                                if v is spec), None), (None,))[0] is a:
            first, by_name = EAL.load_asset(spec["swap_material"]), {}     # a test override
    sk = comp.get_skinned_asset()
    slots = [str(m.get_editor_property("material_slot_name")).lower() for m in sk.get_editor_property("materials")]
    for i in range(comp.get_num_materials()):
        m = by_name.get(slots[i] if i < len(slots) else "") or first
        if m is not None:
            comp.set_material(i, m)


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
    for i, g in enumerate(g_ for g_ in spec.get("garments", []) if not g_.startswith("@")):
        followers.append(add_follower(a, body, g, f"garment_{i}"))
    face_material(key, face)
    if hide_face:
        # a pure generated figure (its own head, hair and hands): the MetaHuman's face and grooms are not drawn
        # (they keep animating, so the face skeleton still measures eyes and head)
        for c in a.get_components_by_class(unreal.SkeletalMeshComponent):
            if c != body and c.get_name().lower() != "body":
                c.set_render_in_main_pass(False)
                c.set_cast_shadow(False)
                # (out of the main pass alone it still wrote depth: the hidden head stood as a black silhouette
                # wherever the figure's own head did not cover it, 11-14)
                try:
                    c.set_editor_property("render_in_depth_pass", False)
                except Exception as e:
                    note(f"render_in_depth_pass: {e}")
                try:
                    c.set_editor_property("visibility_based_anim_tick_option",
                                          unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
                except Exception:
                    pass
        for gc in a.get_components_by_class(unreal.GroomComponent):
            gc.set_visibility(False, True)
        # and whatever else the MetaHuman carries on its head (a groom's cap or cards as a mesh of their own): a
        # dark disc stood behind a whole figure's head and reached a key (11-14)
        left = []
        for pc in a.get_components_by_class(unreal.PrimitiveComponent):
            if pc == body or pc.get_name().lower() == "body" or isinstance(pc, unreal.SkeletalMeshComponent):
                continue
            try:
                pc.set_visibility(False, True)
                pc.set_hidden_in_game(True)
                left.append(f"{pc.get_class().get_name()}:{pc.get_name()}")
            except Exception:
                pass
        note(f"{key}: MetaHuman face and grooms hidden (pure figure); also hidden: {left}")
    if spec.get("hide_hair") and not hide_face:
        # the figure wears its own drawn hair (the fit kept it on the head): the MetaHuman's hair groom is not drawn;
        # its brows, lashes and beard are
        gone_ = []
        for gc in a.get_components_by_class(unreal.GroomComponent):
            nm = gc.get_name().lower()
            if "hair" in nm and "facial" not in nm:
                gc.set_visibility(False, True)
                gone_.append(gc.get_name())
        note(f"{key}: hair groom hidden (the figure's own hair): {gone_}; grooms: "
             f"{[g.get_name() for g in a.get_components_by_class(unreal.GroomComponent)]}")
    if spec.get("body_swap"):
        followers.append(add_follower(a, body, spec["body_swap"], "body_swap", skin=True))
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
        # the shared folder (skeletons, post-process Animation Blueprints, face rig) is written by the build too; left
        # unsaved, every later editor loads the body with no skeleton and no pose ever evaluates on it
        EAL.save_directory(b.common_folder_path, False, True)
        step("saved", common=len(EAL.list_assets(b.common_folder_path, recursive=True, include_folder=False)))
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


# cloth per baked slot (MB_<cloth>): weave map, its tile in cm (UV channel 1 is in centimetres), how strong the
# thread normal and occlusion are, roughness, specular, and either cloth fuzz (wool, linen) or anisotropy (silk);
# never both on one material. Starting values from the outfit research, tuned by eye.
CLOTH = {
    "wool": {"weave": "nap", "tile_cm": 4.0, "n": 0.35, "ao": 0.25, "rough": 0.92, "spec": 0.25, "fuzz": 0.65,
             "fuzz_color": [0.16, 0.16, 0.17]},
    "wool_lapel": {"weave": "nap", "tile_cm": 4.0, "n": 0.3, "ao": 0.25, "rough": 0.88, "spec": 0.25, "fuzz": 0.6,
                   "fuzz_color": [0.16, 0.16, 0.17]},
    "wool_vest": {"weave": "twill22", "tile_cm": 3.429, "n": 0.6, "ao": 0.4, "rough": 0.82, "spec": 0.3,
                  "fuzz": 0.35, "fuzz_color": [0.2, 0.2, 0.2]},
    "wool_trousers": {"weave": "twill22", "tile_cm": 3.429, "n": 0.6, "ao": 0.4, "rough": 0.74, "spec": 0.3,
                      "fuzz": 0.25, "fuzz_color": [0.45, 0.45, 0.44]},
    "cotton": {"weave": "plain", "tile_cm": 2.909, "n": 0.7, "ao": 0.45, "rough": 0.78, "spec": 0.35, "fuzz": 0.15,
               "fuzz_color": [0.9, 0.9, 0.88]},
    "starch": {"weave": "plain", "tile_cm": 2.909, "n": 0.2, "ao": 0.15, "rough": 0.45, "spec": 0.4, "fuzz": 0.0},
    "silk": {"weave": "satin5", "tile_cm": 1.667, "n": 0.25, "ao": 0.15, "rough": 0.5, "spec": 0.35, "aniso": 0.45},
    "leather": {"rough": 0.36, "spec": 0.5},
    "horn": {"rough": 0.3, "spec": 0.5},
    "thread": {"rough": 0.8, "spec": 0.3},
}


def cloth_material(path, texs, weave, c):
    """A two-sided material over the baked maps (UV 0) with a thread-scale weave (UV 1 in cm / tile): normals
    blended (whiteout), weave occlusion in colour and AO, roughness from the cloth with the weave's crown factor,
    low specular; Cloth shading with fuzz for wool and linen, anisotropy for silk."""
    MEL = unreal.MaterialEditingLibrary
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    folder, name = path.rsplit("/", 1)
    m = at.create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    X = {"x": -1200}

    def node(cls, **props):
        X["x"] += 40
        n = MEL.create_material_expression(m, cls, X["x"], 0)
        for k, v in props.items():
            n.set_editor_property(k, v)
        return n

    def link(a, ao, b, bi):
        MEL.connect_material_expressions(a, ao, b, bi)

    def const(v):
        if isinstance(v, (list, tuple)):
            return node(unreal.MaterialExpressionConstant3Vector, constant=unreal.LinearColor(*v, 1.0))
        return node(unreal.MaterialExpressionConstant, r=float(v))

    def op(cls, a, b, ao="", bo=""):
        n = node(cls)
        link(a, ao, n, "A")
        link(b, bo, n, "B")
        return n

    def samp(tex, st, uv=None):
        n = node(unreal.MaterialExpressionTextureSample, texture=tex, sampler_type=st)
        if uv is not None:
            link(uv, "", n, "UVs")
        return n
    S_COL, S_NRM, S_MSK = (unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL,
                           unreal.MaterialSamplerType.SAMPLERTYPE_MASKS)
    bc = samp(texs["BaseColor"], S_COL)
    nm = samp(texs["Normal"], S_NRM)
    rma = samp(texs["RMA"], S_MSK)
    P = unreal.MaterialProperty
    outs = {}
    w = c.get("weave")
    if w and w in weave:
        tc = node(unreal.MaterialExpressionTextureCoordinate, coordinate_index=1)
        uv = op(unreal.MaterialExpressionMultiply, tc, const(1.0 / c["tile_cm"]))
        wn = samp(weave[w]["N"], S_NRM, uv)
        wm = samp(weave[w]["M"], S_MSK, uv)
        # weave normal toward flat by (1 - strength), then whiteout blend with the baked normal
        wl = node(unreal.MaterialExpressionLinearInterpolate)
        link(const([0.0, 0.0, 1.0]), "", wl, "A")
        link(wn, "RGB", wl, "B")
        link(const(c.get("n", 0.5)), "", wl, "Alpha")
        rg_a = node(unreal.MaterialExpressionComponentMask, r=True, g=True, b=False, a=False)
        link(nm, "RGB", rg_a, "")
        rg_b = node(unreal.MaterialExpressionComponentMask, r=True, g=True, b=False, a=False)
        link(wl, "", rg_b, "")
        b_a = node(unreal.MaterialExpressionComponentMask, r=False, g=False, b=True, a=False)
        link(nm, "RGB", b_a, "")
        b_b = node(unreal.MaterialExpressionComponentMask, r=False, g=False, b=True, a=False)
        link(wl, "", b_b, "")
        xy = op(unreal.MaterialExpressionAdd, rg_a, rg_b)
        zz = op(unreal.MaterialExpressionMultiply, b_a, b_b)
        ap = op(unreal.MaterialExpressionAppendVector, xy, zz)
        nn = node(unreal.MaterialExpressionNormalize)
        link(ap, "", nn, "VectorInput")
        outs["MP_NORMAL"] = (nn, "")
        # thread occlusion: lerp(1, weave AO, k)
        aol = node(unreal.MaterialExpressionLinearInterpolate)
        link(const(1.0), "", aol, "A")
        link(wm, "G", aol, "B")
        link(const(c.get("ao", 0.3)), "", aol, "Alpha")
        col = op(unreal.MaterialExpressionMultiply, bc, aol, "RGB", "")
        outs["MP_BASE_COLOR"] = (col, "")
        ao = op(unreal.MaterialExpressionMultiply, rma, aol, "B", "")
        outs["MP_AMBIENT_OCCLUSION"] = (ao, "")
        rf = op(unreal.MaterialExpressionMultiply, wm, const(2.0), "B", "")
        rl = node(unreal.MaterialExpressionLinearInterpolate)
        link(const(1.0), "", rl, "A")
        link(rf, "", rl, "B")
        link(const(0.6), "", rl, "Alpha")
        rough = op(unreal.MaterialExpressionMultiply, rl, const(c.get("rough", 0.8)))
    else:
        outs["MP_BASE_COLOR"] = (bc, "RGB")
        outs["MP_NORMAL"] = (nm, "RGB")
        outs["MP_AMBIENT_OCCLUSION"] = (rma, "B")
        rough = const(c.get("rough", 0.8))
    outs["MP_ROUGHNESS"] = (rough, "")
    outs["MP_SPECULAR"] = (const(c.get("spec", 0.4)), "")
    outs["MP_METALLIC"] = (const(0.0), "")
    if c.get("fuzz", 0) > 0:
        m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_CLOTH)
        outs["MP_SUBSURFACE_COLOR"] = (const(c.get("fuzz_color", [0.2, 0.2, 0.2])), "")
        outs["CustomData0"] = (const(c["fuzz"]), "")    # the cloth amount: only Make Material Attributes has it
    elif c.get("aniso"):
        outs["MP_ANISOTROPY"] = (const(c["aniso"]), "")
    pins = {"MP_BASE_COLOR": "BaseColor", "MP_NORMAL": "Normal", "MP_ROUGHNESS": "Roughness",
            "MP_SPECULAR": "Specular", "MP_METALLIC": "Metallic", "MP_AMBIENT_OCCLUSION": "AmbientOcclusion",
            "MP_SUBSURFACE_COLOR": "SubsurfaceColor", "MP_ANISOTROPY": "Anisotropy", "CustomData0": "CustomData0"}
    if "CustomData0" in outs:
        mma = node(unreal.MaterialExpressionMakeMaterialAttributes)
        for k, (n_, o_) in outs.items():
            ok = MEL.connect_material_expressions(n_, o_, mma, pins[k])
            if not ok and k == "CustomData0":
                ok = MEL.connect_material_expressions(n_, o_, mma, "ClearCoat")
            if not ok:
                note(f"{path}: pin {pins[k]} not connected")
        m.set_editor_property("use_material_attributes", True)
        MEL.connect_material_property(mma, "", P.MP_MATERIAL_ATTRIBUTES)
    else:
        for k, (n_, o_) in outs.items():
            MEL.connect_material_property(n_, o_, getattr(P, k))
    m.set_editor_property("two_sided", True)
    m.set_editor_property("used_with_skeletal_mesh", True)
    MEL.recompile_material(m)
    EAL.save_loaded_asset(m, False)
    return m


def import_weave(wdir, dest):
    out = {}
    for fn in sorted(os.listdir(wdir)):
        if not fn.endswith(".png"):
            continue
        kind, ch = fn[len("T_weave_"):-4].rsplit("_", 1)
        tex = import_texture(os.path.join(wdir, fn), dest, "normal" if ch == "N" else "mask")
        tex.set_editor_property("address_x", unreal.TextureAddress.TA_WRAP)
        tex.set_editor_property("address_y", unreal.TextureAddress.TA_WRAP)
        tex.set_editor_property("never_stream", True)
        EAL.save_loaded_asset(tex, False)
        out.setdefault(kind, {})[ch] = tex
    return out


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
    weave = import_weave(spec["weave"], dest + "/Weave") if spec.get("weave") else {}
    cloth = dict(CLOTH, **spec.get("cloth", {}))
    made = {}
    sk = EAL.load_asset(r["outfit"])
    mats = []
    for sm in sk.get_editor_property("materials"):       # the array holds copies: rebuild it
        sm2 = unreal.SkeletalMaterial()
        slot = str(sm.get_editor_property("material_slot_name"))
        key = slot[3:] if slot.startswith("MB_") else None
        if "skin" in slot.lower():
            # a skin slot (hands carried in the outfit) keeps its own material; refresh() gives it the body's skin
            sm2.set_editor_property("material_interface", sm.get_editor_property("material_interface"))
        elif spec.get("cloth_slots", True) and key in cloth:
            if key not in made:
                made[key] = cloth_material(f"{dest}/M_Cloth_{key}", texs, weave, cloth[key])
            sm2.set_editor_property("material_interface", made[key])
        else:
            sm2.set_editor_property("material_interface", mi)
        sm2.set_editor_property("material_slot_name", sm.get_editor_property("material_slot_name"))
        mats.append(sm2)
    sk.set_editor_property("materials", mats)
    EAL.save_loaded_asset(sk, False)
    r["cloth"] = {k: v.get_path_name() for k, v in made.items()}
    if made:
        JOB["garment_material"] = None          # the slots carry their own cloth now
    r["time_s"] = round(time.time() - t0, 1)
    S["r"]["import_outfit"] = r
    log(f"import_outfit {json.dumps(r)}")
    for spec_c in JOB.get("characters", {}).values():
        spec_c["garments"] = [r["outfit"] if g == "@outfit" else g for g in spec_c.get("garments", [])]
        if spec_c.get("body_swap") == "@trim":
            spec_c["body_swap"] = r.get("trim")


def pose_anim(skel, name, rots):
    """A two-frame body pose built from the skeleton's reference pose: each [bone, parent, degrees] turns that bone
    about the character's lateral axis through its own joint (thighs forward and calves back = seated). The sign that
    brings a thigh's child forward (a calf's child down) is chosen by testing both, so axis conventions cannot flip
    the pose. Returns the AnimSequence path."""
    AP = unreal.AnimPoseExtensions
    WS, LS = unreal.AnimPoseSpaces.WORLD, unreal.AnimPoseSpaces.LOCAL
    pose = AP.get_reference_pose(skel)
    world = {}
    local = {}

    def W(b):
        if b not in world:
            world[b] = AP.get_bone_pose(pose, b, WS)
        return world[b]

    def L(b):
        if b not in local:
            local[b] = AP.get_bone_pose(pose, b, LS)
        return local[b]
    fwd = W("ball_l").translation - W("foot_l").translation
    fwd.z = 0
    fwd = fwd.normal()
    lat = unreal.Vector(fwd.y, -fwd.x, 0.0)
    child = {"thigh_l": "calf_l", "thigh_r": "calf_r", "calf_l": "foot_l", "calf_r": "foot_r",
             "upperarm_l": "lowerarm_l", "upperarm_r": "lowerarm_r", "lowerarm_l": "hand_l", "lowerarm_r": "hand_r"}
    out = {}
    for rot in rots:
        b, parent, deg = rot[:3]
        # axis: "lat" (default; pitch: thighs forward, elbows bend forward) or "fwd" (roll: arms down to the sides,
        # the sign that lowers the child is taken)
        axis_name = rot[3] if len(rot) > 3 else "lat"
        ax_v = fwd if axis_name == "fwd" else lat
        pw = W(parent)
        bw = unreal.MathLibrary.compose_transforms(L(b), pw)       # current world (parent may have moved)
        best = None
        for sgn in (1, -1):
            a = math.radians(deg * sgn) / 2
            q = unreal.Quat(ax_v.x * math.sin(a), ax_v.y * math.sin(a), ax_v.z * math.sin(a), math.cos(a))
            nw = unreal.Transform(bw.translation, unreal.Rotator(), bw.scale3d)
            nw.rotation = q * bw.rotation
            c = child.get(b)
            if c:
                cp = unreal.MathLibrary.compose_transforms(L(c), nw).translation - nw.translation
                score = -cp.z if axis_name == "fwd" else (cp.dot(fwd) if deg > 0 else -cp.z)
            else:
                score = 0
            if best is None or score > best[0]:
                best = (score, nw)
        nw = best[1]
        world[b] = nw
        out[b] = unreal.MathLibrary.compose_transforms(nw, unreal.MathLibrary.invert_transform(pw))
        local[b] = out[b]
        for c in [k for k, v in ((r_[0], r_[1]) for r_ in rots) if v == b]:
            world.pop(c, None)
    folder = "/Game/Char/Anim"
    path = f"{folder}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    f = unreal.AnimSequenceFactory()
    f.set_editor_property("target_skeleton", skel)
    anim = at.create_asset(name, folder, unreal.AnimSequence, f)
    ctrl = anim.get_editor_property("controller")
    ctrl.open_bracket("pose", False)
    ctrl.set_frame_rate(unreal.FrameRate(30, 1), False)
    ctrl.set_number_of_frames(unreal.FrameNumber(2), False)
    for b, t in out.items():
        for fn in ("add_bone_curve", "add_bone_track"):
            try:
                getattr(ctrl, fn)(b, False)
                break
            except Exception:
                continue
        pos, rot, scl = [t.translation] * 3, [t.rotation] * 3, [t.scale3d] * 3
        try:
            ctrl.set_bone_track_keys(b, pos, rot, scl, False)
        except Exception:
            ctrl.set_bone_track_keys(b, [unreal.Vector3f(v.x, v.y, v.z) for v in pos],
                                     [unreal.Quat4f(q.x, q.y, q.z, q.w) for q in rot],
                                     [unreal.Vector3f(v.x, v.y, v.z) for v in scl], False)
    ctrl.close_bracket(False)
    EAL.save_loaded_asset(anim, False)
    return path


def face_from_body_anim(body, face, curves, name):
    """A two-frame face-skeleton AnimSequence that holds the body component's CURRENT pose for every bone the two
    skeletons share (local, parent-bone space, as the body evaluates it now, retargeting included) plus the face
    expression curves. Used for expression stills: a leader-pose follower carries the body pose exactly but not the
    face's own expression curves (measured: jawOpen on a follower left the jaw bone unmoved)."""
    folder = "/Game/Char/Anim"
    path = f"{folder}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    f = unreal.AnimSequenceFactory()
    f.set_editor_property("target_skeleton", face.get_skinned_asset().get_editor_property("skeleton"))
    anim = at.create_asset(name, folder, unreal.AnimSequence, f)
    ctrl = anim.get_editor_property("controller")
    ctrl.open_bracket("face_from_body", False)
    ctrl.set_frame_rate(unreal.FrameRate(30, 1), False)
    ctrl.set_number_of_frames(unreal.FrameNumber(2), False)
    parent_space = unreal.RelativeTransformSpace.RTS_PARENT_BONE_SPACE
    n = 0
    for i in range(face.get_num_bones()):
        b = face.get_bone_name(i)
        if body.get_bone_index(b) < 0:
            continue
        t = body.get_socket_transform(b, parent_space)
        for fn in ("add_bone_curve", "add_bone_track"):
            try:
                getattr(ctrl, fn)(b, False)
                break
            except Exception:
                continue
        pos, rot, scl = [t.translation] * 3, [t.rotation] * 3, [t.scale3d] * 3
        try:
            ctrl.set_bone_track_keys(b, pos, rot, scl, False)
        except Exception:
            ctrl.set_bone_track_keys(b, [unreal.Vector3f(v.x, v.y, v.z) for v in pos],
                                     [unreal.Quat4f(q.x, q.y, q.z, q.w) for q in rot],
                                     [unreal.Vector3f(v.x, v.y, v.z) for v in scl], False)
        n += 1
    for cname, val in (curves or {}).items():
        unreal.AnimationLibrary.add_curve(anim, cname, unreal.RawCurveTrackTypes.RCT_FLOAT, False)
        for tt in (0.0, 1.0 / 30, 2.0 / 30):
            unreal.AnimationLibrary.add_float_curve_key(anim, cname, tt, float(val))
    ctrl.close_bracket(False)
    return anim, n


def turned_body_anim(src_anim, t, turns, name):
    """A two-frame body AnimSequence: src_anim's pose at time t with extra turns through the named joints, applied
    in order: {"neck_02": deg, "head": deg} turns about the world vertical (a head turned while crouching);
    {"thigh_l": [deg, "calf_l", "fwd"], "calf_l": [deg, "foot_l", "down"]} turns about the body's lateral axis, the
    sign chosen so that the named child joint ends up further forward (+Y, where the character faces) or lower
    (seated: thighs forward, shins down)."""
    AP = unreal.AnimPoseExtensions
    WS, LS = unreal.AnimPoseSpaces.WORLD, unreal.AnimPoseSpaces.LOCAL
    pose = AP.get_anim_pose_at_time(src_anim, float(t), unreal.AnimPoseEvaluationOptions())
    bones = [str(b) for b in AP.get_bone_names(pose)]
    local = {b: AP.get_bone_pose(pose, b, LS) for b in bones}

    def setp(tr, b):            # children follow their parent (not adjusted); the binding returns the pose
        nonlocal pose
        res = AP.set_bone_pose(pose, tr, b, LS)
        if res is not None:
            pose = res

    def turned(w, axis, deg):
        a = math.radians(deg) / 2
        q = unreal.Quat(axis[0] * math.sin(a), axis[1] * math.sin(a), axis[2] * math.sin(a), math.cos(a))
        nw = unreal.Transform(w.translation, unreal.Rotator(), w.scale3d)
        nw.rotation = q * w.rotation
        return nw

    for b, spec in turns.items():
        w = AP.get_bone_pose(pose, b, WS)
        parent_w = unreal.MathLibrary.compose_transforms(unreal.MathLibrary.invert_transform(local[b]), w)
        if isinstance(spec, (int, float)):
            nw = turned(w, (0, 0, 1), spec)
        else:
            deg, child, goal = spec
            cl = local[child]
            best = None
            for sgn in (1, -1):
                cand = turned(w, (1, 0, 0), sgn * deg)
                cp = unreal.MathLibrary.compose_transforms(cl, cand).translation
                score = cp.y if goal == "fwd" else -cp.z
                if best is None or score > best[0]:
                    best = (score, cand)
            nw = best[1]
        local[b] = unreal.MathLibrary.compose_transforms(nw, unreal.MathLibrary.invert_transform(parent_w))
        setp(local[b], b)
    folder = "/Game/Char/Anim"
    path = f"{folder}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    f = unreal.AnimSequenceFactory()
    f.set_editor_property("target_skeleton", src_anim.get_editor_property("skeleton"))
    anim = at.create_asset(name, folder, unreal.AnimSequence, f)
    ctrl = anim.get_editor_property("controller")
    ctrl.open_bracket("turned", False)
    ctrl.set_frame_rate(unreal.FrameRate(30, 1), False)
    ctrl.set_number_of_frames(unreal.FrameNumber(2), False)
    for b in bones:
        tr = local[b]
        for fn in ("add_bone_curve", "add_bone_track"):
            try:
                getattr(ctrl, fn)(b, False)
                break
            except Exception:
                continue
        pos, rot, scl = [tr.translation] * 3, [tr.rotation] * 3, [tr.scale3d] * 3
        try:
            ctrl.set_bone_track_keys(b, pos, rot, scl, False)
        except Exception:
            ctrl.set_bone_track_keys(b, [unreal.Vector3f(v.x, v.y, v.z) for v in pos],
                                     [unreal.Quat4f(q_.x, q_.y, q_.z, q_.w) for q_ in rot],
                                     [unreal.Vector3f(v.x, v.y, v.z) for v in scl], False)
    ctrl.close_bracket(False)
    EAL.save_loaded_asset(anim, False)
    return path


def pose_face(sh, info):
    """Make the face (and so the hair, brows, lashes, eyes and teeth bound to it) carry the body's pose.

    The assembled Blueprint's Face is only ATTACHED to the Body component: the runtime MetaHuman component that
    would make it a leader-pose follower of the Body does not run in an editor world, and the face AnimBP does not
    copy the body pose there, so in any body pose the face stayed in its reference pose relative to the actor
    (measured on the deep knee bend: body head at z 141.9, face head at z 169.8). Without face curves the face
    becomes a leader-pose follower of the body (exact, also while a clip plays); with expression curves it plays a
    face anim that holds the body's current pose plus the curves (stills)."""
    a = S["actors"][sh["char"]][0]
    body, face = skel_comps(a)
    if face is None or body is None:
        return
    curves = sh.get("face_curves")
    if curves or sh.get("face_anim"):
        face.set_leader_pose_component(None, True, False)
        body, face = skel_comps(a)        # a Blueprint actor may rebuild its components on a property change
        if sh.get("face_anim"):
            # a face-only clip (no body bone tracks) cannot carry the body pose: follow the body by leader pose
            # is impossible together with it, so warn when the body is posed
            if sh.get("body_anim"):
                note(f"{sh['name']}: face_anim with a posed body; the face keeps the clip's own head pose")
            apply_anim(face, EAL.load_asset(sh["face_anim"]), sh.get("face_t", 0.0))
        else:
            anim, n = face_from_body_anim(body, face, curves, f"AS_{sh['name']}")
            body, face = skel_comps(a)
            apply_anim(face, anim, 1.0 / 30)
            S["face_anim_last"] = anim
            info["face_bones_from_body"] = n
    else:
        clear_anim(face)
        face.set_leader_pose_component(body, True, False)
    info["face_follow"] = "anim" if (curves or sh.get("face_anim")) else "leader"


def head_error_mm(key):
    """Distance between the head and neck bones as the Body and as the Face component place them, in mm."""
    a = S["actors"][key][0]
    body, face = skel_comps(a)
    out = {}
    for b in ("neck_02", "head"):
        try:
            out[b] = round((body.get_socket_location(b) - face.get_socket_location(b)).length() * 10, 2)
        except Exception as e:
            out[b] = str(e)
    return out


def hands_check(key):
    """Both hands visible and attached: the mesh that shows the hands (the trimmed copy when the body is swapped,
    else the body) is drawn, wears skin (never the garment material), and puts hand_l / hand_r within 1 mm of
    where the body places them. Returns (ok, details)."""
    a = S["actors"][key][0]
    body, _ = skel_comps(a)
    sw = S.get("swap", {}).get(id(a))
    if sw is None:      # hands carried in a garment (a skin slot), else the body itself shows them
        sw = next((f for f in S.get("followers", {}).get(id(a), []) if skin_slots(f.skeletal_mesh_component)), None)
    comp = sw.skeletal_mesh_component if sw is not None else body
    d = {"mesh": comp.get_skinned_asset().get_path_name() if comp.get_skinned_asset() else None}
    ok = d["mesh"] is not None and comp.is_visible() and (sw is None or not sw.is_hidden_ed())
    d["visible"] = bool(ok)
    if JOB.get("debug_hands"):
        dbg = {}
        for nm, c_, ac in (("swap", comp, sw), ("garment0", (S.get("followers", {}).get(id(a)) or [sw])[0].skeletal_mesh_component, None)):
            x = {}
            for prop in ("render_in_main_pass", "render_in_depth_pass", "hidden_in_game", "visible", "owner_no_see",
                         "only_owner_see", "bounds_scale", "use_bounds_from_leader_pose_component", "cast_shadow"):
                try:
                    x[prop] = str(c_.get_editor_property(prop))
                except Exception as ex:
                    x[prop] = "ERR"
            try:
                o, e = c_.get_owner().get_actor_bounds(False)
                x["actor_bounds"] = [[round(o.x, 1), round(o.y, 1), round(o.z, 1)], [round(e.x, 1), round(e.y, 1), round(e.z, 1)]]
            except Exception as ex:
                x["actor_bounds"] = str(ex)
            try:
                lb = c_.get_skinned_asset().get_bounds()
                x["mesh_bounds"] = [[round(lb.origin.x, 1), round(lb.origin.y, 1), round(lb.origin.z, 1)],
                                    [round(lb.box_extent.x, 1), round(lb.box_extent.y, 1), round(lb.box_extent.z, 1)]]
            except Exception as ex:
                x["mesh_bounds"] = str(ex)
            try:
                x["leader"] = c_.get_editor_property("leader_pose_component").get_name()
            except Exception as ex:
                x["leader"] = str(ex)
            try:
                x["comp_loc"] = str(c_.get_world_location())
                x["comp_scale"] = str(c_.get_world_scale())
            except Exception as ex:
                pass
            dbg[nm] = x
        d["debug"] = dbg
    gm = JOB.get("garment_material") or "\0"
    mats = [m.get_path_name() if m else None for m in comp.get_materials()]
    if sw is not None and skin_slots(comp):
        mats = [mats[i] for i in skin_slots(comp)]
    d["materials"] = mats
    d["skin"] = bool(mats) and all(m is not None and not m.startswith(gm.split(".")[0]) for m in mats)
    ok = ok and d["skin"]
    for b in ("hand_l", "hand_r"):
        try:
            if comp.get_bone_index(b) < 0:
                raise ValueError("no bone")
            e = round((comp.get_socket_location(b) - body.get_socket_location(b)).length() * 10, 2)
        except Exception as ex:
            e = str(ex)
        d[b + "_mm"] = e
        ok = ok and isinstance(e, float) and e <= 1.0
    # the fingers must hang off the hand (a follower skinned to bones its leader does not evaluate collapses them
    # toward the origin while the hand bone itself still agrees)
    for f, h in (("middle_03_l", "hand_l"), ("middle_03_r", "hand_r"), ("thumb_03_l", "hand_l"),
                 ("thumb_03_r", "hand_r")):
        try:
            e = round((comp.get_socket_location(f) - comp.get_socket_location(h)).length(), 1)
        except Exception as ex:
            e = str(ex)
        d[f + "_cm"] = e
        ok = ok and isinstance(e, float) and 3.0 <= e <= 25.0
    return bool(ok), d


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
    try:
        comp.set_component_tick_enabled(True)
        comp.set_update_animation_in_editor(True)
    except Exception:
        pass
    comp.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    try:
        comp.override_animation_data(anim, False, False, float(t), 0.0)
    except Exception:
        comp.set_animation(anim)
    comp.set_animation(anim)
    m = JOB.get("anim_method", "stop")
    if m == "stop":
        comp.play(False)
        comp.set_position(float(t), False)
        comp.set_play_rate(0.0)
        comp.stop()
        comp.set_position(float(t), False)
    elif m == "rate0":
        comp.play(False)
        comp.set_play_rate(0.0)
        comp.set_position(float(t), True)
    else:
        comp.play(True)
        comp.set_play_rate(0.0001)
        comp.set_position(float(t), True)
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
    res = CUR_RES[0] if CUR_RES else RES
    fb.sensor_width, fb.sensor_height = 36.0, 36.0 * res[1] / res[0]
    cc.set_editor_property("filmback", fb)
    cc.set_editor_property("current_focal_length", focal)
    fs = cc.get_editor_property("focus_settings")
    fs.focus_method = unreal.CameraFocusMethod.MANUAL
    fs.manual_focus_distance = dist
    cc.set_editor_property("focus_settings", fs)
    cc.set_editor_property("current_aperture", 5.6)
    return cam


def project_joints(cam, body, focal=None):
    """Body joints in the still's pixels (pinhole: the camera's 36 mm wide filmback across the image width)."""
    cc = cam.get_cine_camera_component()
    f = float(cc.get_editor_property("current_focal_length"))
    loc = cam.get_actor_location()
    fw, rt, up = cam.get_actor_forward_vector(), cam.get_actor_right_vector(), cam.get_actor_up_vector()
    out = {}
    for b in ("head", "neck_01", "neck_02", "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r", "spine_05",
              "spine_04", "spine_03"):
        try:
            q = body.get_socket_location(b)
        except Exception:
            continue
        d = q - loc
        dot = unreal.MathLibrary.dot_vector_vector
        z = dot(d, fw)
        if z <= 1:
            continue
        out[b] = [round(RES[0] / 2 + f * dot(d, rt) / z / 36.0 * RES[0], 1),
                  round(RES[1] / 2 - f * dot(d, up) / z / 36.0 * RES[0], 1), round(z, 1)]
    return out


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
    set_skin_slots(a, body)
    if spec.get("body_swap"):
        set_swap_materials(a, body)
    if spec.get("body_swap") or spec.get("hide_body"):
        body.set_editor_property("visibility_based_anim_tick_option",
                                 unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
        if spec.get("hide_mode", "visibility") == "visibility":
            body.set_visibility(False, False)      # still posed (always tick), so followers keep following
        else:
            body.set_visibility(True, False)
            body.set_render_in_main_pass(False)
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
    face_material(key, face)
    S["actors"][key] = (a, body, face)
    return a, body, face


def aim_point(sh):
    """World location of the bone a shot aims at ("aim_bone", e.g. hand_r for a hands close-up), from the body."""
    a = S["actors"][sh["char"]][0]
    body, _ = skel_comps(a)
    p = body.get_socket_location(sh["aim_bone"])
    off = sh.get("aim_offset", [0, 0, 0])
    return p + unreal.Vector(*off)


def full_focal(sh, head):
    """Full-length lens: 35 mm by default; with "fill": true the figure (crown at head + 12%) fills ~92% of the
    frame height at the shot's distance (4.2 m unless "dist"), whatever the still's aspect."""
    if not sh.get("fill"):
        return sh.get("focal", 35)
    res = sh.get("res", RES)
    sensor_h = 36.0 * res[1] / res[0]
    return sensor_h * sh.get("dist", 420.0) / (head.z * 1.12 / 0.92)


def frame_camera(sh, head):
    """(location, rotation, focal) of a shot's camera for the head position (or the shot's aim bone)."""
    fr, az, el = sh.get("frame", "close"), sh.get("az", 0), sh.get("el", 3)
    focal = None
    if sh.get("aim_bone"):
        target, dist, focal = aim_point(sh), sh.get("dist", 120), sh.get("focal", 50)
    elif fr == "close":                 # head and shoulders: aim near the eyes (the head joint sits low in the skull)
        target, dist, focal = head + unreal.Vector(0, 0, sh.get("dz", 1)), sh.get("dist", 230), 85
    elif fr == "medium":
        target, dist, focal = head + unreal.Vector(0, 0, -38), 260, 50
    else:
        target, dist, focal, el = (unreal.Vector(head.x, head.y, head.z * 0.52), sh.get("dist", 420),
                                   full_focal(sh, head), sh.get("el", 4))
    a, e = math.radians(az), math.radians(el)
    loc = target + unreal.Vector(math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)) * dist
    return loc, unreal.MathLibrary.find_look_at_rotation(loc, target), target, focal


def reframe(cam, sh, info):
    """Re-aim a shot's camera at the head where the posed body now holds it (the pose takes effect a step after
    the camera was placed)."""
    a = S["actors"][sh["char"]][0]
    body, face = skel_comps(a)
    try:
        head = face.get_socket_location("head")
    except Exception:
        return
    if head.z < 30:
        return
    loc, rot, _, _ = frame_camera(sh, head)
    cam.set_actor_location(loc, False, False)
    cam.set_actor_rotation(rot, False)
    info["head"] = [round(head.x, 1), round(head.y, 1), round(head.z, 1)]


def setup_shot(sh):
    CUR_RES[:] = [sh["res"]] if sh.get("res") else []
    character_actor(sh["char"])
    a, body, face = refresh(sh["char"])
    show_only(sh["char"])
    info = {"name": sh["name"]}
    if sh.get("body_pose") and not sh.get("body_anim"):
        # e.g. {"name": "AS_seated", "rots": [["thigh_l", "pelvis", 85], ["calf_l", "thigh_l", -85], ...]}
        bp = sh["body_pose"]
        # one asset per pose per job: deleting an AnimSequence the body is playing (to rebuild it for the next
        # shot) leaves the component without an instance ("ObjectInstance is null")
        key = json.dumps([bp["name"], bp["rots"]])
        if key not in S.setdefault("poses", {}):
            S["poses"][key] = pose_anim(body.get_skinned_asset().get_editor_property("skeleton"), bp["name"],
                                        bp["rots"])
        sh["body_anim"] = S["poses"][key]
        sh["body_t"] = 0.0
    if sh.get("body_anim"):
        banim = EAL.load_asset(sh["body_anim"])
        t = sh.get("body_t", 0.0)
        if sh.get("body_scan"):
            t, score, length = scan_body(banim, sh["body_scan"])
            info["body_scan"] = {"t": round(t, 3), "score": round(score, 2), "length": round(length, 2)}
        if sh.get("turn"):
            sh["body_anim"] = turned_body_anim(banim, t, sh["turn"], f"AS_{sh['name']}_turn")
            banim, t = EAL.load_asset(sh["body_anim"]), 0.0
            info["turn"] = sh["turn"]
        apply_anim(body, banim, t)
        info["body"] = [sh["body_anim"], round(t, 3)]
    elif body is not None:
        clear_anim(body)
    if face is not None:
        pose_face(sh, info)
        info["face"] = sh.get("face_curves") or sh.get("face_anim")
        body, face = skel_comps(a)
    try:
        head = face.get_socket_location("head")
        if head.z < 50:
            raise ValueError
    except Exception:
        head = unreal.Vector(0, 0, 165)
    info["head"] = [round(head.x, 1), round(head.y, 1), round(head.z, 1)]
    fr, az, el = sh.get("frame", "close"), sh.get("az", 0), sh.get("el", 3)
    if sh.get("aim_bone"):
        cam = camera(sh["name"], aim_point(sh), az, el, sh.get("dist", 120), sh.get("focal", 50))
    elif fr == "close":               # head and shoulders: aim near the eyes (the head joint sits low in the skull)
        cam = camera(sh["name"], head + unreal.Vector(0, 0, sh.get("dz", 1)), az, el, sh.get("dist", 230), 85)
    elif fr == "medium":
        cam = camera(sh["name"], head + unreal.Vector(0, 0, -38), az, el, 260, 50)
    else:
        cam = camera(sh["name"], unreal.Vector(head.x, head.y, head.z * 0.52), az, sh.get("el", 4),
                     sh.get("dist", 420), full_focal(sh, head))
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
            # a paused single-node pose shows only from the second application (measured: one shot late); the
            # face is posed again from the body as it now stands
            a = S["actors"][sh["char"]][0]
            body, face = skel_comps(a)
            if sh.get("body_anim") and "body" in info:
                apply_anim(body, EAL.load_asset(sh["body_anim"]), info["body"][1])
            yield from frames(10)
            pose_face(sh, info)
            yield from frames(10)
        yield from frames(JOB.get("settle_frames", 40))
        # verify before capture: the head and neck must sit where the body puts them (1 mm); re-pose the face and
        # wait if not, and refuse the still if they never agree
        for attempt in range(JOB.get("head_check_tries", 6)):
            err = head_error_mm(sh["char"])
            if all(isinstance(v, float) and v <= 1.0 for v in err.values()):
                break
            note(f"{sh['name']}: face off the body {err} mm; re-posing (try {attempt + 1})")
            pose_face(sh, info)
            yield from frames(15)
        info["head_err_mm"] = err
        # both hands visible and attached (the trimmed copy wears skin and follows the body); one refresh to
        # re-wire it if not, then refuse the still
        hok, hands = hands_check(sh["char"])
        if not hok and not sh.get("hands_optional"):
            note(f"{sh['name']}: hands not right {hands}; refreshing")
            refresh(sh["char"])
            yield from frames(15)
            hok, hands = hands_check(sh["char"])
        info["hands"] = hands
        if JOB.get("debug_comps"):
            a_ = S["actors"][sh["char"]][0]
            lst = []
            for c_ in a_.get_components_by_class(unreal.SkeletalMeshComponent):
                d_ = {"name": c_.get_name(), "mesh": c_.get_skinned_asset().get_name() if c_.get_skinned_asset() else None}
                for f_ in ("is_registered", "is_visible"):
                    try:
                        d_[f_] = getattr(c_, f_)()
                    except Exception as ex:
                        d_[f_] = str(ex)[:40]
                try:
                    d_["mode"] = str(c_.get_animation_mode())
                    d_["hand_r"] = str(c_.get_socket_location("hand_r"))[-60:]
                except Exception as ex:
                    d_["mode"] = str(ex)[:60]
                lst.append(d_)
            info["comps"] = lst
        reframe(cam, sh, info)
        yield from frames(5)
        if not all(isinstance(v, float) and v <= 1.0 for v in err.values()):
            raise RuntimeError(f"{sh['name']}: face does not follow the body ({err} mm); still refused")
        if not hok and not sh.get("hands_optional"):
            raise RuntimeError(f"{sh['name']}: hands not visible and attached ({hands}); still refused")
        path = os.path.join(JOB["out"], f"{sh['name']}.png")
        if os.path.exists(path):
            os.remove(path)
        res = sh.get("res", RES)
        unreal.AutomationLibrary.take_high_res_screenshot(res[0], res[1], path, camera=cam, force_game_view=True)
        t1 = time.time()
        while not (os.path.exists(path) and os.path.getsize(path) > 0):
            if time.time() - t1 > 90:
                raise RuntimeError(f"screenshot not written: {path}")
            yield
        info["path"], info["t_s"] = path, round(time.time() - S["t0"], 1)
        if sh.get("probe_face"):
            # the same still with the face mesh left out of the main pass (its shadows kept): the pixels that differ
            # are where the head mesh is seen; the body joints projected to the still say where the shoulders are
            # the components as they are now: posing the face (pose_face) can make a Blueprint rebuild them
            a = S["actors"][sh["char"]][0]
            body, face = skel_comps(a)
            info["joints_px"] = project_joints(cam, body, sh.get("focal_mm"))
            if face is not None:
                face.set_render_in_main_pass(False)
                yield from frames(JOB.get("settle_frames", 40))
                p2 = os.path.join(JOB["out"], f"{sh['name']}_noface.png")
                if os.path.exists(p2):
                    os.remove(p2)
                unreal.AutomationLibrary.take_high_res_screenshot(RES[0], RES[1], p2, camera=cam, force_game_view=True)
                t1 = time.time()
                while not (os.path.exists(p2) and os.path.getsize(p2) > 0):
                    if time.time() - t1 > 90:
                        raise RuntimeError(f"screenshot not written: {p2}")
                    yield
                face.set_render_in_main_pass(True)
                info["noface_path"] = p2
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
