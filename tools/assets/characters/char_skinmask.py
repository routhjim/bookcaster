"""Engine job: a character's own masked copy of its face skin material, so the head mesh's skin under the clothes is
hidden (the fit bakes the mask: img3d/bl_skinmask.py).

job: {"result": JSON, "face_folder": the MetaHuman's folder (its face mesh is found there), "mask_png": the baked
mask (head UV 0; white = skin kept, black = hidden), "dest": a folder of the character's own (e.g. /Game/Char/X/Skin),
"blueprint": the MetaHuman's assembled Blueprint (optional: a copy of it is made whose face wears the masked skin),
"name": the copy's name suffix}

Steps: import the mask as a linear texture; duplicate the face skin's base material (never edited: the copy is the
character's own) into dest; set the copy's blend mode to masked with the mask's red channel as its opacity mask;
make a material instance of the copy carrying every parameter value of the face's LOD 0 skin instance and the mask.
The result names the instance and the Blueprint copy: the character entry's "blueprint" is the copy (the scene tool
rebuilds face components, so a material set on a spawned face does not last there), and the garment token
"@face_material=<instance path>" (char_render.py) sets it on the studio renders' faces too.
"""
import json
import os

import unreal

EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary
AT = unreal.AssetToolsHelpers.get_asset_tools()


PASS_ATTRS = ("BaseColor", "Metallic", "Specular", "Roughness", "Anisotropy", "EmissiveColor", "Opacity", "OpacityMask",
              "Normal", "Tangent", "WorldPositionOffset", "SubsurfaceColor", "ClearCoat", "ClearCoatRoughness",
              "AmbientOcclusion", "Refraction", "CustomizedUV0", "CustomizedUV1", "CustomizedUV2", "CustomizedUV3",
              "PixelDepthOffset", "ShadingModel", "Displacement")


def face_skin(folder):
    for a in EAL.list_assets(folder, recursive=True, include_folder=False):
        if "FaceMesh" not in a:
            continue
        o = EAL.load_asset(a)
        if not isinstance(o, unreal.SkeletalMesh):
            continue
        for sm in o.get_editor_property("materials"):
            mi = sm.get_editor_property("material_interface")
            if mi is not None and mi.get_name().startswith("MI_Face_Skin") and "LOD0" in mi.get_name():
                return o, mi
    raise RuntimeError(f"no face mesh with an MI_Face_Skin ... LOD0 slot under {folder}")


def import_mask(png, dest):
    t = unreal.AssetImportTask()
    t.set_editor_property("filename", png)
    t.set_editor_property("destination_path", dest)
    t.set_editor_property("destination_name", "T_head_mask")
    t.set_editor_property("automated", True)
    t.set_editor_property("replace_existing", True)
    t.set_editor_property("save", False)
    AT.import_asset_tasks([t])
    tex = EAL.load_asset(f"{dest}/T_head_mask")
    if tex is None:
        raise RuntimeError(f"mask import failed: {png}")
    tex.set_editor_property("srgb", False)
    tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
    EAL.save_loaded_asset(tex)
    return tex


def copy_blueprint(src, dst, mat):
    if EAL.does_asset_exist(dst):
        EAL.delete_asset(dst)
    bp = EAL.duplicate_asset(src.split(".")[0], dst)
    if bp is None:
        raise RuntimeError(f"Blueprint copy failed: {src}")
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    k = 0
    for h in sds.k2_gather_subobject_data_for_blueprint(bp):
        o = unreal.SubobjectDataBlueprintFunctionLibrary.get_object(sds.k2_find_subobject_data_from_handle(h))
        if o is None or not isinstance(o, unreal.SkeletalMeshComponent) or not o.get_name().lower().startswith("face"):
            continue
        mesh = o.get_skinned_asset()
        ov = []
        for sm in mesh.get_editor_property("materials"):
            slot = str(sm.get_editor_property("material_slot_name"))
            ov.append(mat if slot.startswith("head_") else sm.get_editor_property("material_interface"))
            k += slot.startswith("head_")
        o.set_editor_property("override_materials", ov)
    if not k:
        raise RuntimeError(f"no face skin slot in {src}")
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    EAL.save_loaded_asset(bp)
    return bp.get_path_name().split(".")[0]


def run(job):
    dest = job["dest"].rstrip("/")
    mesh, src = face_skin(job["face_folder"])
    base = src
    while isinstance(base, unreal.MaterialInstance):
        base = base.get_editor_property("parent")
    if base is None:
        raise RuntimeError("the face skin instance has no base material")
    tex = import_mask(job["mask_png"], dest)
    mpath = f"{dest}/M_skin_masked"
    if EAL.does_asset_exist(mpath):
        EAL.delete_asset(mpath)
    mat = EAL.duplicate_asset(base.get_path_name().split(".")[0], mpath)
    if mat is None:
        raise RuntimeError(f"duplicate of {base.get_path_name()} failed")
    attrs = bool(mat.get_editor_property("use_material_attributes"))
    mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    mat.set_editor_property("opacity_mask_clip_value", 0.5)
    ex = MEL.create_material_expression(mat, unreal.MaterialExpressionTextureSampleParameter2D, -1400, 1600)
    ex.set_editor_property("parameter_name", "HideMask")
    ex.set_editor_property("texture", tex)
    ex.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_MASKS)
    how = "opacity mask input"
    if attrs:
        # the material's attributes come from one node: break them out and make them again, the opacity mask from the
        # mask texture, every other attribute passed through as it was
        src_node = MEL.get_material_property_input_node(mat, unreal.MaterialProperty.MP_MATERIAL_ATTRIBUTES)
        if src_node is None:
            raise RuntimeError("the skin material's attributes input is not connected")
        out_name = MEL.get_material_property_input_node_output_name(mat, unreal.MaterialProperty.MP_MATERIAL_ATTRIBUTES)
        br = MEL.create_material_expression(mat, unreal.MaterialExpressionBreakMaterialAttributes, -900, 1600)
        mk = MEL.create_material_expression(mat, unreal.MaterialExpressionMakeMaterialAttributes, -500, 1600)
        MEL.connect_material_expressions(src_node, str(out_name or ""), br, "")
        passed = []
        for a in PASS_ATTRS:
            if a == "OpacityMask":
                continue
            if MEL.connect_material_expressions(br, a, mk, a):
                passed.append(a)
        ok = MEL.connect_material_expressions(ex, "R", mk, "OpacityMask")
        ok = ok and MEL.connect_material_property(mk, "", unreal.MaterialProperty.MP_MATERIAL_ATTRIBUTES)
        how = f"attributes broken and made again ({len(passed)} passed through)"
    else:
        ok = MEL.connect_material_property(ex, "R", unreal.MaterialProperty.MP_OPACITY_MASK)
    MEL.recompile_material(mat)
    EAL.save_loaded_asset(mat)
    ipath = f"{dest}/MI_skin_masked"
    if EAL.does_asset_exist(ipath):
        EAL.delete_asset(ipath)
    mi = AT.create_asset("MI_skin_masked", dest, unreal.MaterialInstanceConstant,
                         unreal.MaterialInstanceConstantFactoryNew())
    MEL.set_material_instance_parent(mi, mat)
    n = {"scalar": 0, "vector": 0, "texture": 0, "switch": 0}
    for p in MEL.get_scalar_parameter_names(src):
        MEL.set_material_instance_scalar_parameter_value(mi, p, MEL.get_material_instance_scalar_parameter_value(src, p))
        n["scalar"] += 1
    for p in MEL.get_vector_parameter_names(src):
        MEL.set_material_instance_vector_parameter_value(mi, p, MEL.get_material_instance_vector_parameter_value(src, p))
        n["vector"] += 1
    for p in MEL.get_texture_parameter_names(src):
        if str(p) == "HideMask":
            continue
        v = MEL.get_material_instance_texture_parameter_value(src, p)
        if v is not None:
            MEL.set_material_instance_texture_parameter_value(mi, p, v)
            n["texture"] += 1
    for p in MEL.get_static_switch_parameter_names(src):
        try:
            MEL.set_material_instance_static_switch_parameter_value(
                mi, p, MEL.get_material_instance_static_switch_parameter_value(src, p))
            n["switch"] += 1
        except Exception as e:
            unreal.log_warning(f"static switch {p}: {e}")
    MEL.set_material_instance_texture_parameter_value(mi, "HideMask", tex)
    MEL.update_material_instance(mi)
    EAL.save_loaded_asset(mi)
    bp_copy = None
    if job.get("blueprint"):
        # the scene tool sets editor properties on the spawned Blueprint's components, which re-runs its construction
        # and replaces the face (and any material set on it): the character gets its own copy of its Blueprint whose
        # Face component template wears the masked skin in its skin slots
        bp_copy = copy_blueprint(job["blueprint"], f"{dest}/BP_{job.get('name', 'skin_masked')}", mi)
    out = {"ok": bool(ok), "how": how, "blueprint": bp_copy, "instance": mi.get_path_name().split(".")[0], "material": mpath, "source": src.get_path_name(),
           "base": base.get_path_name(), "face_mesh": mesh.get_path_name(), "mask": tex.get_path_name(), "copied": n}
    os.makedirs(os.path.dirname(job["result"]), exist_ok=True)
    json.dump(out, open(job["result"], "w"), indent=1)
    return out
