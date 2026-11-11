"""UE side: the set builder's master materials and per-set material instances.

Masters (made once per project under /Game/Sets/_Shared/, rebuilt when MASTER_VERSION changes):
  M_SetSurface  opaque, world-aligned procedural: two-colour large-scale variation, grime (optionally stretched
                vertically into streaks), roughness variation, panel/slab joints on a world grid (darker, rougher
                lines on every face, whatever its orientation). No textures, no UVs: a scaled cube and a 60 m wall
                read the same, and nothing stretches.
  M_SetGlass    translucent, surface-lit, see-through glass (tint, opacity, roughness).
  M_SetEmissive unlit light source surface (colour x intensity in nits).
Instances: /Game/Sets/<set_id>/Materials/MI_<name>, parameters from the resolved spec (build_set.py).
"""

import unreal

EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary
AT = unreal.AssetToolsHelpers.get_asset_tools()
SHARED = "/Game/Sets/_Shared"
MASTER_VERSION = 6

# Every parameter reads Custom Primitive Data (per mesh component, saved in the level), so one master serves all
# sets and a new look never compiles a shader. Indices (floats): vectors take 4 slots (RGBA).
CPD = {
    "surface": {"ColorA": 0, "ColorB": 4, "GrimeColor": 8, "GrimeStretch": 12, "PanelSizeM": 16,
                "ColorScaleM": 20, "MottleScaleM": 21, "MottleAmount": 22, "GrimeAmount": 23, "GrimeScaleM": 24,
                "RoughA": 25, "RoughB": 26, "RoughScaleM": 27, "GrimeRough": 28, "JointWidthM": 29,
                "JointDarken": 30, "JointRough": 31, "Metallic": 32, "Specular": 33, "GrimeContrast": 34},
    "glass": {"Tint": 0, "Opacity": 4, "Roughness": 5, "Specular": 6},
    "emissive": {"Color": 0, "Intensity": 4},
}
_CUR = {"master": None}


def _expr(m, cls, x, y, **props):
    e = MEL.create_material_expression(m, cls, x, y)
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def _link(a, b, inp, out=""):
    if not MEL.connect_material_expressions(a, out, b, inp):     # returns False on a wrong pin name: fail loudly
        raise RuntimeError(f"material link failed: {a.get_class().get_name()} -> {b.get_class().get_name()}.{inp!r}")


def _bin(m, cls, a, b, x, y, ia="A", ib="B"):
    e = _expr(m, cls, x, y)
    _link(a, e, ia)
    _link(b, e, ib)
    return e


def _scalar(m, name, v, x, y):
    return _expr(m, unreal.MaterialExpressionScalarParameter, x, y, parameter_name=name, default_value=v,
                 use_custom_primitive_data=True, primitive_data_index=CPD[_CUR["master"]][name])


def _vector(m, name, rgb, x, y):
    e = _expr(m, unreal.MaterialExpressionVectorParameter, x, y, parameter_name=name,
              default_value=unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0),
              use_custom_primitive_data=True, primitive_data_index=CPD[_CUR["master"]][name])
    if name in ("Tint", "ColorA", "ColorB", "GrimeColor", "Color", "GrimeStretch", "PanelSizeM"):
        rgb_ = _expr(m, unreal.MaterialExpressionComponentMask, x + 150, y, r=True, g=True, b=True, a=False)
        _link(e, rgb_, "")
        return rgb_
    return e


def _mask(m, src, ch, x, y):
    e = _expr(m, unreal.MaterialExpressionComponentMask, x, y, r=ch == "r", g=ch == "g", b=ch == "b", a=False)
    _link(src, e, "")
    return e


def _noise(m, pos, levels, x, y):
    e = _expr(m, unreal.MaterialExpressionNoise, x, y, scale=1.0, levels=levels, output_min=0.0, output_max=1.0,
              noise_function=unreal.NoiseFunction.NOISEFUNCTION_GRADIENT_ALU, quality=1)
    _link(pos, e, "")          # the first pin ("World Position" in 5.8; the name "Position" does not connect)
    return e


def _new_material(name):
    path = f"{SHARED}/{name}"
    if EAL.does_asset_exist(path):
        m = EAL.load_asset(path)
        if EAL.get_metadata_tag(m, "set_master_version") == str(MASTER_VERSION):
            return m, False
        MEL.delete_all_material_expressions(m)
        return m, True
    if not EAL.does_directory_exist(SHARED):
        EAL.make_directory(SHARED)
    return AT.create_asset(name, SHARED, unreal.Material, unreal.MaterialFactoryNew()), True


def _finish(m):
    MEL.layout_material_expressions(m)
    MEL.recompile_material(m)
    EAL.set_metadata_tag(m, "set_master_version", str(MASTER_VERSION))
    EAL.save_loaded_asset(m)


def master_surface():
    _CUR["master"] = "surface"
    m, fresh = _new_material("M_SetSurface")
    if not fresh:
        return m
    wp = _expr(m, unreal.MaterialExpressionWorldPosition, -2400, 0)
    to_m = _expr(m, unreal.MaterialExpressionConstant, -2400, 120, r=0.01)
    pm = _bin(m, unreal.MaterialExpressionMultiply, wp, to_m, -2200, 0)              # metres
    # large colour variation
    cscale = _scalar(m, "ColorScaleM", 4.0, -2200, 200)
    p1 = _bin(m, unreal.MaterialExpressionDivide, pm, cscale, -2000, 100)
    n1 = _noise(m, p1, 5, -1800, 100)
    ca = _vector(m, "ColorA", (0.2, 0.2, 0.2), -1800, -200)
    cb = _vector(m, "ColorB", (0.3, 0.3, 0.3), -1800, -100)
    base = _expr(m, unreal.MaterialExpressionLinearInterpolate, -1500, -100)
    _link(ca, base, "A"); _link(cb, base, "B"); _link(n1, base, "Alpha")
    # fine mottle: multiplies the colour by 1 +- MottleAmount
    mscale = _scalar(m, "MottleScaleM", 0.25, -2200, 300)
    p3 = _bin(m, unreal.MaterialExpressionDivide, pm, mscale, -2000, 300)
    n3 = _noise(m, p3, 3, -1800, 300)
    mamt = _scalar(m, "MottleAmount", 0.15, -1800, 450)
    half = _expr(m, unreal.MaterialExpressionConstant, -1800, 520, r=0.5)
    n3c = _bin(m, unreal.MaterialExpressionSubtract, n3, half, -1600, 300)
    n3s = _bin(m, unreal.MaterialExpressionMultiply, n3c, mamt, -1450, 300)
    one = _expr(m, unreal.MaterialExpressionConstant, -1450, 450, r=1.0)
    mott = _bin(m, unreal.MaterialExpressionAdd, n3s, one, -1300, 300)
    base2 = _bin(m, unreal.MaterialExpressionMultiply, base, mott, -1200, -100)
    # grime: noise on position stretched in Z (streaks), thresholded by GrimeAmount
    gstretch = _vector(m, "GrimeStretch", (1.0, 1.0, 1.0), -2200, 600)
    gscale = _scalar(m, "GrimeScaleM", 1.5, -2200, 700)
    pg = _bin(m, unreal.MaterialExpressionMultiply, pm, gstretch, -2000, 600)
    pg2 = _bin(m, unreal.MaterialExpressionDivide, pg, gscale, -1850, 600)
    ng = _noise(m, pg2, 4, -1700, 600)
    gamt = _scalar(m, "GrimeAmount", 0.3, -1700, 760)
    gthr = _bin(m, unreal.MaterialExpressionAdd, ng, gamt, -1500, 600)              # ng + amt - 1, x3, saturate
    gthr2 = _bin(m, unreal.MaterialExpressionSubtract, gthr, one, -1350, 600)
    three = _scalar(m, "GrimeContrast", 2.0, -1350, 720)          # edge hardness of stains
    gmul = _bin(m, unreal.MaterialExpressionMultiply, gthr2, three, -1200, 600)
    gmask = _expr(m, unreal.MaterialExpressionSaturate, -1050, 600)
    _link(gmul, gmask, "")
    gcol = _vector(m, "GrimeColor", (0.05, 0.045, 0.04), -1050, 450)
    col = _expr(m, unreal.MaterialExpressionLinearInterpolate, -900, -100)
    _link(base2, col, "A"); _link(gcol, col, "B"); _link(gmask, col, "Alpha")
    # roughness variation
    rscale = _scalar(m, "RoughScaleM", 3.0, -2200, 900)
    p2 = _bin(m, unreal.MaterialExpressionDivide, pm, rscale, -2000, 900)
    n2 = _noise(m, p2, 4, -1800, 900)
    ra = _scalar(m, "RoughA", 0.4, -1800, 1050)
    rb = _scalar(m, "RoughB", 0.8, -1800, 1120)
    rough = _expr(m, unreal.MaterialExpressionLinearInterpolate, -1500, 900)
    _link(ra, rough, "A"); _link(rb, rough, "B"); _link(n2, rough, "Alpha")
    grough = _scalar(m, "GrimeRough", 0.85, -1500, 1050)
    rough2 = _expr(m, unreal.MaterialExpressionLinearInterpolate, -1300, 900)
    _link(rough, rough2, "A"); _link(grough, rough2, "B"); _link(gmask, rough2, "Alpha")
    # joints: per axis, distance to the nearest grid line, weighted by how much the face runs along that axis
    psize = _vector(m, "PanelSizeM", (1000.0, 1000.0, 1000.0), -2200, 1300)
    pd = _bin(m, unreal.MaterialExpressionDivide, pm, psize, -1850, 1350)
    fr = _expr(m, unreal.MaterialExpressionFrac, -1700, 1350)
    _link(pd, fr, "")
    fr1 = _bin(m, unreal.MaterialExpressionSubtract, one, fr, -1550, 1450)
    fmin = _bin(m, unreal.MaterialExpressionMin, fr, fr1, -1400, 1350)
    dist = _bin(m, unreal.MaterialExpressionMultiply, fmin, psize, -1250, 1350)      # metres to the joint
    jw = _scalar(m, "JointWidthM", 0.01, -1250, 1500)
    dn = _bin(m, unreal.MaterialExpressionDivide, dist, jw, -1100, 1350)
    dsat = _expr(m, unreal.MaterialExpressionSaturate, -950, 1350)
    _link(dn, dsat, "")
    jv = _bin(m, unreal.MaterialExpressionSubtract, one, dsat, -800, 1350)
    nrm = _expr(m, unreal.MaterialExpressionVertexNormalWS, -1100, 1600)
    nab = _expr(m, unreal.MaterialExpressionAbs, -950, 1600)
    _link(nrm, nab, "")
    nw = _bin(m, unreal.MaterialExpressionSubtract, one, nab, -800, 1600)
    jvw = _bin(m, unreal.MaterialExpressionMultiply, jv, nw, -650, 1450)
    jx, jy, jz = (_mask(m, jvw, c, -500, 1400 + 80 * i) for i, c in enumerate("rgb"))
    jxy = _bin(m, unreal.MaterialExpressionMax, jx, jy, -350, 1420)
    joint = _bin(m, unreal.MaterialExpressionMax, jxy, jz, -200, 1450)
    jdark = _scalar(m, "JointDarken", 0.5, -350, 1600)
    jcol = _bin(m, unreal.MaterialExpressionMultiply, col, jdark, -350, -50)
    col2 = _expr(m, unreal.MaterialExpressionLinearInterpolate, -150, -100)
    _link(col, col2, "A"); _link(jcol, col2, "B"); _link(joint, col2, "Alpha")
    jrough = _scalar(m, "JointRough", 0.9, -200, 1700)
    rough3 = _expr(m, unreal.MaterialExpressionLinearInterpolate, -150, 900)
    _link(rough2, rough3, "A"); _link(jrough, rough3, "B"); _link(joint, rough3, "Alpha")
    metal = _scalar(m, "Metallic", 0.0, -150, 700)
    spec = _scalar(m, "Specular", 0.5, -150, 780)
    MEL.connect_material_property(col2, "", unreal.MaterialProperty.MP_BASE_COLOR)
    MEL.connect_material_property(rough3, "", unreal.MaterialProperty.MP_ROUGHNESS)
    MEL.connect_material_property(metal, "", unreal.MaterialProperty.MP_METALLIC)
    MEL.connect_material_property(spec, "", unreal.MaterialProperty.MP_SPECULAR)
    _finish(m)
    return m


def master_glass():
    _CUR["master"] = "glass"
    m, fresh = _new_material("M_SetGlass")
    if not fresh:
        return m
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    m.set_editor_property("translucency_lighting_mode", unreal.TranslucencyLightingMode.TLM_SURFACE)
    m.set_editor_property("two_sided", True)
    tint = _vector(m, "Tint", (0.8, 0.85, 0.85), -600, 0)
    op = _scalar(m, "Opacity", 0.15, -600, 150)
    ro = _scalar(m, "Roughness", 0.05, -600, 250)
    sp = _scalar(m, "Specular", 0.5, -600, 350)
    MEL.connect_material_property(tint, "", unreal.MaterialProperty.MP_BASE_COLOR)
    MEL.connect_material_property(op, "", unreal.MaterialProperty.MP_OPACITY)
    MEL.connect_material_property(ro, "", unreal.MaterialProperty.MP_ROUGHNESS)
    MEL.connect_material_property(sp, "", unreal.MaterialProperty.MP_SPECULAR)
    _finish(m)
    return m


def master_emissive():
    _CUR["master"] = "emissive"
    m, fresh = _new_material("M_SetEmissive")
    if not fresh:
        return m
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    c = _vector(m, "Color", (1, 1, 1), -600, 0)
    i = _scalar(m, "Intensity", 1000.0, -600, 150)
    e = _bin(m, unreal.MaterialExpressionMultiply, c, i, -300, 0)
    MEL.connect_material_property(e, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    _finish(m)
    return m


MASTERS = {"surface": master_surface, "glass": master_glass, "emissive": master_emissive}


def masters():
    """The three masters (built once per project; MASTER_VERSION bumps rebuild them)."""
    return {k: f() for k, f in MASTERS.items()}


def apply_cpd(comp, floats):
    """Write a material's parameter floats (presets.resolve -> "cpd") into a component's saved custom primitive
    data, four at a time."""
    for i in range(0, len(floats), 4):
        q = list(floats[i:i + 4]) + [0.0] * (4 - len(floats[i:i + 4]))
        comp.set_default_custom_primitive_data_vector4(i, unreal.Vector4(q[0], q[1], q[2], q[3]))
