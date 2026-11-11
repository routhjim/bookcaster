"""Material presets for set specs (stdlib). A spec material is {"preset": name, ...overrides}; resolve() turns it
into the UE master + parameters (set_materials.py). Colours are sRGB hex in the spec, linear in UE.

Overrides any preset accepts (all optional):
  color        "#rrggbb"   main colour (ColorA); ColorB is derived from it by `variation`
  color2       "#rrggbb"   second colour for the large-scale variation (instead of the derived one)
  variation    0..1        how different the derived ColorB is (lighter) from color
  roughness    [a, b]      roughness range of the large-scale variation
  metallic     0..1
  grime        0..1        how much of the surface carries dirt (0 clean .. 1 filthy)
  grime_color  "#rrggbb"
  streaks      true/false  stretch the grime vertically (rain/drip streaks on walls)
  panel_m      [x, y, z]   joint grid in metres along set X, Y, Z (0 or null = no joints on that axis)
  joint_m      joint line width in metres
  feature_m    size in metres of the large colour patches
Glass: tint "#rrggbb", opacity 0..1, roughness. Emissive: color "#rrggbb", nits (luminance of the surface).
"""

PRESETS = {
    "worn_concrete": {"master": "surface", "color": "#77746e", "color2": "#97938a", "feature_m": 5.0,
                      "mottle": [0.9, 0.18], "grime": 0.35, "grime_color": "#3b3833", "grime_m": 2.5,
                      "roughness": [0.25, 0.85], "rough_m": 3.5, "grime_rough": 0.9, "panel_m": [6, 6, 0],
                      "joint_m": 0.012, "joint_darken": 0.35},
    "painted_steel": {"master": "surface", "color": "#5d5f5c", "variation": 0.12, "feature_m": 2.0,
                      "mottle": [0.9, 0.08], "grime": 0.3, "grime_color": "#2c2a26", "grime_m": 1.2, "streaks": True,
                      "roughness": [0.35, 0.6], "rough_m": 1.5, "grime_rough": 0.75, "panel_m": [0, 0, 0],
                      "joint_m": 0.008, "joint_darken": 0.5},
    "bare_steel": {"master": "surface", "color": "#7d8083", "variation": 0.15, "feature_m": 1.5, "metallic": 1.0,
                   "mottle": [0.8, 0.12], "grime": 0.25, "grime_color": "#3a3836", "grime_m": 0.8,
                   "roughness": [0.3, 0.55], "rough_m": 0.8, "grime_rough": 0.7, "panel_m": [0, 0, 0]},
    "galvanized_steel": {"master": "surface", "color": "#9a9d9f", "variation": 0.12, "feature_m": 0.6, "metallic": 1.0,
                         "mottle": [0.8, 0.12], "grime": 0.1, "grime_color": "#6d6e6c", "grime_m": 1.0,
                         "roughness": [0.3, 0.5], "rough_m": 0.5, "grime_rough": 0.6, "panel_m": [0, 0, 0]},
    "stainless_steel": {"master": "surface", "color": "#b9bcbe", "variation": 0.05, "feature_m": 1.0, "metallic": 1.0,
                        "mottle": [0.8, 0.04], "grime": 0.0, "roughness": [0.18, 0.28], "rough_m": 0.4,
                        "panel_m": [0, 0, 0]},
    "epoxy_floor": {"master": "surface", "color": "#e3e5e4", "color2": "#eceeed", "feature_m": 3.0,
                    "mottle": [0.8, 0.04], "grime": 0.08, "grime_color": "#b9bbba", "grime_m": 1.5,
                    "roughness": [0.06, 0.16], "rough_m": 2.0, "grime_rough": 0.3, "panel_m": [0, 0, 0]},
    "white_panel": {"master": "surface", "color": "#e8eaea", "variation": 0.03, "feature_m": 3.0,
                    "mottle": [0.8, 0.03], "grime": 0.04, "grime_color": "#c3c6c6", "grime_m": 1.5,
                    "roughness": [0.3, 0.4], "rough_m": 2.0, "grime_rough": 0.45, "panel_m": [1.2, 1.2, 0],
                    "joint_m": 0.006, "joint_darken": 0.6},
    "paint_line": {"master": "surface", "color": "#c79a1c", "variation": 0.1, "feature_m": 1.0,
                   "mottle": [0.6, 0.15], "grime": 0.45, "grime_color": "#4a443a", "grime_m": 0.6,
                   "roughness": [0.45, 0.75], "rough_m": 1.0, "grime_rough": 0.85, "panel_m": [0, 0, 0]},
    "rubber": {"master": "surface", "color": "#1e1f1f", "variation": 0.05, "feature_m": 1.0, "mottle": [0.6, 0.1],
               "grime": 0.1, "grime_color": "#3a3a38", "roughness": [0.75, 0.9], "panel_m": [0, 0, 0]},
    "plastic": {"master": "surface", "color": "#d8dada", "variation": 0.04, "feature_m": 1.0, "mottle": [0.6, 0.05],
                "grime": 0.05, "grime_color": "#a0a2a2", "roughness": [0.35, 0.45], "panel_m": [0, 0, 0]},
    "glass": {"master": "glass", "tint": "#e6eceb", "opacity": 0.12, "roughness": 0.03},
    "frosted_glass": {"master": "glass", "tint": "#eef1f0", "opacity": 0.45, "roughness": 0.25},
    "emissive": {"master": "emissive", "color": "#ffffff", "nits": 8000.0},
}


def srgb_to_linear(h):
    h = h.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return [round(x, 5) for x in out]


def resolve(m):
    if m.get("preset") not in PRESETS:
        raise ValueError(f"unknown material preset {m.get('preset')!r}; known: {sorted(PRESETS)}")
    p = dict(PRESETS[m["preset"]])
    p.update({k: v for k, v in m.items() if k != "preset"})
    if p["master"] == "glass":
        return {"master": "glass", "vectors": {"Tint": srgb_to_linear(p["tint"])},
                "scalars": {"Opacity": p["opacity"], "Roughness": p["roughness"], "Specular": p.get("specular", 0.5)}}
    if p["master"] == "emissive":
        return {"master": "emissive", "vectors": {"Color": srgb_to_linear(p["color"])},
                "scalars": {"Intensity": p["nits"]}}
    a = srgb_to_linear(p["color"])
    if "color2" in p and "variation" not in m:
        b = srgb_to_linear(p["color2"])
    else:
        v = p.get("variation", 0.1)
        b = [min(1.0, c * (1 + v * 2.5)) for c in a]
    panel = [x if x else 1000.0 for x in p.get("panel_m", [0, 0, 0])]
    stretch = [1.0, 1.0, 0.12] if p.get("streaks") else [1.0, 1.0, 1.0]
    mot = p.get("mottle", [0.8, 0.08])
    return {"master": "surface",
            "vectors": {"ColorA": a, "ColorB": b, "GrimeColor": srgb_to_linear(p.get("grime_color", "#333333")),
                        "GrimeStretch": stretch, "PanelSizeM": panel,
                        "PanelOffsetM": p.get("panel_offset_m", [0, 0, 0])},
            "scalars": {"ColorScaleM": p.get("feature_m", 3.0), "MottleScaleM": mot[0], "MottleAmount": mot[1],
                        "GrimeAmount": p.get("grime", 0.2), "GrimeScaleM": p.get("grime_m", 1.5),
                        "RoughA": p["roughness"][0], "RoughB": p["roughness"][1], "RoughScaleM": p.get("rough_m", 2.0),
                        "GrimeRough": p.get("grime_rough", 0.8), "JointWidthM": p.get("joint_m", 0.01),
                        "JointDarken": p.get("joint_darken", 0.5), "JointRough": p.get("joint_rough", 0.9),
                        "Metallic": p.get("metallic", 0.0), "Specular": p.get("specular", 0.5)}}
