"""UE side: (re)build one set level from an expanded scene (layout.py) inside a running editor, then save it.

The level /Game/Sets/<id>/L_<id> is created once; a rebuild loads it, deletes every actor it holds and spawns the
scene again, so the map is always exactly the spec. Structure = the engine's basic cube/cylinder/sphere scaled,
with the set's material instances (set_materials.py). Library assets by their static-mesh path. Lights, sky, fog
and a post-process volume with manual exposure from the scene's "env".
"""

import importlib.util
import os
import time

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
EAL = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
ML = unreal.MathLibrary
MESHES = {"cube": "/Engine/BasicShapes/Cube", "cylinder": "/Engine/BasicShapes/Cylinder",
          "sphere": "/Engine/BasicShapes/Sphere"}


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def V(v):
    return unreal.Vector(float(v[0]), float(v[1]), float(v[2]))


def open_clean_level(level):
    if EAL.does_asset_exist(level):
        w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        if w is None or not w.get_path_name().startswith(level):
            les.load_level(level)
        actors = act.get_all_level_actors()
        if actors:
            act.destroy_actors(actors)
    else:
        folder = level.rsplit("/", 1)[0]
        if not EAL.does_directory_exist(folder):
            EAL.make_directory(folder)
        if not les.new_level(level):
            raise RuntimeError(f"new_level failed for {level}")


def spawn_mesh(mesh, loc, rot, scale, mat, folder):
    a = act.spawn_actor_from_class(unreal.StaticMeshActor, loc, rot)
    c = a.static_mesh_component
    c.set_static_mesh(mesh)
    if mat is not None:
        c.set_material(0, mat)
    a.set_actor_scale3d(scale)
    a.set_folder_path(folder)
    return a


def setp(obj, names, val):
    """Set the first property name this engine version knows (names differ between versions)."""
    for n in names:
        try:
            obj.set_editor_property(n, val)
            return n
        except Exception:
            continue
    unreal.log_warning(f"SETS: none of {names} on {obj.get_class().get_name()}")
    return None


def light_common(comp, kelvin, shadows=True):
    comp.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    if kelvin:
        comp.set_editor_property("use_temperature", True)
        comp.set_editor_property("temperature", float(kelvin))
    comp.set_editor_property("cast_shadows", bool(shadows))


def spawn_light(l):
    """One point / spot / rect light of an expanded scene (layout.light_items) -> its actor."""
    typ = l["type"]
    loc = V(l["c"])
    rot = ML.make_rot_from_x(V(l["dir"])) if "dir" in l else unreal.Rotator(0, 0, 0)
    if typ == "rect":
        a = act.spawn_actor_from_class(unreal.RectLight, loc, rot)
        c = a.get_component_by_class(unreal.RectLightComponent)
        c.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
        c.set_editor_property("source_width", float(l["w_cm"]))
        c.set_editor_property("source_height", float(l["h_cm"]))
        c.set_editor_property("barn_door_angle", float(l.get("barn_door_deg", 88)))
    elif typ == "spot":
        a = act.spawn_actor_from_class(unreal.SpotLight, loc, rot)
        c = a.get_component_by_class(unreal.SpotLightComponent)
        c.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
        c.set_editor_property("outer_cone_angle", float(l.get("cone_deg", 40)))
    else:
        a = act.spawn_actor_from_class(unreal.PointLight, loc, rot)
        c = a.get_component_by_class(unreal.PointLightComponent)
        c.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
        c.set_editor_property("source_radius", float(l.get("source_radius_m", 0.05)) * 100)
    light_common(c, l.get("kelvin", 5000), l.get("shadows", True))
    c.set_editor_property("intensity", float(l["lumens"]))
    c.set_editor_property("attenuation_radius", float(l.get("radius_m", 30)) * 100)
    if "color" in l:
        c.set_editor_property("use_temperature", False)
        rgb = [int(l["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
        # keyword arguments: unreal.Color's positional order is B, G, R, A (blue lights came out red)
        c.set_editor_property("light_color", unreal.Color(r=rgb[0], g=rgb[1], b=rgb[2], a=255))
    a.set_folder_path(l["f"])
    return a


def build(set_id, level, scene):
    t = {}
    t0 = time.time()
    mats = _mod("set_materials")
    ms = mats.masters()
    mi = {name: ms[m["master"]] for name, m in scene["materials"].items()}
    cpd = {name: m["cpd"] for name, m in scene["materials"].items()}
    t["materials"] = round(time.time() - t0, 2)
    t1 = time.time()
    open_clean_level(level)
    t["open_level"] = round(time.time() - t1, 2)
    t1 = time.time()
    meshes = {k: EAL.load_asset(v) for k, v in MESHES.items()}
    for p in scene["prims"]:
        if "x" in p:
            if p["m"] == "cylinder":
                rot = ML.make_rot_from_zx(V(p["z"]), V(p["x"]))
            else:
                rot = ML.make_rot_from_xz(V(p["x"]), V(p["z"]))
        else:
            rot = unreal.Rotator(0, 0, p.get("yaw", 0.0))
        a = spawn_mesh(meshes[p["m"]], V(p["c"]), rot, V(p["s"]), mi[p["mat"]], p["f"])
        mats.apply_cpd(a.static_mesh_component, cpd[p["mat"]])
    t["prims"] = round(time.time() - t1, 2)
    t1 = time.time()
    cache = {}
    for a in scene["assets"]:
        sm = cache.get(a["path"]) or EAL.load_asset(a["path"])
        cache[a["path"]] = sm
        x = act.spawn_actor_from_class(unreal.StaticMeshActor, V(a["c"]), unreal.Rotator(0, 0, a["yaw"]))
        x.static_mesh_component.set_static_mesh(sm)
        x.set_actor_label(a["name"])
        x.set_folder_path(a["f"])
    t["assets"] = round(time.time() - t1, 2)
    t1 = time.time()
    for l in scene["lights"]:
        spawn_light(l)
    t["lights"] = round(time.time() - t1, 2)
    env = scene["env"]
    sun = env.get("sun")
    if sun:
        a = act.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 3000),
                                       ML.make_rot_from_x(V([-x for x in sun["toward_sun_ue"]])))
        a.set_actor_label("Sun")
        c = a.get_component_by_class(unreal.DirectionalLightComponent)
        light_common(c, sun.get("kelvin", 5600))
        c.set_editor_property("intensity", float(sun["lux"]))
        c.set_editor_property("atmosphere_sun_light", True)
        c.set_editor_property("light_source_angle", float(sun.get("angle_deg", 0.53)))
        c.set_editor_property("volumetric_scattering_intensity", float(sun.get("volumetric", 1.0)))
        a.set_folder_path("Environment")
    sky = env.get("sky")
    if sky:
        if sky.get("atmosphere", True):
            act.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0)).set_folder_path("Environment")
        a = act.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 500))
        c = a.get_component_by_class(unreal.SkyLightComponent)
        c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        c.set_editor_property("real_time_capture", True)
        c.set_editor_property("intensity", float(sky.get("intensity", 1.0)))
        if "lower_hemisphere" in sky:
            c.set_editor_property("lower_hemisphere_is_black", True)
        a.set_folder_path("Environment")
    fog = env.get("fog")
    if fog:
        a = act.spawn_actor_from_class(unreal.ExponentialHeightFog, unreal.Vector(0, 0, 0))
        c = a.get_component_by_class(unreal.ExponentialHeightFogComponent)
        c.set_editor_property("fog_density", float(fog.get("density", 0.02)))
        c.set_editor_property("fog_height_falloff", float(fog.get("falloff", 0.05)))
        if fog.get("volumetric"):
            setp(c, ["enable_volumetric_fog", "volumetric_fog", "b_enable_volumetric_fog"], True)
            setp(c, ["volumetric_fog_extinction_scale"], float(fog.get("extinction", 1.0)))
            setp(c, ["volumetric_fog_distance"], float(fog.get("distance_m", 80)) * 100)
        a.set_folder_path("Environment")
    ppv = act.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
    ppv.set_actor_label("SetPost")
    ppv.set_folder_path("Environment")
    ppv.set_editor_property("unbound", True)
    pp = ppv.get_editor_property("settings")
    post = env.get("post", {})

    def ov(name, val):
        try:
            pp.set_editor_property(f"override_{name}", True)
            pp.set_editor_property(name, val)
        except Exception as e:  # property names differ between engine versions: record, do not fail the build
            unreal.log_warning(f"SETS post {name}: {e}")

    ov("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    ov("auto_exposure_apply_physical_camera_exposure", False)
    ov("auto_exposure_bias", -float(env["exposure_ev100"]))
    ov("bloom_intensity", float(post.get("bloom", 0.3)))
    ov("vignette_intensity", float(post.get("vignette", 0.25)))
    ov("lumen_final_gather_quality", float(post.get("lumen_gather_quality", 2.0)))
    ov("lumen_scene_lighting_quality", float(post.get("lumen_scene_quality", 2.0)))
    ov("lumen_reflection_quality", float(post.get("lumen_reflection_quality", 2.0)))
    ov("lumen_max_trace_distance", float(post.get("lumen_trace_m", 200)) * 100)
    ov("lumen_scene_view_distance", float(post.get("lumen_view_m", 200)) * 100)
    ov("lumen_ray_lighting_mode", unreal.LumenRayLightingModeOverride.HIT_LIGHTING_FOR_REFLECTIONS)
    if "saturation" in post:
        s = float(post["saturation"])
        ov("color_saturation", unreal.Vector4(s, s, s, 1.0))
    if "contrast" in post:
        s = float(post["contrast"])
        ov("color_contrast", unreal.Vector4(s, s, s, 1.0))
    ppv.set_editor_property("settings", pp)
    t1 = time.time()
    les.save_current_level()
    t["save"] = round(time.time() - t1, 2)
    t["total"] = round(time.time() - t0, 2)
    return {"times_s": t, "prims": len(scene["prims"]), "assets": len(scene["assets"]), "lights": len(scene["lights"])}
