#!/usr/bin/env python3
"""Bring a HY-World set (tools/world.py output) into Blender, with the exact camera of a gsplat render.

STABLE API (inside Blender; other tools import this, keep the signatures):

    import sys, os; sys.path.insert(0, os.path.expanduser("~/tts_reader/tools")); import world_blender as wb
    s = wb.import_world(world_dir)            # -> dict, see below
    s["objects"]["set"]                       # the set mesh (cut + background layer behind occluders), unlit
    s["objects"].get("backdrop")              # exterior panoramas: far sphere behind the sky (or None)
    s["collection"]                           # bpy collection "set:<slug>" holding them
    s["ground"]   # {"plane_z": 0, "normal": [0,0,1], "eye_height_m", "method"}: the frame's ground plane
    s["capture_camera"], s["bounds_min"], s["bounds_max"], s["world"] (= world.json)
    z = wb.ground_z(s, x, y)                  # terrain height under (x, y) by ray cast (None if off the set)
    cam = wb.camera_from_json(path_or_dict)   # world.py <out>.png.camera.json, or a world.py camera dict
                                              # {"pos","look_at","fov"|"lens","roll"}; becomes scene camera

    import_world(world_dir, mesh="full"|"cut", backdrop=True, material="emission"|"principled",
                 name=None, hide_render=False, holdout=False)
      mesh="full" (default; "stretch" is an old alias): the cut mesh + a background layer that continues the
      far surface behind every occluder edge (tools/world_ldi.py), so a moved camera sees background, not holes.
      mesh="cut": depth jumps left open (clean silhouettes; holes show the backdrop).
      holdout=True: the set renders as a transparent holdout (still occludes, still in depth/Z passes),
      for compositing characters over a painted plate.

Command line (outside Blender; launches Blender -t 16 --factory-startup):
    python3 tools/world_blender.py render <world dir> --jobs jobs.json [--engine workbench|cycles] [--blend f]
        (the same jobs list world.py render takes; each <out> -> <out stem>_bl.png from <out>.camera.json)
    python3 tools/world_blender.py render <world dir> --camera-json <x>.png.camera.json --out bl.png

Frame: world.py's Blender frame (metres, Z up, ground plane z = 0, capture camera at (0, 0, eye height)
looking +Y), so no transform is applied on import. Material "emission" shows the painted colours unlit,
which is what the splat render shows; "principled" lets scene lights shade the set (the colours already
contain the painting's lighting). Large sets: a 1920x960 pano set is ~4.4 M faces (+0.9 M backdrop),
imports in ~2-3 s.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

BLENDER = os.environ.get("BLOCKOUT_BLENDER", os.path.expanduser("~/opt/blender/current/blender"))
THREADS = "16"

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


# ------------------------------------------------------------------------------------------- in Blender

def _color_attr(me):
    names = [a.name for a in me.color_attributes]
    return names[0] if names else None


def _set_material(ob, name: str, material: str):
    import bpy
    me = ob.data
    attr = _color_attr(me)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    if material == "holdout":
        h = nt.nodes.new("ShaderNodeHoldout")
        nt.links.new(h.outputs[0], outn.inputs["Surface"])
    else:
        vc = nt.nodes.new("ShaderNodeVertexColor")
        if attr:
            vc.layer_name = attr
        if material == "emission":
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            nt.links.new(vc.outputs["Color"], em.inputs["Color"])
            nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
        else:
            bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
            bs.inputs["Roughness"].default_value = 0.8
            nt.links.new(vc.outputs["Color"], bs.inputs["Base Color"])
            nt.links.new(bs.outputs["BSDF"], outn.inputs["Surface"])
    me.materials.clear()
    me.materials.append(mat)


def _import_ply(path: Path, name: str, coll):
    import bpy
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=str(path))
    ob = [o for o in bpy.data.objects if o not in before][0]
    ob.name = name
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    coll.objects.link(ob)
    return ob


def import_world(world_dir: str, mesh: str = "full", backdrop: bool = True, material: str = "emission",
                 name: str | None = None, hide_render: bool = False, holdout: bool = False) -> dict:
    """Import a world.py set into the current scene. See the module docstring for the returned dict."""
    import bpy
    wd = Path(os.path.expanduser(str(world_dir)))
    w = json.loads((wd / "world.json").read_text())
    files = w.get("files", {"mesh": "set_mesh.ply"})
    slug = name or w.get("slug") or wd.name
    coll = bpy.data.collections.new(f"set:{slug}")
    bpy.context.scene.collection.children.link(coll)
    key = "mesh_cut" if mesh == "cut" and "mesh_cut" in files else "mesh"
    objs = {"set": _import_ply(wd / files[key], f"{slug}_set", coll), "backdrop": None}
    if backdrop and files.get("backdrop") and (wd / files["backdrop"]).exists():
        objs["backdrop"] = _import_ply(wd / files["backdrop"], f"{slug}_backdrop", coll)
    for k, ob in objs.items():
        if ob is None:
            continue
        _set_material(ob, f"{slug}_{k}_mat", "holdout" if holdout else material)
        ob.hide_render = hide_render
        ob["world_dir"] = str(wd)
    ground = w.get("ground") or {"plane_z": 0.0, "normal": [0, 0, 1],
                                 "eye_height_m": w["capture_camera"]["pos"][2], "method": "legacy"}
    return {"objects": objs, "collection": coll, "ground": ground, "capture_camera": w["capture_camera"],
            "bounds_min": w.get("bounds_min"), "bounds_max": w.get("bounds_max"), "world": w,
            "world_dir": str(wd)}


def ground_z(s: dict, x: float, y: float, top: float | None = None):
    """Height of the set surface under (x, y): the first hit of a ray cast straight down onto the set mesh
    that is not sky (sky sits on a far dome and is skipped, the backdrop is never hit). A miss (e.g. the
    tiny untriangulated pole straight below a panorama's capture point) retries on a 5 cm ring and
    averages. None if nothing is below (off the set)."""
    import bpy
    from mathutils import Vector
    ob = s["objects"]["set"]
    w = s["world"]
    cam = Vector(w["capture_camera"]["pos"])
    sky_r = (w.get("level") or {}).get("sky_radius_m")
    if top is None:
        top = (s.get("bounds_max") or [0, 0, 50])[2] + 5.0
    dg = bpy.context.evaluated_depsgraph_get()
    obe = ob.evaluated_get(dg)
    mw, mw_inv = ob.matrix_world, ob.matrix_world.inverted()
    down = (mw_inv.to_3x3() @ Vector((0, 0, -1))).normalized()

    def cast(px, py):
        """All hits down the vertical line; keep upward-facing ground (normal flipped to face the capture
        camera, the side the surface was seen from: floors face up, ceilings down, stretched faces sideways)."""
        o = mw_inv @ Vector((px, py, top))
        best = None
        for _ in range(32):
            hit, loc, n, _i = obe.ray_cast(o, down)
            if not hit:
                break
            p = mw @ loc
            o = loc + down * 0.02
            if sky_r and (p - cam).length > 0.9 * sky_r:
                continue                          # sky dome
            nw = (mw.to_3x3() @ n).normalized()
            if nw.dot(cam - p) < 0:
                nw = -nw
            if nw.z > 0.3:
                best = float(p.z)                 # first (highest) upward-facing surface
                break
        return best

    z = cast(x, y)
    if z is None:
        ring = [cast(x + 0.05 * math.cos(a), y + 0.05 * math.sin(a)) for a in (0, 1.57, 3.14, 4.71)]
        ring = [r for r in ring if r is not None]
        z = sum(ring) / len(ring) if ring else None
    return z


def _blender_cam_from_dict(cam: dict, W: int = 1536, H: int = 864) -> dict:
    """world.py camera dict -> the same 'blender' block world.py writes (shared math in world.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import world as _w
    return _w.camera_matrices(cam, W, H)[2]


def camera_from_json(cam, name: str = "world_cam", size=None):
    """<out>.png.camera.json path, its parsed dict, or a world.py camera dict -> Blender camera object,
    set as the scene camera (+ resolution when known)."""
    import bpy
    if isinstance(cam, (str, Path)):
        cam = json.loads(Path(os.path.expanduser(str(cam))).read_text())
    if "blender" in cam:
        b = cam["blender"]
    elif "pos" in cam:
        b = _blender_cam_from_dict(cam, *(size or (1536, 864)))
    else:
        b = cam
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = b.get("sensor_fit", "HORIZONTAL")
    cd.sensor_width = float(b.get("sensor_width_mm", 36.0))
    cd.lens = float(b["lens_mm"])
    cd.clip_start, cd.clip_end = 0.05, 10000.0
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = b["location"]
    ob.rotation_mode = "XYZ"
    ob.rotation_euler = b["rotation_euler_xyz"]
    sc = bpy.context.scene
    sc.camera = ob
    if "resolution" in b:
        sc.render.resolution_x, sc.render.resolution_y = b["resolution"]
        sc.render.resolution_percentage = 100
    return ob


def _setup_render(engine: str, samples: int):
    import bpy
    sc = bpy.context.scene
    sc.view_settings.view_transform = "Standard"   # PLY colours are display sRGB, as in the splat render
    sc.view_settings.look = "None"
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    if engine == "workbench":
        sc.render.engine = "BLENDER_WORKBENCH"
        sh = sc.display.shading
        sh.light = "FLAT"
        sh.color_type = "VERTEX"
        sc.display.render_aa = "8"
    else:
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = samples
        sc.cycles.use_denoising = False
        sc.cycles.max_bounces = 0
    w = bpy.data.worlds.new("black")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    sc.world = w


def _selftest(job: dict):
    import bpy
    import time
    bpy.ops.wm.read_factory_settings(use_empty=True)
    t = time.time()
    s = import_world(job["world"])
    out = {"import_s": round(time.time() - t, 2), "collection": s["collection"].name,
           "objects": {k: (o.name, len(o.data.polygons)) if o else None for k, o in s["objects"].items()},
           "ground": s["ground"]}
    cc = s["capture_camera"]["pos"]
    out["ground_z_under_capture"] = ground_z(s, cc[0], cc[1])
    out["ground_z_samples"] = {f"{x},{y}": ground_z(s, x, y) for x, y in ((0, 3), (0, 6), (-2, 4), (2, 4))}
    cam = camera_from_json({"pos": [0, -1, 1.6], "look_at": [0, 5, 1.2], "fov": 60}, size=(640, 360))
    out["camera"] = {"location": list(cam.location), "rot": list(cam.rotation_euler), "lens": cam.data.lens,
                     "res": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y]}
    print("SELFTEST " + json.dumps(out), flush=True)


def _blender_main(job: dict):
    import bpy
    import time
    if job.get("selftest"):
        return _selftest(job)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    t0 = time.time()
    import_world(job["world"], mesh=job.get("mesh", "full"), backdrop=job.get("backdrop", True))
    t_imp = time.time() - t0
    _setup_render(job.get("engine", "workbench"), int(job.get("samples", 16)))
    for v in job["views"]:
        for o in [o for o in bpy.data.objects if o.type == "CAMERA"]:
            bpy.data.objects.remove(o)
        camera_from_json(v["camera_json"])
        bpy.context.scene.render.filepath = v["out"]
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print(json.dumps({"out": v["out"], "render_s": round(time.time() - t1, 2)}), flush=True)
    if job.get("blend"):
        bpy.ops.wm.save_as_mainfile(filepath=job["blend"])
    print(json.dumps({"import_s": round(t_imp, 2)}), flush=True)


# ----------------------------------------------------------------------------------------------- CLI

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("world")
    r.add_argument("--camera-json")
    r.add_argument("--out")
    r.add_argument("--jobs", help="world.py render jobs list; uses each <out>.camera.json, writes <stem>_bl.png")
    r.add_argument("--engine", choices=["workbench", "cycles"], default="workbench")
    r.add_argument("--samples", type=int, default=16)
    r.add_argument("--blend", help="also save the scene as a .blend")
    r.add_argument("--suffix", default="_bl", help="--jobs output name suffix")
    r.add_argument("--mesh", choices=["full", "stretch", "cut"], default="full")
    r.add_argument("--no-backdrop", action="store_true")
    st = sub.add_parser("selftest", help="import_world + ground_z + camera_from_json inside Blender")
    st.add_argument("world")
    a = ap.parse_args()
    wd = Path(a.world).expanduser().resolve()
    if a.cmd == "selftest":
        cmd = [BLENDER, "--background", "--factory-startup", "-t", THREADS, "--python", str(Path(__file__).resolve()),
               "--", json.dumps({"world": str(wd), "selftest": True})]
        subprocess.run(cmd, check=True, timeout=600)
        return
    if not (wd / "set_mesh.ply").exists():
        sys.exit(f"no set_mesh.ply in {wd}")
    views = []
    if a.jobs:
        for j in json.loads(Path(a.jobs).read_text()):
            o = Path(j["out"])
            views.append({"camera_json": str(o) + ".camera.json", "out": str(o.with_name(o.stem + a.suffix + ".png"))})
    elif a.camera_json and a.out:
        views.append({"camera_json": str(Path(a.camera_json).resolve()), "out": str(Path(a.out).resolve())})
    else:
        sys.exit("render needs --camera-json and --out, or --jobs")
    job = {"world": str(wd), "views": views, "engine": a.engine, "samples": a.samples,
           "mesh": a.mesh, "backdrop": not a.no_backdrop,
           "blend": str(Path(a.blend).resolve()) if a.blend else None}
    cmd = [BLENDER, "--background", "--factory-startup", "-t", THREADS, "--python", str(Path(__file__).resolve()),
           "--", json.dumps(job)]
    subprocess.run(cmd, check=True, timeout=1800)


if __name__ == "__main__":
    if IN_BLENDER:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        if argv:
            _blender_main(json.loads(argv[0]))
    else:
        _cli()
