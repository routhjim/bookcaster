"""Blender side of tools/assets/run_asset.py: one model script + spec -> checked, baked, exported asset.

    blender -b --factory-startup -t 16 --python tools/assets/bl_harness.py -- JOB.json

JOB {"spec": {...asset spec, see CONTRACT.md...}, "model": model script path, "out": asset folder,
     "mode": "final" | "preview" (preview = steps 1-3 + a small unbaked sheet; no bake, no export),
     "formats": ["glb"] (also "fbx", "usd" for format tests), "sheet": true, "source_sheet": false,
     "device": "CPU" | "HIP", "hip_bus": "05" (the card whose lock the caller holds)}

Steps (CONTRACT.md has the rules):
1. MODEL: empty scene, metric units; run the model script (it may read SPEC; it may define build(SPEC)).
2. CONTRACT: drop non-mesh objects, apply modifiers and transforms, join into one mesh SM_<id>,
   floor centre of the bounding box to the origin.
3. CHECKS: size against spec.size_m, triangles against the tier, materials, normals, loose geometry,
   floating parts, degenerate faces.
4. UV + BAKE (Cycles CPU): smart-UV unless spec.keep_uvs; procedural materials -> T_<id>_BaseColor,
   T_<id>_Normal (OpenGL, +Y), T_<id>_ORM (R occlusion, G roughness, B metallic); one material M_<id>.
5. SHEET: 8 views of the baked asset on a neutral 10 cm checker stage -> sheet/<id>_sheet.png.
6. EXPORT: export/<id>.glb (+ other formats if asked); source/<id>.blend; blender_report.json.
"""

import json
import math
import os
import runpy
import sys
import time
import traceback

import bmesh
import bpy
import numpy as np
from mathutils import Vector

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
SPEC = JOB["spec"]
AID = SPEC["id"]
OUT = JOB["out"]
TIERS = {"hero": (250_000, 2048), "mid": (60_000, 1024), "background": (8_000, 512)}
REPORT = {"id": AID, "blender": bpy.app.version_string, "times_s": {}, "checks": [], "warnings": [], "files": {}}
for d in ("source", "export", "textures", "sheet"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)


def save_report(status):
    REPORT["status"] = status
    with open(os.path.join(OUT, "blender_report.json"), "w") as f:
        json.dump(REPORT, f, indent=1)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        REPORT["times_s"][self.name] = round(time.time() - self.t, 2)


def check(name, ok, value=None, limit=None, note=""):
    REPORT["checks"].append({"name": name, "ok": bool(ok), "value": value, "limit": limit, "note": note})


# ---------------------------------------------------------------- 1. MODEL
def run_model():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    us = bpy.context.scene.unit_settings
    us.system, us.scale_length, us.length_unit = "METRIC", 1.0, "METERS"
    g = runpy.run_path(JOB["model"], init_globals={"SPEC": SPEC}, run_name="__asset_model__")
    if not [o for o in bpy.context.scene.objects if o.type == "MESH"] and callable(g.get("build")):
        g["build"](SPEC)


# ---------------------------------------------------------------- 2. CONTRACT
def apply_contract():
    sc = bpy.context.scene
    for o in list(sc.objects):
        if o.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT"):
            if o.type != "EMPTY":
                REPORT["warnings"].append(f"removed non-mesh object {o.name} ({o.type})")
            bpy.data.objects.remove(o, do_unlink=True)
    objs = [o for o in sc.objects if o.visible_get()]
    for o in list(sc.objects):
        if o not in objs:
            REPORT["warnings"].append(f"removed hidden object {o.name}")
            bpy.data.objects.remove(o, do_unlink=True)
    if not objs:
        raise RuntimeError("the model script made no visible mesh")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.convert(target="MESH")              # applies modifiers, curves -> mesh
    objs = [o for o in sc.objects if o.type == "MESH"]
    for o in objs:
        o.parent = None
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(objs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = ob.data.name = f"SM_{AID}"
    vs = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", vs)
    vs = vs.reshape(-1, 3)
    lo, hi = vs.min(0), vs.max(0)
    shift = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    ob.data.transform(__import__("mathutils").Matrix.Translation(Vector(shift)))
    ob.data.update()
    ob.location = (0, 0, 0)
    REPORT["origin_shift_m"] = [round(float(x), 4) for x in shift]
    return ob


# ---------------------------------------------------------------- 3. CHECKS
def islands(bm):
    comp, seen = [], set()
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, cur = [v], [v.index]
        seen.add(v.index)
        while stack:
            x = stack.pop()
            for e in x.link_edges:
                y = e.other_vert(x)
                if y.index not in seen:
                    seen.add(y.index)
                    cur.append(y.index)
                    stack.append(y)
        comp.append(cur)
    return comp


def run_checks(ob):
    me = ob.data
    tier = SPEC.get("tier", "mid")
    max_tris = TIERS[tier][0]
    vs = np.array([v.co[:] for v in me.vertices])
    dims = (vs.max(0) - vs.min(0)).tolist()
    REPORT["dims_m"] = [round(d, 4) for d in dims]
    want = SPEC.get("size_m")
    tol = float(SPEC.get("size_tolerance", 0.15))
    if want:
        bad = [f"{'xyz'[i]} {dims[i]:.3f} m v {want[i]:.3f} m" for i in range(3)
               if want[i] and abs(dims[i] - want[i]) > tol * want[i]]
        check("size", not bad, [round(d, 3) for d in dims], want,
              "; ".join(bad) or f"each axis within {tol:.0%}")
    me.calc_loop_triangles()
    tris = len(me.loop_triangles)
    REPORT["tris"] = tris
    check("triangles", tris <= max_tris, tris, max_tris, f"tier {tier}")
    slots = [s.material for s in ob.material_slots]
    used = {p.material_index for p in me.polygons}
    empty = [i for i in used if i >= len(slots) or slots[i] is None]
    check("materials", bool(slots) and not empty, [m.name for m in slots if m], None,
          f"faces on empty slots {empty}" if empty else ("no material" if not slots else "every face has a material"))
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    loose_v = sum(1 for v in bm.verts if not v.link_edges)
    loose_e = sum(1 for e in bm.edges if not e.link_faces)
    check("loose_geometry", loose_v + loose_e == 0, {"verts": loose_v, "edges": loose_e}, 0)
    flipped = 0
    nonman = 0
    for e in bm.edges:
        lf = e.link_loops
        if len(lf) == 2:
            if lf[0].vert == lf[1].vert:
                flipped += 1
        elif len(lf) != 0:
            nonman += 1
    bmt = bm.copy()
    bmesh.ops.triangulate(bmt, faces=bmt.faces)
    vol = 0.0
    for f in bmt.faces:
        a, b, c = (v.co for v in f.verts)
        vol += a.dot(b.cross(c)) / 6.0
    bmt.free()
    check("normals_consistent", flipped == 0, flipped, 0, "edges whose two faces wind the same way")
    check("normals_outward", vol > 0, round(vol, 6), "> 0", "signed volume of the mesh (negative = inside out)")
    REPORT["non_manifold_edges"] = nonman
    degen = sum(1 for f in bm.faces if f.calc_area() < 1e-10)
    check("degenerate_faces", degen == 0, degen, 0)
    comps = islands(bm)
    boxes = []
    for c in comps:
        p = vs[c]
        boxes.append((p.min(0), p.max(0), len(c)))
    eps = 0.002
    floating = []
    for i, (lo, hi, n) in enumerate(boxes):
        if lo[2] <= eps:
            continue
        touch = any(j != i and np.all(lo <= boxes[j][1] + eps) and np.all(boxes[j][0] <= hi + eps)
                    for j in range(len(boxes)))
        if not touch:
            floating.append({"verts": n, "min": lo.round(3).tolist(), "max": hi.round(3).tolist()})
    REPORT["parts"] = len(comps)
    check("floating_parts", not floating, floating, [],
          "parts off the floor whose bounding box touches no other part")
    bm.free()


# ---------------------------------------------------------------- device
def set_device():
    """Cycles device for bake and sheet. JOB device "HIP" + hip_bus "05": that one card only (no CPU, no iGPU);
    the caller holds the card's lock. Anything else, or no matching card: CPU."""
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    if JOB.get("device") == "HIP":
        p = bpy.context.preferences.addons["cycles"].preferences
        p.compute_device_type = "HIP"
        p.get_devices()
        want = f"0000:{JOB.get('hip_bus', '')}:00"
        hit = False
        for d in p.devices:
            d.use = d.type == "HIP" and d.id.endswith(want)
            hit = hit or d.use
        if hit:
            sc.cycles.device = "GPU"
            REPORT["device"] = f"HIP {want}"
            return
        REPORT["warnings"].append(f"no HIP device {want}; using the CPU")
    sc.cycles.device = "CPU"
    REPORT["device"] = "CPU"


# ---------------------------------------------------------------- 4. UV + BAKE
def triangulate_ngons(ob):
    """Faces with more than 4 corners -> triangles (tangents for the normal map need tris or quads)."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    ng = [f for f in bm.faces if len(f.verts) > 4]
    if ng:
        bmesh.ops.triangulate(bm, faces=ng, quad_method="BEAUTY", ngon_method="BEAUTY")
        bm.to_mesh(ob.data)
    bm.free()
    REPORT["ngons_triangulated"] = len(ng)


def ensure_uvs(ob):
    me = ob.data
    if SPEC.get("keep_uvs") and me.uv_layers:
        check("uvs", True, len(me.uv_layers), None, "kept the model's UVs")
        return
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    me.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004, area_weight=0.0)
    bpy.ops.uv.pack_islands(margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = np.empty(len(me.loops) * 2)
    me.uv_layers[0].data.foreach_get("uv", uv)
    inside = bool(uv.min() >= -1e-4 and uv.max() <= 1 + 1e-4)
    check("uvs", inside, len(me.uv_layers), None, "smart UV project + pack, all in 0-1" if inside else "UVs outside 0-1")


def _principled(mat):
    if not mat.use_nodes:
        return None, None
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) \
        or next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    pr = None
    if out and out.inputs["Surface"].is_linked and out.inputs["Surface"].links[0].from_node.type == "BSDF_PRINCIPLED":
        pr = out.inputs["Surface"].links[0].from_node
    pr = pr or next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    return out, pr


def _img(name, res, color):
    im = bpy.data.images.new(name, res, res, alpha=False, float_buffer=False)
    im.colorspace_settings.name = "sRGB" if color else "Non-Color"
    return im


def bake(ob):
    sc = bpy.context.scene
    tier = SPEC.get("tier", "mid")
    res = int(SPEC.get("texture_px", TIERS[tier][1]))
    sc.render.engine = "CYCLES"
    set_device()
    sc.render.bake.margin = max(4, res // 128)
    sc.render.bake.margin_type = "EXTEND"
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    sc.world.light_settings.distance = 0.15 * max(REPORT["dims_m"])
    mats = [s.material for s in ob.material_slots if s.material]
    for m in mats:
        m.use_nodes = True
    imgs = {k: _img(f"T_{AID}_{k}", res, k == "BaseColor")
            for k in ("BaseColor", "Normal", "Rough", "Metal", "AO")}
    nodes = {}
    for m in mats:
        tn = m.node_tree.nodes.new("ShaderNodeTexImage")
        nodes[m.name] = tn
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob

    def target(key):
        for m in mats:
            tn = nodes[m.name]
            tn.image = imgs[key]
            for n in m.node_tree.nodes:
                n.select = False
            tn.select = True
            m.node_tree.nodes.active = tn

    def emit_bake(key, socket_name, samples=4):
        saved = []
        for m in mats:
            nt = m.node_tree
            out, pr = _principled(m)
            if out is None:
                out = nt.nodes.new("ShaderNodeOutputMaterial")
            old = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].is_linked else None
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            if pr is None:
                REPORT["warnings"].append(f"material {m.name} has no Principled BSDF; baked as grey")
                em.inputs["Color"].default_value = {"BaseColor": (0.5, 0.5, 0.5, 1), "Rough": (0.6,) * 3 + (1,),
                                                    "Metal": (0, 0, 0, 1)}[key]
            else:
                src = pr.inputs[socket_name]
                if src.is_linked:
                    nt.links.new(src.links[0].from_socket, em.inputs["Color"])
                else:
                    v = src.default_value
                    em.inputs["Color"].default_value = tuple(v) if hasattr(v, "__len__") else (v, v, v, 1)
            nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
            saved.append((nt, out, old, em))
        target(key)
        sc.cycles.samples = samples
        bpy.ops.object.bake(type="EMIT", use_clear=True)
        for nt, out, old, em in saved:
            nt.nodes.remove(em)
            if old is not None:
                nt.links.new(old, out.inputs["Surface"])

    with Timer("bake"):
        emit_bake("BaseColor", "Base Color")
        emit_bake("Rough", "Roughness")
        emit_bake("Metal", "Metallic")
        target("Normal")
        sc.cycles.samples = 4
        bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_clear=True)
        target("AO")
        sc.cycles.samples = int(SPEC.get("ao_samples", {"hero": 32, "mid": 16, "background": 8}[tier]))
        bpy.ops.object.bake(type="AO", use_clear=True)

    def px(im):
        a = np.empty(res * res * 4, dtype=np.float32)
        im.pixels.foreach_get(a)
        return a.reshape(res, res, 4)

    orm = _img(f"T_{AID}_ORM", res, False)
    o = np.ones((res, res, 4), dtype=np.float32)
    o[..., 0] = px(imgs["AO"])[..., 0]
    o[..., 1] = px(imgs["Rough"])[..., 0]
    o[..., 2] = px(imgs["Metal"])[..., 0]
    orm.pixels.foreach_set(o.ravel())
    files = {}
    for key, im in (("BaseColor", imgs["BaseColor"]), ("Normal", imgs["Normal"]), ("ORM", orm)):
        p = os.path.join(OUT, "textures", f"T_{AID}_{key}.png")
        im.filepath_raw = p
        im.file_format = "PNG"
        im.save()
        im.source = "FILE"
        im.filepath = p
        files[key] = p
    REPORT["files"]["textures"] = files
    REPORT["texture_px"] = res
    # one material on the baked maps
    mat = bpy.data.materials.new(f"M_{AID}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    pr = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(pr.outputs["BSDF"], out.inputs["Surface"])
    tb = nt.nodes.new("ShaderNodeTexImage")
    tb.image = imgs["BaseColor"]
    nt.links.new(tb.outputs["Color"], pr.inputs["Base Color"])
    to = nt.nodes.new("ShaderNodeTexImage")
    to.image = orm
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(to.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], pr.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], pr.inputs["Metallic"])
    tn = nt.nodes.new("ShaderNodeTexImage")
    tn.image = imgs["Normal"]
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], pr.inputs["Normal"])
    # glTF occlusion goes through a "glTF Material Output" group (the exporter reads its Occlusion input)
    grp = bpy.data.node_groups.get("glTF Material Output") or bpy.data.node_groups.new("glTF Material Output",
                                                                                        "ShaderNodeTree")
    if not any(getattr(i, "name", "") == "Occlusion" for i in grp.interface.items_tree):
        grp.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    gn = nt.nodes.new("ShaderNodeGroup")
    gn.node_tree = grp
    nt.links.new(sep.outputs["Red"], gn.inputs["Occlusion"])
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    for p in ob.data.polygons:
        p.material_index = 0
    for m in mats:
        if m.users == 0:
            bpy.data.materials.remove(m)
    for k in ("Rough", "Metal", "AO"):
        bpy.data.images.remove(imgs[k])


# ---------------------------------------------------------------- 5. SHEET
VIEWS = [("front", 0, 12), ("right", 90, 12), ("back", 180, 12), ("left", 270, 12),
         ("front_left_high", -45, 35), ("front_right_high", 45, 35), ("back_right_high", 135, 35),
         ("back_left_high", 225, 35)]


def build_stage(ob):
    sc = bpy.context.scene
    dims = Vector(REPORT["dims_m"])
    r = max(dims.length / 2, 0.05)
    bpy.ops.mesh.primitive_plane_add(size=max(40 * r, 4), location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.name = "stage_floor"
    m = bpy.data.materials.new("stage_floor")
    m.use_nodes = True
    nt = m.node_tree
    pr = nt.nodes["Principled BSDF"]
    ck = nt.nodes.new("ShaderNodeTexChecker")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], ck.inputs["Vector"])
    ck.inputs["Scale"].default_value = 5.0      # Object coords are metres: 5 -> 0.1 m squares
    ck.inputs["Color1"].default_value = (0.20, 0.20, 0.20, 1)
    ck.inputs["Color2"].default_value = (0.15, 0.15, 0.15, 1)
    nt.links.new(ck.outputs["Color"], pr.inputs["Base Color"])
    pr.inputs["Roughness"].default_value = 0.85
    fl.data.materials.append(m)
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.35, 0.36, 0.38, 1)
    bg.inputs["Strength"].default_value = 0.6
    for name, loc, energy, size in (("key", (-3, -4, 5), 900, 3), ("rim", (3, 4, 3), 400, 2)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy * (r / 0.4) ** 2
        ld.size = size * r / 0.4 + 0.5
        lo = bpy.data.objects.new(name, ld)
        lo.location = Vector(loc) * (r / 0.4 + 0.3)
        sc.collection.objects.link(lo)
        d = -lo.location + Vector((0, 0, dims.z / 2))
        lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cd = bpy.data.cameras.new("cam")
    cd.lens = 50
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam, r, dims


def render_sheet(ob, tag, px=512, samples=32):
    sc = bpy.context.scene
    cam, r, dims = build_stage(ob) if "cam" not in bpy.data.objects else (bpy.data.objects["cam"],
                                                                        max(Vector(REPORT["dims_m"]).length / 2, 0.05),
                                                                        Vector(REPORT["dims_m"]))
    sc.render.engine = "CYCLES"
    set_device()
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x = sc.render.resolution_y = px
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "AgX"
    centre = Vector((0, 0, dims.z / 2))
    fov = 2 * math.atan(18 / cam.data.lens)
    dist = r / math.sin(fov / 2) * 1.05
    tiles = []
    for name, az, el in VIEWS:
        a, e = math.radians(az), math.radians(el)
        cam.location = centre + dist * Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
        p = os.path.join(OUT, "sheet", f"{tag}_{name}.png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        tiles.append(p)
    rows = []
    for i in range(0, 8, 4):
        row = []
        for p in tiles[i:i + 4]:
            im = bpy.data.images.load(p)
            a = np.empty(px * px * 4, dtype=np.float32)
            im.pixels.foreach_get(a)
            row.append(a.reshape(px, px, 4))
            bpy.data.images.remove(im)
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows[::-1], axis=0)       # Blender pixel rows run bottom-up
    out = bpy.data.images.new(f"{tag}_sheet", 4 * px, 2 * px, alpha=False)
    out.pixels.foreach_set(sheet.ravel())
    sp = os.path.join(OUT, "sheet", f"{AID}_{tag}.png")
    out.filepath_raw = sp
    out.file_format = "PNG"
    out.save()
    for p in tiles:
        os.remove(p)
    return sp


# ---------------------------------------------------------------- 6. EXPORT
def export(ob, fmt):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    p = os.path.join(OUT, "export", f"{AID}.{fmt}")
    if fmt == "glb":
        bpy.ops.export_scene.gltf(filepath=p, export_format="GLB", use_selection=True, export_apply=True,
                                  export_yup=True, export_image_format="AUTO", export_tangents=True)
    elif fmt == "fbx":
        bpy.ops.export_scene.fbx(filepath=p, use_selection=True, apply_unit_scale=True,
                                 apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z", axis_up="Y",
                                 path_mode="COPY", embed_textures=True, mesh_smooth_type="FACE",
                                 use_tspace=True, bake_space_transform=False)
    elif fmt == "usd":
        p = p[:-4] + ".usdc"
        bpy.ops.wm.usd_export(filepath=p, selected_objects_only=True, export_materials=True)
    return p


def main():
    t0 = time.time()
    try:
        with Timer("model"):
            run_model()
        with Timer("contract"):
            ob = apply_contract()
        with Timer("checks"):
            run_checks(ob)
        if JOB.get("mode") == "preview":
            # the modelling loop: shape and checks only, unbaked materials, small sheet
            with Timer("preview_sheet"):
                REPORT["files"]["preview_sheet"] = render_sheet(ob, "preview", int(JOB.get("preview_px", 320)),
                                                                int(JOB.get("preview_samples", 12)))
            REPORT["times_s"]["total"] = round(time.time() - t0, 2)
            save_report("ok")
            return
        if JOB.get("source_sheet"):
            with Timer("source_sheet"):
                bpy.context.scene.render.engine = "CYCLES"
                REPORT["files"]["source_sheet"] = render_sheet(ob, "source")
        with Timer("uv"):
            triangulate_ngons(ob)
            ensure_uvs(ob)
        for o in [o for o in bpy.context.scene.objects if o.name.startswith("stage_")]:
            o.hide_render = True
        bake(ob)
        for o in [o for o in bpy.context.scene.objects if o.name.startswith("stage_")]:
            o.hide_render = False
        if JOB.get("sheet", True):
            with Timer("sheet"):
                REPORT["files"]["sheet"] = render_sheet(ob, "sheet")
        with Timer("export"):
            REPORT["files"]["export"] = {f: export(ob, f) for f in JOB.get("formats", ["glb"])}
        bp = os.path.join(OUT, "source", f"{AID}.blend")
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=bp, compress=True)
        REPORT["files"]["blend"] = bp
        REPORT["times_s"]["total"] = round(time.time() - t0, 2)
        save_report("ok")
    except Exception:
        REPORT["error"] = traceback.format_exc()
        REPORT["times_s"]["total"] = round(time.time() - t0, 2)
        save_report("error")
        sys.stderr.write(REPORT["error"])
        sys.exit(2)


main()
