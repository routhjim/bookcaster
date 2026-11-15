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
    p = os.path.join(OUT, "blender_report.json")
    with open(p + ".tmp", "w") as f:                  # atomic: the scheduler reads it while Blender runs
        json.dump(REPORT, f, indent=1)
    os.replace(p + ".tmp", p)


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
GEOM = ("MESH", "CURVE", "SURFACE", "META", "FONT")


def keep_coordinates(objs):
    """Shader coordinates survive the contract: before transforms are applied, the parts joined and the mesh
    recentred, every part stores its own Object, Generated and world-Position coordinates as vertex attributes,
    and material links from Texture Coordinate (Object, Generated; no other object set) and Geometry (Position)
    are rewired to those attributes. Masks and decals placed by position bake where the model script put them."""
    rewired = 0
    for o in objs:
        me = o.data
        n = len(me.vertices)
        if not n:
            continue
        co = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        lo, hi = co.min(0), co.max(0)
        gen = (co - lo) / np.maximum(hi - lo, 1e-9)
        M = np.array(o.matrix_world)
        wc = co @ M[:3, :3].T + M[:3, 3]
        for name, arr in (("_obj_co", co), ("_gen_co", gen), ("_world_co", wc)):
            a = me.attributes.get(name) or me.attributes.new(name, "FLOAT_VECTOR", "POINT")
            a.data.foreach_set("vector", arr.astype(np.float32).ravel())
    mats = {s.material for o in objs for s in o.material_slots if s.material and s.material.use_nodes}
    for m in mats:
        nt = m.node_tree
        for n in list(nt.nodes):
            if n.type == "TEX_COORD" and getattr(n, "object", None) is None:
                outs = {"Object": "_obj_co", "Generated": "_gen_co"}
            elif n.type == "NEW_GEOMETRY":
                outs = {"Position": "_world_co"}
            else:
                continue
            for sock_name, attr in outs.items():
                sock = n.outputs[sock_name]
                if not sock.is_linked:
                    continue
                at = nt.nodes.new("ShaderNodeAttribute")
                at.attribute_type = "GEOMETRY"
                at.attribute_name = attr
                for lk in list(sock.links):
                    nt.links.new(at.outputs["Vector"], lk.to_socket)
                    rewired += 1
    REPORT["coords_rewired"] = rewired


def degenerate_by_part(objs):
    """Which part (object of the model script) has zero-area faces, and where (world centres of up to 3)."""
    out = []
    for o in objs:
        me = o.data
        n = len(me.polygons)
        if not n:
            continue
        area = np.empty(n, dtype=np.float32)
        me.polygons.foreach_get("area", area)
        bad = np.nonzero(area < 1e-10)[0]
        if len(bad):
            M = o.matrix_world
            out.append({"part": o.name, "faces": int(len(bad)),
                        "at": [[round(x, 4) for x in (M @ me.polygons[int(i)].center)] for i in bad[:3]]})
    REPORT["degenerate_by_part"] = out


def apply_contract():
    sc = bpy.context.scene
    # modifiers first, while every object they may reference (empties for Mirror/Array, hidden boolean cutters)
    # still exists; then everything that is not a visible part goes
    objs = [o for o in sc.objects if o.type in GEOM and o.visible_get()]
    if not objs:
        raise RuntimeError("the model script made no visible mesh")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    names = {o.name for o in objs}
    bpy.ops.object.convert(target="MESH")              # applies modifiers, curves -> mesh
    objs = [o for o in sc.objects if o.type == "MESH" and (o.name in names or o.select_get())]
    if not objs:
        raise RuntimeError("the model script made no visible mesh")
    bpy.context.view_layer.update()
    world = {o.name: o.matrix_world.copy() for o in objs}       # before parents (empties) go
    for o in list(sc.objects):
        if o not in objs:
            if o.type != "EMPTY":
                REPORT["warnings"].append(f"removed {'hidden' if o.type in GEOM else 'non-mesh'} object "
                                          f"{o.name} ({o.type})")
            bpy.data.objects.remove(o, do_unlink=True)
    for o in objs:
        o.parent = None
        o.matrix_world = world[o.name]
    bpy.context.view_layer.update()
    keep_coordinates(objs)
    degenerate_by_part(objs)
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
    if SPEC.get("seen_from_inside"):                      # a shell people are filmed inside: its faces look inward
        check("normals_inward", vol < 0, round(vol, 6), "< 0", "signed volume of the mesh (a shell seen from inside)")
    else:
        check("normals_outward", vol > 0, round(vol, 6), "> 0", "signed volume of the mesh (negative = inside out)")
    REPORT["non_manifold_edges"] = nonman
    degen = sum(1 for f in bm.faces if f.calc_area() < 1e-10)
    check("degenerate_faces", degen == 0, degen, 0,
          "; ".join(f"{d['part']}: {d['faces']} at {d['at']}" for d in REPORT.get("degenerate_by_part", []))
          + (" (world positions before recentring)" if REPORT.get("degenerate_by_part") else ""))
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
            p.use_hiprt = bool(JOB.get("hiprt", False))
            sc.cycles.device = "GPU"
            if hasattr(sc.cycles, "denoising_use_gpu"):
                sc.cycles.denoising_use_gpu = True        # OIDN on the card, not the CPU
            REPORT["device"] = f"HIP {want}" + (" hiprt" if p.use_hiprt else "")
            return
        raise RuntimeError(f"no HIP device {want}: the chain never bakes or renders on the CPU")
    sc.cycles.device = "CPU"                              # only when the job asked for "CPU" (manual debugging)
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


def _linked_or(pr, name, test):
    """True if the Principled input is linked, or its default passes test."""
    if pr is None or name not in pr.inputs:
        return False
    i = pr.inputs[name]
    return i.is_linked or test(i.default_value)


def is_glass(m):
    """A glass slot: spec "glass_materials" names it, the material has a custom property asset_glass, or its
    Principled transmission is >= 0.5 (unlinked). Glass is not baked: it exports as its own transmissive
    material (KHR_materials_transmission) with constant factors."""
    if m.name in SPEC.get("glass_materials", []) or m.get("asset_glass"):
        return True
    _, pr = _principled(m)
    if pr is None or "Transmission Weight" not in pr.inputs:
        return False
    t = pr.inputs["Transmission Weight"]
    return (not t.is_linked) and t.default_value >= 0.5


def has_emission(m):
    _, pr = _principled(m)
    if pr is None:
        return False
    col = _linked_or(pr, "Emission Color", lambda v: max(v[:3]) > 1e-4)
    stren = _linked_or(pr, "Emission Strength", lambda v: v > 1e-4)
    return col and stren


def has_normal_detail(m):
    """Anything that bends the shading normal: a linked Principled Normal input or a material displacement."""
    out, pr = _principled(m)
    return bool((pr is not None and pr.inputs["Normal"].is_linked)
                or (out is not None and out.inputs["Displacement"].is_linked))


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
    glass = [m for m in mats if is_glass(m)]
    emissive = any(has_emission(m) for m in mats if m not in glass)
    REPORT["glass_materials"] = [m.name for m in glass]
    for m in mats:
        m.use_nodes = True
    imgs = {k: _img(f"T_{AID}_{k}", res, k == "BaseColor")
            for k in ("BaseColor", "Normal", "RMA")}
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

    defaults = {"Base Color": (0.5, 0.5, 0.5, 1), "Roughness": 0.6, "Metallic": 0.0,
                "Emission Color": (0, 0, 0, 1), "Emission Strength": 0.0}

    def emit_bake(key, source, samples=4):
        """Bake through an Emission shader. source: a Principled input name (colour), or three entries packed into
        R, G, B: Principled input names or "AO" (an Ambient Occlusion node). One pass instead of one per map."""
        saved = []
        for m in mats:
            nt = m.node_tree
            out, pr = _principled(m)
            if out is None:
                out = nt.nodes.new("ShaderNodeOutputMaterial")
            if pr is None:
                REPORT["warnings"].append(f"material {m.name} has no Principled BSDF; baked as defaults")
            old = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].is_linked else None
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            tmp = [em]

            def feed(name, dst):
                if name == "AO":
                    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
                    ao.samples = int(JOB.get("ao_node_samples") or {"hero": 16, "mid": 8, "background": 4}[tier])
                    ao.inputs["Distance"].default_value = sc.world.light_settings.distance
                    tmp.append(ao)
                    nt.links.new(ao.outputs["AO"], dst)
                    return
                src = pr.inputs[name] if pr is not None else None
                if src is not None and src.is_linked:
                    nt.links.new(src.links[0].from_socket, dst)
                else:
                    v = src.default_value if src is not None else defaults[name]
                    if dst.type == "RGBA":
                        dst.default_value = tuple(v) if hasattr(v, "__len__") else (v, v, v, 1)
                    else:
                        dst.default_value = v if not hasattr(v, "__len__") else v[0]

            if source == "Emission":                  # colour x strength (HDR: baked to a float image)
                feed("Emission Color", em.inputs["Color"])
                feed("Emission Strength", em.inputs["Strength"])
            elif isinstance(source, str):
                feed(source, em.inputs["Color"])
            else:
                cc = nt.nodes.new("ShaderNodeCombineColor")
                tmp.append(cc)
                for name, ch in zip(source, ("Red", "Green", "Blue")):
                    feed(name, cc.inputs[ch])
                nt.links.new(cc.outputs["Color"], em.inputs["Color"])
            nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
            saved.append((nt, out, old, tmp))
        target(key)
        sc.cycles.samples = samples
        bpy.ops.object.bake(type="EMIT", use_clear=True)
        for nt, out, old, tmp in saved:
            for n in tmp:
                nt.nodes.remove(n)
            if old is not None:
                nt.links.new(old, out.inputs["Surface"])

    es = int(JOB.get("bake_samples", 4))
    with Timer("bake"):
        with Timer("bake_basecolor"):
            emit_bake("BaseColor", "Base Color", es)
        with Timer("bake_rough_metal_ao"):      # one pass: R roughness, G metallic, B occlusion (AO node)
            emit_bake("RMA", ["Roughness", "Metallic", "AO"], es)
        with Timer("bake_normal"):
            if any(has_normal_detail(m) for m in mats if m not in glass):
                target("Normal")
                sc.cycles.samples = es
                bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_clear=True)
            else:                                 # nothing bends the normal: a flat map, no bake (saves ~1/3)
                flat = np.empty((res, res, 4), dtype=np.float32)
                flat[...] = (0.5, 0.5, 1.0, 1.0)
                imgs["Normal"].pixels.foreach_set(flat.ravel())
                REPORT["normal_bake"] = "flat (no normal input linked)"
        emis_strength = 0.0
        if emissive:
            with Timer("bake_emissive"):
                imgs["Emissive"] = bpy.data.images.new(f"T_{AID}_Emissive_f", res, res, alpha=False,
                                                       float_buffer=True)
                imgs["Emissive"].colorspace_settings.name = "Linear Rec.709"
                emit_bake("Emissive", "Emission", es)
                e = np.empty(res * res * 4, dtype=np.float32)
                imgs["Emissive"].pixels.foreach_get(e)
                e = e.reshape(res, res, 4)
                emis_strength = float(max(e[..., :3].max(), 1e-6))
                em8 = _img(f"T_{AID}_Emissive", res, True)
                lin = np.clip(e[..., :3] / emis_strength, 0, 1)
                enc = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
                o8 = np.ones((res, res, 4), dtype=np.float32)
                o8[..., :3] = enc
                em8.colorspace_settings.name = "Non-Color"    # write the encoded values as they are...
                em8.pixels.foreach_set(o8.ravel())
                bpy.data.images.remove(imgs["Emissive"])
                imgs["Emissive"] = em8
                REPORT["emission_strength"] = round(emis_strength, 4)

    def px(im):
        a = np.empty(res * res * 4, dtype=np.float32)
        im.pixels.foreach_get(a)
        return a.reshape(res, res, 4)

    orm = _img(f"T_{AID}_ORM", res, False)
    o = np.ones((res, res, 4), dtype=np.float32)
    rma = px(imgs["RMA"])
    # a failed bake comes back all zeros (seen 11-10 when many bakes shared the cards: every map black); real
    # occlusion is never zero everywhere, so fail the job instead of shipping a black asset
    if float(rma[..., 2].max()) <= 0.0 and float(px(imgs["BaseColor"])[..., :3].max()) <= 0.0:
        raise RuntimeError("bake came back empty (BaseColor and occlusion all zero): GPU bake failed, re-run")
    o[..., 0], o[..., 1], o[..., 2] = rma[..., 2], rma[..., 0], rma[..., 1]
    orm.pixels.foreach_set(o.ravel())
    files = {}
    maps = [("BaseColor", imgs["BaseColor"]), ("Normal", imgs["Normal"]), ("ORM", orm)]
    if "Emissive" in imgs:
        maps.append(("Emissive", imgs["Emissive"]))
    for key, im in maps:
        p = os.path.join(OUT, "textures", f"T_{AID}_{key}.png")
        im.filepath_raw = p
        im.file_format = "PNG"
        im.save()
        if key == "Emissive":
            im.colorspace_settings.name = "sRGB"            # ...then read them back as colour
        im.source = "FILE"
        im.filepath = p
        files[key] = p
    REPORT["files"]["textures"] = files
    REPORT["texture_px"] = res
    # one material on the baked maps
    mat = bpy.data.materials.new(f"M_{AID}")
    mat.use_nodes = True
    if SPEC.get("seen_from_inside"):                      # a shell's walls have no back: a camera outside sees in
        mat.use_backface_culling = True
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
    if "Emissive" in imgs:
        te = nt.nodes.new("ShaderNodeTexImage")
        te.image = imgs["Emissive"]
        nt.links.new(te.outputs["Color"], pr.inputs["Emission Color"])
        pr.inputs["Emission Strength"].default_value = emis_strength
    # one transmissive material per glass slot, from that slot's constant factors (not baked): M_<id>_Glass,
    # M_<id>_Glass2, ... (a frosted globe and a clear chimney keep their own looks); emission carries over (a lit globe)
    gslot = {}                                       # old slot index -> new slot index
    gmats = []
    for i, sl in enumerate(ob.material_slots):
        if sl.material in glass and sl.material not in [g for g, _ in gmats]:
            _, gp = _principled(sl.material)
            gmat = bpy.data.materials.new(f"M_{AID}_Glass" + (str(len(gmats) + 1) if gmats else ""))
            gmat.use_nodes = True
            gpr = gmat.node_tree.nodes["Principled BSDF"]
            for k in ("Base Color", "Roughness", "IOR", "Transmission Weight", "Metallic", "Alpha",
                      "Emission Color", "Emission Strength"):
                if gp is not None and k in gp.inputs and not gp.inputs[k].is_linked:
                    gpr.inputs[k].default_value = gp.inputs[k].default_value
            if gpr.inputs["Transmission Weight"].default_value < 0.5:
                gpr.inputs["Transmission Weight"].default_value = 1.0
            gmats.append((sl.material, gmat))
        if sl.material in glass:
            gslot[i] = 1 + [g for g, _ in gmats].index(sl.material)
    REPORT["glass"] = [{k: (list(gm.node_tree.nodes["Principled BSDF"].inputs[k].default_value)
                            if hasattr(gm.node_tree.nodes["Principled BSDF"].inputs[k].default_value, "__len__")
                            else gm.node_tree.nodes["Principled BSDF"].inputs[k].default_value)
                        for k in ("Base Color", "Roughness", "IOR", "Transmission Weight", "Emission Strength")}
                       for _, gm in gmats]
    old_idx = np.empty(len(ob.data.polygons), dtype=np.int32)
    ob.data.polygons.foreach_get("material_index", old_idx)
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    for _, gm in gmats:
        ob.data.materials.append(gm)
    lut = np.zeros(max(len(ob.material_slots), int(old_idx.max(initial=0)) + 1), dtype=np.int32)
    for i, j in gslot.items():
        lut[i] = j
    new_idx = lut[old_idx]
    ob.data.polygons.foreach_set("material_index", new_idx)
    REPORT["glass_faces"] = int((new_idx > 0).sum())
    for m in mats:
        if m.users == 0:
            bpy.data.materials.remove(m)
    bpy.data.images.remove(imgs["RMA"])


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
    sc.cycles.use_denoising = bool(JOB.get("denoise", True))   # OIDN runs on the CPU here (0.4-4 s a tile)
    sc.render.resolution_x = sc.render.resolution_y = px
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "AgX"
    centre = Vector((0, 0, dims.z / 2))
    cam.data.lens = 50
    fov = 2 * math.atan(18 / cam.data.lens)
    dist = r / math.sin(fov / 2) * 1.05
    hx, hy = dims.x / 2, dims.y / 2
    corners = [Vector((x, y, z)) for x in (-hx, hx) for y in (-hy, hy) for z in (0, dims.z)]
    tiles = []
    for name, az, el in VIEWS:
        a, e = math.radians(az), math.radians(el)
        cam.location = centre + dist * Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        # frame the tile on the bounding box as seen from here (a long, flat asset fills the tile)
        inv = cam.matrix_world.inverted()
        t = 0.0
        for cn in corners:
            q = inv @ cn
            if q.z < -1e-6:
                t = max(t, abs(q.x / -q.z), abs(q.y / -q.z))
        cam.data.lens = 18.0 / max(t * 1.06, 1e-4)
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
        with Timer("export"):
            REPORT["files"]["export"] = {f: export(ob, f) for f in JOB.get("formats", ["glb"])}
        save_report("exported")                       # the scheduler starts the UE import now, beside the sheet
        if JOB.get("sheet", True):
            with Timer("sheet"):
                REPORT["files"]["sheet"] = render_sheet(ob, "sheet", int(JOB.get("sheet_px", 512)),
                                                          int(JOB.get("sheet_samples", 32)))
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
