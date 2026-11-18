"""Model script (run_asset.py) for a picture-to-3D asset: the .glb that img3d.py wrote, with its own texture.

Spec keys:
    "source_glb"   img3d.py's .glb (base-colour texture on xatlas UVs)
    "yaw_deg"      turn about Z after import so the asset's front faces -Y (contract)       [0]
    "height_m"     scale so the height is this (else the spec's size_m[2])
    "max_tris"     decimate to this many triangles when over it (tier budget)              [the tier's budget - 2%]
    "bump_m"       detail-normal depth from the colour's luminance (0 = none)              [0.002 x height]
    "rough"        [lo, hi]: roughness from luminance, bright -> lo, dark -> hi              [0.45, 0.8]
    "keep_uvs": true (the port's atlas is non-overlapping and inside 0-1)
    "seen_from_inside": true for a shell whose faces look inward (img3d/pano_shell.py): its winding is kept
    "emit_texture" strength: the texture also as emission (an outdoor place's land and sky give their own light)

The port gives base colour only. A normal map and a roughness map are derived here as node maths on the colour (a
bump from its luminance, roughness from its luminance) and the harness bakes them on its GPU like any material.
The decimation (Blender's collapse) is the one CPU step: ~1-3 s for a few tens of thousands of faces.
"""
import math
import time

import bpy
from mathutils import Matrix

aid = SPEC["id"]
SPECK_POINTS = 40           # a loose piece of at most this many points that touches no other piece is a speck
TIER_TRIS = {"hero": 250000, "mid": 60000, "background": 8000}
t0 = time.time()
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=SPEC["source_glb"])
meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
for o in bpy.data.objects:
    if o not in before and o.type != "MESH":
        o.select_set(False)
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
ob.parent = None
ob.matrix_world = ob.matrix_world.copy()
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
yaw = SPEC.get("yaw_deg", 0)
if yaw == "auto":
    # the generator's front is not reliable for three-quarter drawings (a fireplace came out facing +X). The front
    # axis: the one whose extent ratio matches the spec's width:depth; the front side of it: the less flat end
    # (the back of furniture and architecture is a flat panel or a wall face)
    import numpy as np
    P = np.array([v.co[:] for v in ob.data.vertices])
    lo_, hi_ = P.min(0), P.max(0)
    ext = hi_ - lo_
    sw, sd = SPEC["size_m"][0], SPEC["size_m"][1]
    swap = (sw > sd * 1.15 and ext[0] < ext[1] * 0.87) or (sd > sw * 1.15 and ext[1] < ext[0] * 0.87)
    ax = 0 if swap else 1

    def flat(end):
        lim = (hi_[ax] - 0.02 * ext[ax]) if end > 0 else (lo_[ax] + 0.02 * ext[ax])
        return float(((P[:, ax] >= lim) if end > 0 else (P[:, ax] <= lim)).mean())
    front_end = 1 if flat(1) < flat(-1) else -1
    if abs(flat(1) - flat(-1)) < 0.004 and not swap:
        front_end = -1                     # no clear back: keep the generator's front (-Y)
    yaw = {(1, 1): -90, (1, -1): 90, (0, 1): 180, (0, -1): 0}[(1 if swap else 0, front_end)] if True else 0
    if ax == 1:
        yaw = 180 if front_end > 0 else 0
    print(f"[img3d model] auto yaw: front axis {'X' if swap else 'Y'}, front end {front_end:+d}, flat "
          f"+{flat(1):.3f} -{flat(-1):.3f} -> yaw {yaw}")
yaw = float(yaw)
ob.data.transform(Matrix.Rotation(math.radians(yaw), 4, "Z"))
zs = [v.co.z for v in ob.data.vertices]
h = max(zs) - min(zs)
want_h = float(SPEC.get("height_m") or SPEC["size_m"][2])
if SPEC.get("longest_m"):               # a thing sized by its longest side, whichever way up it came
    xs_, ys_ = [v.co.x for v in ob.data.vertices], [v.co.y for v in ob.data.vertices]
    want_h = float(SPEC["longest_m"]) * h / max(h, max(xs_) - min(xs_), max(ys_) - min(ys_))
ob.data.transform(Matrix.Scale(want_h / h, 4))
ob.data.update()
# a cabin people sit inside is opened where they look out: the shape model closes a canopy with a solid skin, and from
# the seats nothing outside could be seen; the faces inside the spec's "sight_cut_m" box (asset frame, metres) go
cut = SPEC.get("sight_cut_m")
if cut:
    import bmesh as _bm
    b_ = _bm.new()
    b_.from_mesh(ob.data)
    # (the box is in the finished asset's frame: the middle of its footprint at the origin, standing on z = 0)
    cs = [v.co for v in b_.verts]
    off = [(min(c[0] for c in cs) + max(c[0] for c in cs)) / 2, (min(c[1] for c in cs) + max(c[1] for c in cs)) / 2,
           min(c[2] for c in cs)]
    gone = [f for f in b_.faces if all(cut[k][0] <= f.calc_center_median()[i] - off[i] <= cut[k][1]
                                       for i, k in enumerate("xyz"))]
    _bm.ops.delete(b_, geom=gone, context="FACES")
    b_.to_mesh(ob.data)
    b_.free()
    print(f"[img3d model] sight cut: {len(gone)} faces removed in {cut}")
# triangle budget
limit = int(SPEC.get("max_tris") or TIER_TRIS[SPEC.get("tier", "hero")] * 0.98)
tris = sum(len(p.vertices) - 2 for p in ob.data.polygons)
t_dec = 0.0
if tris > limit:
    t = time.time()
    md = ob.modifiers.new("dec", "DECIMATE")
    md.ratio = limit / tris
    md.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=md.name)
    t_dec = time.time() - t
import bmesh  # noqa: E402

bm = bmesh.new()
bm.from_mesh(ob.data)
# specks: a piece of the shape with almost none of its points, joined to nothing (the generator leaves a few; one of
# four points beside a helmet failed the whole object as a floating part)
bm.verts.ensure_lookup_table()
seen_, isl_ = set(), []
for v0 in bm.verts:
    if v0.index in seen_:
        continue
    stack, grp = [v0], []
    seen_.add(v0.index)
    while stack:
        v_ = stack.pop()
        grp.append(v_)
        for e_ in v_.link_edges:
            o_ = e_.other_vert(v_)
            if o_.index not in seen_:
                seen_.add(o_.index)
                stack.append(o_)
    isl_.append(grp)
if len(isl_) > 1:
    # (the shape comes in many pieces, one for each patch of its texture: a speck is a piece of a few points
    # whose box touches no other piece's, the harness's own measure of a floating part)
    import numpy as _np
    bx_ = [(_np.min([v_.co[:] for v_ in g_], 0), _np.max([v_.co[:] for v_ in g_], 0)) for g_ in isl_]
    eps_ = 0.002                                        # (2 mm: the mesh is at its finished size here)
    gone_ = [v_ for i_, g_ in enumerate(isl_) if len(g_) <= SPECK_POINTS and not any(
        j_ != i_ and _np.all(bx_[i_][0] <= bx_[j_][1] + eps_) and _np.all(bx_[j_][0] <= bx_[i_][1] + eps_)
        for j_ in range(len(isl_))) for v_ in g_]
    if gone_:
        bmesh.ops.delete(bm, geom=gone_, context="VERTS")
        print(f"[img3d model] {len(gone_)} points of specks removed ({len(isl_)} pieces in all)")
loose = [e for e in bm.edges if not e.link_faces]
bmesh.ops.delete(bm, geom=loose, context="EDGES")
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
if not SPEC.get("seen_from_inside"):                    # (a shell made to face inward keeps its own winding)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)   # the collapse can leave a face wound against its neighbours
bm.to_mesh(ob.data)
bm.free()
print(f"[img3d model] tris {tris} -> {sum(len(p.vertices) - 2 for p in ob.data.polygons)} (decimate {t_dec:.1f}s on "
      f"CPU); height {h:.3f} -> {want_h} m; yaw {yaw}")
# material: the port's colour, plus derived bump and roughness
src = None
for m in ob.data.materials:
    if m and m.use_nodes:
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None:
                src = n.image
                break
    if src:
        break
if src is None:
    raise RuntimeError("the .glb has no base-colour image")
m = bpy.data.materials.new(f"img3d_{aid}")
m.use_nodes = True
nt = m.node_tree
pr = nt.nodes["Principled BSDF"]
uv = nt.nodes.new("ShaderNodeUVMap")
uv.uv_map = ob.data.uv_layers[0].name
tx = nt.nodes.new("ShaderNodeTexImage")
tx.image = src
tx.image.colorspace_settings.name = "sRGB"
tx.interpolation = "Cubic"
nt.links.new(uv.outputs["UV"], tx.inputs["Vector"])
nt.links.new(tx.outputs["Color"], pr.inputs["Base Color"])
lum = nt.nodes.new("ShaderNodeRGBToBW")
nt.links.new(tx.outputs["Color"], lum.inputs["Color"])
lo, hi = SPEC.get("rough", [0.45, 0.8])
mr = nt.nodes.new("ShaderNodeMapRange")
mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.05, 0.6
mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = hi, lo
nt.links.new(lum.outputs["Val"], mr.inputs["Value"])
nt.links.new(mr.outputs["Result"], pr.inputs["Roughness"])
bump_m = float(SPEC.get("bump_m", 0.002 * want_h))
if bump_m > 0:
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Distance"].default_value = bump_m
    bp.inputs["Strength"].default_value = 1.0
    nt.links.new(lum.outputs["Val"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], pr.inputs["Normal"])
pr.inputs["Specular IOR Level"].default_value = 0.4
pr.inputs["Metallic"].default_value = 0.0
if SPEC.get("emit_texture"):            # a place's land and sky that give their own picture's light (img3d/shell_one.py)
    nt.links.new(tx.outputs["Color"], pr.inputs["Emission Color"])
    pr.inputs["Emission Strength"].default_value = float(SPEC["emit_texture"])
ob.data.materials.clear()
ob.data.materials.append(m)
for p in ob.data.polygons:
    p.material_index = 0
# regions (as models/from_views.py): faces whose texture colour (hsv) and place (box, 0..1 of the bounds) match get a
# real material instead: "glass" (transmissive; the harness exports it as glass), "emit" (a flame, coals: emissive in
# the region's colour x strength). The generator makes glass and flames solid and unlit; this gives them back.
regs = SPEC.get("regions") or []
if regs:
    import colorsys
    import numpy as np
    W_, H_ = src.size
    pxa = np.empty(W_ * H_ * 4, np.float32)
    src.pixels.foreach_get(pxa)
    pxa = pxa.reshape(H_, W_, 4)
    me = ob.data
    uvl = me.uv_layers.active.data
    co = np.array([v.co[:] for v in me.vertices])
    lo, hi = co.min(0), co.max(0)
    for ri, r in enumerate(regs):
        mat = bpy.data.materials.new(f"{r.get('name', 'region')}_{aid}")
        mat.use_nodes = True
        bs = mat.node_tree.nodes["Principled BSDF"]
        c = r.get("color", [1.0, 0.8, 0.5])
        bs.inputs["Base Color"].default_value = (*c, 1.0)
        if r["material"] == "glass":
            bs.inputs["Transmission Weight"].default_value = 1.0
            bs.inputs["Roughness"].default_value = r.get("rough", 0.05)
            bs.inputs["IOR"].default_value = r.get("ior", 1.5)
        else:
            bs.inputs["Emission Color"].default_value = (*r.get("glow", c), 1.0)
            bs.inputs["Emission Strength"].default_value = r.get("strength", 8.0)
        ob.data.materials.append(mat)
        k = len(ob.data.materials) - 1
        n = 0
        for p in me.polygons:
            u = np.mean([uvl[li].uv[:] for li in p.loop_indices], axis=0)
            rgb = pxa[min(H_ - 1, int(u[1] * H_)), min(W_ - 1, int(u[0] * W_)), :3]
            h, s_, v = colorsys.rgb_to_hsv(*rgb)
            ok = True
            if "hsv" in r:
                (h0, h1), (s0, s1), (v0, v1) = r["hsv"]
                ok = (h0 <= h <= h1 or (h0 > h1 and (h >= h0 or h <= h1))) and s0 <= s_ <= s1 and v0 <= v <= v1
            if ok and "box" in r:
                cc = (np.array(p.center[:]) - lo) / np.maximum(hi - lo, 1e-6)
                b = r["box"]
                ok = b[0] <= cc[0] <= b[1] and b[2] <= cc[1] <= b[3] and b[4] <= cc[2] <= b[5]
            if ok:
                p.material_index = k
                n += 1
        print(f"[img3d model] region {r.get('name')}: {n} faces -> {r['material']}")
ob.name = f"img3d_{aid}"
print(f"[img3d model] done in {time.time() - t0:.1f}s")
