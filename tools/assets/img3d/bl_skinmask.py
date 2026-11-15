"""The head mesh's skin that the fitted garment covers, as an opacity mask in the head mesh's UVs (called from
bl_combo.py with its scene). In the engine the character's face gets a copy of its skin material that is masked by
this texture (characters/char_skinmask.py), so skin under cloth is hidden rather than pushed against or lined.

A head vertex is hidden when it lies below the garment's top round the neck at its angle (per 15-degree sector:
the highest cloth within 13 cm of the neck's axis, less MARGIN) AND cloth covers it: its outward ray meets the
garment within REACH, or the garment is within NEAR of it, or (skin in a gap of the cloth, as between a bodice and
a puffed sleeve) rays in at least COVER_FRAC of 17 directions round its normal meet the garment within 15 cm, or it
is a shoulder (out to the side beyond the neck's radius plus 3 cm: skin poking out through a lower cloth shoulder
has no cloth over it to find). Round the neck a hidden vertex must also have cloth outside it MARGIN higher up (the
collar's own edge, not its highest point). Skin in an open neckline (no cloth over it, in front of the neck) stays. The per-vertex values are rasterised into the UV map by a Cycles
emission bake on the GPU, then the hidden region is grown by GROW_PX texels (so no seam of skin shows at its edge).
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SKIN_SLOT_KEY = "head"     # the MetaHuman face mesh's skin section (engine slot head_shader_shader; its export
#                            names the material M_GrayTexture_Head)


def build(HEAD, fig, J, JOB, REP, out_png):
    C = JOB.get("skin_mask") or {}
    margin = float(C.get("margin_m", 0.012))
    reach = float(C.get("reach_m", 0.12))
    near = float(C.get("near_m", 0.03))
    res = int(C.get("res", 1024))
    grow = int(C.get("grow_px", 3))
    nk = np.array(J("neck_01"))
    H = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
    Hn = np.array([(HEAD.matrix_world.to_3x3() @ v.normal).normalized() for v in HEAD.data.vertices])
    P = np.array([fig.matrix_world @ v.co for v in fig.data.vertices])
    rr = np.linalg.norm(P[:, :2] - nk[:2], axis=1)
    sec = ((np.arctan2(P[:, 1] - nk[1], P[:, 0] - nk[0]) + math.pi) / (2 * math.pi) * 24).astype(int) % 24
    top = np.full(24, -9.0)
    for k in range(24):
        m = (sec == k) & (rr < 0.13) & (P[:, 2] < float(J("head").z))
        if m.any():
            top[k] = float(P[m, 2].max())
    hs = ((np.arctan2(H[:, 1] - nk[1], H[:, 0] - nk[0]) + math.pi) / (2 * math.pi) * 24).astype(int) % 24
    below = H[:, 2] < top[hs] - margin
    bvh = BVHTree.FromObject(fig, bpy.context.evaluated_depsgraph_get())
    hide = np.zeros(len(H), bool)
    # directions round each vertex's normal (the normal, 8 at 45 degrees, 8 at 75): skin in a gap of the cloth (between
    # a bodice and a puffed sleeve) has cloth over it in most of them, skin in an open neckline in few
    cover_min = float(C.get("cover_frac", 0.5))
    reach_c = float(C.get("cover_reach_m", 0.15))
    dirs_local = [np.array([0.0, 0.0, 1.0])]
    for tilt in (45.0, 75.0):
        for k in range(8):
            a, t_ = 2 * math.pi * k / 8, math.radians(tilt)
            dirs_local.append(np.array([math.sin(t_) * math.cos(a), math.sin(t_) * math.sin(a), math.cos(t_)]))
    covered = np.zeros(len(H))
    # the shoulders: below the collar line and out to the side beyond the neck, skin is always under the garment's
    # shoulder, so it is hidden even where it pokes out through the cloth (no cloth "over" it to find)
    ring = np.abs(H[:, 2] - nk[2]) < 0.015
    neck_r = float(np.median(np.linalg.norm(H[ring, :2] - nk[:2], axis=1))) if ring.sum() > 20 else 0.06
    side_x = neck_r + float(C.get("shoulder_beyond_neck_m", 0.03))
    shoulder = below & (np.abs(H[:, 0] - nk[0]) > side_x)
    hide |= shoulder
    for i in np.nonzero(below & ~shoulder)[0]:
        hit = bvh.ray_cast(Vector(H[i] + Hn[i] * 0.001), Vector(Hn[i]), reach)
        if hit[0] is not None:
            hide[i] = True
            continue
        loc, nrm, _, d = bvh.find_nearest(Vector(H[i]), near)
        if loc is not None:
            hide[i] = True
            continue
        n_ = Hn[i]
        t1 = np.cross(n_, [0.0, 0.0, 1.0] if abs(n_[2]) < 0.9 else [1.0, 0.0, 0.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n_, t1)
        hits = 0
        for dl in dirs_local:
            dv = dl[0] * t1 + dl[1] * t2 + dl[2] * n_
            if bvh.ray_cast(Vector(H[i] + n_ * 0.001), Vector(dv), reach_c)[0] is not None:
                hits += 1
        covered[i] = hits / len(dirs_local)
        hide[i] = covered[i] >= cover_min
    # round the neck the collar's own edge decides, not its highest point at that angle: a neck vertex stays hidden
    # only when the cloth still stands outside it MARGIN higher up (a collar dipping in front, or standing off the
    # neck, left a notch of hidden skin above its edge that showed as a hole)
    neck_zone = hide & ~shoulder & (np.linalg.norm(H[:, :2] - nk[:2], axis=1) < side_x)
    kept_edge = 0
    # a collar standing off the neck shows the neck down into the gap as deep as the gap is wide (seen from 45 degrees
    # above, background showed through where that skin was hidden, 11-12): the test height is the larger of MARGIN
    # and the gap to the cloth outside the vertex
    for i in np.nonzero(neck_zone)[0]:
        d = np.array([H[i][0] - nk[0], H[i][1] - nk[1], 0.0])
        if np.linalg.norm(d) < 1e-4:
            continue
        d /= np.linalg.norm(d)
        h0 = bvh.ray_cast(Vector(H[i] + d * 0.001), Vector(d), float(C.get("collar_reach_m", 0.10)))
        gap = float(h0[3]) if h0[0] is not None else 0.0
        o = H[i] + np.array([0.0, 0.0, max(margin, gap)])
        if bvh.ray_cast(Vector(o), Vector(d), float(C.get("collar_reach_m", 0.10)))[0] is None:
            hide[i] = False
            kept_edge += 1
    if C.get("debug_npz"):
        np.savez_compressed(C["debug_npz"], head=H, hide=hide, below=below, top=top, neck=nk,
                            uv=np.array([[0, 0]]) if not HEAD.data.uv_layers else
                            np.array([HEAD.data.uv_layers[0].data[li].uv[:] for li in range(len(HEAD.data.loops))]),
                            loop_v=np.array([lp.vertex_index for lp in HEAD.data.loops]),
                            cloth=P[np.random.default_rng(0).choice(len(P), min(len(P), 40000), replace=False)])
    REP["skin_mask"] = {"hidden_verts": int(hide.sum()), "below_collar": int(below.sum()), "res": res,
                        "by_cover": int((covered >= cover_min).sum()), "by_shoulder": int(shoulder.sum()),
                        "neck_r_m": round(neck_r, 3), "kept_at_collar_edge": kept_edge}
    # bake: an emission of the vertex value onto an image through UV map 0, on a copy of the head mesh holding only
    # its skin section (the eyes, teeth, lashes and eye shells share the UV square and would paint over the skin)
    names = [(m.name if m else "") for m in HEAD.data.materials]
    skin_slots = [k for k, n in enumerate(names) if SKIN_SLOT_KEY in n.lower()]
    if not skin_slots:
        raise SystemExit(f"skin mask: no skin section among the head mesh's materials {names}")
    ob = HEAD.copy()
    ob.data = HEAD.data.copy()
    bpy.context.scene.collection.objects.link(ob)
    me = ob.data
    idx = np.zeros(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", idx)
    at = me.color_attributes.new("keep", "FLOAT_COLOR", "POINT")
    keep = (~hide).astype(np.float32)
    at.data.foreach_set("color", np.c_[keep, keep, keep, np.ones(len(keep), np.float32)].ravel())
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if idx[f.index] not in skin_slots], context="FACES")
    bm.to_mesh(me)
    bm.free()
    img = bpy.data.images.new("T_head_mask", res, res, alpha=False, float_buffer=False)
    img.generated_color = (1.0, 1.0, 1.0, 1.0)    # texels no face covers stay "kept": no dark seam bleeds in at mips
    mat = bpy.data.materials.new("bakemask")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    a_ = nt.nodes.new("ShaderNodeAttribute")
    a_.attribute_name = "keep"
    em = nt.nodes.new("ShaderNodeEmission")
    o_ = nt.nodes.new("ShaderNodeOutputMaterial")
    tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = img
    nt.links.new(a_.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], o_.inputs["Surface"])
    nt.nodes.active = tx
    me.materials.clear()
    me.materials.append(mat)
    me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), np.int32))
    if me.uv_layers:
        me.uv_layers.active = me.uv_layers[0]
        me.uv_layers[0].active_render = True
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    pr = bpy.context.preferences.addons["cycles"].preferences
    pr.compute_device_type = "HIP"
    pr.get_devices()
    bus = str(JOB.get("bake_bus", os.environ.get("ASSETS_IGPU_BUS", "c4")))    # the fit runs on the iGPU station
    hit = False
    for d in pr.devices:
        d.use = d.type == "HIP" and d.id.endswith(f"0000:{bus}:00")
        hit = hit or d.use
    if not hit:
        raise SystemExit(f"skin mask: no HIP device on bus {bus}")
    sc.cycles.device = "GPU"
    sc.cycles.samples = 1
    bpy.ops.object.select_all(action="DESELECT")
    ob.hide_render = False
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.bake(type="EMIT", margin=4, use_clear=False)     # texels outside the skin keep the white
    px = np.array(img.pixels[:], np.float32).reshape(res, res, 4)[..., 0]
    # grow the hidden (dark) region by `grow` texels: the edge of the skin under the collar line never shows
    hid = px < 0.5
    for _ in range(grow):
        g = hid.copy()
        g[1:] |= hid[:-1]
        g[:-1] |= hid[1:]
        g[:, 1:] |= hid[:, :-1]
        g[:, :-1] |= hid[:, 1:]
        hid = g
    outp = np.where(hid, 0.0, 1.0).astype(np.float32)
    img.pixels.foreach_set(np.dstack([outp, outp, outp, np.ones_like(outp)]).ravel())
    img.filepath_raw = out_png
    img.file_format = "PNG"
    img.save()
    REP["skin_mask"]["hidden_texel_frac"] = round(float(hid.mean()), 4)
    REP["skin_mask"]["skin_slots"] = [names[k] for k in skin_slots]
    bpy.data.objects.remove(ob)
    return out_png
