"""Blender side of the 3D-model stage (tools/blockout_model.py runs it; not for direct use).

    blender -b --factory-startup --python tools/blockout_model_bl.py -- JOB.json

JOB {"op": "prep", "mesh": in.glb, "height": 1.8, "front": "+z"|..., "out_blend": x.blend,
     "front_png": front.png, "size": 1024, "texture": {"image": cut.png} (optional)}
    Imports the image-to-3D mesh, turns it Z-up facing +X with its feet at z = 0 and the given height,
    centres it, decimates it, optionally paints vertex colours by projecting the source picture from the
    front, saves the .blend and renders an orthographic front view (for the 2D pose detector) and
    writes front.json (the ortho camera's scale) next to it.
JOB {"op": "rig", "blend": x.blend, "kps": [[u, v] | null] * 18 (pixels in front.png), "out_blend": y.blend,
     "out_json": rig.json}
    Lifts the detected 2D joints to 3D (depth = middle of the mesh along the view ray), builds a named
    armature (the joint names of tools/blockout_rig.py) and skins the mesh to it with Blender's bone-heat
    automatic weights (falling back to envelope weights), and saves.
"""

import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
job = json.load(open(argv[0]))

FRONT_ROT = {   # rotate the importer's axes so the model faces +X and stands on +Z
    "+z": Matrix.Rotation(math.pi / 2, 4, "Z"),         # glTF import: model already Z-up (Y-up -> Z-up), front -Y
}


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mesh_objects():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def import_mesh(path):
    if path.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    elif path.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    obs = mesh_objects()
    # join into one object
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    ob.name = "model"
    return ob


def normalise(ob, height, yaw_deg):
    me = ob.data
    me.transform(Matrix.Rotation(math.radians(yaw_deg), 4, "Z"))
    vs = np.array([v.co[:] for v in me.vertices])
    lo, hi = vs.min(0), vs.max(0)
    s = height / float(hi[2] - lo[2])
    c = (lo + hi) / 2
    me.transform(Matrix.Scale(s, 4) @ Matrix.Translation(Vector((-c[0], -c[1], -lo[2]))))
    me.update()


def decimate(ob, faces):
    n = len(ob.data.polygons)
    if n <= faces:
        return
    m = ob.modifiers.new("dec", "DECIMATE")
    m.ratio = faces / n
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=m.name)


def ortho_front(ob, png, size):
    """Orthographic camera on +X looking at the model (which faces +X)."""
    vs = np.array([v.co[:] for v in ob.data.vertices])
    lo, hi = vs.min(0), vs.max(0)
    span = float(max(hi[1] - lo[1], hi[2] - lo[2])) * 1.1
    cd = bpy.data.cameras.new("front")
    cd.type = "ORTHO"
    cd.ortho_scale = span
    cam = bpy.data.objects.new("front", cd)
    cy, cz = float((lo[1] + hi[1]) / 2), float((lo[2] + hi[2]) / 2)
    cam.location = (float(hi[0]) + 5.0, cy, cz)
    cam.rotation_euler = (math.pi / 2, 0, math.pi / 2)          # looks along -X, Z up
    bpy.context.scene.collection.objects.link(cam)
    sc = bpy.context.scene
    sc.camera = cam
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3
    sun.rotation_euler = (math.radians(60), 0, math.radians(110))
    sc.collection.objects.link(sun)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 16
    sc.cycles.use_denoising = True
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.filepath = png
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    sc.world = w
    sc.view_settings.view_transform = "Standard"
    bpy.ops.render.render(write_still=True)
    meta = {"cy": cy, "cz": cz, "span": span, "size": size}
    with open(os.path.splitext(png)[0] + ".json", "w") as f:
        json.dump(meta, f)
    bpy.data.objects.remove(cam)
    bpy.data.objects.remove(sun)
    return meta


def _srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


HEAD_FRAC = 0.13                  # the head band: the top 13 % of a standing figure (crown to chin, ~1/7.5 of the height)


def _bbox_uv(ob, image_path, back=False, view=None):
    """Per-vertex (u, v) into a front (or back) picture: the figure's bbox in the picture mapped onto the mesh's
    bbox (front: screen right = +Y, the model's left; back: screen right = -Y); also returns the picture's
    pixels (as stored: sRGB-encoded) and size."""
    img = bpy.data.images.load(image_path)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, -1)[::-1]   # top row first
    rgb = px[..., :3]
    if px.shape[2] == 4 and px[..., 3].min() < 0.99:
        fg = px[..., 3] > 0.5
    else:   # a cut-out on a plain card: foreground = differs from the corner colour
        bg = rgb[:8, :8].reshape(-1, 3).mean(0)
        fg = np.abs(rgb - bg).sum(-1) > 0.08
    ys, xs = np.nonzero(fg)
    u0, u1, v0, v1 = xs.min(), xs.max(), ys.min(), ys.max()
    me = ob.data
    vs = np.array([v.co[:] for v in me.vertices])
    lo, hi = vs.min(0), vs.max(0)
    # model faces +X; the ortho camera on +X looking -X has screen right = +Y (the model's LEFT side on
    # screen right, as in any front view; checked with world_to_camera_view 10-19), screen up = +Z
    su = (vs[:, 1] - lo[1]) / max(hi[1] - lo[1], 1e-6)
    if back or view == "back":
        su = 1.0 - su
    elif view == "left":            # camera on +Y (the model's left side) looking -Y: screen right = -X (its back)
        su = (hi[0] - vs[:, 0]) / max(hi[0] - lo[0], 1e-6)
    elif view == "right":           # camera on -Y looking +Y: screen right = +X (its front)
        su = (vs[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-6)
    sv = (hi[2] - vs[:, 2]) / max(hi[2] - lo[2], 1e-6)
    pu = u0 + su * (u1 - u0)
    pv = v0 + sv * (v1 - v0)
    # the HEAD registered on its own (10-22: the whole-figure bbox put the Knight's side paint a few cm off his front
    # paint - a double face - and left Edric's back of head unpainted): the drawing's head band (the top HEAD_FRAC
    # of the figure) mapped onto the mesh's head band, horizontally, blended into the body mapping over the neck
    if job.get("head_register", True):
        Hm = max(hi[2] - lo[2], 1e-6)
        zt = hi[2] - HEAD_FRAC * Hm
        head = vs[:, 2] >= zt
        if head.sum() > 50:
            c = {"left": -vs[:, 0], "right": vs[:, 0]}.get(view, -vs[:, 1] if (back or view == "back") else vs[:, 1])
            ch = c[head]
            hc0, hc1 = np.percentile(ch, 1), np.percentile(ch, 99)
            vb = int(v0 + HEAD_FRAC * (v1 - v0))
            band = fg[v0:vb + 1]
            rows = [np.nonzero(r)[0] for r in band if r.any()]
            if len(rows) > 5 and hc1 - hc0 > 1e-3:
                hu0 = float(np.percentile([r.min() for r in rows], 10))
                hu1 = float(np.percentile([r.max() for r in rows], 90))
                pu_h = hu0 + (c - hc0) / (hc1 - hc0) * (hu1 - hu0)
                wgt = np.clip((vs[:, 2] - (zt - 0.04 * Hm)) / (0.04 * Hm), 0.0, 1.0)
                pu = wgt * pu_h + (1 - wgt) * pu
    return img, rgb, pu, pv, w, h


def _project(ob, image_path, back=False, view=None):
    img, rgb, pu, pv, w, h = _bbox_uv(ob, image_path, back, view)
    return rgb[np.clip(pv.round().astype(int), 0, h - 1), np.clip(pu.round().astype(int), 0, w - 1)]


def _bled_image(img, rgb, w, h, name):
    """The drawing with its colour BLED outward over the transparent background (iterated neighbour fill) and
    no alpha: mesh areas just outside the drawing's silhouette (a tail curled differently, a wing edge) get the
    nearest figure colour instead of transparent/black."""
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, -1)
    a = px[..., 3] > 0.5 if px.shape[2] == 4 else np.ones((h, w), bool)
    col = px[..., :3].copy()
    col[~a] = 0
    filled = a.copy()
    for _ in range(int(max(w, h) / 6)):
        if filled.all():
            break
        acc = np.zeros_like(col)
        cnt = np.zeros((h, w), np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh_f = np.roll(filled, (dy, dx), (0, 1))
            sh_c = np.roll(col, (dy, dx), (0, 1))
            acc += sh_c * sh_f[..., None]
            cnt += sh_f
        new = (~filled) & (cnt > 0)
        col[new] = acc[new] / cnt[new][:, None]
        filled |= new
    if not filled.all():
        col[~filled] = col[filled].mean(0)
    tmp = bpy.data.images.new(name, w, h, alpha=False)
    tmp.pixels.foreach_set(np.concatenate([col, np.ones((h, w, 1), np.float32)], -1).ravel())
    path = os.path.join(os.path.dirname(job["out_blend"]), f"{name}.png")
    tmp.filepath_raw = path                     # written to disk and loaded back: a generated image does not
    tmp.file_format = "PNG"                     # survive packing reliably (the dragon rendered black)
    tmp.save()
    bpy.data.images.remove(tmp)
    out = bpy.data.images.load(path)
    out.colorspace_settings.name = "sRGB"
    return out


def _uv_layer(ob, name, image_path, back=False, view=None):
    """A UV map that projects the picture onto the mesh at the picture's FULL resolution (per face corner);
    returns the BLED, alpha-free texture."""
    img, rgb, pu, pv, w, h = _bbox_uv(ob, image_path, back, view)
    me = ob.data
    uv = me.uv_layers.new(name=name)
    U = pu / w
    Vv = 1.0 - pv / h                     # Blender UV origin = bottom left
    vi = np.zeros(len(me.loops), int)
    me.loops.foreach_get("vertex_index", vi)
    uv.data.foreach_set("uv", np.stack([U[vi], Vv[vi]], 1).ravel())
    return _bled_image(img, rgb, w, h, f"{name}_tex")


VIEW_DIRS = {"front": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0), "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0)}


def _visible(ob, dirs) -> np.ndarray:
    """(n_verts, n_dirs) 1.0 where a ray from the vertex (nudged off the surface) toward the view hits nothing of the
    mesh, else 0.0 (an orthographic view from far along that direction)."""
    from mathutils.bvhtree import BVHTree
    me = ob.data
    bm_ = bmesh.new()
    bm_.from_mesh(me)
    tree = BVHTree.FromBMesh(bm_)
    bm_.free()
    co = [v.co.copy() for v in me.vertices]
    nr = [v.normal.copy() for v in me.vertices]
    out = np.ones((len(co), len(dirs)), np.float32)
    for j, d in enumerate(dirs):
        dv = Vector(tuple(float(x) for x in d))
        for i, (c, n) in enumerate(zip(co, nr)):
            if n.dot(dv) <= 0.0:
                continue                             # facing away: its weight is 0 anyway
            hit = tree.ray_cast(c + n * 0.004 + dv * 0.002, dv, 10.0)
            if hit[0] is not None:
                out[i, j] = 0.0
    return out


def paint_views(ob, views: dict):
    """FOUR (or more) drawings, each projected at full resolution through its own UV map and blended per vertex by
    how squarely the REST surface faces that view (cos^2 of the normal; 10-21: front + back only = every head's
    sides were the front picture stretched sideways -> melted talk faces). The blend factors are baked per vertex
    (posing turns the body by its bones); the shader mixes the textures in order with cumulative factors."""
    me = ob.data
    nrm = np.array([v.normal[:] for v in me.vertices])
    names = [v for v in VIEW_DIRS if views.get(v)]
    W = np.stack([np.clip(nrm @ np.array(VIEW_DIRS[v]), 0, None) ** 2 for v in names], 1)
    # a view paints only what it SEES (10-22: in the side drawings the hanging hand sits in front of the hip; projected,
    # it painted a hand on the hip, a third hand once the arm moved): a vertex hidden from a view by the mesh itself
    # (the A-pose arm in front of the hip) takes nothing from that view
    side_ = [i for i, v in enumerate(names) if v in ("left", "right")]      # (the side views: there the hanging arm
    if side_:                                                                 # covers the hip; front/back see A-pose
        W[:, side_] *= _visible(ob, [np.array(VIEW_DIRS[names[i]], float) for i in side_])   # arms held clear)
    W[:, 0] += 0.02                                     # straight up/down: the front picture
    W /= W.sum(1, keepdims=True)
    cols = [_project(ob, views[v], view=v) for v in names]
    col = sum(c * W[:, i:i + 1] for i, c in enumerate(cols))
    col = _srgb_to_linear(col)
    if not me.color_attributes.get("Col"):
        me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    me.color_attributes["Col"].data.foreach_set("color", np.concatenate([col, np.ones((len(col), 1), np.float32)], 1).ravel())
    cum = np.cumsum(W, 1)
    mat = bpy.data.materials.new("painted")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    out = None
    for i, v in enumerate(names):
        im = _uv_layer(ob, v, views[v], view=v)
        im.colorspace_settings.name = "sRGB"
        im.pack()
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = im
        t.interpolation = "Cubic"
        u = nt.nodes.new("ShaderNodeUVMap")
        u.uv_map = v
        nt.links.new(u.outputs["UV"], t.inputs["Vector"])
        if out is None:
            out = t.outputs["Color"]
            continue
        fa = me.attributes.get(f"w_{v}") or me.attributes.new(f"w_{v}", "FLOAT", "POINT")
        fa.data.foreach_set("value", (W[:, i] / np.maximum(cum[:, i], 1e-6)).astype(np.float32))
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_name = f"w_{v}"
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(at.outputs["Fac"], mix.inputs[0])
        nt.links.new(out, mix.inputs[6])
        nt.links.new(t.outputs["Color"], mix.inputs[7])
        out = mix.outputs[2]
    nt.links.new(out, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    me.materials.clear()
    me.materials.append(mat)
    fw = me.attributes.get("front_w") or me.attributes.new("front_w", "FLOAT", "POINT")   # old readers
    fw.data.foreach_set("value", W[:, 0].astype(np.float32))


def paint_from_image(ob, image_path, back_path=None):
    """The drawings ARE the texture: front (and back) pictures projected through UV maps at full resolution,
    blended by the surface normal (+X = front), un-lit albedo as drawn (sRGB image textures). Vertex colours
    (converted to LINEAR: sRGB values written as linear rendered washed out) remain as the Workbench fallback."""
    me = ob.data
    col = _project(ob, image_path)
    nx = np.array([v.normal[0] for v in me.vertices])
    wv1 = np.clip((nx + 0.5) / 0.35, 0.0, 1.0)                  # the back drawing only where clearly turned away
    # the blend weight from the REST normal, baked per vertex: posing turns the body by its bones, so any
    # render-time normal (world or object space) would flip front/back with the way the character faces
    fa = me.attributes.get("front_w") or me.attributes.new("front_w", "FLOAT", "POINT")
    fa.data.foreach_set("value", wv1.astype(np.float32) if back_path else np.ones(len(nx), np.float32))
    if back_path:
        colb = _project(ob, back_path, back=True)
        wv = wv1[:, None]
        col = col * wv + colb * (1 - wv)
    col = _srgb_to_linear(col)
    if not me.color_attributes.get("Col"):
        me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    attr = me.color_attributes["Col"]
    flat = np.concatenate([col, np.ones((len(col), 1), np.float32)], 1).ravel()
    attr.data.foreach_set("color", flat)
    imf = _uv_layer(ob, "front", image_path)
    imb = _uv_layer(ob, "back", back_path, back=True) if back_path else None
    for im in (imf, imb):
        if im is not None:
            im.colorspace_settings.name = "sRGB"
            im.pack()
    mat = bpy.data.materials.new("painted")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def tex(im, uvname):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = im
        t.interpolation = "Cubic"
        u = nt.nodes.new("ShaderNodeUVMap")
        u.uv_map = uvname
        nt.links.new(u.outputs["UV"], t.inputs["Vector"])
        return t
    tf = tex(imf, "front")
    if imb is not None:
        tb = tex(imb, "back")
        at = nt.nodes.new("ShaderNodeAttribute")             # the baked REST-normal weight (1 = front)
        at.attribute_name = "front_w"
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(at.outputs["Fac"], mix.inputs[0])
        nt.links.new(tb.outputs["Color"], mix.inputs[6])
        nt.links.new(tf.outputs["Color"], mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    else:
        nt.links.new(tf.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    me.materials.clear()
    me.materials.append(mat)


def clean_mesh(ob, min_frac=0.005, head_frac=0.2, k=3.0, iters=3):
    """Image-to-3D leftovers (10-21: stringy hair spikes and loose bits around the heads): loose pieces smaller than
    min_frac of the mesh are deleted; in the top head_frac of the figure, vertices that stand out from their
    neighbours' mean by > k x the median such offset (thin spikes) are pulled back to it - only the outliers, so the
    face is not smoothed away. -> {"removed_islands", "despiked"}"""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    seen, islands = set(), []
    for v0 in bm.verts:
        if v0.index in seen:
            continue
        stack, isl = [v0], []
        seen.add(v0.index)
        while stack:
            v = stack.pop()
            isl.append(v)
            for e in v.link_edges:
                o = e.other_vert(v)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        islands.append(isl)
    n = len(bm.verts)
    small = [isl for isl in islands if len(isl) < min_frac * n]
    for isl in small:
        bmesh.ops.delete(bm, geom=isl, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.update()
    V = np.array([v.co[:] for v in me.vertices])
    E = np.array([e.vertices[:] for e in me.edges])
    z0, z1 = V[:, 2].min(), V[:, 2].max()
    head = V[:, 2] > z1 - head_frac * (z1 - z0)
    moved = 0
    for _ in range(iters):
        acc = np.zeros_like(V)
        cnt = np.zeros(len(V))
        np.add.at(acc, E[:, 0], V[E[:, 1]])
        np.add.at(acc, E[:, 1], V[E[:, 0]])
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        mean = acc / np.maximum(cnt, 1)[:, None]
        off = np.linalg.norm(V - mean, axis=1)
        med = np.median(off[head]) if head.any() else 0.0
        out = head & (cnt > 0) & (off > k * max(med, 1e-6))
        if not out.any():
            break
        V[out] = mean[out]
        moved += int(out.sum())
    me.vertices.foreach_set("co", V.ravel())
    me.update()
    print("MESH_CLEAN", json.dumps({"removed_islands": len(small), "despiked": moved}))
    return {"removed_islands": len(small), "despiked": moved}


def op_prep():
    clear()
    ob = import_mesh(job["mesh"])
    if not job.get("keep_frame"):                     # a cleaned mesh is already in its rig's frame
        normalise(ob, float(job.get("height", 1.8)), float(job.get("yaw", 0.0)))
    decimate(ob, int(job.get("faces", 60000)))
    if not job.get("keep_frame") and not job.get("no_clean"):
        clean_mesh(ob)
    for p in ob.data.polygons:
        p.use_smooth = True
    if job.get("texture"):
        tx = job["texture"]
        if tx.get("left") or tx.get("right"):         # the side views exist: the 4-view paint
            paint_views(ob, {"front": tx["image"], "back": tx.get("back"), "left": tx.get("left"), "right": tx.get("right")})
        else:
            paint_from_image(ob, tx["image"], tx.get("back"))
    if job.get("front_png"):
        ortho_front(ob, job["front_png"], int(job.get("size", 1024)))
    bpy.ops.wm.save_as_mainfile(filepath=job["out_blend"])


BONES = [  # name, head joint, tail joint, parent
    ("pelvis", "pelvis", "spine_mid", None),
    ("spine", "spine_mid", "neck", "pelvis"),
    ("neck", "neck", "head", "spine"),
    ("head", "head", "head_top", "neck"),
    ("r_clav", "neck", "r_shoulder", "spine"), ("l_clav", "neck", "l_shoulder", "spine"),
    ("r_uarm", "r_shoulder", "r_elbow", "r_clav"), ("l_uarm", "l_shoulder", "l_elbow", "l_clav"),
    ("r_farm", "r_elbow", "r_wrist", "r_uarm"), ("l_farm", "l_elbow", "l_wrist", "l_uarm"),
    ("r_hand", "r_wrist", "r_hand", "r_farm"), ("l_hand", "l_wrist", "l_hand", "l_farm"),
    ("r_hipb", "pelvis", "r_hip", "pelvis"), ("l_hipb", "pelvis", "l_hip", "pelvis"),
    ("r_thigh", "r_hip", "r_knee", "r_hipb"), ("l_thigh", "l_hip", "l_knee", "l_hipb"),
    ("r_shin", "r_knee", "r_ankle", "r_thigh"), ("l_shin", "l_knee", "l_ankle", "l_thigh"),
    ("r_foot", "r_ankle", "r_toe", "r_shin"), ("l_foot", "l_ankle", "l_toe", "l_shin"),
]
COCO = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip",
        "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]


def depth_mid(ob, y, z):
    """Middle of the mesh along the +X view ray through (y, z); None if the ray misses."""
    deps = bpy.context.evaluated_depsgraph_get()
    hits = []
    origin = Vector((50.0, y, z))
    d = Vector((-1.0, 0, 0))
    for _ in range(12):
        ok, loc, nrm, idx, obj, mat = bpy.context.scene.ray_cast(deps, origin, d)
        if not ok:
            break
        hits.append(loc.x)
        origin = loc + d * 1e-4
    if not hits:
        return None
    return (max(hits) + min(hits)) / 2


def op_rig():
    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    ob = bpy.data.objects["model"]
    meta = json.load(open(job["front_json"]))
    S, size = meta["span"], meta["size"]
    vs = np.array([v.co[:] for v in ob.data.vertices])
    lo, hi = vs.min(0), vs.max(0)
    H = float(hi[2] - lo[2])
    J = {}
    for name, p in zip(COCO, job["kps"]):
        if p is None:
            continue
        u, v = p
        y = meta["cy"] + (u / size - 0.5) * S          # screen right = +Y (the model's left)
        z = meta["cz"] - (v / size - 0.5) * S
        x = depth_mid(ob, y, z)
        if x is None:   # the joint fell just outside the silhouette: nudge toward the body centre
            for k in range(1, 8):
                x = depth_mid(ob, y + (0 - y) * 0.08 * k, z)
                if x is not None:
                    break
        J[name] = np.array([x if x is not None else 0.0, y, z])
    if "r_hip" in J and "l_hip" in J:
        J["pelvis"] = (J["r_hip"] + J["l_hip"]) / 2 + np.array([0, 0, 0.02 * H])
    J["spine_mid"] = J["pelvis"] * 0.5 + J["neck"] * 0.5
    head_c = J.get("nose", J["neck"] + np.array([0, 0, 0.1 * H]))
    J["head"] = np.array([J["neck"][0], J["neck"][1], (J["neck"][2] + head_c[2]) / 2 + 0.02 * H])
    J["head_top"] = np.array([J["head"][0], J["head"][1], float(hi[2])])
    for s in ("r", "l"):
        fa = J[f"{s}_wrist"] - J[f"{s}_elbow"]
        J[f"{s}_hand"] = J[f"{s}_wrist"] + fa / max(np.linalg.norm(fa), 1e-6) * 0.09 * H
        a = J[f"{s}_ankle"].copy()
        a[2] = max(a[2], 0.04 * H)
        J[f"{s}_ankle"] = a
        J[f"{s}_toe"] = np.array([a[0] + 0.12 * H, a[1], 0.015 * H])
    # armature
    ad = bpy.data.armatures.new("rig")
    arm = bpy.data.objects.new("rig", ad)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for name, h, t, par in BONES:
        b = ad.edit_bones.new(name)
        b.head, b.tail = Vector(J[h]), Vector(J[t])
        if (b.tail - b.head).length < 1e-3:
            b.tail = b.head + Vector((0, 0, 0.02))
        b.roll = 0
        if par:
            b.parent = eb[par]
            b.use_connect = False
        eb[name] = b
    bpy.ops.object.mode_set(mode="OBJECT")
    for b in ("r_hipb", "l_hipb", "r_clav", "l_clav", "pelvis"):
        ad.bones[b].use_deform = b in ("r_clav", "l_clav", "pelvis")
    # skin
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    how = "bone heat"
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError:
        how = "envelope"
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    empty = sum(1 for v in ob.data.vertices if not v.groups)
    if empty > 0.2 * len(ob.data.vertices) and how == "bone heat":
        how = "envelope (bone heat left %d unweighted)" % empty
        for vg in list(ob.vertex_groups):
            ob.vertex_groups.remove(vg)
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    bpy.ops.wm.save_as_mainfile(filepath=job["out_blend"])
    with open(job["out_json"], "w") as f:
        json.dump({"joints": {k: v.tolist() for k, v in J.items()}, "skin": how,
                   "unweighted": empty, "verts": len(ob.data.vertices), "height": H}, f, indent=1)
    print("RIG", how, empty, len(ob.data.vertices))


{"prep": op_prep, "rig": op_rig}[job["op"]]()
