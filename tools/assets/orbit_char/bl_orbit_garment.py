"""Blender 4.5: turn a skinned image-to-3D figure (bl_orbit_skin.py output) into a garment shell for a rigged body.

    blender -b -t 4 --factory-startup --python bl_orbit_garment.py -- job.json

job: {"body_fbx": "...Body.fbx" (the character's body, exported from UE: skeleton + skin weights, centimetres),
      "figure_blend": ".../orbit_fig.blend", "figure": "orbit_fig", "out": dir,
      "height_m": null (default: the body's crown + 0.03 m for shoes and hair),
      "gap": 0.006, "ease_min": 1.0, "skirt": {"dist": 0.03, "span": 0.05},
      "skin": {"hue_max": 0.13, "sat_min": 0.16, "val_min": 0.22}, "head_cut_pad": 0.004,
      "min_island": 200, "export": true}

Steps (mechanics only; all CPU work here is mesh bookkeeping of a few seconds, no rendering or baking):
 1. body + skeleton in metres (FBX from UE: centimetre empty dropped, transforms applied);
 2. the figure scaled to the body's height (and widened when its chest is shallower than the body's);
 3. the body's arms posed to hang where the figure's hands are (one rotation per upper arm), and that pose
    applied as the rest pose (the garment binds arms-down, as it was drawn; UE's leader pose then drives it from
    the body's own pose);
 4. the figure's head, hair and every skin-coloured surface (face, neck, hands: read from its baked colour map at
    each vertex) cut away, small islands dropped: what is left is the clothing shell (coat, waistcoat, trousers,
    collar, cravat, shoes as one outer surface);
 5. the shell pushed at least `gap` outside the body (nearest point along its normal), lightly smoothed;
 6. weights copied from the body (nearest face, interpolated); where the shell stands off the legs (coat skirt)
    the weights blend toward the pelvis so the skirt does not split between the legs;
 7. a trimmed body (hands, wrists, neck kept: the parts the shell leaves visible);
 8. FBX in centimetres (shell + skeleton; trim + skeleton) and textures named for the character tools'
    import (T_outfit_BaseColor / Normal / RMA).
"""

import colorsys
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
JOB = json.load(open(argv[0]))
OUT = JOB["out"]
os.makedirs(OUT, exist_ok=True)
REP = {"times_s": {}, "warnings": []}
T0 = time.time()


class Timer:
    def __init__(self, k):
        self.k = k

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        REP["times_s"][self.k] = round(time.time() - self.t, 2)


def import_fbx(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path, use_anim=False, ignore_leaf_bones=False,
                             automatic_bone_orientation=False)
    return [o for o in bpy.data.objects if o not in before]


def load_body():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objs = import_fbx(JOB["body_fbx"])
    arm = next(o for o in objs if o.type == "ARMATURE")
    meshes = [o for o in objs if o.type == "MESH"]
    body = max(meshes, key=lambda o: len(o.data.vertices))
    for o in meshes:
        if o is not body:
            bpy.data.objects.remove(o)
    for o in list(bpy.data.objects):
        if o.type == "EMPTY":
            for ch in list(o.children):
                mw = ch.matrix_world.copy()
                ch.parent = None
                ch.matrix_world = mw
            bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in (arm, body):
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    body.name = "body"
    return arm, body


def load_figure():
    with bpy.data.libraries.load(JOB["figure_blend"], link=False) as (src, dst):
        dst.objects = [JOB.get("figure", "orbit_fig")]
    ob = dst.objects[0]
    bpy.context.scene.collection.objects.link(ob)
    return ob


def bone_class(n):
    n = n.lower()
    if "toe" in n or n.startswith(("foot", "ball")):
        return "foot"
    if n.startswith(("thumb", "index", "middle", "ring", "pinky", "hand")):
        return "hand"
    if n.startswith(("lowerarm", "wrist")):
        return "lowerarm"
    if n.startswith("upperarm"):
        return "upperarm"
    if n.startswith(("calf", "knee")):
        return "calf"
    if n.startswith("thigh"):
        return "thigh"
    if n.startswith(("neck", "head")):
        return "neck"
    return "torso"


def classify(ob):
    gcls = [bone_class(g.name) for g in ob.vertex_groups]
    cls = []
    for v in ob.data.vertices:
        acc = defaultdict(float)
        for g in v.groups:
            acc[gcls[g.group]] += g.weight
        cls.append(max(acc, key=acc.get) if acc else "none")
    return cls


def co_np(ob):
    a = np.zeros(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def evaluated_co(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    oe = ob.evaluated_get(dg)
    me = oe.to_mesh()
    a = np.array([v.co[:] for v in me.vertices])
    oe.to_mesh_clear()
    return a


def joint(arm, name, tail=False):
    pb = arm.pose.bones[name]
    return arm.matrix_world @ (pb.tail if tail else pb.head)


def fit_scale(fig, body, cls):
    bc = co_np(body)
    crown = bc[:, 2].max()
    # the head mesh is not in the body FBX: the crown is the body's top (neck) + the head joint's offset
    H = JOB.get("height_m") or float(JOB.get("crown_m", 1.875) + 0.03)
    fc = co_np(fig)
    s = H / (fc[:, 2].max() - fc[:, 2].min())
    fc = fc * s
    fc[:, 2] -= fc[:, 2].min()
    # chest depth: body torso vertices vs figure centre band, at 72 % of the height
    zc = 0.72 * H
    tor = np.array([c == "torso" for c in cls])
    bb = bc[tor & (np.abs(bc[:, 2] - zc) < 0.02)]
    fb = fc[(np.abs(fc[:, 2] - zc) < 0.02) & (np.abs(fc[:, 0]) < 0.08)]
    k = 1.0
    if len(bb) and len(fb):
        bd = bb[:, 1].max() - bb[:, 1].min()
        fd = fb[:, 1].max() - fb[:, 1].min()
        k = max(JOB.get("ease_min", 1.0), (bd + 2 * JOB.get("gap", 0.006) + 0.01) / max(fd, 1e-3))
        REP["chest_depth_m"] = {"body": round(float(bd), 3), "figure": round(float(fd), 3)}
    # centre the figure over the body (x, y of the torso at hip height)
    hip = bc[tor & (np.abs(bc[:, 2] - 0.55 * H) < 0.03)]
    cx, cy = (hip[:, 0].mean(), hip[:, 1].mean()) if len(hip) else (0.0, 0.0)
    fx = fc[np.abs(fc[:, 2] - 0.55 * H) < 0.03]
    fc[:, 0] = (fc[:, 0] - fx[:, 0].mean()) * k + cx
    fc[:, 1] = (fc[:, 1] - fx[:, 1].mean()) * k + cy
    fig.data.vertices.foreach_set("co", fc.ravel())
    fig.data.update()
    REP["figure_scale"] = {"height_m": round(H, 3), "s": round(float(s), 4), "xy_widen": round(float(k), 3),
                           "body_top_m": round(float(crown), 3)}


def skin_mask(fig):
    """Per vertex: True where the baked colour map reads as skin (warm hue, some saturation, not dark)."""
    me = fig.data
    img = None
    for n in me.materials[0].node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image:
            img = n.image
    w, h = img.size
    px = np.asarray(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    uv = np.zeros(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    vi = np.zeros(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", vi)
    x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
    rgb = px[y, x, :3]
    col = np.zeros((len(me.vertices), 3))
    cnt = np.zeros(len(me.vertices))
    np.add.at(col, vi, rgb)
    np.add.at(cnt, vi, 1)
    col /= np.maximum(cnt, 1)[:, None]
    p = JOB.get("skin", {})
    out = np.zeros(len(col), dtype=bool)
    for i, c in enumerate(col):
        hh, ss, vv = colorsys.rgb_to_hsv(*c)
        out[i] = (hh < p.get("hue_max", 0.13) or hh > 0.95) and ss > p.get("sat_min", 0.16) and vv > p.get("val_min", 0.22)
    return out, col


def pose_arms(arm, fig, skin):
    """Rotate each upper arm so the line shoulder -> hand points at the figure's hand (skin below the chest, out
    to the side)."""
    fc = co_np(fig)
    H = fc[:, 2].max()
    res = {}
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for side in ("l", "r"):
        sh = joint(arm, f"upperarm_{side}")
        hand = joint(arm, f"hand_{side}")
        sgn = 1 if sh.x > 0 else -1
        sel = skin & (fc[:, 2] < 0.62 * H) & (fc[:, 2] > 0.3 * H) & (fc[:, 0] * sgn > 0.08)
        if sel.sum() < 20:
            REP["warnings"].append(f"no hand skin found on side {side}")
            continue
        hp = fc[sel]
        # the wrist: the top of the hand cluster
        top = hp[hp[:, 2] > np.percentile(hp[:, 2], 80)].mean(0)
        tgt = Vector(top.tolist())
        a = (hand - sh).normalized()
        b = (tgt - sh).normalized()
        q = a.rotation_difference(b)
        pb = arm.pose.bones[f"upperarm_{side}"]
        mw = arm.matrix_world @ pb.matrix
        loc = mw.to_translation()
        R = q.to_matrix().to_4x4()
        new = Matrix.Translation(loc) @ R @ Matrix.Translation(-loc) @ mw
        pb.matrix = arm.matrix_world.inverted() @ new
        bpy.context.view_layer.update()
        res[side] = {"angle_deg": round(math.degrees(a.angle(b)), 1), "wrist_target": [round(x, 3) for x in top]}
    bpy.ops.object.mode_set(mode="OBJECT")
    REP["arm_pose"] = res


def apply_rest(arm, body):
    """The current pose becomes the rest pose for skeleton and body."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    mods = [m for m in body.modifiers if m.type == "ARMATURE"]
    for m in mods:
        bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    m = body.modifiers.new("arm", "ARMATURE")
    m.object = arm


def cut_shell(fig, skin, body_neck_z):
    """Remove head/hair (everything above the lowest face skin over the collar) and every skin-coloured vertex."""
    fc = co_np(fig)
    H = fc[:, 2].max()
    face = skin & (fc[:, 2] > 0.78 * H) & (np.abs(fc[:, 0]) < 0.12)
    cut_z = float(np.percentile(fc[face, 2], 2)) + JOB.get("head_cut_pad", 0.004) if face.sum() > 20 else body_neck_z
    drop = skin | (fc[:, 2] > cut_z)
    bm = bmesh.new()
    bm.from_mesh(fig.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if drop[v.index]], context="VERTS")
    # small islands (skin specks the colour test missed, bits of hair)
    seen, islands = set(), []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack, isl = [f], []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            isl.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        islands.append(isl)
    bm.faces.ensure_lookup_table()
    small = [f for isl in islands if len(isl) < JOB.get("min_island", 200) for f in isl]
    bmesh.ops.delete(bm, geom=small, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(fig.data)
    bm.free()
    fig.data.update()
    REP["cut"] = {"head_cut_z": round(cut_z, 3), "skin_verts": int(skin.sum()), "dropped": int(drop.sum()),
                  "islands": len(islands), "small_island_faces": len(small), "verts_left": len(fig.data.vertices)}


def collider(objs):
    verts, polys = [], []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        off = len(verts)
        mw = o.matrix_world
        verts += [mw @ v.co for v in me.vertices]
        polys += [[off + i for i in p.vertices] for p in me.polygons]
        oe.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys, all_triangles=False)


def push_out(ob, bvh, gap, max_dist=0.3):
    moved = 0
    for v in ob.data.vertices:
        hit = bvh.find_nearest(v.co, max_dist)
        if hit[0] is None:
            continue
        loc, n = hit[0], hit[1]
        d = (v.co - loc).dot(n)
        if d < gap:
            v.co = loc + n * gap if d < 0 else v.co + n * (gap - d)
            moved += 1
    ob.data.update()
    return moved


def smooth(ob, iters, factor=0.5):
    m = ob.modifiers.new("sm", "SMOOTH")
    m.iterations = iters
    m.factor = factor
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=m.name)


def transfer_weights(dst, src):
    for g in list(dst.vertex_groups):
        dst.vertex_groups.remove(g)
    for g in src.vertex_groups:
        dst.vertex_groups.new(name=g.name)
    m = dst.modifiers.new("dt", "DATA_TRANSFER")
    m.object = src
    m.use_vert_data = True
    m.data_types_verts = {"VGROUP_WEIGHTS"}
    m.vert_mapping = "POLYINTERP_NEAREST"
    m.layers_vgroup_select_src = "ALL"
    m.layers_vgroup_select_dst = "NAME"
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.modifier_apply(modifier=m.name)


def skirt_blend(ob, bvh, arm):
    """Coat skirt: shell vertices standing off the legs between knee and hip take pelvis weight."""
    p = JOB.get("skirt", {})
    dist0, span = p.get("dist", 0.03), p.get("span", 0.05)
    hip_z = joint(arm, "pelvis").z
    knee_z = (joint(arm, "calf_l").z + joint(arm, "calf_r").z) / 2
    pel = ob.vertex_groups.get("pelvis")
    if pel is None:
        REP["warnings"].append("no pelvis group")
        return
    gi = {g.index: g for g in ob.vertex_groups}
    n = 0
    for v in ob.data.vertices:
        if not (knee_z - 0.05 < v.co.z < hip_z):
            continue
        hit = bvh.find_nearest(v.co, 0.5)
        if hit[0] is None:
            continue
        d = (v.co - hit[0]).length
        a = min(1.0, max(0.0, (d - dist0) / span))
        if a <= 0:
            continue
        a *= min(1.0, (hip_z - v.co.z) / 0.08)
        for g in v.groups:
            gi[g.group].add([v.index], g.weight * (1 - a), "REPLACE")
        pel.add([v.index], a, "ADD")
        n += 1
    REP["skirt_blend_verts"] = n


def body_trim(body, arm, cls):
    co = co_np(body)
    keep = np.array([c in ("hand", "neck") for c in cls])
    for s in ("l", "r"):
        h = np.array(joint(arm, f"hand_{s}")[:])
        e = np.array(joint(arm, f"lowerarm_{s}")[:])
        ax = (h - e) / np.linalg.norm(h - e)
        d = (co - h)
        keep |= (np.array([c == "lowerarm" for c in cls]) & (d @ ax > -0.04) & (np.linalg.norm(d, axis=1) < 0.12))
    neck_z = joint(arm, "neck_01").z
    keep |= np.array([c == "torso" for c in cls]) & (co[:, 2] > neck_z - 0.01)
    me = body.data.copy()
    ob = bpy.data.objects.new("body_trim", me)
    bpy.context.scene.collection.objects.link(ob)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep[v.index]], context="VERTS")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    ob.parent = arm
    ob.modifiers.new("arm", "ARMATURE").object = arm
    return ob


def to_centimetres(objs, arm):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs + [arm]:
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    arm.scale = (100, 100, 100)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def export_fbx(objs, arm, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"ARMATURE", "MESH"},
                             apply_unit_scale=False, global_scale=1.0, apply_scale_options="FBX_SCALE_NONE",
                             axis_forward="-Z", axis_up="Y", add_leaf_bones=False, use_armature_deform_only=False,
                             primary_bone_axis="Y", secondary_bone_axis="X", armature_nodetype="NULL",
                             bake_anim=False, mesh_smooth_type="FACE", use_tspace=True, path_mode="STRIP")


def textures(fig):
    tdir = os.path.join(OUT, "textures")
    os.makedirs(tdir, exist_ok=True)
    img = None
    for n in fig.data.materials[0].node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image:
            img = n.image
    shutil.copy(bpy.path.abspath(img.filepath), os.path.join(tdir, "T_outfit_BaseColor.png"))
    for name, rgb, cs in (("Normal", (0.5, 0.5, 1.0), "Non-Color"),
                          ("RMA", (JOB.get("roughness", 0.8), 0.0, 1.0), "Non-Color")):
        im = bpy.data.images.new(f"T_outfit_{name}", 8, 8, alpha=False)
        im.colorspace_settings.name = cs
        im.pixels[:] = (list(rgb) + [1.0]) * 64
        im.filepath_raw = os.path.join(tdir, f"T_outfit_{name}.png")
        im.file_format = "PNG"
        im.save()
    return tdir


def main():
    with Timer("load"):
        arm, body = load_body()
        fig = load_figure()
        cls = classify(body)
    with Timer("scale"):
        fit_scale(fig, body, cls)
    with Timer("skin_mask"):
        skin, col = skin_mask(fig)
    with Timer("pose_arms"):
        pose_arms(arm, fig, skin)
        apply_rest(arm, body)
    with Timer("cut"):
        cut_shell(fig, skin, joint(arm, "neck_02").z)
    with Timer("fit"):
        bvh = collider([body])
        m1 = push_out(fig, bvh, JOB.get("gap", 0.006))
        smooth(fig, 2, 0.4)
        m2 = push_out(fig, bvh, JOB.get("gap", 0.006))
        REP["pushed"] = [m1, m2]
    with Timer("weights"):
        transfer_weights(fig, body)
        skirt_blend(fig, bvh, arm)
        bpy.context.view_layer.objects.active = fig
        bpy.ops.object.select_all(action="DESELECT")
        fig.select_set(True)
        bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=8)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
        fig.parent = arm
        fig.modifiers.new("arm", "ARMATURE").object = arm
        fig.name = "outfit"
    with Timer("trim"):
        trim = body_trim(body, arm, cls)
    tdir = textures(fig)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "garment.blend"))
    if JOB.get("export", True):
        with Timer("export"):
            to_centimetres([fig, trim, body], arm)
            export_fbx([fig], arm, os.path.join(OUT, "outfit.fbx"))
            export_fbx([trim], arm, os.path.join(OUT, "body_trim.fbx"))
    REP["textures"] = tdir
    REP["total_s"] = round(time.time() - T0, 1)
    json.dump(REP, open(os.path.join(OUT, "garment_report.json"), "w"), indent=1)
    print("GARMENT_REPORT " + json.dumps(REP))


main()
