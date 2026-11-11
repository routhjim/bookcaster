"""Blender 4.5 (background): build an original period outfit on a MetaHuman body, skin it to the body's skeleton,
bake cloth materials on a GPU and export it for UE.

    blender -b --factory-startup --python bl_outfit.py -- job.json

job = {"body_fbx": body skeletal mesh FBX (exported from UE), "head_fbx": head FBX (collision only) | null,
       "out": folder, "outfit": outfit spec (see examples/outfit_frock.json), "device": "HIP", "hip_bus": "05",
       "bake_px": 4096, "bake_samples": 4, "preview": {"px": [640, 900], "samples": 24} | null,
       "steps": ["build", "bake", "export", "preview", "pose_check"]}

Route (fitted pieces): every fitted garment starts as the body's own faces, chosen by the skeleton region that drives
them (bone weights), smoothed so muscles and toes melt into cloth, then pushed out to a gap above everything under it
(body, head, inner layers), so nothing pokes through at rest. Loose pieces are constructed: trouser legs and sleeves
are straightened about the limb axis; the coat skirt is lofted from convex-hull cross-sections of hips and legs down
to the knee with a flare; collar, cravat, lapels, cuffs, pocket flaps and buttons are built shapes. Weights are copied
from the body (data transfer, nearest face); the skirt blends pelvis and thigh weights by depth. The body is cut to the
parts the clothes leave visible (hands, neck) and exported beside the outfit, so the engine can hide the full body.

Outputs in <out>: outfit.fbx (one skeletal mesh, one material), body_trim.fbx, textures/T_outfit_{BaseColor,Normal,
RMA}.png (R roughness, G metallic, B occlusion), outfit.blend, report.json (times, poke-through counts at rest and in
a test pose, joint heights), preview/*.png.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = JOB["out"]
SPEC = JOB.get("outfit", {})
os.makedirs(OUT, exist_ok=True)
REPORT = {"times": {}, "warnings": [], "poke": {}}
T0 = time.time()


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        REPORT["times"][self.name] = round(time.time() - self.t, 2)
        save_report()


def save_report():
    REPORT["total_s"] = round(time.time() - T0, 1)
    json.dump(REPORT, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)


def S(key, default):
    return SPEC.get(key, default)


# ------------------------------------------------------------------ import
def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_fbx(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path, use_anim=False, ignore_leaf_bones=False,
                             automatic_bone_orientation=False)
    return [o for o in bpy.data.objects if o not in before]


def load_body():
    objs = import_fbx(JOB["body_fbx"])
    arm = next(o for o in objs if o.type == "ARMATURE")
    meshes = [o for o in objs if o.type == "MESH"]
    body = max(meshes, key=lambda o: len(o.data.vertices))
    for o in meshes:
        if o is not body:
            bpy.data.objects.remove(o)
    head = None
    if JOB.get("head_fbx"):
        hobjs = import_fbx(JOB["head_fbx"])
        hm = [o for o in hobjs if o.type == "MESH"]
        head = max(hm, key=lambda o: len(o.data.vertices))
        for o in hobjs:
            if o is not head:
                bpy.data.objects.remove(o)
        # keep the head in its imported rest place, free of its own rig
        mw = head.matrix_world.copy()
        head.parent = None
        head.matrix_world = mw
        for m in list(head.modifiers):
            head.modifiers.remove(m)
        head.name = "head_ref"
    # The FBX from UE comes in centimetres: an empty (the file's top node, scale 0.01) over the armature object
    # (named after the skeleton's root joint). Drop the empty, keeping world transforms, then bake every object
    # transform into the data so the armature, the body and the head all live in metres with identity matrices.
    for o in list(bpy.data.objects):
        if o.type == "EMPTY":
            for ch in list(o.children):
                mw = ch.matrix_world.copy()
                ch.parent = None
                ch.matrix_world = mw
            bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in (arm, body, head):
        if o is not None:
            o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    body.name = "body"
    return arm, body, head


# ------------------------------------------------------------------ skeleton frame
def bone_class(n):
    n = n.lower()
    if "toe" in n or n.startswith(("foot", "ball")):
        return "foot"
    if n.startswith(("thumb", "index", "middle", "ring", "pinky", "hand")):
        return "hand"
    if n.startswith("lowerarm") or n.startswith("wrist"):
        return "lowerarm"
    if n.startswith("upperarm"):
        return "upperarm"
    if n.startswith("calf") or n.startswith("knee"):
        return "calf"
    if n.startswith("thigh"):
        return "thigh"
    if n.startswith(("neck", "head")):
        return "neck"
    return "torso"


class Frame:
    """Joint positions and the body's own axes (forward from heel to ball, up = +Z)."""

    def __init__(self, arm):
        self.arm = arm
        self.j = {b.name: arm.matrix_world @ b.head_local for b in arm.data.bones}
        f = (self.j["ball_l"] - self.j["foot_l"]) + (self.j["ball_r"] - self.j["foot_r"])
        f.z = 0
        self.fwd = f.normalized()
        self.up = Vector((0, 0, 1))
        self.lat = self.fwd.cross(self.up).normalized()
        self.center = (self.j["pelvis"].x, self.j["pelvis"].y)
        z = lambda n: self.j[n].z
        self.knee_z = (z("calf_l") + z("calf_r")) / 2
        self.ankle_z = (z("foot_l") + z("foot_r")) / 2
        self.neck_z = z("neck_01")
        self.head_z = z("head")
        self.pelvis_z = z("pelvis")
        self.waist_z = S("waist_z", None) or (z("spine_02") + z("spine_03")) / 2
        self.shoulder_z = (z("upperarm_l") + z("upperarm_r")) / 2
        REPORT["joints_z"] = {n: round(self.j[n].z, 3) for n in
                              ("pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01",
                               "head", "upperarm_l", "lowerarm_l", "hand_l", "thigh_l", "calf_l", "foot_l", "ball_l")
                              if n in self.j}
        REPORT["frame"] = {"fwd": list(map(lambda x: round(x, 3), self.fwd)), "waist_z": round(self.waist_z, 3)}

    def local(self, p):
        """(lateral, forward, z) relative to the pelvis axis."""
        d = Vector((p.x - self.center[0], p.y - self.center[1], 0))
        return d.dot(self.lat), d.dot(self.fwd), p.z

    def world(self, lat, fwd, z):
        return Vector((self.center[0], self.center[1], 0)) + self.lat * lat + self.fwd * fwd + Vector((0, 0, z))


def classify(ob):
    names = [g.name for g in ob.vertex_groups]
    gcls = [bone_class(n) for n in names]
    cls, frac = [], []
    for v in ob.data.vertices:
        acc = defaultdict(float)
        for g in v.groups:
            acc[gcls[g.group]] += g.weight
        tot = sum(acc.values()) or 1.0
        cls.append(max(acc, key=acc.get) if acc else "none")
        frac.append({k: w / tot for k, w in acc.items()})
    return cls, frac


# ------------------------------------------------------------------ mesh helpers
def new_obj(name, me):
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def body_subset(name, body, keep):
    """Copy of the body keeping only vertices where keep[i] (faces with all vertices kept), weights and UVs kept."""
    me = body.data.copy()
    ob = new_obj(name, me)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep[v.index]], context="VERTS")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    for m in list(ob.modifiers):
        ob.modifiers.remove(m)
    return ob


def bm_of(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    return bm


def smooth(ob, iters, factor=0.5, pin_boundary=False):
    bm = bm_of(ob)
    vs = [v for v in bm.verts if not (pin_boundary and v.is_boundary)]
    for _ in range(iters):
        bmesh.ops.smooth_vert(bm, verts=vs, factor=factor, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(ob.data)
    bm.free()


def collider(objs):
    """One BVH over the evaluated, world-space meshes of objs."""
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


def push_out(ob, bvh, gap, max_dist=0.25, normals_from=None):
    """Move every vertex to at least `gap` outside the collider along the collider's normal (nearest point)."""
    moved = 0
    for v in ob.data.vertices:
        p = ob.matrix_world @ v.co
        hit = bvh.find_nearest(p, max_dist)
        if hit[0] is None:
            continue
        loc, n, _, _ = hit
        d = (p - loc).dot(n)
        if d < gap:
            v.co = ob.matrix_world.inverted() @ (loc + n * gap if d < 0 else p + n * (gap - d))
            moved += 1
    ob.data.update()
    return moved


def fit_layer(ob, under, gap, rounds=3, smooth_iters=4):
    """Alternate light smoothing and push-out so the cloth bridges hollows but never dips under `gap`."""
    bvh = collider(under)
    for _ in range(rounds):
        push_out(ob, bvh, gap)
        smooth(ob, smooth_iters, 0.5)
    return push_out(ob, bvh, gap)


def straighten(ob, a, b, mix, extra=0.0, keep_fn=None):
    """Loosen a limb tube: radial distance about the axis a->b goes toward the tube's largest radius at that
    station (a fuller sleeve or trouser leg instead of a muscle copy)."""
    ax = (b - a)
    L = ax.length
    ax.normalize()
    stations = defaultdict(list)
    data = []
    for v in ob.data.vertices:
        if keep_fn and not keep_fn(v):
            continue
        p = v.co
        t = (p - a).dot(ax)
        if t < -0.03 or t > L + 0.05:
            continue
        c = a + ax * t
        r = (p - c).length
        k = int(t / 0.02)
        stations[k].append(r)
        data.append((v, t, c, r, k))
    # a round tube at each station: the mean radius there (averaged with the neighbours) plus ease
    mean = {k: sum(rs) / len(rs) for k, rs in stations.items()}
    for v, t, c, r, k in data:
        nb = [mean[j] for j in (k - 1, k, k + 1) if j in mean]
        R = sum(nb) / len(nb) + extra
        if r < 1e-6:
            continue
        nr = r + (R - r) * mix
        v.co = c + (v.co - c) * (nr / r)
    ob.data.update()


def fix_strays(ob, bvh, limit, rounds=8):
    """Vertices pushed or smoothed far from the body (spikes) go to the mean of their neighbours, repeatedly."""
    bm = bm_of(ob)
    fixed = 0
    for _ in range(rounds):
        bad = []
        for v in bm.verts:
            h = bvh.find_nearest(v.co, 2.0)
            if h[0] is None or h[3] > limit:
                bad.append(v)
        if not bad:
            break
        for v in bad:
            nb = [e.other_vert(v).co for e in v.link_edges]
            if nb:
                v.co = sum(nb, Vector()) / len(nb)
        fixed += len(bad)
    still = [v for v in bm.verts if not v.link_faces or (lambda h: h[0] is None or h[3] > limit)(
        bvh.find_nearest(v.co, 2.0))]
    if still:
        bmesh.ops.delete(bm, geom=still, context="VERTS")      # loose or unreachable: remove
    bm.to_mesh(ob.data)
    bm.free()
    return [fixed, len(still)]


def solidify(ob, t):
    m = ob.modifiers.new("solid", "SOLIDIFY")
    m.thickness = t
    m.offset = -1.0
    m.use_even_offset = False          # even offset spikes on sharp or degenerate corners
    m.use_quality_normals = True
    m.use_rim = True
    apply_mods(ob)


def apply_mods(ob):
    bpy.context.view_layer.objects.active = ob
    for m in list(ob.modifiers):
        if m.type == "ARMATURE":
            continue
        bpy.ops.object.modifier_apply(modifier=m.name)


def transfer_weights(dst, src):
    """Vertex groups by nearest interpolated face of src (the body)."""
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


def set_mat(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)


def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    objs[0].name = name
    return objs[0]


def convex_hull_2d(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def ray_hull(hull, ang):
    """Distance from the origin to a convex polygon (containing the origin) along direction ang."""
    d = (math.cos(ang), math.sin(ang))
    best = 0.0
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = (b[0] - a[0], b[1] - a[1])
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-9:
            continue
        t = (a[0] * e[1] - a[1] * e[0]) / den
        s = (a[0] * d[1] - a[1] * d[0]) / den
        if t > 0 and -1e-6 <= s <= 1 + 1e-6:
            best = max(best, t)
    return best


# ------------------------------------------------------------------ materials (procedural; baked later)
def principled(name, color, rough, sheen=0.0, coat=0.0, spec=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Sheen Weight"].default_value = sheen
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Specular IOR Level"].default_value = spec
    return m, nt, p


def weave(nt, p, scale, kind, strength, color=None, vary=0.0):
    """Cloth surface from object-space procedural textures: twill (diagonal wale) for wool, plain weave for cotton,
    soft folds for silk; a little colour variation (heather) for wool."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    if kind == "twill":
        mp.inputs["Rotation"].default_value = (0, math.radians(45), math.radians(45))
        w = nt.nodes.new("ShaderNodeTexWave")
        w.wave_type = "BANDS"
        w.inputs["Scale"].default_value = 1.0
        w.inputs["Distortion"].default_value = 0.6
        w.inputs["Detail"].default_value = 1.0
        nt.links.new(mp.outputs["Vector"], w.inputs["Vector"])
        h = w.outputs["Fac"]
    elif kind == "plain":
        c = nt.nodes.new("ShaderNodeTexChecker")
        c.inputs["Scale"].default_value = 1.0
        nt.links.new(mp.outputs["Vector"], c.inputs["Vector"])
        h = c.outputs["Fac"]
    else:
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 1.0
        n.inputs["Detail"].default_value = 2.0
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        h = n.outputs["Fac"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale * 0.08
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY_ADD"
    nt.links.new(h, mix.inputs[0])
    mix.inputs[1].default_value = 0.7
    nt.links.new(noise.outputs["Fac"], mix.inputs[2])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(mix.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    if color is not None and vary > 0:
        ramp = nt.nodes.new("ShaderNodeMix")
        ramp.data_type = "RGBA"
        ramp.inputs["Factor"].default_value = 1.0
        hsv = nt.nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Value"].default_value = 1.0
        mr = nt.nodes.new("ShaderNodeMapRange")
        nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
        mr.inputs["To Min"].default_value = 1.0 - vary
        mr.inputs["To Max"].default_value = 1.0 + vary
        nt.links.new(mr.outputs["Result"], hsv.inputs["Value"])
        hsv.inputs["Color"].default_value = (*color, 1)
        nt.links.new(hsv.outputs["Color"], p.inputs["Base Color"])
        nt.nodes.remove(ramp)


def materials():
    M = {}
    for key, d in S("materials", {}).items():
        col = tuple(d["color"])
        m, nt, p = principled(f"M_{key}", col, d.get("rough", 0.7), d.get("sheen", 0.0), d.get("coat", 0.0),
                              d.get("spec", 0.5))
        if d.get("weave"):
            weave(nt, p, d.get("scale", 900), d["weave"], d.get("bump", 0.25), col, d.get("vary", 0.0))
        M[key] = m
    return M


# ------------------------------------------------------------------ garments
def build(arm, body, head):
    F = Frame(arm)
    cls, frac = classify(body)
    co = [body.matrix_world @ v.co for v in body.data.vertices]
    loc = [F.local(p) for p in co]
    M = materials()
    under = [body] + ([head] if head else [])
    parts = {}
    collar_top = F.neck_z + S("collar_height", 0.075)
    REPORT["collar_top"] = round(collar_top, 3)

    def wrist_t(i, side_hand, cut):
        """distance of vertex i past the wrist joint along the forearm (negative = toward the elbow)."""
        h = F.j[side_hand]
        e = F.j[side_hand.replace("hand", "lowerarm")]
        ax = (h - e).normalized()
        return (co[i] - h).dot(ax) + cut

    def side(i):
        return "l" if loc[i][0] * (1 if F.local(F.j["thigh_l"])[0] > 0 else -1) > 0 else "r"

    # ---- shirt: torso + arms to the wrist, collar band up the neck
    sh = S("shirt", {})
    keep = []
    for i, c in enumerate(cls):
        k = c in ("torso", "upperarm", "lowerarm") and co[i].z > F.pelvis_z - 0.06
        if c == "lowerarm" and wrist_t(i, f"hand_{side(i)}", sh.get("cuff_past_wrist", -0.01)) > 0:
            k = False
        keep.append(k)
    shirt = body_subset("shirt", body, keep)
    smooth(shirt, 6)
    fit_layer(shirt, under, sh.get("gap", 0.004))
    parts["shirt"] = shirt

    # ---- collar: lofted rings around the neck (head + body points), stand collar
    collar = loft_collar(F, under, F.neck_z - 0.01, collar_top, sh.get("collar_gap", 0.007), "collar")
    parts["collar"] = collar

    # ---- trousers
    tr = S("trousers", {})
    top = F.waist_z + tr.get("rise_above_waist", 0.0)
    keep = [(c in ("thigh", "calf") or (c == "torso" and co[i].z < top)) and co[i].z > F.ankle_z + 0.02
            for i, c in enumerate(cls)]
    trousers = body_subset("trousers", body, keep)
    smooth(trousers, 8)
    for s_ in ("l", "r"):
        knee = F.j[f"calf_{s_}"]
        ank = F.j[f"foot_{s_}"]
        hip = F.j[f"thigh_{s_}"]
        sgn = 1 if F.local(hip)[0] > 0 else -1

        def on_leg(v, sgn=sgn):
            p = trousers.matrix_world @ v.co
            return F.local(p)[0] * sgn > 0.01
        straighten(trousers, hip, knee, tr.get("thigh_ease", 0.5), 0.0, on_leg)
        straighten(trousers, knee, ank, tr.get("leg_ease", 0.85), tr.get("leg_extra", 0.012), on_leg)
    extend_hems(trousers, F, F.ankle_z + 0.03, tr.get("hem_z", 0.035), tr.get("hem_flare", 1.12))
    fit_layer(trousers, under, tr.get("gap", 0.006), rounds=2, smooth_iters=2)
    parts["trousers"] = trousers

    # ---- shoes: lofted along each foot from heel to toe, convex-hull cross-sections + gap (no toes)
    so = S("shoes", {})
    objs = []
    for s_ in ("l", "r"):
        sgn = 1 if F.local(F.j[f"foot_{s_}"])[0] > 0 else -1
        pts = [co[i] for i, c in enumerate(cls)
               if (c == "foot" or (c == "calf" and co[i].z < F.ankle_z + so.get("top_above_ankle", 0.03)))
               and loc[i][0] * sgn > 0]
        objs.append(shoe(F, pts, s_, so))
    parts["shoes"] = join(objs, "shoes")

    # ---- waistcoat
    wc = S("waistcoat", {})
    vz = F.neck_z - wc.get("v_depth", 0.17)
    vhalf = wc.get("v_half_top", 0.075)
    keep = []
    for i, c in enumerate(cls):
        lat, fw, z = loc[i]
        if c != "torso" or frac[i].get("upperarm", 0) > 0.15:
            keep.append(False)
            continue
        front = fw > 0
        bottom = F.waist_z - wc.get("hem_below_waist", 0.03)
        if front:
            bottom -= max(0.0, wc.get("point_drop", 0.06) - wc.get("point_slope", 0.45) * abs(lat))
        k = bottom < z < F.neck_z - 0.005
        if front and z > vz and abs(lat) < vhalf * (z - vz) / max(1e-3, F.neck_z - vz):
            k = False
        keep.append(k)
    waistcoat = body_subset("waistcoat", body, keep)
    smooth(waistcoat, 8)
    fit_layer(waistcoat, under + [shirt, trousers], wc.get("gap", 0.004))
    parts["waistcoat"] = waistcoat
    parts["wc_buttons"] = buttons_line(F, waistcoat, vz - 0.012, F.waist_z - 0.05, wc.get("buttons", 6),
                                       wc.get("button_r", 0.0055), "wc_buttons")

    # ---- cravat: band over the collar, knot, two tails
    parts["cravat"] = cravat(F, collar, shirt, S("cravat", {}))

    # ---- coat body + sleeves
    ct = S("coat", {})
    keep = []
    for i, c in enumerate(cls):
        lat, fw, z = loc[i]
        k = c in ("torso", "upperarm", "lowerarm") and z > F.waist_z - 0.02
        if c == "lowerarm" and wrist_t(i, f"hand_{side(i)}", ct.get("sleeve_short_of_wrist", -0.025)) > 0:
            k = False
        if c == "neck" and z < F.neck_z + ct.get("back_neck_rise", 0.02) and fw < 0:
            k = True                      # the back of the neck base, so the coat meets its collar
        if c == "torso" and z > F.neck_z + ct.get("back_neck_rise", 0.02):
            k = False
        if k and fw > 0 and c in ("torso", "neck") and z > F.neck_z - 0.06:
            k = False                     # front of the neck: the shirt collar and cravat show here
        if k and fw > 0 and c == "torso":
            t = (z - F.waist_z) / max(1e-3, F.neck_z - F.waist_z)
            half = ct.get("open_waist", 0.085) + ct.get("open_widen", 0.03) * t
            if abs(lat) < half:
                k = False
        keep.append(k)
    coat = body_subset("coat", body, keep)
    smooth(coat, ct.get("smooth", 25))
    for s_ in ("l", "r"):
        sgn = 1 if F.local(F.j[f"upperarm_{s_}"])[0] > 0 else -1

        def on_arm(v, sgn=sgn):
            return F.local(coat.matrix_world @ v.co)[0] * sgn > 0.17
        straighten(coat, F.j[f"upperarm_{s_}"], F.j[f"lowerarm_{s_}"], ct.get("sleeve_ease", 0.6), 0.008, on_arm)
        straighten(coat, F.j[f"lowerarm_{s_}"], F.j[f"hand_{s_}"], ct.get("sleeve_ease", 0.6), 0.012, on_arm)
    fit_layer(coat, under + [shirt, waistcoat, parts["wc_buttons"], parts["cravat"]], ct.get("gap", 0.006))
    cuffs(F, coat, ct)
    parts["coat"] = coat
    parts["lapels"] = lapels(F, coat, ct)
    parts["coat_collar"] = loft_collar(F, under + [collar], F.neck_z - 0.03, F.neck_z + ct.get("collar_rise", 0.035),
                                       ct.get("collar_gap", 0.012), "coat_collar", back_only=True)
    legs = [co[i] for i, c in enumerate(cls) if c in ("torso", "thigh", "calf")]
    parts["skirt"] = skirt(F, legs, under + [trousers, waistcoat], ct)
    parts["flaps"] = flaps(F, parts["skirt"], ct)
    parts["coat_buttons"] = coat_buttons(F, coat, parts["skirt"], ct)

    # strays, then thickness and materials
    for key, th in S("thickness", {}).items():
        if key in parts:
            solidify(parts[key], th)
    cb = collider(under)
    loose = {"skirt": 0.3, "flaps": 0.3, "coat_buttons": 0.3, "coat": 0.16, "lapels": 0.16}
    REPORT["strays_fixed"] = {k: fix_strays(o, cb, loose.get(k, 0.12)) for k, o in parts.items()}
    for key, ob in parts.items():
        set_mat(ob, M[S("material_of", {}).get(key, "wool")])
    REPORT["parts"] = {k: len(o.data.vertices) for k, o in parts.items()}
    return F, parts


def loft_collar(F, under, z0, z1, gap, name, back_only=False, nang=72, nring=8):
    """Stand collar: rings around the neck from z0 to z1, each at the largest radius of the neck points near that
    height (head and body meshes) plus a gap, so the collar clears both."""
    pts = []
    for o in under:
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co
            if z0 - 0.02 < p.z < z1 + 0.02:
                pts.append(p)
    cx = sum(p.x for p in pts) / len(pts)
    cy = sum(p.y for p in pts) / len(pts)
    rings = []
    for k in range(nring + 1):
        z = z0 + (z1 - z0) * k / nring
        band = [p for p in pts if abs(p.z - z) < 0.012]
        # neck cross-section points only (exclude shoulders): within 11 cm of the axis
        hull = convex_hull_2d([(round(p.x - cx, 4), round(p.y - cy, 4)) for p in band
                               if math.hypot(p.x - cx, p.y - cy) < 0.11])
        ring = []
        for a in range(nang):
            ang = 2 * math.pi * a / nang
            r = ray_hull(hull, ang) + gap if len(hull) >= 3 else 0.07
            ring.append(Vector((cx + r * math.cos(ang), cy + r * math.sin(ang), z)))
        rings.append(ring)
    # keep the angles off the front opening (and the front half entirely for a back-only collar)
    front_ang = math.atan2(F.fwd.y, F.fwd.x)
    keep_a = []
    for a in range(nang):
        ang = 2 * math.pi * a / nang
        dfront = abs((ang - front_ang + math.pi) % (2 * math.pi) - math.pi)
        keep_a.append(dfront > (math.radians(100) if back_only else math.radians(9)))
    return grid_mesh(name, rings, keep_a, closed=all(keep_a))


def grid_mesh(name, rings, keep_a, closed):
    me = bpy.data.meshes.new(name)
    nang = len(rings[0])
    verts = [p for ring in rings for p in ring]
    faces = []
    for k in range(len(rings) - 1):
        for a in range(nang):
            b = (a + 1) % nang
            if not closed and (not keep_a[a] or not keep_a[b] or b == 0 and not closed):
                continue
            faces.append((k * nang + a, k * nang + b, (k + 1) * nang + b, (k + 1) * nang + a))
    me.from_pydata([tuple(v) for v in verts], [], faces)
    ob = new_obj(name, me)
    bm = bm_of(ob)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def extend_hems(ob, F, below_z, hem_z, flare):
    """Extrude the bottom boundary loops of the trouser legs down to hem_z, opened out by flare."""
    bm = bm_of(ob)
    edges = [e for e in bm.edges if e.is_boundary and all(v.co.z < below_z for v in e.verts)]
    if not edges:
        REPORT["warnings"].append("trousers: no hem boundary found")
        bm.free()
        return
    for _ in range(3):
        ret = bmesh.ops.extrude_edge_only(bm, edges=edges)
        nv = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMVert)]
        groups = defaultdict(list)
        for v in nv:
            groups[1 if F.local(v.co)[0] > 0 else -1].append(v)
        for vs in groups.values():
            c = sum((v.co for v in vs), Vector()) / len(vs)
            for v in vs:
                d = v.co - c
                d.z = 0
                v.co = Vector((c.x, c.y, 0)) + d * flare ** (1 / 3)
                v.co.z = max(hem_z, v.co.z - 0.01)
        edges = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMEdge) and g.is_boundary]
    for v in bm.verts:
        if v.is_boundary and v.co.z < below_z:
            v.co.z = hem_z
    bm.to_mesh(ob.data)
    bm.free()


def buttons_line(F, surf, z_top, z_bot, n, r, name, lat=0.0, out=0.002):
    bvh = collider([surf])
    objs = []
    for k in range(n):
        z = z_top + (z_bot - z_top) * k / max(1, n - 1)
        p = F.world(lat, 0.4, z)
        hit = bvh.ray_cast(p, -F.fwd, 1.0)
        if hit[0] is None:
            continue
        objs.append(button(hit[0] + hit[1] * (out + r * 0.4), hit[1], r))
    return join(objs, name) if objs else new_obj(name, bpy.data.meshes.new(name))


def button(p, n, r):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=r)
    for v in bm.verts:
        v.co.z *= 0.45
    me = bpy.data.meshes.new("btn")
    bm.to_mesh(me)
    bm.free()
    ob = new_obj("btn", me)
    ob.matrix_world = Matrix.Translation(p) @ n.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return ob


def cravat(F, collar, shirt, cv):
    """A band around the lower collar, a knot at the throat and two hanging tails over the shirt front."""
    objs = []
    bm = bm_of(collar)
    zs = [v.co.z for v in bm.verts]
    z0, z1 = min(zs), max(zs)
    band_top = z0 + (z1 - z0) * cv.get("band_frac", 0.55)
    me = collar.data.copy()
    band = new_obj("cravat_band", me)
    bmb = bm_of(band)
    bmesh.ops.delete(bmb, geom=[v for v in bmb.verts if v.co.z > band_top + 1e-4], context="VERTS")
    c = Vector((sum(v.co.x for v in bmb.verts) / len(bmb.verts), sum(v.co.y for v in bmb.verts) / len(bmb.verts), 0))
    for v in bmb.verts:
        d = v.co - Vector((c.x, c.y, v.co.z))
        v.co += d.normalized() * cv.get("band_gap", 0.004)
    bmb.to_mesh(band.data)
    bmb.free()
    objs.append(band)
    bvh = collider([collar])
    hit = bvh.ray_cast(F.world(0, 0.4, (z0 + band_top) / 2), -F.fwd, 1.0)
    knot_p = (hit[0] if hit[0] is not None else F.world(0, 0.07, z0)) + F.fwd * 0.012
    bmk = bmesh.new()
    bmesh.ops.create_uvsphere(bmk, u_segments=16, v_segments=10, radius=1.0)
    kx, ky, kz = cv.get("knot", [0.022, 0.012, 0.016])
    for v in bmk.verts:
        v.co = Vector((v.co.x * kx, v.co.y * ky, v.co.z * kz))
    me = bpy.data.meshes.new("knot")
    bmk.to_mesh(me)
    bmk.free()
    knot = new_obj("cravat_knot", me)
    knot.matrix_world = Matrix.Translation(knot_p) @ Matrix.Rotation(math.atan2(F.fwd.x, -F.fwd.y), 4, "Z")
    objs.append(knot)
    # tails: tapered strips hanging from the knot, fanned, laid over the shirt front
    sbvh = collider([shirt])
    L = cv.get("tail_len", 0.11)
    for sgn in (-1, 1):
        rows = []
        for k in range(9):
            t = k / 8
            w = cv.get("tail_w", 0.03) * (0.6 + 0.6 * t)
            lat = sgn * (0.008 + cv.get("tail_fan", 0.018) * t)
            z = knot_p.z - 0.008 - L * t
            row = []
            for s in (-0.5, 0.5):
                p = F.world(lat + s * w, 0.4, z)
                h = sbvh.ray_cast(p, -F.fwd, 1.0)
                q = (h[0] + h[1] * (0.006 + 0.003 * (1 - t))) if h[0] is not None else F.world(lat + s * w, 0.09, z)
                row.append(q)
            rows.append(row)
        me = bpy.data.meshes.new("tail")
        verts = [p for r in rows for p in r]
        faces = [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(len(rows) - 1)]
        me.from_pydata([tuple(v) for v in verts], [], faces)
        objs.append(new_obj("cravat_tail", me))
    for o in objs:
        bpy.context.view_layer.objects.active = o
    ob = join(objs, "cravat")
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return ob


def cuffs(F, coat, ct):
    """Turned-back cuffs: the last stretch of each sleeve stands a little proud."""
    L = ct.get("cuff_len", 0.075)
    for s_ in ("l", "r"):
        h = F.j[f"hand_{s_}"]
        e = F.j[f"lowerarm_{s_}"]
        ax = (h - e).normalized()
        sgn = 1 if F.local(h)[0] > 0 else -1
        end = max(((v.co - h).dot(ax) for v in coat.data.vertices if F.local(v.co)[0] * sgn > 0.2), default=0)
        for v in coat.data.vertices:
            if F.local(v.co)[0] * sgn <= 0.2:
                continue
            t = (v.co - h).dot(ax)
            if end - L < t <= end + 1e-4:
                c = e + ax * (v.co - e).dot(ax)
                d = v.co - c
                v.co = c + d * (1 + ct.get("cuff_proud", 0.006) / max(d.length, 1e-3))
    coat.data.update()


def lapels(F, coat, ct):
    """Lapels: the coat faces next to the front opening above the roll point, copied and laid proud of the coat,
    widest at the chest, with a notch below the collar."""
    roll_t = ct.get("roll_t", 0.35)
    wmax = ct.get("lapel_w", 0.07)
    bm = bm_of(coat)
    keepf = []
    for f in bm.faces:
        c = f.calc_center_median()
        lat, fw, z = F.local(c)
        t = (z - F.waist_z) / max(1e-3, F.neck_z - F.waist_z)
        if fw <= 0 or t < roll_t or t > 1.02:
            continue
        half = ct.get("open_waist", 0.085) + ct.get("open_widen", 0.03) * t
        u = (t - roll_t) / (1 - roll_t)
        w = wmax * min(1.0, u * 1.6) * (0.35 if ct.get("notch_t", 0.86) < u < ct.get("notch_t", 0.86) + 0.07 else 1)
        if half <= abs(lat) < half + w:
            keepf.append(f.index)
    bm.free()
    me = coat.data.copy()
    ob = new_obj("lapels", me)
    bml = bm_of(ob)
    bml.faces.ensure_lookup_table()
    ks = set(keepf)
    bmesh.ops.delete(bml, geom=[f for f in bml.faces if f.index not in ks], context="FACES")
    bmesh.ops.delete(bml, geom=[v for v in bml.verts if not v.link_faces], context="VERTS")
    bml.normal_update()
    for v in bml.verts:
        v.co += v.normal * ct.get("lapel_proud", 0.005)
    bml.to_mesh(ob.data)
    bml.free()
    return ob


def skirt(F, pts, under, ct, nang=96, nring=16):
    """Coat skirt from the waist seam to below the knee: convex-hull cross-sections of hips and legs (so it bridges
    between the legs like cloth), plus a gap and a flare that grows downward; open at the front."""
    z_top = F.waist_z - 0.0
    z_hem = F.knee_z + ct.get("hem_vs_knee", -0.05)
    cx, cy = F.center
    open_top = ct.get("skirt_open_top_deg", 34)
    open_hem = ct.get("skirt_open_hem_deg", 44)
    front_ang = math.atan2(F.fwd.y, F.fwd.x)
    rings = []
    prev = None
    bvh = collider(under)
    for k in range(nring + 1):
        t = k / nring
        z = z_top + (z_hem - z_top) * t
        band = [(p.x - cx, p.y - cy) for p in pts if abs(p.z - z) < 0.015 and math.hypot(p.x - cx, p.y - cy) < 0.4]
        # above the crotch the band holds the hips only; below it both legs (the hull bridges them)
        hull = convex_hull_2d([(round(a, 4), round(b, 4)) for a, b in band])
        ring = []
        for a in range(nang):
            ang = 2 * math.pi * a / nang
            r = ray_hull(hull, ang) + ct.get("skirt_gap", 0.02) + ct.get("skirt_flare", 0.07) * t ** 1.3
            if prev is not None:
                r = max(r, prev[a] * 0.995)          # cloth hangs: never tucks back in below
            ring.append(r)
        prev = ring
        rings.append([Vector((cx + r * math.cos(2 * math.pi * a / nang), cy + r * math.sin(2 * math.pi * a / nang), z))
                      for a, r in enumerate(ring)])
    keep_a = []
    for a in range(nang):
        ang = 2 * math.pi * a / nang
        keep_a.append(abs((ang - front_ang + math.pi) % (2 * math.pi) - math.pi) > math.radians(open_top))
    ob = grid_mesh("skirt", rings, keep_a, closed=False)
    # cut-away: the front edges open further toward the hem
    bm = bm_of(ob)
    kill = []
    for f in bm.faces:
        c = f.calc_center_median()
        t = (z_top - c.z) / max(1e-3, z_top - z_hem)
        ang = math.atan2(c.y - cy, c.x - cx)
        if abs((ang - front_ang + math.pi) % (2 * math.pi) - math.pi) < math.radians(open_top + (open_hem - open_top) * t):
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()
    smooth_skirt(ob, z_top, ct.get("skirt_smooth", 6))
    push_out(ob, bvh, ct.get("skirt_gap", 0.02) * 0.6)
    ob["z_top"], ob["z_hem"] = z_top, z_hem
    return ob


def smooth_skirt(ob, z_top, iters):
    """Even out the hull facets around each ring (horizontal smoothing only; heights stay)."""
    bm = bm_of(ob)
    vs = [v for v in bm.verts if v.co.z < z_top - 1e-4]
    for _ in range(iters):
        bmesh.ops.smooth_vert(bm, verts=vs, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=False)
    bm.to_mesh(ob.data)
    bm.free()


def shoe(F, pts, side, so, nst=22, nang=40):
    """One closed shoe: stations from heel to toe along the foot's forward axis; each station a ring around the
    hull of the foot points near it (lateral x height plane), widened by a gap, flat at the sole, rounded toe."""
    fwd = F.fwd
    fs = [p.dot(fwd) for p in pts]
    f0 = min(fs) - so.get("heel_back", 0.008)
    f1 = max(fs) + so.get("toe_ext", 0.02)
    lat = F.lat
    rings = []
    top = F.ankle_z + so.get("top_above_ankle", 0.03)
    sole = so.get("sole_bottom", -0.004)
    gap = so.get("gap", 0.007)
    for k in range(nst + 1):
        t = k / nst
        f = f0 + (f1 - f0) * t
        band = [p for p, fp in zip(pts, fs) if abs(fp - f) < 0.012]
        if len(band) < 4:
            band = sorted(pts, key=lambda p: abs(p.dot(fwd) - f))[:30]
        cl = sum((p.dot(lat) for p in band)) / len(band)
        cz = max(0.02, sum(p.z for p in band) / len(band))
        hull = convex_hull_2d([(round(p.dot(lat) - cl, 4), round(p.z - cz, 4)) for p in band])
        # toe: taper the last stretch to a rounded point
        taper = 1.0 if t < 0.8 else max(0.25, 1 - (t - 0.8) / 0.2 * 0.75) ** 0.5
        ring = []
        for a in range(nang):
            ang = 2 * math.pi * a / nang
            r = (ray_hull(hull, ang) + gap) * taper if len(hull) >= 3 else 0.02
            x, z = cl + r * math.cos(ang), cz + r * math.sin(ang)
            z = min(max(z, sole), top)
            base = Vector((F.center[0], F.center[1], 0)) + fwd * f + lat * x
            ring.append(Vector((base.x, base.y, z)))
        rings.append(ring)
    ob = grid_mesh(f"shoe_{side}", rings, [True] * nang, closed=True)
    bm = bm_of(ob)
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    smooth(ob, 3, 0.5)
    return ob


def flaps(F, skirt_ob, ct):
    bvh = collider([skirt_ob])
    objs = []
    z = F.waist_z - ct.get("flap_below_waist", 0.07)
    for sgn in (-1, 1):
        lat = sgn * ct.get("flap_lat", 0.15)
        p = F.world(lat * 2.5, 0.6, z)
        tgt = F.world(lat, 0.0, z)
        d = (tgt - p).normalized()
        hit = bvh.ray_cast(p, d, 2.0)
        if hit[0] is None:
            continue
        n = hit[1]
        tang = n.cross(Vector((0, 0, 1))).normalized()
        w, h = ct.get("flap_w", 0.16), ct.get("flap_h", 0.05)
        c = hit[0] + n * 0.003
        quad = [c + tang * (-w / 2) + Vector((0, 0, 0)), c + tang * (w / 2), c + tang * (w / 2) + Vector((0, 0, -h)),
                c + tang * (-w / 2) + Vector((0, 0, -h))]
        me = bpy.data.meshes.new("flap")
        me.from_pydata([tuple(q) for q in quad], [], [(0, 1, 2, 3)])
        objs.append(new_obj("flap", me))
    ob = join(objs, "flaps")
    return ob


def coat_buttons(F, coat, skirt_ob, ct):
    objs = []
    bvh = collider([coat, skirt_ob])
    half = ct.get("open_waist", 0.085)
    r = ct.get("button_r", 0.009)
    for sgn in (-1, 1):
        for dz in ct.get("button_dz", [0.06, -0.0, -0.08]):
            z = F.waist_z + dz
            p = F.world(sgn * (half + 0.025), 0.4, z)
            hit = bvh.ray_cast(p, -F.fwd, 1.0)
            if hit[0] is not None:
                objs.append(button(hit[0] + hit[1] * (0.002 + r * 0.4), hit[1], r))
    return join(objs, "coat_buttons") if objs else new_obj("coat_buttons", bpy.data.meshes.new("cb"))


# ------------------------------------------------------------------ skinning
def skin(arm, body, F, parts):
    for key, ob in parts.items():
        if key in ("shirt", "trousers", "waistcoat", "coat", "shoes", "lapels"):
            # copied from the body already, but the push-out moved vertices: re-transfer for clean weights
            transfer_weights(ob, body)
        elif key in ("skirt", "flaps"):
            skirt_weights(ob, body, F)
        else:
            transfer_weights(ob, body)
        ob.parent = arm
        m = ob.modifiers.new("arm", "ARMATURE")
        m.object = arm
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.select_all(action="DESELECT")
        ob.select_set(True)
        bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=8)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)


def skirt_weights(ob, body, F):
    """Top of the skirt rides the pelvis; lower down each side follows its thigh (and a little its calf), the back
    centre splits between both, so a step or a seated pose swings the skirt without tearing it."""
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    gp = ob.vertex_groups.new(name="pelvis")
    gl = ob.vertex_groups.new(name="thigh_l")
    gr = ob.vertex_groups.new(name="thigh_r")
    sl = 1 if F.local(F.j["thigh_l"])[0] > 0 else -1
    z_top = ob.get("z_top", F.waist_z)
    z_hem = ob.get("z_hem", F.knee_z)
    for v in ob.data.vertices:
        t = min(1.0, max(0.0, (z_top - v.co.z) / max(1e-3, z_top - z_hem)))
        lat = F.local(v.co)[0] * sl
        s = max(-1.0, min(1.0, lat / 0.09))
        leg = 0.85 * t ** 0.8
        wl = leg * (1 + s) / 2
        wr = leg * (1 - s) / 2
        gp.add([v.index], max(0.0, 1 - leg), "REPLACE")
        if wl > 1e-3:
            gl.add([v.index], wl, "REPLACE")
        if wr > 1e-3:
            gr.add([v.index], wr, "REPLACE")


def body_trim(body, F, cls):
    """The body cut to what the outfit leaves visible: hands and wrists, and the neck. Faces with any vertex outside
    the kept set go, so the cut edge stays under a cuff or the collar."""
    keep = [c in ("hand", "neck") for c in cls]
    co = [v.co for v in body.data.vertices]
    for i, c in enumerate(cls):
        if c == "lowerarm":
            for s_ in ("l", "r"):
                h = F.j[f"hand_{s_}"]
                e = F.j[f"lowerarm_{s_}"]
                ax = (h - e).normalized()
                if -0.05 < (co[i] - h).dot(ax) and (co[i] - h).length < 0.12:
                    keep[i] = True
        if c == "torso" and co[i].z > F.neck_z - 0.01:
            keep[i] = True
    ob = body_subset("body_trim", body, keep)
    ob.data.materials.clear()
    for m in body.data.materials:
        ob.data.materials.append(m)
    return ob


# ------------------------------------------------------------------ poke-through
def poke_counts(layers, core):
    """Poke-through per (inner, outer) layer pair, independent of how garment normals face: for each inner vertex,
    take the nearest point on the core (body + head) and its normal; a ray from there along the normal meets the
    outer layer at some distance; the inner vertex pokes through when it lies further out than that. Inner vertices
    whose ray meets no outer layer within 10 cm are not covered by it and are not counted. Returns [pokes, covered]."""
    dg = bpy.context.evaluated_depsgraph_get()
    cbvh = collider(core)
    res = {}

    def world_verts(o):
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        vs = [o.matrix_world @ v.co for v in me.vertices]
        oe.to_mesh_clear()
        return vs
    for inner, outer in layers:
        obvh = collider(outer)
        n = tot = 0
        for o in inner:
            for p in world_verts(o):
                hit = cbvh.find_nearest(p, 0.15)
                if hit[0] is None:
                    continue
                loc = hit[0]
                d_in = (p - loc).length
                # outward = from the core surface to the inner vertex (BVH face normals of imported meshes may
                # face either way)
                nrm = (p - loc) / d_in if d_in > 1e-5 else hit[1]
                h = obvh.ray_cast(loc + nrm * 0.0005, nrm, 0.10)
                if h[0] is None:
                    continue
                tot += 1
                if d_in > h[3] + 0.0005 + 0.0005:
                    n += 1
        res[f"{'+'.join(o.name for o in inner)} > {'+'.join(o.name for o in outer)}"] = [n, tot]
    return res


def far_verts(parts, core, limit=0.12, loose=("skirt", "flaps", "coat_buttons")):
    """Stray vertices: further from the body than any garment should be (a spike or a mis-hit), per part."""
    bvh = collider(core)
    res = {}
    for key, o in parts.items():
        lim = 0.3 if key in loose else limit
        far = []
        for v in o.data.vertices:
            p = o.matrix_world @ v.co
            h = bvh.find_nearest(p, 2.0)
            d = h[3] if h[0] is not None else 9.0
            if d > lim:
                far.append((round(d, 3), [round(x, 3) for x in p]))
        res[key] = [len(far), sorted(far, reverse=True)[:3]]
    return res


def test_pose(arm, kind):
    pb = arm.pose.bones
    for b in pb:
        b.rotation_mode = "QUATERNION"
        b.rotation_quaternion = (1, 0, 0, 0)
        b.rotation_euler = (0, 0, 0)
        b.location = (0, 0, 0)

    def rot_world(name, axis, deg):
        """rotate a pose bone about a world axis through its head (rest frame), via its local matrix."""
        b = pb[name]
        R = Matrix.Rotation(math.radians(deg), 4, axis)
        rest = b.bone.matrix_local
        loc = rest.inverted().to_3x3() @ R.to_3x3() @ rest.to_3x3()
        b.rotation_mode = "QUATERNION"
        b.rotation_quaternion = (b.rotation_quaternion.to_matrix() @ loc).to_quaternion()
    F = Frame(arm)
    if kind == "seated":
        for s_ in ("l", "r"):
            rot_world(f"thigh_{s_}", F.lat, 80)
            rot_world(f"calf_{s_}", F.lat, -80)
    elif kind == "arm_up":
        rot_world("upperarm_r", F.fwd, -70 if F.local(F.j["upperarm_r"])[0] < 0 else 70)
        rot_world("lowerarm_r", F.fwd, 0)
    bpy.context.view_layer.update()


# ------------------------------------------------------------------ bake + export
def set_device():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    p = bpy.context.preferences.addons["cycles"].preferences
    p.compute_device_type = "HIP"
    p.get_devices()
    want = f"0000:{JOB.get('hip_bus', '05')}:00"
    hit = False
    for d in p.devices:
        d.use = d.type == "HIP" and d.id.endswith(want)
        hit = hit or d.use
    if not hit:
        raise RuntimeError(f"no HIP device {want}: never bake or render on the CPU")
    sc.cycles.device = "GPU"
    if hasattr(sc.cycles, "denoising_use_gpu"):
        sc.cycles.denoising_use_gpu = True
    REPORT["device"] = f"HIP {want}"


def uv_unwrap(ob):
    me = ob.data
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    me.uv_layers.new(name="UVMap")
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.002, area_weight=0.0)
    bpy.ops.uv.pack_islands(margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")


def bake(ob, res, samples):
    """Emission bakes of base colour and (roughness, metallic, AO), then a tangent normal bake; one texture set
    for the whole outfit (conventions of the asset chain)."""
    sc = bpy.context.scene
    set_device()
    sc.render.bake.margin = 8
    sc.render.bake.margin_type = "EXTEND"
    imgs = {}
    for k, color in (("BaseColor", True), ("Normal", False), ("RMA", False)):
        im = bpy.data.images.new(f"T_outfit_{k}", res, res, alpha=False)
        im.colorspace_settings.name = "sRGB" if color else "Non-Color"
        imgs[k] = im
    mats = [s.material for s in ob.material_slots if s.material]
    tex = {}
    for m in mats:
        tn = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex[m.name] = tn
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob

    def target(k):
        for m in mats:
            tn = tex[m.name]
            tn.image = imgs[k]
            for n in m.node_tree.nodes:
                n.select = False
            tn.select = True
            m.node_tree.nodes.active = tn

    def emit(k, srcs):
        saved = []
        for m in mats:
            nt = m.node_tree
            out = nt.nodes["Material Output"]
            pr = nt.nodes["Principled BSDF"]
            old = out.inputs["Surface"].links[0].from_socket
            em = nt.nodes.new("ShaderNodeEmission")
            tmp = [em]
            if isinstance(srcs, str):
                i = pr.inputs[srcs]
                if i.is_linked:
                    nt.links.new(i.links[0].from_socket, em.inputs["Color"])
                else:
                    em.inputs["Color"].default_value = i.default_value
            else:
                cc = nt.nodes.new("ShaderNodeCombineColor")
                tmp.append(cc)
                for name, ch in zip(srcs, ("Red", "Green", "Blue")):
                    if name == "AO":
                        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
                        ao.samples = 8
                        ao.inputs["Distance"].default_value = 0.05
                        tmp.append(ao)
                        nt.links.new(ao.outputs["AO"], cc.inputs[ch])
                    else:
                        i = pr.inputs[name]
                        if i.is_linked:
                            nt.links.new(i.links[0].from_socket, cc.inputs[ch])
                        else:
                            cc.inputs[ch].default_value = i.default_value
                nt.links.new(cc.outputs["Color"], em.inputs["Color"])
            nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
            saved.append((nt, out, old, tmp))
        target(k)
        sc.cycles.samples = samples
        bpy.ops.object.bake(type="EMIT", use_clear=True)
        for nt, out, old, tmp in saved:
            for n in tmp:
                nt.nodes.remove(n)
            nt.links.new(old, out.inputs["Surface"])

    with Timer("bake_basecolor"):
        emit("BaseColor", "Base Color")
    with Timer("bake_rma"):
        emit("RMA", ["Roughness", "Metallic", "AO"])
    with Timer("bake_normal"):
        target("Normal")
        sc.cycles.samples = samples
        bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_clear=True)
    os.makedirs(os.path.join(OUT, "textures"), exist_ok=True)
    for k, im in imgs.items():
        im.filepath_raw = os.path.join(OUT, "textures", f"T_outfit_{k}.png")
        im.file_format = "PNG"
        im.save()
    return imgs


def baked_material(ob, imgs):
    """One material on the baked textures (what UE gets), replacing the procedural ones."""
    m = bpy.data.materials.new("M_Outfit")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    bc = nt.nodes.new("ShaderNodeTexImage")
    bc.image = imgs["BaseColor"]
    nt.links.new(bc.outputs["Color"], p.inputs["Base Color"])
    rma = nt.nodes.new("ShaderNodeTexImage")
    rma.image = imgs["RMA"]
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(rma.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Red"], p.inputs["Roughness"])
    nt.links.new(sep.outputs["Green"], p.inputs["Metallic"])
    nm = nt.nodes.new("ShaderNodeTexImage")
    nm.image = imgs["Normal"]
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nm.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], p.inputs["Normal"])
    ob.data.materials.clear()
    ob.data.materials.append(m)


def export_fbx(objs, arm, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"ARMATURE", "MESH"},
                             apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", axis_forward="-Z",
                             axis_up="Y", add_leaf_bones=False, use_armature_deform_only=False,
                             primary_bone_axis="Y", secondary_bone_axis="X", armature_nodetype="NULL",
                             bake_anim=False, mesh_smooth_type="FACE", use_tspace=True, path_mode="STRIP")


# ------------------------------------------------------------------ preview
def preview(objs, tag, views=(("front", 0), ("three_q", 35), ("side", 90), ("back", 180)), px=(640, 900),
            samples=24, target_z=None, dist=3.2, focal=50):
    sc = bpy.context.scene
    set_device()
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = px
    sc.render.image_settings.file_format = "PNG"
    if sc.world is None:
        sc.world = bpy.data.worlds.new("W")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.42, 0.42, 0.42, 1)
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_render = o not in objs
    F = Frame(next(o for o in bpy.data.objects if o.type == "ARMATURE"))
    tz = target_z if target_z is not None else 0.95
    if "key_light" not in bpy.data.objects:
        for name, ang, e, en in (("key_light", -40, 35, 700), ("fill_light", 50, 15, 250), ("rim_light", 160, 40, 400)):
            ld = bpy.data.lights.new(name, "AREA")
            ld.energy = en
            ld.size = 2.0
            lo = new_obj(name, ld)
            d = (F.fwd * math.cos(math.radians(ang)) + F.lat * math.sin(math.radians(ang))) * math.cos(math.radians(e))
            lo.location = Vector((0, 0, tz)) + (d + Vector((0, 0, math.sin(math.radians(e))))) * 3.5
            lo.rotation_euler = (Vector((0, 0, tz)) - lo.location).to_track_quat("-Z", "Y").to_euler()
    cd = bpy.data.cameras.new("cam")
    cd.lens = focal
    cam = new_obj("cam", cd)
    sc.camera = cam
    paths = []
    for name, az in views:
        a = math.radians(az)
        d = F.fwd * math.cos(a) + F.lat * math.sin(a)
        tgt = Vector((F.center[0], F.center[1], tz))
        cam.location = tgt + d * dist + Vector((0, 0, 0.05))
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        p = os.path.join(OUT, "preview", f"{tag}_{name}.png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
    bpy.data.objects.remove(cam)
    return paths


def skin_mat():
    m, nt, p = principled("M_skin_preview", (0.62, 0.48, 0.42), 0.5)
    return m


# ------------------------------------------------------------------ main
def main():
    steps = JOB.get("steps", ["build", "bake", "export", "preview", "pose_check"])
    clear()
    with Timer("import"):
        arm, body, head = load_body()
    with Timer("build"):
        F, parts = build(arm, body, head)
    cls, _ = classify(body)
    with Timer("trim"):
        trim = body_trim(body, F, cls)
    with Timer("skin"):
        skin(arm, body, F, parts)
        trim.parent = arm
        trim.modifiers.new("arm", "ARMATURE").object = arm
    garments = list(parts.values())
    core = [body] + ([head] if head else [])
    layers = [([head] if head else [], [parts["shirt"], parts["collar"]]),
              ([parts["shirt"]], [parts["waistcoat"], parts["coat"]]),
              ([parts["waistcoat"]], [parts["coat"]]),
              ([parts["trousers"]], [parts["skirt"]]),
              ([parts["shoes"]], [parts["trousers"]])]
    REPORT["far_verts"] = far_verts(parts, core)
    with Timer("poke_rest"):
        REPORT["poke"]["rest"] = poke_counts(layers, core)
    sm = skin_mat()
    for o in ([head] if head else []) + [trim]:
        o.data.materials.clear()
        o.data.materials.append(sm)
    body.hide_render = True
    if "preview" in steps:
        with Timer("preview_rest"):
            REPORT["preview"] = preview(garments + [trim] + ([head] if head else []), "rest",
                                        **JOB.get("preview_args", {}))
    if "pose_check" in steps:
        for kind in ("seated", "arm_up"):
            test_pose(arm, kind)
            with Timer(f"poke_{kind}"):
                REPORT["poke"][kind] = poke_counts(layers, core)
            if "preview" in steps:
                with Timer(f"preview_{kind}"):
                    REPORT.setdefault("preview_pose", []).extend(
                        preview(garments + [trim] + ([head] if head else []), kind,
                                views=(("front", 0), ("side", 90)), **JOB.get("preview_args", {})))
        test_pose(arm, "rest")
    outfit = None
    if "bake" in steps or "export" in steps:
        bpy.ops.object.select_all(action="DESELECT")
        outfit = join([o for o in garments], "outfit")
        with Timer("uv"):
            uv_unwrap(outfit)
    if "bake" in steps:
        with Timer("bake"):
            imgs = bake(outfit, JOB.get("bake_px", 4096), JOB.get("bake_samples", 4))
        baked_material(outfit, imgs)
    if "export" in steps:
        with Timer("export"):
            export_fbx([outfit], arm, os.path.join(OUT, "outfit.fbx"))
            export_fbx([trim], arm, os.path.join(OUT, "body_trim.fbx"))
            if JOB.get("export_body_roundtrip"):
                export_fbx([body], arm, os.path.join(OUT, "body_roundtrip.fbx"))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "outfit.blend"))
    REPORT["ok"] = True
    save_report()


try:
    main()
except Exception:
    import traceback
    REPORT["error"] = traceback.format_exc()
    save_report()
    raise
