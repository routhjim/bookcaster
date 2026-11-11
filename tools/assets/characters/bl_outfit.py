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
REPORT = {"times": {}, "warnings": [], "poke": {}, "drape_times": JOB.get("drape_times")}
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
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)   # the body is split along its UV seams: weld
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


def smooth(ob, iters, factor=0.5, pin_boundary=True):
    """Laplacian smoothing; open edges stay put by default (an unpinned neckline or hem creeps several cm)."""
    bm = bm_of(ob)
    vs = [v for v in bm.verts if not (pin_boundary and v.is_boundary)]
    for _ in range(iters):
        bmesh.ops.smooth_vert(bm, verts=vs, factor=factor, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(ob.data)
    bm.free()


def smooth_edges(ob, iters=12, factor=0.5):
    """Smooth open edges along themselves (each boundary vertex toward its boundary neighbours): a cut that follows
    the body's faces is stair-stepped; this makes it a clean line without letting it creep."""
    bm = bm_of(ob)
    for _ in range(iters):
        new = {}
        for v in bm.verts:
            if not v.is_boundary:
                continue
            nb = [e.other_vert(v).co for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new[v] = v.co * (1 - factor) + (nb[0] + nb[1]) * (factor / 2)
        for v, c in new.items():
            v.co = c
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


def _math(nt, op, a, b=None, c=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for i, x in enumerate((a, b, c)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            nt.links.new(x, n.inputs[i])
    return n.outputs[0]


def _plane_dist(nt, pos, p0, n):
    """dot(P - p0, n) of the shading point in object space (= world: garments keep identity transforms)."""
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    nt.links.new(pos, sub.inputs[0])
    sub.inputs[1].default_value = tuple(p0)
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    nt.links.new(sub.outputs[0], dot.inputs[0])
    dot.inputs[1].default_value = tuple(Vector(n).normalized())
    return dot.outputs["Value"]


def _ramp(nt, x, a, b, va, vb):
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.interpolation_type = "SMOOTHSTEP"
    nt.links.new(x, mr.inputs["Value"])
    mr.inputs["From Min"].default_value = a
    mr.inputs["From Max"].default_value = b
    mr.inputs["To Min"].default_value = va
    mr.inputs["To Max"].default_value = vb
    return mr.outputs["Result"]


def seam_mask(nt, pos, seams):
    """1 on any seam line, falling to 0 over its half-width; each seam is a plane cut by half-space masks."""
    acc = None
    for sm in seams:
        d = _math(nt, "ABSOLUTE", _plane_dist(nt, pos, sm["p"], sm["n"]))
        g = _ramp(nt, d, 0.0, sm.get("w", 0.0014), 1.0, 0.0)
        for mp, mn in sm.get("masks", []):
            g = _math(nt, "MULTIPLY", g, _ramp(nt, _plane_dist(nt, pos, mp, mn), -0.004, 0.004, 0.0, 1.0))
        acc = g if acc is None else _math(nt, "MAXIMUM", acc, g)
    return acc


def cloth(key, d, seams):
    """A cloth surface from object-space procedural textures: twill (diagonal wale) for wool, plain weave for linen
    and cotton, soft folds for silk, grain for leather; heather colour variation; seams as stitched grooves (darker,
    sunk) and wear on raised edges (lighter, smoother), all baked into the outfit's maps."""
    col = tuple(d["color"])
    m, nt, p = principled(f"M_{key}", col, d.get("rough", 0.7), d.get("sheen", 0.0), d.get("coat", 0.0),
                          d.get("spec", 0.5))
    kind = d.get("weave")
    if not kind:
        return m
    scale = d.get("scale", 900)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    pos = tc.outputs["Object"]
    mp = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(pos, mp.inputs["Vector"])
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    if kind == "twill":
        mp.inputs["Rotation"].default_value = (0, math.radians(45), math.radians(45))
        w = nt.nodes.new("ShaderNodeTexWave")
        w.wave_type = "BANDS"
        w.inputs["Scale"].default_value = 1.0
        w.inputs["Distortion"].default_value = 0.4
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
        n.inputs["Detail"].default_value = 3.0
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        h = n.outputs["Fac"]
    fib = nt.nodes.new("ShaderNodeTexNoise")          # slub and fibre: irregularity across the weave
    fib.inputs["Scale"].default_value = scale * 0.25
    fib.inputs["Detail"].default_value = 4.0
    nt.links.new(pos, fib.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")        # heather, at a few centimetres
    noise.inputs["Scale"].default_value = d.get("heather_scale", 40.0)
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(pos, noise.inputs["Vector"])
    bw = S("bake_weave", 1.0)        # the engine lays thread-scale weave over the bake: keep the baked one faint
    height = _math(nt, "MULTIPLY_ADD", h, 0.7 * bw, _math(nt, "MULTIPLY", fib.outputs["Fac"], 0.5 * bw))
    colv = _ramp(nt, noise.outputs["Fac"], 0.3, 0.7, 1.0 - d.get("vary", 0.0), 1.0 + d.get("vary", 0.0))
    colv = _math(nt, "MULTIPLY", colv, _ramp(nt, h, 0.0, 1.0, 1.0 - d.get("weave_tone", 0.12), 1.0))
    if d.get("stripe"):
        per, amt = d["stripe"]                        # vertical stripes: bands across the lateral axis
        st = nt.nodes.new("ShaderNodeTexWave")
        st.wave_type = "BANDS"
        st.bands_direction = "X"
        st.inputs["Scale"].default_value = 1.0 / per / 2.0
        st.inputs["Distortion"].default_value = 0.0
        nt.links.new(pos, st.inputs["Vector"])
        colv = _math(nt, "MULTIPLY", colv, _ramp(nt, st.outputs["Fac"], 0.8, 0.95, 1.0, 1.0 - amt))
    if seams:
        sm = seam_mask(nt, pos, seams)
        height = _math(nt, "MULTIPLY_ADD", sm, -d.get("seam_depth", 2.5), height)
        colv = _math(nt, "MULTIPLY", colv, _math(nt, "MULTIPLY_ADD", sm, -d.get("seam_dark", 0.45), 1.0))
    if d.get("wear", 0) > 0:
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        wear = _ramp(nt, geo.outputs["Pointiness"], 0.51, 0.6, 0.0, d["wear"])
        colv = _math(nt, "MULTIPLY", colv, _math(nt, "ADD", wear, 1.0))
    # tone per panel: a face attribute "tone" (offset around 0) set per garment part before the bake
    ta = nt.nodes.new("ShaderNodeAttribute")
    ta.attribute_type = "GEOMETRY"
    ta.attribute_name = "tone"
    colv = _math(nt, "MULTIPLY", colv, _math(nt, "ADD", ta.outputs["Fac"], 1.0))
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Color"].default_value = (*col, 1)
    nt.links.new(colv, hsv.inputs["Value"])
    nt.links.new(hsv.outputs["Color"], p.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = d.get("bump", 0.25)
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(height, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m


def seam_specs(F):
    """Seam lines as planes (+ half-space masks) from the body's own joints: centre back, back side seams, shoulder
    seams, armholes, sleeve under-seams (coat); centre front and side seams (waistcoat); outside seams, inseams and
    fly (trousers). Units: metres, world = object space."""
    up = Vector((0, 0, 1))
    fwd, lat = F.fwd, F.lat
    back = (F.world(0, -0.02, 0), -fwd)
    front = (F.world(0, 0.02, 0), fwd)

    def above(z):
        return (Vector((0, 0, z)), up)

    def below(z):
        return (Vector((0, 0, z)), -up)
    wool, vest, trou = [], [], []
    wool.append({"p": F.world(0, 0, 0), "n": lat, "masks": [back, below(F.neck_z - 0.025)]})
    for s in (-1, 1):
        wool.append({"p": F.world(s * 0.115, 0, 0), "n": lat,
                     "masks": [back, above(F.waist_z), below(F.armpit_z - 0.02)]})
        wool.append({"p": F.world(0, -0.018, 0), "n": fwd,
                     "masks": [above(F.armpit_z + 0.06), (F.world(s * 0.07, 0, 0), s * lat),
                               (F.world(s * 0.20, 0, 0), -s * lat)]})
        side = "l" if s == F.sgn_l else "r"
        j0, j1 = F.j[f"upperarm_{side}"], F.j[f"lowerarm_{side}"]
        ax = (j1 - j0).normalized()
        wool.append({"p": j0 + ax * 0.01, "n": ax, "masks": [(F.world(s * 0.12, 0, 0), s * lat)], "w": 0.0016})
        nfw = (fwd - ax * fwd.dot(ax)).normalized()
        n2 = ax.cross(nfw)
        n2 = n2 if n2.z > 0 else -n2
        wool.append({"p": j0, "n": nfw, "masks": [(j0 + ax * 0.05, ax), (j0, -n2)]})
        vest.append({"p": F.world(0, -0.02, 0), "n": fwd, "masks": [(F.world(s * 0.09, 0, 0), s * lat)]})
        t = F.j[f"thigh_{side}"]
        trou.append({"p": t, "n": fwd, "masks": [(t, s * lat), below(F.crotch_z + 0.08)]})
        trou.append({"p": t, "n": fwd, "masks": [(t, -s * lat), below(F.crotch_z - 0.04)]})
    vest.append({"p": F.world(0, 0, 0), "n": lat, "masks": [front], "w": 0.0018})
    trou.append({"p": F.world(0, 0, 0), "n": lat, "masks": [front, above(F.crotch_z)]})
    trou.append({"p": F.world(0, 0, 0), "n": lat, "masks": [back, above(F.crotch_z)]})
    trou.append({"p": F.world(F.sgn_l * 0.035, 0, 0), "n": lat,
                 "masks": [front, above(F.crotch_z + 0.07), below(F.waist_z - 0.03)], "w": 0.0009})
    return {"wool": wool, "wool_vest": vest, "wool_trousers": trou}


def materials(F):
    seams = seam_specs(F) if S("seams", True) else {}
    return {key: cloth(key, d, seams.get(key, [])) for key, d in S("materials", {}).items()}


# ------------------------------------------------------------------ measurements (tape-measure girths)
def local_np(F, co):
    """(N, 3) world points -> (N, 3) (lateral, forward, z) about the pelvis axis."""
    rel = co[:, :2] - np.array(F.center)
    lat = np.array(F.lat[:2])
    fwd = np.array(F.fwd[:2])
    return np.stack([rel @ lat, rel @ fwd, co[:, 2]], 1)


def world_np(F, L):
    """(N, 3) (lateral, forward, z) -> world."""
    c = np.array([F.center[0], F.center[1], 0.0])
    lat = np.array([F.lat.x, F.lat.y, 0.0])
    fwd = np.array([F.fwd.x, F.fwd.y, 0.0])
    return c + L[:, :1] * lat + L[:, 1:2] * fwd + L[:, 2:3] * np.array([0, 0, 1.0])


def get_co(ob):
    a = np.zeros(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def set_co(ob, co):
    ob.data.vertices.foreach_set("co", np.asarray(co, float).ravel())
    ob.data.update()


def hull_np(q):
    h = convex_hull_2d([(round(float(a), 4), round(float(b), 4)) for a, b in q])
    return np.array(h, float)


def ray_hull_np(h, d):
    """Distances from the origin along unit directions d (A, 2) to a convex polygon h (E, 2) around the origin."""
    a = h
    e = np.roll(h, -1, 0) - h
    den = d[:, :1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
        s = (a[None, :, 0] * d[:, 1:2] - a[None, :, 1] * d[:, :1]) / den
    ok = (np.abs(den) > 1e-12) & (t > 0) & (s >= -1e-6) & (s <= 1 + 1e-6)
    return np.where(ok, t, 0.0).max(1)


def dirs(ang):
    ang = np.asarray(ang, float)
    return np.stack([np.cos(ang), np.sin(ang)], 1)


class Girth:
    """Girths as a tape measure takes them: per height band, the convex hull of the body's cross-section, kept as
    radii along fixed rays from an axis (in the lateral/forward plane, angle 0 = lateral, pi/2 = forward). Hollows
    between muscles are bridged the way cloth bridges them. hang(): below a bulge (chest, shoulder blades, seat) cloth
    falls from it with a limited taper instead of following the body back in."""

    def __init__(self, P, zs, nang=144, band=0.012, rmax=0.34, center=(0.0, 0.0)):
        self.zs = np.asarray(zs, float)
        self.dz = float(self.zs[1] - self.zs[0])
        self.ang = np.linspace(0, 2 * np.pi, nang, endpoint=False)
        self.center = np.array(center, float)
        d = dirs(self.ang)
        R = np.zeros((len(self.zs), nang))
        for i, z in enumerate(self.zs):
            q = P[np.abs(P[:, 2] - z) < band][:, :2] - self.center
            q = q[np.hypot(q[:, 0], q[:, 1]) < rmax]
            if len(q) >= 3:
                h = hull_np(q)
                if len(h) >= 3:
                    R[i] = ray_hull_np(h, d)
        ok = np.where(R.max(1) > 0)[0]
        for i in np.where(R.max(1) <= 0)[0]:
            R[i] = R[ok[np.argmin(np.abs(ok - i))]]
        self.R = R

    def smooth(self, na=3, nz=1, iters=2):
        n = len(self.zs)
        for _ in range(iters):
            self.R = sum(np.roll(self.R, s, 1) for s in range(-na, na + 1)) / (2 * na + 1)
            if nz:
                Pd = np.pad(self.R, ((nz, nz), (0, 0)), mode="edge")
                self.R = sum(Pd[nz + s: nz + s + n] for s in range(-nz, nz + 1)) / (2 * nz + 1)
        return self

    def by_angle(self, front, side, back):
        s, c = np.sin(self.ang), np.cos(self.ang)
        return front * np.maximum(s, 0) ** 2 + back * np.maximum(-s, 0) ** 2 + side * c ** 2

    def add(self, front, side, back):
        self.R = self.R + self.by_angle(front, side, back)[None, :]
        return self

    def hang(self, front, side, back):
        """Top-down running maximum with a taper (metres of radius per metre of drop) by direction."""
        tap = self.by_angle(front, side, back) * self.dz
        for i in range(len(self.zs) - 2, -1, -1):
            self.R[i] = np.maximum(self.R[i], self.R[i + 1] - tap)
        return self

    def r(self, z, ang):
        z = np.atleast_1d(np.asarray(z, float))
        ang = np.atleast_1d(np.asarray(ang, float))
        nz, n = len(self.zs), len(self.ang)
        fz = np.clip((z - self.zs[0]) / self.dz, 0, nz - 1.000001)
        i0 = np.floor(fz).astype(int)
        tz = fz - i0
        i1 = np.minimum(i0 + 1, nz - 1)
        fa = np.mod(ang, 2 * np.pi) / (2 * np.pi) * n
        a0 = np.floor(fa).astype(int) % n
        ta = fa - np.floor(fa)
        a1 = (a0 + 1) % n
        R = self.R
        return ((R[i0, a0] * (1 - ta) + R[i0, a1] * ta) * (1 - tz) + (R[i1, a0] * (1 - ta) + R[i1, a1] * ta) * tz)


def arm_frac(ob):
    """Per vertex: share of its skin weight on arm bones (upper arm, forearm, hand)."""
    gc = [bone_class(g.name) for g in ob.vertex_groups]
    out = np.zeros(len(ob.data.vertices))
    for v in ob.data.vertices:
        tot = arm = 0.0
        for g in v.groups:
            tot += g.weight
            if gc[g.group] in ("upperarm", "lowerarm", "hand"):
                arm += g.weight
        out[v.index] = arm / tot if tot > 0 else 0.0
    return out


def ease_torso(ob, F, G, ease, z_lo, z_hi, fade=0.06, arm_limit=0.3):
    """Move torso vertices out to the girth table (+ ease, by direction): no vertex stays inside the tailored shape.
    Fades out toward the armholes (arm-weighted vertices) and above z_hi (shoulders: smoothed instead)."""
    co = get_co(ob)
    L = local_np(F, co)
    af = arm_frac(ob)
    r = np.hypot(L[:, 0], L[:, 1])
    ang = np.arctan2(L[:, 1], L[:, 0])
    s_, c_ = np.sin(ang), np.cos(ang)
    e = ease[0] * np.maximum(s_, 0) ** 2 + ease[2] * np.maximum(-s_, 0) ** 2 + ease[1] * c_ ** 2
    tgt = G.r(L[:, 2], ang) + e
    w = (np.clip((z_hi + fade - L[:, 2]) / fade, 0, 1) * np.clip((L[:, 2] - z_lo) / fade + 1, 0, 1)
         * np.clip(1 - af / arm_limit, 0, 1))
    nr = r + w * np.maximum(0.0, tgt - r)
    k = np.where(r > 1e-6, nr / np.maximum(r, 1e-6), 1.0)
    L[:, 0] *= k
    L[:, 1] *= k
    set_co(ob, world_np(F, L))
    return int((w > 0).sum())


def tube_fit(ob, joints, ease, sel, taper=0.16, ramp=0.07, station=0.015, pct=92, mix=1.0):
    """Sleeve or trouser leg as a round tube with ease: each vertex goes to the radius of the tube about the limb axis
    (a polyline through the joints): per station the girth (a high percentile of the limb's radii), never narrowing
    faster than `taper` toward the far end (cloth hangs straight), plus ease. Blends in over `ramp` from the first
    joint so the armhole stays put."""
    co = get_co(ob)
    J = [np.array(tuple(j)) for j in joints]
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return 0
    p = co[idx]
    best_d = np.full(len(p), 1e9)
    best_s = np.zeros(len(p))
    best_c = np.zeros_like(p)
    cum = 0.0
    for a, b in zip(J[:-1], J[1:]):
        ax = b - a
        Ls = np.linalg.norm(ax)
        ax = ax / Ls
        t = np.clip((p - a) @ ax, 0, Ls)
        c = a + t[:, None] * ax
        d = np.linalg.norm(p - c, axis=1)
        m = d < best_d
        best_d[m], best_s[m], best_c[m] = d[m], cum + t[m], c[m]
        cum += Ls
    k = (best_s / station).astype(int)
    nst = k.max() + 1
    R = np.zeros(nst)
    for i in range(nst):
        rr = best_d[k == i]
        R[i] = np.percentile(rr, pct) if len(rr) else 0
    for i in range(nst):
        if R[i] <= 0:
            R[i] = R[i - 1] if i else R[R > 0][0]
    R = np.convolve(np.pad(R, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    for i in range(1, nst):
        R[i] = max(R[i], R[i - 1] - taper * station)
    tgt = np.interp(best_s, (np.arange(nst) + 0.5) * station, R) + ease
    w = mix * np.clip(best_s / ramp, 0, 1)
    nr = best_d + w * (tgt - best_d)
    kk = np.where(best_d > 1e-6, nr / np.maximum(best_d, 1e-6), 1.0)
    co[idx] = best_c + (p - best_c) * kk[:, None]
    set_co(ob, co)
    return len(idx)


def boundary_snap(ob, F, fn):
    """Open-edge vertices for which fn(lat, fwd, z) returns a new lateral value get it (clean straight edges)."""
    bm = bm_of(ob)
    n = 0
    for v in bm.verts:
        if not v.is_boundary:
            continue
        lat, fw, z = F.local(v.co)
        nl = fn(lat, fw, z)
        if nl is not None:
            v.co = F.world(nl, fw, z)
            n += 1
    bm.to_mesh(ob.data)
    bm.free()
    return n


def orient_out(ob, F, axis_z=None):
    """Recalculate normals, then flip the whole mesh if most face normals point toward the body axis."""
    bm = bm_of(ob)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    acc = 0.0
    for f in bm.faces:
        c = f.calc_center_median()
        lat, fw, z = F.local(c)
        out = F.lat * lat + F.fwd * fw
        acc += f.normal.dot(out) * f.calc_area()
    if acc < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()


def mesh_from_grid(name, rows, closed_u=False, closed_v=False):
    """rows: list of rows of 3D points (same length); quads between neighbours."""
    nr, nc = len(rows), len(rows[0])
    verts = [tuple(p) for row in rows for p in row]
    faces = []
    for i in range(nr - 1 + (1 if closed_v else 0)):
        i2 = (i + 1) % nr
        for j in range(nc - 1 + (1 if closed_u else 0)):
            j2 = (j + 1) % nc
            faces.append((i * nc + j, i * nc + j2, i2 * nc + j2, i2 * nc + j))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    ob = new_obj(name, me)
    bm = bm_of(ob)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.calc_area() < 1e-10], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def surface_point(bvh, F, lat, z, fwd0=0.6, fallback=None):
    """First hit on a surface seen from the front at (lat, z): (point, outward normal) or None."""
    o = F.world(lat, fwd0, z)
    h = bvh.ray_cast(o, -F.fwd, 2.0)
    if h[0] is None:
        return None
    n = h[1] if h[1].dot(F.fwd) >= 0 else -h[1]
    return h[0], n


# ------------------------------------------------------------------ drape (contact solver, see drape_run.py)
class DrapeExported(Exception):
    """Raised after a layer's drape input is written: the solver runs, then the build is re-run with its result."""


def panels_of(ob, F, panel_fn, inset=0.0015):
    """Split a garment along its seams into panels (panel_fn(lat, fwd, z) -> id per face): every vertex shared by n
    panels becomes n vertices, joined by n-1 stitch pairs. Returns (V, tris, stitches, copies[orig] -> [new])."""
    me = ob.data
    co = [ob.matrix_world @ v.co for v in me.vertices]
    idx, V, T = {}, [], []
    copies = defaultdict(list)
    pa = me.attributes.get("panel") or me.attributes.new("panel", "INT", "FACE")
    for poly in me.polygons:
        pid = panel_fn(*F.local(ob.matrix_world @ poly.center))
        pa.data[poly.index].value = pid
        vs = []
        for vi in poly.vertices:
            k = (vi, pid)
            if k not in idx:
                idx[k] = len(V)
                V.append(tuple(co[vi]))
                copies[vi].append(idx[k])
            vs.append(idx[k])
        for i in range(1, len(vs) - 1):
            T.append((vs[0], vs[i], vs[i + 1]))
    S = [(cs[0], c) for cs in copies.values() for c in cs[1:]]
    V = np.array(V)
    T = np.array(T, np.int32)
    # seam copies touch edge to edge, which the solver counts as intersecting: pull each copy a little into its
    # own panel (toward the centre of its faces there); the stitches close the seam again
    seamv = np.array([c for cs in copies.values() if len(cs) > 1 for c in cs], int)
    if len(seamv):
        acc = np.zeros_like(V)
        cnt = np.zeros(len(V))
        cen = V[T].mean(1)
        for k in range(3):
            np.add.at(acc, T[:, k], cen)
            np.add.at(cnt, T[:, k], 1)
        d = acc[seamv] / np.maximum(cnt[seamv], 1)[:, None] - V[seamv]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        V[seamv] += d / np.maximum(n, 1e-9) * np.minimum(n, inset)
    return V, T, np.array(S, np.int64).reshape(-1, 2), copies


def tri_soup(objs, keep=None):
    V, T = [], []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        off = len(V)
        V += [tuple(o.matrix_world @ v.co) for v in me.vertices]
        for p in me.polygons:
            vs = [off + i for i in p.vertices]
            if keep is not None and not any(keep(V[i]) for i in vs):
                continue
            for i in range(1, len(vs) - 1):
                T.append((vs[0], vs[i], vs[i + 1]))
        oe.to_mesh_clear()
    V = np.array(V)
    T = np.array(T, np.int32)
    used = np.unique(T)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return V[used], remap[T]


def drape_layer(key, ob, F, colliders, panel_fn, pin_fn, collide_keep=None):
    """Drape hook for one layer. Export pass: write the panels, stitches, pins and the fixed collider (body and the
    settled layers under it) and stop. Result pass: put the settled positions back (each original vertex at the mean
    of its panel copies, i.e. the seam closed)."""
    dj = JOB.get("drape") or {}
    if key not in dj.get("parts", []):
        return False
    V, T, Sp, copies = panels_of(ob, F, panel_fn)
    res = (dj.get("result") or {}).get(key)
    if res:
        R = np.load(res)["V"]
        co = get_co(ob)
        for vi, cs in copies.items():
            co[vi] = R[cs].mean(0)
        set_co(ob, co)
        REPORT.setdefault("drape", {})[key] = {"verts": len(V), "stitches": len(Sp),
                                                "seconds": float(np.load(res)["seconds"])}
        return True
    if dj.get("export") and dj.get("export_part") == key:
        CV, CF = tri_soup(colliders, collide_keep)
        pin = [c for vi, cs in copies.items() if pin_fn(*F.local(ob.matrix_world @ ob.data.vertices[vi].co))
               for c in cs]
        params = dict(dj.get("params", {}).get("default", {}), **dj.get("params", {}).get(key, {}))
        params["name"] = f"outfit_{key}"
        np.savez(dj["export"], V=V, F=T, S=Sp, pin=np.array(pin, np.int64), CV=CV, CF=CF,
                 params=json.dumps(params))
        REPORT["drape_export"] = {"part": key, "verts": len(V), "tris": len(T), "stitches": len(Sp),
                                  "pins": len(pin), "collider_verts": len(CV)}
        raise DrapeExported(key)
    return False


# ------------------------------------------------------------------ garments
def build(arm, body, head):
    F = Frame(arm)
    cls, frac = classify(body)
    co = [body.matrix_world @ v.co for v in body.data.vertices]
    loc = [F.local(p) for p in co]
    CO = np.array([tuple(p) for p in co])
    P = local_np(F, CO)
    C = np.array(cls)
    AF = np.array([f.get("upperarm", 0) + f.get("lowerarm", 0) + f.get("hand", 0) for f in frac])
    under = [body] + ([head] if head else [])
    parts = {}
    sgn_l = 1 if F.local(F.j["thigh_l"])[0] > 0 else -1
    sgn_r = -sgn_l
    F.sgn_l, F.sgn_r = sgn_l, sgn_r

    # measurements
    arm_pts = P[(C == "upperarm") & (np.abs(P[:, 0]) < 0.22)]
    F.armpit_z = float(arm_pts[:, 2].min()) if len(arm_pts) else F.shoulder_z - 0.1
    mid = (np.abs(P[:, 0]) < 0.012) & (P[:, 2] > F.knee_z) & (P[:, 2] < F.pelvis_z)
    F.crotch_z = float(P[mid, 2].min())
    torso_pts = P[(C == "torso") & (AF < 0.3)]
    zs = np.arange(F.crotch_z - 0.02, F.armpit_z + 0.08, 0.01)
    G0 = Girth(torso_pts, zs, nang=144).smooth(na=2, nz=1, iters=2)
    REPORT["measure"] = {"armpit_z": round(F.armpit_z, 3), "crotch_z": round(F.crotch_z, 3),
                         "waist_girth": round(float(np.mean(G0.r(np.full(144, F.waist_z), G0.ang)) * 2 * np.pi), 3),
                         "chest_girth": round(float(np.mean(G0.r(np.full(144, F.armpit_z - 0.03), G0.ang))
                                                    * 2 * np.pi), 3)}
    M = materials(F)

    def girth(hang, add=(0, 0, 0)):
        g = Girth.__new__(Girth)
        g.zs, g.dz, g.ang, g.center, g.R = G0.zs, G0.dz, G0.ang, G0.center, G0.R.copy()
        return g.add(*add).hang(*hang)

    def wrist_t(i, side_hand, cut):
        h = F.j[side_hand]
        e = F.j[side_hand.replace("hand", "lowerarm")]
        ax = (h - e).normalized()
        return (co[i] - h).dot(ax) + cut

    def side(i):
        return "l" if loc[i][0] * sgn_l > 0 else "r"

    def arm_axis(s_):
        return [F.j[f"upperarm_{s_}"], F.j[f"lowerarm_{s_}"], F.j[f"hand_{s_}"]]

    def fit_sleeves(ob, ease, taper):
        af = arm_frac(ob)
        L = local_np(F, get_co(ob))
        for s_ in ("l", "r"):
            sg = sgn_l if s_ == "l" else sgn_r
            sel = (af > 0.5) & (L[:, 0] * sg > 0.12)
            tube_fit(ob, arm_axis(s_), ease, sel, taper=taper)

    # ---- shirt: torso + arms to the wrist (tailored: girths + ease), collar band up the neck
    sh = S("shirt", {})
    keep = []
    for i, c in enumerate(cls):
        k = c in ("torso", "upperarm", "lowerarm") and co[i].z > F.waist_z - sh.get("tuck_below_waist", 0.07)
        if c == "neck" and co[i].z < F.neck_z + sh.get("neck_rise", 0.01):
            k = True
        if c == "lowerarm" and wrist_t(i, f"hand_{side(i)}", sh.get("cuff_past_wrist", -0.01)) > 0:
            k = False
        keep.append(k)
    shirt = body_subset("shirt", body, keep)
    smooth_edges(shirt)
    smooth(shirt, 6)
    fit_sleeves(shirt, sh.get("sleeve_ease", 0.008), 0.12)
    ease_torso(shirt, F, girth((0.5, 0.8, 0.8)), (0.004, 0.004, 0.004), F.waist_z - 0.1, F.armpit_z)
    smooth(shirt, 3)
    fit_layer(shirt, under, sh.get("gap", 0.004))

    def near_wrist(p):
        return any((p - F.j[f"hand_{s_}"]).length < 0.09 for s_ in ("l", "r"))
    REPORT["tidy_cuff"] = tidy_edges(shirt, near_wrist, iters=20)
    if head is not None:
        # the body has an opening at the neck and upper chest that the head mesh fills: cover it with the head
        # mesh's own faces below the collar line
        hz = F.neck_z + sh.get("neck_rise", 0.01)
        hkeep = [(head.matrix_world @ v.co).z < hz for v in head.data.vertices]
        if any(hkeep):
            patch = body_subset("shirt_chest", head, hkeep)
            smooth_edges(patch)
            smooth(patch, 6)
            fit_layer(patch, under + [shirt], sh.get("gap", 0.004))
            shirt = weld(join([shirt, patch], "shirt"))
    parts["shirt"] = shirt
    if sh.get("cuffs", True):
        parts["shirt_cuffs"] = shirt_cuffs(F, co, cls, sh)

    # ---- standing collar: a smooth cone around the neck (not the neck's contour), rounded top, open at the front
    collar, cinfo = stand_collar(F, under + [shirt], F.neck_z - sh.get("collar_below_neck", 0.03), sh)
    parts["collar"] = collar

    # ---- shoes
    so = S("shoes", {})
    objs = []
    for s_ in ("l", "r"):
        sgn = 1 if F.local(F.j[f"foot_{s_}"])[0] > 0 else -1
        pts = [co[i] for i, c in enumerate(cls)
               if (c == "foot" or (c == "calf" and co[i].z < F.ankle_z + so.get("top_above_ankle", 0.03)))
               and loc[i][0] * sgn > 0]
        sh_ob = shoe(F, pts, s_, so)
        objs.append(sh_ob)
        objs.extend(sole_and_heel(F, sh_ob, s_, so))
    parts["shoes"] = join(objs, "shoes")

    # ---- trousers: lofted from girths, two legs and a seat with a dropped crotch (no wrap of the body)
    tr = S("trousers", {})
    trousers = trousers_loft(F, P, C, tr)
    smooth(trousers, tr.get("smooth", 4), 0.5)
    # not pushed off the shoes: that folds the hem rings (the leg rings already hold the foot points + ease)
    fit_layer(trousers, under + [shirt], tr.get("gap", 0.006), rounds=2, smooth_iters=2)
    hip_fwd = F.local(F.j["thigh_l"])[1]
    t_top = F.waist_z + tr.get("rise_above_waist", 0.01)
    # panels: front and back leg per side (side seam and inseam on the plane through the hip joints, centre seams
    # on the middle plane); the waistband is held while the legs settle
    drape_layer("trousers", trousers, F, [body, parts["shoes"]],
                lambda lat, fw, z: (2 if lat >= 0 else 0) + (1 if fw > hip_fwd else 0),
                lambda lat, fw, z: z > t_top - 0.012,
                collide_keep=lambda p: p[2] < F.waist_z + 0.12)
    REPORT["tidy_hem"] = tidy_edges(trousers, lambda p: p.z < F.ankle_z + 0.06, iters=16)
    parts["trousers"] = trousers

    # ---- waistcoat
    wc = S("waistcoat", {})
    vz = F.neck_z - wc.get("v_depth", 0.17)
    F.wc_vz = vz
    vhalf = wc.get("v_half_top", 0.075)

    def v_half(z):
        return vhalf * (z - vz) / max(1e-3, F.neck_z - vz)

    def wc_bottom(lat, front):
        b = F.waist_z - wc.get("hem_below_waist", 0.03)
        if not front:
            return b
        lp = wc.get("point_lat", 0.03)
        a = abs(lat)
        if a < lp:      # the notch between the two points
            return b - wc.get("point_drop", 0.07) + wc.get("notch_rise", 0.035) * (1 - a / lp)
        return b - max(0.0, wc.get("point_drop", 0.07) - wc.get("point_slope", 0.45) * (a - lp))
    keep = []
    for i, c in enumerate(cls):
        lat, fw, z = loc[i]
        if c != "torso" or (frac[i].get("upperarm", 0) > wc.get("armhole_frac", 0.35)
                            and abs(lat) > wc.get("armhole_lat", 0.10)):
            keep.append(False)
            continue
        front = fw > 0
        k = wc_bottom(lat, front) < z < F.neck_z - 0.005
        if front and z > vz and abs(lat) < v_half(z):
            k = False
        keep.append(k)
    waistcoat = body_subset("waistcoat", body, keep)
    smooth_edges(waistcoat)
    smooth(waistcoat, 8)
    ease_torso(waistcoat, F, girth(wc.get("hang", (0.12, 0.5, 0.35))), wc.get("ease", (0.012, 0.010, 0.010)),
               F.waist_z - 0.15, F.armpit_z)
    smooth(waistcoat, 3)
    fit_layer(waistcoat, under + [shirt, trousers], wc.get("gap", 0.004))
    boundary_snap(waistcoat, F, lambda lat, fw, z: (math.copysign(v_half(z), lat)
                                                    if fw > 0 and vz + 0.01 < z < F.neck_z - 0.02
                                                    and abs(lat) < v_half(z) + 0.03 else None))
    parts["waistcoat"] = waistcoat
    wcb = collider([waistcoat])
    parts["wc_lapels"] = edge_strip(F, waistcoat, lambda z: v_half(z) if z > vz else None,
                                    wc.get("lapel_w", 0.024), wc.get("lapel_proud", 0.003), "wc_lapels",
                                    z_lo=vz + 0.01, z_hi=F.neck_z - 0.03)
    parts["wc_buttons"] = buttons_line(F, waistcoat, vz - 0.012, F.waist_z - 0.05, wc.get("buttons", 6),
                                       wc.get("button_r", 0.0055), "wc_buttons")
    parts["wc_welts"] = welts(F, wcb, [(s * wc.get("welt_lat", 0.095), F.waist_z - 0.0, s * 0.12)
                                       for s in (-1, 1)], wc.get("welt_w", 0.075), 0.012, "wc_welts")

    # ---- cravat: a band wound over the collar, a knot with volume, two full ends on the shirt front
    parts["cravat"] = cravat_tied(F, cinfo, shirt, S("cravat", {}))

    # ---- coat body + sleeves
    ct = S("coat", {})
    e0 = ct.get("edge_lat", 0.05)
    nh = ct.get("neck_half", 0.075)
    zb = F.waist_z + ct.get("break_above_waist", 0.08)
    zn = F.neck_z - ct.get("neck_point_below", 0.02)

    def open_half(z):
        if z <= zb:
            return e0
        if z >= zn:
            return nh
        return e0 + (nh - e0) * (z - zb) / (zn - zb)
    F.open_half = open_half
    keep = []
    for i, c in enumerate(cls):
        lat, fw, z = loc[i]
        k = c in ("torso", "upperarm", "lowerarm") and z > F.waist_z - 0.02
        if c == "lowerarm" and wrist_t(i, f"hand_{side(i)}", ct.get("sleeve_short_of_wrist", -0.025)) > 0:
            k = False
        if c == "neck" and z < F.neck_z + ct.get("back_neck_rise", 0.015) and (fw < 0.02 or abs(lat) > nh):
            k = True
        if c == "torso" and z > F.neck_z + ct.get("back_neck_rise", 0.015):
            k = False
        if k and fw > 0 and c in ("torso", "neck") and abs(lat) < open_half(z):
            k = False
        keep.append(k)
    coat = body_subset("coat", body, keep)
    if head is not None:
        # the body is open at the neck and upper chest (the head mesh fills it): fill the coat there from the head
        hk = []
        for v in head.data.vertices:
            lat, fw, z = F.local(head.matrix_world @ v.co)
            hk.append(z < F.neck_z + ct.get("back_neck_rise", 0.015) and not (fw > 0 and abs(lat) < open_half(z))
                      and (fw < 0.02 or abs(lat) > nh or z < F.neck_z - 0.01))
        if any(hk):
            coat = weld(join([coat, body_subset("coat_chest", head, hk)], "coat"))
    smooth_edges(coat)
    smooth(coat, ct.get("smooth", 12))
    fit_sleeves(coat, ct.get("sleeve_ease", 0.022), ct.get("sleeve_taper", 0.16))
    ease_torso(coat, F, girth(ct.get("hang", (0.08, 0.3, 0.15))), ct.get("ease", (0.030, 0.026, 0.026)),
               F.waist_z - 0.1, F.armpit_z)
    smooth(coat, 4)
    shoulder_pad(coat, F, ct.get("shoulder_pad", 0.0), ct.get("shoulder_pad_r", 0.11))
    fit_layer(coat, under + [shirt, collar, waistcoat, parts["wc_buttons"], parts["wc_lapels"], parts["cravat"]],
              ct.get("gap", 0.006))
    boundary_snap(coat, F, lambda lat, fw, z: (math.copysign(open_half(z), lat)
                                               if fw > 0 and z < zn + 0.005 and abs(lat) < open_half(z) + 0.04
                                               else None))
    if ct.get("sleeve_end_t") is not None:
        REPORT["sleeve_extended"] = extend_sleeves(coat, F, ct["sleeve_end_t"])
    cuffs(F, coat, ct)
    parts["coat"] = coat
    cb = collider([coat])
    lap, lap_info = lapels_built(F, cb, ct, open_half, e0, zb, zn)
    parts["lapels"] = lap
    parts["coat_collar"] = coat_collar(F, under + [collar, shirt], ct, lap_info, collider([coat, lap]))
    legs = [co[i] for i, c in enumerate(cls) if c in ("torso", "thigh", "calf")]
    parts["skirt"] = skirt_built(F, legs, under + [trousers, waistcoat], coat, ct, e0)
    cbs = collider([coat, parts["skirt"]])
    parts["coat_buttons"] = coat_buttons(F, cbs, ct, e0, zb)
    parts["buttonholes"] = buttonholes(F, cbs, ct, e0, zb)
    parts["welts"] = welts(F, cb, [(sgn_l * ct.get("breast_welt_lat", 0.11), F.armpit_z - 0.02,
                                    -sgn_l * 0.1)], ct.get("breast_welt_w", 0.10), 0.02, "welts")

    # faces no camera can see (under the coat and waistcoat) go: they can only poke through
    def shirt_seen(lat, fw, z, p):
        if z > F.neck_z - 0.04:
            return True
        if fw > 0 and z > vz - 0.03 and abs(lat) < v_half(max(z, vz)) + 0.035:
            return True
        for s_ in ("l", "r"):
            h, e = F.j[f"hand_{s_}"], F.j[f"lowerarm_{s_}"]
            if (p - h).dot((h - e).normalized()) > -0.1 and (p - h).length < 0.2:
                return True
        return False
    REPORT["culled"] = {"shirt": cull(parts["shirt"], F, shirt_seen),
                        "waistcoat": cull(parts["waistcoat"], F, lambda lat, fw, z, p: fw > -0.02)}
    for key, th in S("thickness", {}).items():
        if key in parts:
            solidify(parts[key], th)
    cbody = collider(under)
    loose = {"skirt": 0.3, "flaps": 0.3, "coat_buttons": 0.3, "buttonholes": 0.3, "coat": 0.16, "lapels": 0.16,
             "coat_collar": 0.16}
    REPORT["strays_fixed"] = {k: fix_strays(o, cbody, loose.get(k, 0.12)) for k, o in parts.items()}
    for key, ob in parts.items():
        set_mat(ob, M[S("material_of", {}).get(key, "wool")])
        if key not in S("flat_parts", ["buttonholes", "welts", "wc_welts"]):
            ob.data.shade_smooth()
    REPORT["parts"] = {k: len(o.data.vertices) for k, o in parts.items()}
    return F, parts


def weld(ob, dist=2e-4):
    bm = bm_of(ob)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def cull(ob, F, seen):
    """Delete faces whose centre fails seen(lat, fwd, z, point)."""
    bm = bm_of(ob)
    kill = []
    for f in bm.faces:
        c = f.calc_center_median()
        lat, fw, z = F.local(c)
        if not seen(lat, fw, z, c):
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()
    return len(kill)


def trousers_loft(F, P, C, tr):
    """Trousers from girths, not from the body's faces. Above a crotch dropped below the body's, both legs share one
    seat ring per height (hull of hips and thighs, + ease, hanging from the seat and the front); below it each leg is
    a ring about its own axis (hull of that leg, + ease) that hangs nearly straight to the hem. The rings of each leg
    run front-centre -> outside -> back-centre (outer arc) and back along the inside (inner arc); above the crotch the
    inner arcs lie on the centre plane and are dropped, and the two legs weld there, so the crotch is a gusset clear
    of the body."""
    A = tr.get("nang", 64)
    M = A // 2
    K = tr.get("inner", 10)
    is_leg = np.isin(C, ["thigh", "calf", "foot"])
    zc = F.crotch_z - tr.get("crotch_drop", 0.035)
    z_top = F.waist_z + tr.get("rise_above_waist", 0.01)
    z_hem = F.ankle_z + tr.get("hem_above_ankle", -0.01)
    zs_u = np.arange(zc - 0.01, z_top + 0.02, 0.01)
    Gu = Girth(P[(C == "torso") | is_leg], zs_u, nang=A, rmax=0.36).smooth(na=1, nz=1, iters=2)
    Gu.add(*tr.get("seat_ease", (0.016, 0.012, 0.02))).hang(*tr.get("seat_hang", (0.3, 0.6, 0.15)))
    up_z = list(np.linspace(zc, z_top, max(3, int((z_top - zc) / 0.025) + 1)))
    n_lo = max(4, int((zc - z_hem) / 0.025))
    lo_z = list(np.linspace(z_hem, zc, n_lo + 1))[:-1]
    i_front, i_back = A // 4, 3 * A // 4

    def upper_ring(z, sgn):
        ang_i = [(i_front - sgn * j) % A for j in range(M + 1)]
        r = Gu.r(np.full(len(ang_i), z), Gu.ang[ang_i])
        pts = [[r[j] * math.cos(Gu.ang[a]), r[j] * math.sin(Gu.ang[a]), z] for j, a in enumerate(ang_i)]
        pts[0][0] = pts[-1][0] = 0.0
        f0, b0 = pts[0][1], pts[-1][1]
        inner = [[0.0, b0 + (f0 - b0) * k / K, z] for k in range(1, K)]
        return pts + inner, (f0, b0)

    legs = []
    for sgn in (1, -1):
        ring_c, (f0, b0) = upper_ring(zc, sgn)
        # leg rings: hull of this leg's points per height, about the leg's own centre
        lp = P[is_leg & (P[:, 0] * sgn > 0.0)]
        cents, hulls = [], []
        for z in lo_z:
            q = lp[np.abs(lp[:, 2] - z) < 0.012][:, :2]
            h = hull_np(q) if len(q) >= 3 else None
            hulls.append(h)
            cents.append(h.mean(0) if h is not None and len(h) >= 3 else None)
        for i in range(len(cents)):
            if cents[i] is None:
                j = min((j for j in range(len(cents)) if cents[j] is not None), key=lambda j: abs(j - i))
                cents[i], hulls[i] = cents[j], hulls[j]
        cents = np.array(cents)
        cents = np.array([np.convolve(np.pad(cents[:, d], 2, mode="edge"), np.ones(5) / 5, mode="valid")
                          for d in (0, 1)]).T
        c_top = cents[-1]
        th_f = math.atan2(f0 - c_top[1], 0.0 - c_top[0])
        th_b = math.atan2(b0 - c_top[1], 0.0 - c_top[0])
        dirn = -sgn
        span_o = (dirn * (th_b - th_f)) % (2 * math.pi)
        thetas = [th_f + dirn * span_o * j / M for j in range(M + 1)]
        thetas += [th_b + dirn * (2 * math.pi - span_o) * k / K for k in range(1, K)]
        D = dirs(thetas)
        Rl = np.array([ray_hull_np(h - c, D) for h, c in zip(hulls, cents)])
        z_arr = np.array(lo_z)
        ease = np.interp(z_arr, [z_hem, F.knee_z, zc], tr.get("leg_ease", [0.03, 0.026, 0.02]))
        Rl = Rl + ease[:, None]
        tap = tr.get("leg_taper", 0.06) * (lo_z[1] - lo_z[0])
        for i in range(len(lo_z) - 2, -1, -1):          # top-down: the leg hangs from the thigh
            Rl[i] = np.maximum(Rl[i], Rl[i + 1] - tap)
        rows = []
        for i, z in enumerate(lo_z):
            zz = z
            pts = []
            for j, th in enumerate(thetas):
                x = cents[i][0] + Rl[i, j] * math.cos(th)
                y = cents[i][1] + Rl[i, j] * math.sin(th)
                if i == 0:     # hem: higher at the front (resting on the instep), lower at the heel
                    zz = z + tr.get("hem_tilt", 0.012) * math.sin(th)
                pts.append([x, y, zz])
            rows.append(pts)
        rows.append(ring_c)
        for z in up_z[1:]:
            rows.append(upper_ring(z, sgn)[0])
        N = M + K
        verts = [tuple(world_np(F, np.array([p]))[0]) for row in rows for p in row]
        faces = []
        n_lo_rows = len(lo_z)
        for i in range(len(rows) - 1):
            for j in range(N):
                if i >= n_lo_rows and j >= M:
                    continue                      # inner arc above the crotch lies on the centre plane: no faces
                j2 = (j + 1) % N
                faces.append((i * N + j, i * N + j2, (i + 1) * N + j2, (i + 1) * N + j))
        me = bpy.data.meshes.new(f"trouser_{sgn}")
        me.from_pydata(verts, [], faces)
        legs.append(new_obj(f"trouser_{sgn}", me))
    ob = join(legs, "trousers")
    bm = bm_of(ob)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob["z_crotch"] = zc
    orient_out(ob, F)
    return ob


def neck_sections(F, under, z0, z1, nslice=10, nang=96):
    """The neck as a tape takes it, slice by slice from z0 to z1: per slice the convex hull of the head/body points
    round the neck axis, as radii along fixed rays, and its centre. Slices that the jaw, chin or skull reach (girth
    more than 8% above the neck's narrowest) are not neck: they take the last clean slice's radii and the clean
    slices' centre line carried on, so a collar built on them follows the neck, not the jaw."""
    pts = []
    for o in under:
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co
            if z0 - 0.012 < p.z < z1 + 0.012:
                pts.append((p.x, p.y, p.z))
    Pn = np.array(pts)
    ang = np.linspace(0, 2 * np.pi, nang, endpoint=False)
    D = dirs(ang)
    zs = np.linspace(z0, z1, nslice)
    ax = np.array([F.j["neck_01"].x, F.j["neck_01"].y])
    if "neck_02" in F.j:
        ax2 = np.array([F.j["neck_02"].x, F.j["neck_02"].y])
        ax2z = F.j["neck_02"].z
    else:
        ax2, ax2z = ax, F.neck_z + 0.05
    Rs, Cs, G = [], [], []
    for z in zs:
        t = (z - F.neck_z) / max(1e-3, ax2z - F.neck_z)
        c = ax + (ax2 - ax) * t
        b = Pn[np.abs(Pn[:, 2] - z) < 0.006][:, :2] - c
        b = b[np.hypot(b[:, 0], b[:, 1]) < 0.095]
        if len(b) < 8:
            Rs.append(None)
            Cs.append(c)
            G.append(np.inf)
            continue
        h = hull_np(b)
        c2 = c + h.mean(0)
        R = ray_hull_np(h - h.mean(0), D)
        Rs.append(R)
        Cs.append(c2)
        G.append(float(R.mean() * 2 * np.pi))
    G = np.array(G)
    gmin = G.min()
    clean = G <= gmin * 1.08
    # the neck is clean from the bottom up to where the jaw first reaches; above that nothing is trusted
    first_bad = next((i for i in range(nslice) if not clean[i] and i > int(np.argmin(G))), nslice)
    ok = [i for i in range(first_bad) if clean[i] and Rs[i] is not None]
    if not ok:
        ok = [int(np.argmin(G))]
    Cs = np.array(Cs)
    if len(ok) >= 2:
        kx = np.polyfit(zs[ok], Cs[ok, 0], 1)
        ky = np.polyfit(zs[ok], Cs[ok, 1], 1)
        Cl = np.stack([np.polyval(kx, zs), np.polyval(ky, zs)], 1)
    else:
        Cl = np.repeat(Cs[ok[0]][None], nslice, 0)
    Rl = []
    last = Rs[ok[0]]
    for i in range(nslice):
        if i in ok:
            last = Rs[i]
        Rl.append(last if i >= ok[0] else Rs[ok[0]])
    Rl = np.array(Rl)
    for _ in range(3):
        Rl = sum(np.roll(Rl, k, 1) for k in range(-3, 4)) / 7
    REPORT["neck"] = {"girth_cm": [round(float(g) * 100, 1) if np.isfinite(g) else None for g in G],
                      "z": [round(float(z), 3) for z in zs], "clean_to": round(float(zs[ok[-1]]), 3),
                      "girth_used_cm": round(float(Rl[ok[-1]].mean() * 2 * np.pi * 100), 1)}
    return zs, Cl, Rl, ang, D


def stand_collar(F, under, z0, sh, nang=96):
    """Standing collar built from the neck's own measured sections (neck_sections: hull radii per direction and the
    neck's centre line, the jaw left out), + a small gap, so it sits close all round; taller at the back than the
    front, a rolled top edge, open at the front, with the front corners turned down as wing points. Starched: it
    does not turn with the head."""
    hb, hf = sh.get("collar_h_back", 0.07), sh.get("collar_h_front", 0.055)
    gap, flare = sh.get("collar_gap", 0.004), sh.get("collar_flare", 0.002)
    open_deg = sh.get("collar_open_deg", 10)
    zs, Cl, Rl, ang, D = neck_sections(F, under, z0, z0 + hb, nang=nang)

    def at(z):
        i = float(np.clip((z - zs[0]) / max(1e-6, zs[-1] - zs[0]) * (len(zs) - 1), 0, len(zs) - 1))
        i0 = min(int(i), len(zs) - 2)
        f = i - i0
        return Cl[i0] * (1 - f) + Cl[i0 + 1] * f, Rl[i0] * (1 - f) + Rl[i0 + 1] * f

    front = math.atan2(F.fwd.y, F.fwd.x)
    df = np.abs((ang - front + np.pi) % (2 * np.pi) - np.pi)
    keep = df > math.radians(open_deg)
    fdeg = math.radians(open_deg)
    top = z0 + hb + (hf - hb) * ((1 + np.cos(df)) / 2) ** 2
    order = [a for a in range(nang) if keep[a]]
    a_start = int(np.argmin(np.where(keep, df, 9)))
    order = sorted(order, key=lambda a: (ang[a] - ang[a_start]) % (2 * np.pi))
    wing_deg = math.radians(sh.get("wing_deg", 26))
    wing_h = sh.get("wing_h", 0.42)          # share of the collar height folded down at the front corners
    rows = []
    nring = 12

    def ring_pt(a, z, extra):
        c, R = at(z)
        r = R[a] + gap + extra
        return (c[0] + r * D[a, 0], c[1] + r * D[a, 1], z)

    for k in range(nring + 1):
        t = k / nring
        row = []
        for a in order:
            # the band under a wing stops at the fold line
            wf = max(0.0, 1 - (df[a] - fdeg) / wing_deg)
            tt = min(t, 1 - wing_h * min(1.0, 1.6 * wf)) if wf > 0 else t
            row.append(ring_pt(a, z0 + tt * (top[a] - z0), flare * tt))
        rows.append(row)
    for dr, dz in ((0.0014, 0.001), (0.0028, -0.0005)):      # rolled top edge (behind the wings)
        row = []
        for a in order:
            wf = max(0.0, 1 - (df[a] - fdeg) / wing_deg)
            tt = 1 - wing_h * min(1.0, 1.6 * wf) if wf > 0 else 1.0
            p = ring_pt(a, z0 + tt * (top[a] - z0) + dz, flare * tt + dr)
            row.append(p)
        rows.append(row)
    ob = mesh_from_grid("collar", rows)
    objs = [ob]
    # wing points: from the fold line, the corner folds out and down, a triangle that narrows away from the opening
    for sgn in (-1, 1):
        wrows = []
        nw, nd = 8, 6
        for i in range(nd + 1):
            d = i / nd
            row = []
            for j in range(nw + 1):
                u = j / nw
                aa = front + sgn * (fdeg + wing_deg * u * (1 - 0.75 * d))
                a = int(round((aa % (2 * np.pi)) / (2 * np.pi) * nang)) % nang
                zf = z0 + (1 - wing_h) * (top[a] - z0)
                c, R = at(zf)
                L = wing_h * (top[a] - z0) * (1 - 0.35 * u)
                # clear of the cravat band wound below the fold (wing_clear), more toward the tip
                r = R[a] + gap + flare + 0.0015 + L * d * 0.75 + sh.get("wing_clear", 0.0) * (0.4 + 0.6 * d)
                z = zf - L * d * 0.55 + 0.0015
                da = np.array([math.cos(aa), math.sin(aa)])
                row.append((c[0] + r * da[0], c[1] + r * da[1], z))
            wrows.append(row)
        objs.append(mesh_from_grid(f"collar_wing{sgn}", wrows))
    ob = join(objs, "collar")
    orient_out(ob, F)
    zb = z0 + sh.get("band_at", 0.022)
    cb, Rb = at(zb)
    info = {"c": (float(cb[0]), float(cb[1])), "R": Rb, "ang": ang, "gap": gap, "flare": flare, "z0": z0,
            "top": top, "D": D, "front": front}
    REPORT["collar"] = {"z0": round(z0, 3), "top_back": round(z0 + hb, 3), "gap": gap,
                        "girth_cm_at_band": round(float((Rb.mean() + gap) * 2 * np.pi * 100), 1)}
    return ob, info


def cravat_tied(F, ci, shirt, cv):
    """Silk cravat (an ascot): a smooth band once round the collar base, a soft knot at the throat (a rounded
    cushion with two broad folds, not a lump), and two blades that spread from the knot over the shirt front in a
    few long, soft lengthwise folds, the front blade over the back one, both ending in a clean straight edge tucked
    under the waistcoat opening."""
    cx, cy = ci["c"]
    R, ang, D = ci["R"], ci["ang"], ci["D"]
    rcol = R + ci["gap"] + ci["flare"] * 0.5
    z0 = ci["z0"]
    zc = z0 + cv.get("band_z", 0.022)
    hz, ht = cv.get("band_h", 0.017), cv.get("band_t", 0.006)
    nps = 14
    rows = []
    for a in range(len(ang)):
        row = []
        for k in range(nps):
            ps = 2 * math.pi * k / nps
            crease = cv.get("band_crease", 0.0005) * math.sin(3 * ang[a] + 2 * ps)
            rr = rcol[a] + 0.0015 + ht + ht * math.cos(ps) + crease
            row.append((cx + rr * D[a, 0], cy + rr * D[a, 1], zc + hz * math.sin(ps)))
        rows.append(row)
    band = mesh_from_grid("cravat_band", rows, closed_u=True, closed_v=True)
    objs = [band]
    front = ci["front"]
    af = int(round((front % (2 * math.pi)) / (2 * math.pi) * len(ang))) % len(ang)
    r_front = rcol[af] + 2 * ht
    kc = Vector((cx + r_front * math.cos(front), cy + r_front * math.sin(front), zc - 0.004))
    kx, ky, kz = cv.get("knot", [0.021, 0.014, 0.019])
    bmk = bmesh.new()
    bmesh.ops.create_uvsphere(bmk, u_segments=32, v_segments=18, radius=1.0)
    for v in bmk.verts:
        th = math.atan2(v.co.y, v.co.x)
        ph = math.asin(max(-1, min(1, v.co.z)))
        # a cushion: flatter at the back (against the throat), two broad diagonal folds across the front
        fold = 1 + cv.get("knot_fold", 0.06) * math.cos(2 * th + 2 * ph) * math.cos(ph)
        back = 0.55 if v.co.y > 0 else 1.0
        v.co = Vector((v.co.x * kx * fold, v.co.y * ky * fold * back, v.co.z * kz * fold))
    me = bpy.data.meshes.new("knot")
    bmk.to_mesh(me)
    bmk.free()
    knot = new_obj("cravat_knot", me)
    knot.matrix_world = Matrix.Translation(kc) @ Matrix.Rotation(math.atan2(F.fwd.x, -F.fwd.y), 4, "Z")
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    knot.select_set(True)
    bpy.context.view_layer.objects.active = knot
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    objs.append(knot)
    sb = collider([shirt])
    k_lat, _, k_z = F.local(kc)
    z_end = getattr(F, "wc_vz", k_z - 0.14) - cv.get("tuck", 0.015)
    nfold = cv.get("folds", 3)
    for n_end, sgn in enumerate((1, -1)):          # back blade first, then the front blade over it
        L = max(0.04, (k_z - 0.012) - z_end - 0.008 * n_end)
        ns, nw = 28, 16
        rows = []
        last = None
        for i in range(ns + 1):
            s_ = i / ns
            cl = k_lat + sgn * (0.003 + cv.get("end_fan", 0.012) * s_)
            z = k_z - 0.012 - L * s_
            w = cv.get("end_w0", 0.03) + (cv.get("end_w1", 0.08) - cv.get("end_w0", 0.03)) * min(1.0, s_ * 1.6) ** 0.6
            row = []
            for j in range(nw + 1):
                u = j / nw - 0.5
                lat = cl + u * w
                hit = surface_point(sb, F, lat, z)
                if hit is not None:
                    last = hit
                p, nrm = (hit if hit is not None else (last if last is not None else (F.world(lat, 0.1, z), F.fwd)))
                if hit is None and last is not None:
                    p = F.world(lat, F.local(last[0])[1], z)
                # soft lengthwise folds, deepest mid-blade, flat at the knot and at the tucked end
                env = math.sin(math.pi * min(1.0, s_ * 1.15))
                fold = cv.get("fold_amp", 0.004) * env * (0.5 + 0.5 * math.cos(2 * math.pi * nfold * u + n_end))
                bulge = cv.get("end_bulge", 0.006) * (1 - (2 * u) ** 2) * (0.5 + 0.5 * env)
                off = cv.get("end_gap", 0.006) + 0.004 * n_end + bulge + fold + 0.008 * (1 - s_) ** 3
                # the tucked end lies flat on the shirt, under the waistcoat
                tk = min(1.0, max(0.0, (z - z_end) / 0.035))
                off = 0.0022 + (off - 0.0022) * tk * tk * (3 - 2 * tk)
                row.append(tuple(p + nrm * off))
            rows.append(row)
        objs.append(mesh_from_grid(f"cravat_end{n_end}", rows))
    ob = join(objs, "cravat")
    return ob


def edge_strip(F, ob, half_fn, w, proud, name, z_lo, z_hi):
    """Faces of a front panel next to an opening edge (|lat| between half_fn(z) and +w), copied and laid proud: a
    narrow lapel or facing."""
    bm = bm_of(ob)
    keepf = set()
    for f in bm.faces:
        c = f.calc_center_median()
        lat, fw, z = F.local(c)
        if fw <= 0 or not (z_lo < z < z_hi):
            continue
        h = half_fn(z)
        if h is not None and h <= abs(lat) < h + w:
            keepf.add(f.index)
    bm.free()
    me = ob.data.copy()
    o2 = new_obj(name, me)
    bml = bm_of(o2)
    bml.faces.ensure_lookup_table()
    bmesh.ops.delete(bml, geom=[f for f in bml.faces if f.index not in keepf], context="FACES")
    bmesh.ops.delete(bml, geom=[v for v in bml.verts if not v.link_faces], context="VERTS")
    bml.normal_update()
    for v in bml.verts:
        n = v.normal if v.normal.dot(F.fwd) >= 0 else -v.normal
        v.co += n * proud
    bml.to_mesh(o2.data)
    bml.free()
    smooth(o2, 4, 0.5, pin_boundary=False)
    orient_out(o2, F)
    return o2


def lapels_built(F, cb, ct, open_half, e0, zb, zn):
    """Notch lapels as cut pieces: in the front view a polygon from the break point B up the roll line to the gorge
    G, out along the gorge to the notch Q, the lapel point P, and a bellied front edge back down to B; draped onto
    the coat front (rays from the front), lifted off it most along the roll line, where a fold strip joins it to the
    coat's front edge."""
    objs = []
    info = {}
    zG = zn - ct.get("gorge_below_neckpoint", 0.02)
    zP = F.neck_z - ct.get("notch_below_neck", 0.115)
    P_ = (open_half(zP) + ct.get("lapel_w", 0.085), zP)
    Q_ = (P_[0] - ct.get("notch_w", 0.03), zP + ct.get("notch_rise", 0.012))
    G_ = (open_half(zG), zG)
    B_ = (e0, zb)
    belly = ct.get("lapel_belly", 0.012)
    # outer path B -> P (quadratic, bellied outward) -> Q -> G
    mx, mz = (B_[0] + P_[0]) / 2, (B_[1] + P_[1]) / 2
    dx, dz = P_[0] - B_[0], P_[1] - B_[1]
    ln = math.hypot(dx, dz)
    ctrl = (mx + dz / ln * belly * 2, mz - dx / ln * belly * 2)
    path = []
    for i in range(16):
        t = i / 15
        path.append(((1 - t) ** 2 * B_[0] + 2 * t * (1 - t) * ctrl[0] + t * t * P_[0],
                     (1 - t) ** 2 * B_[1] + 2 * t * (1 - t) * ctrl[1] + t * t * P_[1]))
    for i in range(1, 4):
        t = i / 3
        path.append((P_[0] + (Q_[0] - P_[0]) * t, P_[1] + (Q_[1] - P_[1]) * t))
    for i in range(1, 7):
        t = i / 6
        path.append((Q_[0] + (G_[0] - Q_[0]) * t, Q_[1] + (G_[1] - Q_[1]) * t))
    seg = [0.0]
    for a, b in zip(path[:-1], path[1:]):
        seg.append(seg[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = seg[-1]

    def outer(u):
        s = u * total
        for i in range(len(seg) - 1):
            if seg[i + 1] >= s:
                t = (s - seg[i]) / max(1e-9, seg[i + 1] - seg[i])
                return (path[i][0] + (path[i + 1][0] - path[i][0]) * t, path[i][1] + (path[i + 1][1] - path[i][1]) * t)
        return path[-1]
    nu, nv = 28, 8
    lift0, roll = ct.get("lapel_lift", 0.004), ct.get("lapel_roll", 0.009)
    uQ = seg[16 + 2] / total
    for sgn in (F.sgn_l, F.sgn_r):
        rows = []
        for i in range(nu + 1):
            u = i / nu
            zr = B_[1] + (G_[1] - B_[1]) * u
            rl = open_half(zr)
            ol, oz = outer(u)
            row = []
            # fold strip: from the coat surface at the edge up to the lapel's rolled inner edge
            for j in range(-1, nv + 1):
                v = max(0.0, j / nv)
                lat = rl + (ol - rl) * v + (0.0 if j < 0 else 0.002)
                z = zr + (oz - zr) * v
                hit = surface_point(cb, F, sgn * lat, z)
                if hit is None:
                    hit = surface_point(cb, F, sgn * (lat + 0.01), z)
                if hit is None:
                    row.append(tuple(F.world(sgn * lat, 0.12, z)))
                    continue
                p, n = hit
                rise = min(1.0, u * 5) * min(1.0, (1 - u) * 6 + 0.3)
                off = 0.0 if j < 0 else lift0 + roll * (1 - v) ** 1.6 * rise
                row.append(tuple(p + n * off))
            rows.append(row)
        ob = mesh_from_grid("lapel", rows)
        orient_out(ob, F)
        objs.append(ob)
        # 3D points for the collar: gorge on the roll line, and the collar end just above the notch
        g = surface_point(cb, F, sgn * G_[0], G_[1])
        q = surface_point(cb, F, sgn * (Q_[0] - 0.004), Q_[1] + 0.016)
        info[sgn] = {"G": (g[0] + g[1] * lift0) if g else F.world(sgn * G_[0], 0.1, G_[1]),
                     "C": (q[0] + q[1] * (lift0 + 0.004)) if q else F.world(sgn * Q_[0], 0.12, Q_[1])}
    REPORT["lapel"] = {"B": B_, "P": P_, "Q": Q_, "G": G_, "uQ": round(uQ, 3)}
    return join(objs, "lapels"), info


def coat_collar(F, under, ct, li, surf):
    """Coat collar: from one gorge round the back of the neck to the other. Per direction a profile: a stand up the
    neck from below the coat's neckline, a rolled top, and a fall that turns down and lies on the coat (each fall
    point is the nearest point of the coat or lapel surface, lifted a few millimetres). Toward the ends the base runs
    down the neckline to the gorge and the fall ends on the lapel just above the notch."""
    nz = F.neck_z
    pts = []
    for o in under:
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co
            if nz - 0.01 < p.z < nz + 0.04:
                pts.append((p.x, p.y))
    Pn = np.array(pts)
    cx, cy = F.j["neck_01"].x, F.j["neck_01"].y
    q = Pn - (cx, cy)
    q = q[np.hypot(q[:, 0], q[:, 1]) < 0.1]
    hull = hull_np(q)
    gl, gr = li[F.sgn_l]["G"], li[F.sgn_r]["G"]
    a_l = math.atan2(gl.y - cy, gl.x - cx)
    a_r = math.atan2(gr.y - cy, gr.x - cx)
    back = math.atan2(-F.fwd.y, -F.fwd.x)
    span = (a_r - a_l) % (2 * math.pi)
    if abs(((a_l + span / 2) - back + math.pi) % (2 * math.pi) - math.pi) > math.pi / 2:
        span = span - 2 * math.pi
    n = 56
    hb, he = ct.get("collar_stand", 0.028), ct.get("collar_stand_end", 0.008)
    fall_b, fall_e = ct.get("collar_fall", 0.04), ct.get("collar_fall_end", 0.03)
    gap = ct.get("collar_gap", 0.012)
    lift = ct.get("collar_lift", 0.004)
    c0 = Vector((cx, cy, 0))
    cols = []
    for j in range(n + 1):
        t = j / n
        a = a_l + span * t
        e = min(t, 1 - t) * 2                    # 0 at the ends, 1 at the back
        sm = min(1.0, e / 0.5)
        sm = sm * sm * (3 - 2 * sm)
        d = Vector((math.cos(a), math.sin(a), 0))
        end = li[F.sgn_l] if t < 0.5 else li[F.sgn_r]
        rb = float(ray_hull_np(hull, dirs([a]))[0]) + gap
        ring_base = c0 + d * rb + Vector((0, 0, nz - 0.006))
        base = end["G"].lerp(ring_base, sm)
        h = he + (hb - he) * sm
        fall = fall_e + (fall_b - fall_e) * sm
        top = base + Vector((0, 0, h)) + d * 0.002
        roll = top + d * 0.007 + Vector((0, 0, 0.003))
        col = [base, base.lerp(top, 0.5), top, roll]
        for k, f in enumerate((0.35, 0.7, 1.0)):
            want = top + d * (0.012 + 0.022 * f) - Vector((0, 0, fall * f))
            if sm < 1.0:
                want = want.lerp(end["C"], (1 - sm) * f)
            # land it from outside: a ray from beyond the coat toward the point meets the coat's outer face
            src = want + d * 0.12 + Vector((0, 0, 0.06))
            dv = (want - src).normalized()
            hit = surf.ray_cast(src, dv, 0.3)
            if hit[0] is not None:
                nrm = hit[1] if hit[1].dot(dv) < 0 else -hit[1]
                p = hit[0] + nrm * (lift + 0.003 * (1 - f))
                if k == 0:                         # the fall turns over the roll: never below the coat surface
                    p = p.lerp(want, 0.5)
                col.append(p)
            else:
                col.append(want)
        cols.append(col)
    rows = [[cols[j][k] for j in range(n + 1)] for k in range(len(cols[0]))]
    ob = mesh_from_grid("coat_collar", rows)
    orient_out(ob, F)
    return ob


def skirt_built(F, pts, under, coat, ct, e0, nang=120, nring=18):
    """Coat skirt from the waist seam to below the knee. The top ring sits just outside the coat body at the seam;
    below, rings around the hull of hips and both legs (bridging between the legs like cloth) + gap + a flare that
    grows downward, never tucking back in, with soft folds that deepen toward the hem. Fronts cut on straight edges
    that part a little toward the hem."""
    z_top = F.waist_z
    z_hem = F.knee_z + ct.get("hem_vs_knee", -0.05)
    cx, cy = F.center
    e_hem = ct.get("skirt_edge_hem", 0.09)
    cbvh = collider([coat])
    angs = [2 * math.pi * a / nang for a in range(nang)]
    r0 = []
    for ang in angs:
        # from the axis outward: the first hit is the coat body (a ray from outside meets the sleeves first)
        d = Vector((math.cos(ang), math.sin(ang), 0))
        h = cbvh.ray_cast(Vector((cx, cy, z_top + 0.005)), d, 0.3)
        r0.append((h[3] + 0.003) if h[0] is not None else None)
    ok = [r for r in r0 if r]
    r0 = [r if r else (min(ok) if ok else 0.2) for r in r0]
    rings = []
    prev = None
    fold_n, fold_a = ct.get("skirt_folds", 9), ct.get("skirt_fold_amp", 0.008)
    for k in range(nring + 1):
        t = k / nring
        z = z_top + (z_hem - z_top) * t
        band = [(p.x - cx, p.y - cy) for p in pts if abs(p.z - z) < 0.015 and math.hypot(p.x - cx, p.y - cy) < 0.4]
        hull = convex_hull_2d([(round(a, 4), round(b, 4)) for a, b in band])
        ring = []
        for a, ang in enumerate(angs):
            if k == 0:
                r = r0[a]
            else:
                r = ray_hull(hull, ang) + ct.get("skirt_gap", 0.02) + ct.get("skirt_flare", 0.07) * t ** 1.3
                r = max(r, prev[a] * 0.997)
            ring.append(r)
        prev = ring
        pts_r = []
        for a, ang in enumerate(angs):
            fw = math.sin(ang - math.atan2(F.fwd.y, F.fwd.x) + math.pi / 2)   # +1 front, -1 back
            fold = fold_a * t ** 1.5 * (0.5 + 0.5 * math.sin(fold_n * ang)) * (0.4 + 0.6 * max(0.0, -fw + 0.3))
            rr = ring[a] + fold
            pts_r.append(Vector((cx + rr * math.cos(ang), cy + rr * math.sin(ang), z)))
        rings.append(pts_r)
    ob = grid_mesh("skirt", rings, [True] * nang, closed=True)
    bm = bm_of(ob)
    kill = []
    for f in bm.faces:
        c = f.calc_center_median()
        lat, fw, z = F.local(c)
        t = (z_top - z) / max(1e-3, z_top - z_hem)
        if fw > 0 and abs(lat) < e0 + (e_hem - e0) * t:
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()

    def edge(lat, fw, z):
        if fw <= 0 or z > z_top - 0.002 or z < z_hem + 0.002:
            return None
        t = (z_top - z) / max(1e-3, z_top - z_hem)
        e = e0 + (e_hem - e0) * t
        return math.copysign(e, lat) if abs(lat) < e + 0.05 else None
    boundary_snap(ob, F, edge)
    smooth_skirt(ob, z_top, ct.get("skirt_smooth", 3))
    push_out(ob, collider(under), ct.get("skirt_gap", 0.02) * 0.6)
    orient_out(ob, F)
    ob["z_top"], ob["z_hem"] = z_top, z_hem
    return ob


def button(p, n, r, h=0.38, seg=20):
    """A button: domed rim, dished centre, flat back (a lathe profile), oriented along n at p."""
    prof = [(0.0, 0.16), (0.3, 0.14), (0.55, 0.2), (0.75, 0.42), (0.9, 0.42), (1.0, 0.2), (1.0, 0.0)]
    verts = [(0, 0, prof[0][1] * h * r)]
    for rho, z in prof[1:]:
        for s in range(seg):
            a = 2 * math.pi * s / seg
            verts.append((rho * r * math.cos(a), rho * r * math.sin(a), z * h * r))
    faces = [(0, 1 + s, 1 + (s + 1) % seg) for s in range(seg)]
    for k in range(len(prof) - 2):
        b0, b1 = 1 + k * seg, 1 + (k + 1) * seg
        faces += [(b0 + s, b1 + s, b1 + (s + 1) % seg, b0 + (s + 1) % seg) for s in range(seg)]
    last = 1 + (len(prof) - 2) * seg
    faces.append(tuple(last + s for s in range(seg - 1, -1, -1)))
    me = bpy.data.meshes.new("btn")
    me.from_pydata(verts, [], faces)
    ob = new_obj("btn", me)
    ob.matrix_world = Matrix.Translation(p) @ n.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bm = bm_of(ob)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def buttons_line(F, surf, z_top, z_bot, n, r, name, lat=0.0, out=0.0015):
    bvh = collider([surf])
    objs = []
    for k in range(n):
        z = z_top + (z_bot - z_top) * k / max(1, n - 1)
        hit = surface_point(bvh, F, lat, z)
        if hit is None:
            continue
        objs.append(button(hit[0] + hit[1] * out, hit[1], r))
    return join(objs, name) if objs else new_obj(name, bpy.data.meshes.new(name))


def coat_buttons(F, bvh, ct, e0, zb):
    """Buttons down the wearer's right front from the break, two at the back waist seam."""
    objs = []
    r = ct.get("button_r", 0.0095)
    for dz in ct.get("button_dz", [0.0, -0.09, -0.18, -0.27]):
        hit = surface_point(bvh, F, F.sgn_r * (e0 + ct.get("button_in", 0.024)), zb + dz)
        if hit is not None:
            objs.append(button(hit[0] + hit[1] * 0.001, hit[1], r))
    for sgn in (-1, 1):
        o = F.world(sgn * ct.get("back_button_lat", 0.055), -0.6, F.waist_z + 0.005)
        h = bvh.ray_cast(o, F.fwd, 1.0)
        if h[0] is not None:
            n = h[1] if h[1].dot(F.fwd) < 0 else -h[1]
            objs.append(button(h[0] + n * 0.001, n, r))
    return join(objs, "coat_buttons") if objs else new_obj("coat_buttons", bpy.data.meshes.new("cb"))


def slab(name, p, n, along, w, h, t):
    """A small box lying on a surface: centre p, outward normal n, long axis `along`, size w x h, thickness t."""
    n = n.normalized()
    u = (along - n * along.dot(n)).normalized()
    v = n.cross(u)
    verts = []
    for dz in (0.0, t):
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(tuple(p + u * (a * w / 2) + v * (b * h / 2) + n * dz))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    ob = new_obj(name, me)
    bm = bm_of(ob)
    bmesh.ops.bevel(bm, geom=[e for e in bm.edges], offset=min(t, h) * 0.35, segments=2, affect="EDGES")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def buttonholes(F, bvh, ct, e0, zb):
    """Worked buttonholes on the wearer's left front, level with the buttons: short raised bars of thread."""
    objs = []
    for dz in ct.get("button_dz", [0.0, -0.09, -0.18, -0.27]):
        lat = F.sgn_l * (e0 + 0.016)
        hit = surface_point(bvh, F, lat, zb + dz)
        if hit is not None:
            p, n = hit
            objs.append(slab("bh", p + F.lat * (F.sgn_l * 0.004), n, F.lat, 0.022, 0.0035, 0.0012))
    return join(objs, "buttonholes") if objs else new_obj("buttonholes", bpy.data.meshes.new("bh"))


def welts(F, bvh, spots, w, h, name):
    """Welt pockets: thin raised strips on the surface at (lat, z, tilt radians)."""
    objs = []
    for lat, z, tilt in spots:
        hit = surface_point(bvh, F, lat, z)
        if hit is None:
            continue
        p, n = hit
        along = F.lat * math.cos(tilt) + F.up * math.sin(tilt)
        objs.append(slab("welt", p, n, along, w, h, 0.0025))
    return join(objs, name) if objs else new_obj(name, bpy.data.meshes.new(name))


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
        taper = 1.0 if t < 0.8 else max(so.get("toe_round", 0.55), 1 - (t - 0.8) / 0.2 * 0.75) ** 0.5
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


def extend_sleeves(ob, F, t_end, steps=3, taper=0.1):
    """The body's wrist is weighted to the hand, so a sleeve cut from the body by class stops well short of it.
    Carry each sleeve's open end on along the forearm axis to t_end (metres past the hand joint) as a straight
    tube: the end loop is extruded in steps, each ring keeping the loop's shape, the last ring square to the arm."""
    out = {}
    for s_ in ("l", "r"):
        h, e = F.j[f"hand_{s_}"], F.j[f"lowerarm_{s_}"]
        ax = (h - e).normalized()
        bm = bm_of(ob)
        loop_e = [ed for ed in bm.edges if ed.is_boundary and all(
            (v.co - h).length < 0.25 and -0.25 < (v.co - h).dot(ax) < t_end - 0.005 and (v.co - h).dot(ax) > -0.2
            and ((v.co - h) - ax * (v.co - h).dot(ax)).length < 0.09 for v in ed.verts)]
        if not loop_e:
            bm.free()
            continue
        vs = {v for ed in loop_e for v in ed.verts}
        t0 = float(np.mean([(v.co - h).dot(ax) for v in vs]))
        cur = loop_e
        for k in range(1, steps + 1):
            res = bmesh.ops.extrude_edge_only(bm, edges=cur)
            nv = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
            target = t0 + (t_end - t0) * k / steps
            for v in nv:
                v.co = v.co + ax * (target - (v.co - h).dot(ax))
                c = h + ax * (v.co - h).dot(ax)
                v.co = c + (v.co - c) * (1 - taper / steps)       # a sleeve narrows a little to the wrist
            cur = [g for g in res["geom"] if isinstance(g, bmesh.types.BMEdge) and g.is_boundary]
        bm.to_mesh(ob.data)
        bm.free()
        out[s_] = round(t_end - t0, 3)
    return out


def shoulder_pad(ob, F, pad, rad):
    """A little structure at the shoulder: round each shoulder point the cloth is carried out (lateral) and a
    touch up, falling off smoothly, so the coat's shoulder line is wider and squarer than the lean body's."""
    if pad <= 0:
        return
    co = get_co(ob)
    for s_ in ("l", "r"):
        j = np.array(F.j[f"upperarm_{s_}"])
        sg = 1 if F.local(F.j[f"upperarm_{s_}"])[0] > 0 else -1
        q = j + np.array(F.lat) * sg * 0.02 + np.array([0, 0, 0.035])
        d = np.linalg.norm(co - q, axis=1)
        w = np.clip(1 - d / rad, 0, 1) ** 2
        co += w[:, None] * (np.array(F.lat) * sg * pad + np.array([0, 0, pad * 0.35]))
    set_co(ob, co)
    REPORT["shoulder_pad"] = pad


def tidy_edges(ob, near_fn, iters=12, ring=2):
    """Clean an open edge (a cuff end, a hem): boundary loops whose vertices pass near_fn are smoothed along the
    loop (Laplacian on the loop only), then their inner rings follow, so the edge is a clean curve, not the
    stair-step of the faces it was cut from."""
    bm = bm_of(ob)
    bv = [v for v in bm.verts if v.is_boundary and near_fn(v.co)]
    sel = set(v.index for v in bv)
    if not sel:
        bm.free()
        return 0
    nb = {v.index: [e.other_vert(v).index for e in v.link_edges if e.is_boundary and e.other_vert(v).index in sel]
          for v in bv}
    co = np.array([v.co[:] for v in bm.verts])
    start = co.copy()
    for _ in range(iters):
        new = co.copy()
        for i, ns in nb.items():
            if len(ns) == 2:
                new[i] = 0.5 * co[i] + 0.25 * (co[ns[0]] + co[ns[1]])
        co = new
    moved = co - start
    # inner rings follow with falloff
    front = set(sel)
    for k in range(1, ring + 1):
        nxt = set()
        for i in front:
            for e in bm.verts[i].link_edges:
                j = e.other_vert(bm.verts[i]).index
                if j not in sel and j not in nxt:
                    nxt.add(j)
        for j in nxt:
            ns = [e.other_vert(bm.verts[j]).index for e in bm.verts[j].link_edges]
            src = [moved[n] for n in ns if n in front]
            if src:
                co[j] = co[j] + np.mean(src, 0) * (1 - k / (ring + 1))
        sel |= nxt
        front = nxt
    for v in bm.verts:
        v.co = co[v.index]
    bm.to_mesh(ob.data)
    bm.free()
    return len(nb)


def shirt_cuffs(F, co, cls, sh, nang=48):
    """Stiff single cuffs, constructed: an elliptical tube round the wrist (the wrist's own section from the body,
    + ease), from inside the coat sleeve to cuff_end past the wrist joint, with a rolled edge, so the cuff shows as
    a clean band instead of the cut end of the shirt sleeve."""
    objs = []
    for s_ in ("l", "r"):
        h, e = F.j[f"hand_{s_}"], F.j[f"lowerarm_{s_}"]
        ax = (h - e).normalized()
        u = ax.cross(Vector((0, 0, 1)))
        if u.length < 1e-3:
            u = ax.cross(Vector((0, 1, 0)))
        u.normalize()
        v = ax.cross(u).normalized()
        q = []
        for i, c in enumerate(cls):
            if c not in ("lowerarm", "hand"):
                continue
            d = co[i] - h
            t = d.dot(ax)
            if -0.035 < t < 0.0 and d.length < 0.09:
                q.append((d.dot(u), d.dot(v)))
        if len(q) < 8:
            continue
        Q = np.array(q)
        c2 = Q.mean(0)
        hull = hull_np(Q - c2)
        ang = np.linspace(0, 2 * np.pi, nang, endpoint=False)
        R = ray_hull_np(hull, dirs(ang)) + sh.get("cuff_ease", 0.008)
        for _ in range(3):
            R = sum(np.roll(R, k) for k in range(-2, 3)) / 5
        t0, t1 = sh.get("cuff_start", -0.06), sh.get("cuff_end", 0.04)
        rows = []
        for t in np.linspace(t0, t1, 8):
            rows.append([h + ax * t + u * (c2[0] + R[a] * math.cos(ang[a])) + v * (c2[1] + R[a] * math.sin(ang[a]))
                         for a in range(nang)])
        for dr, dt in ((0.0012, 0.0012), (0.0024, -0.0004)):        # rolled edge at the open end
            rows.append([h + ax * (t1 + dt) + u * (c2[0] + (R[a] + dr) * math.cos(ang[a]))
                         + v * (c2[1] + (R[a] + dr) * math.sin(ang[a])) for a in range(nang)])
        objs.append(grid_mesh(f"cuff_{s_}", rows, [True] * nang, closed=True))
    return join(objs, "shirt_cuffs")


def prism(name, F, poly, z0, z1):
    """A closed slab: polygon (lat, fwd) pairs in the body frame, from z0 to z1, rounded off by one bevel."""
    n = len(poly)
    verts = [tuple(F.world(la, fw, z0)) for la, fw in poly] + [tuple(F.world(la, fw, z1)) for la, fw in poly]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    ob = new_obj(name, me)
    bm = bm_of(ob)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.bevel(bm, geom=[e for e in bm.edges if len(e.link_faces) == 2 and
                              abs(e.link_faces[0].normal.dot(e.link_faces[1].normal)) < 0.5],
                    offset=min(0.0025, (z1 - z0) * 0.3), segments=2, affect="EDGES", profile=0.5)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def sole_and_heel(F, shoe_ob, side, so):
    """A sole that stands a welt's width proud of the upper all round, and a stacked heel under the back third."""
    co = np.array([v.co[:] for v in shoe_ob.data.vertices])
    zb = so.get("sole_bottom", -0.004)
    low = co[co[:, 2] < zb + 0.015]
    L = local_np(F, low)[:, :2]
    h = hull_np(L)
    c = h.mean(0)
    welt = so.get("welt", 0.004)
    poly = [tuple(c + (p - c) * (1 + welt / max(1e-4, np.linalg.norm(p - c)))) for p in h]
    # the foot stands on the floor, so the sole and heel are built up from it around the upper's base: the welt
    # edge and the heel stack show, the upper sits inside them
    sole = prism(f"sole_{side}", F, poly, zb, zb + so.get("sole_t", 0.011))
    f0, f1 = L[:, 1].min(), L[:, 1].max()
    cut = f0 + (f1 - f0) * so.get("heel_frac", 0.28)
    hh = hull_np(L[L[:, 1] <= cut])
    hc = hh.mean(0)
    hp = [tuple(hc + (p - hc) * (1 + welt * 0.5 / max(1e-4, np.linalg.norm(p - hc)))) for p in hh]
    heel = prism(f"heel_{side}", F, hp, zb, zb + so.get("heel_h", 0.026))
    return [sole, heel]


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


def smooth_skirt(ob, z_top, iters):
    """Even out the hull facets around each ring (horizontal smoothing only; heights stay)."""
    bm = bm_of(ob)
    vs = [v for v in bm.verts if v.co.z < z_top - 1e-4]
    for _ in range(iters):
        bmesh.ops.smooth_vert(bm, verts=vs, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=False)
    bm.to_mesh(ob.data)
    bm.free()


# ------------------------------------------------------------------ skinning
def weights_of(ob):
    names = [g.name for g in ob.vertex_groups]
    out = []
    for v in ob.data.vertices:
        out.append({names[g.group]: g.weight for g in v.groups if g.weight > 1e-4})
    return out


def set_weights(ob, W):
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    groups = {}
    for i, w in enumerate(W):
        tot = sum(w.values()) or 1.0
        for n, x in w.items():
            if n not in groups:
                groups[n] = ob.vertex_groups.new(name=n)
            groups[n].add([i], x / tot, "REPLACE")


def transfer_from(dst, srcs):
    """Weights from the nearest of several already-weighted garments (a temporary joined copy as the source), so a
    button or lapel moves with the cloth it sits on, not with the body under it."""
    copies = []
    for o in srcs:
        c = o.copy()
        c.data = o.data.copy()
        for m in list(c.modifiers):
            c.modifiers.remove(m)
        c.parent = None
        bpy.context.scene.collection.objects.link(c)
        copies.append(c)
    src = join(copies, "_wsrc") if len(copies) > 1 else copies[0]
    transfer_weights(dst, src)
    me = src.data
    bpy.data.objects.remove(src)
    bpy.data.meshes.remove(me)


# garments on the body directly, and the pieces that ride on another garment
RIDES_ON = {"lapels": ["coat"], "coat_collar": ["coat"], "coat_buttons": ["coat", "skirt"],
            "buttonholes": ["coat", "skirt"], "welts": ["coat"], "wc_lapels": ["waistcoat"],
            "wc_buttons": ["waistcoat"], "wc_welts": ["waistcoat"], "cravat": ["collar", "shirt"]}


def skin(arm, body, F, parts):
    order = [k for k in parts if k not in RIDES_ON] + [k for k in parts if k in RIDES_ON]
    for key in order:
        ob = parts[key]
        if key == "skirt":
            skirt_weights(ob, body, F)
        elif key in RIDES_ON and all(s in parts for s in RIDES_ON[key]):
            transfer_from(ob, [parts[s] for s in RIDES_ON[key]])
        else:
            transfer_weights(ob, body)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.select_all(action="DESELECT")
        ob.select_set(True)
        bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=8)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    for ob in parts.values():
        ob.parent = arm
        m = ob.modifiers.new("arm", "ARMATURE")
        m.object = arm


def skirt_weights(ob, body, F):
    """The skirt hangs from the waist seam: at the seam it carries exactly the body weights the coat body carries
    there (no tear at the seam); lower down it blends to a designed set: the front panels follow their thigh (they lie
    on it when sitting or stepping), the back panels stay mostly with the pelvis (they hang behind, and do not fold
    into the legs when the knees come up), the centre back splits between both thighs."""
    transfer_weights(ob, body)
    Wt = weights_of(ob)
    sl = F.sgn_l
    z_top = ob.get("z_top", F.waist_z)
    z_hem = ob.get("z_hem", F.knee_z)
    W = []
    for v, wt in zip(ob.data.vertices, Wt):
        lat, fw, z = F.local(v.co)
        t = min(1.0, max(0.0, (z_top - z) / max(1e-3, z_top - z_hem)))
        front = min(1.0, max(0.0, 0.5 + fw / 0.16))
        leg = (0.35 + 0.55 * front) * t ** 0.8
        s = max(-1.0, min(1.0, lat * sl / 0.09))
        d = {"pelvis": 1 - leg}
        if leg * (1 + s) / 2 > 1e-3:
            d["thigh_l"] = leg * (1 + s) / 2
        if leg * (1 - s) / 2 > 1e-3:
            d["thigh_r"] = leg * (1 - s) / 2
        a = min(1.0, t / 0.2)
        a = a * a * (3 - 2 * a)
        tot = sum(wt.values()) or 1.0
        out = {n: (1 - a) * x / tot for n, x in wt.items()}
        for n, x in d.items():
            out[n] = out.get(n, 0.0) + a * x
        W.append(out)
    set_weights(ob, W)


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


def far_verts(parts, core, limit=0.12, loose=("skirt", "coat_buttons", "buttonholes")):
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
    elif kind == "knee_bend":
        for s_ in ("l", "r"):
            rot_world(f"thigh_{s_}", F.lat, 70)
            rot_world(f"calf_{s_}", F.lat, -110)
    elif kind == "head_turn":
        for b_, d_ in (("neck_01", 10), ("neck_02", 12), ("head", 13)):
            rot_world(b_, F.up, d_)
    bpy.context.view_layer.update()


# ------------------------------------------------------------------ bake + export
HOLD_FILE = os.path.expanduser(os.environ.get("CHAR_GPU_HOLD", "~/ue/GPU_HOLD"))
LOCK = {"fd": None, "bus": None}


def gpu_take(kind):
    """Take a GPU lock only for the seconds a render or bake runs: previews try the iGPU (when idle) then card 05,
    bakes card 05 (job "bake_cards"). Non-blocking flock polled; waits while the hold file exists; never the CPU."""
    import fcntl
    cards = JOB.get("preview_cards", ["c4", "05"]) if kind == "preview" else JOB.get("bake_cards", ["05"])
    t0 = time.time()
    while True:
        if not os.path.exists(HOLD_FILE):
            for bus in cards:
                if bus == "c4":
                    try:
                        busy = int(open("/sys/bus/pci/devices/0000:c4:00.0/gpu_busy_percent").read())
                    except Exception:
                        busy = 100
                    if busy > 10:
                        continue
                lock = "/tmp/gpu-igpu.lock" if bus == "c4" else f"/tmp/gpu-{bus}.lock"
                fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    LOCK.update(fd=fd, bus=bus)
                    try:
                        set_device(bus)
                    except RuntimeError as e:
                        REPORT["warnings"].append(str(e))
                        gpu_release()
                        continue
                    REPORT.setdefault("gpu_waits", []).append([kind, bus, round(time.time() - t0, 1)])
                    return bus
                except BlockingIOError:
                    os.close(fd)
        time.sleep(2)


def gpu_release():
    if LOCK["fd"] is not None:
        os.close(LOCK["fd"])
        LOCK.update(fd=None, bus=None)


def set_device(bus):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    p = bpy.context.preferences.addons["cycles"].preferences
    p.compute_device_type = "HIP"
    p.get_devices()
    want = f"0000:{bus}:00"
    hit = False
    for d in p.devices:
        d.use = d.type == "HIP" and d.id.endswith(want)
        hit = hit or d.use
    if not hit:
        raise RuntimeError(f"no HIP device {want}: never bake or render on the CPU")
    sc.cycles.device = "GPU"
    if hasattr(sc.cycles, "denoising_use_gpu"):
        sc.cycles.denoising_use_gpu = True
    REPORT.setdefault("device", []).append(f"HIP {want}")


# texel density per garment, relative: what a face-and-collar close-up sees gets more of the atlas
UV_DENSITY = {"collar": 2.2, "cravat": 2.0, "lapels": 1.6, "coat_collar": 1.8, "wc_lapels": 1.4, "coat_buttons": 1.5,
              "wc_buttons": 1.5, "buttonholes": 1.5, "shirt": 1.3, "waistcoat": 1.1, "coat": 1.0, "trousers": 0.7,
              "skirt": 0.75, "shoes": 0.8}


def tag_tone(parts):
    """Slight tone variation across panels: each island of each part gets its own small offset."""
    rng = np.random.default_rng(S("tone_seed", 11))
    amp = S("tone_vary", 0.05)
    for key, ob in parts.items():
        me = ob.data
        bm = bm_of(ob)
        bm.faces.ensure_lookup_table()
        isl = np.full(len(bm.faces), -1)
        n = 0
        for f in bm.faces:
            if isl[f.index] >= 0:
                continue
            stack = [f]
            isl[f.index] = n
            while stack:
                g = stack.pop()
                for e in g.edges:
                    for h in e.link_faces:
                        if isl[h.index] < 0:
                            isl[h.index] = n
                            stack.append(h)
            n += 1
        bm.free()
        off = rng.uniform(-amp, amp, max(n, 1))
        a = me.attributes.get("tone") or me.attributes.new("tone", "FLOAT", "FACE")
        a.data.foreach_set("value", off[isl].astype(float))


def tag_density(parts):
    for key, ob in parts.items():
        a = ob.data.attributes.new("uvk", "FLOAT", "FACE")
        k = S("uv_density", UV_DENSITY).get(key, 1.0)
        a.data.foreach_set("value", [k] * len(ob.data.polygons))


def unwrap_part(ob):
    """UVs for one garment before the join. A part cut into panels (face attribute "panel", from the drape split)
    gets its panel outlines: seams marked where the panel changes, conformal unwrap, so each island is the flat
    pattern piece at a uniform scale. Other parts: smart projection."""
    me = ob.data
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    me.uv_layers.new(name="UVMap")
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    if "panel" in me.attributes and len(me.polygons):
        pid = np.zeros(len(me.polygons), int)
        me.attributes["panel"].data.foreach_get("value", pid)
        bm = bm_of(ob)
        bm.faces.ensure_lookup_table()
        for e in bm.edges:
            fs = e.link_faces
            e.seam = len(fs) == 2 and pid[fs[0].index] != pid[fs[1].index]
        bm.to_mesh(me)
        bm.free()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.unwrap(method="CONFORMAL", margin=0.002)
        bpy.ops.uv.average_islands_scale()
    else:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.002, area_weight=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    # metric scale: UV area = surface area, so densities compare across parts
    a3 = sum(p.area for p in me.polygons)
    uv = np.zeros(len(me.loops) * 2)
    me.uv_layers[0].data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    a2 = 0.0
    for p in me.polygons:
        q = uv[p.loop_start:p.loop_start + p.loop_total]
        a2 += 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
    if a2 > 0:
        uv = uv * math.sqrt(a3 / a2)
        me.uv_layers[0].data.foreach_set("uv", uv.ravel())
    # a second channel in centimetres, kept through the pack: the engine tiles thread-scale weave maps with it at a
    # real thread density
    cm = me.uv_layers.new(name="UVcm")
    cm.data.foreach_set("uv", (uv * 100.0).ravel())
    me.uv_layers.active_index = 0


def uv_unwrap(ob):
    me = ob.data
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    if "uvk" in me.attributes:
        # scale each garment's islands by its density, then pack (packing keeps relative island sizes)
        k = np.zeros(len(me.polygons))
        me.attributes["uvk"].data.foreach_get("value", k)
        ls = np.zeros(len(me.polygons), int)
        me.polygons.foreach_get("loop_start", ls)
        lt = np.zeros(len(me.polygons), int)
        me.polygons.foreach_get("loop_total", lt)
        per_loop = np.repeat(k, lt)
        uv = np.zeros(len(me.loops) * 2)
        me.uv_layers[0].data.foreach_get("uv", uv)
        uv = (uv.reshape(-1, 2) * per_loop[:, None]).ravel()
        me.uv_layers[0].data.foreach_set("uv", uv)
    # pack works on selected UVs of the active layer; in a background session UV selection is not the mesh
    # selection unless synced
    me.uv_layers.active_index = 0
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, scale=True, margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = np.zeros(len(me.loops) * 2)
    me.uv_layers[0].data.foreach_get("uv", uv)
    REPORT["uv_range"] = [round(float(uv.min()), 4), round(float(uv.max()), 4)]
    if uv.min() < -1e-3 or uv.max() > 1.001:
        raise RuntimeError(f"UV pack left islands outside the atlas: {REPORT['uv_range']}")


def bake(ob, res, samples):
    """Emission bakes of base colour and (roughness, metallic, AO), then a tangent normal bake; one texture set
    for the whole outfit (conventions of the asset chain)."""
    sc = bpy.context.scene
    gpu_take("bake")
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
    gpu_release()
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
    # one slot per cloth, all on the same baked maps: the slot name (MB_<cloth>) tells the engine which cloth
    # shading and weave to put over the bake
    for i, slot in enumerate(ob.material_slots):
        src = slot.material
        if src is None or src is m:
            continue
        key = src.name[2:] if src.name.startswith("M_") else src.name
        mb = bpy.data.materials.get(f"MB_{key}") or m.copy()
        mb.name = f"MB_{key}"
        ob.data.materials[i] = mb
    if not ob.material_slots:
        ob.data.materials.append(m)


def to_centimetres(objs, arm):
    """Scale the armature and its meshes to centimetres (data, identity transforms). UE reads Blender's FBX with
    the unit scale on the root bone otherwise (a root scaled x100), and a leader pose from the character's body
    then flings every vertex; with the data in centimetres and no unit scale the root bone is identity."""
    if arm.get("cm"):
        return
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs + [arm]:
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    arm.scale = (100, 100, 100)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    arm["cm"] = True


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


# ------------------------------------------------------------------ preview
def preview(objs, tag, views=(("front", 0), ("three_q", 35), ("side", 90), ("back", 180)), px=(640, 900),
            samples=24, target_z=None, dist=3.2, focal=50):
    sc = bpy.context.scene
    gpu_take("preview")
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
    for view in views:
        name, az = view[0], view[1]
        vz = view[2] if len(view) > 2 else tz
        vd = view[3] if len(view) > 3 else dist
        cd.lens = view[4] if len(view) > 4 else focal
        a = math.radians(az)
        d = F.fwd * math.cos(a) + F.lat * math.sin(a)
        tgt = Vector((F.center[0], F.center[1], vz))
        cam.location = tgt + d * vd + Vector((0, 0, 0.05))
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        p = os.path.join(OUT, "preview", f"{tag}_{name}.png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
    gpu_release()
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
        try:
            F, parts = build(arm, body, head)
        except DrapeExported as e:
            REPORT["ok"] = True
            REPORT["stopped_for_drape"] = str(e)
            save_report()
            return
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
        for kind in ("seated", "arm_up", "knee_bend", "head_turn"):
            test_pose(arm, kind)
            with Timer(f"poke_{kind}"):
                REPORT["poke"][kind] = poke_counts(layers, core)
            if "preview" in steps:
                with Timer(f"preview_{kind}"):
                    REPORT.setdefault("preview_pose", []).extend(
                        preview(garments + [trim] + ([head] if head else []), kind,
                                **{**JOB.get("preview_args", {}), "views": (("front", 0), ("side", 90))}))
        test_pose(arm, "rest")
    outfit = None
    if "bake" in steps or "export" in steps:
        bpy.ops.object.select_all(action="DESELECT")
        tag_density(parts)
        tag_tone(parts)
        with Timer("uv_parts"):
            for o in garments:
                unwrap_part(o)
        outfit = join([o for o in garments], "outfit")
        with Timer("uv"):
            uv_unwrap(outfit)
    if "bake" in steps:
        with Timer("bake"):
            imgs = bake(outfit, JOB.get("bake_px", 4096), JOB.get("bake_samples", 4))
        baked_material(outfit, imgs)
    if outfit is not None and JOB.get("hands_in_outfit", True):
        # the visible skin (hands, wrists) rides in the outfit mesh as a second material slot "M_skin": a separate
        # trimmed-body skeletal mesh rendered nowhere in UE (the hands vanished); in the outfit it shares the
        # outfit's skeleton and bind, and UE gives the slot the body's own skin material
        hands = trim.copy()
        hands.data = trim.data.copy()
        bpy.context.scene.collection.objects.link(hands)
        hme = hands.data
        while len(hme.uv_layers) > 1:
            hme.uv_layers.remove(hme.uv_layers[-1])
        if hme.uv_layers:
            hme.uv_layers[0].name = outfit.data.uv_layers[0].name
        hme.materials.clear()
        hme.materials.append(bpy.data.materials.new("M_skin"))
        for m in list(hands.modifiers):
            hands.modifiers.remove(m)
        hv = sum(1 for v in hme.vertices if any(hands.vertex_groups[g.group].name.startswith("hand_")
                                                    for g in v.groups))
        REPORT["hands_in_outfit"] = {"verts": len(hme.vertices), "hand_weighted": hv}
        if hv < 50:
            raise RuntimeError(f"the trimmed body has no hands ({hv} hand-weighted vertices)")
        outfit = join([outfit, hands], "outfit")
        outfit.parent = arm
        if not any(m.type == "ARMATURE" for m in outfit.modifiers):
            outfit.modifiers.new("arm", "ARMATURE").object = arm
    if "export" in steps:
        to_centimetres([o for o in (outfit, trim, body) if o is not None], arm)
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
