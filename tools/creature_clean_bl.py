"""Blender side: CLEAN an image-to-3D creature body into one watertight mesh (any template creature).

    blender -b --factory-startup -t 16 --python tools/creature_clean_bl.py -- JOB.json

JOB {"rigged": a tools/blockout_creature_bl.py rigged.blend (its skin tells which vertices are wing / head / limb),
     "rig_json", "head": optional higher-detail head mesh (.glb, e.g. Hunyuan3D from a head-only reference),
     "voxel": remesh voxel size in metres (default 1/300 of the height), "faces": target face count,
     "limb_out_deg": fore limbs swung out from the body before remeshing (so the remesh cannot weld them to the
     flanks again), "out_glb", "stats_json"}
Steps: cut the faces that weld a fore limb to anything outside its chain; pose the fore limbs out and bake; delete
the fused wing shell (the wings are rebuilt procedurally by tools/creature_parts_bl.py) and, with "head", the old
head; fit the new head onto the neck (scale + rigid alignment by chamfer distance to the old head); join; VOXEL
REMESH (watertight, one surface); delete loose islands (< 0.5 % of the vertices: smoke wisps, strands); smooth;
decimate; report islands / non-manifold edges / boundary loops (holes) / vertices.
"""

import json
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
bpy.ops.wm.open_mainfile(filepath=job["rigged"])
ob = bpy.data.objects["model"]
arm = bpy.data.objects["rig"]
RJ = json.load(open(job["rig_json"]))
J = {k: np.asarray(v, float) for k, v in RJ["joints"].items()}
H = float(RJ.get("height", 6.0))
gi = {g.index: g.name for g in ob.vertex_groups}


def dom(v):
    return gi[max(v.groups, key=lambda g: g.weight).group] if v.groups else None


def stats(o):
    bm = bmesh.new()
    bm.from_mesh(o.data)
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    bnd = [e for e in bm.edges if e.is_boundary]
    # boundary loops = holes
    seen, loops = set(), 0
    adj = {}
    for e in bnd:
        for v in e.verts:
            adj.setdefault(v.index, []).append(e)
    for e in bnd:
        if e.index in seen:
            continue
        loops += 1
        stack = [e]
        while stack:
            x = stack.pop()
            if x.index in seen:
                continue
            seen.add(x.index)
            for v in x.verts:
                stack += [y for y in adj.get(v.index, []) if y.index not in seen]
    # islands
    bm.verts.ensure_lookup_table()
    lab = [-1] * len(bm.verts)
    isl = []
    for v in bm.verts:
        if lab[v.index] >= 0:
            continue
        n, st = 0, [v]
        lab[v.index] = len(isl)
        while st:
            x = st.pop()
            n += 1
            for e in x.link_edges:
                y = e.other_vert(x)
                if lab[y.index] < 0:
                    lab[y.index] = len(isl)
                    st.append(y)
        isl.append(n)
    out = {"verts": len(bm.verts), "faces": len(bm.faces), "non_manifold_edges": nm, "holes": loops,
           "islands": len(isl), "largest_island": max(isl) if isl else 0}
    bm.free()
    return out


before = stats(ob)
print("CLEAN before", json.dumps(before))
# 1. cut the welds between a fore limb and anything outside its chain (their faces stretch into webs)
bm = bmesh.new()
bm.from_mesh(ob.data)
dl = bm.verts.layers.deform.verify()
vdom = []
for v in bm.verts:
    dv = v[dl]
    vdom.append(gi[max(dv.items(), key=lambda kv: kv[1])[0]] if dv else None)


def grp(n):
    if n is None:
        return None
    for pre in ("fleg_l", "fleg_r"):
        if n.startswith(pre):
            return pre
    return "o:" + n


cut = []
for f in bm.faces:
    gs = {grp(vdom[v.index]) for v in f.verts}
    limbs = {g for g in gs if g and not g.startswith("o:")}
    others = {g[2:] for g in gs if g and g.startswith("o:")} - {"chest", "spine3", "neck1"}
    if limbs and (len(limbs) > 1 or others):
        cut.append(f)
bmesh.ops.delete(bm, geom=cut, context="FACES_ONLY")
CORE = [J[k] for k in ("pelvis", "spine1", "spine2", "spine3", "chest", "neck1", "neck2", "neck3", "neck4", "head")
        if k in J]


def core_dist(p):
    """Distance from the body's spine-neck axis (polyline pelvis..head)."""
    best = 1e9
    for a, b in zip(CORE[:-1], CORE[1:]):
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(float(ab @ ab), 1e-9), 0, 1))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


NECK_C = J.get("neck4", J["head"])
NECK_D = (J["head"] - J.get("neck3", NECK_C)) / max(np.linalg.norm(J["head"] - J.get("neck3", NECK_C)), 1e-9)
HEAD_R = 0.25 * H


# 2. delete the fused wing shell, and the old head when a new head is given (keep the old head's points)
old_head = []
kill = set()
for v in bm.verts:
    n = vdom[v.index]
    if n and n.startswith("wing_") and core_dist(np.asarray(v.co, float)) > 0.06 * H:
        kill.add(v)                                    # the fused wing shell; the back and neck under it stay
    if job.get("head"):
        # the old head = beyond ONE plane through the neck4 joint across the neck (a clean ring cut that the new
        # head's neck stub covers), not every vertex the head bones happen to own (that gouged the neck)
        p_ = np.asarray(v.co, float)
        if np.dot(p_ - NECK_C, NECK_D) > 0 and np.linalg.norm(p_ - J["head"]) < HEAD_R:
            old_head.append(p_)
            kill.add(v)
bmesh.ops.delete(bm, geom=list(kill), context="VERTS")
# CLOSE every opening (the voxel remesh needs a closed volume: an open shell remeshes into fragments)
filled = bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
bm.to_mesh(ob.data)
bm.free()
ob.data.update()
print("CLEAN cut", len(cut), "weld faces; removed", len(kill), "wing/head verts; capped", len(filled.get("faces", [])))

# 3. pose the fore limbs OUT from the body and bake (the remesh then cannot re-weld them)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
deg = float(job.get("limb_out_deg", 14.0))
_bad = {"l": "r", "r": "l"}.get(job.get("mirror_arm"))  # the bad arm is replaced by a mirror: never pose it
for s, sg in [x for x in (("l", 1), ("r", -1)) if x[0] != _bad]:
    pbn = arm.pose.bones.get(f"fleg_{s}_knee")              # template: *_knee bone = the upper arm
    if pbn is None:
        continue
    M = arm.matrix_world @ pbn.matrix
    hd = M.translation.copy()
    R = Matrix.Rotation(math.radians(sg * deg), 4, Vector((1, 0, 0)))
    pbn.matrix = arm.matrix_world.inverted() @ (Matrix.Translation(hd) @ R @ Matrix.Translation(-hd) @ M)
    bpy.context.view_layer.update()
bpy.ops.object.mode_set(mode="OBJECT")
JO = {k: v.copy() for k, v in J.items()}
for s_, sg_ in (("l", 1), ("r", -1)):                  # the joints follow the fore limbs posed out
    piv = J.get(f"fleg_{s_}_hip")
    if piv is None:
        continue
    Rn = np.array(Matrix.Rotation(math.radians(sg_ * deg), 3, Vector((1, 0, 0))))
    for k in ("knee", "ankle", "toe"):
        nm_ = f"fleg_{s_}_{k}"
        if nm_ in JO:
            JO[nm_] = piv + Rn @ (J[nm_] - piv)
dg = bpy.context.evaluated_depsgraph_get()
baked = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
body = bpy.data.objects.new("body", baked)
bpy.context.scene.collection.objects.link(body)
body.matrix_world = ob.matrix_world.copy()
for o in (ob, arm):
    bpy.data.objects.remove(o, do_unlink=True)
for m in list(body.modifiers):
    body.modifiers.remove(m)

# MIRROR AN ARM: image-to-3D fused one fore limb into the flank; replace the bad arm by the mirror of the good
# arm (the region around its bone chain, from a third down the upper arm to the claws); the rest of the body,
# legs and tail keep their own geometry
if job.get("mirror_arm") in ("l", "r"):
    g_, b_ = job["mirror_arm"], ("r" if job["mirror_arm"] == "l" else "l")
    ch = lambda s_: [JO[f"fleg_{s_}_{k}"] for k in ("hip", "knee", "ankle", "toe") if f"fleg_{s_}_{k}" in JO]
    rad = float(job.get("arm_radius", 0.06 * H))

    def chain_d(p, C):
        best = (1e9, 0.0)
        Ltot = sum(float(np.linalg.norm(b - a)) for a, b in zip(C[:-1], C[1:]))
        acc = 0.0
        for a, b in zip(C[:-1], C[1:]):
            ab = b - a
            L_ = float(np.linalg.norm(ab))
            t = float(np.clip(np.dot(p - a, ab) / max(L_ * L_, 1e-9), 0, 1))
            d = float(np.linalg.norm(p - (a + t * ab)))
            if d < best[0]:
                best = (d, (acc + t * L_) / max(Ltot, 1e-9))
            acc += L_
        return best
    Cg, Cb = ch(g_), ch(b_)
    frac0 = float(job.get("arm_from", 0.3))
    bm = bmesh.new()
    bm.from_mesh(body.data)
    Lb = [JO[f"hleg_{b_}_{k}"] for k in ("hip", "knee", "ankle", "toe") if f"hleg_{b_}_{k}" in JO]
    kill_ = []
    for v in bm.verts:
        p_ = np.asarray(v.co, float)
        d, u = chain_d(p_, Cb)
        # the ARM only: never the thigh it rested on or the flank (nearer the leg / the body axis than the arm)
        if d < rad * 1.6 and u > frac0 and d < 0.8 * min(chain_d(p_, Lb)[0] if Lb else 9.0, core_dist(p_)):
            kill_.append(v)
    bmesh.ops.delete(bm, geom=kill_, context="VERTS")
    Lg = [JO[f"hleg_{g_}_{k}"] for k in ("hip", "knee", "ankle", "toe") if f"hleg_{g_}_{k}" in JO]
    sel = []
    for f in bm.faces:
        ok = True
        for v in f.verts:
            p_ = np.asarray(v.co, float)
            d, u = chain_d(p_, Cg)
            if not (d < rad * 1.6 and u > frac0 and d < 0.8 * min(chain_d(p_, Lg)[0] if Lg else 9.0, core_dist(p_))):
                ok = False
                break
        if ok:
            sel.append(f)
    dup = bmesh.ops.duplicate(bm, geom=sel)
    for v in [g for g in dup["geom"] if isinstance(g, bmesh.types.BMVert)]:
        v.co.y = -v.co.y
    bmesh.ops.reverse_faces(bm, faces=[g for g in dup["geom"] if isinstance(g, bmesh.types.BMFace)])
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)   # close both arm ends
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    for k in ("hip", "knee", "ankle", "toe"):
        if f"fleg_{g_}_{k}" in JO:
            m_ = np.asarray(JO[f"fleg_{g_}_{k}"], float).copy()
            m_[1] = -m_[1]
            JO[f"fleg_{b_}_{k}"] = m_
    print("CLEAN arm", b_, "<- mirror of", g_, ": removed", len(kill_), "verts, copied", len(sel), "faces")

parts = [body]
# 4. the new head, fitted onto the neck
if job.get("head") and old_head:
    bpy.ops.import_scene.gltf(filepath=job["head"])
    hobs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in hobs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = hobs[0]
    if len(hobs) > 1:
        bpy.ops.object.join()
    hd = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # drop its loose bits (a smoke wisp drawn into the reference becomes geometry)
    bm = bmesh.new()
    bm.from_mesh(hd.data)
    bm.verts.ensure_lookup_table()
    lab = {}
    isl = []
    for v in bm.verts:
        if v.index in lab:
            continue
        st, mem = [v], []
        lab[v.index] = len(isl)
        while st:
            x = st.pop()
            mem.append(x)
            for e in x.link_edges:
                y = e.other_vert(x)
                if y.index not in lab:
                    lab[y.index] = len(isl)
                    st.append(y)
        isl.append(mem)
    big = max(len(m) for m in isl)
    bmesh.ops.delete(bm, geom=[v for m in isl if len(m) < 0.02 * big for v in m], context="VERTS")
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)   # close the neck stub
    bm.to_mesh(hd.data)
    bm.free()
    HV = np.array([v.co[:] for v in hd.data.vertices])
    OH = np.array(old_head)
    # a head-only drawing's mesh: Z up; its NECK stub is the lowest slice; FORWARD = from the neck stub to the
    # snout tip (the point farthest out horizontally). Seat it on the body: the neck stub on the body's head
    # joint (overlapping into the neck), forward = the body's head direction, length = the old head's length.
    z0, z1 = HV[:, 2].min(), HV[:, 2].max()
    neckc = HV[HV[:, 2] < z0 + 0.08 * (z1 - z0)].mean(0)
    # the SNOUT: the farthest point out among the LOWER part of the head (horns and crest are high and
    # curve back; they must not set the forward direction)
    low = HV[(HV[:, 2] > z0 + 0.12 * (z1 - z0)) & (HV[:, 2] < z0 + 0.55 * (z1 - z0))]
    hz = low - neckc
    hz[:, 2] = 0
    tip = low[np.argmax(np.linalg.norm(hz, axis=1))]
    fwd_h = tip - neckc
    fwd_h[2] = 0
    fwd_h /= max(np.linalg.norm(fwd_h), 1e-9)
    # the body's head direction: neck4 -> head-end tip (horizontal), and the old head's length along it
    hb_tip = OH[np.argmax((OH - J["head"]) @ np.array([1.0, 0, 0]))]
    fwd_b = np.array([1.0, 0.0, 0.0])
    len_old = float(np.ptp(OH @ fwd_b))
    len_new = float(np.ptp(HV @ fwd_h))
    s = len_old / max(len_new, 1e-6) * float(job.get("head_scale", 1.15))
    ang = math.atan2(fwd_b[1], fwd_b[0]) - math.atan2(fwd_h[1], fwd_h[0])
    Rz = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
    # the neck stub goes a little INTO the body's neck (the voxel union merges them): at the neck4 joint
    # the capped neck stub sinks well into the body's neck (no flat block under the head)
    seat = J.get("neck3", J.get("neck4", J["head"])) + (J["head"] - J.get("neck3", J["head"])) * 0.35
    A = Rz * s
    off = seat - A @ neckc
    T = Matrix.Identity(4)
    for i_ in range(3):
        for j_ in range(3):
            T[i_][j_] = float(A[i_, j_])
        T[i_][3] = float(off[i_])
    d, s_ = 0.0, s
    hd.data.transform(T)
    hd.data.update()
    # NECK FILL: image-to-3D necks under fused wings come out thin and torn; a tapered tube along the chest ->
    # neck joints -> the seated head's neck stub (the voxel union blends it into one smooth neck)
    stub = HV[HV[:, 2] < z0 + 0.08 * (z1 - z0)]
    r_stub = 0.5 * float(np.ptp(stub[:, :2], axis=0).mean()) * s
    radii = RJ.get("radii", {})
    r_chest = max(float(radii.get("chest", 0.0)), float(radii.get("spine3", 0.0)), 0.07 * H)
    chain = [J[k] for k in ("spine3", "chest", "neck1", "neck2", "neck3", "neck4") if k in J] + [seat]
    pts, rs = [], []
    for a_, b_ in zip(chain[:-1], chain[1:]):
        for t_ in np.linspace(0, 1, 6, endpoint=False):
            pts.append(a_ + (b_ - a_) * t_)
    pts.append(chain[-1])
    for k_, p_ in enumerate(pts):
        u_ = k_ / max(len(pts) - 1, 1)
        rs.append(r_chest * (1 - u_) + r_stub * u_)
    for p_, r_ in zip(pts, rs):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(r_), location=tuple(float(x) for x in p_),
                                             segments=24, ring_count=12)
        parts.append(bpy.context.active_object)
    # the new head's landmarks: snout tip = its forward-most point; the mouth line at ~40 % of the snout's
    # height; the head joint stays at the seat (the hinge); the jaw joint on the mouth line 40 % back from the tip
    W_ = (HV @ A.T) + off
    fx_ = np.array([1.0, 0, 0])
    snout = W_[np.argmax(W_ @ fx_)]
    front = W_[(W_ @ fx_) > (snout @ fx_) - 0.25 * len_old * float(job.get("head_scale", 1.15))]
    mz = float(np.percentile(front[:, 2], 40))
    # the JAW HINGE: on the mouth line, at the back of the mouth (~65 % back from the snout tip): the jaw bone
    # turns about its head joint, so a high skull-top pivot barely opened the mouth
    JO["head"] = snout + (seat - snout) * 0.65
    JO["head"][2] = mz + 0.015 * H
    JO["jaw"] = np.array([JO["head"][0] + (snout[0] - JO["head"][0]) * 0.6, 0.0, mz])
    SNOUT_TIP = np.array([snout[0], 0.0, mz + 0.02 * H])
    print("CLEAN neck fill: r_chest", round(r_chest, 3), "r_stub", round(r_stub, 3), "spheres", len(pts))
    print("CLEAN head seated: forward", fwd_h.round(2).tolist(), "scale", round(s, 3), "seat", seat.round(2).tolist())
    hd.name = "newhead"
    parts.append(hd)

# 5. one watertight surface: join, voxel remesh, islands, smooth, decimate
bpy.ops.object.select_all(action="DESELECT")
for o in parts:
    o.select_set(True)
bpy.context.view_layer.objects.active = body
if len(parts) > 1:
    bpy.ops.object.join()
body = bpy.context.view_layer.objects.active
vox = float(job.get("voxel") or H / 300.0)
rm = body.modifiers.new("remesh", "REMESH")
rm.mode = "VOXEL"
rm.voxel_size = vox
rm.adaptivity = 0.0
bpy.ops.object.modifier_apply(modifier=rm.name)
# FLAPS: thin double-sided sheets (an apron of skin left from a fused limb, wing-root plates) survive a voxel
# remesh; a vertex whose inward ray hits the opposite surface within ~3.5 cm is a flap: delete, close, remesh
from mathutils.bvhtree import BVHTree
for _pass in range(int(job.get("flap_passes", 0))):   # opt-in (experimental)
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.normal_update()
    tree = BVHTree.FromBMesh(bm)
    thin = float(job.get("flap_thickness", 0.035 * H / 6.0))
    flap = []
    for v in bm.verts:
        n_ = v.normal
        if n_.length < 1e-6:
            continue
        p_ = np.asarray(v.co, float)
        # only the TORSO region (horns, claws, spikes and the tail tip are legitimately thin)
        if not (float(J["pelvis"][2]) - 0.12 * H < p_[2] < float(J.get("neck1", J["chest"])[2]) and
                core_dist(p_) < 0.16 * H):
            continue
        hit = tree.ray_cast(v.co - n_ * 1e-4, -n_, thin)
        if hit[0] is not None:
            flap.append(v)
    bmesh.ops.delete(bm, geom=flap, context="VERTS")
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    print("CLEAN flap pass", _pass, "removed", len(flap), "thin verts")
    rm2 = body.modifiers.new("remesh2", "REMESH")
    rm2.mode = "VOXEL"
    rm2.voxel_size = vox
    rm2.adaptivity = 0.0
    bpy.ops.object.modifier_apply(modifier=rm2.name)
bm = bmesh.new()
bm.from_mesh(body.data)
bm.verts.ensure_lookup_table()
lab, isl = {}, []
for v in bm.verts:
    if v.index in lab:
        continue
    st, mem = [v], []
    lab[v.index] = len(isl)
    while st:
        x = st.pop()
        mem.append(x)
        for e in x.link_edges:
            y = e.other_vert(x)
            if y.index not in lab:
                lab[y.index] = len(isl)
                st.append(y)
    isl.append(mem)
big = max(len(m) for m in isl)
bmesh.ops.delete(bm, geom=[v for m in isl if len(m) < 0.05 * big for v in m], context="VERTS")   # strands + INNER shells (a cavity)
bm.to_mesh(body.data)
bm.free()
sm = body.modifiers.new("smooth", "CORRECTIVE_SMOOTH")
sm.iterations = 4
sm.factor = 0.5
sm.use_only_smooth = True
bpy.ops.object.modifier_apply(modifier=sm.name)
dc = body.modifiers.new("dec", "DECIMATE")
dc.ratio = min(1.0, float(job.get("faces", 120000)) / max(len(body.data.polygons), 1))
bpy.ops.object.modifier_apply(modifier=dc.name)
# the TORSO surface (chest, belly, back): flaps left where a fused limb was cut away -> smoothed (volume kept);
# limbs, head and tail keep their detail
bm = bmesh.new()
bm.from_mesh(body.data)
z_lo = float(J["pelvis"][2]) - 0.05 * H
z_hi = float(J.get("neck1", J["chest"])[2])
tors = [v for v in bm.verts if z_lo < v.co.z < z_hi and core_dist(np.asarray(v.co, float)) < 0.11 * H]
for _ in range(int(job.get("torso_smooth", 12))):
    bmesh.ops.smooth_laplacian_vert(bm, verts=tors, lambda_factor=0.6, lambda_border=0.0, use_x=True, use_y=True,
                                    use_z=True, preserve_volume=True)
bm.to_mesh(body.data)
bm.free()
print("CLEAN torso smoothed", len(tors), "verts")
for p in body.data.polygons:
    p.use_smooth = True
after = stats(body)
print("CLEAN after", json.dumps(after))
body.name = "model"
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
bpy.ops.export_scene.gltf(filepath=job["out_glb"], use_selection=True, export_format="GLB", export_apply=True)
json.dump({"before": before, "after": after, "voxel_m": vox, "weld_faces_cut": len(cut)}, open(job["stats_json"], "w"), indent=1)
if job.get("joints_out"):
    json.dump({"joints": {k: np.asarray(v).tolist() for k, v in JO.items()},
               "snout_tip": SNOUT_TIP.tolist() if job.get("head") and old_head else None},
              open(job["joints_out"], "w"), indent=1)
