"""Rig a procedural creature body (tools/creature_sdf.py output) on its skeleton: heat weights, checks, rig.json.

    blender -b --factory-startup -t 16 --python tools/creature_sdf_bl.py -- OUT_DIR

Reads OUT_DIR/body.npz + skeleton.json; writes OUT_DIR/rigged.blend (objects "model" + "rig", the same layout as
the other creature rigs, so creature_qa_bl.py / the renderer / the motion tools read it) and rig.json (joints,
anchors, qa_poses, checks). Checks (generic): 1 island, watertight (no boundary / non-manifold edges), 0 zero-weight
vertices, no vertex far from the body box.
"""

import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

out = Path(sys.argv[sys.argv.index("--") + 1])
SK = json.loads((out / "skeleton.json").read_text())
J = {k: np.asarray(v, float) for k, v in SK["joints"].items()}
S = float(SK["height_m"])
bpy.ops.wm.read_factory_settings(use_empty=True)
z = np.load(out / "body.npz")
V, F = z["verts"].astype(float), z["faces"].astype(int)
me = bpy.data.meshes.new("model")
me.from_pydata(V.tolist(), [], F.tolist())
me.update()
ob = bpy.data.objects.new("model", me)
bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.objects.active = ob
ob.select_set(True)

# one island (marching cubes can leave specks), consistent normals, a lighter mesh
bm = bmesh.new()
bm.from_mesh(me)
bm.verts.ensure_lookup_table()
lab, isl = {}, []
for v in bm.verts:
    if v.index in lab:
        continue
    st, mem = [v], []
    lab[v.index] = 1
    while st:
        x = st.pop()
        mem.append(x)
        for e in x.link_edges:
            y = e.other_vert(x)
            if y.index not in lab:
                lab[y.index] = 1
                st.append(y)
    isl.append(mem)
big = max(isl, key=len)
drop = [v for m in isl if m is not big for v in m]
bmesh.ops.delete(bm, geom=drop, context="VERTS")
bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
bm.to_mesh(me)
bm.free()
dec = ob.modifiers.new("dec", "DECIMATE")
dec.ratio = min(1.0, float(SK.get("faces", 120000)) / max(len(me.polygons), 1))
bpy.ops.object.modifier_apply(modifier=dec.name)
sm = ob.modifiers.new("sm", "CORRECTIVE_SMOOTH")
sm.iterations = 3
sm.use_only_smooth = True
bpy.ops.object.modifier_apply(modifier=sm.name)
for p in me.polygons:
    p.use_smooth = True

# the skeleton
ad = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("rig", ad)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for b, par, h, t in SK["bones"]:
    e = ad.edit_bones.new(b)
    e.head = Vector(J[h].tolist())
    e.tail = Vector(J[t].tolist())
    if np.linalg.norm(J[t] - J[h]) < 1e-4:
        e.tail = e.head + Vector((0, 0, 0.05 * S))
    eb[b] = (e, par)
for b, (e, par) in eb.items():
    if par:
        e.parent = eb[par][0]
bpy.ops.object.mode_set(mode="OBJECT")
# heat weights
bpy.ops.object.select_all(action="DESELECT")
ob.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_NAME")      # empty groups; the weights are computed below
gi = {g.index: g.name for g in ob.vertex_groups}


def bone_dist(p, b):
    _, _, h, t = b
    a_, b_ = J[h], J[t]
    ab = b_ - a_
    hh = np.clip(((p - a_) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a_ + hh * ab)))


# THICKNESS-NORMALISED distance weights (heat diffusion fails on a body this size: flat 0.25 weights over many
# bones). For each vertex, q_b = distance to bone b / b's own radius (so the belly belongs to the spine, not to
# the arm passing beside it); weights fall off over 0.35 in q from the best bone, top 4, normalised
Pv0 = np.array([v.co[:] for v in ob.data.vertices])
BN = [b for b in SK["bones"] if b[0] not in ("head_end",) or True]
Dm = np.zeros((len(Pv0), len(BN)))
for j, b in enumerate(BN):
    a_, b_ = J[b[2]], J[b[3]]
    ab = b_ - a_
    hh = np.clip(((Pv0 - a_) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    Dm[:, j] = np.linalg.norm(Pv0 - (a_ + hh[:, None] * ab), axis=1)
own = np.argmin(Dm, axis=1)
rad = np.array([np.percentile(Dm[own == j, j], 50) if np.sum(own == j) > 20 else 0.05 * S for j in range(len(BN))])
rad = np.maximum(rad, 0.01 * S)
Q = Dm / rad[None, :]
qmin = Q.min(axis=1, keepdims=True)
W = np.clip(1.0 - (Q - qmin) / 0.35, 0, 1) ** 2
for i in range(len(Pv0)):
    top = np.argsort(-W[i])[:4]
    tot = W[i, top].sum()
    for j in top:
        if W[i, j] > 1e-3:
            (ob.vertex_groups.get(BN[j][0]) or ob.vertex_groups.new(name=BN[j][0])).add([i], float(W[i, j] / tot), "REPLACE")
gi = {g.index: g.name for g in ob.vertex_groups}

# zero-weight vertices (heat can miss thin parts): the nearest bone, then normalise and limit to 4 bones
nz = 0
for v in ob.data.vertices:
    if sum(g.weight for g in v.groups) < 1e-4:
        p = np.asarray(v.co, float)
        b = min(SK["bones"], key=lambda b_: bone_dist(p, b_))
        g = ob.vertex_groups.get(b[0]) or ob.vertex_groups.new(name=b[0])
        g.add([v.index], 1.0, "REPLACE")
        nz += 1
# the MEMBRANES: their skin follows the WING bones only (heat weights let a nearby arm or the pelvis grab parts of
# the sheet): every vertex on a membrane slab gets inverse-distance weights of its two nearest wing bones
def _tri_d(P, p0, p1, p2):
    e0, e1 = p1 - p0, p2 - p0
    n = np.cross(e0, e1)
    n = n / max(np.linalg.norm(n), 1e-12)
    w = P - p0
    d00, d01, d11 = e0 @ e0, e0 @ e1, e1 @ e1
    d20, d21 = w @ e0, w @ e1
    den = d00 * d11 - d01 * d01
    vv = (d11 * d20 - d01 * d21) / den
    uu = (d00 * d21 - d01 * d20) / den
    inside = (uu >= -0.02) & (vv >= -0.02) & (uu + vv <= 1.02)
    return np.where(inside, np.abs(w @ n), 1e9)


Pv = np.array([v.co[:] for v in ob.data.vertices])
nmem = 0
for mb in SK.get("membranes", []):
    poly = [J[n] for n in mb["outline"]]
    fan = J[mb["fan"]]
    dmin = np.full(len(Pv), 1e9)
    for i in range(len(poly)):
        a2, b2 = poly[i], poly[(i + 1) % len(poly)]
        if np.allclose(a2, fan) or np.allclose(b2, fan):
            continue
        dmin = np.minimum(dmin, _tri_d(Pv, fan, a2, b2))
    side = mb["outline"][0].split("_")[1]
    wb = [b for b in SK["bones"] if b[0].startswith(f"wing_{side}_")]
    on = np.where(dmin < 1.5 * mb["thick"] * S + 0.05)[0]
    for vi in on:
        p = Pv[vi]
        ds = sorted((bone_dist(p, b), b[0]) for b in wb)[:2]
        ws = [1.0 / max(d_, 1e-3) ** 2 for d_, _ in ds]
        tot = sum(ws)
        v = ob.data.vertices[int(vi)]
        for g in list(v.groups):
            ob.vertex_groups[g.group].remove([int(vi)])
        for (d_, bn), w_ in zip(ds, ws):
            (ob.vertex_groups.get(bn) or ob.vertex_groups.new(name=bn)).add([int(vi)], w_ / tot, "REPLACE")
        nmem += 1
print("SDF membrane verts reweighted", nmem)
gi = {g.index: g.name for g in ob.vertex_groups}

# the JAW: below the mouth plane and in front of the mouth corners = rigid on the jaw; behind the corners a
# stretch band blends jaw -> skull/neck (no tear line); above the plane: none
jw = SK.get("jaw")
if jw and ob.vertex_groups.get("jaw"):
    a_ = np.asarray(jw["plane"][0], float) * S
    b_ = np.asarray(jw["plane"][1], float) * S
    dd = (b_ - a_) / np.linalg.norm(b_ - a_)
    nrm = np.cross(dd, [0, 1.0, 0])
    nrm = -nrm if nrm[2] < 0 else nrm
    cx, band, hr = jw["corner_x"] * S, jw["band"] * S, jw["head_r"] * S
    hc = J["head"]
    gj = ob.vertex_groups["jaw"]
    for v in ob.data.vertices:
        p = np.asarray(v.co, float)
        dom = max(v.groups, key=lambda g: g.weight).group if len(v.groups) else None
        if np.linalg.norm(p - (hc + dd * 0.5 * np.linalg.norm(b_ - a_))) > hr or \
                gi.get(dom) not in ("head", "head_end", "jaw", "neck5"):
            # outside the head's own skin: NO jaw weight at all (a horn tip, the chest, a hand)
            if any(gi.get(g.group) == "jaw" for g in v.groups):
                wj = next(g.weight for g in v.groups if gi.get(g.group) == "jaw")
                gj.remove([v.index])
                hg = ob.vertex_groups.get("head_end") or ob.vertex_groups.get("head")
                hg.add([v.index], wj, "ADD")
            continue
        below = float(np.dot(p - a_, nrm)) < 0
        if not below:
            w_ = 0.0
        elif p[0] >= cx:
            w_ = 1.0
        else:
            w_ = max(0.0, 1.0 - (cx - p[0]) / band)
        others = [(g.group, g.weight) for g in v.groups if gi.get(g.group) != "jaw"]
        tot = sum(w for _, w in others) or 1.0
        for g_, w in others:
            ob.vertex_groups[g_].add([v.index], w / tot * (1 - w_), "REPLACE")
        if w_ > 0:
            gj.add([v.index], w_, "REPLACE")
        elif any(gi.get(g.group) == "jaw" for g in v.groups):
            gj.remove([v.index])
            if not others:
                hg = ob.vertex_groups.get("head_end") or ob.vertex_groups.get("head")
                hg.add([v.index], 1.0, "REPLACE")
bpy.context.view_layer.objects.active = ob
bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
bpy.ops.object.vertex_group_normalize_all(lock_active=False)

# checks
bm = bmesh.new()
bm.from_mesh(ob.data)
nb = sum(1 for e in bm.edges if e.is_boundary)
nm = sum(1 for e in bm.edges if not e.is_manifold)
bm.free()
P = np.array([v.co[:] for v in ob.data.vertices])
lo_, hi_ = np.min(list(J.values()), axis=0) - 0.2 * S, np.max(list(J.values()), axis=0) + 0.2 * S
far = int(np.sum(np.any((P < lo_) | (P > hi_), axis=1)))
zero = sum(1 for v in ob.data.vertices if sum(g.weight for g in v.groups) < 1e-4)
checks = {"islands_in": len(isl), "islands": 1, "boundary_edges": nb, "nonmanifold_edges": nm, "far_verts": far,
          "zero_weight": zero, "zero_weight_fixed": nz, "verts": len(P), "faces": len(ob.data.polygons)}
checks["ok"] = nb == 0 and nm == 0 and far == 0 and zero == 0
print("SDF_CHECK", json.dumps(checks))
bpy.ops.wm.save_as_mainfile(filepath=str(out / "rigged.blend"))

# rig.json in the creature layout (human joint keys too, for the motion tools)
blen = float(np.linalg.norm(J["chest"] - J["pelvis"]))
anchors = {k: {"joint": a["joint"], "offset": (np.asarray(a["offset"]) * S / blen).round(4).tolist()}
           for k, a in SK.get("anchors", {}).items()}
alias = {"r_shoulder": "r_clav", "r_elbow": "r_uarm", "r_wrist": "r_farm", "r_hip": "r_hipb", "r_knee": "r_thigh",
         "r_ankle": "r_shin", "r_toe": "r_foot", "spine_mid": "spine", "head_end": "snout"}
joints = {k: v.round(4).tolist() for k, v in J.items()}
for a, b in alias.items():
    for s in ("r", "l"):
        a2, b2 = (a.replace("r_", s + "_", 1), b.replace("r_", s + "_", 1)) if a.startswith("r_") else (a, b)
        if b2 in J:
            joints[a2] = J[b2].round(4).tolist()
rj = {"template": SK["template"], "clean": True, "procedural": True, "height": S, "joints": joints, "anchors": anchors,
      "qa_poses": SK.get("qa_poses", {}), "skin": "thickness-normalised distance", "check": checks, "verts": len(P),
      "bones": [b[0] for b in SK["bones"]], "parents": {b[3]: b[2] for b in SK["bones"]}}
(out / "rig.json").write_text(json.dumps(rj, indent=1))
