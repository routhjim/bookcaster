"""Blender side: rig an image-to-3D CREATURE mesh to a shared template skeleton (tools/stage_templates/
creatures.json, written by tools/motion_templates.py), so creature motions retarget by joint name.

    blender -b --factory-startup -t 16 --python tools/blockout_creature_bl.py -- JOB.json

JOB {"blend": prepped model.blend (blockout_model_bl.py prep: Z-up, facing +X, feet at 0),
     "template": "winged_quadruped", "templates": "/abs/creatures.json",
     "stance": "upright"|"quadruped", "out_blend": rigged.blend, "out_json": rig.json}

Joint placement by mesh LANDMARKS (vertex statistics, any mesh of that template; no names of characters):
  body column = vertices near the centre plane; hind feet = lowest clusters either side; pelvis = top of
  the legs; chest/neck/head up the body column (head = the top of the column, toward +X); tail = the
  vertices behind the body, binned along -X; fore limbs = the lateral masses beside the chest, claws at
  their lowest point; wings (winged) = the lateral extremes above the shoulders: tip = highest outer point,
  fingers = outer points down the trailing edge, elbow/wrist on the leading edge.
Bones: one per non-root joint, from its parent joint to it (bone name = joint name); bone-heat skinning
(envelope fallback). rig.json = {"template", "joints": {name: xyz}, "anchors": {name: xyz}, "skin"}.
"""

import json
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
job = json.load(open(argv[0]))


def landmarks(V, tpl):
    """Template joints placed on an UPRIGHT winged/biped-creature mesh (or a quadruped: stance)."""
    z0, z1 = V[:, 2].min(), V[:, 2].max()
    H = z1 - z0
    J = {}
    core = V[np.abs(V[:, 1]) < 0.12 * H]
    # the body column: the front-most mass at each height (the tail is behind, low)
    def col(z, w=0.04):
        b = core[(core[:, 2] > z - w * H) & (core[:, 2] < z + w * H)]
        if len(b) < 10:
            return None
        xr = np.percentile(b[:, 0], [55, 100])
        b = b[b[:, 0] >= xr[0] - 0.25 * H]
        return np.array([np.median(b[:, 0]), 0.0, z])
    # hind feet: lowest clusters either side
    low = V[V[:, 2] < z0 + 0.05 * H]
    for s, sg in (("l", 1), ("r", -1)):
        side = low[low[:, 1] * sg > 0.02 * H]
        side = side if len(side) > 5 else low
        toe = side[np.argmax(side[:, 0])]
        heel = np.median(side, 0)
        J[f"hleg_{s}_toe"] = np.array([toe[0], heel[1], z0])
        J[f"hleg_{s}_ankle"] = np.array([heel[0], heel[1], z0 + 0.06 * H])
    # pelvis at ~38 % of the height (top of the legs in an upright stance)
    zp = z0 + 0.36 * H
    J["pelvis"] = col(zp)
    for s, sg in (("l", 1), ("r", -1)):
        J[f"hleg_{s}_hip"] = J["pelvis"] + np.array([0.0, sg * 0.07 * H, -0.02 * H])
        a = J[f"hleg_{s}_ankle"]
        k = 0.5 * (J[f"hleg_{s}_hip"] + a)
        J[f"hleg_{s}_knee"] = k + np.array([0.06 * H, 0, 0])            # knees forward
    zs = [0.45, 0.53, 0.60]
    for i, f in enumerate(zs):
        J[f"spine{i + 1}"] = col(z0 + f * H)
    J["chest"] = col(z0 + 0.66 * H)
    # neck up to the head: the top of the column; head forward
    top = core[core[:, 2] > z0 + 0.88 * H]
    hc = np.median(top, 0) if len(top) > 10 else J["chest"] + np.array([0.1 * H, 0, 0.25 * H])
    J["head"] = np.array([hc[0], 0.0, min(hc[2], z0 + 0.90 * H)])
    for i in range(4):
        t = (i + 1) / 5
        J[f"neck{i + 1}"] = J["chest"] * (1 - t) + J["head"] * t
    snout = top[np.argmax(top[:, 0])] if len(top) else J["head"] + np.array([0.08 * H, 0, 0])
    J["jaw"] = np.array([0.5 * (J["head"][0] + snout[0]), 0.0, J["head"][2] - 0.03 * H])
    # tail: everything behind the pelvis and below the chest, binned along -X
    # (below the pelvis: wing membranes hang behind the body higher up)
    tail = V[(V[:, 0] < J["pelvis"][0] - 0.10 * H) & (V[:, 2] < J["pelvis"][2]) & (np.abs(V[:, 1]) < 0.25 * H)]
    if len(tail) > 50:
        xs = np.linspace(J["pelvis"][0] - 0.08 * H, tail[:, 0].min(), 8)
        for i, x in enumerate(xs):
            b = tail[np.abs(tail[:, 0] - x) < 0.04 * H]
            if len(b):                      # the tail tube: the lowest dense band in this slice
                zc = np.percentile(b[:, 2], 30)
                b = b[np.abs(b[:, 2] - zc) < 0.06 * H]
            J[f"tail{i + 1}"] = np.median(b, 0) if len(b) else None
        pts = [J[f"tail{i + 1}"] for i in range(8)]
        good = [i for i, q in enumerate(pts) if q is not None]
        for i in range(8):                  # empty slices: interpolate between found neighbours
            if pts[i] is None:
                lo_ = max([g for g in good if g < i], default=None)
                hi_ = min([g for g in good if g > i], default=None)
                if lo_ is not None and hi_ is not None:
                    t = (i - lo_) / (hi_ - lo_)
                    J[f"tail{i + 1}"] = pts[lo_] * (1 - t) + pts[hi_] * t
                else:
                    J[f"tail{i + 1}"] = pts[lo_ if lo_ is not None else hi_]
    else:
        for i in range(8):
            J[f"tail{i + 1}"] = J["pelvis"] + np.array([-(i + 1) * 0.08 * H, 0, -0.03 * H * i])
    # fore limbs (arms of an upright creature): lateral masses beside/below the chest, in front
    for s, sg in (("l", 1), ("r", -1)):
        sh = J["chest"] + np.array([0.02 * H, sg * 0.09 * H, -0.02 * H])
        arm = V[(V[:, 1] * sg > 0.07 * H) & (V[:, 1] * sg < 0.20 * H) & (V[:, 2] > z0 + 0.30 * H) &
                (V[:, 2] < J["chest"][2]) & (V[:, 0] > J["pelvis"][0] - 0.02 * H)]
        claw = arm[np.argmin(arm[:, 2])] if len(arm) > 10 else sh + np.array([0.05 * H, 0, -0.25 * H])
        J[f"fleg_{s}_hip"] = sh
        J[f"fleg_{s}_toe"] = claw
        J[f"fleg_{s}_ankle"] = claw + (sh - claw) * 0.12
        J[f"fleg_{s}_knee"] = 0.5 * (sh + J[f"fleg_{s}_ankle"]) + np.array([-0.03 * H, sg * 0.01 * H, 0])
    # wings: lateral extremes above the shoulders
    if "wing_l_shoulder" in tpl["joints"]:
        for s, sg in (("l", 1), ("r", -1)):
            back = J["spine3"] + np.array([-0.08 * H, sg * 0.05 * H, 0.02 * H])
            wv = V[(V[:, 1] * sg > 0.15 * H)]
            tip = wv[np.argmax(wv[:, 2] + 0.3 * np.abs(wv[:, 1]))]
            outer = wv[np.argsort(-np.abs(wv[:, 1]))[:max(20, len(wv) // 50)]]
            ys = np.abs(wv[:, 1])
            J[f"wing_{s}_shoulder"] = back
            J[f"wing_{s}_wrist"] = back + (tip - back) * 0.70
            J[f"wing_{s}_elbow"] = back + (J[f"wing_{s}_wrist"] - back) * 0.45 + np.array([0, 0, 0.04 * H])
            J[f"wing_{s}_tip"] = tip
            # fingers: outer trailing-edge points at three heights below the wrist
            for i, f in enumerate((0.78, 0.6, 0.45)):
                zz = z0 + f * H
                b = wv[np.abs(wv[:, 2] - zz) < 0.05 * H]
                J[f"wing_{s}_finger{i + 1}"] = b[np.argmax(np.abs(b[:, 1]))] if len(b) else tip - np.array([0, 0, (i + 1) * 0.1 * H])
            del outer, ys
    for k in tpl["joints"]:
        if k not in J or J[k] is None:
            raise RuntimeError(f"landmark {k} not placed")
    return {k: np.asarray(J[k], float) for k in tpl["joints"]}


def _bvh(ob):
    from mathutils.bvhtree import BVHTree
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.transform(ob.matrix_world)
    t = BVHTree.FromBMesh(bm)
    bm.free()
    return t


def _hits(t, p, d, far=50.0):
    out, o = [], Vector(p) - Vector(d) * far
    for _ in range(64):
        loc, nrm, idx, dist = t.ray_cast(o, Vector(d), 2 * far)
        if loc is None:
            break
        out.append(loc.copy())
        o = loc + Vector(d) * 1e-4
    return out


def inside(t, p):
    """Inside a closed mesh: the nearest surface point's outward normal faces away from the joint."""
    p = Vector(p)
    loc, nrm, idx, dist = t.find_nearest(p)
    if loc is None:
        return False
    return (p - loc).dot(nrm) < 0 or dist < 1e-4


def snap_mid(t, p, axes=((0, 1, 0), (1, 0, 0))):
    """Move a joint to the middle of the mesh cross-section around it (both sides along each axis)."""
    q = Vector(p)
    for ax in axes:
        a = Vector(ax)
        hs = _hits(t, q, a)
        if len(hs) < 2:
            continue
        ss = sorted(hs, key=lambda h: (h - q).dot(a))
        neg = [h for h in ss if (h - q).dot(a) <= 0]
        pos = [h for h in ss if (h - q).dot(a) > 0]
        if neg and pos:
            lo_, hi_ = neg[-1], pos[0]                # the nearest wall on each side
            if (hi_ - lo_).length < 2.5:              # a real local cross-section, not across a gap
                q = q + a * (((lo_ + hi_) / 2 - q).dot(a))
    return np.array(q[:])


def resample(points, n):
    P = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    out = []
    for s_ in np.linspace(0, cum[-1], n):
        i = min(int(np.searchsorted(cum, s_, side="right")) - 1, len(P) - 2)
        u = (s_ - cum[i]) / max(seg[i], 1e-9)
        out.append(P[i] * (1 - u) + P[i + 1] * u)
    return out


def refine(J, ob, tpl):
    """Joints INSIDE the mesh: body/neck/tail/leg joints snapped to the cross-section middle; the tail
    resampled evenly along its centreline; wing joints (thin membrane) left on the membrane."""
    t = _bvh(ob)
    tail = [k for k in tpl["joints"] if k.startswith("tail")]
    def push_in(q):
        q = Vector(q)
        for _ in range(3):
            if inside(t, q):
                break
            loc, nrm, idx, dist = t.find_nearest(q)
            if loc is None:
                break
            q = loc - nrm * 0.06                     # just under the skin, then centred below
        return np.array(q[:])
    for k in tpl["joints"]:
        if k.startswith("wing_") and not k.endswith("shoulder"):
            continue
        mid = k in ("pelvis", "chest", "head", "jaw") or k.startswith(("spine", "neck"))
        if mid:                                       # bilateral body: the midline joints stay on y = 0
            q0 = np.array([J[k][0], 0.0, J[k][2]])
            q = snap_mid(t, push_in(q0), axes=((1, 0, 0),))
            q[1] = 0.0
            if not inside(t, q):
                q = push_in(q)
        else:
            q = snap_mid(t, push_in(J[k]))
        J[k] = q if inside(t, q) else push_in(q)
    if tail:
        pts = [J["pelvis"]] + [J[k] for k in tail]
        res = resample(pts, len(tail) + 1)[1:]
        for k, q in zip(tail, res):
            q = snap_mid(t, push_in(q))
            J[k] = q if inside(t, q) else push_in(q)
    return J, t


def rig_report(J, t, tpl):
    ins = {k: bool(inside(t, v)) for k, v in J.items() if not (k.startswith("wing_") and not k.endswith("shoulder"))}
    mono = {}
    for ch, js in tpl.get("chains", {}).items():
        js = [j for j in js if j in J]
        if len(js) < 3:
            continue
        d = [np.asarray(J[b]) - np.asarray(J[a]) for a, b in zip(js, js[1:])]
        back = sum(1 for a, b in zip(d, d[1:]) if np.dot(a, b) < 0)       # a chain doubling back
        mono[ch] = back
    return {"inside_pct": round(100 * sum(ins.values()) / max(len(ins), 1), 1),
            "outside": [k for k, v in ins.items() if not v], "chain_reversals": mono}


def fix_limb_bleed(ob, arm, take_from, give_to):
    """Bone heat bleeds between limbs that touch in the rest pose (a wing membrane hanging beside a fore leg
    gets the leg's weights, and tears when the leg lifts). For every vertex whose dominant bone is in
    take_from, if the nearest give_to bone SEGMENT is closer than the nearest take_from segment, move all its
    take_from weight onto that give_to bone. Generic by bone-name prefixes."""
    segs = {}
    for b in arm.data.bones:
        h = np.array(arm.matrix_world @ b.head_local)
        t = np.array(arm.matrix_world @ b.tail_local)
        segs[b.name] = (h, t)

    def sd(p, h, t):
        ab = t - h
        u = np.clip(((p - h) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
        return float(np.linalg.norm(p - (h + u * ab)))
    vg = {g.name: g for g in ob.vertex_groups}
    gi = {g.index: g.name for g in ob.vertex_groups}
    A = [n for n in segs if n.startswith(take_from) and n in vg]
    B = [n for n in segs if n.startswith(give_to) and n in vg]
    moved = 0
    for v in ob.data.vertices:
        if not v.groups:
            continue
        if not any(gi[g.group].startswith(take_from) and g.weight > 0.01 for g in v.groups):
            continue
        p = np.array(ob.matrix_world @ v.co)
        da = min(sd(p, *segs[n]) for n in A)
        db, nb = min((sd(p, *segs[n]), n) for n in B)
        if db < da:
            w = sum(g.weight for g in v.groups if gi[g.group].startswith(take_from))
            for g in list(v.groups):
                if gi[g.group].startswith(take_from):
                    vg[gi[g.group]].remove([v.index])
            vg[nb].add([v.index], w, "ADD")
            moved += 1
    print("BLEEDFIX", take_from, "->", give_to, moved)
    return moved


def _pt_tri_dist(P, a, b, c):
    """Distance from points P (n,3) to triangle abc (vectorised; exact, region by region)."""
    ab, ac, ap = b - a, c - a, P - a
    n = np.cross(ab, ac)
    nn = max(float(n @ n), 1e-12)
    # barycentric of the projection
    d00, d01, d11 = ab @ ab, ab @ ac, ac @ ac
    d20, d21 = ap @ ab, ap @ ac
    den = max(d00 * d11 - d01 * d01, 1e-12)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1 - v - w
    inside = (u >= 0) & (v >= 0) & (w >= 0)
    dplane = np.abs(ap @ n) / math.sqrt(nn)

    def seg(p, q):
        pq = q - p
        t = np.clip(((P - p) @ pq) / max(float(pq @ pq), 1e-12), 0, 1)
        return np.linalg.norm(P - (p + t[:, None] * pq), axis=1)
    dedge = np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return np.where(inside, dplane, dedge)


def membrane_dist(P, J, tpl):
    """Distance of each point to the template's wing-membrane polygons (fan triangles), per side."""
    out = {}
    for side, names in (tpl.get("membrane") or {}).items():
        pts = [np.asarray(J[n], float) for n in names if n in J]
        if len(pts) < 3:
            continue
        c = np.mean(pts, 0)
        d = np.full(len(P), 1e9)
        for i in range(len(pts)):
            d = np.minimum(d, _pt_tri_dist(P, c, pts[i], pts[(i + 1) % len(pts)]))
        out[side] = d
    return out


def segment_skin(ob, arm, J, tpl=None, k=4, power=2.0, no_wings=False):
    """Geometric skin for image-to-3D creature meshes (bone heat fails on them: on the upright dragon the jaw
    bone took the WHOLE head and the chest owned 2 vertices, so the torso followed the arms/wings).
    Each vertex -> its k nearest bone SEGMENTS, weights ~ 1/d^power. Region rules (generic, template names):
      * the head (vertices nearest head/jaw/head_end): split by the MOUTH LINE -- the line from the jaw hinge
        (head joint) toward the snout tip, lowered to the jaw joint's height at the jaw joint: below = jaw,
        above = head_end (skull), so opening the jaw opens only the lower jaw;
      * fore limbs never own a vertex BEHIND the limb column (rest pose, -X of the fore-limb hips minus the
        limb radius): that is wing membrane / back, it goes to the nearest non-limb bone.
    Returns the number of vertices skinned."""
    segs = []
    for b in arm.data.bones:
        h = np.array(arm.matrix_world @ b.head_local)
        t = np.array(arm.matrix_world @ b.tail_local)
        if b.parent is None and not b.name.startswith("pelvis"):
            continue
        segs.append((b.name, h, t))
    names = [n for n, _, _ in segs]
    H_ = np.array([h for _, h, _ in segs])
    T_ = np.array([t for _, _, t in segs])
    P = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
    AB = T_ - H_
    L2 = np.maximum((AB * AB).sum(1), 1e-9)
    D = np.empty((len(P), len(segs)))
    for i in range(0, len(P), 4096):
        p = P[i:i + 4096, None, :]
        u = np.clip(((p - H_[None]) * AB[None]).sum(2) / L2[None], 0, 1)
        D[i:i + 4096] = np.linalg.norm(p - (H_[None] + u[..., None] * AB[None]), axis=2)
    nm = np.array(names)
    limb = np.array([n.startswith("fleg_") for n in names])
    # fore-limb column: behind it = not a limb vertex
    fl = [J[j] for j in J if j.startswith("fleg_") and j.endswith("_hip")]
    if fl:
        lim_x = min(np.asarray(f)[0] for f in fl)
        own_l = D[:, limb].min(1) <= D[:, ~limb].min(1)
        rad = float(np.median(D[own_l][:, limb].min(1))) if own_l.any() else 0.1
        behind = P[:, 0] < lim_x - 1.5 * rad
        D[np.ix_(behind, limb)] = 1e9
        # limb FLESH lies within ~1.8 limb radii of its bone: anything farther (a membrane hanging beside
        # the arm, the flank) is never a fore-limb vertex
        far = D[:, limb].min(1) > 1.8 * rad
        D[np.ix_(far, limb)] = 1e9
        print("LIMB far-from-bone verts kept off the fore limbs:", int(far.sum()), "radius", round(rad, 3))
    # wing MEMBRANE: a vertex lying on a template membrane polygon (closer to the sheet than to any fore-limb
    # bone's own surface) is membrane: only the wing bones (and the body edge it is sewn to) may own it
    if tpl is not None and fl:
        md = membrane_dist(P, J, tpl)
        if md:
            dm = np.min(np.stack(list(md.values())), 0)
            dl = D[:, limb].min(1) if limb.any() else np.full(len(P), 1e9)
            on_sheet = (dm < 1.0 * rad) & (dm < 0.8 * dl)
            D[np.ix_(on_sheet, limb)] = 1e9
            print("MEMBRANE verts kept off the fore limbs:", int(on_sheet.sum()))
    # head split by the mouth line
    hi = [names.index(n) for n in ("head", "jaw", "head_end") if n in names]
    if len(hi) == 3 and "jaw" in J and "head" in J:
        hinge = np.asarray(J["head"], float)
        jawj = np.asarray(J["jaw"], float)
        tip = T_[names.index("head_end")]
        # the mouth line through the jaw joint, parallel to hinge->tip
        d = tip - hinge
        d[1] = 0
        d /= max(np.linalg.norm(d), 1e-9)
        near_head = np.argmin(D, 1)
        inhead = np.isin(near_head, hi) & (P[:, 0] > hinge[0] - 0.02)
        rel = P - jawj
        # signed height above the mouth line (in the x-z plane)
        above = rel[:, 2] - rel[:, 0] * (d[2] / max(abs(d[0]), 1e-6))
        jaw_i, sk_i = names.index("jaw"), names.index("head_end")
        D[np.ix_(inhead & (above < 0), [sk_i])] = 1e9
        D[np.ix_(inhead & (above >= 0), [jaw_i])] = 1e9
        for x in (names.index("head"),):
            D[np.ix_(inhead, [x])] = 1e9
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    vg = {n: ob.vertex_groups.new(name=n) for n in names}
    # TORSO vertices (nearest bone = spine/chest/pelvis/neck) never take limb weight in the blend: the chest
    # beside a resting arm otherwise follows the arm when it lifts (stretched slabs)
    if no_wings:                                       # a cleaned body: the wings are separate procedural parts
        wcols = np.array([n.startswith("wing_") for n in names])
        D[:, wcols] = 1e9
    core = np.array([n.startswith(("spine", "chest", "pelvis", "neck")) for n in names])
    near0 = np.argmin(D, 1)
    torso = core[near0]
    # the ARMPIT: torso vertices near a shoulder joint may blend with that arm (a smooth stretch, not a tear)
    for s_ in ("l", "r"):
        sh_ = J.get(f"fleg_{s_}_hip")
        if sh_ is not None:
            near_sh = np.linalg.norm(P - np.asarray(sh_, float), axis=1) < 0.08 * float(np.ptp(P[:, 2]))
            torso = torso & ~near_sh
    D[np.ix_(torso, limb | np.array([n.startswith("hleg_") for n in names]))] = 1e9
    idx = np.argsort(D, 1)[:, :k]
    # LIMB vertices (nearest bone is a fore/hind limb bone): sharp weights within their own limb chain and its
    # parent only (a smooth blend onto the torso smears a raised arm into a band under linear-blend skinning)
    par_ = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    chain_ = {}
    for n in names:
        if n.startswith(("fleg_", "hleg_")):
            side = n.split("_")[0] + "_" + n.split("_")[1]
            chain_[n] = {m for m in names if m.startswith(side + "_")} | {par_.get(side + "_hip")}
    for vi in range(len(P)):
        n0 = nm[idx[vi][0]]
        if n0 in chain_:
            allowed = [names.index(m) for m in chain_[n0] if m in names]
            order = sorted(allowed, key=lambda j: D[vi, j])[:2]
            dd = D[vi, order]
            w = 1.0 / np.maximum(dd, 1e-3) ** 4.0
            w = w / w.sum()
            for j_, w_, d_ in zip(order, w, dd):
                if w_ > 0.02 and d_ < 1e8:
                    vg[nm[j_]].add([vi], float(w_), "REPLACE")
            continue
        dd = D[vi, idx[vi]]
        w = 1.0 / np.maximum(dd, 1e-3) ** power
        w = w / w.sum()
        for j_, w_ in zip(idx[vi], w):
            if w_ > 0.02 and dd[list(idx[vi]).index(j_)] < 1e8:
                vg[nm[j_]].add([vi], float(w_), "REPLACE")
    if ob.find_armature() is None:
        mod = ob.modifiers.new("rig", "ARMATURE")
        mod.object = arm
    print("SEGSKIN", len(P), "verts", len(segs), "bones")
    return len(P)


def cut_fused(ob, arm):
    """Image-to-3D meshes weld limbs that touch at rest (a hanging fore limb fused to the thigh, the flank or a
    wing): those bridging faces stretch into a band when the limb moves. Delete every face whose vertices'
    dominant bones mix a fore limb with anything outside that limb's own chain and its torso attachment
    (chest, spine3, neck1, its clavicle). Returns the number of faces cut."""
    import bmesh
    gi = {g.index: g.name for g in ob.vertex_groups}
    dom = []
    for v in ob.data.vertices:
        dom.append(gi[max(v.groups, key=lambda g: g.weight).group] if v.groups else None)

    def grp(n):
        if n is None:
            return None
        for pre in ("fleg_l", "fleg_r"):
            if n.startswith(pre):
                return pre
        return "other:" + n
    ok_attach = {"chest", "spine3", "neck1"}
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    cut = []
    for f in bm.faces:
        gs = {grp(dom[v.index]) for v in f.verts}
        limbs = {g for g in gs if g and not g.startswith("other:")}
        if not limbs:
            continue
        others = {g[6:] for g in gs if g and g.startswith("other:")}
        if len(limbs) > 1 or (others - ok_attach):
            cut.append(f)
    bmesh.ops.delete(bm, geom=cut, context="FACES_ONLY")
    # cap the openings the cut leaves (the flank under the arm, the arm's inner side): small boundary loops only
    bnd = [e for e in bm.edges if e.is_boundary]
    filled = bmesh.ops.holes_fill(bm, edges=bnd, sides=300)
    print("FUSED holes capped:", len(filled.get("faces", [])))
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    print("FUSED faces cut:", len(cut))
    return len(cut)


def symmetry_repair(J, H):
    """Paired limbs must mirror each other (y -> -y): when one side's chain was placed on the wrong mass (a fore
    limb landmarked on a hanging wing, 0.5 m behind the other arm), replace it by the mirror of the better side.
    Fore limbs: the better side is the one further FORWARD (+X, the facing); others: the side nearer the body
    midline. Clavicle (*_hip) joints keep a minimum lateral offset. Returns the repaired chains."""
    fixed = []
    for pre in ("fleg", "hleg"):
        ks = [k for k in ("hip", "knee", "ankle", "toe") if f"{pre}_l_{k}" in J and f"{pre}_r_{k}" in J]
        if not ks:
            continue
        L = np.array([J[f"{pre}_l_{k}"] for k in ks])
        R = np.array([J[f"{pre}_r_{k}"] for k in ks])
        err = float(np.mean(np.linalg.norm(L - R * np.array([1, -1, 1]), axis=1)))
        if err < 0.1 * H:
            continue
        if pre == "fleg":
            good = "l" if L[:, 0].mean() >= R[:, 0].mean() else "r"
        else:
            good = "l" if np.abs(L[:, 1]).mean() <= np.abs(R[:, 1]).mean() else "r"
        bad = "r" if good == "l" else "l"
        for k in ks:
            g = np.asarray(J[f"{pre}_{good}_{k}"], float)
            J[f"{pre}_{bad}_{k}"] = np.array([g[0], -g[1], g[2]])
        fixed.append(f"{pre}_{bad} <- mirror of {pre}_{good} (error {err:.2f} m)")
    for s_, sg in (("l", 1), ("r", -1)):
        k = f"fleg_{s_}_hip"
        if k in J and sg * J[k][1] < 0.04 * H:
            J[k] = np.array([J[k][0], sg * 0.04 * H, J[k][2]])
            fixed.append(f"{k} lateral offset")
    return fixed


def main():
    tpls = json.load(open(job["templates"]))["templates"]
    tpl = tpls[job["template"]]
    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    ob = bpy.data.objects["model"]
    V = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
    if job.get("joints_json"):                        # a CLEANED mesh: its joints come from the cleanup
        JJ = json.load(open(job["joints_json"]))
        J = {k: np.asarray(v, float) for k, v in JJ["joints"].items() if k in tpl["joints"]}
        SNOUT = np.asarray(JJ["snout_tip"], float) if JJ.get("snout_tip") else None
        if job.get("frozen"):                           # a FROZEN skeleton: the joints are used exactly
            _, bvh = refine(dict(J), ob, tpl)
        else:
            J, bvh = refine(J, ob, tpl)                    # snap the carried joints INTO the cleaned mesh
    else:
        SNOUT = None
        J = landmarks(V, tpl)
        J, bvh = refine(J, ob, tpl)
        H_ = float(V[:, 2].max() - V[:, 2].min())
        for msg in symmetry_repair(J, H_):
            print("SYMMETRY", msg)
        # (no second refine: its snapping pulls repaired joints back onto the midline)
    report = rig_report(J, bvh, tpl)
    print("RIGCHECK", json.dumps(report))
    ad = bpy.data.armatures.new("rig")
    arm = bpy.data.objects.new("rig", ad)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    par = tpl["parents"]
    for j in tpl["joints"]:
        p = par.get(j)
        if p is None:
            b = ad.edit_bones.new(j)
            b.head = Vector(J[j])
            b.tail = Vector(J[j] + np.array([0.0, 0.0, 0.05 * (V[:, 2].max() - V[:, 2].min())]))
        else:
            b = ad.edit_bones.new(j)
            b.head, b.tail = Vector(J[p]), Vector(J[j])
            if (b.tail - b.head).length < 1e-3:
                b.tail = b.head + Vector((0, 0, 0.02))
            b.parent = eb[p]
        eb[j] = b
    # helper LEAF bones (rig-only, not template joints; they follow their parent when posed): the skull
    # beyond the head joint (so a jaw bone opens only the lower jaw) and the claws beyond the toes
    Hh = float(V[:, 2].max() - V[:, 2].min())
    tpa = tpl.get("anchors", {})
    blen = float(np.linalg.norm(np.asarray(J["chest"]) - np.asarray(J["pelvis"]))) if "chest" in J else 0.2 * Hh
    extra = {}
    if "head" in J and "snout" in tpa:
        tip = SNOUT if SNOUT is not None else np.asarray(J["head"]) + np.asarray(tpa["snout"]["offset"]) * blen
        # keep the skull tip above the jaw joint so the two bones split upper/lower head
        tip[2] = max(tip[2], np.asarray(J["jaw"])[2] + 0.04 * Hh) if "jaw" in J else tip[2]
        extra["head_end"] = ("head", J["head"], tip)
    for j in tpl["joints"]:
        if j.endswith("_toe") and j.startswith("fleg"):
            pj = par[j]
            d_ = np.asarray(J[j]) - np.asarray(J[pj])
            extra[j + "_end"] = (j, J[j], np.asarray(J[j]) + d_ * 0.6)
    for name, (pname, h_, t_) in extra.items():
        b = ad.edit_bones.new(name)
        b.head, b.tail = Vector(h_), Vector(t_)
        b.parent = eb[pname]
    bpy.ops.object.mode_set(mode="OBJECT")
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
        how = f"envelope (bone heat left {empty} unweighted)"
        for vg in list(ob.vertex_groups):
            ob.vertex_groups.remove(vg)
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    if job.get("skin", "segments") == "segments":
        segment_skin(ob, arm, J, tpl, no_wings=bool(job.get("joints_json")))
        if job.get("cut_fused", True):                 # a cleaned (creature_clean_bl) mesh has no welds left
            cut_fused(ob, arm)
        how = "segments"
        empty = sum(1 for v in ob.data.vertices if not v.groups)
    else:
        fix_limb_bleed(ob, arm, ("fleg_",), ("wing_",))
    bpy.ops.wm.save_as_mainfile(filepath=job["out_blend"])
    # per-bone radius: the median distance of the vertices the bone owns (weight > 0.5) to its segment
    # (creature VOLUMES for the cleanup: props and people must not enter them)
    gi = {g.index: g.name for g in ob.vertex_groups}
    own = {}
    for v in ob.data.vertices:
        if v.groups:
            g = max(v.groups, key=lambda x: x.weight)
            if g.weight > 0.5:
                own.setdefault(gi[g.group], []).append(np.array(ob.matrix_world @ v.co))
    radii = {}
    for j in tpl["joints"]:
        p_ = par.get(j)
        if p_ is None or j not in own:
            continue
        a_, b_ = np.asarray(J[p_]), np.asarray(J[j])
        ab = b_ - a_
        P_ = np.array(own[j])
        t_ = np.clip(((P_ - a_) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
        d_ = np.linalg.norm(P_ - (a_ + t_[:, None] * ab), axis=1)
        radii[j] = round(float(np.median(d_)), 4)
    anchors = {}
    for name, a in tpl.get("anchors", {}).items():
        anchors[name] = {"joint": a["joint"], "offset": a["offset"]}
    json.dump({"template": job["template"], "clean": bool(job.get("joints_json")),
               "joints": {k: v.tolist() for k, v in J.items()}, "anchors": anchors,
               "skin": how, "unweighted": empty, "verts": len(ob.data.vertices), "check": report, "radii": radii,
               "height": float(V[:, 2].max() - V[:, 2].min())}, open(job["out_json"], "w"), indent=1)
    print("RIG", how, empty, len(ob.data.vertices))


main()
