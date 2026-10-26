"""Blender side: rebuild the parts an image-to-3D creature mesh cannot articulate (any template creature).

    blender -b --factory-startup -t 8 --python tools/creature_parts_bl.py -- RIGGED.blend RIG.json TEMPLATES.json OUT.blend

Image-to-3D (Hunyuan3D) fuses a bat wing into one thick shell with the body and has no mouth: skinning tears the
wing and a jaw bone has nothing to open. After tools/blockout_creature_bl.py has rigged the mesh:
  WINGS  the fused wing shell (vertices owned by wing bones beyond the shoulder root) is deleted and the
         openings capped; each wing is rebuilt PROCEDURALLY on the template's wing bones: a bat membrane fanned
         from the wrist over the template "membrane" outline (tip, fingers, the trailing edge sewn to the body at
         spine2/pelvis, shoulder, elbow), scalloped between finger tips, skinned by barycentric weights to the
         corner bones (it stretches with the fingers and can never tear), plus bone tubes along the arm and the
         fingers. Colours come from the nearest vertex of the old wing shell (the drawing's paint survives).
  JAW    the head is cut along the MOUTH LINE (through the jaw joint, parallel to the head->snout direction):
         the part below and in front of the hinge is the lower jaw (rigid on the jaw bone), the rest the skull
         (head_end); both cut faces are capped with a dark mouth interior and lined with teeth, so an opening
         of 25-40 degrees reads as an open mouth.
Writes OUT.blend (objects "model" + "rig", same armature, same template joints).
"""

import json
from pathlib import Path
import re
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

argv = sys.argv[sys.argv.index("--") + 1:]
blend, rigj, tpls, out = argv[:4]
bpy.ops.wm.open_mainfile(filepath=blend)
ob = bpy.data.objects["model"]
arm = bpy.data.objects["rig"]
RJ = json.load(open(rigj))
J = {k: np.asarray(v, float) for k, v in RJ["joints"].items()}
TPL = json.load(open(tpls))["templates"][RJ["template"]]
H = float(RJ.get("height", 6.0))
mw = ob.matrix_world
assert all(abs(mw[i][j] - (1.0 if i == j else 0.0)) < 1e-6 for i in range(4) for j in range(4)), "model has a transform"
gi = {g.index: g.name for g in ob.vertex_groups}


def dominant(v):
    return gi[max(v.groups, key=lambda g: g.weight).group] if v.groups else None


def vcol_layer(me):
    a = me.color_attributes.get("Col")
    return a


# ------------------------------------------------------------------------------------------------ colours
col_attr = vcol_layer(ob.data)
old_wing = {"l": [], "r": []}
for v in ob.data.vertices:
    d = dominant(v)
    if d and d.startswith("wing_") and not d.endswith("_shoulder"):
        s = d.split("_")[1]
        c = tuple(col_attr.data[v.index].color[:3]) if col_attr else (0.3, 0.25, 0.22)
        old_wing[s].append((Vector(v.co), c))
kd = {}
for s, lst in old_wing.items():
    t = KDTree(max(len(lst), 1))
    for i, (p, _) in enumerate(lst):
        t.insert(p, i)
    t.balance()
    kd[s] = (t, lst)


def wing_colour(s, p):
    t, lst = kd[s]
    if not lst:
        return (0.3, 0.25, 0.22)
    near = t.find_n(Vector(p), 24)                     # a smooth average: the shell's paint is patchy
    cs = np.array([lst[i][1] for _, i, _ in near])
    c = np.median(cs, 0)
    return (float(c[0]) * 0.9, float(c[1]) * 0.9, float(c[2]) * 0.9)


# ------------------------------------------------------------------------------------------------ cut the wing shell
bm = bmesh.new()
bm.from_mesh(ob.data)
bm.verts.ensure_lookup_table()
dl = bm.verts.layers.deform.verify()
wing_bones = {n for n in gi.values() if n.startswith("wing_") and not n.endswith("_shoulder")}
gidx = {n: i for i, n in gi.items()}
kill = []
for v in bm.verts:
    dv = v[dl]
    if not dv:
        continue
    top = max(dv.items(), key=lambda kv: kv[1])
    if gi[top[0]] in wing_bones:
        kill.append(v)
# the wing shoulder root that sticks out beyond the back: also wing (keep only what is near the body column)
for v in bm.verts:
    dv = v[dl]
    if not dv:
        continue
    top = max(dv.items(), key=lambda kv: kv[1])
    n = gi[top[0]]
    if n.endswith("_shoulder") and n.startswith("wing_"):
        s = n.split("_")[1]
        if np.linalg.norm(np.asarray(v.co) - J[f"wing_{s}_shoulder"]) > 0.08 * H:
            kill.append(v)
if RJ.get("clean"):                                   # a cleaned mesh has no fused wing shell left
    kill = []
bmesh.ops.delete(bm, geom=list(set(kill)), context="VERTS")
bnd = [e for e in bm.edges if e.is_boundary]
capped = bmesh.ops.holes_fill(bm, edges=bnd, sides=400)
print("PARTS wing shell verts removed:", len(set(kill)), "caps:", len(capped.get("faces", [])))

# ------------------------------------------------------------------------------------------------ flaps
# THIN-SHELL FLAPS (generic): a double-sided sheet of skin on the trunk (an apron left from a fused limb, a torn
# neck edge): a vertex whose inward ray hits the OPPOSITE side of the SAME body segment within ~3.5 cm (scaled by
# the creature's height) is in a flap. Clusters of them on the trunk/neck FRONT and sides (never the dorsal
# spikes, horns, claws, wings, tail tip) are deleted and each hole capped and relaxed.
from mathutils.bvhtree import BVHTree


def _dom(v):
    return max(v[dl].items(), key=lambda kv: kv[1])[0] if len(v[dl]) else None


TRUNK = re.compile(r"^(spine|chest|pelvis|neck|hip|belly|torso|root)")
bm.normal_update()
bm.verts.ensure_lookup_table()
bm.faces.ensure_lookup_table()
tree = BVHTree.FromBMesh(bm)
thin = float(RJ.get("flap_thickness", 0.035 * H / 6.0))
spine_pts = np.array([J[k] for k in J if TRUNK.match(k)] or [J.get("pelvis", np.zeros(3))])
fwd = np.array([1.0, 0.0, 0.0])
if "head" in J and "pelvis" in J:
    fwd = np.array([J["head"][0] - J["pelvis"][0], 0.0, 0.0])
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
flap = set()
for v in bm.verts:
    g = _dom(v)
    if g is None or not TRUNK.match(gi[g]):
        continue
    p_ = np.asarray(v.co, float)
    k_ = int(np.argmin(np.linalg.norm(spine_pts - p_, axis=1)))
    if np.dot(p_ - spine_pts[k_], fwd) < -0.02 * H:
        continue                                        # the back (dorsal spikes are legitimately thin)
    n_ = v.normal
    if n_.length < 1e-6:
        continue
    hit = tree.ray_cast(v.co - n_ * 1e-4, -n_, thin)
    if hit[0] is None or hit[2] is None:
        continue
    fo = bm.faces[hit[2]]
    if any(_dom(u) == g for u in fo.verts):
        flap.add(v)
# clusters >= 15 verts only (a lone thin spot is a crease, not a flap)
lab, clusters = {}, []
for v in flap:
    if v in lab:
        continue
    st_, mem = [v], []
    lab[v] = 1
    while st_:
        x = st_.pop()
        mem.append(x)
        for e in x.link_edges:
            y = e.other_vert(x)
            if y in flap and y not in lab:
                lab[y] = 1
                st_.append(y)
    clusters.append(mem)
dead = [v for m in clusters if len(m) >= 15 for v in m]
before_b = {e for e in bm.edges if e.is_boundary}
bmesh.ops.delete(bm, geom=dead, context="VERTS")
newb = [e for e in bm.edges if e.is_boundary and e not in before_b]
filled = bmesh.ops.holes_fill(bm, edges=newb, sides=0).get("faces", [])
if filled:
    bmesh.ops.triangulate(bm, faces=filled)


def fan_fill(bm, edges):
    """Close every open loop among edges with a fan around its centre (weights averaged): holes_fill gives up on
    non-planar / self-touching loops."""
    left = {e for e in edges if e.is_valid and e.is_boundary}
    made = 0
    while left:
        e0 = left.pop()
        loop = [e0.verts[0], e0.verts[1]]
        prev = e0
        ok = True
        while loop[-1] is not loop[0]:
            nxt = [e for e in loop[-1].link_edges if e.is_boundary and e is not prev and e in left]
            if not nxt:
                ok = False
                break
            prev = nxt[0]
            left.discard(prev)
            loop.append(prev.other_vert(loop[-1]))
        if not ok or len(loop) < 4:
            continue
        loop = loop[:-1]
        if len(loop) <= 4:
            try:
                bm.faces.new(loop)
            except ValueError:                          # a degenerate sliver: collapse it to a point
                bmesh.ops.pointmerge(bm, verts=loop, merge_co=Vector(np.mean([np.asarray(v.co) for v in loop], 0)))
            made += 1
            continue
        cv = bm.verts.new(Vector(np.mean([np.asarray(v.co) for v in loop], 0)))
        w = {}
        for v in loop:
            for g_, x in v[dl].items():
                w[g_] = w.get(g_, 0.0) + x / len(loop)
        for g_, x in w.items():
            cv[dl][g_] = x
        for i_ in range(len(loop)):
            try:
                bm.faces.new((loop[i_], loop[(i_ + 1) % len(loop)], cv))
            except ValueError:
                pass
        made += 1
    return made


extra_caps = fan_fill(bm, [e for e in bm.edges if e.is_boundary])   # the body is closed (sheets come later)
bm.normal_update()
bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
ring = {v for e in newb if e.is_valid for v in e.verts}
for _ in range(6):
    ring2 = set(ring)
    for v in ring:
        for e in v.link_edges:
            ring2.add(e.other_vert(v))
    ring = ring2 if len(ring2) < 20000 else ring
for _ in range(8):                                     # plain neighbour averaging (Laplacian with volume
    bmesh.ops.smooth_vert(bm, verts=list(ring), factor=0.5,   # preservation exploded on fan caps: verts at -88 m)
                          use_axis_x=True, use_axis_y=True, use_axis_z=True)
FLAP_STATS = {"thin_verts": len(flap), "clusters": len(clusters), "deleted_verts": len(dead),
              "deleted_clusters": sum(1 for m in clusters if len(m) >= 15), "caps": len(filled), "fan_caps": extra_caps}
print("PARTS flaps", json.dumps(FLAP_STATS))

# ------------------------------------------------------------------------------------------------ mirror weights
# a limb REBUILT by mirroring (creature_clean mirror_arm) gets the SOURCE limb's weights mirrored (y -> -y,
# _l_ <-> _r_), not whatever the auto-weights gave the new geometry (rig.json "mirror_weights": "l" = copy l -> r)
_ms = RJ.get("mirror_weights")
if _ms in ("l", "r"):
    _md = "r" if _ms == "l" else "l"
    def _swap(n):
        return re.sub(rf"(^|_){_ms}(_|$)", rf"\g<1>{_md}\g<2>", n)
    src_v = [v for v in bm.verts if _dom(v) is not None and re.search(rf"(^|_){_ms}(_|$)", gi[_dom(v)])
             and not gi[_dom(v)].startswith("wing")]
    kd_m = KDTree(max(len(src_v), 1))
    for i_, v in enumerate(src_v):
        kd_m.insert(Vector((v.co.x, -v.co.y, v.co.z)), i_)
    kd_m.balance()
    n_m = 0
    for v in bm.verts:
        g = _dom(v)
        if g is None or not re.search(rf"(^|_){_md}(_|$)", gi[g]) or gi[g].startswith("wing"):
            continue
        _, k_, dist_ = kd_m.find(v.co)
        if k_ is None or dist_ > 0.03 * H:
            continue
        sv = src_v[k_]
        v[dl].clear()
        for g_, w_ in sv[dl].items():
            nm_ = _swap(gi[g_])
            if nm_ in gidx:
                v[dl][gidx[nm_]] = w_
        n_m += 1
    print("PARTS mirrored limb weights", _ms, "->", _md, n_m, "verts")

# VALIDITY (generic): no vertex flung away from the body, no zero-weight vertex, no absurd edge (fills / smoothing)
_bb = np.array([v.co[:] for v in bm.verts])
_lo, _hi = np.percentile(_bb, 0.5, axis=0) - 0.1 * H, np.percentile(_bb, 99.5, axis=0) + 0.1 * H
_far = [v for v in bm.verts if np.any(np.asarray(v.co) < _lo) or np.any(np.asarray(v.co) > _hi)]
if _far:
    bmesh.ops.delete(bm, geom=_far, context="VERTS")
    fan_fill(bm, [e for e in bm.edges if e.is_boundary])
for v in bm.verts:
    if sum(v[dl].values()) < 1e-6:                     # a new vertex without weights: its neighbours' average
        nb = [e.other_vert(v) for e in v.link_edges]
        acc = {}
        for u in nb:
            for g_, w_ in u[dl].items():
                acc[g_] = acc.get(g_, 0.0) + w_ / max(len(nb), 1)
        for g_, w_ in acc.items():
            v[dl][g_] = w_
_long = sum(1 for e in bm.edges if e.calc_length() > 0.05 * H)
VALID = {"far_verts_removed": len(_far), "long_edges": _long,
         "zero_weight": sum(1 for v in bm.verts if sum(v[dl].values()) < 1e-6)}
print("PARTS validity", json.dumps(VALID))

# ------------------------------------------------------------------------------------------------ jaw split
jaw_ok = all(k in J for k in ("head", "jaw")) and "jaw" in gidx
n_caps = 0
if jaw_ok:
    hinge = J["head"]
    jj = J["jaw"]
    tipb = arm.data.bones.get("head_end")
    tip = np.array(arm.matrix_world @ tipb.tail_local) if tipb else hinge + (jj - hinge) * 2.5
    d = tip - hinge
    d[1] = 0
    d /= max(np.linalg.norm(d), 1e-9)
    # plane through the jaw joint (at the midline), containing Y and d: normal = d x Y
    p0 = np.array([jj[0], 0.0, jj[2]])
    nrm = np.cross(d, np.array([0.0, 1.0, 0.0]))
    if nrm[2] < 0:
        nrm = -nrm
    head_r = 0.8 * float(np.linalg.norm(tip - hinge)) + 0.05 * H     # reaches the SNOUT TIP (the mouth must be cut to the front)
    # only the head region: faces whose centre is in front of the hinge and near the head
    HEADG = {gidx[n] for n in ("head", "head_end", "jaw") if n in gidx}

    def _headish(f):                                    # the HEAD's own skin (never a forearm raised near the throat)
        return sum(1 for v in f.verts if len(v[dl]) and max(v[dl].items(), key=lambda kv: kv[1])[0] in HEADG) \
            >= 0.5 * len(f.verts)
    region = [f for f in bm.faces
              if (np.asarray(f.calc_center_median()) - hinge)[0] > -0.02 * H
              and np.linalg.norm(np.asarray(f.calc_center_median()) - (hinge + tip) / 2) < head_r and _headish(f)]
    region_set = set(region)
    geom = list({e for f in region for e in f.edges}) + region + list({v for f in region for v in f.verts})
    res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=Vector(p0), plane_no=Vector(nrm), clear_inner=False,
                                 clear_outer=False)
    cut_edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
    mouth_pts = np.array([np.asarray(v.co, float) for e in cut_edges for v in e.verts])
    bmesh.ops.split_edges(bm, edges=cut_edges)
    # classify: lower jaw = head-region faces below the plane and in front of the hinge
    bm.faces.ensure_lookup_table()
    lower = set()
    for f in bm.faces:
        c = np.asarray(f.calc_center_median())
        # the LOWER JAW: below the mouth plane, in FRONT of the hinge by a margin (never the throat behind it)
        if (c - hinge)[0] > 0.12 * float(np.linalg.norm(tip - hinge)) and \
                np.linalg.norm(c - (hinge + tip) / 2) < head_r and np.dot(c - p0, nrm) < 0 and _headish(f) and \
                (np.dot(c - p0, nrm) > -float(RJ.get("jaw_depth", 0.3)) * float(np.linalg.norm(tip - hinge)) or
                 float(np.dot(c - hinge, d)) > 0.55 * float(np.linalg.norm(tip - hinge))):   # the chin: all jaw
            lower.add(f)                                # (the jaw's own depth: the throat below it stretches)
    # weights: lower jaw rigid on "jaw", the rest of the head region on "head_end"
    jaw_g = gidx["jaw"]
    sk_g = gidx.get("head_end", gidx.get("head"))
    lv = {v for f in lower for v in f.verts}
    for v in lv:
        v[dl].clear()
        v[dl][jaw_g] = 1.0
    # the UPPER head is RIGID on the skull bone (no neck weight reaching into cheeks, eyes, snout: no face
    # stretch); only a seam band just past the neck plane keeps a blend with the neck
    nk = J.get("neck4", hinge)
    nd = (hinge - J.get("neck3", nk))
    nd = nd / max(np.linalg.norm(nd), 1e-9)
    Lh = float(np.linalg.norm(tip - hinge))
    up_v = set()
    for f in bm.faces:
        if f in lower:
            continue
        c = np.asarray(f.calc_center_median())
        if np.linalg.norm(c - (hinge + tip) / 2) < head_r and np.dot(c - nk, nd) > 0:
            for v in f.verts:
                if v not in lv:
                    up_v.add(v)
    for v in up_v:
        dpl = float(np.dot(np.asarray(v.co) - nk, nd))
        if dpl > 0.18 * Lh:
            v[dl].clear()
            v[dl][sk_g] = 1.0
        else:                                       # seam band: blend neck -> skull
            w_ = dpl / (0.18 * Lh)
            for g_ in list(v[dl].keys()):
                v[dl][g_] *= (1 - w_)
            v[dl][sk_g] = v[dl].get(sk_g, 0.0) + w_
    # STRETCH BAND (generic): the skin next to the rigid lower jaw (the throat and cheek below the mouth line, and
    # the mouth corners) blends jaw <-> skull/neck over a band, so opening the jaw STRETCHES that skin instead of
    # tearing it along a hard weight step; the jaw itself (and the teeth added later) stay rigid
    from mathutils.kdtree import KDTree as _KD
    if lv:
        kdj = _KD(len(lv))
        for i_, v in enumerate(lv):
            kdj.insert(v.co, i_)
        kdj.balance()
        band = float(RJ.get("jaw_band", 0.3)) * Lh
        corners = []
        if len(mouth_pts):
            adm = (mouth_pts - p0) @ d
            for sy in (-1, 1):
                sd_ = mouth_pts[(mouth_pts[:, 1] * sy) > 0]
                if len(sd_):
                    corners.append(sd_[np.argmin((sd_ - p0) @ d)])
        a_corner = float(min(((q - p0) @ d) for q in corners)) if corners else -1e9
        nb_ = 0
        for v in bm.verts:
            if v in lv:
                continue
            c = np.asarray(v.co, float)
            if np.linalg.norm(c - (hinge + tip) / 2) > head_r * 1.3:
                continue
            depth = -float(np.dot(c - p0, nrm))                     # > 0 below the mouth plane
            dist_ = kdj.find(v.co)[2]
            fall = max(0.0, 1.0 - dist_ / band) * 0.85
            if float(np.dot(c - p0, d)) > a_corner + 0.02 * Lh:
                # the CUT region (lips): the lower side follows the jaw, the upper lip stays on the skull
                w_ = fall if depth > 0.003 * H else 0.0     # ON the cut (upper-lip copies of the split): skull
            else:
                # BEHIND the mouth corners the skin is uncut: blend smoothly ACROSS the mouth plane too (no shear
                # line where above-plane skin has 0 jaw weight and the skin just below has 0.85)
                w_ = fall * float(np.clip(0.5 + depth / (0.24 * Lh), 0.0, 1.0))
            if w_ <= 0.01:
                continue
            for g_ in list(v[dl].keys()):
                v[dl][g_] *= (1 - w_)
            v[dl][jaw_g] = v[dl].get(jaw_g, 0.0) + w_
            nb_ += 1
        print("PARTS jaw stretch band verts", nb_, "band", round(band, 3))
        # NO stray jaw weight anywhere else (a horn or neck vertex the rigging gave some jaw weight tears open)
        band_v = set()
        nstray = 0
        for v in bm.verts:
            if v in lv or jaw_g not in v[dl]:
                continue
            c = np.asarray(v.co, float)
            if kdj.find(v.co)[2] < band:
                continue                                # the stretch band itself
            w_ = v[dl][jaw_g]
            del v[dl][jaw_g]
            v[dl][sk_g] = v[dl].get(sk_g, 0.0) + w_
            nstray += 1
        print("PARTS jaw stray weights moved to the skull", nstray)
    # caps along the cut: the open boundary loops at the mouth line -> mouth interior faces
    bnd = [e for e in bm.edges if e.is_boundary and abs(np.dot(np.asarray(e.verts[0].co) - p0, nrm)) < 0.03 * H]
    # the mouth loop is capped as TWO halves (the palate on the skull, the tongue floor on the jaw), never one face
    # across both lips (that face becomes a torn sheet when the jaw opens): a fan from each half's centre
    caps = []
    for half_lower in (False, True):
        es = [e for e in bnd if all((v in lv) == half_lower for v in e.verts) or
              (half_lower and any(v in lv for v in e.verts) and not all(v in lv for v in e.verts))]
        es = [e for e in bnd if sum(1 for v in e.verts if v in lv) >= (1 if half_lower else 0) and
              (half_lower or not any(v in lv for v in e.verts))]
        if not es:
            continue
        vs = {v for e in es for v in e.verts}
        cen = np.mean([np.asarray(v.co) for v in vs], 0)
        # sink the centre into the mouth (toward the head's midline, off the cut plane) so the cap reads as a cavity
        cen = cen - nrm * (0.01 * H if not half_lower else -0.01 * H) * 0.0
        cv = bm.verts.new(Vector(cen))
        g_ = jaw_g if half_lower else sk_g
        cv[dl][g_] = 1.0
        for e in es:
            try:
                caps.append(bm.faces.new((e.verts[0], e.verts[1], cv)))
            except ValueError:
                pass
    fan_fill(bm, [e for e in bm.edges if e.is_boundary])   # any small gap left at the mouth corners
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    n_caps = len(caps)
    print("PARTS jaw: lower faces", len(lower), "mouth caps", len(caps))

bm.to_mesh(ob.data)
bm.free()
ob.data.update()

# mouth interior material on the caps (dark red), by face index
if jaw_ok:
    mi = bpy.data.materials.new("mouth")
    mi.use_nodes = True
    mi.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.18, 0.03, 0.03, 1)
    mi.diffuse_color = (0.18, 0.03, 0.03, 1)
    ob.data.materials.append(mi)
    mslot = len(ob.data.materials) - 1
    # recolour the mouth caps (vertex colour too, for Workbench); the mesh was rebuilt: fetch the layer again
    col_attr = ob.data.color_attributes.get("Col")
    if n_caps:
        for p in ob.data.polygons:
            c = np.asarray(p.center)
            if abs(np.dot(c - p0, nrm)) < 0.012 * H and (c - J["head"])[0] > 0:
                p.material_index = mslot
                if col_attr:
                    for vi in p.vertices:
                        col_attr.data[vi].color = (0.18, 0.03, 0.03, 1.0)

# ------------------------------------------------------------------------------------------------ new parts
parts_v, parts_f, parts_w, parts_c = [], [], [], []      # vertices, faces, [(bone, w)...], colour


def add_tri_patch(A, B, C, wa, wb, wc, n, colf, scallop=None):
    """Subdivided triangle A,B,C (rows from A); weights blended barycentrically. scallop: (u-edge index) the
    B-C edge bows toward A by that fraction at its middle."""
    base = len(parts_v)
    idx = {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            b = i / n
            c = j / n
            a = 1 - b - c
            p = a * A + b * B + c * C
            if scallop and i + j == n and 0 < i < n:          # on the B-C edge
                t = i / n
                p = p + (A - p) * scallop * 4 * t * (1 - t)
            w = {}
            for wd, k in ((wa, a), (wb, b), (wc, c)):
                for bn, bw in wd.items():
                    w[bn] = w.get(bn, 0.0) + bw * k
            idx[(i, j)] = len(parts_v)
            parts_v.append(p)
            parts_w.append(w)
            parts_c.append(colf(p))
    for i in range(n):
        for j in range(n - i):
            parts_f.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
            if i + j < n - 1:
                parts_f.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    return base


def add_tube(P, Q, r0, r1, wd, col, seg=8):
    d = Q - P
    L = np.linalg.norm(d)
    if L < 1e-6:
        return
    d = d / L
    a = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(d, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    base = len(parts_v)
    for k, (C, r) in enumerate(((P, r0), (Q, r1))):
        for s in range(seg):
            t = 2 * math.pi * s / seg
            parts_v.append(C + (a * math.cos(t) + b * math.sin(t)) * r)
            parts_w.append(wd)
            parts_c.append(col)
    for s in range(seg):
        s2 = (s + 1) % seg
        parts_f.append((base + s, base + s2, base + seg + s2, base + seg + s))
    # caps
    for k in (0, 1):
        ctr = len(parts_v)
        parts_v.append(P if k == 0 else Q)
        parts_w.append(wd)
        parts_c.append(col)
        for s in range(seg):
            s2 = (s + 1) % seg
            parts_f.append((ctr, base + k * seg + s, base + k * seg + s2))


for s in ("l", "r"):
    names = TPL["membrane"].get(s)
    if not names or any(n not in J for n in names):
        continue
    sgn = 1.0 if s == "l" else -1.0
    S, E, W, T = (J[f"wing_{s}_{k}"] for k in ("shoulder", "elbow", "wrist", "tip"))
    F = [J[f"wing_{s}_finger{i}"] for i in (1, 2, 3) if f"wing_{s}_finger{i}" in J]
    # the body edge the membrane is sewn to: beside the spine (spine2) and the pelvis, on this side
    # the membrane is sewn to the BACK close under the shoulder (a short, tapered root, not a slab down the
    # flank to the pelvis)
    B1 = J.get("spine3", J["spine2"]) + np.array([-0.02 * H, sgn * 0.035 * H, 0])
    B2 = J["spine2"] + np.array([-0.03 * H, sgn * 0.03 * H, 0.0])
    bw = lambda n: {n: 1.0}
    outline = [(T, bw(f"wing_{s}_tip"))] + [(f, bw(f"wing_{s}_finger{i + 1}")) for i, f in enumerate(F)] + \
              [(B2, bw("pelvis")), (B1, bw("spine2")), (S, bw(f"wing_{s}_shoulder")), (E, bw(f"wing_{s}_elbow"))]
    wW = bw(f"wing_{s}_wrist")
    n = 10
    colf = lambda p, s=s: wing_colour(s, p)
    for k in range(len(outline) - 1):
        (P1, w1), (P2, w2) = outline[k], outline[k + 1]
        between_fingers = k < len(F) + 1                    # tip->f1, f1->f2, f2->f3, f3->pelvis
        add_tri_patch(W, P1, P2, wW, w1, w2, n, colf, scallop=0.12 if between_fingers else 0.04)
    # bones: arm + fingers as tubes, dark
    dark = (0.08, 0.07, 0.07)
    r = 0.012 * H
    add_tube(S, E, r * 1.4, r * 1.2, {f"wing_{s}_elbow": 1.0}, dark)
    add_tube(E, W, r * 1.2, r, {f"wing_{s}_wrist": 1.0}, dark)
    add_tube(W, T, r, r * 0.35, {f"wing_{s}_tip": 1.0}, dark)
    for i, f in enumerate(F):
        add_tube(W, f, r * 0.8, r * 0.25, {f"wing_{s}_finger{i + 1}": 1.0}, dark)

# teeth along the mouth line (cones), upper on the skull, lower on the jaw
teeth_f0 = len(parts_f)
if jaw_ok:
    tooth = (0.85, 0.8, 0.7)
    L = float(np.linalg.norm(tip - J["head"]))
    # teeth only along the ACTUAL mouth outline (the cut), inset from the lip edge toward the midline
    mp = mouth_pts if len(mouth_pts) else np.zeros((0, 3))
    ad = (mp - p0) @ d if len(mp) else np.zeros(0)
    tlist = []
    if len(mp) > 8:
        a0_, a1_ = np.percentile(ad, 25), np.percentile(ad, 92)        # from behind the snout tip back
        for k in range(8):
            a_ = a0_ + (a1_ - a0_) * k / 7
            near = mp[np.abs(ad - a_) < (a1_ - a0_) / 10]
            if len(near) < 2:
                continue
            for side_y in (-1, 1):
                side = near[(near[:, 1] * side_y) > 0]
                if not len(side):
                    continue
                edge = side[np.argmax(np.abs(side[:, 1]))]
                tlist.append(np.array([edge[0], edge[1] * 0.82, edge[2]]))
    for base_p in tlist:
        if True:
            h_ = 0.006 * H
            for up, bone in ((1, "head_end" if "head_end" in gidx else "head"), (-1, "jaw")):
                P = base_p + nrm * up * 0.004 * H
                Q = base_p - nrm * up * h_
                add_tube(P, Q, 0.004 * H, 0.0005 * H, {bone: 1.0}, tooth, seg=5)

teeth_f1 = len(parts_f)
# build the parts mesh, weight it, join into the model
me2 = bpy.data.meshes.new("parts")
me2.from_pydata([tuple(p) for p in parts_v], [], [list(f) for f in parts_f])
me2.update()
# the body's PROJECTED-DRAWING UVs are linear in (y, z) (a front view): fit them on the body, apply to the new
# parts, so the membranes carry the drawing's own wing texture (veins, colour), not a flat grey
PV = np.array(parts_v)
for uvn in [l.name for l in ob.data.uv_layers]:
    lay = ob.data.uv_layers[uvn]
    n = len(ob.data.loops)
    vi = np.zeros(n, int)
    ob.data.loops.foreach_get("vertex_index", vi)
    uv = np.zeros(n * 2)
    lay.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    BV = np.array([ob.data.vertices[i].co[:] for i in vi])
    A = np.stack([BV[:, 1], BV[:, 2], np.ones(len(BV))], 1)
    cu, *_ = np.linalg.lstsq(A, uv[:, 0], rcond=None)
    cv, *_ = np.linalg.lstsq(A, uv[:, 1], rcond=None)
    l2 = me2.uv_layers.new(name=uvn)
    vi2 = np.zeros(len(me2.loops), int)
    me2.loops.foreach_get("vertex_index", vi2)
    P2 = np.stack([PV[vi2, 1], PV[vi2, 2], np.ones(len(vi2))], 1)
    l2.data.foreach_set("uv", np.stack([P2 @ cu, P2 @ cv], 1).ravel())
    print("PARTS uv", uvn, "fit", np.round(cu, 4).tolist(), np.round(cv, 4).tolist())
pob = bpy.data.objects.new("parts", me2)
bpy.context.scene.collection.objects.link(pob)
ca = me2.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
for i, c in enumerate(parts_c):
    ca.data[i].color = (c[0], c[1], c[2], 1.0)
for i, w in enumerate(parts_w):
    tot = sum(w.values()) or 1.0
    for bn, bwv in w.items():
        if bwv / tot < 0.01:
            continue
        g = pob.vertex_groups.get(bn) or pob.vertex_groups.new(name=bn)
        g.add([i], bwv / tot, "REPLACE")
for mat in ob.data.materials:
    me2.materials.append(mat)
tm = bpy.data.materials.new("teeth")
tm.use_nodes = True
tm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.82, 0.78, 0.68, 1)
tm.diffuse_color = (0.82, 0.78, 0.68, 1)
me2.materials.append(tm)
ti = len(me2.materials) - 1
for k, poly in enumerate(me2.polygons):
    poly.material_index = ti if teeth_f0 <= k < teeth_f1 else 0
N_BODY = len(ob.data.vertices)                         # the joined mesh: body verts first, then the new parts
bpy.ops.object.select_all(action="DESELECT")
pob.select_set(True)
ob.select_set(True)
bpy.context.view_layer.objects.active = ob
bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
ob.name = "model"
if not any(m.type == "ARMATURE" for m in ob.modifiers):
    m = ob.modifiers.new("rig", "ARMATURE")
    m.object = arm
# CHECKS (generic): open boundary loops that are not intended sheets (membranes) = a hole; thin-shell flaps left
bm = bmesh.new()
bm.from_mesh(ob.data)
dl2 = bm.verts.layers.deform.verify()
gi2 = {g.index: g.name for g in ob.vertex_groups}
bnd = [e for e in bm.edges if e.is_boundary]
adj = {}
for e in bnd:
    for v in e.verts:
        adj.setdefault(v, []).append(e.other_vert(v))
seen, holes, sheets = set(), [], 0
for v0 in adj:
    if v0 in seen:
        continue
    st_, mem = [v0], []
    seen.add(v0)
    while st_:
        x = st_.pop()
        mem.append(x)
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                st_.append(y)
    doms = [gi2[max(v[dl2].items(), key=lambda kv: kv[1])[0]] if len(v[dl2]) else "" for v in mem]
    if sum(1 for v in mem if v.index >= N_BODY) > 0.5 * len(mem) or \
            sum(1 for g in doms if g.startswith("wing_")) > 0.5 * len(doms):
        sheets += 1                                     # a membrane's outline: an intended sheet
    else:
        holes.append({"verts": len(mem), "at": np.mean([np.asarray(v.co) for v in mem], 0).round(3).tolist()})
bm.free()
chk = {"open_holes": len(holes), "holes": holes[:20], "membrane_outlines": sheets, "flaps": FLAP_STATS,
       "validity": VALID, "ok": len(holes) == 0 and VALID["zero_weight"] == 0 and VALID["long_edges"] == 0}
json.dump(chk, open(str(Path(out).with_name("parts_check.json")), "w"), indent=1)
print("PARTS check", json.dumps({k: v for k, v in chk.items() if k != "holes"}))
bpy.ops.wm.save_as_mainfile(filepath=out)
print("PARTS done", len(parts_v), "new verts")
