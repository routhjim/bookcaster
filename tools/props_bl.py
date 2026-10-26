"""Blender side of tools/props_build.py: one prop mesh -> painted, aligned onto its library proxy, joints split.

    blender -b --factory-startup -t 8 --python tools/props_bl.py -- JOB.json

JOB {"mesh": shape.glb (Hunyuan3D: Z-up after import, the drawing's view along +Y), "ref": the drawing cut-out
     (RGBA), "proxy": the library entry's parts in the PROP FRAME (held: grip at the origin, handle along +X; set:
     floor centre at the origin, Z up), "size_m": the longest extent when there is no proxy, "joints": [...],
     "out_blend", "out_json", "qa_prefix"}
1. PAINT (in the import frame): front projection of the cut-out (screen right = +X, up = +Z).
2. ALIGN: centre, scale the longest principal extent to the proxy's, try the 24 axis-aligned rotations of the
   principal frame onto the proxy's principal frame, keep the lowest symmetric chamfer distance (the "fit",
   as a fraction of the extent). The proxy keeps doing contacts; the mesh is what renders.
3. JOINTS: each joint's part = the vertices beyond a slab ({"axis": "z", "above": 0.55}: the top 45 % along z)
   split into object "joint_<name>" with its origin at the pivot (fractions of the box, -0.5..0.5 per axis).
4. QA: 3 views (front, side, 3/4 from above) + a 1.8 m figure for scale (Workbench, vertex colours).
"""

import json
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
bpy.ops.wm.read_factory_settings(use_empty=True)
DERIVE = job.get("derive_from")
QA_ONLY = job.get("qa_only")


def _islands(me_):
    bm_ = bmesh.new()
    bm_.from_mesh(me_)
    seen_, n_ = set(), 0
    for v_ in bm_.verts:
        if v_.index in seen_:
            continue
        n_ += 1
        st_ = [v_]
        seen_.add(v_.index)
        while st_:
            x_ = st_.pop()
            for e_ in x_.link_edges:
                y_ = e_.other_vert(x_)
                if y_.index not in seen_:
                    seen_.add(y_.index)
                    st_.append(y_)
    bm_.free()
    return n_


def _rest_check(V2_):
    e2_ = np.ptp(V2_, axis=0)
    br_ = 0.0
    for f_ in (0.06, 0.15):      # 15 %: legs reach the floor at different heights (the trestle bellows: 2 feet at 6 %)
        b2_ = V2_[V2_[:, 2] < V2_[:, 2].min() + f_ * e2_[2]]
        if len(b2_) > 3:
            br_ = max(br_, float(np.ptp(b2_[:, 0]) * np.ptp(b2_[:, 1]) / max(e2_[0] * e2_[1], 1e-9)))
    rc_ = {"rest": job.get("rest") or ("flat_base" if job.get("kind") == "set" else None),
           "longest_horizontal": bool(e2_[2] < max(e2_[0], e2_[1])), "base_ratio": round(br_, 3)}
    rc_["ok"] = job.get("soft") or (job.get("kind") != "set" and not job.get("rest")) or (
        (rc_["longest_horizontal"] or rc_["rest"] != "long_horizontal") and
        (not rc_["longest_horizontal"] or rc_["rest"] != "upright") and br_ >= 0.15)
    return rc_


if QA_ONLY:
    # REGRESSION / gate re-check of an existing model: no rebuild, the same QA renders + checks
    bpy.ops.wm.open_mainfile(filepath=QA_ONLY)
    ob = bpy.data.objects["prop"]
    me = ob.data
    # the whole prop: the body + its split joint parts (in world space), as the build measured it before the split
    V2 = np.array([tuple(o.matrix_world @ v.co) for o in bpy.data.objects if o.type == "MESH"
                   for v in o.data.vertices])
    lo2, hi2 = V2.min(0), V2.max(0)
    ctr = (lo2 + hi2) / 2
    json.dump({"extent_m": np.ptp(V2, axis=0).round(3).tolist(), "rest_check": _rest_check(V2),
               "islands": _islands(me), "verts": len(V2)}, open(job["out_json"], "w"), indent=1)
elif DERIVE:
    # a MATERIAL / CONDITION state of an object (heated, cooled, cracked, flowing, dark blue): the SAME geometry as
    # the base prop (its model.blend), recoloured (tint), glowing (emission), cracked (dark crack veins + a slight
    # displacement); never a new shape from a new picture
    bpy.ops.wm.open_mainfile(filepath=DERIVE)
    ob = bpy.data.objects["prop"]
    me = ob.data
    fx = job.get("state_fx") or {}
    tint = np.array(fx.get("tint") or [1.0, 1.0, 1.0], np.float32)
    ca = me.color_attributes.get("Col")
    V = np.array([v.co[:] for v in me.vertices])
    if ca is not None:
        C = np.zeros(len(ca.data) * 4, np.float32)
        ca.data.foreach_get("color", C)
        C = C.reshape(-1, 4)
        lum = C[:, :3].mean(1, keepdims=True)
        if fx.get("tint"):                             # clearly the state's colour, the drawing's shading kept
            C[:, :3] = np.clip(0.15 * C[:, :3] + 0.85 * np.minimum(lum * 1.8, 1.0) * tint, 0, 1)
        if fx.get("shimmer"):                          # a faint inner shimmer: sparse light specks
            rng_ = np.random.default_rng(3)
            sp_ = rng_.random(len(C)) < 0.03
            C[sp_, :3] = np.clip(tint * 2.2 + 0.25, 0, 1)
        if fx.get("crack"):
            rng = np.random.default_rng(7)
            ext = float(np.ptp(V, axis=0).max())
            seeds = V[rng.choice(len(V), 9, replace=False)]
            dmin = np.sort(np.linalg.norm(V[:, None, :] - seeds[None], axis=2), axis=1)
            vein = (dmin[:, 1] - dmin[:, 0]) < 0.025 * ext          # Voronoi borders = crack lines
            C[vein, :3] *= 0.15
            if fx.get("emission"):
                C[vein, :3] = tint
            for v, isv in zip(me.vertices, vein):
                if isv:
                    v.co -= v.normal * 0.004 * ext
        ca.data.foreach_set("color", C.ravel())
    if fx.get("emission"):
        for m in me.materials:
            b_ = m.node_tree.nodes.get("Principled BSDF") if m and m.use_nodes else None
            if b_:
                b_.inputs["Emission Color"].default_value = (float(tint[0]), float(tint[1]), float(tint[2]), 1)
                b_.inputs["Emission Strength"].default_value = float(fx["emission"])
    me.update()
    V2 = np.array([v.co[:] for v in me.vertices])
    lo2, hi2 = V2.min(0), V2.max(0)
    ctr = (lo2 + hi2) / 2
    bpy.ops.wm.save_as_mainfile(filepath=job["out_blend"])
    bj = json.load(open(str(DERIVE).replace("model.blend", "model.json")))
    json.dump(dict(bj, derived_from=DERIVE, state_fx=fx), open(job["out_json"], "w"), indent=1)
else:
    if job.get("procedural") == "straps":
        # thin flexible leather (reins, straps): two flat straps sagging from the hand to the bit, a bit loop
        L, wdt, th = float(job.get("size_m", 1.2)), 0.025, 0.004
        verts, faces = [], []
        for strap, off in ((0, -0.03), (1, 0.03)):
            base = len(verts)
            n = 40
            for i in range(n + 1):
                t = i / n
                x = t * L
                z = -0.25 * L * 4 * t * (1 - t)               # the slack: a catenary-like sag
                y = off * (1 - t) + 0.0 * t
                for dy, dz in ((-wdt / 2, -th / 2), (wdt / 2, -th / 2), (wdt / 2, th / 2), (-wdt / 2, th / 2)):
                    verts.append((x, y + dy, z + dz))
            for i in range(n):
                for k in range(4):
                    a, b = base + i * 4 + k, base + i * 4 + (k + 1) % 4
                    faces.append((a, b, b + 4, a + 4))
        # the bit ring at the far end
        import math as _m
        c0 = len(verts)
        for i in range(24):
            a_ = 2 * _m.pi * i / 24
            for dz in (-0.004, 0.004):
                verts.append((L + 0.04 * _m.cos(a_), 0.04 * _m.sin(a_), dz))
        for i in range(24):
            j = (i + 1) % 24
            faces.append((c0 + 2 * i, c0 + 2 * j, c0 + 2 * j + 1, c0 + 2 * i + 1))
        me_ = bpy.data.meshes.new("straps")
        me_.from_pydata(verts, [], faces)
        me_.update()
        o_ = bpy.data.objects.new("straps", me_)
        bpy.context.scene.collection.objects.link(o_)
    else:
        bpy.ops.import_scene.gltf(filepath=job["mesh"])
    obs = [o for o in bpy.data.objects if o.type == "MESH"]
    for o in bpy.data.objects:
        o.select_set(o in obs)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(bpy.data.objects):
        if o is not ob:
            bpy.data.objects.remove(o)
    ob.name = "prop"
    me = ob.data
    # loose FRAGMENTS of image-to-3D (a sliver of the drawing's background, a shadow): keep the islands with
    # >= 15 % of the largest island's vertices
    _bm = bmesh.new()
    _bm.from_mesh(me)
    _bm.verts.ensure_lookup_table()
    _lab, _isl = {}, []
    for _v in _bm.verts:
        if _v.index in _lab:
            continue
        _st, _mem = [_v], []
        _lab[_v.index] = 1
        while _st:
            _x = _st.pop()
            _mem.append(_x)
            for _e in _x.link_edges:
                _y = _e.other_vert(_x)
                if _y.index not in _lab:
                    _lab[_y.index] = 1
                    _st.append(_y)
        _isl.append(_mem)
    _big = max(len(m) for m in _isl) if _isl else 0
    _main = max(_isl, key=len) if _isl else []
    _MP = np.array([v.co[:] for v in _main]) if _main else np.zeros((1, 3))
    _lo, _hi = _MP.min(0), _MP.max(0)
    _pad = 0.08 * float(np.ptp(_MP, axis=0).max())

    def _inside(m):                                     # a part of the object lies within the main body's box
        c_ = np.array([v.co[:] for v in m]).mean(0)
        return bool(np.all(c_ >= _lo - _pad) and np.all(c_ <= _hi + _pad))
    _drop = [v for m in _isl if m is not _main and (len(m) < 0.15 * _big or not _inside(m)) for v in m]
    if _drop:
        bmesh.ops.delete(_bm, geom=_drop, context="VERTS")
        _bm.to_mesh(me)
        print("PROP fragments removed", len(_drop), "verts")
    _bm.free()
    V = np.array([v.co[:] for v in me.vertices])

    # ------------------------------------------------------------------------------------------------ paint
    if job.get("procedural"):
        img = None
    else:
        img = bpy.data.images.load(job["ref"])
    if img is None:
        w = h = 4
        px = np.tile(np.array([0.22, 0.13, 0.07, 1.0], np.float32), (4, 4, 1))
    else:
        w, h = img.size
        px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, -1)[::-1]
    rgb = px[..., :3]
    fg = px[..., 3] > 0.5 if px.shape[2] == 4 else np.abs(rgb - rgb[:8, :8].reshape(-1, 3).mean(0)).sum(-1) > 0.08
    ys, xs = np.nonzero(fg)
    u0, u1, v0, v1 = xs.min(), xs.max(), ys.min(), ys.max()
    lo, hi = V.min(0), V.max(0)
    su = (V[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-6)
    sv = (hi[2] - V[:, 2]) / max(hi[2] - lo[2], 1e-6)
    pu = np.clip((u0 + su * (u1 - u0)).round().astype(int), 0, w - 1)
    pv = np.clip((v0 + sv * (v1 - v0)).round().astype(int), 0, h - 1)
    col = rgb[pv, pu]
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.concatenate([col, np.ones((len(col), 1), np.float32)], 1).ravel())
    mat = bpy.data.materials.new("painted")
    mat.use_nodes = True
    vc = mat.node_tree.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    me.materials.clear()
    me.materials.append(mat)


    # ------------------------------------------------------------------------------------------------ proxy
    def proxy_points(parts, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        P = []
        vols = []
        for p in parts:
            t = p.get("t")
            if t == "box":
                hf = np.asarray(p.get("half", [0.05] * 3), float)
                vols.append(8 * np.prod(hf))
            elif t == "ellipsoid":
                hf = np.asarray(p.get("half", [0.05] * 3), float)
                vols.append(4.19 * np.prod(hf))
            elif t == "capsule":
                r = float(p.get("r", p.get("r_min_dh", 0.05)))
                L = np.linalg.norm(np.asarray(p["b"], float) - np.asarray(p["a"], float))
                vols.append(math.pi * r * r * L + 4.19 * r ** 3)
            else:
                vols.append(0.0)
        tot = sum(vols) or 1.0
        for p, vv in zip(parts, vols):
            k = max(20, int(n * vv / tot))
            t = p.get("t")
            if t in ("box", "ellipsoid"):
                c = np.asarray(p["c"], float)
                hf = np.asarray(p.get("half", [0.05] * 3), float)
                q = rng.uniform(-1, 1, (k * 2, 3))
                if t == "ellipsoid":
                    q = q[(q ** 2).sum(1) <= 1]
                P.append(c + q[:k] * hf)
            elif t == "capsule":
                a, b = np.asarray(p["a"], float), np.asarray(p["b"], float)
                r = float(p.get("r", p.get("r_min_dh", 0.05)))
                tt = rng.uniform(0, 1, (k, 1))
                d = rng.normal(size=(k, 3))
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                P.append(a + (b - a) * tt + d * r * rng.uniform(0, 1, (k, 1)) ** (1 / 3))
        return np.vstack(P) if P else np.zeros((0, 3))


    def pca(X):
        c = X.mean(0)
        U, S, Wt = np.linalg.svd(X - c, full_matrices=False)
        R = Wt                            # rows: principal axes
        if np.linalg.det(R) < 0:
            R[2] *= -1
        return c, R


    def _kd(X):
        from mathutils.kdtree import KDTree
        t = KDTree(len(X))
        for i, p in enumerate(X):
            t.insert(Vector(p), i)
        t.balance()
        return t


    def chamfer(A, B, tB=None):
        tB = tB or _kd(B)
        tA = _kd(A)
        da = np.mean([tB.find(Vector(p))[2] for p in A])
        db = np.mean([tA.find(Vector(p))[2] for p in B])
        return float(da + db) / 2


    Q = proxy_points(job.get("proxy") or [])
    rng = np.random.default_rng(1)
    Vs = V[rng.choice(len(V), min(len(V), 4000), replace=False)]
    cV, RV = pca(Vs)
    extV = np.ptp((Vs - cV) @ RV.T, axis=0).max()
    if len(Q) >= 50:
        cQ, RQ = pca(Q)
        extQ = np.ptp((Q - cQ) @ RQ.T, axis=0).max()
    else:
        cQ, RQ = np.zeros(3), np.eye(3)
        extQ = float(job.get("size_m", 0.5))
    s = extQ / max(extV, 1e-6)
    # the 24 proper rotations of the axis-aligned cube group
    cands = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for sg in ([1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]):
            P_ = np.zeros((3, 3))
            for i in range(3):
                P_[i, perm[i]] = sg[i]
            if np.linalg.det(P_) > 0:
                cands.append(P_)
    if job.get("kind") == "set":
        # a FIXED prop RESTS ON ITS LARGEST FLAT FACE with its LONG axis horizontal (an anvil drawn or shaped on its
        # end must lie down): of the 24 principal-frame turns keep those with the longest axis horizontal, score by
        # the footprint of the lowest slice (a flat base) and, with a proxy, the chamfer fit
        # job "rest": long_horizontal (anvil, bench, table: the class says so) | upright (architecture: the longest
        # axis vertical) | flat_base (default: the largest flat base, any axis)
        rest_ = job.get("rest") or "flat_base"
        keep = []
        for P_ in cands:
            M = RQ.T @ P_ @ RV if len(Q) >= 50 else P_ @ RV
            X = (Vs - cV) @ M.T
            ext_ = np.ptp(X, axis=0)
            if rest_ == "long_horizontal" and ext_[2] >= max(ext_[0], ext_[1]) * 0.98:
                continue                                   # the longest axis would stand vertical
            if rest_ == "upright" and ext_[2] < max(ext_[0], ext_[1]) * 0.98:
                continue
            z0_ = X[:, 2].min()
            base = X[X[:, 2] < z0_ + 0.06 * ext_[2]]
            area = float(np.ptp(base[:, 0]) * np.ptp(base[:, 1])) if len(base) > 3 else 0.0
            keep.append((area, P_, float((M @ np.array([0.0, 0.0, 1.0]))[2])))
        keep.sort(key=lambda a: -a[0])
        if keep:                                       # only the turns that really rest on (nearly) the largest face
            amax = keep[0][0]
            # ... UNLESS the shape's own up survives: image-to-3D keeps the reference's up, and the
            # references are drawn standing (10-21: the trestle bellows lay on its side - its flank was the
            # "largest face")
            # (no footprint bar: legs touch the floor only at their tips - the bellows trestle's upright turn had 5 %
            # of the flank's "base"; an object drawn on its end is already out by the rest rule above)
            upright_ = [P_ for a_, P_, up_ in keep if up_ > 0.9 and a_ >= 0.02 * amax]
            print("REST candidates (base area, own-up):", [(round(a_ / max(amax, 1e-9), 2), round(up_, 2)) for a_, _, up_ in keep])
            cands = upright_[:8] or [P_ for a_, P_, _ in keep if a_ >= 0.8 * amax][:8] or [keep[0][1]]
    best = None
    tQ = _kd(Q[::3]) if len(Q) >= 50 else None
    for P_ in cands:
        if isinstance(P_, tuple):     # a world-frame candidate (fixed props: yaw only, no PCA permutation)
            M = P_[1]
            cV_ = Vs.mean(0)
            ext_xy = np.ptp(Vs[:, :2], axis=0).max()
            extQ_xy = np.ptp(Q[:, :2], axis=0).max() if len(Q) >= 50 else extQ
            s = extQ_xy / max(ext_xy, 1e-6)
            X = (Vs - cV_) @ M.T * s + (cQ if len(Q) >= 50 else 0)
            sc = chamfer(X[::4], Q[::3], tQ) if len(Q) >= 50 else 0.0
            if best is None or sc < best[0]:
                best = (sc, M, s, cV_)
            continue
        M = RQ.T @ P_ @ RV            # V principal frame -> (permuted) -> proxy principal frame -> world
        X = (Vs - cV) @ M.T * s + cQ
        if len(Q) >= 50:
            sc = chamfer(X[::4], Q[::3], tQ)
        else:                         # no proxy: longest along X, then Z up (flat props lie down)
            e = np.ptp(X, axis=0)
            sc = -(e[0] - e[2])
        if best is None or sc < best[0]:
            best = (sc, M, s, cV)
    sc, M, s, cV = best
    fit = round(float(sc) / max(extQ, 1e-6), 4) if len(Q) >= 50 else None
    T = Matrix.Identity(4)
    A = M * s
    for i in range(3):
        for j in range(3):
            T[i][j] = float(A[i, j])
    off = cQ - A @ cV
    for i in range(3):
        T[i][3] = float(off[i])
    me.transform(T)
    me.update()
    # FLAT THINGS LIE FLAT (10-21: the oilcloth, a thin square sheet, stood 0.8 m tall on its edge near Hulda - a
    # "black slab"; a generic proxy box let the fit pick any turn): a resting prop whose thinnest principal extent is
    # < 1/4 of its longest gets that thin axis vertical. Architecture (rest "upright": doors, panels) is exempt.
    if job.get("kind") == "set" and job.get("rest") != "upright":     # resting cloth (a held one hangs by its grip)
        _V = np.array([v.co[:] for v in me.vertices])
        _c = _V.mean(0)
        _ev, _evec = np.linalg.eigh(np.cov((_V - _c).T))
        _ext = np.ptp((_V - _c) @ _evec, axis=0)
        _n = _evec[:, int(np.argmin(_ext))]
        _z0 = _V[:, 2].min()
        _b = _V[_V[:, 2] < _z0 + 0.06 * np.ptp(_V[:, 2])]
        _br = float(np.ptp(_b[:, 0]) * np.ptp(_b[:, 1]) / max(np.ptp(_V[:, 0]) * np.ptp(_V[:, 1]), 1e-9)) if len(_b) > 3 else 0.0
        # ... when it is CLOTH (soft): a thin rigid frame on legs (the trestle bellows, 0.28 m deep) stands; the
        # footprint can't tell them apart (the standing trestle touches the floor at two feet: base 0.05, as an
        # oilcloth on its edge)
        if _ext.min() < 0.25 * _ext.max() and abs(_n[2]) < 0.95 and job.get("soft"):
            _rot = Vector(_n.tolist()).rotation_difference(Vector((0, 0, 1 if _n[2] >= 0 else -1))).to_matrix().to_4x4()
            me.transform(Matrix.Translation(Vector(_c.tolist())) @ _rot @ Matrix.Translation(Vector((-_c).tolist())))
            me.update()
            print("PROP laid flat: thin axis", np.round(_n, 2).tolist(), "extents", np.round(_ext, 3).tolist(), "base", round(_br, 3))
    if job.get("kind") == "set":                     # set props stand on the floor
        zmin = min(v.co.z for v in me.vertices)
        me.transform(Matrix.Translation(Vector((0, 0, -zmin))))
    # the CLASS SIZE wins (a proxy of the wrong scale made a 2.7 m helm, a 0.94 m anvil): longest extent = size_m,
    # scaled about the prop origin (the grip / the floor centre)
    if job.get("size_m"):
        _V = np.array([v.co[:] for v in me.vertices])
        _k = float(job["size_m"]) / max(float(np.ptp(_V, axis=0).max()), 1e-6)
        if abs(_k - 1) > 0.03:
            me.transform(Matrix.Scale(_k, 4))
            s = s * _k
            print("PROP scaled to the class size", job["size_m"], "x", round(_k, 3))
    for p in me.polygons:
        p.use_smooth = True

    # ------------------------------------------------------------------------------------------------ joints
    joints_out = []
    V2 = np.array([v.co[:] for v in me.vertices])
    lo2, hi2 = V2.min(0), V2.max(0)
    ctr, ext = (lo2 + hi2) / 2, (hi2 - lo2)
    AX = {"x": 0, "y": 1, "z": 2}
    for jd in job.get("joints") or []:
        a = AX[jd["part"]["axis"]]
        thr = lo2[a] + float(jd["part"]["above"]) * ext[a] if jd["part"]["above"] > 0 else ctr[a]
        pivot = ctr + np.asarray(jd["pivot"], float) * ext
        bm = bmesh.new()
        bm.from_mesh(me)
        sel = [f for f in bm.faces if f.calc_center_median()[a] > thr]
        if not sel:
            bm.free()
            continue
        dup = bmesh.ops.duplicate(bm, geom=sel)
        part = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMFace)]
        bmesh.ops.delete(bm, geom=sel, context="FACES")
        # move the duplicated part into its own mesh
        bm2 = bmesh.new()
        pm = bpy.data.meshes.new(f"joint_{jd['name']}")
        vmap = {}
        for f in part:
            vs = []
            for v in f.verts:
                if v not in vmap:
                    vmap[v] = bm2.verts.new(v.co - Vector(pivot))
                vs.append(vmap[v])
            try:
                bm2.faces.new(vs)
            except ValueError:
                pass
        bmesh.ops.delete(bm, geom=part, context="FACES")
        bm.to_mesh(me)
        bm.free()
        bm2.to_mesh(pm)
        bm2.free()
        po = bpy.data.objects.new(f"joint_{jd['name']}", pm)
        bpy.context.scene.collection.objects.link(po)
        po.location = Vector(pivot)
        po.parent = ob
        pm.materials.append(mat)
        # carry the paint onto the part (nearest original vertex colour)
        from mathutils.kdtree import KDTree
        kd = KDTree(len(V2))
        for i, p in enumerate(V2):
            kd.insert(p, i)
        kd.balance()
        pc = pm.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        for i, v in enumerate(pm.vertices):
            _, k, _ = kd.find(Vector(pivot) + v.co)
            c = col[k] if k < len(col) else (0.5, 0.5, 0.5)
            pc.data[i].color = (float(c[0]), float(c[1]), float(c[2]), 1.0)
        joints_out.append(dict(jd, object=po.name, pivot_m=pivot.round(4).tolist(), verts=len(pm.vertices)))
        print("JOINT", jd["name"], len(pm.vertices), "verts, pivot", pivot.round(3).tolist())

    bpy.ops.wm.save_as_mainfile(filepath=job["out_blend"])
    # QA CHECK: how the prop rests (a fixed prop: on a flat base; long_horizontal: the long axis level)
    e2 = np.ptp(V2, axis=0)
    b2 = V2[V2[:, 2] < V2[:, 2].min() + 0.06 * e2[2]]
    base_ratio = float(np.ptp(b2[:, 0]) * np.ptp(b2[:, 1]) / max(e2[0] * e2[1], 1e-9)) if len(b2) > 3 else 0.0
    rest_check = {"rest": job.get("rest") or ("flat_base" if job.get("kind") == "set" else None),
                  "longest_horizontal": bool(e2[2] < max(e2[0], e2[1])), "base_ratio": round(base_ratio, 3)}
    rest_check["ok"] = (job.get("kind") != "set" and not job.get("rest")) or (
        (rest_check["longest_horizontal"] or rest_check["rest"] != "long_horizontal") and
        (not rest_check["longest_horizontal"] or rest_check["rest"] != "upright") and base_ratio >= 0.15)
    json.dump({"fit": fit, "scale": float(s), "joints": joints_out, "extent_m": np.ptp(V2, axis=0).round(3).tolist(),
               "rest_check": _rest_check(V2), "islands": _islands(me)},
              open(job["out_json"], "w"), indent=1)

# ------------------------------------------------------------------------------------------------ QA renders
sc_ = bpy.context.scene
sc_.render.engine = "BLENDER_WORKBENCH"
sc_.display.shading.light = "STUDIO"
sc_.display.shading.color_type = "VERTEX"
sc_.render.resolution_x = sc_.render.resolution_y = 512
sc_.world = bpy.data.worlds.new("w")
sc_.world.color = (0.16, 0.16, 0.17)
R_ = float(max(np.ptp(V2, axis=0))) * 1.6
cam_d = bpy.data.cameras.new("c")
cam_d.type = "ORTHO"
cam = bpy.data.objects.new("c", cam_d)
sc_.collection.objects.link(cam)
sc_.camera = cam
c3 = Vector(ctr)
for name, d in (("front", (0, -1, 0.0)), ("side", (1, 0, 0.0)), ("3q", (0.7, -0.7, 0.55))):
    dv = Vector(d).normalized()
    cam_d.ortho_scale = R_
    cam.location = c3 + dv * 10
    cam.rotation_euler = (-dv).to_track_quat("-Z", "Y").to_euler()
    sc_.render.filepath = f"{job['qa_prefix']}_{name}.png"
    bpy.ops.render.render(write_still=True)
# scale: a 1.8 m figure beside the prop
bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=1.8, location=(float(hi2[0]) + 0.5, float(ctr[1]),
                                                                       0.9 if job.get("kind") == "set" else float(ctr[2])))
fig = bpy.context.active_object
fig.color = (0.8, 0.8, 0.8, 1)
sc_.display.shading.color_type = "VERTEX"
dv = Vector((0, -1, 0.15)).normalized()
span = max(R_, 2.4)
cam_d.ortho_scale = span
mid = Vector(((float(hi2[0]) + 0.5 + float(lo2[0])) / 2, float(ctr[1]), 0.9))
cam.location = mid + dv * 12
cam.rotation_euler = (-dv).to_track_quat("-Z", "Y").to_euler()
sc_.render.filepath = f"{job['qa_prefix']}_zscale.png"
bpy.ops.render.render(write_still=True)
print("PROP done fit", fit, "joints", len(joints_out))
