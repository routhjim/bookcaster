"""Constructed neckwear for bl_combo.py: the generated shirt front, cravat and shirt collar come out ragged (torn
light patches, a collar that flips with the pose), so that one region is replaced by clean built pieces, measured
from the MetaHuman's own neck and chest, while the generated coat, waistcoat and trousers stay.

Pieces (all in the outfit mesh, its one material, coloured through small swatches written into a free corner of
the outfit's atlas):
  - shirt front: a plate laid 6 mm over the body's chest between the lapels (hidden under the coat and waistcoat
    wherever they lie, which stand at least the fit's 1.4 cm ease off the body);
  - standing collar: rings round the neck from the neck's base up to just under the jaw, each at the neck's own
    largest radius at that height (the head mesh's points) plus a gap, open at the front; style "wing" folds the
    two front top corners out and down;
  - cravat: a band round the collar's lower part, a knot at the throat, two bow loops and two short tails over the
    shirt front.
Height: the collar's front top sits `visible_neck_frac` of the head's height (chin to crown) under the chin, the
proportion read off the character's sheet; it rises `back_rise_m` toward the nape (a collar is cut higher behind).
Weights: the collar's bottom rides the upper spine (spine_05), its top blends into neck_01 (`top_neck_w`), so it
turns a little with the head and never with the arms; the plate rides spine_04/05; the cravat the collar's base.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _hull(pts):
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


def _ray_hull(hull, ang):
    d = (math.cos(ang), math.sin(ang))
    best = 0.0
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = (b[0] - a[0], b[1] - a[1])
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        t = (a[0] * e[1] - a[1] * e[0]) / den
        s = (a[0] * d[1] - a[1] * d[0]) / den
        if t > 0 and -1e-9 <= s <= 1 + 1e-9:
            best = max(best, t)
    return best


def free_swatches(fig, img, colours, cell=32):
    """Write one even block per colour into a part of the atlas no UV triangle touches; returns UVs of their
    centres. Coverage is marked at cell resolution (conservative: each triangle's bounding box)."""
    W, H = img.size
    me = fig.data
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    nx, ny = W // cell, H // cell
    used = np.zeros((ny, nx), bool)
    # each face marks the cells under its corners, centre and edge midpoints (faces are far smaller than a cell
    # on these meshes); a cell is free when it and its eight neighbours are unmarked (bilinear and mip margin)
    li = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", li)
    starts = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", starts)
    tots = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_total", tots)
    pts = [uv]
    for k in range(3):
        a_ = uv[starts + np.minimum(k, tots - 1)]
        b_ = uv[starts + np.minimum(k + 1, tots - 1) % np.maximum(tots, 1)]
        pts.append((a_ + b_) / 2)
    cen = np.add.reduceat(uv, starts) / tots[:, None]
    pts.append(cen)
    P_ = np.vstack(pts)
    xi = np.clip((P_[:, 0] * nx).astype(int), 0, nx - 1)
    yi = np.clip((P_[:, 1] * ny).astype(int), 0, ny - 1)
    used[yi, xi] = True
    grown = used.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(used, dy, 0), dx, 1)
    used = grown
    free = np.argwhere(~used)
    if len(free) < len(colours):
        raise RuntimeError("no free atlas cells for the collar swatches")
    px = np.empty(W * H * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(H, W, 4)
    out = []
    for (cy, cx), c in zip(free, colours):
        px[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell, :3] = c
        px[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell, 3] = 1.0
        out.append(((cx + 0.5) / nx, (cy + 0.5) / ny))
    img.pixels.foreach_set(px.ravel())
    img.update()
    return out


def _obj(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _ellipsoid(c, axes, R, seg=14, ring=8):
    """Ellipsoid at c with semi-axes along R's columns (lateral, out, up)."""
    verts, faces = [], []
    for i in range(ring + 1):
        th = math.pi * i / ring
        for j in range(seg):
            ph = 2 * math.pi * j / seg
            p = np.array([math.sin(th) * math.cos(ph) * axes[0], math.sin(th) * math.sin(ph) * axes[1],
                          math.cos(th) * axes[2]])
            verts.append(c + R @ p)
    for i in range(ring):
        for j in range(seg):
            a, b = i * seg + j, i * seg + (j + 1) % seg
            faces.append((a, b, b + seg, a + seg))
    return verts, faces


def build(J, HEAD, body, arm, img, JOB, REP, colours, PH=None):
    """-> list of new objects (weighted, UV'd onto the swatches), not yet joined. colours: {"collar", "cravat"}."""
    C = JOB.get("collar_build") or {}
    rep = {}
    nx_ = np.array(J("neck_01"))
    cx, cy = float(nx_[0]), float(nx_[1])
    hv = np.array([HEAD.matrix_world @ v.co for v in HEAD.data.vertices])
    top = hv[:, 2].max()
    # the chin: the lowest head-mesh point on the face's centre line ahead of the neck, above the neck's top joint
    # the chin's underside: on the face's centre-line profile (frontmost point per 3 mm slice), the biggest step
    # forward going up from the throat to the chin
    prof = hv[np.abs(hv[:, 0] - cx) < 0.012]
    zs = np.arange(J("neck_02").z - 0.03, J("head").z + 0.02, 0.003)
    fy = np.array([prof[np.abs(prof[:, 2] - z) < 0.0015, 1].min() if (np.abs(prof[:, 2] - z) < 0.0015).any() else np.nan
                   for z in zs])
    ok_ = ~np.isnan(fy)
    zs, fy = zs[ok_], fy[ok_]
    step = fy[1:] - fy[:-1]
    chin = float(zs[int(np.argmin(step))]) if len(step) else float(J("head").z - 0.07)
    head_h = top - chin
    frac = float(C.get("visible_neck_frac", 0.03))
    z_top = chin - frac * head_h
    z_bot = float(J("neck_01").z) - float(C.get("below_neck01_m", 0.035))
    rep.update(chin_z=round(chin, 3), head_h=round(head_h, 3), collar_top_front_z=round(z_top, 3),
               visible_neck_frac=frac)
    # neck rings: head + body points near each height, within 9 cm of the axis
    bev = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    bv = np.array([v.co[:] for v in bev.data.vertices])
    pts = np.vstack([hv, bv])
    pts = pts[(pts[:, 2] > z_bot - 0.03) & (pts[:, 2] < z_top + 0.05)
              & (np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < float(C.get("neck_r_max_m", 0.09)))]
    NA, NR = 64, 7
    gap = float(C.get("gap_m", 0.007))
    rise = float(C.get("back_rise_m", 0.012))
    open_deg = float(C.get("open_deg", 0))
    fa = math.atan2(-1.0, 0.0)          # the front: -Y
    rings, girth = [], []
    R_top = None
    flare = float(C.get("flare_m", 0.010))
    for k in range(NR, -1, -1):          # top ring first: the lower rings flare at most `flare` past it
        t = k / NR
        ring = []
        for a in range(NA):
            ang = fa + 2 * math.pi * a / NA
            back = 0.5 - 0.5 * math.cos(ang - fa)          # 0 at the front, 1 at the back
            z = z_bot + (z_top + rise * back - z_bot) * t
            # the neck's section, never the jaw's: heights near the chin take the section 2.5 cm under it
            band = pts[np.abs(pts[:, 2] - min(z, chin - 0.025)) < 0.01]
            hull = _hull([(round(p[0] - cx, 4), round(p[1] - cy, 4)) for p in band])
            r = (_ray_hull(hull, ang) if len(hull) >= 3 else 0.06) + gap + 0.003 * (1 - t)
            ring.append(np.array([cx + r * math.cos(ang), cy + r * math.sin(ang), z]))
        # a smooth ring: the hull's ray lengths jitter with the mesh; average radii over neighbours
        R_ = np.array([np.hypot(p[0] - cx, p[1] - cy) for p in ring])
        for _ in range(3):
            R_ = 0.5 * R_ + 0.25 * (np.roll(R_, 1) + np.roll(R_, -1))
        if R_top is None:
            R_top = R_.copy()
        R_ = np.minimum(R_, R_top + flare * (1 - t))
        for a in range(NA):
            ang = fa + 2 * math.pi * a / NA
            ring[a][0], ring[a][1] = cx + R_[a] * math.cos(ang), cy + R_[a] * math.sin(ang)
        rings.insert(0, ring)
        girth.insert(0, float(np.sum(np.linalg.norm(np.diff(np.array(ring + ring[:1])[:, :2], axis=0), axis=1))))
    rep["girth_m"] = [round(girth[0], 3), round(girth[-1], 3)]
    keep = [abs(((2 * math.pi * a / NA) + math.pi) % (2 * math.pi) - math.pi) > math.radians(open_deg) for a in range(NA)]
    verts = [p for ring in rings for p in ring]
    faces = []
    for k in range(NR):
        for a in range(NA):
            b = (a + 1) % NA
            if keep[a] and keep[b]:
                faces.append((k * NA + a, k * NA + b, (k + 1) * NA + b, (k + 1) * NA + a))
    objs = []
    white_top = C.get("white_top_m")
    if white_top:
        # a dress's own high collar in the dress colour with a narrow white band (lace) along its top edge
        kw = max(1, min(NR - 1, int(round(NR * float(white_top) / max(z_top - z_bot, 1e-3)))))
        lo = [f for f in faces if f[0] < (NR - kw) * NA]
        hi = [f for f in faces if f[0] >= (NR - kw) * NA]
        objs.append(("dress", _obj("collar_dress", verts, lo)))
        objs.append(("collar", _obj("collar", verts, hi)))
    else:
        objs.append(("collar", _obj("collar", verts, faces)))
    # wing tips: the two front top corners turned out and down
    if C.get("style", "wing") == "wing":
        wv, wf = [], []
        wl = float(C.get("wing_m", 0.022))
        for sgn in (1, -1):
            a0 = next(a for a in (range(NA) if sgn > 0 else range(NA - 1, -1, -1)) if keep[a])
            a1 = (a0 + sgn * 3) % NA
            p0, p1 = rings[NR][a0], rings[NR][a1]
            out0 = np.array([p0[0] - cx, p0[1] - cy, 0])
            out0 /= np.linalg.norm(out0)
            tip = p0 + out0 * 0.012 + np.array([0, 0, -wl])
            mid = p1 + out0 * 0.004 + np.array([0, 0, -0.004])
            base = len(wv)
            wv += [p0 + out0 * 0.002, mid, tip]
            wf.append((base, base + 1, base + 2))
        objs.append(("collar", _obj("wings", wv, wf)))
    # shirt front plate: rays from the front onto the body (and head skin), between the lapels
    dg = bpy.context.evaluated_depsgraph_get()
    bvhs = [BVHTree.FromObject(o, dg) for o in (body, HEAD)]
    pw = float(C.get("plate_half_w_m", 0.075))
    z_pl0 = float(J("spine_03").z)
    z_pl1 = z_bot + 0.02
    gx, gz = 13, 16
    pv, pf = [], []
    for iz in range(gz + 1):
        z = z_pl0 + (z_pl1 - z_pl0) * iz / gz
        for ix in range(gx + 1):
            x = cx - pw + 2 * pw * ix / gx
            best = None
            for bvh in bvhs:
                h = bvh.ray_cast(Vector((x, cy - 0.5, z)), Vector((0, 1, 0)), 1.0)
                if h[0] is not None and (best is None or h[3] < best[3]):
                    best = h
            p = (np.array(best[0]) + np.array(best[1]) * float(C.get("plate_off_m", 0.006))) if best else np.array([x, cy - 0.10, z])
            pv.append(p)
    for iz in range(gz):
        for ix in range(gx):
            a = iz * (gx + 1) + ix
            pf.append((a, a + 1, a + gx + 2, a + gx + 1))
    if C.get("plate", True):
        objs.append(("collar", _obj("shirt_front", pv, pf)))
    if C.get("cuffs") and PH is not None:
        # white cuffs: a band round each wrist where the sleeve ends, at the body's wrist radius + the sleeve ease
        cw = float(C["cuffs"].get("width_m", 0.03)) if isinstance(C["cuffs"], dict) else 0.03
        bev2 = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        BV2 = np.array([v.co[:] for v in bev2.data.vertices])
        for side in ("l", "r"):
            E_, W_ = np.array(PH(f"lowerarm_{side}")), np.array(PH(f"hand_{side}"))
            ax = (W_ - E_) / np.linalg.norm(W_ - E_)
            t_ = (BV2 - W_) @ ax
            rr = np.linalg.norm((BV2 - W_) - t_[:, None] * ax[None, :], axis=1)
            near = (np.abs(t_ + 0.02) < 0.015) & (rr < 0.06)
            r0 = (float(np.quantile(rr[near], 0.95)) if near.sum() > 8 else 0.03) + float(C.get("cuff_ease_m", 0.016))
            u_ = np.cross(ax, [0, 0, 1.0])
            u_ = u_ / np.linalg.norm(u_) if np.linalg.norm(u_) > 1e-6 else np.array([1.0, 0, 0])
            v_ = np.cross(ax, u_)
            cv_, cf_ = [], []
            for k in range(3):
                c0 = W_ - ax * (0.005 + cw * k / 2)
                for a in range(24):
                    ang = 2 * math.pi * a / 24
                    cv_.append(c0 + (u_ * math.cos(ang) + v_ * math.sin(ang)) * (r0 + 0.002 * (1 - k / 2)))
            for k in range(2):
                for a in range(24):
                    b = (a + 1) % 24
                    cf_.append((k * 24 + a, k * 24 + b, (k + 1) * 24 + b, (k + 1) * 24 + a))
            ob_ = _obj(f"cuff_{side}", cv_, cf_)
            objs.append(("collar", ob_))
    if C.get("cravat", True):
        cv = C.get("cravat_shape") or {}
        # band: the collar's lower rings, 4 mm out
        kb = max(2, int(NR * float(cv.get("band_frac", 0.4))))
        bvv, bff = [], []
        for k in range(kb + 1):
            for a in range(NA):
                p = rings[k][a]
                o = np.array([p[0] - cx, p[1] - cy, 0])
                bvv.append(p + o / np.linalg.norm(o) * 0.004)
        for k in range(kb):
            for a in range(NA):
                b = (a + 1) % NA
                bff.append((k * NA + a, k * NA + b, (k + 1) * NA + b, (k + 1) * NA + a))
        objs.append(("cravat", _obj("cravat_band", bvv, bff)))
        f0 = rings[0][0]
        zk = z_bot + (z_top - z_bot) * float(cv.get("knot_frac", 0.4))
        knot_c = np.array([cx, f0[1] - 0.010, zk])
        Rk = np.eye(3)
        kdef = [0.012, 0.009, 0.013] if C.get("neckwear", "bow") == "tie" else [0.016, 0.011, 0.014]
        kv, kf = _ellipsoid(knot_c, cv.get("knot", kdef), Rk)
        objs.append(("cravat", _obj("cravat_knot", kv, kf)))
        tie = C.get("neckwear", "bow") == "tie"
        # bow: two loops and two short tails fanned out; tie: one long blade down the shirt front
        for sgn in ((0,) if tie else (1, -1)):
            if not tie:
                ang = math.radians(cv.get("loop_deg", 18)) * sgn
                Rl = np.array([[math.cos(ang), 0, -math.sin(ang)], [0, 1, 0], [math.sin(ang), 0, math.cos(ang)]])
                lc = knot_c + np.array([sgn * cv.get("loop_off_m", 0.026), -0.002, 0.002])
                lv, lf = _ellipsoid(lc, cv.get("loop", [0.024, 0.009, 0.014]), Rl)
                objs.append(("cravat", _obj("cravat_loop", lv, lf)))
            # a tail: a tapered strip from under the knot, laid over the shirt front
            L = float(cv.get("tail_m", 0.30 if tie else 0.085))
            w0, w1 = (0.028, 0.055) if tie else (0.016, 0.028)
            rows = []
            nrow = 14 if tie else 8
            for k in range(nrow):
                t = k / (nrow - 1)
                w = w0 + (w1 - w0) * t
                x = cx + sgn * (0.006 + 0.022 * t)
                z = zk - 0.008 - L * t
                row = []
                for s_ in (-0.5, 0.5):
                    q = Vector((x + s_ * w, cy - 0.5, z))
                    best = None
                    for bvh in bvhs:
                        h = bvh.ray_cast(q, Vector((0, 1, 0)), 1.0)
                        if h[0] is not None and (best is None or h[3] < best[3]):
                            best = h
                    off = 0.009 if tie else 0.011
                    row.append(np.array(best[0]) + np.array(best[1]) * off if best else np.array([q.x, cy - 0.11, z]))
                rows.append(row)
            if tie:                       # the blade's point
                tipz = rows[-1][0][2] - 0.025
                mid = (rows[-1][0] + rows[-1][1]) / 2
                rows.append([mid + np.array([0, 0, tipz - mid[2]]), mid + np.array([0, 0, tipz - mid[2]])])
            tv = [p for r in rows for p in r]
            tf = [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(len(rows) - 1)]
            objs.append(("cravat", _obj("cravat_tail", tv, tf)))
    # weights and UVs
    uvs = colours["_uv"]
    top_w = float(C.get("top_neck_w", 0.45))
    out = []
    for kind, ob in objs:
        me = ob.data
        P = np.array([v.co[:] for v in me.vertices])
        g5 = ob.vertex_groups.new(name="spine_05")
        gn = ob.vertex_groups.new(name="neck_01")
        g4 = ob.vertex_groups.new(name="spine_04")
        for i, p in enumerate(P):
            t = float(np.clip((p[2] - z_bot) / max(z_top - z_bot, 1e-3), 0, 1))
            if ob.name.startswith("cuff_"):
                sd = ob.name[-1]
                g5.remove([i])
                (ob.vertex_groups.get(f"lowerarm_{sd}") or ob.vertex_groups.new(name=f"lowerarm_{sd}")).add([i], 1.0, "REPLACE")
                continue
            if ob.name.startswith("shirt_front"):
                u = float(np.clip((p[2] - z_pl0) / max(J("spine_05").z - z_pl0, 1e-3), 0, 1))
                g4.add([i], 1 - u, "REPLACE")
                g5.add([i], u, "REPLACE")
            elif p[2] < z_bot:
                g5.add([i], 1.0, "REPLACE")
            else:
                g5.add([i], 1 - top_w * t, "REPLACE")
                gn.add([i], top_w * t, "REPLACE")
        uv = uvs[kind]
        if not me.uv_layers:
            me.uv_layers.new(name=colours.get("_uv_name", "UVMap"))
        me.uv_layers[0].data.foreach_set("uv", np.tile(np.array(uv, np.float32), len(me.loops)))
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        out.append(ob)
    rep["pieces"] = {o.name: len(o.data.vertices) for o in out}
    REP["collar_build"] = rep
    return out
