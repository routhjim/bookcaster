"""Background layer for a single-viewpoint set (a 2-layer depth image), so a moved camera sees plausible
background behind occluders instead of holes (cut mesh) or rubber walls (stretched mesh).

Rule (after "3D photo inpainting", Shih et al. 2020): near-side pixels of depth edges, out to `width` px,
that are notably nearer than the farthest surface within that reach, are re-synthesised as BACKGROUND:
depth inpainted in log space and colour inpainted from the surrounding background, then meshed as a
second layer that sits behind the occluder and stitches to the real background along its 1-px ring.
Pure numpy/scipy/OpenCV, CPU. Used by tools/world.py _finish_world.
"""
from __future__ import annotations

import numpy as np


def push_pull_fill(a, known):
    """Fill a (H,W) or (H,W,C) float array where known==False, smoothly from the known pixels: a
    normalised-convolution pyramid (push: area-downsample value*weight and weight; pull: upsample the
    coarse estimate into the holes). OpenCV's Telea inpaint is NOT smooth on float depth (log-depth steps
    of 2.6 inside a 50 px hole), which tore the layer apart."""
    import cv2
    a = a.astype(np.float32)
    w = known.astype(np.float32)
    if a.ndim == 3:
        aw = a * w[..., None]
    else:
        aw = a * w
    levels = [(aw, w)]
    while min(levels[-1][1].shape[:2]) > 2 and (levels[-1][1] <= 1e-6).any():
        pa, pw = levels[-1]
        h, ww = pw.shape[:2]
        size = (max(1, ww // 2), max(1, h // 2))
        levels.append((cv2.resize(pa, size, interpolation=cv2.INTER_AREA),
                       cv2.resize(pw, size, interpolation=cv2.INTER_AREA)))
    pa, pw = levels[-1]
    est = pa / np.maximum(pw, 1e-6)[..., None] if pa.ndim == 3 else pa / np.maximum(pw, 1e-6)
    for pa, pw in reversed(levels[:-1]):
        h, ww = pw.shape[:2]
        up = cv2.resize(est, (ww, h), interpolation=cv2.INTER_LINEAR)
        cur = pa / np.maximum(pw, 1e-6)[..., None] if pa.ndim == 3 else pa / np.maximum(pw, 1e-6)
        t = np.clip(pw, 0, 1)
        if pa.ndim == 3:
            t = t[..., None]
        est = t * cur + (1 - t) * up
    return est


def background_layer(P, C, cam, valid, sky, wrap=False, width=60, edge_ratio=1.04):
    """P (H,W,3) points, C (H,W,3) uint8 RGB, cam (3,), valid/sky (H,W) bool.
    Edge-anchored propagation: every pixel nearer than a 4-neighbour by edge_ratio (the same 4% jump at
    which the cut mesh opens a seam) takes that far neighbour's depth and colour as its BACKGROUND, which
    then spreads into the occluder, one pixel per step for `width` steps, only into pixels clearly in
    front of it (fronto-parallel extension of the background surface; colour copied from the far side).
    Returns (P_layer, C_layer, valid_layer, region): valid_layer = region + its 1-px ring, so the layer
    stitches to the real background (stitches onto the occluder are long edges and get cut by the mesh)."""
    from scipy import ndimage
    H, W = valid.shape
    geo = valid & ~sky
    d = np.linalg.norm(P - np.asarray(cam), axis=-1)
    d = np.where(geo, d, 0.0).astype(np.float32)
    col = C.astype(np.uint8)
    bgd = np.zeros((H, W), np.float32)
    bgc = np.zeros((H, W, 3), np.uint8)

    def shifts(a):
        """4-neighbour views of a (H,W[,3]) array: up, down, left, right (columns wrap for panoramas)."""
        up = np.concatenate([a[:1], a[:-1]], 0)
        dn = np.concatenate([a[1:], a[-1:]], 0)
        if wrap:
            lf, rt = np.roll(a, 1, 1), np.roll(a, -1, 1)
        else:
            lf = np.concatenate([a[:, :1], a[:, :-1]], 1)
            rt = np.concatenate([a[:, 1:], a[:, -1:]], 1)
        return [up, dn, lf, rt]

    # seed: near side of every depth edge gets the FARTHEST geometry within a (2*seed_r+1)^2 window
    # (MoGe depth ramps over 2-3 px at silhouettes; seeding from the 4-neighbour put the layer mid-ramp)
    seed_r = 3
    for dy in range(-seed_r, seed_r + 1):
        for dx in range(-seed_r, seed_r + 1):
            if dy == 0 and dx == 0:
                continue
            nd = np.roll(np.roll(d, dy, 0), dx, 1)
            nc = np.roll(np.roll(col, dy, 0), dx, 1)
            if not wrap and dx:
                (nd[:, :dx] if dx > 0 else nd[:, dx:])[:] = 0
            if dy:
                (nd[:dy] if dy > 0 else nd[dy:])[:] = 0
            take = geo & (nd > edge_ratio * np.maximum(d, 1e-6)) & (nd > bgd)
            bgd = np.where(take, nd, bgd)
            bgc = np.where(take[..., None], nc, bgc)
    region = bgd > 0
    if not region.any():
        return P, C, np.zeros_like(valid), region
    # propagate into the occluder
    for _ in range(width):
        best = np.zeros_like(bgd)
        bestc = np.zeros_like(bgc)
        for nd, nc in zip(shifts(bgd), shifts(bgc)):
            t = nd > best
            best = np.where(t, nd, best)
            bestc = np.where(t[..., None], nc, bestc)
        grow = geo & ~region & (best > edge_ratio * np.maximum(d, 1e-6))
        if not grow.any():
            break
        bgd = np.where(grow, best, bgd)
        bgc = np.where(grow[..., None], bestc, bgc)
        region |= grow
    # soften the copied colours a little (pure copies streak along the propagation direction)
    sm = ndimage.uniform_filter(bgc.astype(np.float32), size=(5, 5, 1), mode="nearest")
    bgc = np.where(region[..., None], sm, bgc).astype(np.uint8)
    dirs = (P - np.asarray(cam)) / np.maximum(d, 1e-6)[..., None]
    P_bg = np.asarray(cam) + dirs * bgd[..., None]
    ring = ndimage.binary_dilation(region, iterations=1) & geo & ~region
    layer_valid = region | ring
    P_layer = np.where(region[..., None], P_bg, P)
    C_layer = np.where(region[..., None], bgc, C).astype(np.uint8)
    return P_layer, C_layer, layer_valid, region


def despeckle_depth(depth, valid=None, wrap=False, size=5, log_jump=0.2, max_blob=30):
    """Remove isolated near-depth specks (MoGe puts candle flames, sparks and bright dots metres in front of
    their wall): pixels nearer than their size x size median by more than log_jump (0.2 = 18 %) whose
    connected blob is under max_blob pixels take the median depth. Thin real props (tools, stumps, poles)
    form larger blobs and are kept. Returns (depth, n_fixed)."""
    from scipy import ndimage
    d = np.asarray(depth, np.float32)
    ok = np.isfinite(d) & (d > 0) if valid is None else (valid & np.isfinite(d) & (d > 0))
    ld = np.where(ok, np.log(np.maximum(d, 1e-6)), np.nan).astype(np.float32)
    fill = np.where(ok, ld, np.nanmedian(ld[ok]) if ok.any() else 0).astype(np.float32)
    pw = size // 2 if wrap else 0
    fp = np.concatenate([fill[:, -pw:], fill, fill[:, :pw]], 1) if pw else fill
    med = ndimage.median_filter(fp, size=size, mode="nearest")
    med = med[:, pw:pw + fill.shape[1]] if pw else med
    near = ok & (ld < med - log_jump)
    if not near.any():
        return d, 0
    lab, n = ndimage.label(near)
    sizes = ndimage.sum(near, lab, index=np.arange(1, n + 1))
    small = np.zeros(n + 1, bool)
    small[1:] = sizes < max_blob
    fix = small[lab] & near
    out = np.where(fix, np.exp(med), d).astype(np.float32)
    return out, int(fix.sum())


def floater_pixels(P, faces, cam, max_frac=0.01, open_ratio=1.25, open_frac=0.6, sky=None, grid=None):
    """Floating islands of a set mesh, judged in IMAGE space. Components of `faces` (triangles as pixel indices
    into the (H, W) grid of P) are FLOATERS when small (< max_frac of all faces) AND hovering: of the
    component's bottom-edge pixels, more than open_frac see something more than open_ratio x farther (or sky /
    nothing) directly BELOW them in the panorama -- open space under it.
    A distance-to-large-geometry test was tried first and REJECTED (10-20): it removed a building across a
    river, stall interiors seen through windows, distant stumps and candles -- all naturally separated by depth
    edges but resting on something below them in the image.
    Returns (drop mask over the flattened pixels, stats dict)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P = np.asarray(P, np.float64)
    H, W = (grid or P.shape[:2])
    Pf = P.reshape(-1, 3)
    n = len(Pf)
    cam = np.asarray(cam, np.float64)
    dist = np.linalg.norm(Pf - cam, axis=1)
    skyf = np.zeros(n, bool) if sky is None else np.asarray(sky).reshape(-1)
    faces = np.asarray(faces, np.int64)
    if len(faces) == 0:
        return np.zeros(n, bool), {"components": 0, "removed": 0, "removed_faces": 0, "largest_removed": []}
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    nf = np.bincount(lab[faces[:, 0]], minlength=ncomp)
    total = int(nf.sum())
    used = np.zeros(n, bool)
    used[faces.ravel()] = True
    vlab = np.where(used, lab, -1)
    cand = np.where((nf > 0) & (nf < max_frac * total))[0]
    order = np.argsort(vlab, kind="stable")
    vl_sorted = vlab[order]
    starts = np.searchsorted(vl_sorted, cand, "left")
    ends = np.searchsorted(vl_sorted, cand, "right")
    drop = np.zeros(n, bool)
    removed = []
    for c, a, b in zip(cand, starts, ends):
        vid = order[a:b]
        r, col = vid // W, vid % W
        below = np.where(r + 1 < H, vid + W, -1)
        inside = np.zeros(len(vid), bool)
        ok = below >= 0
        inside[ok] = vlab[below[ok]] == c
        edge = ~inside                                   # bottom-edge pixels of the component
        bel = below[edge]
        if len(bel) == 0:
            continue
        d0 = dist[vid[edge]]
        re_, ce_ = r[edge], col[edge]
        # nearest real surface in the next 3 rows below (steps over the uncut seam pixels at depth edges);
        # sky / nothing counts as open, any surface counts by its depth
        near = np.full(len(re_), np.inf)
        for k in (1, 2, 3):
            rr = re_ + k
            okk = rr < H
            pid = rr[okk] * W + ce_[okk]
            dk = np.where(skyf[pid] | ~np.isfinite(dist[pid]), np.inf, dist[pid])
            near[okk] = np.minimum(near[okk], dk)
        openb = near > open_ratio * d0
        fr = float(openb.mean())
        if fr <= open_frac:
            continue
        drop[vid] = True
        removed.append({"faces": int(nf[c]), "open_below": round(fr, 2), "dist_cam_m": round(float(np.median(dist[vid])), 1),
                        "z_min": round(float(Pf[vid, 2].min()), 2)})
    removed.sort(key=lambda r_: -r_["faces"])
    return drop, {"components": int(ncomp), "removed": len(removed),
                  "removed_faces": int(sum(r_["faces"] for r_ in removed)), "largest_removed": removed[:5]}
