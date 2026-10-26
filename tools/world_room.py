"""Usable floor area of an interior set (walls as planes), so camera / figure solvers keep everything inside
the room. CPU only, from the set's levelled points (world.json frame: metres, Z up, floor z = 0).

    room = room_bounds(points (N,3) or (H,W,3), cam_xy)      -> dict for world.json["room"]
    inside_room(room, xyz, margin=0.0) -> bool
    clamp_into_room(room, xyz)         -> xyz moved toward the capture point until inside

world.json["room"] = {
  "floor_polygon_xy": [[x, y], ...]   free floor around the capture point (star-shaped, CCW), already shrunk
                                       by margin_m from the nearest obstacle at body height,
  "walls": [{"n": [nx, ny, 0], "d": d, "inliers": k}, ...]   vertical planes, inside where n.p + d >= 0,
                                       APPROXIMATE (RANSAC on the free-space boundary, sorted by support;
                                       furniture can make a 'wall'): the polygon is authoritative,
  "ceiling_z": metres (camera tops must stay below), "floor_z": 0.0,
  "bbox_xy": [xmin, ymin, xmax, ymax] of the polygon, "margin_m": 0.25, "method": ...}
"""
from __future__ import annotations

import math

import numpy as np

BINS = 360
BODY = (0.3, 2.0)          # obstacle slab above the floor (m)
MARGIN = 0.25


def room_bounds(points, cam_xy=(0.0, 0.0), margin: float = MARGIN, bins: int = BINS) -> dict:
    P = np.asarray(points, np.float64).reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    cx, cy = float(cam_xy[0]), float(cam_xy[1])
    slab = P[(P[:, 2] > BODY[0]) & (P[:, 2] < BODY[1])]
    dx, dy = slab[:, 0] - cx, slab[:, 1] - cy
    r = np.hypot(dx, dy)
    ang = (np.arctan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
    b = np.minimum((ang / (2 * math.pi) * bins).astype(int), bins - 1)
    rmin = np.full(bins, np.nan)
    for i in range(bins):
        ri = r[b == i]
        if len(ri) >= 5:
            rmin[i] = np.percentile(ri, 5)            # nearest obstacle on this bearing, robust to specks
    # fill empty bearings from neighbours (circular)
    if np.isnan(rmin).all():
        rmin[:] = 3.0
    idx = np.arange(bins)
    good = ~np.isnan(rmin)
    rmin = np.interp(idx, idx[good], rmin[good], period=bins)
    # a little circular smoothing (min-preserving: take the min of a 3-bin window)
    rs = np.minimum(np.minimum(rmin, np.roll(rmin, 1)), np.roll(rmin, -1))
    rr = np.maximum(rs - margin, 0.05)
    th = (idx + 0.5) / bins * 2 * math.pi
    poly = np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)
    # walls: RANSAC lines on the obstacle boundary points (unshrunk)
    bpts = np.stack([cx + rmin * np.cos(th), cy + rmin * np.sin(th)], 1)
    walls = _fit_walls(bpts, (cx, cy))
    top = P[P[:, 2] > 1.0]
    above = top[np.hypot(top[:, 0] - cx, top[:, 1] - cy) < max(1.0, float(np.median(rmin)) * 0.5)]
    ceiling = float(np.percentile(above[:, 2], 10)) if len(above) > 50 else float(np.percentile(P[:, 2], 99))
    return {"floor_polygon_xy": np.round(poly, 3).tolist(), "walls": walls, "ceiling_z": round(ceiling, 3),
            "floor_z": 0.0, "bbox_xy": np.round([poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(),
                                                   poly[:, 1].max()], 3).tolist(),
            "margin_m": margin, "capture_xy": [cx, cy],
            "method": f"polar free space from the capture point: nearest obstacle at {BODY[0]}-{BODY[1]} m per "
                      f"{360 / bins:.0f} deg bearing (5th pct), minus margin; walls = RANSAC lines on that boundary"}


def _fit_walls(pts, c, max_walls: int = 8, tol: float = 0.12, min_inliers: int = 12):
    rng = np.random.default_rng(0)
    left = pts.copy()
    walls = []
    for _ in range(max_walls):
        if len(left) < min_inliers:
            break
        best = None
        for _ in range(300):
            i, j = rng.choice(len(left), 2, replace=False)
            d = left[j] - left[i]
            n = np.array([-d[1], d[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n /= nn
            dist = np.abs((left - left[i]) @ n)
            inl = dist < tol
            if best is None or inl.sum() > best[0].sum():
                best = (inl, n, left[i])
        if best is None or best[0].sum() < min_inliers:
            break
        inl, n, p0 = best
        q = left[inl]
        cen = q.mean(0)
        _, _, vt = np.linalg.svd(q - cen, full_matrices=False)
        n = np.array([-vt[0][1], vt[0][0]])
        dd = -n @ cen
        if n @ np.asarray(c) + dd < 0:             # inward = toward the capture point
            n, dd = -n, -dd
        walls.append({"n": [round(float(n[0]), 4), round(float(n[1]), 4), 0.0], "d": round(float(dd), 4),
                      "inliers": int(inl.sum())})
        left = left[~inl]
    return walls


def _in_poly(poly, x, y):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def inside_room(room: dict, xyz, margin: float = 0.0) -> bool:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2]) if len(xyz) > 2 else 1.6
    if z > room["ceiling_z"] - 0.15 - margin or z < room.get("floor_z", 0.0) + 0.05:
        return False
    if margin:
        cx, cy = room["capture_xy"]
        dx, dy = x - cx, y - cy
        rr = math.hypot(dx, dy)
        if rr > 1e-6:
            x, y = cx + dx * (rr + margin) / rr, cy + dy * (rr + margin) / rr
    return _in_poly(room["floor_polygon_xy"], x, y)


def clamp_into_room(room: dict, xyz, steps: int = 40):
    """Move a point along the line to the capture point (at its own height, clamped under the ceiling)
    until it is inside the room."""
    x, y, z = (float(v) for v in xyz)
    z = min(max(z, room.get("floor_z", 0.0) + 0.1), room["ceiling_z"] - 0.2)
    cx, cy = room["capture_xy"]
    for k in range(steps + 1):
        t = k / steps
        px, py = x + (cx - x) * t, y + (cy - y) * t
        if inside_room(room, (px, py, z)):
            return [px, py, z]
    return [cx, cy, z]
