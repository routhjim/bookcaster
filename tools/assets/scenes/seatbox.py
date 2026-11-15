"""A seat's box from its own mesh: where a seated person goes (numpy; reads the library asset's exported glTF).

    python tools/assets/scenes/seatbox.py <library>/prop/<id> [...] [--write]

Per seat, in the seat's own frame (metres; x across, y depth with the FRONT toward -y, z up, origin the footprint
centre on the floor: the set builder's convention "front -Y"), plus "front_asset": the mesh's real front in the
asset's frame, found from its shape (the back is the high end), so a mesh that breaks the convention still seats
its sitter facing the right way:
  {"seat_z_m":   top of the seat cushion under the sitter (the lowest of the centre line's top surface, front half),
   "front_y_m":  the seat's front edge,
   "back_y_m":   the front face of the back at cushion + 0.1..0.35 m (where a sitter's back rests),
   "inner_x_m":  [left, right] inner faces of the arms at cushion + 0.06 m and above (or the seat's own edges
                 when it has no arms),
   "arm_z_m":    top of the arms (null without arms),
   "width_m":    between the arms, "depth_m": front edge to back,
   "places":     [x, ...] sitter centres across the seat (one for a chair, more for a sofa: one per width_per_place)}
Written into the asset's library record as "seat" with --write (the set and prop builders should write the same
block for every new seat; see README "Seats"). Plain geometry bookkeeping over the vertex positions: no rendering.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

PLACE_W = 0.5            # the width one seated adult takes on a sofa


def glb_points(path: Path) -> np.ndarray:
    b = path.read_bytes()
    n = struct.unpack_from("<I", b, 12)[0]
    j = json.loads(b[20:20 + n])
    off = 20 + n
    blen = struct.unpack_from("<I", b, off)[0]
    bin_ = b[off + 8: off + 8 + blen]
    pts = []
    for node in j["nodes"]:
        if "mesh" not in node:
            continue
        if any(k in node for k in ("matrix", "rotation", "scale")):
            raise SystemExit(f"{path}: node transforms are not handled here")
        t = np.array(node.get("translation", [0, 0, 0]), np.float32)
        for p in j["meshes"][node["mesh"]]["primitives"]:
            a = j["accessors"][p["attributes"]["POSITION"]]
            bv = j["bufferViews"][a["bufferView"]]
            o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            stride = bv.get("byteStride", 12)
            raw = np.frombuffer(bin_, np.uint8, a["count"] * stride, o).reshape(a["count"], stride)
            pts.append(raw[:, :12].copy().view(np.float32).reshape(-1, 3) + t)
    g = np.concatenate(pts)
    # glTF y-up (exported from Blender: x, z, -y) -> the asset frame x, y, z (front -y)
    return np.stack([g[:, 0], -g[:, 2], g[:, 1]], 1)


def seat_box(P: np.ndarray) -> dict:
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    y0, y1 = float(y.min()), float(y.max())
    # the centre line's top surface, slice by slice front to back
    mid = np.abs(x) < 0.12
    ys = np.arange(y0, y1, 0.02)
    tops = []
    for a in ys:
        m = mid & (y >= a) & (y < a + 0.02) & (z > 0.2)
        tops.append(float(z[m].max()) if m.any() else np.nan)
    tops = np.array(tops)
    front_half = ys < (y0 + y1) / 2
    seat_z = float(np.nanpercentile(tops[front_half], 40))
    # the seat's front edge: the first slice whose top is near the cushion
    near = np.where(front_half & (tops > seat_z - 0.06))[0]
    front_y = float(ys[near[0]]) if len(near) else y0
    # the back's front face where a back rests
    band = mid & (z > seat_z + 0.10) & (z < seat_z + 0.35) & (y > front_y + 0.15)
    back_y = float(y[band].min()) if band.any() else y1
    # arms: the inner faces at the middle of the seat's depth
    ym = (front_y + back_y) / 2
    sl = (np.abs(y - ym) < 0.12) & (z > seat_z + 0.06)
    xl = x[sl & (x < -0.05)]
    xr = x[sl & (x > 0.05)]
    xs_all = x[(np.abs(y - ym) < 0.12) & (z > seat_z - 0.05)]
    inner_l = float(xl.max()) if len(xl) else float(xs_all.min())
    inner_r = float(xr.min()) if len(xr) else float(xs_all.max())
    arms = len(xl) > 50 and len(xr) > 50
    arm_z = None
    if arms:
        # the arm rests only: the front part of the seat's depth, below a wing's height
        outer = (((x <= inner_l) | (x >= inner_r)) & (z > seat_z + 0.06) & (z < seat_z + 0.40)
                 & (y > front_y) & (y < front_y + 0.6 * (back_y - front_y)))
        arm_z = float(np.percentile(z[outer], 98)) if outer.any() else None
    width = inner_r - inner_l
    n = max(1, int(width / PLACE_W + 1e-6))
    places = [round(inner_l + width * (i + 0.5) / n, 3) for i in range(n)]
    return {"seat_z_m": round(seat_z, 3), "front_y_m": round(front_y, 3), "back_y_m": round(back_y, 3),
            "inner_x_m": [round(inner_l, 3), round(inner_r, 3)], "arm_z_m": None if arm_z is None else round(arm_z, 3),
            "width_m": round(width, 3), "depth_m": round(back_y - front_y, 3), "places": places}


def front_of(P: np.ndarray) -> list:
    """The mesh's own front, from its shape: of the two plan axes, the one whose two ends differ most in height; the
    higher end is the back. A seat whose front is not -y breaks the library's "front -Y" contract (11-11: the leather
    armchair faced +x, so a sitter placed by the set's facing sat across its arm)."""
    best = (0.0, [0.0, -1.0])
    for ax in (0, 1):
        c = P[:, ax]
        lo, hi = c.min(), c.max()
        span = hi - lo
        mid = np.abs(P[:, 1 - ax] - P[:, 1 - ax].mean()) < 0.25 * (P[:, 1 - ax].max() - P[:, 1 - ax].min())
        a = P[mid & (c < lo + 0.15 * span), 2]
        b = P[mid & (c > hi - 0.15 * span), 2]
        if not len(a) or not len(b):
            continue
        ta, tb = float(np.percentile(a, 98)), float(np.percentile(b, 98))
        if abs(ta - tb) > best[0]:
            f = [0.0, 0.0]
            f[ax] = 1.0 if ta > tb else -1.0          # the low end is the front
            best = (abs(ta - tb), f)
    return best[1]


PAN_MIN_M = 0.35                        # the least of a seat pan, front to back


def cabin_seat_box(P: np.ndarray, seats: int = 1) -> dict:
    """The seats INSIDE an enclosure (a cockpit, a cab, a carriage): a chair's measure reads the top of the centre
    line, which in a cabin is its roof or canopy. Here a seat is a level patch with headroom above it (no geometry
    for 0.55 m), walled in on both sides within 0.45 m, with the cabin rising on one end only: together they are the
    cabin's interior. It is split side to side into as many lanes as the cabin has seats abreast (the record's
    "seats_abreast", read from the place's concept picture), and the first lane's low level is its seat. The sitter
    faces the cabin's tall end (its controls and windscreen), not away from a back as on a chair. -> the seat block,
    in the seat's frame (front -y), with "front_asset" the facing in the asset's frame."""
    import collections
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    C = 0.04
    x0, y0 = float(x.min()), float(y.min())
    ix, iy = ((x - x0) / C).astype(int), ((y - y0) / C).astype(int)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    cells = collections.defaultdict(list)
    for i, j, zz in zip(ix, iy, z):
        cells[(int(i), int(j))].append(float(zz))
    surf, roofed = {}, {}
    for k, zs in cells.items():
        zs = sorted(zs)
        for n, zc in enumerate(zs):
            if (zs[n + 1] if n + 1 < len(zs) else 9.0) - zc >= 0.55 and 0.15 < zc < 1.2:
                surf[k] = zc
                roofed[k] = n + 1 < len(zs)
                break

    def has(i0, i1, j0, j1, lo, hi):
        return any(lo < v < hi for i in range(max(i0, 0), min(i1, nx)) for j in range(max(j0, 0), min(j1, ny))
                   for v in cells.get((i, j), ()))
    R = int(0.45 / C)
    cand = {}
    for (i, j), zc in surf.items():
        if not (has(i - R, i - 2, j - 1, j + 2, zc + 0.2, zc + 1.0) and has(i + 3, i + R + 1, j - 1, j + 2, zc + 0.2, zc + 1.0)):
            continue
        bp = has(i - 1, i + 2, j + 3, j + R + 1, zc + 0.3, zc + 1.0)
        bm = has(i - 1, i + 2, j - R, j - 2, zc + 0.3, zc + 1.0)
        if bp != bm:
            cand[(i, j)] = 1 if bp else -1
    if not cand:
        raise SystemExit("no seat found inside the cabin")
    # a made cabin has other level, walled patches: the deck of its nose, a sill behind its seats. The interior is
    # the patch under the canopy (geometry overhead, past the headroom); with no canopy over any, the largest. The
    # largest alone sat a pilot out on the nose, facing back into the windscreen (11-14).
    left, comps = set(cand), []
    while left:
        stack, comp = [left.pop()], []
        while stack:
            k = stack.pop()
            comp.append(k)
            for q in [(k[0] + a, k[1] + b) for a in range(-2, 3) for b in range(-2, 3)]:
                if q in left:
                    left.remove(q)
                    stack.append(q)
        comps.append(comp)
    best = max(comps, key=lambda c: (sum(roofed[k] for k in c) >= 0.5 * len(c), len(c)))
    cand = {k: cand[k] for k in best}
    ks = np.array(list(cand))
    f = 1.0 if np.mean(list(cand.values())) > 0 else -1.0             # the sitter faces the cabin's tall end
    xl, xr = x0 + ks[:, 0].min() * C, x0 + (ks[:, 0].max() + 1) * C     # the walled interior, side to side
    lane_w = (xr - xl) / max(1, seats)
    # the first lane from the asset's -x side is measured; its seat is its low level surface
    lane = [k for k in cand if xl <= x0 + (k[0] + 0.5) * C < xl + lane_w]
    zs = np.array([surf[k] for k in lane])
    seat_z = float(np.percentile(zs, 30))
    on = np.array([k for k in lane if abs(surf[k] - seat_z) < 0.12])
    ya, yb = y0 + on[:, 1].min() * C, y0 + (on[:, 1].max() + 1) * C
    sx = -f                                                             # asset -> seat frame (front -y)
    xs = sorted([sx * xl, sx * (xl + lane_w)])
    ys = sorted([-f * ya, -f * yb])
    xs, ys, lane_w = [float(v) for v in xs], [float(v) for v in ys], float(lane_w)
    # the interior's rear end is the cabin's sill or bulkhead, not the seat's back: the sitter's back rests a seat
    # back's thickness ahead of it (measured at the end itself, the pilot sat on the rear sill behind his seat)
    # (only where the level runs on behind a seat's own depth: a patch no longer than a seat pan is the pan itself)
    if ys[1] - ys[0] >= 0.30 + PAN_MIN_M:
        ys[1] -= 0.30
    return {"seat_z_m": round(seat_z, 3), "front_y_m": round(ys[0], 3), "back_y_m": round(ys[1], 3),
            "inner_x_m": [round(xs[0], 3), round(xs[1], 3)], "arm_z_m": None, "width_m": round(lane_w, 3),
            "depth_m": round(ys[1] - ys[0], 3), "places": [round((xs[0] + xs[1]) / 2, 3)],
            "front_asset": [0.0, f], "inside": True, "seats_abreast": int(seats)}


def for_asset(asset_dir: Path) -> dict:
    aj = json.loads((asset_dir / "asset.json").read_text())
    P = glb_points(asset_dir / aj["export"]["glb"])
    if aj.get("piece_kind") == "vehicle_cabin" or "vehicle_cabin" in (aj.get("tags") or []):
        return cabin_seat_box(P, int(aj.get("seats_abreast") or 1))
    f = front_of(P)
    # turn the points so the front is -y, measure there, keep the mesh's own front with the box
    yaw = np.arctan2(f[1], f[0]) - np.arctan2(-1.0, 0.0)
    c, s = np.cos(-yaw), np.sin(-yaw)
    Q = np.stack([P[:, 0] * c - P[:, 1] * s, P[:, 0] * s + P[:, 1] * c, P[:, 2]], 1)
    out = seat_box(Q)
    out["front_asset"] = [round(float(v), 3) for v in f]
    if f != [0.0, -1.0]:
        out["warning"] = f"the mesh's front is {out['front_asset']} in its own frame, not -Y (library contract)"
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in args:
        d = Path(a)
        s = for_asset(d)
        print(d.name, json.dumps(s))
        if "--write" in sys.argv:
            p = d / "asset.json"
            aj = json.loads(p.read_text())
            aj["seat"] = dict(s, source="scenes/seatbox.py from the exported mesh")
            p.write_text(json.dumps(aj, indent=1))


if __name__ == "__main__":
    main()
