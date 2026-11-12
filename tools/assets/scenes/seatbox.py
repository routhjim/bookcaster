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


def for_asset(asset_dir: Path) -> dict:
    aj = json.loads((asset_dir / "asset.json").read_text())
    P = glb_points(asset_dir / aj["export"]["glb"])
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
