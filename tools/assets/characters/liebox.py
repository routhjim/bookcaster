"""A bed's lie box from its own mesh: where a lying person goes (numpy; reads the library asset's exported glTF).

    python tools/assets/characters/liebox.py <library>/prop/<id> [...] [--write]

The top surface is read on a 2 cm grid in plan as the highest vertex in each cell. The lying surface is the mode of
the heights over the bed's middle (the counterpane or sheet on the mattress); the lie box is the connected run of
cells within LIE_TOL of it around the middle (posts, rails and the head- and footboards stand above it and bound it).
The head end is the end whose board stands higher; the pillow is the cells near the head end standing PILLOW_MIN
above the surface. In the asset's frame (metres; x across, y along, front -Y, z up), one entry of the prop record's
"supports" list (format agreed with the scene side, 11-12):
  {"posture": "lying", "name": "mattress", "box": {"top_z_m", "x_m": [x0, x1], "y_m": [y0, y1]},
   "head": "+y" | "-y" | "+x" | "-x", "pillow_z_m": top of the pillow or null, "pillow_y_m": [y0, y1] or null,
   "cover_z_m": top of the bed's own cover over the mattress (the surface read here; the mattress under it is not
   seen from above) or null, "front_asset": [x, y], "source": "characters/liebox.py from the exported mesh"}
--write adds or replaces that entry in the asset's library record. Plain geometry over the vertex positions.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

CELL = 0.02
LIE_TOL = 0.05           # cells within this of the lying surface belong to it (a counterpane's creases)
PILLOW_MIN = 0.04        # a pillow stands at least this above the surface


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
    return np.stack([g[:, 0], -g[:, 2], g[:, 1]], 1)        # glTF y-up -> asset x, y, z (front -y)


def lie_box(P: np.ndarray) -> dict:
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    x0, y0 = x.min(), y.min()
    ix = ((x - x0) / CELL).astype(int)
    iy = ((y - y0) / CELL).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    top = np.full((nx, ny), np.nan)
    order = np.argsort(z)
    top[ix[order], iy[order]] = z[order]                    # last write wins: the highest vertex per cell
    # a sparse mesh leaves cells with no vertex (a flat counterpane's big faces): filled from the highest of the
    # cells round them, up to 3 cells away
    for _ in range(3):
        nan_ = np.isnan(top)
        if not nan_.any():
            break
        pad = np.pad(top, 1, constant_values=np.nan)
        nb = np.stack([pad[1 + dx:nx + 1 + dx, 1 + dy:ny + 1 + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        with np.errstate(all="ignore"):
            fill = np.nanmax(np.where(np.isnan(nb), -np.inf, nb), axis=0)
        fill[np.isinf(fill)] = np.nan
        top[nan_] = fill[nan_]
    cx, cy = nx // 2, ny // 2
    mid = top[max(cx - 10, 0):cx + 10, max(cy - 20, 0):cy + 20]
    vals = mid[~np.isnan(mid)]
    hist, edges = np.histogram(vals, bins=np.arange(vals.min(), vals.max() + 0.02, 0.01))
    surf = float(edges[np.argmax(hist)] + 0.005)
    ok = np.abs(top - surf) < LIE_TOL
    ok[np.isnan(top)] = False
    # the connected run of surface cells round the middle
    seen = np.zeros_like(ok)
    stack = [(cx, cy)] if ok[cx, cy] else []
    while stack:
        i, k = stack.pop()
        if i < 0 or k < 0 or i >= nx or k >= ny or seen[i, k] or not ok[i, k]:
            continue
        seen[i, k] = True
        stack += [(i + 1, k), (i - 1, k), (i, k + 1), (i, k - 1)]
    # the box: the rows and columns where most of the run is (a crease or a post corner does not set an edge)
    cols = np.nonzero(seen.sum(1) > 0.5 * seen.sum(1).max())[0]
    rows = np.nonzero(seen.sum(0) > 0.5 * seen.sum(0).max())[0]
    bx = [float(x0 + cols.min() * CELL), float(x0 + (cols.max() + 1) * CELL)]
    by = [float(y0 + rows.min() * CELL), float(y0 + (rows.max() + 1) * CELL)]
    # the head end: the end whose board stands higher (beyond the box along y, within the bed's width)
    inx = (x > bx[0]) & (x < bx[1])
    hi_lo = float(z[inx & (y < by[0])].max()) if (inx & (y < by[0])).any() else 0.0
    hi_hi = float(z[inx & (y > by[1])].max()) if (inx & (y > by[1])).any() else 0.0
    head = "+y" if hi_hi >= hi_lo else "-y"
    # the pillow: cells inside the bed's width near the head end, standing above the surface
    pil = (top > surf + PILLOW_MIN)
    xs_ok = np.zeros(nx, bool)
    xs_ok[cols.min():cols.max() + 1] = True
    pil &= xs_ok[:, None]
    yc = y0 + (np.arange(ny) + 0.5) * CELL
    near_head = (yc > by[1] - 0.6) if head == "+y" else (yc < by[0] + 0.6)
    inside_y = (yc > by[0] - 0.3) & (yc < by[1] + 0.3)
    board_free = top < surf + 0.30                       # not the headboard
    pil &= near_head[None, :] & inside_y[None, :] & board_free
    pz = float(np.nanmax(np.where(pil, top, np.nan))) if pil.any() else None
    py = None
    if pil.any():
        kk = np.nonzero(pil.any(0))[0]
        py = [round(float(y0 + kk.min() * CELL), 3), round(float(y0 + (kk.max() + 1) * CELL), 3)]
    return {"posture": "lying", "name": "mattress",
            "box": {"top_z_m": round(surf, 3), "x_m": [round(bx[0], 3), round(bx[1], 3)],
                    "y_m": [round(by[0], 3), round(by[1], 3)]},
            "head": head, "pillow_z_m": None if pz is None else round(pz, 3), "pillow_y_m": py,
            "cover_z_m": round(surf, 3), "front_asset": [0.0, -1.0] if head == "+y" else [0.0, 1.0],
            "length_m": round(by[1] - by[0], 3), "width_m": round(bx[1] - bx[0], 3),
            "source": "characters/liebox.py from the exported mesh"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    for d in args:
        d = Path(d)
        rec = json.loads((d / "asset.json").read_text())
        P = glb_points(d / rec["export"]["glb"])
        box = lie_box(P)
        print(d.name, json.dumps(box))
        if write:
            sup = [s for s in rec.get("supports", []) if not (s.get("posture") == "lying" and s.get("name") == "mattress")]
            sup.append(box)
            rec["supports"] = sup
            (d / "asset.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
