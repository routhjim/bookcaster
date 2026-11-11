"""Mesh-side gate for a generated spread-pose figure: do the arms come free of the body (no web of cloth between the
sleeve's underside and the garment's side)? A drawing can pass pose_gate.py and its mesh still web.

    python -I mesh_gate.py FIGURE.glb [...] [--min-free 0.70] [--min-gap 0.05] [--elbow 0.68]

The figure's vertices are projected on its front plane (x across, z up, z = 0 at the feet, as a fraction of its
height H) and sliced every 0.4 % of H; in each slice the occupied x cells (0.4 % of H wide) form runs. Per side, a
slice is separated when the outermost run on that side is not the run that holds the body's centre line. Measured:
  free   the highest slice, walking up from the hands, of the unbroken stretch of separated slices (a real armpit,
         with the arm out ~40-45 degrees, is at ~0.73 of the height);
  gap    the air between the body's run and the arm's run at elbow height (--elbow, the spread pose's elbow), / H.
Passes when both sides have free >= --min-free and gap >= --min-gap. One JSON line per figure; exit 1 on any fail.
Plain numpy over the vertex list (milliseconds; no meshing).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load_points(path):
    import trimesh
    m = trimesh.load(path, force="mesh")
    P = np.asarray(m.vertices)
    # glTF is Y-up: front plane x-y -> x, z
    return np.c_[P[:, 0], P[:, 1]]


def runs(occ):
    d = np.diff(np.r_[0, occ.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def gate(P2, min_free, min_gap, elbow, cell=0.004):
    x, z = P2[:, 0], P2[:, 1]
    H = z.max() - z.min()
    zn = (z - z.min()) / H
    cx = np.median(x[(zn > 0.45) & (zn < 0.6)])
    xn = (x - cx) / H
    nx = int(1.4 / cell)
    xi = np.clip(((xn + 0.7) / cell).astype(int), 0, nx - 1)
    zs = np.arange(0.40, 0.86, cell)
    occ = np.zeros((len(zs), nx), bool)
    zi = np.clip(((zn - 0.40) / cell).astype(int), -1, len(zs))
    ok = (zi >= 0) & (zi < len(zs))
    occ[zi[ok], xi[ok]] = True
    # close the surface sampling's speckle: a cell with occupied cells on both sides within 1 counts as occupied
    occ = occ | (np.roll(occ, 1, 1) & np.roll(occ, -1, 1))
    occ = occ | (np.roll(occ, 1, 0) & np.roll(occ, -1, 0))
    c0 = int(0.7 / cell)
    out = {"height": round(float(H), 3)}
    allok = True
    for side, sg in (("left_of_picture", -1), ("right_of_picture", 1)):
        sep, gaps = [], []
        for k in range(len(zs)):
            rs = runs(occ[k])
            body = [r for r in rs if r[0] <= c0 < r[1]]
            if not body:
                sep.append(False)
                gaps.append(0.0)
                continue
            b = body[0]
            outer = [r for r in rs if (r[0] >= b[1] if sg > 0 else r[1] <= b[0])]
            if not outer:
                sep.append(False)
                gaps.append(0.0)
                continue
            near = min(outer, key=lambda r: (r[0] - b[1]) if sg > 0 else (b[0] - r[1]))
            g = ((near[0] - b[1]) if sg > 0 else (b[0] - near[1])) * cell
            sep.append(True)
            gaps.append(g)
        sep = np.array(sep) & (np.array(gaps) >= 2 * cell)
        # free: walking down from the shoulders, the first slice from which the arm stays apart from the body for
        # 0.04 of the height below it (a real armpit; a single parted slice in the shoulder does not count)
        win = int(0.04 / cell)
        free = None
        for k in range(len(zs) - 1, win, -1):
            if zs[k] > 0.84:
                continue
            if sep[k - win:k + 1].all():
                free = float(zs[k])
                break
        ke = int(round((elbow - 0.40) / cell))
        gap = float(max(gaps[max(ke - 2, 0):ke + 3])) if 0 <= ke < len(zs) else 0.0
        p = free is not None and free >= min_free and gap >= min_gap
        out[side] = {"free": round(free, 3) if free is not None else None, "gap_at_elbow": round(gap, 3), "pass": p}
        allok &= p
    out["pass"] = bool(allok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figures", nargs="+")
    ap.add_argument("--min-free", type=float, default=0.70)
    ap.add_argument("--min-gap", type=float, default=0.05)
    ap.add_argument("--elbow", type=float, default=0.68)
    a = ap.parse_args()
    ok = True
    for f in a.figures:
        r = gate(load_points(f), a.min_free, a.min_gap, a.elbow)
        r["file"] = f
        print(json.dumps(r), flush=True)
        ok &= r["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
