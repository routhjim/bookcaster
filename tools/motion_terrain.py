"""The SET's surface for the motion engine (any story): a pano world's depth seen from its capture point -> the
terrain height under any (x, y), and whether the set blocks a line of sight. numpy only (no Blender).

A HY-World pano set is a depth surface around its capture point (world.json capture_camera; points.npy = the
equirect point map in world metres). The set's surface along any direction from the capture point is the depth
there; a point farther than that depth is inside / behind the set. terrain(x, y) marches straight down from above the
set and returns the first height that is inside the surface (rock tops, slopes, the ground) - what a falling creature
or a tail lands on. Without a world (a plain ground set), the terrain is the flat ground z = 0.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

_CACHE: dict = {}


def world_dir_of(shot: dict) -> str | None:
    st = shot.get("set") or {}
    wdir = (st.get("world") or {}).get("dir")
    if wdir:
        return wdir
    where = st.get("where") or shot.get("where")
    bk = shot.get("book")
    cands = [shot.get("assets"), os.environ.get("BLOCKOUT_ASSETS")]
    if bk and os.path.isdir(bk):
        cands.append(os.path.join(bk, "work", "video", "stage", "assets.json"))
    elif bk:
        try:
            import characters as CH
            cands.append(str(CH.vdir_of(bk) / "stage" / "assets.json"))
        except Exception:  # noqa: BLE001
            pass
    for c in [c for c in cands if c and os.path.exists(c)]:
        ent = next((e for e in json.load(open(c)).get("sets", []) if e.get("where") == where), None)
        if ent:
            return ent["world_dir"]
    return None


class World:
    def __init__(self, wdir: str | None):
        self.ok = False
        self.center = self.R = None
        self.O = np.array([0.0, 0.0, 1.6])
        self.ground = 0.0
        if not wdir or not os.path.exists(os.path.join(wdir, "world.json")):
            return
        W = json.load(open(os.path.join(wdir, "world.json")))
        vr = W.get("view_region") or {}
        self.O = np.asarray(W["capture_camera"]["pos"], float)
        self.center = np.asarray(vr.get("center") or self.O, float)
        self.R = float(vr.get("radius_m", 6.0))
        self.ground = float((W.get("ground") or {}).get("plane_z", 0.0))
        self.lo = np.asarray(W.get("bounds_min") or [-20, -20, -1], float)
        self.hi = np.asarray(W.get("bounds_max") or [20, 20, 8], float)
        pp = os.path.join(wdir, "points.npy")
        if not os.path.exists(pp):
            return
        P = np.load(pp, mmap_mode="r")[::2, ::2].reshape(-1, 3).astype(float)
        q = P - self.O
        d = np.linalg.norm(q, axis=1)
        ok = np.isfinite(d) & (d > 0.2)
        q, d = q[ok], d[ok]
        self.na, self.ne = 720, 360
        ia, ie = self._bins(q, d)
        dep = np.full((self.ne, self.na), np.inf)
        np.minimum.at(dep, (ie, ia), d)
        for _ in range(4):                                  # empty bins from their neighbours
            bad = ~np.isfinite(dep)
            if not bad.any():
                break
            nb = np.stack([np.roll(dep, s, ax) for ax in (0, 1) for s in (-1, 1)])
            nb[~np.isfinite(nb)] = np.nan
            with np.errstate(all="ignore"):
                fill = np.nanmin(nb, 0)
            dep[bad] = fill[bad]
        dep[~np.isfinite(dep)] = 1e4
        self.dep = dep
        self.ok = True
        self._hm = None

    def _bins(self, q, r):
        az = np.arctan2(q[..., 1], q[..., 0])
        el = np.arcsin(np.clip(q[..., 2] / np.maximum(r, 1e-9), -1, 1))
        ia = ((az + math.pi) / (2 * math.pi) * self.na).astype(int) % self.na
        ie = np.clip(((el + math.pi / 2) / math.pi * self.ne).astype(int), 0, self.ne - 1)
        return ia, ie

    def solid(self, X):
        """(n,3) -> bool: inside / behind the set surface (seen from the capture point)."""
        X = np.atleast_2d(np.asarray(X, float))
        if not self.ok:
            return X[:, 2] < self.ground
        q = X - self.O
        r = np.linalg.norm(q, axis=1)
        ia, ie = self._bins(q, r)
        return r >= self.dep[ie, ia]

    def heightmap(self, cell=0.4):
        if self._hm is None:
            xs = np.arange(self.lo[0], self.hi[0] + cell, cell)
            ys = np.arange(self.lo[1], self.hi[1] + cell, cell)
            zs = np.arange(self.hi[2] + 1.0, self.ground - 0.5, -0.05)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            H = np.full(X.shape, self.ground)
            done = np.zeros(X.shape, bool)
            for z in zs:
                pts = np.stack([X, Y, np.full(X.shape, z)], -1).reshape(-1, 3)
                s = self.solid(pts).reshape(X.shape) & ~done
                H[s] = z
                done |= s
            H[~done] = self.ground
            self._hm = (xs, ys, cell, H)
        return self._hm

    def terrain(self, x, y):
        """Height of the set surface under (x, y) (array-friendly); the flat ground off the map / without a world."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if not self.ok:
            return np.full(np.broadcast(x, y).shape, self.ground)
        xs, ys, c, H = self.heightmap()
        fx = np.clip((x - xs[0]) / c, 0, len(xs) - 1.001)
        fy = np.clip((y - ys[0]) / c, 0, len(ys) - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        a, b = fx - i, fy - j
        h = (H[i, j] * (1 - a) * (1 - b) + H[i + 1, j] * a * (1 - b) + H[i, j + 1] * (1 - a) * b + H[i + 1, j + 1] * a * b)
        off = (x < self.lo[0]) | (x > self.hi[0]) | (y < self.lo[1]) | (y > self.hi[1])
        return np.where(off, self.ground, h)


def world_for(shot: dict) -> World:
    wd = world_dir_of(shot)
    if wd not in _CACHE:
        _CACHE[wd] = World(wd)
    return _CACHE[wd]
