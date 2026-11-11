"""Delete the hidden inner wall of a picture-to-3D mesh (torch on a GPU; trimesh for the file).

    python -I inner_wall.py IN.glb OUT.glb [--voxel-frac 0.0016] [--push 1.5]

TRELLIS.2's surface comes out as a hollow double wall: a second surface about 0.4% of the height (one remesh band
each side) behind ~98% of the visible surface, the two joined through small holes; for some shapes (the dragon) two
separate full shells. Rigging, cutting and weighting all go wrong on it. Method: rasterise the surface into a voxel
grid (voxel = voxel-frac x the largest extent, ~3 mm on a 1.8 m figure), grow it by close-frac (the gap between the
walls is ~0.4% of the height and opens to the outside through small holes), flood the air in from the grid's border
(repeated 3x3x3 max-pooling restricted to empty voxels, on the GPU), then keep a face when its centre moved `push` + close
voxels along its normal lands in that outside air. Faces of the inner wall look into the sealed gap or the hollow
inside and go. Writes OUT.glb (texture and UVs kept) and prints the share of faces kept. No CPU fallback.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--voxel-frac", type=float, default=0.0016)
ap.add_argument("--push", type=float, default=1.5)
ap.add_argument("--close-frac", type=float, default=0.006, help="gap closing, fraction of the largest extent")
ap.add_argument("--method", default="rays", choices=("rays", "air"))
ap.add_argument("--rays", type=int, default=24)
ap.add_argument("--chunk", type=int, default=32768)
a = ap.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("no GPU visible (HIP_VISIBLE_DEVICES): this step never runs on the CPU")
dev = "cuda"
t0 = time.time()
sc = trimesh.load(a.inp, process=False)
geoms = list(sc.geometry.items()) if isinstance(sc, trimesh.Scene) else [("mesh", sc)]
name, m = max(geoms, key=lambda kv: len(kv[1].faces))
V = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev)
Fc = torch.tensor(np.asarray(m.faces), dtype=torch.long, device=dev)
lo, hi = V.min(0).values, V.max(0).values
vox = float((hi - lo).max()) * a.voxel_frac
pad = 3
dims = (((hi - lo) / vox).ceil().long() + 2 * pad).tolist()
origin = lo - pad * vox
occ = torch.zeros(dims, dtype=torch.bool, device=dev)
T = V[Fc]                                                        # (F, 3, 3)
# rasterise: points on each face at better than half-voxel spacing
e = torch.stack([(T[:, 1] - T[:, 0]).norm(dim=1), (T[:, 2] - T[:, 1]).norm(dim=1), (T[:, 0] - T[:, 2]).norm(dim=1)], 1)
n_sub = int(min(64, max(2, float(e.max() / (0.5 * vox)) + 1)))
for i in range(n_sub + 1):
    for j in range(n_sub + 1 - i):
        u, v = i / n_sub, j / n_sub
        P = T[:, 0] * (1 - u - v) + T[:, 1] * u + T[:, 2] * v
        idx = ((P - origin) / vox).long()
        for k in range(3):
            idx[:, k].clamp_(0, dims[k] - 1)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
# close the gap between the two walls (and the small holes that join it to the outside): grow the solid by
# `close` voxels before flooding, so the outside air cannot run into the gap
close = max(1, int(round(a.close_frac * float((hi - lo).max()) / vox)))
occ0 = occ
if close:
    occ = F.max_pool3d(occ.float()[None, None], 2 * close + 1, 1, close)[0, 0] > 0.5
# flood the outside air from the border through empty voxels
air = torch.zeros_like(occ, dtype=torch.float32)
air[0], air[-1], air[:, 0], air[:, -1], air[:, :, 0], air[:, :, -1] = 1, 1, 1, 1, 1, 1
free = (~occ).float()
air *= free
it = 0
while True:
    nxt = F.max_pool3d(air[None, None], 3, 1, 1)[0, 0] * free
    it += 1
    if it % 16 == 0 and torch.equal(nxt, air):
        break
    air = nxt
    if it > 4 * max(dims):
        break
# grow the air back into the creases the closing filled (folds, eye sockets), through free voxels of the original
# surface only, so it stops at the outer wall and does not enter the gap behind it
free0 = (~occ0).float()
def grow6(x):
    """one step of 6-connected growth (a one-voxel surface does not leak diagonally, as a 3x3x3 window would)"""
    y = x.clone()
    y[1:] = torch.maximum(y[1:], x[:-1]); y[:-1] = torch.maximum(y[:-1], x[1:])
    y[:, 1:] = torch.maximum(y[:, 1:], x[:, :-1]); y[:, :-1] = torch.maximum(y[:, :-1], x[:, 1:])
    y[:, :, 1:] = torch.maximum(y[:, :, 1:], x[:, :, :-1]); y[:, :, :-1] = torch.maximum(y[:, :, :-1], x[:, :, 1:])
    return y


for _ in range(close + 1):
    air = torch.maximum(air, grow6(air) * free0)
air = air > 0.5
C = T.mean(1)
N0 = torch.linalg.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
N0 = N0 / N0.norm(dim=1, keepdim=True).clamp_min(1e-12)
if a.method == "rays":
    # visibility: march a cone of rays from each face, on both sides, through the surface voxels (occ0) and keep
    # the face when any ray leaves the grid without touching the surface again; the inner wall's rays all meet the
    # outer wall within a few voxels or stay in the sealed hollow
    g = torch.Generator(device=dev).manual_seed(0)
    dirs = torch.randn(a.rays, 3, device=dev, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    occf = occ0
    dimt = torch.tensor(dims, device=dev)
    keep = torch.zeros(len(Fc), dtype=torch.bool, device=dev)
    steps = torch.arange(1.5, float(max(dims)) * 1.8, 0.8, device=dev)
    for c0 in range(0, len(Fc), a.chunk):
        cc, nn_ = C[c0:c0 + a.chunk], N0[c0:c0 + a.chunk]
        esc = torch.zeros(len(cc), dtype=torch.bool, device=dev)
        for sg in (1.0, -1.0):
            n = nn_ * sg
            # rays in the hemisphere around n: flip the random directions that point away from it, lean them to n
            d = dirs[None, :, :].expand(len(cc), -1, -1)
            d = torch.where(((d * n[:, None, :]).sum(-1, keepdim=True) < 0), -d, d)
            d = d + n[:, None, :]
            d = d / d.norm(dim=-1, keepdim=True)
            o = (cc - origin) / vox
            alive = torch.ones(d.shape[:2], dtype=torch.bool, device=dev)
            done = torch.zeros(d.shape[:2], dtype=torch.bool, device=dev)
            for t in steps:
                p = (o[:, None, :] + d * t).long()
                out = ((p < 0) | (p >= dimt)).any(-1)
                done |= out & alive
                alive &= ~out
                if not alive.any():
                    break
                pc = p.clamp(min=0)
                pc = torch.minimum(pc, dimt - 1)
                hit = occf[pc[..., 0], pc[..., 1], pc[..., 2]]
                alive &= ~hit
            esc |= done.any(1)
        keep[c0:c0 + a.chunk] = esc
    keep = keep.cpu().numpy()
else:
    keep = None
# keep the faces that look out into that air
C = T.mean(1)
N = torch.linalg.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
N = N / N.norm(dim=1, keepdim=True).clamp_min(1e-12)
keep_air = torch.zeros(len(Fc), dtype=torch.bool, device=dev)
for s in (a.push, a.push + 1.0, a.push + 2.0):                  # a few depths: thin parts, grazing normals
    for sg in (1.0, -1.0):                                       # either side: the port's winding is not reliable
        q = ((C + sg * N * s * vox - origin) / vox).long()
        for k in range(3):
            q[:, k].clamp_(0, dims[k] - 1)
        keep_air |= air[q[:, 0], q[:, 1], q[:, 2]]
if keep is None:
    keep = keep_air.cpu().numpy()
m2 = m.copy()
m2.update_faces(keep)
m2.remove_unreferenced_vertices()
if isinstance(sc, trimesh.Scene):
    sc.geometry[name] = m2
    sc.export(a.out)
else:
    m2.export(a.out)
rep = {"faces_in": int(len(keep)), "faces_kept": int(keep.sum()), "kept_frac": round(float(keep.mean()), 3),
       "voxel_m": round(vox, 5), "close_vox": close, "grid": dims, "flood_iters": it, "seconds": round(time.time() - t0, 1)}
print("INNER_WALL", json.dumps(rep), flush=True)
