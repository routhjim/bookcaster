"""Scanner-style shape from all orbit frames, on one card (PyTorch): per-frame consistency, a visual hull from the
masks, a TSDF from the per-frame depth, and the surface extracted on the GPU (marching tetrahedra), with a vertex
colour pass weighted towards the first frame (the drawing).

    HIP_VISIBLE_DEVICES=<card> python -I fuse.py RECON_DIR OUT_DIR [--res 384] [--drop-frac 0.1]

RECON_DIR: recon_chunks.py's pred.npz plus masks_cams.py's masks/ and cams.json.
OUT_DIR/consistency.json (per frame: azimuth, mask agreement with all other frames, depth agreement with its
neighbours, weight), OUT_DIR/mesh.ply (vertex colours, asset frame: metres, Z up, origin on the floor under the
subject centre, X along the subject's long axis), OUT_DIR/frame.json (world -> asset transform, metres per world
unit, cameras in the asset frame).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("recon")
ap.add_argument("out")
ap.add_argument("--res", type=int, default=384, help="voxels along the longest axis")
ap.add_argument("--length-m", type=float, default=10.0, help="the subject's length in metres")
ap.add_argument("--drop-frac", type=float, default=0.1, help="drop this fraction of the least consistent frames")
ap.add_argument("--carve-votes", type=int, default=3, help="a voxel is carved when this many views see it outside")
ap.add_argument("--trunc-vox", type=float, default=4.0)
ap.add_argument("--use-depth", type=int, default=1)
ap.add_argument("--smooth", type=int, default=8)
a = ap.parse_args()
t0 = time.time()
dev = "cuda"
R, O = Path(a.recon), Path(a.out)
O.mkdir(parents=True, exist_ok=True)
P = np.load(R / "pred.npz")
cams = json.loads((R / "cams.json").read_text())
depth = torch.from_numpy(P["depth"]).to(dev).float()
conf = torch.from_numpy(P["conf"]).to(dev).float()
img = torch.from_numpy(P["img"]).to(dev).float() / 255
S, H, W = depth.shape
mask = torch.stack([torch.from_numpy(np.asarray(Image.open(R / "masks" / f"m_{i:03d}.png"))).to(dev)
                    for i in range(S)]).float() / 255 > 0.5
c2w = torch.tensor(np.array([f["c2w"] for f in cams["frames"]]), device=dev, dtype=torch.float32)
K = torch.from_numpy(P["K"]).to(dev).float()
w2c = torch.linalg.inv(c2w)
up = torch.tensor(cams["up"], device=dev)
az = np.array([f["az"] for f in cams["frames"]])


def project(X, i):
    """X [N,3] world -> u, v (pixels), z for frame i."""
    Xc = X @ w2c[i, :3, :3].T + w2c[i, :3, 3]
    z = Xc[:, 2]
    u = K[i, 0, 0] * Xc[:, 0] / z + K[i, 0, 2]
    v = K[i, 1, 1] * Xc[:, 1] / z + K[i, 1, 2]
    return u, v, z


def sample(t, i, u, v):
    """nearest-pixel lookup of t[i] (H,W[,C]) at u,v; returns values and an in-image flag."""
    ui, vi = u.floor().long(), v.floor().long()
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi = ui.clamp(0, W - 1), vi.clamp(0, H - 1)
    return t[i][vi, ui], ok


vv, uu = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")


def backproject(i, sel):
    z = depth[i][sel]
    x = (uu[sel] - K[i, 0, 2]) / K[i, 0, 0] * z
    y = (vv[sel] - K[i, 1, 2]) / K[i, 1, 1] * z
    return torch.stack([x, y, z], -1) @ c2w[i, :3, :3].T + c2w[i, :3, 3]


# ---------------- per-frame consistency
mask_d = F.max_pool2d(mask[:, None].float(), 5, 1, 2)[:, 0] > 0.5          # 2 px slack
sil_in = np.zeros(S)
dep_err = np.zeros(S)
gen = torch.Generator(device=dev).manual_seed(0)
order = np.argsort(az)
for i in range(S):
    sel = mask[i]
    X = backproject(i, sel)
    X = X[torch.randperm(len(X), device=dev, generator=gen)[:4000]]
    fr = []
    for j in range(S):
        if j == i:
            continue
        u, v, z = project(X, j)
        m, ok = sample(mask_d, j, u, v)
        ok = ok & (z > 0)
        if ok.sum() > 100:
            fr.append(float(m[ok].float().mean()))
    sil_in[i] = np.mean(fr)
    # depth agreement with the frames nearest in azimuth (either clip), visible points only
    nb = np.argsort(np.abs(((az - az[i] + 180) % 360) - 180))[1:5]
    errs = []
    for j in nb:
        u, v, z = project(X, int(j))
        dj, ok = sample(depth, int(j), u, v)
        mj, _ = sample(mask, int(j), u, v)
        ok = ok & mj & (z > 0)
        if ok.sum() > 100:
            r = ((z[ok] - dj[ok]) / dj[ok]).abs()
            errs.append(float(r.median()))
    dep_err[i] = np.mean(errs) if errs else np.nan
# weight: a frame whose points fall outside the others' masks or disagree with its neighbours counts less
bad = (1 - sil_in) / max(np.median(1 - sil_in), 1e-4) + dep_err / max(np.nanmedian(dep_err), 1e-4)
wf = 1.0 / (1.0 + np.maximum(bad - 2, 0))
drop = set(np.argsort(-bad)[:int(round(a.drop_frac * S))].tolist()) - {0}
keep = [i for i in range(S) if i not in drop]
wf_t = torch.tensor(wf, device=dev, dtype=torch.float32)
t_cons = time.time() - t0

# ---------------- grid in an up-aligned box round the subject
sub = torch.from_numpy(np.load(R / "subject_points.npy")).to(dev)
centre = torch.tensor(cams["centre"], device=dev)
e1 = torch.tensor(cams["e1"], device=dev)
e2 = torch.cross(up, e1, dim=0)
rel = sub - centre
h = rel @ up
# long axis: principal direction in the ground plane
gp = torch.stack([rel @ e1, rel @ e2], -1)
ev, evec = torch.linalg.eigh(gp.T @ gp)
ax = evec[:, 1]
xa = ax[0] * e1 + ax[1] * e2
ya = torch.cross(up, xa, dim=0)
B = torch.stack([xa, ya, up])                                        # rows: asset axes in world
loc = rel @ B.T
lo = torch.quantile(loc[::3], 0.002, dim=0)
hi = torch.quantile(loc[::3], 0.998, dim=0)
pad = (hi - lo) * 0.08
lo, hi = lo - pad, hi + pad
lo[2] = cams["floor_h"] - float(centre @ up) - 0.002 * float(hi[2] - lo[2])
vox = float((hi - lo).max()) / a.res
dims = ((hi - lo) / vox).ceil().long() + 1
nx, ny, nz = [int(x) for x in dims]
gx = lo[0] + vox * torch.arange(nx, device=dev)
gy = lo[1] + vox * torch.arange(ny, device=dev)
gz = lo[2] + vox * torch.arange(nz, device=dev)
print(f"grid {nx}x{ny}x{nz}, voxel {vox:.4f} world units", flush=True)


def world_of(ix_slab):
    """world points of an x-slab of the grid [ny*nz*len,3]"""
    X, Y, Z = torch.meshgrid(gx[ix_slab], gy, gz, indexing="ij")
    L = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    return L @ B + centre


tau = a.trunc_vox * vox
tsdf = torch.zeros(nx, ny, nz, device=dev)
wsum = torch.zeros(nx, ny, nz, device=dev)
outv = torch.zeros(nx, ny, nz, device=dev, dtype=torch.int16)
seen = torch.zeros(nx, ny, nz, device=dev, dtype=torch.int16)
slab = max(1, int(4e6 // (ny * nz)))
for s0 in range(0, nx, slab):
    ix = torch.arange(s0, min(nx, s0 + slab), device=dev)
    X = world_of(ix)
    ts = torch.zeros(len(X), device=dev)
    ws = torch.zeros(len(X), device=dev)
    ov = torch.zeros(len(X), device=dev, dtype=torch.int16)
    sn = torch.zeros(len(X), device=dev, dtype=torch.int16)
    for i in keep:
        u, v, z = project(X, i)
        m, ok = sample(mask_d, i, u, v)
        ok = ok & (z > 0)
        sn += ok.short()
        ov += (ok & ~m).short()
        if a.use_depth:
            d, _ = sample(depth, i, u, v)
            mm, _ = sample(mask, i, u, v)
            c, _ = sample(conf, i, u, v)
            sdf = d - z
            use = ok & mm & (sdf > -tau)
            w = wf_t[i] * c.clamp(max=10) * (3.0 if i == 0 else 1.0)
            ts += torch.where(use, sdf.clamp(max=tau) * w, 0)
            ws += torch.where(use, w, 0)
    sl = (slice(s0, s0 + len(ix)),)
    tsdf[sl] = (ts / ws.clamp(min=1e-6)).view(len(ix), ny, nz)
    wsum[sl] = ws.view(len(ix), ny, nz)
    outv[sl] = ov.view(len(ix), ny, nz)
    seen[sl] = sn.view(len(ix), ny, nz)
hull = (outv < a.carve_votes) & (seen > len(keep) * 0.3)
# field: negative inside. Hull alone gives -tau inside, +tau outside; the depth TSDF refines it where observed.
field = torch.where(hull, -tau * torch.ones_like(tsdf), tau * torch.ones_like(tsdf))
if a.use_depth:
    obs = wsum > 1e-3
    field = torch.where(hull & obs, tsdf.clamp(-tau, tau), field)
    # keep the hull boundary where depth says "inside" but the masks carve it away
    field = torch.where(~hull, torch.maximum(field, torch.full_like(field, tau)), field)
# light 3-D blur of the field (box 3) for a cleaner zero set
f5 = F.avg_pool3d(field[None, None], 3, 1, 1)[0, 0]
field = f5
t_fuse = time.time() - t0 - t_cons
print(f"hull {int(hull.sum())} voxels; consistency {t_cons:.1f}s fuse {t_fuse:.1f}s", flush=True)

# ---------------- marching tetrahedra (Kuhn: six tets per cube on the 0-7 diagonal) on the GPU
corner = torch.tensor([[(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)], device=dev)
tets = []
import itertools  # noqa: E402
for p in itertools.permutations(range(3)):
    path, cur = [0], 0
    for axis in p:
        cur |= 1 << axis
        path.append(cur)
    tets.append(path)
tets = torch.tensor(tets, device=dev)                                # 6,4 corner ids
inside = field < 0
# cubes with a sign change among their 8 corners
cmin = -F.max_pool3d(-(inside.float())[None, None], 2, 1)[0, 0]
cmax = F.max_pool3d(inside.float()[None, None], 2, 1)[0, 0]
cubes = torch.nonzero(cmax != cmin)                                   # [M,3]
cv = cubes[:, None, :] + corner[None]                                 # M,8,3
gid = (cv[..., 0] * ny + cv[..., 1]) * nz + cv[..., 2]                # M,8 global vertex ids
fv = field.reshape(-1)[gid]                                           # M,8
T = gid[:, tets]                                                      # M,6,4
TV = fv[:, tets]
T, TV = T.reshape(-1, 4), TV.reshape(-1, 4)
ins = TV < 0
cnt = ins.sum(1)
keept = (cnt > 0) & (cnt < 4)
T, TV, ins, cnt = T[keept], TV[keept], ins[keept], cnt[keept]
# sort each tet's corners inside-first
key_ = (~ins).float() * 2 + torch.arange(4, device=dev) * 0.1
srt = key_.argsort(1)
T, TV = T.gather(1, srt), TV.gather(1, srt)
edge_keys, edge_t = [], []


def ekey(a_, b_):
    lo_, hi_ = torch.minimum(a_, b_), torch.maximum(a_, b_)
    return lo_ * (nx * ny * nz) + hi_


tri_list = []
# one inside corner (0 in, 1..3 out): triangle (0,1),(0,2),(0,3)
# three inside (0..2 in, 3 out): triangle (0,3),(1,3),(2,3)
# two inside (0,1 in; 2,3 out): quad (0,2),(0,3),(1,3),(1,2)
cases = {1: [[(0, 1), (0, 2), (0, 3)]], 3: [[(0, 3), (1, 3), (2, 3)]],
         2: [[(0, 2), (0, 3), (1, 3)], [(0, 2), (1, 3), (1, 2)]]}
for c_, tris in cases.items():
    sel = cnt == c_
    for tri in tris:
        tri_list.append(torch.stack([ekey(T[sel, i_], T[sel, j_]) for i_, j_ in tri], 1))
F_keys = torch.cat(tri_list)                                          # Ftri,3 edge keys
uk, inv = torch.unique(F_keys.reshape(-1), return_inverse=True)
faces = inv.view(-1, 3)
ia, ib = uk // (nx * ny * nz), uk % (nx * ny * nz)
fa, fb = field.reshape(-1)[ia], field.reshape(-1)[ib]
tt = (fa / (fa - fb)).clamp(0, 1)


def gpos(g):
    z_ = g % nz
    y_ = (g // nz) % ny
    x_ = g // (ny * nz)
    return torch.stack([gx[x_], gy[y_], gz[z_]], -1)


V = gpos(ia) * (1 - tt[:, None]) + gpos(ib) * tt[:, None]             # asset-local (world units)
# orient each face along the field gradient (outward = increasing field)
gz_, gy_, gx_ = torch.gradient(field)[2], torch.gradient(field)[1], torch.gradient(field)[0]
cen = V[faces].mean(1)
ci = ((cen - lo) / vox).round().long()
ci = torch.stack([ci[:, 0].clamp(0, nx - 1), ci[:, 1].clamp(0, ny - 1), ci[:, 2].clamp(0, nz - 1)], -1)
g = torch.stack([gx_[ci[:, 0], ci[:, 1], ci[:, 2]], gy_[ci[:, 0], ci[:, 1], ci[:, 2]],
                 gz_[ci[:, 0], ci[:, 1], ci[:, 2]]], -1)
n = torch.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]], dim=1)
flip = (n * g).sum(1) < 0
faces = torch.where(flip[:, None], faces[:, [0, 2, 1]], faces)
# drop degenerate faces
faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
# keep the largest connected piece (label propagation on the GPU)
lab = torch.arange(len(V), device=dev)
for _ in range(2000):
    m_ = torch.minimum(torch.minimum(lab[faces[:, 0]], lab[faces[:, 1]]), lab[faces[:, 2]])
    new = lab.clone()
    for k in range(3):
        new.scatter_reduce_(0, faces[:, k], m_, reduce="amin")
    new = new[new]
    if torch.equal(new, lab):
        break
    lab = new
big = torch.bincount(lab[faces[:, 0]]).argmax()
faces = faces[lab[faces[:, 0]] == big]
used = torch.unique(faces)
remap = torch.full((len(V),), -1, device=dev, dtype=torch.long)
remap[used] = torch.arange(len(used), device=dev)
V, faces = V[used], remap[faces]
# Taubin smoothing (lambda/mu) on the GPU
E = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
E = torch.cat([E, E.flip(1)])
deg = torch.bincount(E[:, 0], minlength=len(V)).float().clamp(min=1)
for it in range(a.smooth * 2):
    lam = 0.5 if it % 2 == 0 else -0.53
    avg = torch.zeros_like(V).index_add_(0, E[:, 0], V[E[:, 1]]) / deg[:, None]
    V = V + lam * (avg - V)
t_mesh = time.time() - t0 - t_cons - t_fuse

# ---------------- vertex colour: visible views (depth test against the TSDF surface itself), drawing first
Vw = V @ B + centre
fn = torch.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]], dim=1)
vn = torch.zeros_like(V).index_add_(0, faces.reshape(-1), fn.repeat_interleave(3, 0))
vn = F.normalize(vn, dim=1) @ B                                       # world
col = torch.zeros(len(V), 3, device=dev)
cw = torch.zeros(len(V), device=dev)
for i in keep:
    u, v, z = project(Vw, i)
    d, ok = sample(depth, i, u, v)
    mm, _ = sample(mask, i, u, v)
    camc = c2w[i, :3, 3]
    facing = (F.normalize(camc - Vw, dim=1) * vn).sum(1)
    vis = ok & mm & (z > 0) & ((z - d).abs() < 3 * tau) & (facing > 0.1)
    rgb, _ = sample(img, i, u, v)
    w = facing.clamp(min=0) ** 2 * wf_t[i] * (8.0 if i == 0 else 1.0)
    col += torch.where(vis[:, None], rgb * w[:, None], 0)
    cw += torch.where(vis, w, 0)
col = col / cw.clamp(min=1e-6)[:, None]
seen_v = cw > 0
t_col = time.time() - t0 - t_cons - t_fuse - t_mesh

# ---------------- asset frame: metres, Z up, origin on the floor under the centre, X along the long axis
length = float(V[:, 0].max() - V[:, 0].min())
mpu = a.length_m / length
Va = (V - torch.tensor([float((V[:, 0].max() + V[:, 0].min()) / 2), float((V[:, 1].max() + V[:, 1].min()) / 2),
                        float(lo[2])], device=dev)) * mpu
off = torch.tensor([float((V[:, 0].max() + V[:, 0].min()) / 2), float((V[:, 1].max() + V[:, 1].min()) / 2),
                    float(lo[2])], device=dev)
Vn, Fn, Cn = Va.cpu().numpy(), faces.cpu().numpy(), (col.clamp(0, 1) * 255).byte().cpu().numpy()


def write_ply(p, Vn, Fn, Cn):
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(Vn)}\nproperty float x\nproperty float y\n"
           f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
           f"element face {len(Fn)}\nproperty list uchar int vertex_indices\nend_header\n")
    vd = np.zeros(len(Vn), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    vd["p"], vd["c"] = Vn, Cn
    fd = np.zeros(len(Fn), dtype=[("n", "u1"), ("f", "<i4", 3)])
    fd["n"], fd["f"] = 3, Fn
    with open(p, "wb") as fh:
        fh.write(hdr.encode())
        fh.write(vd.tobytes())
        fh.write(fd.tobytes())


write_ply(O / "mesh.ply", Vn, Fn, Cn)
# cameras in the asset frame (OpenCV c2w), processed-size K
Bn, cn, offn = B.cpu().numpy(), centre.cpu().numpy(), off.cpu().numpy()
fr = []
for i in range(S):
    cw_ = c2w[i].cpu().numpy()
    Rc = Bn @ cw_[:3, :3]
    tc = (Bn @ (cw_[:3, 3] - cn) - offn) * mpu
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Rc, tc
    fr.append({"i": i, "path": cams["frames"][i]["path"], "c2w": M.tolist(), "K": P["K"][i].tolist(),
               "az": float(az[i]), "weight": round(float(wf[i]), 3), "kept": i in keep})
(O / "frame.json").write_text(json.dumps({"B": Bn.tolist(), "centre": cn.tolist(), "offset": offn.tolist(),
                                          "m_per_unit": mpu, "size": [W, H], "pp": json.loads(str(P["pp"])),
                                          "frames": fr}, indent=1))
ext = (Va.max(0).values - Va.min(0).values).cpu().numpy()
cons = {"frames": [{"i": i, "path": cams["frames"][i]["path"].split("/")[-2] + "/" +
                    cams["frames"][i]["path"].split("/")[-1], "az": round(float(az[i]), 1),
                    "mask_agree": round(float(sil_in[i]), 3), "depth_err_rel": round(float(dep_err[i]), 4),
                    "badness": round(float(bad[i]), 2), "weight": round(float(wf[i]), 3), "kept": i in keep}
                   for i in range(S)],
        "dropped": sorted(drop), "grid": [nx, ny, nz], "voxel_m": vox * mpu, "verts": len(Vn), "faces": len(Fn),
        "seen_vert_frac": round(float(seen_v.float().mean()), 3), "extent_m": ext.tolist(),
        "seconds": {"consistency": round(t_cons, 1), "fuse": round(t_fuse, 1), "mesh": round(t_mesh, 1),
                    "colour": round(t_col, 1), "total": round(time.time() - t0, 1)}}
(O / "consistency.json").write_text(json.dumps(cons, indent=1))
print(json.dumps({k: cons[k] for k in ("grid", "voxel_m", "verts", "faces", "seen_vert_frac", "extent_m", "seconds")}))
