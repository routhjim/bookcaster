"""Register an image-to-3D mesh to the orbit frames' cameras by silhouette (torch, on a GPU), then express mesh and
cameras in the asset contract's frame (metres, Z up, front -Y, base on Z = 0, origin at the floor centre).

    HIP_VISIBLE_DEVICES=<card> <python with torch + trimesh> -I fit.py RECON_DIR MESH.glb OUT_DIR
        [--front-frame I] [--size-m 11.7] [--iters 400] [--refine 300]

RECON_DIR: masks_cams.py output (cams.json, masks/). MESH.glb: Y-up, as Hunyuan emits it (front +Z in glTF, i.e.
-Y once Z is up). Steps:
1. global similarity (scale, rotation, translation) mesh -> recon world: a yaw grid seeds it (mesh up = the
   orbit's up, mesh front turned toward --front-frame's camera +- the grid), then Adam on a soft point-splat
   silhouette against every frame's mask at two blur scales;
2. per-frame camera correction (small rotation, translation, focal scale; L2-regularised), since video frames
   are not a rigid capture;
3. writes OUT_DIR/mesh_asset.ply (asset frame, metres), OUT_DIR/cams_asset.json (per frame: Blender camera
   matrix_world, lens for a 36 mm sensor, full-resolution size, principal-point shift, the frame's path, mask path,
   silhouette IoU before/after refinement), OUT_DIR/fit.json (losses, IoUs, transform, times).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("recon")
ap.add_argument("mesh")
ap.add_argument("out")
ap.add_argument("--front-frame", type=int, default=None)
ap.add_argument("--size-m", type=float, default=11.7, help="largest bounding-box side in metres")
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--refine", type=int, default=300)
ap.add_argument("--points", type=int, default=160000)
ap.add_argument("--full-size", type=int, nargs=2, default=None, help="source frame size W H (default: read it)")
a = ap.parse_args()
t0 = time.time()
dev = "cuda"
R = Path(a.recon)
O = Path(a.out)
O.mkdir(parents=True, exist_ok=True)
C = json.load(open(R / "cams.json"))
S = len(C["frames"])
W, H = C["size"]
up = np.array(C["up"])
centre = np.array(C["centre"])

# ---- mesh (glTF Y-up -> Z-up: (x, y, z) -> (x, -z, y)), normalised: bbox centre at 0, height 1
tm = trimesh.load(a.mesh, force="mesh")
V = np.asarray(tm.vertices, np.float64)
V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)
Fc = np.asarray(tm.faces)
lo, hi = V.min(0), V.max(0)
V = (V - (lo + hi) / 2) / (hi[2] - lo[2])
Vt = torch.from_numpy(V).float().to(dev)
Ft = torch.from_numpy(Fc).long().to(dev)
tri = Vt[Ft]
area = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1)
g = torch.Generator(device=dev).manual_seed(0)


def sample(n):
    fi = torch.multinomial(area, n, replacement=True, generator=g)
    r1 = torch.rand(n, 1, device=dev, generator=g).sqrt()
    r2 = torch.rand(n, 1, device=dev, generator=g)
    t = tri[fi]
    return (1 - r1) * t[:, 0] + r1 * (1 - r2) * t[:, 1] + r1 * r2 * t[:, 2]


PTS = sample(a.points)

# ---- frames: masks at half the processed size, cameras (OpenCV)
sc = 0.5
w, h = int(W * sc), int(H * sc)
masks = []
for i in range(S):
    m = np.asarray(Image.open(R / "masks" / f"m_{i:03d}.png")).astype(np.float32) / 255
    masks.append(torch.from_numpy(m)[None, None])
M = F.interpolate(torch.cat(masks).to(dev), size=(h, w), mode="area")       # S,1,h,w
c2w = torch.tensor([f["c2w"] for f in C["frames"]], dtype=torch.float32, device=dev)
K = torch.tensor([f["K"] for f in C["frames"]], dtype=torch.float32, device=dev)
Kl = K.clone()
Kl[:, :2] *= sc
# video frames have square pixels; WorldMirror's separate fov_x / fov_y can disagree by ~10 %. Start square
# (fy = fx) and learn one global aspect correction with the pose.
Kl[:, 1, 1] = Kl[:, 0, 0]
asp = torch.zeros(1, device=dev, requires_grad=True)


def blur(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, 1)


def so3(v):
    th = v.norm() + 1e-12
    k = v / th
    Kx = torch.zeros(3, 3, device=dev)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=dev) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx) if th.item() > 1e-9 else \
        torch.eye(3, device=dev) + Kx * th


def so3b(v):
    """batched, differentiable Rodrigues"""
    th = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k = v / th
    z = torch.zeros_like(k[..., 0])
    Kx = torch.stack([torch.stack([z, -k[..., 2], k[..., 1]], -1), torch.stack([k[..., 2], z, -k[..., 0]], -1),
                      torch.stack([-k[..., 1], k[..., 0], z], -1)], -2)
    s_, c_ = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=dev) + s_ * Kx + (1 - c_) * (Kx @ Kx)


def render(Pw, idx, dR=None, dT=None, dF=None):
    """soft silhouettes of world points Pw (N,3) in frames idx: (len(idx),1,h,w)"""
    out = []
    for j, i in enumerate(idx):
        Rc = c2w[i, :3, :3]
        Tc = c2w[i, :3, 3]
        if dR is not None:
            Rc = Rc @ so3b(dR[i])
            Tc = Tc + dT[i]
        pc = (Pw - Tc) @ Rc                                    # world -> camera (OpenCV)
        z = pc[:, 2].clamp(min=1e-4)
        fx, fy = Kl[i, 0, 0], Kl[i, 1, 1] * torch.exp(asp)[0]
        if dF is not None:
            fx, fy = fx * torch.exp(dF[i]), fy * torch.exp(dF[i])
        x = pc[:, 0] / z * fx + Kl[i, 0, 2] - 0.5
        y = pc[:, 1] / z * fy + Kl[i, 1, 2] - 0.5
        x0, y0 = torch.floor(x), torch.floor(y)
        img = torch.zeros(h * w, device=dev)
        for ox in (0, 1):
            for oy in (0, 1):
                xi, yi = x0 + ox, y0 + oy
                wgt = (1 - (x - xi).abs()) * (1 - (y - yi).abs())
                ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                img = img.index_add(0, (yi[ok] * w + xi[ok]).long(), wgt[ok])
        out.append(img.view(1, 1, h, w))
    d = torch.cat(out)
    return 1 - torch.exp(-blur(d, 1) * 3.0)


def loss_fn(sil, tgt):
    l = (sil - tgt).pow(2).mean()
    for r in (3, 9):
        l = l + (blur(sil, r) - blur(tgt, r)).pow(2).mean()
    return l


def iou(sil, tgt):
    a_, b_ = sil > 0.5, tgt > 0.5
    return ((a_ & b_).sum((1, 2, 3)).float() / (a_ | b_).sum((1, 2, 3)).clamp(min=1).float())


# ---- init: mesh +Z -> up; mesh front (-Y) toward the front camera, yaw grid
cams = c2w[:, :3, 3].cpu().numpy()
ff = a.front_frame if a.front_frame is not None else 0
fdir = cams[ff] - centre
fdir -= up * (fdir @ up)
fdir /= np.linalg.norm(fdir)
height = float(C["top_above_floor"])


def base_rot(yaw):
    f = fdir * math.cos(yaw) + np.cross(up, fdir) * math.sin(yaw)    # where mesh -Y should point
    xa = np.cross(f, up)                                              # mesh +X: front x up = right-hand side... check
    # mesh axes in world: X = xa, Y = -f, Z = up (right-handed: X = Y x Z = (-f) x up = up x f)
    xa = np.cross(up, f)
    return np.stack([xa, -f, up], 1)                                  # columns = mesh axes in world


floor_pt = centre - up * ((centre @ up) - C["floor_h"])
results = []
idx_all = list(range(S))
for yaw_deg in range(-180, 180, 30):
    Rm = torch.tensor(base_rot(math.radians(yaw_deg)), dtype=torch.float32, device=dev)
    s0 = height
    # mesh bottom (z=-0.5) on the floor, centre over the subject centre
    t0_ = torch.tensor(floor_pt + up * 0.5 * s0, dtype=torch.float32, device=dev)
    Pw = s0 * PTS @ Rm.T + t0_
    with torch.no_grad():
        l = loss_fn(render(Pw, idx_all[::3]), M[::3]).item()
    results.append((l, yaw_deg))
results.sort()
yaw0 = results[0][1]
Rm0 = torch.tensor(base_rot(math.radians(yaw0)), dtype=torch.float32, device=dev)
rv = torch.zeros(3, device=dev, requires_grad=True)
ls = torch.zeros(1, device=dev, requires_grad=True)
tt = torch.tensor(floor_pt + up * 0.5 * height, dtype=torch.float32, device=dev).requires_grad_(True)
opt = torch.optim.Adam([{"params": [rv], "lr": 0.01}, {"params": [ls], "lr": 0.01},
                        {"params": [tt], "lr": 0.002 * height * 5},
                        {"params": [asp], "lr": 0.003}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=0)
hist = []


def world_pts():
    Rg = so3b(rv) @ Rm0
    return height * torch.exp(ls) * PTS @ Rg.T + tt, Rg


for it in range(a.iters):
    idx = list(np.random.default_rng(it).choice(S, size=min(S, 8), replace=False))
    Pw, _ = world_pts()
    l = loss_fn(render(Pw, idx), M[idx])
    opt.zero_grad()
    l.backward()
    opt.step()
    sched.step()
    if it % 50 == 0:
        hist.append(round(l.item(), 5))
with torch.no_grad():
    Pw, Rg = world_pts()
    sil = render(Pw, idx_all)
    iou_g = iou(sil, M).cpu().numpy()
# ---- per-frame refinement
dR = torch.zeros(S, 3, device=dev, requires_grad=True)
dT = torch.zeros(S, 3, device=dev, requires_grad=True)
dF = torch.zeros(S, device=dev, requires_grad=True)
opt2 = torch.optim.Adam([{"params": [dR], "lr": 0.002}, {"params": [dT], "lr": 0.002 * height},
                         {"params": [dF], "lr": 0.002}])
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, a.refine, eta_min=0)
Pw = Pw.detach()
for it in range(a.refine):
    idx = list(np.random.default_rng(1000 + it).choice(S, size=min(S, 8), replace=False))
    l = loss_fn(render(Pw, idx, dR, dT, dF), M[idx])
    reg = 0.5 * dR.pow(2).sum() + 0.5 * (dT / height).pow(2).sum() + 0.5 * dF.pow(2).sum()
    opt2.zero_grad()
    (l + 0.01 * reg / S).backward()
    opt2.step()
    sched2.step()
with torch.no_grad():
    sil2 = render(Pw, idx_all, dR, dT, dF)
    iou_r = iou(sil2, M).cpu().numpy()
    # debug sheet: mask (red) v silhouette (green)
    sheet = torch.cat([M, sil2, torch.zeros_like(M)], 1)
    n = S
    cols = 6
    rows = (n + cols - 1) // cols
    canvas = torch.zeros(3, rows * h, cols * w, device=dev)
    for i in range(n):
        r_, c_ = divmod(i, cols)
        canvas[:, r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = sheet[i]
    Image.fromarray((canvas.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).resize(
        (cols * w // 2, rows * h // 2)).save(O / "fit_sheet.png")

# ---- asset frame: mesh frame (Z up, front -Y), scaled to metres, base on Z = 0, origin at the floor centre
sc_w = float(height * torch.exp(ls.detach()))                 # world units per mesh unit
Rg = Rg.detach().cpu().numpy()
tw = tt.detach().cpu().numpy()
ext = V.max(0) - V.min(0)
Mscale = a.size_m / ext.max()                         # metres per mesh unit
org = np.array([(V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2, V[:, 2].min()])
Va = (V - org) * Mscale
trimesh.Trimesh(Va, Fc, process=False).export(O / "mesh_asset.ply")
FW, FH = a.full_size if a.full_size else Image.open(C["frames"][0]["path"]).size
flip = np.diag([1.0, -1.0, -1.0])
out = []
dRn, dTn, dFn = so3b(dR).detach().cpu().numpy(), dT.detach().cpu().numpy(), dF.detach().cpu().numpy()
for i, f in enumerate(C["frames"]):
    c = np.array(f["c2w"])
    Rc = c[:3, :3] @ dRn[i]
    Tc = c[:3, 3] + dTn[i]
    # world -> mesh: Xm = Rg^T (Xw - tw) / sc_w ; mesh -> asset: (Xm - org) * Mscale
    pos = ((Rg.T @ (Tc - tw)) / sc_w - org) * Mscale
    rot = Rg.T @ Rc @ flip                             # Blender camera: -Z forward, +Y up
    Mw = np.eye(4)
    Mw[:3, :3] = rot
    Mw[:3, 3] = pos
    k = np.array(f["K"])
    sx, sy = FW / W, FH / H
    fx = k[0, 0] * sx * math.exp(float(dFn[i]))
    cx, cy = k[0, 2] * sx, k[1, 2] * sy
    out.append({"name": f"v{i:03d}", "frame": f["path"], "mask": str(R / "masks" / f"m_{i:03d}.png"),
                "az": f["az"], "size": [FW, FH], "matrix_world": Mw.tolist(), "sensor_width": 36.0,
                "lens": fx / FW * 36.0, "shift_x": -(cx - FW / 2) / FW, "shift_y": (cy - FH / 2) / FW,
                "fy_over_fx": math.exp(float(asp.detach())) * sy / sx,
                "iou_global": round(float(iou_g[i]), 4), "iou_refined": round(float(iou_r[i]), 4)})
json.dump({"views": out, "bbox": [Va.min(0).tolist(), Va.max(0).tolist()]}, open(O / "cams_asset.json", "w"),
          indent=1)
info = {"aspect_fy_over_fx": math.exp(float(asp.detach())), "yaw_grid": results[:4], "loss_hist": hist, "iou_global_mean": round(float(iou_g.mean()), 4),
        "iou_refined_mean": round(float(iou_r.mean()), 4), "iou_refined_min": round(float(iou_r.min()), 4),
        "scale_world_per_mesh": sc_w, "metres_per_mesh": Mscale, "dims_m": (Va.max(0) - Va.min(0)).tolist(),
        "faces": int(len(Fc)), "seconds": round(time.time() - t0, 1)}
json.dump(info, open(O / "fit.json", "w"), indent=1)
print(json.dumps(info))
