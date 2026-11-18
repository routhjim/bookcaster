"""Gaussian-splat (2DGS surfels by default, or 3DGS) fit to every orbit frame at full resolution, on one card, with
gsplat (ROCm build; plain rasterisation kernels, nothing that needs host calls). Per-frame camera corrections and
per-frame colour gain/bias absorb the video model's drift; the first frame (the drawing) is fixed and drawn most.

    HIP_VISIBLE_DEVICES=<card> python -I splat.py FUSE_DIR MASK_DIR OUT [--iters 15000] [--mode 2dgs]

FUSE_DIR: fuse.py's frame.json (cameras in the asset frame) and mesh.ply (the start points).
MASK_DIR: m_###.png masks at the processed size (masks_cams.py).
OUT/splats.ply (standard 3DGS layout), OUT/render/pred.npz + masks/ + cams.json + subject_points.npy (rendered depth,
alpha, colour of every frame, in fuse.py's input layout, so fuse.py can mesh the splats), OUT/fit.json (per frame:
photometric error, alpha IoU with the mask, camera correction), OUT/views_*.jpg.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization, rasterization_2dgs
from gsplat.strategy import DefaultStrategy

ap = argparse.ArgumentParser()
ap.add_argument("fuse")
ap.add_argument("masks")
ap.add_argument("out")
ap.add_argument("--iters", type=int, default=15000)
ap.add_argument("--mode", default="2dgs", choices=["2dgs", "3dgs"])
ap.add_argument("--init-n", type=int, default=300000)
ap.add_argument("--drawing-weight", type=float, default=6.0, help="how much more often the first frame is drawn")
ap.add_argument("--pose-opt", type=int, default=1)
ap.add_argument("--use-dropped", type=int, default=0)
ap.add_argument("--render-scale", type=float, default=1.0, help="training resolution, fraction of the frames'")
ap.add_argument("--clips", default="", help="comma list of clip folder names to train on (the first frame always)")
ap.add_argument("--holdout", type=int, default=0, help="hold out every Nth frame (never the first) for PSNR")
ap.add_argument("--app-opt", type=int, default=1, help="per-frame colour gain/bias")
ap.add_argument("--mesh-renders", type=int, default=1)
ap.add_argument("--init-recon", default="", help="start from the masked depth points of this recon (pred.npz + masks)")
a = ap.parse_args()
t0 = time.time()
dev = "cuda"
torch.manual_seed(0)
FD, O = Path(a.fuse), Path(a.out)
O.mkdir(parents=True, exist_ok=True)
fj = json.loads((FD / "frame.json").read_text())
pp = fj["pp"]
frames = [f for f in fj["frames"] if f["kept"] or a.use_dropped or f["i"] == 0]
if a.clips:
    frames = [f for f in frames if f["i"] == 0 or Path(f["path"]).parent.name in a.clips.split(",")]
S = len(frames)
held = np.zeros(S, bool)
if a.holdout:
    held[np.arange(a.holdout // 2, S, a.holdout)] = True
    held[0] = False

# ---- images (full size) and masks (processed size, upsampled)
im0 = Image.open(frames[0]["path"])
W0, H0 = im0.size
W, H = int(round(W0 * a.render_scale)), int(round(H0 * a.render_scale))
imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(f["path"]).convert("RGB").resize((W, H), Image.BICUBIC)))
                    for f in frames]).to(dev)                                    # S,H,W,3 uint8
mk = torch.stack([torch.from_numpy(np.asarray(Image.open(Path(a.masks) / f"m_{f['i']:03d}.png")))
                  for f in frames]).to(dev).float()[:, None] / 255
# processed image = resize(source) then centre crop: map back to the source frame
sx, sy = pp["scale_x"], pp["scale_y"]
full = torch.zeros(S, 1, int(round(pp["new_h"])), int(round(pp["new_w"])), device=dev)
full[:, :, pp["crop_y"]:pp["crop_y"] + mk.shape[2], pp["crop_x"]:pp["crop_x"] + mk.shape[3]] = mk
masks = F.interpolate(full, size=(H, W), mode="bilinear", align_corners=False)[:, 0].clamp(0, 1)
Ks = []
for f in frames:
    K = np.array(f["K"], dtype=np.float64)
    Kf = np.array([[K[0, 0] / sx, 0, (K[0, 2] + pp["crop_x"]) / sx], [0, K[1, 1] / sy, (K[1, 2] + pp["crop_y"]) / sy],
                   [0, 0, 1]]) * np.array([[a.render_scale], [a.render_scale], [1]])
    Ks.append(Kf)
Ks = torch.tensor(np.array(Ks), device=dev, dtype=torch.float32)
c2w0 = torch.tensor(np.array([f["c2w"] for f in frames]), device=dev, dtype=torch.float32)
wfr = np.array([max(f["weight"], 0.05) for f in frames])
wfr[0] *= a.drawing_weight
wfr[held] = 0
prob = torch.tensor(wfr / wfr.sum(), device=dev)

# ---- start points: area-weighted samples on the fused mesh, with its vertex colours
raw = (FD / "mesh.ply").read_bytes()
hdr_end = raw.index(b"end_header\n") + len(b"end_header\n")
hdr = raw[:hdr_end].decode()
nv = int(hdr.split("element vertex ")[1].split()[0])
nf = int(hdr.split("element face ")[1].split()[0])
vd = np.frombuffer(raw, dtype=[("p", "<f4", 3), ("c", "u1", 3)], count=nv, offset=hdr_end)
fdd = np.frombuffer(raw, dtype=[("n", "u1"), ("f", "<i4", 3)], count=nf, offset=hdr_end + vd.nbytes)
V = torch.from_numpy(vd["p"].copy()).to(dev)
C = torch.from_numpy(vd["c"].copy()).to(dev).float() / 255
Fc = torch.from_numpy(fdd["f"].astype(np.int64)).to(dev)
tri = V[Fc]
area = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1).norm(dim=1) / 2
pick = torch.multinomial(area / area.sum(), a.init_n, replacement=True)
r1, r2 = torch.rand(a.init_n, 1, device=dev), torch.rand(a.init_n, 1, device=dev)
s1 = r1.sqrt()
bw = torch.cat([1 - s1, s1 * (1 - r2), s1 * r2], 1)
means = (tri[pick] * bw[..., None]).sum(1)
cols = (C[Fc[pick]] * bw[..., None]).sum(1)
spacing = math.sqrt(float(area.sum()) / a.init_n)
if a.init_recon:
    # every masked depth pixel of the training frames, in the asset frame (fuse.py's transform)
    RP = np.load(Path(a.init_recon) / "pred.npz")
    Bm = torch.tensor(fj["B"], device=dev, dtype=torch.float32)
    cen = torch.tensor(fj["centre"], device=dev, dtype=torch.float32)
    off = torch.tensor(fj["offset"], device=dev, dtype=torch.float32)
    pts_, cols_ = [], []
    for f in frames:
        i_ = f["i"]
        d_ = torch.from_numpy(RP["depth"][i_]).to(dev)
        m_ = torch.from_numpy(np.asarray(Image.open(Path(a.masks) / f"m_{i_:03d}.png"))).to(dev) > 127
        Kp = torch.from_numpy(RP["K"][i_]).to(dev)
        w2c_ = torch.eye(4, device=dev)
        w2c_[:3] = torch.from_numpy(RP["w2c"][i_]).to(dev)
        c2w_ = torch.linalg.inv(w2c_)
        vv_, uu_ = torch.nonzero(m_, as_tuple=True)
        z_ = d_[vv_, uu_]
        xc = torch.stack([(uu_ + 0.5 - Kp[0, 2]) / Kp[0, 0] * z_, (vv_ + 0.5 - Kp[1, 2]) / Kp[1, 1] * z_, z_], -1)
        Xw = xc @ c2w_[:3, :3].T + c2w_[:3, 3]
        pts_.append(((Xw - cen) @ Bm.T - off) * fj["m_per_unit"])
        cols_.append(torch.from_numpy(RP["img"][i_]).to(dev)[vv_, uu_].float() / 255)
    P_ = torch.cat(pts_)
    C_ = torch.cat(cols_)
    sel_ = torch.randperm(len(P_), device=dev)[:a.init_n]
    means, cols = P_[sel_], C_[sel_]
    del RP, pts_, cols_, P_, C_
scene = float((V.max(0).values - V.min(0).values).max())
SH0 = 0.28209479177387814
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(means),
    "scales": torch.nn.Parameter(torch.full((a.init_n, 3), math.log(spacing), device=dev)),
    "quats": torch.nn.Parameter(F.normalize(torch.randn(a.init_n, 4, device=dev), dim=1)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((a.init_n,), 0.5, device=dev))),
    "sh0": torch.nn.Parameter(((cols - 0.5) / SH0)[:, None, :]),
})
lrs = {"means": 1.6e-4 * scene, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / a.iters))
strategy = DefaultStrategy(verbose=False, refine_stop_iter=int(a.iters * 0.6), reset_every=3000,
                           refine_every=100, refine_start_iter=500, grow_grad2d=0.0002,
                           key_for_gradient="gradient_2dgs" if a.mode == "2dgs" else "means2d")
strategy.check_sanity(params, opts)
sstate = strategy.initialize_state(scene_scale=scene)

# ---- per-frame camera correction (axis-angle + translation) and colour gain/bias; frame 0 fixed
dpose = torch.nn.Parameter(torch.zeros(S, 6, device=dev))
app = torch.nn.Parameter(torch.zeros(S, 6, device=dev))
opt_x = torch.optim.Adam([{"params": [dpose], "lr": 2e-4 if a.pose_opt else 0.0},
                          {"params": [app], "lr": 2e-3 if a.app_opt else 0.0}])


def rodrigues(r):
    th = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / th
    Kx = torch.zeros(r.shape[0], 3, 3, device=r.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th[..., None]
    return torch.eye(3, device=r.device) + th.sin() * Kx + (1 - th.cos()) * Kx @ Kx


def cam(idx):
    dp = dpose[idx] * (idx != 0).float()[:, None]
    M = c2w0[idx].clone()
    Rd = rodrigues(dp[:, :3] * 0.1)
    M = torch.cat([torch.cat([M[:, :3, :3] @ Rd, M[:, :3, 3:] + dp[:, 3:, None] * scene * 0.01], 2), M[:, 3:]], 1)
    return torch.linalg.inv(M)


def render(idx, bg, mode="RGB+ED"):
    vm = cam(idx)
    q, s, o = params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"])
    if a.mode == "2dgs":
        rc, ra, rn, rnd, rdist, rmed, info = rasterization_2dgs(
            params["means"], q, s, o, params["sh0"], vm, Ks[idx], W, H, sh_degree=0, backgrounds=bg,
            render_mode=mode, distloss=True, depth_mode="median")
        return rc, ra, info, (rn, rnd, rdist, rmed)
    rc, ra, info = rasterization(params["means"], q, s, o, params["sh0"], vm, Ks[idx], W, H, sh_degree=0,
                                 backgrounds=bg, render_mode=mode, packed=False)
    return rc, ra, info, None


def ssim(x, y):
    g = torch.exp(-((torch.arange(11, device=dev) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None]).expand(3, 1, 11, 11)
    mu_x, mu_y = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


t_setup = time.time() - t0
log = []
for step in range(a.iters):
    i = int(torch.multinomial(prob, 1))
    idx = torch.tensor([i], device=dev)
    bg = torch.rand(1, 3, device=dev)
    rc, ra, info, extra = render(idx, bg)
    rgb, alpha = rc[0, ..., :3], ra[0, ..., 0]
    gain, bias = 1 + app[i, :3] * (i != 0), app[i, 3:] * (i != 0)
    rgb = (rgb - bg[0] * (1 - alpha[..., None])) * gain + bias + bg[0] * (1 - alpha[..., None])
    m = masks[i]
    gt = imgs[i].float() / 255 * m[..., None] + bg[0] * (1 - m[..., None])
    l1 = (rgb - gt).abs().mean()
    loss = 0.8 * l1 + 0.2 * (1 - ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
    loss = loss + 0.5 * (alpha - m).abs().mean()
    if extra is not None:
        rn, rnd, rdist, _ = extra
        if step > 3000:
            nerr = (1 - (rn[0] * (rnd[0] * alpha[..., None].detach())).sum(-1)) * m
            loss = loss + 0.05 * nerr.mean()
        if step > 1500:
            loss = loss + 0.01 * rdist.mean()
    loss = loss + 1e-3 * dpose.pow(2).mean() + 1e-3 * app.pow(2).mean()
    strategy.step_pre_backward(params, opts, sstate, step, info)
    loss.backward()
    # the ROCm port's backward leaves non-finite gradients on some splats (measured in the toy fit: culled and
    # tiny ones); their true gradient is zero, so zero them before Adam and the densifier read them
    nonfinite = 0
    for p_ in list(params.values()) + [dpose, app]:
        if p_.grad is not None:
            bad_ = ~torch.isfinite(p_.grad)
            nonfinite += int(bad_.any(-1).sum()) if p_.grad.dim() > 1 else int(bad_.sum())
            p_.grad.nan_to_num_(0.0, 0.0, 0.0)
    for k_ in ("means2d", "gradient_2dgs"):
        if k_ in info and getattr(info[k_], "grad", None) is not None:
            info[k_].grad.nan_to_num_(0.0, 0.0, 0.0)
    for o_ in opts.values():
        o_.step()
        o_.zero_grad(set_to_none=True)
    opt_x.step()
    opt_x.zero_grad(set_to_none=True)
    sched.step()
    strategy.step_post_backward(params, opts, sstate, step, info, packed=False)
    if step % 500 == 0 or step == a.iters - 1:
        msg = (f"step {step} loss {float(loss):.4f} l1 {float(l1):.4f} n {len(params['means'])} "
               f"nonfinite-grad rows {nonfinite} {time.time() - t0:.0f}s")
        print(msg, flush=True)
        log.append(msg)
t_train = time.time() - t0 - t_setup

# ---- per-frame fit: photometric error inside the mask, alpha IoU, camera correction; renders for meshing
R = O / "render"
(R / "masks").mkdir(parents=True, exist_ok=True)
fit = []
D_all, A_all, I_all = [], [], []
with torch.no_grad():
    for k in range(S):
        idx = torch.tensor([k], device=dev)
        bg = torch.zeros(1, 3, device=dev)
        rc, ra, info, extra = render(idx, bg)
        rgb, alpha = rc[0, ..., :3], ra[0, ..., 0]
        dep = extra[3][0, ..., 0] if extra is not None else rc[0, ..., 3]
        m = masks[k] > 0.5
        gain, bias = 1 + app[k, :3] * (k != 0), app[k, 3:] * (k != 0)
        pred = rgb * gain + bias
        e = (pred - imgs[k].float() / 255).abs().mean(-1)
        # PSNR over the subject's box (mask dilated), composited on the frame's own background
        mb = F.max_pool2d(masks[k][None, None], 31, 1, 15)[0, 0] > 0.5
        comp = pred + (1 - alpha[..., None]) * (imgs[k].float() / 255)
        mse = ((comp - imgs[k].float() / 255) ** 2)[mb].mean()
        psnr = float(-10 * torch.log10(mse))
        am = alpha > 0.5
        iou = float((am & m).sum() / (am | m).sum().clamp(min=1))
        dp = dpose[k] * (k != 0)
        if held[k] or k == 0 or k % 24 == 0:
            vis = torch.cat([imgs[k].float() / 255, comp.clamp(0, 1)], 1)
            (O / "views").mkdir(exist_ok=True)
            Image.fromarray((vis * 255).byte().cpu().numpy()).save(O / "views" / f"{'held' if held[k] else 'train'}_{frames[k]['i']:03d}.jpg")
        fit.append({"i": frames[k]["i"], "az": frames[k]["az"], "held": bool(held[k]), "psnr": round(psnr, 2),
                    "l1_in_mask": round(float(e[m].mean()), 4),
                    "alpha_iou": round(iou, 4), "rot_fix_deg": round(float(dp[:3].norm() * 0.1 * 180 / math.pi), 3),
                    "move_fix_m": round(float(dp[3:].norm() * scene * 0.01), 4)})
        D_all.append(dep.cpu().numpy().astype(np.float32))
        A_all.append(alpha.cpu().numpy().astype(np.float32))
        I_all.append((rgb.clamp(0, 1) * 255).byte().cpu().numpy())
        Image.fromarray((am.cpu().numpy() * 255).astype(np.uint8)).save(R / "masks" / f"m_{k:03d}.png")
    c2w_fix = torch.linalg.inv(cam(torch.arange(S, device=dev))).cpu().numpy()
    Ksn = Ks.cpu().numpy()
np.savez(R / "pred.npz", depth=np.stack(D_all), conf=np.stack(A_all), img=np.stack(I_all),
         w2c=np.linalg.inv(c2w_fix)[:, :3, :4].astype(np.float32), K=Ksn.astype(np.float32),
         paths=np.array([f["path"] for f in frames]),
         pp=json.dumps({"orig_w": W0, "orig_h": H0, "new_w": W, "new_h": H, "crop_x": 0, "crop_y": 0,
                        "final_w": W, "final_h": H, "scale_x": W / W0, "scale_y": H / H0}))
mn = params["means"].detach()
op = torch.sigmoid(params["opacities"].detach())
pts = mn[op > 0.3]
ctr = pts.mean(0)
(R / "cams.json").write_text(json.dumps({
    "frames": [{"i": k, "path": frames[k]["path"], "c2w": c2w_fix[k].tolist(), "K": Ksn[k].tolist(),
                "az": frames[k]["az"]} for k in range(S)],
    "centre": ctr.tolist(), "up": [0, 0, 1], "e1": [1, 0, 0], "e2": [0, 1, 0], "floor_h": 0.0}, indent=1))
np.save(R / "subject_points.npy", pts[torch.randperm(len(pts), device=dev)[:200000]].cpu().numpy())

# ---- splats as a standard 3DGS ply (2DGS surfels get a thin third axis)
sc = params["scales"].detach().clone()
if a.mode == "2dgs":
    sc[:, 2] = math.log(1e-4 * scene)
q = F.normalize(params["quats"].detach(), dim=1)
cols = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"]
arr = torch.cat([mn, torch.zeros_like(mn), params["sh0"].detach()[:, 0], params["opacities"].detach()[:, None], sc,
                 q], 1).cpu().numpy().astype("<f4")
with open(O / "splats.ply", "wb") as fh:
    fh.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(arr) +
              "".join(f"property float {c}\n" for c in cols) + "end_header\n").encode())
    fh.write(arr.tobytes())
l1s = np.array([f["l1_in_mask"] for f in fit])
ps = np.array([f["psnr"] for f in fit])
hf = np.array([f["held"] for f in fit])
ious = np.array([f["alpha_iou"] for f in fit])
summary = {"mode": a.mode, "iters": a.iters, "frames": S, "train_res": [W, H], "splats": len(arr),
           "seconds": {"setup": round(t_setup, 1), "train": round(t_train, 1), "total": round(time.time() - t0, 1)},
           "l1_in_mask_median": float(np.median(l1s)),
           "psnr_train_median": float(np.median(ps[~hf])), "psnr_held_median": float(np.median(ps[hf])) if hf.any() else None,
           "psnr_held_min": float(ps[hf].min()) if hf.any() else None, "psnr_drawing": float(ps[0]), "alpha_iou_median": float(np.median(ious)),
           "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), "log": log}
(O / "fit.json").write_text(json.dumps({"summary": summary, "frames": fit}, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "log"}))
