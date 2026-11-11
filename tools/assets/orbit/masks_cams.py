"""After a WorldMirror pass over orbit frames (recon pred.npz: img, depth, conf, w2c, K): object masks and cameras,
all on the GPU (torch). No segmentation model: the subject stands on a plain grey studio floor, so

  mask = (chroma or lightness off a smooth 2-D background model) AND (its reconstructed point stands above the
         floor plane)

which keeps cream horns and pale membrane (chroma) and drops the soft floor shadows (on the plane).

    python -I masks_cams.py RECON_DIR [--above 0.03]

Writes RECON_DIR/masks/m_###.png (processed size), RECON_DIR/cams.json: per frame the source path, OpenCV c2w,
K (processed size), azimuth around the subject (degrees, first frame = 0, unwrapped), mask cover; and the subject
centre, up vector, floor height, the azimuth basis, the subject's extent.
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
ap.add_argument("--above", type=float, default=0.03, help="height above the floor, fraction of the subject height")
ap.add_argument("--chroma", type=float, default=0.06)
ap.add_argument("--light", type=float, default=0.10)
a = ap.parse_args()
t0 = time.time()
dev = "cuda"
R = Path(a.recon)
P = np.load(R / "pred.npz")
img = torch.from_numpy(P["img"]).to(dev).float() / 255           # S,H,W,3
depth = torch.from_numpy(P["depth"]).to(dev).float()
w2c = np.asarray(P["w2c"])[:, :3, :4]
K = np.asarray(P["K"])
S, H, W = depth.shape
paths = [str(x) for x in P["paths"]]


def blur(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, 1)


def dil(m, r):
    return F.max_pool2d(m, 2 * r + 1, 1, r)


def ero(m, r):
    return -F.max_pool2d(-m, 2 * r + 1, 1, r)


# ---- colour key against a smooth background (normalised convolution of the background-looking pixels)
x = img.permute(0, 3, 1, 2)                                          # S,3,H,W
chroma = (x.max(1, keepdim=True)[0] - x.min(1, keepdim=True)[0])
lum = (x * torch.tensor([0.2126, 0.7152, 0.0722], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
obj = (chroma > a.chroma).float()
for _ in range(4):
    w = (1 - dil(obj, 4)) * (chroma < a.chroma).float()
    sm = 4
    lw = F.avg_pool2d(lum * w, sm)
    ww = F.avg_pool2d(w, sm)
    r = 12
    bgl = blur(lw, r) / blur(ww, r).clamp(min=1e-4)
    bgl = F.interpolate(bgl, size=(H, W), mode="bilinear", align_corners=False)
    off = (lum - bgl).abs()
    obj = ((chroma > a.chroma) | (off > a.light)).float()
key = obj

# ---- cameras, points, floor
E = np.tile(np.eye(4), (S, 1, 1))
E[:, :3, :4] = w2c
c2w = np.linalg.inv(E)
up = -c2w[:, :3, 1].mean(0)
up /= np.linalg.norm(up)
c2w_t = torch.from_numpy(c2w).to(dev).float()
Kt = torch.from_numpy(K).to(dev).float()
v, u = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
pts = []
for i in range(S):
    z = depth[i]
    xc = torch.stack([(u - Kt[i, 0, 2]) / Kt[i, 0, 0] * z, (v - Kt[i, 1, 2]) / Kt[i, 1, 1] * z, z], -1)
    pts.append(xc @ c2w_t[i, :3, :3].T + c2w_t[i, :3, 3])
pts = torch.stack(pts)                                               # S,H,W,3
upt = torch.from_numpy(up).to(dev).float()
h = pts @ upt                                                        # height along up
# floor = the dominant height among the lower-image background-looking pixels
lowbg = (key[:, 0] < 0.5) & (v[None] > H * 0.6)
hs = h[lowbg]
lo_, hi_ = torch.quantile(hs[::7], 0.02), torch.quantile(hs[::7], 0.98)
hist = torch.histc(hs, bins=400, min=float(lo_), max=float(hi_))
floor = float(lo_ + (hist.argmax().float() + 0.5) * (hi_ - lo_) / 400)
# subject height from the keyed pixels above the floor
hk = h[key[:, 0] > 0.5] - floor
top = float(torch.quantile(hk[::7], 0.99))
above = (h - floor) > a.above * top
m = key * above[:, None].float()
# dark, grey-looking patches inside the body (shaded scales) fail the colour key: fill them where the geometry
# says subject (above the floor and nearer than the backdrop) and the colour mask closes around them
dk = torch.stack([torch.quantile(depth[i][m[i, 0] > 0.5][::3], 0.995) if m[i, 0].sum() > 100 else depth[i].max()
                  for i in range(S)])
geo = (above & (depth < dk[:, None, None]))[:, None].float()
m = torch.maximum(m, ero(dil(m, 8), 8) * geo)
m = ero(dil(m, 2), 2)
m = dil(ero(m, 1), 1)
# keep the main blob: grow from the strongest core
core = ero(m, 6)
grow = core
for _ in range(600):
    nxt = dil(grow, 1) * m
    if torch.equal(nxt, grow):
        break
    grow = nxt
m = grow
(R / "masks").mkdir(exist_ok=True)
mn = m[:, 0].cpu().numpy()
for i in range(S):
    Image.fromarray((mn[i] * 255).astype(np.uint8)).save(R / "masks" / f"m_{i:03d}.png")

# ---- subject centre, extent and azimuths
sel = m[:, 0] > 0.5
P3 = pts[sel]
c = P3.median(0).values
d = (P3 - c).norm(dim=1)
P3 = P3[d < 3 * torch.quantile(d[::5], 0.9)]
c = P3.mean(0).cpu().numpy()
cams = c2w[:, :3, 3]
e1 = cams[0] - c
e1 -= up * (e1 @ up)
e1 /= np.linalg.norm(e1)
e2 = np.cross(up, e1)
rel = cams - c
az = np.degrees(np.unwrap(np.arctan2(rel @ e2, rel @ e1)))
sub = P3[torch.randperm(len(P3), device=dev)[:200000]].cpu().numpy()
np.save(R / "subject_points.npy", sub.astype(np.float32))
info = {"frames": [{"i": i, "path": paths[i], "c2w": c2w[i].tolist(), "K": K[i].tolist(), "az": round(float(az[i]), 2),
                    "cover": round(float(mn[i].mean()), 4)} for i in range(S)],
        "size": [int(W), int(H)], "centre": c.tolist(), "up": up.tolist(), "e1": e1.tolist(), "e2": e2.tolist(),
        "floor_h": floor, "top_above_floor": top,
        "radius_mean": float(np.linalg.norm(rel - np.outer(rel @ up, up), axis=1).mean()),
        "seconds": round(time.time() - t0, 2)}
(R / "cams.json").write_text(json.dumps(info, indent=1))
print(json.dumps({"frames": S, "az_range": [round(float(az.min()), 1), round(float(az.max()), 1)], "floor": floor,
                  "top": top, "seconds": info["seconds"]}))
