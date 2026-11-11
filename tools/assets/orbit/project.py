"""Orbit skin: project the orbit video frames onto the mesh's UV atlas (torch, on a GPU) and write the contract's
three maps. Each side of the subject comes from the clip that saw it.

    HIP_VISIBLE_DEVICES=<card> <python with torch> -I project.py JOB.json
    JOB {"id": asset id, "prep": bl_prep.py out folder (texel_pos.npy, texel_nrm.npy), "cams": cams_asset.json,
         "views_dir": bl_views.py out folder, "out": textures folder,
         "sides": {"<clip tag>": "-x" | "+x" | "any"}  (clip tag = the letters after the frame's index in its file
                  name, e.g. 012_c048.png -> "c"; a side clip skins only its half, blended over "band_m"),
         "band_m": 0.25, "exclude": [view names], "params": {...}}

Per texel: every view whose rendered surface point at the texel's pixel is the texel itself (visibility), inside
the frame's own mask, away from silhouettes and depth steps, weighted by facing (cos^2 for the low band, cos^8
x density^2 for the high band: the squarest, nearest view gives the fine detail, the rest average the colour).
Low and high frequencies are split per view (masked blur), per-view gains matched to the consensus. Unseen texels
take the colour of the nearest seen surface (sparse 3-D grids, finest level with data). Detail normal and
roughness from the luminance high-pass, as in the painting stage. Rows: arrays are bottom-up (Blender), PNGs
top-down.
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

JOB = json.load(open(sys.argv[1]))
P = {"p_soft": 2.0, "p_sharp": 8.0, "low_px": 10, "edge_px": 6, "vis_tol": 0.004, "fill_levels": [0.03, 0.06, 0.12,
     0.25, 0.5, 1.0], "detail_gain": 1.0, "normal_strength": 2.5, "rough_base": 0.55, "rough_strength": 0.3,
     "height_px": 3, "margin_px": 16, "mask_erode_px": 2}
P.update(JOB.get("params", {}))
dev = "cuda"
t0 = time.time()
T = {}
AID = JOB["id"]
out = Path(JOB["out"])
out.mkdir(parents=True, exist_ok=True)
prep = Path(JOB["prep"])
pos = torch.from_numpy(np.load(prep / "texel_pos.npy")).to(dev)
nrm = torch.from_numpy(np.load(prep / "texel_nrm.npy").astype(np.float32)).to(dev)
res = pos.shape[0]
cov = pos[..., 0] > -1e3
idx = cov.view(-1).nonzero()[:, 0]
tp = pos.view(-1, 3)[idx]
tn = F.normalize(nrm.view(-1, 3)[idx], dim=1)
NT = len(idx)
size = float((tp.max(0).values - tp.min(0).values).norm())
V = json.load(open(JOB["cams"]))["views"]
excl = set(JOB.get("exclude", []))
sides = JOB.get("sides", {})
band = float(JOB.get("band_m", 0.25))


def blur(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, 1)


def srgb_dec(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def srgb_enc(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def bilinear(img, x, y):
    """img C,H,W; x,y pixel coords (centres at +0.5) -> N,C"""
    C_, H_, W_ = img.shape
    gx = (x / W_) * 2 - 1
    gy = (y / H_) * 2 - 1
    g = torch.stack([gx, gy], -1).view(1, 1, -1, 2)
    return F.grid_sample(img[None], g, mode="bilinear", align_corners=False)[0, :, 0].T


def clip_tag(path):
    m = re.match(r"\d+_([a-zA-Z]+)", Path(path).name)
    return m.group(1) if m else ""


acc_l = torch.zeros(NT, 3, device=dev)
acc_lw = torch.zeros(NT, device=dev)
acc_h = torch.zeros(NT, 3, device=dev)
acc_hw = torch.zeros(NT, device=dev)
nviews = torch.zeros(NT, device=dev)
stats = []
per = []
dens_ref = None
for v in V:
    if v["name"] in excl:
        continue
    pp = Path(JOB["views_dir"]) / f"{v['name']}_pos.npy"
    if not pp.exists():
        continue
    W, H = v["size"]
    pv = torch.from_numpy(np.load(pp)).to(dev)                       # H,W,4 bottom-up
    alpha = pv[..., 3]
    posv = pv[..., :3]
    fr = np.asarray(Image.open(v["frame"]).convert("RGB").resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    fr = torch.from_numpy(fr[::-1].copy()).to(dev)                   # bottom-up
    mk = np.asarray(Image.open(v["mask"]).convert("L").resize((W, H), Image.BILINEAR)).astype(np.float32) / 255
    mk = torch.from_numpy(mk[::-1].copy()).to(dev)[None, None]
    r = int(P["mask_erode_px"])
    mk = -F.max_pool2d(-mk, 2 * r + 1, 1, r)[0, 0]
    inside = ((alpha > 0.99) & (mk > 0.5)).float()
    Mw = torch.tensor(v["matrix_world"], dtype=torch.float32, device=dev)
    Mi = torch.linalg.inv(Mw)
    camp = Mw[:3, 3]
    dist = (posv - camp).norm(dim=-1)
    di = dist * inside
    md = blur(di[None, None], 1)[0, 0] / blur(inside[None, None], 1)[0, 0].clamp(min=1e-6)
    med = dist[inside > 0].median() if inside.sum() > 0 else torch.tensor(1.0, device=dev)
    step = (((dist - md).abs() > 0.01 * med) & (inside > 0)).float()
    er = int(P["edge_px"])
    e1 = blur(inside[None, None], er)[0, 0]
    e2 = 1 - (blur(step[None, None], er)[0, 0] * 4).clamp(0, 1)
    wimg = ((e1 - 0.5) * 2).clamp(0, 1) * e2 * inside
    lin = srgb_dec(fr).permute(2, 0, 1)                               # 3,H,W linear
    lr = int(P["low_px"])
    low = blur(lin[None] * inside, lr)[0] / blur(inside[None, None], lr)[0].clamp(min=1e-4)
    high = lin - low
    # texels -> this view (Blender camera: -Z forward, +Y up; rows bottom-up)
    pc = tp @ Mi[:3, :3].T + Mi[:3, 3]
    z = -pc[:, 2]
    f = v["lens"] / v["sensor_width"] * W
    x = W / 2 + pc[:, 0] / z.clamp(min=1e-6) * f + v.get("shift_x", 0.0) * W
    y = H / 2 + pc[:, 1] / z.clamp(min=1e-6) * f * v.get("fy_over_fx", 1.0) + v.get("shift_y", 0.0) * W
    ok = (z > 1e-6) & (x >= 1) & (x < W - 1) & (y >= 1) & (y < H - 1)
    ii = ok.nonzero()[:, 0]
    xi, yi = x[ii], y[ii]
    rp = posv[yi.long(), xi.long()]
    tol = P["vis_tol"] * size + z[ii] / f * 1.5
    vis = ((rp - tp[ii]).norm(dim=1) < tol) & (inside[yi.long(), xi.long()] > 0)
    ii, xi, yi = ii[vis], xi[vis], yi[vis]
    vd = camp - tp[ii]
    dd = vd.norm(dim=1)
    cosv = ((tn[ii] * vd).sum(1) / dd).clamp(0, 1)
    dens = f / dd
    if dens_ref is None:
        dens_ref = float(dens.median()) if len(dens) else 1.0
    wb = bilinear(wimg[None], xi, yi)[:, 0]
    side = sides.get(clip_tag(v["frame"]), "any")
    if side in ("-x", "+x"):
        sx = tp[ii, 0] * (1 if side == "+x" else -1)                 # >0 on this clip's home side
        sw = ((sx + band / 2) / band).clamp(0, 1)
    else:
        sw = torch.ones_like(wb)
    lowv = bilinear(low, xi, yi)
    highv = bilinear(high, xi, yi)
    dn = dens / dens_ref
    ws = wb * cosv ** P["p_soft"] * dn * sw
    wh = wb * cosv ** P["p_sharp"] * dn ** 2 * sw
    per.append((v["name"], ii, ws, wh, lowv, highv))
    stats.append({"view": v["name"], "clip": clip_tag(v["frame"]), "texels": int(len(ii)),
                  "weight": round(float(ws.sum()), 1)})
T["views"] = round(time.time() - t0, 2)
# per-view gain against the consensus low band (two passes)
gains = {n: torch.ones(3, device=dev) for n, *_ in per}
for _ in range(2):
    acc_l.zero_()
    acc_lw.zero_()
    for n, ii, ws, wh, lowv, highv in per:
        acc_l.index_add_(0, ii, ws[:, None] * lowv * gains[n])
        acc_lw.index_add_(0, ii, ws)
    cons = acc_l / acc_lw.clamp(min=1e-9)[:, None]
    for n, ii, ws, wh, lowv, highv in per:
        good = ws > 0.05
        if good.sum() > 500:
            g = (cons[ii][good] * ws[good, None]).sum(0) / (lowv[good] * ws[good, None]).sum(0).clamp(min=1e-6)
            gains[n] = g.clamp(0.7, 1.4)
acc_l.zero_()
acc_lw.zero_()
for n, ii, ws, wh, lowv, highv in per:
    g = gains[n]
    acc_l.index_add_(0, ii, ws[:, None] * lowv * g)
    acc_lw.index_add_(0, ii, ws)
    acc_h.index_add_(0, ii, wh[:, None] * highv * g)
    acc_hw.index_add_(0, ii, wh)
    nviews.index_add_(0, ii, (ws > 0.05).float())
seen = acc_lw > 0.02
low = acc_l / acc_lw.clamp(min=1e-9)[:, None]
high = acc_h / acc_hw.clamp(min=1e-9)[:, None] * (acc_hw / (acc_hw + 0.05))[:, None]
alb = (low + high * P["detail_gain"]).clamp(0, 1)
T["blend"] = round(time.time() - t0, 2)
# unseen texels: colour of the nearest seen surface, finest 3-D grid level with data
fill = torch.zeros_like(alb)
done = seen.clone()
fill[seen] = alb[seen]
lo3 = tp.min(0).values
for cell in P["fill_levels"]:
    if bool(done.all()):
        break
    key = ((tp - lo3) / (cell * size / 10)).floor().long()
    kmax = key.max(0).values + 1
    lin_k = (key[:, 0] * kmax[1] + key[:, 1]) * kmax[2] + key[:, 2]
    uk, inv = torch.unique(lin_k, return_inverse=True)
    s = torch.zeros(len(uk), 3, device=dev).index_add_(0, inv[seen], alb[seen])
    c = torch.zeros(len(uk), device=dev).index_add_(0, inv[seen], torch.ones(int(seen.sum()), device=dev))
    have = c[inv] > 0
    take = (~done) & have
    fill[take] = (s[inv[take]] / c[inv[take], None])
    done |= take
alb = torch.where(seen[:, None], alb, fill)
T["fill"] = round(time.time() - t0, 2)
full = torch.zeros(res * res, 3, device=dev)
full[idx] = alb
full = full.view(res, res, 3)
valid = cov.clone().float()
# gutter: grow the colours into empty texels (UV seams, margins)
img = full.permute(2, 0, 1)[None] * valid
wv = valid[None, None]
for _ in range(int(P["margin_px"])):
    s_ = F.avg_pool2d(img, 3, 1, 1)
    w_ = F.avg_pool2d(wv, 3, 1, 1)
    grow = (wv == 0) & (w_ > 0)
    img = torch.where(grow, s_ / w_.clamp(min=1e-6), img)
    wv = torch.where(grow, torch.ones_like(wv), wv)
full = img[0].permute(1, 2, 0)
full[wv[0, 0] == 0] = full[wv[0, 0] > 0].mean(0)


def save(a, name):
    """a: res,res,C in [0,1], rows bottom-up -> PNG top-down"""
    arr = (a.clamp(0, 1).flip(0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(out / name)
    return str(out / name)


files = {"BaseColor": save(srgb_enc(full), f"T_{AID}_BaseColor.png")}
# detail normal + roughness from the luminance high-pass
L = (full * torch.tensor([0.2126, 0.7152, 0.0722], device=dev)).sum(-1)
vm = (wv[0, 0] > 0).float()
hr = int(P["height_px"]) * 3
Lb = blur((L * vm)[None, None], hr)[0, 0] / blur(vm[None, None], hr)[0, 0].clamp(min=1e-6)
hp = ((L - Lb) / (Lb + 0.02)).clamp(-1, 1) * vm
hb = blur(hp[None, None], 1)[0, 0]
gx = (torch.roll(hb, -1, 1) - torch.roll(hb, 1, 1)) * 0.5
gy = (torch.roll(hb, -1, 0) - torch.roll(hb, 1, 0)) * 0.5
ns = P["normal_strength"]
nn_ = F.normalize(torch.stack([-gx * ns, -gy * ns, torch.ones_like(gx)], -1), dim=-1)
files["Normal"] = save(nn_ * 0.5 + 0.5, f"T_{AID}_Normal.png")
rough = (P["rough_base"] * (1 - P["rough_strength"] * hp)).clamp(0.05, 1)
orm = torch.stack([torch.ones_like(rough), rough, torch.zeros_like(rough)], -1)
files["ORM"] = save(orm, f"T_{AID}_ORM.png")
cm = torch.zeros(res * res, 3, device=dev)
cm[idx, 0] = (~seen).float()
cm[idx, 1] = (nviews / 4).clamp(0, 1)
cm[idx, 2] = 0.2
save(cm.view(res, res, 3), f"_coverage_{AID}.png")
T["total"] = round(time.time() - t0, 2)
rep = {"texels": NT, "seen_fraction": round(float(seen.float().mean()), 4), "views": stats,
       "gains": {k: [round(float(x), 3) for x in g] for k, g in gains.items()}, "files": files, "times_s": T}
json.dump(rep, open(out / "project.json", "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("texels", "seen_fraction", "times_s")}))
