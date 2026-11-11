"""Cameras and depth for a whole orbit (hundreds of frames) in one world with WorldMirror (HY-World 2.0), on one card.

A 24 GB card holds about 40 frames per forward pass at 672 px, so the orbit is cut into chunks. Every chunk carries
the same anchor frames (spread evenly round the orbit, the first frame of the list first: WorldMirror puts its world
on its first frame), and each chunk is mapped onto the reference chunk by a similarity transform fitted to the
anchors' camera centres. The anchors' spread after alignment is the chunk-to-chunk consistency.

    HIP_VISIBLE_DEVICES=<card> python -I recon_chunks.py LIST.txt OUT [--anchors 12] [--chunk 40] [--size 672]

LIST.txt: one frame path per line, in orbit order (the first one is the world's origin frame).
OUT/pred.npz: depth [S,H,W], conf, w2c [S,3,4] (OpenCV), K [S,3,3] (processed size), img [S,H,W,3] uint8, paths,
pp (preprocessing transform: processed -> source pixels). OUT/chunks.json: chunk lists, scales, anchor residuals.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("list")
ap.add_argument("out")
ap.add_argument("--anchors", type=int, default=12)
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--size", type=int, default=672)
a = ap.parse_args()

REPO = Path(os.environ.get("WORLD_REPO", Path.home() / "src" / "HY-World-2.0"))
WM_DIR = Path(os.environ.get("WORLD_MODELS", Path.home() / "models" / "hy-world")) / "HY-World-2.0"
sys.path.insert(0, str(REPO))
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline  # noqa: E402
from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (  # noqa: E402
    compute_adaptive_target_size, compute_preprocessing_transform, vector_to_camera_matrices)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
paths = [ln.strip() for ln in Path(a.list).read_text().splitlines() if ln.strip()]
N = len(paths)
anchors = sorted(set(np.linspace(0, N, a.anchors, endpoint=False).round().astype(int).tolist()))
rest = [i for i in range(N) if i not in anchors]
per = a.chunk - len(anchors)
nch = int(np.ceil(len(rest) / per))
bounds = np.linspace(0, len(rest), nch + 1).round().astype(int)
chunks = [anchors + rest[bounds[c]:bounds[c + 1]] for c in range(nch)]

t0 = time.time()
pipe = WorldMirrorPipeline.from_pretrained(str(WM_DIR), subfolder="HY-WorldMirror-2.0", enable_bf16=True,
                                           disable_heads=["normal", "points", "gs"])
t_load = time.time() - t0
eff = compute_adaptive_target_size(paths[:1], a.size)
pp = compute_preprocessing_transform(paths[:1], eff)
res = []
t_inf_all = 0.0
for c, idx in enumerate(chunks):
    with torch.no_grad():
        p, imgs, t_inf = pipe._run_inference([paths[i] for i in idx], eff, None, None)
    t_inf_all += t_inf
    S, H, W = p["depth"].shape[1:4]
    e3x4, intr = vector_to_camera_matrices(p["camera_params"], image_hw=(H, W))
    r = {"idx": idx, "depth": p["depth"][0, ..., 0].float().cpu().numpy(),
         "img": (imgs[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8),
         "w2c": e3x4[0].float().cpu().numpy(), "K": intr[0].float().cpu().numpy()}
    r["conf"] = p["depth_conf"][0].float().cpu().numpy() if "depth_conf" in p else np.ones_like(r["depth"])
    res.append(r)
    del p, imgs
    torch.cuda.empty_cache()
    print(f"chunk {c + 1}/{nch}: {len(idx)} frames, infer {t_inf:.1f}s, peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
peak = torch.cuda.max_memory_allocated() / 2**30
del pipe
torch.cuda.empty_cache()
t_gpu = time.time() - t0


def c2w_of(w2c):
    E = np.tile(np.eye(4), (len(w2c), 1, 1))
    E[:, :3, :4] = w2c
    return np.linalg.inv(E)


def umeyama(src, dst):
    """s, R, t with dst ~ s R src + t."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    Sg = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        Sg[2, 2] = -1
    R = U @ Sg @ Vt
    s = np.trace(np.diag(D) @ Sg) / xs.var(0).sum()
    return s, R, md - s * R @ ms


na = len(anchors)
ref = c2w_of(res[0]["w2c"][:na])
radius = np.linalg.norm(ref[:, :3, 3] - ref[:, :3, 3].mean(0), axis=1).mean()
aligned_anchor = []
info = {"anchors": anchors, "chunks": [], "radius": float(radius)}
for c, r in enumerate(res):
    cw = c2w_of(r["w2c"])
    s, R, t = umeyama(cw[:na, :3, 3], ref[:, :3, 3])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    cw2 = T @ cw
    cw2[:, :3, :3] /= s                                   # rotation part back to orthonormal; centre carries s
    r["c2w"] = cw2
    r["depth"] = r["depth"] * s
    dc = np.linalg.norm(cw2[:na, :3, 3] - ref[:, :3, 3], axis=1) / radius
    dr = [np.degrees(np.arccos(np.clip((np.trace(cw2[k, :3, :3].T @ ref[k, :3, :3]) - 1) / 2, -1, 1)))
          for k in range(na)]
    info["chunks"].append({"frames": len(r["idx"]), "scale": float(s), "anchor_centre_err_frac_radius":
                           [round(float(x), 4) for x in dc], "anchor_rot_err_deg": [round(float(x), 2) for x in dr]})
    aligned_anchor.append(cw2[:na])

# anchors: mean over chunks (centres averaged, rotations by SVD of the summed matrices)
aa = np.stack(aligned_anchor)                             # C,na,4,4
anc = np.tile(np.eye(4), (na, 1, 1))
anc[:, :3, 3] = aa[:, :, :3, 3].mean(0)
for k in range(na):
    U, _, Vt = np.linalg.svd(aa[:, k, :3, :3].sum(0))
    anc[k, :3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
_, H, W = res[0]["depth"].shape
D = np.zeros((N, H, W), np.float32)
Cf = np.zeros((N, H, W), np.float32)
IM = np.zeros((N, H, W, 3), np.uint8)
C2W = np.zeros((N, 4, 4))
KK = np.zeros((N, 3, 3))
for c, r in enumerate(res):
    for j, i in enumerate(r["idx"]):
        if j < na:
            if c > 0:
                continue
            C2W[i] = anc[j]
            # anchor depth/K: the reference chunk's (all chunks' depths agree up to the fitted scale)
        else:
            C2W[i] = r["c2w"][j]
        D[i], Cf[i], IM[i], KK[i] = r["depth"][j], r["conf"][j], r["img"][j], r["K"][j]
W2C = np.linalg.inv(C2W)[:, :3, :4]
np.savez(out / "pred.npz", depth=D, conf=Cf, img=IM, w2c=W2C.astype(np.float32), K=KK.astype(np.float32),
         paths=np.array(paths), pp=json.dumps(pp))
info.update({"frames": N, "size": [W, H], "load_s": round(t_load, 1), "infer_s": round(t_inf_all, 1),
             "gpu_total_s": round(t_gpu, 1), "peak_gib": round(peak, 1), "pp": pp})
(out / "chunks.json").write_text(json.dumps(info, indent=1))
print(json.dumps({k: info[k] for k in ("frames", "load_s", "infer_s", "gpu_total_s", "peak_gib")}))
for c, ch in enumerate(info["chunks"]):
    print(f"chunk {c}: scale {ch['scale']:.3f} anchor centre err (x radius) max "
          f"{max(ch['anchor_centre_err_frac_radius']):.3f}, rot err max {max(ch['anchor_rot_err_deg']):.2f} deg")
