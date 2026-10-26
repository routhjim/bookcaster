"""Image -> 3D mesh with Hunyuan3D-2.1's shape model (DiT flow matching + ShapeVAE), on one XTX under ROCm.

    HIP_VISIBLE_DEVICES=$(~/bin/hip-index 65) flock /tmp/gpu-65.lock \
        ~/tts-ab/.venv-hy3d/bin/python -I tools/blockout_shape.py in.png out.glb [--steps 30] [--octree 256]
        [--seed 0] [--faces 60000]
    ... blockout_shape.py --views front=f.png back=b.png [left=l.png right=r.png] out.glb [same flags]

in.png: one object on a plain or transparent background (tools/cutout.py output works; the background is
keyed out here by its corner colour if there is no alpha). Writes out.glb (untextured, Y-up as Hunyuan
emits it) and prints a JSON line with timings and VRAM.

Licence: Tencent Hunyuan 3D Community / non-commercial licence (excludes the EU, UK and South Korea).
Code: ~/src/Hunyuan3D-2.1 (hy3dshape, pure PyTorch: SDPA attention, skimage marching cubes, no CUDA
extensions needed for the shape stage). Weights: ~/models/hunyuan3d-2.1 (dit 7.4 GB fp16 + vae 0.7 GB).

--views (10-21): multi-view shape from Tencent's Hunyuan3D-2mv (~/models/hunyuan3d-2mv/hunyuan3d-dit-v2-mv,
4.9 GB fp16 = flux-style DiT 16+32 blocks + dinov2-giant conditioner + 2.0 VAE decoder, all in one file;
no separate dinov2 download). Its config names the 2.0 hy3dgen.shapegen.* classes; MV_REMAP points them at
the 2.1 equivalents and load_mv() checks keys itself (only the encode-side VAE encoder + pre_kl, absent from 2.0
releases, may be missing). Views: front/left/back/right, Tencent's order = front, front rotated clockwise 90, back, front
rotated clockwise 270; any subset that includes front works. Each view is keyed like in.png, then
recentred and scaled on its own (larger bbox side -> 85% of frame), so a side view whose height is not
its larger side would come out a different size than the front. Edric front+back: 21 s load, 49 s gen
(30 steps, octree 256), 5.4 GB peak on XTX 65; rounder, less spiky head than the 2.1 single-view mesh.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC = Path(os.environ.get("HY3D_SRC", Path.home() / "src" / "Hunyuan3D-2.1"))
WEIGHTS = Path(os.environ.get("HY3D_WEIGHTS", Path.home() / "models" / "hunyuan3d-2.1"))


def rgba(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode == "RGBA" and np.asarray(im)[..., 3].min() < 250:
        return im
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    bg = np.concatenate([a[:6, :6].reshape(-1, 3), a[:6, -6:].reshape(-1, 3)]).mean(0)
    d = np.abs(a - bg).sum(-1)
    alpha = np.clip((d - 18) * 12, 0, 255).astype(np.uint8)
    out = np.concatenate([a.astype(np.uint8), alpha[..., None]], -1)
    return Image.fromarray(out, "RGBA")


MV_WEIGHTS = Path(os.environ.get("HY3D_MV_WEIGHTS", Path.home() / "models" / "hunyuan3d-2mv"))
MV_REMAP = {  # 2.0 class path in the 2mv config.yaml -> 2.1 equivalent (10-21)
    "hy3dgen.shapegen.models.Hunyuan3DDiT": "hy3dshape.models.denoisers.hunyuan3ddit.Hunyuan3DDiT",
    "hy3dgen.shapegen.models.ShapeVAE": "hy3dshape.models.autoencoders.ShapeVAE",
    "hy3dgen.shapegen.models.SingleImageEncoder": "hy3dshape.models.conditioner.SingleImageEncoder",
    "hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler":
        "hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler",
    "hy3dgen.shapegen.preprocessors.MVImageProcessorV2": "hy3dshape.preprocessors.MVImageProcessorV2",
}
VIEWS = ("front", "left", "back", "right")


def load_mv(subfolder: str = "hunyuan3d-dit-v2-mv"):
    """2mv checkpoint into the 2.1 pipeline; strict except the VAE encoder + pre_kl (decoder-only 2.0 release)."""
    import yaml
    import safetensors.torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config
    d = MV_WEIGHTS / subfolder
    cfg = yaml.safe_load(open(d / "config.yaml"))
    for k in ("model", "vae", "conditioner", "scheduler", "image_processor"):
        cfg[k]["target"] = MV_REMAP[cfg[k]["target"]]
    ckpt: dict = {}
    if (d / "model.fp16.safetensors").exists():  # flat "model.x" keys; same tensors as the .ckpt
        for k, v in safetensors.torch.load_file(str(d / "model.fp16.safetensors"), device="cpu").items():
            mod, rest = k.split(".", 1)
            ckpt.setdefault(mod, {})[rest] = v
    else:
        ckpt = torch.load(d / "model.fp16.ckpt", map_location="cpu", weights_only=True)
    parts, report = {}, {}
    for k in ("model", "vae", "conditioner"):
        m = instantiate_from_config(cfg[k])
        r = m.load_state_dict(ckpt.pop(k), strict=False)
        bad_missing = [x for x in r.missing_keys  # encode-only VAE parts; decode() never touches them
                       if not (k == "vae" and x.split(".")[0] in ("encoder", "pre_kl"))]
        if bad_missing or r.unexpected_keys:
            raise RuntimeError(f"2mv {k}: missing {bad_missing[:8]} ({len(bad_missing)}), "
                               f"unexpected {r.unexpected_keys[:8]} ({len(r.unexpected_keys)})")
        report[k] = {"missing": len(r.missing_keys), "unexpected": 0}
        parts[k] = m
    if ckpt:
        raise RuntimeError(f"2mv: unused checkpoint groups {list(ckpt)}")
    pipe = Hunyuan3DDiTFlowMatchingPipeline(
        vae=parts["vae"], model=parts["model"], conditioner=parts["conditioner"],
        scheduler=instantiate_from_config(cfg["scheduler"]),
        image_processor=instantiate_from_config(cfg["image_processor"]), device="cuda", dtype=torch.float16)
    return pipe, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="in.png out.glb, or just out.glb with --views")
    ap.add_argument("--views", nargs="+", default=None, help="front=a.png back=b.png [left=.. right=..]")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--faces", type=int, default=80000)
    a = ap.parse_args()
    views = {}
    if a.views:  # nargs="+" swallows a trailing out.glb: move '='-less tokens back to the positionals
        a.paths += [v for v in a.views if "=" not in v]
        for v in (v for v in a.views if "=" in v):
            tag, path = v.split("=", 1)
            if tag not in VIEWS:
                ap.error(f"view tag {tag!r} not in {VIEWS}")
            views[tag] = path
        if "front" not in views or len(a.paths) != 1:
            ap.error("--views needs front=... and exactly one out.glb")
        a.image, a.out = None, a.paths[0]
    elif len(a.paths) == 2:
        a.image, a.out = a.paths
    else:
        ap.error("expected in.png out.glb")
    sys.path.insert(0, str(SRC / "hy3dshape"))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer

    t0 = time.time()
    keys = None
    if views:
        pipe, keys = load_mv()
        img = {t: rgba(p) for t, p in views.items()}
    else:
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(WEIGHTS), subfolder="hunyuan3d-dit-v2-1",
                                                                use_safetensors=False, device="cuda",
                                                                dtype=torch.float16)
        img = rgba(a.image)
    t1 = time.time()
    g = torch.Generator("cuda").manual_seed(a.seed)
    with torch.inference_mode():
        mesh = pipe(image=img, num_inference_steps=a.steps, octree_resolution=a.octree,
                    guidance_scale=a.guidance, generator=g, num_chunks=20000, output_type="trimesh")[0]
    t2 = time.time()
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=a.faces)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(a.out)
    extra = {"views": sorted(views, key=VIEWS.index), "keys": keys} if views else {}
    print(json.dumps({"out": a.out, **extra, "load_s": round(t1 - t0, 1), "gen_s": round(t2 - t1, 1),
                      "post_s": round(time.time() - t2, 1), "faces": int(len(mesh.faces)),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
