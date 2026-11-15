"""Tileable thread-scale cloth maps from a weave's interlacement matrix, computed on a GPU (torch, ROCm).

    python tools/assets/characters/weave_maps.py OUT_DIR [--px 1024] [--card 05] [--kinds plain,twill22,satin5,nap]

For each kind writes OUT_DIR/T_weave_<kind>_N.png (tangent-space normal, OpenGL green flipped to DirectX for UE) and
OUT_DIR/T_weave_<kind>_M.png (R height, G ambient occlusion, B roughness factor around 0.5 = no change). A tile holds
a whole number of yarn repeats, so it tiles; how many threads a tile carries is written to OUT_DIR/weave.json with
the tile's intended size in centimetres (threads / threads_per_cm), which the engine material uses to scale UVs in
real units.

Kinds: plain (linen: one over one under, uneven yarn width), twill22 (worsted: 2/2 twill, 45 degree wale), satin5
(silk: 5-end warp-faced satin, long floats, nearly smooth), nap (broadcloth: fulled and napped wool, no visible
weave, fine fibre noise along the nap).

The card is taken with a non-blocking flock on /tmp/gpu-<card>.lock, polled; HIP_VISIBLE_DEVICES comes from
`hip-index <card>`. Refuses to run on the CPU. Needs torch with ROCm and Pillow (CHAR_TORCH_PY names an interpreter
that has them; the script re-executes itself with it when the current one lacks torch).
"""

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time

KINDS = {
    # threads per tile, threads per cm (coarsened toward what a close-up resolves), yarn fill, crimp, fibre
    "plain": {"tile_threads": 64, "per_cm": 22.0, "fill": 0.86, "crimp": 0.35, "fibre": 0.10, "jitter": 0.12,
              "rough_crown": 0.95},
    "twill22": {"tile_threads": 96, "per_cm": 28.0, "fill": 0.92, "crimp": 0.28, "fibre": 0.08, "jitter": 0.06,
                "rough_crown": 0.9},
    "satin5": {"tile_threads": 100, "per_cm": 60.0, "fill": 0.98, "crimp": 0.12, "fibre": 0.02, "jitter": 0.03,
               "rough_crown": 0.8},
    "nap": {"tile_threads": 64, "per_cm": 16.0, "fill": 1.0, "crimp": 0.0, "fibre": 0.6, "jitter": 0.0,
            "rough_crown": 1.0},
}


def interlace(kind, i, j):
    """1 where the warp (vertical yarn, index i) lies over the weft (horizontal, index j)."""
    import torch
    if kind == "plain":
        return ((i + j) % 2 == 0).float()
    if kind == "twill22":
        return (((i + j) % 4) < 2).float()
    if kind == "satin5":
        return ((2 * i - j) % 5 != 0).float()            # warp floats over four, under one
    return torch.ones_like(i, dtype=torch.float32)


def take_card(card):
    fd = os.open(f"/tmp/gpu-{card}.lock", os.O_RDWR | os.O_CREAT, 0o666)
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd, round(time.time() - t0, 1)
        except BlockingIOError:
            time.sleep(2)


def periodic_noise(torch, n, px, sx, sy, dev, seed):
    """Tileable noise: white noise filtered in the frequency domain with an anisotropic Gaussian (sx, sy in px)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn((n, px, px), generator=g, device=dev)
    f = torch.fft.fftfreq(px, device=dev)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    k = torch.exp(-2 * (math.pi ** 2) * ((fx * sx) ** 2 + (fy * sy) ** 2))
    out = torch.fft.ifft2(torch.fft.fft2(w) * k).real
    return out / (out.std(dim=(-1, -2), keepdim=True) + 1e-6)


def make(kind, px, dev, seed=7):
    import torch
    p = KINDS[kind]
    T = p["tile_threads"]
    s = (torch.arange(px, device=dev, dtype=torch.float32) + 0.5) / px * T
    v, u = torch.meshgrid(s, s, indexing="ij")          # u across (warp index), v down (weft index)
    i, j = torch.floor(u).long(), torch.floor(v).long()
    fu, fv = u - i, v - j
    g = torch.Generator(device=dev).manual_seed(seed)
    wj = 1 + p["jitter"] * torch.randn(T, generator=g, device=dev)       # per-yarn width
    hj = 1 + p["jitter"] * torch.randn(T, generator=g, device=dev)
    if kind == "nap":
        # fulled wool: no yarns; fibres laid along the nap (down the garment), soft
        fib = periodic_noise(torch, 2, px, 0.8, 6.0, dev, seed)
        h = 0.5 + 0.18 * fib[0] + 0.08 * periodic_noise(torch, 1, px, 3.0, 3.0, dev, seed + 1)[0]
        rough = torch.ones_like(h)
    else:
        def profile(f, w):
            x = (2 * f - 1) / w.clamp(0.5, 1.2) / p["fill"]
            return torch.sqrt(torch.clamp(1 - x * x, min=0))

        # crimp: a warp yarn rises where it is over and dips where it is under, eased between crossings
        m_here = interlace(kind, i, j)
        m_next = interlace(kind, i, j + torch.where(fv > 0.5, 1, -1))
        e = torch.abs(fv - 0.5)
        warp_c = (m_here * (1 - e) + m_next * e) * 2 - 1
        m_next_w = interlace(kind, i + torch.where(fu > 0.5, 1, -1), j)
        e2 = torch.abs(fu - 0.5)
        weft_c = -((m_here * (1 - e2) + m_next_w * e2) * 2 - 1)
        warp = profile(fu, wj[i % T]) * hj[i % T] * 0.5 + p["crimp"] * 0.5 * warp_c
        weft = profile(fv, wj[j % T]) * hj[j % T] * 0.5 + p["crimp"] * 0.5 * weft_c
        warp = torch.where(profile(fu, wj[i % T]) > 0, warp, torch.full_like(warp, -1.0))
        weft = torch.where(profile(fv, wj[j % T]) > 0, weft, torch.full_like(weft, -1.0))
        h = torch.maximum(warp, weft)
        h = torch.clamp(h, min=-0.45)
        # fibre: streaks along each yarn's own direction
        fib = periodic_noise(torch, 2, px, 0.6, 4.0, dev, seed + 3)
        along = torch.where(warp >= weft, fib[0], fib[1].transpose(0, 1))
        h = h + p["fibre"] * 0.08 * along
        h = (h - h.min()) / (h.max() - h.min() + 1e-6)
        rough = torch.where(h > 0.6, torch.full_like(h, p["rough_crown"]), torch.ones_like(h))
    # occlusion: how far below its blurred neighbourhood each point sits (periodic blur via FFT)
    f = torch.fft.fftfreq(px, device=dev)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    rad = px / T * 0.8
    k = torch.exp(-2 * (math.pi ** 2) * ((fx * rad) ** 2 + (fy * rad) ** 2))
    hb = torch.fft.ifft2(torch.fft.fft2(h) * k).real
    ao = torch.clamp(1 - 2.2 * torch.clamp(hb - h, min=0), 0.25, 1.0)
    # normal from the periodic gradient; strength in "yarn heights per pixel"
    amp = 0.9 * T / px * 6.0
    dx = (torch.roll(h, -1, 1) - torch.roll(h, 1, 1)) * 0.5 * px / T * amp
    dy = (torch.roll(h, -1, 0) - torch.roll(h, 1, 0)) * 0.5 * px / T * amp
    n = torch.stack([-dx, dy, torch.ones_like(h)], 0)          # +y up in texture space (row 0 at the top)
    n = n / n.norm(dim=0, keepdim=True)
    n_dx = torch.stack([n[0], -n[1], n[2]], 0)                 # DirectX (UE) convention: green down
    return h, ao, rough * 0.5, n_dx, p


def to_png(path, chans):
    import numpy as np
    from PIL import Image
    a = (np.stack([c.clamp(0, 1).cpu().numpy() for c in chans], -1) * 255 + 0.5).astype("uint8")
    Image.fromarray(a, "RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--px", type=int, default=1024)
    ap.add_argument("--card", default="05")
    ap.add_argument("--kinds", default="plain,twill22,satin5,nap")
    a = ap.parse_args()
    try:
        import torch  # noqa: F401
    except ImportError:
        py = os.environ.get("CHAR_TORCH_PY")
        if not py or os.path.realpath(py) == os.path.realpath(sys.executable):
            raise SystemExit("no torch here; set CHAR_TORCH_PY to an interpreter with ROCm torch and Pillow")
        os.execv(py, [py] + sys.argv)
    if "HIP_VISIBLE_DEVICES" not in os.environ:
        idx = subprocess.check_output(["hip-index", a.card], text=True).strip()
        os.environ["HIP_VISIBLE_DEVICES"] = idx
        os.execv(sys.executable, [sys.executable] + sys.argv)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: refusing to compute weave maps on the CPU")
    fd, waited = take_card(a.card)
    t0 = time.time()
    try:
        dev = torch.device("cuda:0")
        os.makedirs(a.out, exist_ok=True)
        meta = {"device": torch.cuda.get_device_name(0), "card": a.card, "lock_wait_s": waited, "kinds": {}}
        for kind in a.kinds.split(","):
            h, ao, rough, n, p = make(kind, a.px, dev)
            to_png(os.path.join(a.out, f"T_weave_{kind}_N.png"), [n[0] * 0.5 + 0.5, n[1] * 0.5 + 0.5, n[2] * 0.5 + 0.5])
            to_png(os.path.join(a.out, f"T_weave_{kind}_M.png"), [h, ao, rough])
            meta["kinds"][kind] = {"tile_threads": p["tile_threads"], "per_cm": p["per_cm"],
                                   "tile_cm": round(p["tile_threads"] / p["per_cm"], 3)}
        torch.cuda.synchronize()
        meta["seconds"] = round(time.time() - t0, 2)
        with open(os.path.join(a.out, "weave.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(json.dumps(meta))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
