"""Guide passes for an image model from a scene's engine depth (GPU, torch): run at the image station.

    python -I tools/assets/redraw/guides.py SCENE_OUT_DIR [SCENE_OUT_DIR ...]

The scene tool (tools/assets/scenes, spec "guides": true) writes per still the camera's linear depth (UE cm, .exr)
and each person's own depth (only that person drawn). From those, per still, into <scene>/guides/:
  <cam>_depth.png      disparity (near bright), 1st..99th percentile stretched: the ControlNet depth picture
  <cam>_normal.png     camera-space surface normals from the depth (R right, G up, B toward the camera)
  <cam>_lines.png      outlines: depth jumps, creases (normal turns) and every person's silhouette, white on black
  <cam>_mask_<key>.png the person's visible part (where their own depth meets the scene's)
and <scene>/truth.json: per still, per person the visible box (0-1, x0 y0 x1 y1), area, the fraction of the
person that is visible, with the scene's angles (head and body to the camera) and camera distance copied from
result.json. Device: cuda (pin it with HIP_VISIBLE_DEVICES); it refuses to run on the CPU.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F

FAR_CM = 1.0e5          # a person's own depth is cleared to the far value where they are not drawn
VIS_TOL = 0.015         # visible where the person's depth is within 1.5 % (or 2 cm) of the scene's
CREASE_DEG = 35.0
JUMP = 0.04             # a depth jump of 4 % between neighbours is an outline


def read_exr(p: Path) -> dict:
    """{channel: float32 array} of an UNCOMPRESSED scanline EXR (the scene tool writes them so; no OpenEXR here)."""
    import struct
    b = p.read_bytes()
    if b[:4] != b"\x76\x2f\x31\x01":
        raise RuntimeError(f"{p}: not an EXR file")
    i, hdr = 8, {}
    while b[i] != 0:
        j = b.index(b"\0", i)
        name = b[i:j].decode()
        k = b.index(b"\0", j + 1)
        typ = b[j + 1:k].decode()
        size = struct.unpack_from("<i", b, k + 1)[0]
        hdr[name] = (typ, b[k + 5:k + 5 + size])
        i = k + 5 + size
    i += 1
    chans, v = [], hdr["channels"][1]
    q = 0
    while v[q] != 0:
        e = v.index(b"\0", q)
        chans.append((v[q:e].decode(), struct.unpack_from("<i", v, e + 1)[0]))
        q = e + 17
    if hdr["compression"][1][0] != 0:
        raise RuntimeError(f"{p}: compressed EXR ({hdr['compression'][1][0]}); write it with compression_quality 1")
    x0, y0, x1, y1 = struct.unpack("<4i", hdr["dataWindow"][1])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    offs = np.frombuffer(b, "<u8", h, i)
    dt = {1: np.float16, 2: np.float32, 0: np.uint32}
    out = {c: np.empty((h, w), np.float32) for c, _ in chans}
    for row, o in enumerate(offs):
        y = struct.unpack_from("<i", b, int(o))[0] - y0
        q = int(o) + 8
        for c, t in chans:
            n = w * np.dtype(dt[t]).itemsize
            out[c][y] = np.frombuffer(b, dt[t], w, q)
            q += n
    return out


def load_depth(p: Path, dev) -> torch.Tensor:
    a = read_exr(p)
    return torch.from_numpy(a["R"]).to(dev).float()          # scene depth (cm) is in R


def save(t: torch.Tensor, p: Path):
    a = t.clamp(0, 255).round().byte().cpu().numpy()
    if a.ndim == 3:
        a = a[..., ::-1]                         # RGB -> BGR for OpenCV
    cv2.imwrite(str(p), np.ascontiguousarray(a))


def normals(d: torch.Tensor, focal_mm: float) -> torch.Tensor:
    h, w = d.shape
    fx = (w / 2) / math.tan(math.atan(18.0 / focal_mm))
    ys, xs = torch.meshgrid(torch.arange(h, device=d.device, dtype=torch.float32),
                            torch.arange(w, device=d.device, dtype=torch.float32), indexing="ij")
    X = (xs - w / 2 + 0.5) * d / fx
    Y = -(ys - h / 2 + 0.5) * d / fx
    P = torch.stack([X, Y, -d], -1)                                   # camera looks along -Z, Y up
    # central differences (one-sided at the border)
    gx = torch.zeros_like(P)
    gy = torch.zeros_like(P)
    gx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    gx[:, 0], gx[:, -1] = P[:, 1] - P[:, 0], P[:, -1] - P[:, -2]
    gy[1:-1] = P[:-2] - P[2:]
    gy[0], gy[-1] = P[0] - P[1], P[-2] - P[-1]
    n = torch.cross(gx, gy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return n


def edges_bool(m: torch.Tensor) -> torch.Tensor:
    """Boundary pixels of a boolean map."""
    f = m.float()[None, None]
    mx = F.max_pool2d(f, 3, 1, 1)
    mn = -F.max_pool2d(-f, 3, 1, 1)
    return (mx - mn)[0, 0] > 0


def one_scene(sd: Path, dev):
    res = json.loads((sd / "result.json").read_text())
    g = sd / "guides"
    truth = {"scene": sd.name, "shots": {}}
    for sh in res.get("shots", []):
        gd = sh.get("guides") or {}
        if not gd.get("depth"):
            continue
        cam = sh["name"]
        d = load_depth(Path(gd["depth"]), dev)
        h, w = d.shape
        valid = (d > 1) & (d < FAR_CM * 0.5)
        disp = torch.where(valid, 1.0 / d.clamp_min(1.0), torch.zeros_like(d))
        lo, hi = torch.quantile(disp[valid][::7], torch.tensor([0.01, 0.99], device=dev))
        dimg = ((disp - lo) / (hi - lo + 1e-9)).clamp(0, 1) * 255
        save(dimg, g / f"{cam}_depth.png")
        n = normals(d, sh["camera"]["focal_mm"])
        save(((n * 0.5 + 0.5) * 255), g / f"{cam}_normal.png")
        # outlines: depth jumps, creases, people's silhouettes
        r = d.clamp_min(1.0)
        jx = torch.zeros_like(d, dtype=torch.bool)
        jx[:, 1:] = (r[:, 1:] - r[:, :-1]).abs() / torch.minimum(r[:, 1:], r[:, :-1]) > JUMP
        jy = torch.zeros_like(jx)
        jy[1:] = (r[1:] - r[:-1]).abs() / torch.minimum(r[1:], r[:-1]) > JUMP
        cos = math.cos(math.radians(CREASE_DEG))
        cx = torch.zeros_like(jx)
        cx[:, 1:] = (n[:, 1:] * n[:, :-1]).sum(-1) < cos
        cy = torch.zeros_like(jx)
        cy[1:] = (n[1:] * n[:-1]).sum(-1) < cos
        lines = jx | jy | cx | cy
        people = {}
        for k, path in gd.items():
            if not k.startswith("depth_"):
                continue
            key = k[len("depth_"):]
            dp = load_depth(Path(path), dev)
            drawn = (dp > 1) & (dp < FAR_CM * 0.5)
            vis = drawn & ((dp - d).abs() <= torch.maximum(d * VIS_TOL, torch.full_like(d, 2.0)))
            save(vis.float() * 255, g / f"{cam}_mask_{key}.png")
            lines |= edges_bool(vis)
            nv, nd = int(vis.sum()), int(drawn.sum())
            box = None
            if nv:
                ys, xs = torch.nonzero(vis, as_tuple=True)
                box = [round(xs.min().item() / w, 4), round(ys.min().item() / h, 4),
                       round((xs.max().item() + 1) / w, 4), round((ys.max().item() + 1) / h, 4)]
            people[key] = {"box": box, "area": round(nv / (w * h), 5), "visible_frac": round(nv / nd, 3) if nd else 0.0,
                           **(sh.get("truth") or {}).get(key, {})}
        lines = F.max_pool2d(lines.float()[None, None], 3, 1, 1)[0, 0] > 0.5     # 3 px lines
        save(lines.float() * 255, g / f"{cam}_lines.png")
        truth["shots"][cam] = {"still": sh["path"], "kind": sh.get("kind"), "camera": sh["camera"], "people": people,
                               "depth_range_cm": [round(float(d[valid].min()), 1), round(float(d[valid].max()), 1)]}
    (sd / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: guides run on a card (HIP_VISIBLE_DEVICES), never on the CPU")
    dev = torch.device("cuda")
    for a in sys.argv[1:]:
        t = one_scene(Path(a), dev)
        for cam, s in t["shots"].items():
            print(cam, s["depth_range_cm"], {k: (v["box"], v["area"], v.get("head_to_camera_deg")) for k, v in s["people"].items()},
                  flush=True)


if __name__ == "__main__":
    main()
