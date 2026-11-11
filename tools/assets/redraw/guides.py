"""Guide passes for an image model from a scene's engine depth (GPU, torch): run at the image station.

    python -I tools/assets/redraw/guides.py [--set LIBRARY_SET_DIR] SCENE_OUT_DIR [SCENE_OUT_DIR ...]

The scene tool (tools/assets/scenes, spec "guides": true) writes per still the camera's linear depth (UE cm, .exr)
and each person's own depth (only that person drawn). From those, per still, into <scene>/guides/:
  <cam>_depth.png      disparity (near bright), 1st..99th percentile stretched: the ControlNet depth picture
  <cam>_normal.png     camera-space surface normals from the depth (R right, G up, B toward the camera)
  <cam>_lines.png      outlines: depth jumps, creases (normal turns) and every person's silhouette, white on black
  <cam>_mask_<key>.png the person's visible part (where their own depth meets the scene's)
  <cam>_mask_people.png all people, grown a few pixels (a per-pixel strength mask)
  <cam>_pose.png       the people's own bones as an OpenPose picture (left/right by colour; face points only where
                       the head shows them), when the scene recorded joints
and <scene>/truth.json: per still, per person the visible box (0-1, x0 y0 x1 y1), area, the fraction of the
person that is visible, with the scene's angles (head and body to the camera) and camera distance copied from
result.json. With --set (the scene's library set): the set's things (setfacts.py) seen in each still, each with its
visible box, the visible fraction of its sample points (occlusion from the depth) and whether it is lit or a fire.
Device: cuda (pin it with HIP_VISIBLE_DEVICES); it refuses to run on the CPU.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import framing  # noqa: E402
import setfacts  # noqa: E402

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


def seen_things(things, cam: dict, d: torch.Tensor, n: int = 7) -> list:
    """The set's things in this still: sample points over each thing's box projected through the camera, kept where
    the engine depth says nothing nearer covers them."""
    h, w = d.shape
    out = []
    for t in things:
        c, hf = t["center"], t["half"]
        pts = [[c[0] + hf[0] * (2 * i / (n - 1) - 1), c[1] + hf[1] * (2 * j / (n - 1) - 1), c[2] + hf[2] * (2 * k / (n - 1) - 1)]
               for i in range(n) for j in range(n) for k in range(n)]
        pr = [framing.project(cam, q, (w, h)) for q in pts]
        inside = [q for q in pr if q and 0 <= q[0] < 1 and 0 <= q[1] < 1]
        if not inside:
            continue
        uv = torch.tensor([[q[0] * w, q[1] * h, q[2] * 100.0] for q in inside], device=d.device)
        dm = d[uv[:, 1].long().clamp(0, h - 1), uv[:, 0].long().clamp(0, w - 1)]
        vis = dm >= uv[:, 2] - torch.clamp(uv[:, 2] * 0.04, min=6.0)
        nv = int(vis.sum())
        if nv < 3:
            continue
        v = uv[vis]
        box = [round(float(v[:, 0].min()) / w, 3), round(float(v[:, 1].min()) / h, 3),
               round(float(v[:, 0].max()) / w, 3), round(float(v[:, 1].max()) / h, 3)]
        flame = None
        if t.get("light_pos"):                     # is the light's own source (flame, grate) in view and uncovered
            q = framing.project(cam, t["light_pos"], (w, h))
            if q and 0 <= q[0] < 1 and 0 <= q[1] < 1:
                dd = float(d[min(h - 1, int(q[1] * h)), min(w - 1, int(q[0] * w))])
                flame = dd >= q[2] * 100.0 - max(q[2] * 100.0 * 0.06, 10.0)
            else:
                flame = False
        out.append({"name": t["name"], "kind": t["kind"], "words": t["words"], "lit": t["lit"], "fire": t["fire"],
                    "flame_in_view": flame,
                    "box": box, "visible_frac": round(nv / len(pts), 3), **({"view": t["view"]} if t.get("view") else {})})
    return out


# OpenPose 18-point body (COCO order) and its usual colours: the ControlNet Union reads left and right from them
OP_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
             [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
             [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
OP_LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
            [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]


def op_points(m: dict, cam: dict) -> list:
    """18 set-frame points (None where the camera cannot see them) from a person's measured joints: the face points
    (nose, eyes) only when the head faces the camera enough to show them, each ear unless the head hides it."""
    j = m.get("joints") or {}
    need = ("upperarm_l", "upperarm_r", "head", "FACIAL_L_Eye", "FACIAL_R_Eye")
    if not all(n in j for n in need):
        return []
    mid = lambda a, b: [(a[i] + b[i]) / 2 for i in range(3)]
    f = m["head_fwd_xy"]
    right = [f[1], -f[0], 0.0]                          # the person's right, in plan
    eye_l, eye_r = j["FACIAL_L_Eye"], j["FACIAL_R_Eye"]
    em = mid(eye_l, eye_r)
    nose = [em[0] + f[0] * 0.07, em[1] + f[1] * 0.07, em[2] - 0.04]
    hd = j["head"]
    ear_r = [hd[0] + right[0] * 0.075, hd[1] + right[1] * 0.075, em[2] - 0.01]
    ear_l = [hd[0] - right[0] * 0.075, hd[1] - right[1] * 0.075, em[2] - 0.01]
    to_cam = [cam["position"][i] - hd[i] for i in range(2)]
    n = math.hypot(*to_cam) or 1.0
    to_cam = [to_cam[0] / n, to_cam[1] / n]
    facing = f[0] * to_cam[0] + f[1] * to_cam[1]        # 1 = facing the camera, -1 = turned away
    side_r = right[0] * to_cam[0] + right[1] * to_cam[1]
    show_face = facing > -0.17                          # nose and eyes: up to ~100 deg off the camera
    pts = [nose if show_face else None, mid(j["upperarm_l"], j["upperarm_r"]),
           j["upperarm_r"], j.get("lowerarm_r"), j.get("hand_r"), j["upperarm_l"], j.get("lowerarm_l"), j.get("hand_l"),
           j.get("thigh_r"), j.get("calf_r"), j.get("foot_r"), j.get("thigh_l"), j.get("calf_l"), j.get("foot_l"),
           eye_r if show_face and (facing > 0.5 or side_r > 0) else None,
           eye_l if show_face and (facing > 0.5 or side_r < 0) else None,
           ear_r if side_r > -0.6 else None, ear_l if side_r < 0.6 else None]
    return pts


def pose_picture(res: dict, cam: dict, w: int, h: int, path: Path):
    """The engine's own bones as an OpenPose picture (black ground, coloured limbs, left and right by colour)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(im)
    n = 0
    for key, sub in res.get("subjects", {}).items():
        pts = op_points(sub.get("measured") or {}, cam)
        if not pts:
            continue
        uv = []
        for q in pts:
            pr = framing.project(cam, q, (w, h)) if q else None
            uv.append((pr[0] * w, pr[1] * h) if pr else None)
        sw = max(3, int(w / 1152 * 5))
        for i, (a, b) in enumerate(OP_LIMBS):
            pa, pb = uv[a - 1], uv[b - 1]
            if pa and pb:
                d.line([pa, pb], fill=tuple(int(c * 0.6) for c in OP_COLORS[i]), width=sw * 2)
        for i, q in enumerate(uv):
            if q:
                d.ellipse([q[0] - sw, q[1] - sw, q[0] + sw, q[1] + sw], fill=tuple(OP_COLORS[i]))
        n += 1
    im.save(path)
    return n


def one_scene(sd: Path, dev, things=None, day=None):
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
        if people:                       # all people at once, grown a little: the per-pixel strength mask
            for name, keep in (("people", lambda v: True),
                               ("front", lambda v: (v.get("body_to_camera_deg") or 0) <= 100)):
                allp = torch.zeros_like(d, dtype=torch.bool)
                for key, v in people.items():
                    if keep(v):
                        allp |= torch.from_numpy(cv2.imread(str(g / f"{cam}_mask_{key}.png"), 0) > 127).to(dev)
                allp = F.max_pool2d(allp.float()[None, None], 9, 1, 4)[0, 0] > 0.5
                save(allp.float() * 255, g / f"{cam}_mask_{name}.png")
            # three levels for a per-pixel strength (qwen_image reads the grey as a blend of its two strengths):
            # people facing within 100 deg white, the room mid grey, people turned past 100 deg black (they keep the
            # guide's body: at higher strength the model swung a turned-away body to profile, 11-11 round 1)
            tier = torch.full_like(d, 128.0)
            for val, keep in ((255.0, lambda v: (v.get("body_to_camera_deg") or 0) <= 100),
                              (0.0, lambda v: (v.get("body_to_camera_deg") or 0) > 100)):
                for key, v in people.items():
                    if keep(v):
                        m = torch.from_numpy(cv2.imread(str(g / f"{cam}_mask_{key}.png"), 0) > 127).to(dev)
                        m = F.max_pool2d(m.float()[None, None], 9, 1, 4)[0, 0] > 0.5
                        tier[m] = val
            save(tier, g / f"{cam}_mask_tiered.png")
        lines = F.max_pool2d(lines.float()[None, None], 3, 1, 1)[0, 0] > 0.5     # 3 px lines
        save(lines.float() * 255, g / f"{cam}_lines.png")
        truth["shots"][cam] = {"still": sh["path"], "kind": sh.get("kind"), "camera": sh["camera"], "people": people,
                               "speaker": sh.get("speaker"), "reject": sh.get("reject"),
                               "depth_range_cm": [round(float(d[valid].min()), 1), round(float(d[valid].max()), 1)]}
        if pose_picture(res, sh["camera"], w, h, g / f"{cam}_pose.png"):
            truth["shots"][cam]["pose"] = str(g / f"{cam}_pose.png")
        if things is not None:
            truth["shots"][cam]["things"] = seen_things(things, sh["camera"], d)
            truth["shots"][cam]["daylight"] = day
    (sd / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: guides run on a card (HIP_VISIBLE_DEVICES), never on the CPU")
    dev = torch.device("cuda")
    args = sys.argv[1:]
    things = day = None
    if args and args[0] == "--set":
        things, day = setfacts.things(args[1]), setfacts.daylight(args[1])
        args = args[2:]
    for a in args:
        t = one_scene(Path(a), dev, things, day)
        for cam, s in t["shots"].items():
            print(cam, [(x["name"], x["visible_frac"]) for x in s.get("things", [])], s["depth_range_cm"], {k: (v["box"], v["area"], v.get("head_to_camera_deg")) for k, v in s["people"].items()},
                  flush=True)


if __name__ == "__main__":
    main()
