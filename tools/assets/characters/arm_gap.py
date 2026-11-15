"""Arm separation and hand visibility in a scene's stills, from the scene tool's guides (person mask + person depth).

    python arm_gap.py SCENE_OUT_DIR [--scene-tool DIR]   -> JSON: {camera: {person key: {side: ...}}}

Per camera, per person and side ("l", "r" forearm; "l_upper", "r_upper" upper arm): rows sampled along the bone,
mask_gap rows, separated rows (mask gap or depth jump) and, for the forearm, whether the palm is the visible surface. A row (elbow->wrist, 10 samples) is separated when the straight line from the forearm's centre to the
body's centre line at that height crosses background (mask) or a depth jump of >= 4 cm (the person's own depth)."""
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import argparse  # noqa: E402

framing = None


def read_exr(p):
    b = Path(p).read_bytes()
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
    chans, v, q = [], hdr["channels"][1], 0
    while v[q] != 0:
        e = v.index(b"\0", q)
        chans.append((v[q:e].decode(), struct.unpack_from("<i", v, e + 1)[0]))
        q = e + 17
    x0, y0, x1, y1 = struct.unpack("<4i", hdr["dataWindow"][1])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    offs = np.frombuffer(b, "<u8", h, i)
    dt = {1: np.float16, 2: np.float32, 0: np.uint32}
    out = {c: np.empty((h, w), np.float32) for c, _ in chans}
    for o in offs:
        y = struct.unpack_from("<i", b, int(o))[0] - y0
        q = int(o) + 8
        for c, t in chans:
            n = w * np.dtype(dt[t]).itemsize
            out[c][y] = np.frombuffer(b, dt[t], w, q)
            q += n
    return out["R"]


def measure(d, key):
    d = Path(d)
    r = json.loads((d / "result.json").read_text())
    J = r["subjects"][key]["measured"]["joints"]
    res = {}
    for s in r["shots"]:
        cam = s["camera"]
        m = np.array(Image.open(d / "guides" / f"{s['name']}_mask_{key}.png").convert("L")) > 127
        dep = read_exr(d / "guides" / f"{s['name']}_depth_{key}.exr")
        scene_dep = read_exr(d / "guides" / f"{s['name']}_depth.exr")
        H, W = m.shape

        def px(p):
            u, v, z = framing.project(cam, p)
            return np.array([u * W, v * H]), z

        out = {}
        mid = (np.array(J["thigh_l"]) + np.array(J["thigh_r"])) / 2
        neck = np.array(J["neck_01"])
        for side, seg in [(sd, sg) for sd in ("l", "r") for sg in ("fore", "upper")]:
            if seg == "fore":
                el, wr = np.array(J[f"lowerarm_{side}"]), np.array(J[f"hand_{side}"])
            else:
                el, wr = np.array(J[f"upperarm_{side}"]), np.array(J[f"lowerarm_{side}"])
            n_mask = n_sep = n = n_low = n_low_rows = 0
            for k_t, t in enumerate(np.linspace(0.3, 1.0, 10)):
                a3 = el + t * (wr - el)
                # the body's centre line at that height
                u = np.clip((a3[2] - mid[2]) / (neck[2] - mid[2]), 0, 1)
                c3 = mid + u * (neck - mid)
                c3[2] = a3[2]
                a2, _ = px(a3)
                c2, _ = px(c3)
                if not (0 <= a2[0] < W and 0 <= a2[1] < H):
                    continue
                n += 1
                L = int(np.linalg.norm(c2 - a2)) + 1
                xs = np.clip(np.linspace(a2[0], c2[0], L).astype(int), 0, W - 1)
                ys = np.clip(np.linspace(a2[1], c2[1], L).astype(int), 0, H - 1)
                mm = m[ys, xs]
                dd = dep[ys, xs]
                gap = (~mm[1:-1]).any() if L > 2 else False
                vis = mm & (np.abs(dd - scene_dep[ys, xs]) < 2.0)
                jump = (np.abs(np.diff(dd[vis])) > max(4.0, 0.015 * float(np.median(dd[vis])))).any() if vis.sum() > 2 else False
                n_mask += bool(gap)
                n_sep += bool(gap or jump)
                if k_t >= 5:                    # the lower half of the forearm, toward the wrist
                    n_low_rows += 1
                    n_low += bool(gap or jump)
            # hand: the palm 8 cm past the wrist, seen when the person's visible surface there is within 6 cm of it
            pal = wr + 0.08 * (wr - el) / np.linalg.norm(wr - el)
            p2, z = px(pal)
            hv = None
            if 0 <= p2[0] < W and 0 <= p2[1] < H:
                x_, y_ = int(p2[0]), int(p2[1])
                win = (slice(max(y_ - 3, 0), y_ + 4), slice(max(x_ - 3, 0), x_ + 4))
                seen = m[win] & (np.abs(dep[win] - scene_dep[win]) < 2.0)
                hv = bool((seen & (np.abs(dep[win] - z * 100) < 6.0)).any())
            key_ = side if seg == "fore" else f"{side}_upper"
            out[key_] = {"rows": n, "mask_gap": n_mask, "separated": n_sep, "lower_rows": n_low_rows,
                         "lower_separated": n_low, "hand_visible": hv if seg == "fore" else None}
        res[s["name"]] = out
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--scene-tool", default=str(Path(__file__).resolve().parents[1] / "scenes"))
    a = ap.parse_args()
    sys.path.insert(0, a.scene_tool)
    import framing as _f  # noqa: E402
    framing = _f
    r = json.loads((Path(a.scene_dir) / "result.json").read_text())
    out = {}
    for key in r["subjects"]:
        for cam, v in measure(a.scene_dir, key).items():
            out.setdefault(cam, {})[key] = v
    print(json.dumps(out))
