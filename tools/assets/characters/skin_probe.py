"""Skin through the clothes, measured on flat-lit studio stills (char_render.py shots with "probe_face": true).

Each probed shot has a normal still and the same still with the face mesh left out of the main pass. The pixels that
differ are where the head mesh is seen. Counted in the shoulder region: below the base of the neck (neck_01 plus
NECK_DROP of the shoulder span) down to the chest (spine_04), across the shoulder span plus a quarter each side.
A pixel counts when it is seen head mesh AND skin-coloured (the face's own chroma, from the cheeks of the same still,
which a flat white light leaves as it is). ok when every shot's count is under MAX_FRAC of the region.

    python skin_probe.py RESULT.json [--max-frac 0.002] [--out PROBE.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

NECK_DROP = 0.12
MAX_FRAC = 0.002


def probe_shot(sh, max_frac=MAX_FRAC):
    im = np.asarray(Image.open(sh["path"]).convert("RGB")).astype(np.float32) / 255
    nf = np.asarray(Image.open(sh["noface_path"]).convert("RGB")).astype(np.float32) / 255
    J = sh.get("joints_px") or {}
    H, W = im.shape[:2]
    seen = np.abs(im - nf).max(2) > 0.08
    k = seen.astype(np.uint8)
    nb = sum(np.roll(np.roll(k, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    seen &= nb >= 6                                 # edge shimmer between the two stills is not a surface
    if not all(b in J for b in ("neck_01", "upperarm_l", "upperarm_r", "spine_04")):
        return {"name": sh["name"], "ok": None, "why": "joints not in view"}
    span = max(abs(J["upperarm_l"][0] - J["upperarm_r"][0]), 0.25 * abs(J["spine_04"][1] - J["neck_01"][1]), 8.0)
    x0 = min(J["upperarm_l"][0], J["upperarm_r"][0]) - 0.25 * span
    x1 = max(J["upperarm_l"][0], J["upperarm_r"][0]) + 0.25 * span
    y0 = J["neck_01"][1] + NECK_DROP * span
    y1 = J["spine_04"][1]
    reg = np.zeros((H, W), bool)
    reg[int(max(y0, 0)):int(min(max(y1, y0 + 1), H)), int(max(x0, 0)):int(min(x1, W))] = True
    # the face's own chroma: head-mesh pixels above the neck that are not dark (hair) - cheeks, forehead, chin
    face = seen.copy()
    face[int(max(J["neck_01"][1] - 0.2 * span, 0)):] = False
    fp = im[face]
    fp = fp[fp.mean(1) > 0.15]
    out = {"name": sh["name"], "region_px": int(reg.sum()), "head_mesh_px": int((seen & reg).sum())}
    if len(fp) < 30 or reg.sum() < 200:
        out.update(ok=None, why="face or shoulders not in view")
        return out
    rg0 = float(np.median(fp[:, 0] / np.maximum(fp[:, 1], 1e-3)))
    bg0 = float(np.median(fp[:, 2] / np.maximum(fp[:, 1], 1e-3)))
    rg = im[..., 0] / np.maximum(im[..., 1], 1e-3)
    bg = im[..., 2] / np.maximum(im[..., 1], 1e-3)
    skin = (np.abs(rg / rg0 - 1) < 0.25) & (np.abs(bg - bg0) < 0.15) & (im.mean(2) > 0.08)
    n = int((seen & reg & skin).sum())
    out.update(skin_px=n, skin_frac=round(n / reg.sum(), 5), ok=bool(n / reg.sum() < max_frac))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--max-frac", type=float, default=MAX_FRAC)
    ap.add_argument("--out")
    ap.add_argument("--prefix", default="", help="only shots whose name starts with this")
    a = ap.parse_args()
    r = json.loads(Path(a.result).read_text())
    shots = [s for s in r["shots"] if s.get("noface_path") and s["name"].startswith(a.prefix)]
    rows = [probe_shot(s, a.max_frac) for s in shots]
    res = {"ok": bool(rows) and all(x.get("ok") is not False for x in rows), "max_frac": a.max_frac, "shots": rows}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
