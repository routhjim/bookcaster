"""Skin through the clothes, measured on flat-lit studio stills (char_render.py shots with "probe_face": true).

Each probed shot has a normal still and the same still with the face mesh left out of the main pass. The pixels that
differ are where the head mesh is seen. Counted in the shoulder region: below the base of the neck (neck_01 plus
NECK_DROP of the shoulder span) down to the chest (spine_04), across the shoulder span plus a quarter each side,
less a column the neck's width round its axis (skin in an open collar is the neck, by design).
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
NECK_HALF = 0.2          # of the shoulder span: half the neck column left out
MAX_FRAC = 0.002
# a view fails only when its skin pixels also exceed this count: a percentage of a small region (a side view) let a
# handful of edge pixels decide. Set from data (11-12): the largest count on the known-clean views of the five
# passing characters was 15; the smallest on the known-bad control's failing views (the unmasked woman) was 99;
# 40 lies between them (about their geometric mean, 38.5)
MIN_PX = 40
NECK_JOIN = 0.1           # how far past the column (share of the span) skin joined to the neck counts as the opening
HUE_MIN = 0.08           # least chroma change (r/g plus b/g) for a pixel to count as another surface
DEBUG_DIR = None         # --debug DIR: each shot with its counted pixels in magenta, the region darkened


def probe_shot(sh, max_frac=MAX_FRAC, neck_half=None, open_neck=False):
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
    # the shoulders, not the neck: a column the neck's width round its axis is left out (the neck seen in an open
    # collar or a V of the coat is skin by design; a lifted collar standing off the neck showed it, 11-11)
    nx = J["neck_01"][0]
    nh = NECK_HALF if neck_half is None else neck_half
    reg[:, int(max(nx - nh * span, 0)):int(min(nx + nh * span, W))] = False
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
    # skin through the cloth is head mesh over cloth: where the still without the head shows skin there too (the
    # body's own skin in an open neckline, under the head's shadow) the head mesh hides no cloth (11-12)
    rgn = nf[..., 0] / np.maximum(nf[..., 1], 1e-3)
    bgn = nf[..., 2] / np.maximum(nf[..., 1], 1e-3)
    skin_nf = (np.abs(rgn / rg0 - 1) < 0.25) & (np.abs(bgn - bg0) < 0.15) & (nf.mean(2) > 0.08)
    # and a surface really changed: the colour itself differs, not only its light (the head's shadow and bounce on a
    # tan drape's folds changed their brightness, 11-12)
    hue_d = np.abs(rg - rgn) + np.abs(bg - bgn)
    seen = seen & (hue_d > HUE_MIN)
    if open_neck:
        # an open or base neckline: head-mesh skin joined to the neck's own skin above the region (the neck carried on
        # down into the opening, seen past the column in a turned view) is the opening, not skin through the cloth,
        # which shows as patches inside the cloth (11-12)
        # only near the opening (within NECK_JOIN of the span beyond the column): skin joined to the neck that spreads
        # over the shoulders is still skin through the cloth (the unmasked control's)
        near_ = np.zeros((H, W), bool)
        near_[:, int(max(nx - (nh + NECK_JOIN) * span, 0)):int(min(nx + (nh + NECK_JOIN) * span, W))] = True
        cnt = seen & skin & ~skin_nf & near_
        grown = cnt & (np.arange(H)[:, None] < int(max(y0, 0)))
        for _ in range(400):
            g2 = grown.copy()
            g2[1:] |= grown[:-1]
            g2[:-1] |= grown[1:]
            g2[:, 1:] |= grown[:, :-1]
            g2[:, :-1] |= grown[:, 1:]
            g2 &= cnt
            if (g2 == grown).all():
                break
            grown = g2
        out["joined_to_neck_px"] = int((grown & reg).sum())
        skin_nf = skin_nf | grown
    n = int((seen & reg & skin & ~skin_nf).sum())
    if DEBUG_DIR:
        ov = (im * 255).astype(np.uint8).copy()
        ov[reg & ~(seen & skin & ~skin_nf)] = (ov[reg & ~(seen & skin & ~skin_nf)] * 0.6).astype(np.uint8)
        ov[seen & reg & skin & ~skin_nf] = [255, 0, 255]
        Image.fromarray(ov).save(Path(DEBUG_DIR) / f"probe_{sh['name']}.png")
    out.update(skin_px=n, skin_frac=round(n / reg.sum(), 5), ok=bool(n / reg.sum() < max_frac or n <= MIN_PX))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--max-frac", type=float, default=MAX_FRAC)
    ap.add_argument("--out")
    ap.add_argument("--debug", default=None, help="write each shot's counted pixels over the still into this folder")
    ap.add_argument("--prefix", default="", help="only shots whose name starts with this")
    ap.add_argument("--neck-half", type=float, default=None,
                    help="half the left-out column as a share of the shoulder span: the garment's measured neck "
                         "opening (bl_combo neck_opening) plus a margin; used for a base or open neckline")
    ap.add_argument("--neckline", default=None, choices=["chin", "base", "open"],
                    help="the garment's neckline fact: at the base of the neck or lower, the left-out column is the "
                         "neckline's width (the collarbones in it are skin by design)")
    a = ap.parse_args()
    global DEBUG_DIR
    DEBUG_DIR = a.debug
    r = json.loads(Path(a.result).read_text())
    shots = [s for s in r["shots"] if s.get("noface_path") and s["name"].startswith(a.prefix)]
    # a neckline at the base of the neck or lower leaves the collarbones bare by design: the left-out column is that
    # neckline's width; the shoulders beyond it are still measured
    nh = {"base": 0.32, "open": 0.42}.get(a.neckline, NECK_HALF)
    if a.neck_half is not None and a.neckline in ("base", "open"):
        nh = a.neck_half          # the opening measured on this figure (plus a margin), not a fixed share
    rows = [probe_shot(s, a.max_frac, nh, a.neckline in ("base", "open")) for s in shots]
    res = {"ok": bool(rows) and all(x.get("ok") is not False for x in rows), "max_frac": a.max_frac, "shots": rows}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
