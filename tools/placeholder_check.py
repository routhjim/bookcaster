"""Untextured placeholder props in a finished frame (any book, CPU, ~50 ms per frame).

    python3 tools/placeholder_check.py FRAME.png [...]     -> JSON per frame: {"placeholder": bool, "area": .., "box": ..}

10-21 knight ch1 forge: props without an approved mesh were staged as plain boxes and survived the paint as flat
white/grey blocks (an anvil-shaped white slab in the foreground of 5 strike shots). A placeholder = a large region
that is at once flat (no texture: low local variance), unsaturated, bright, and box-like (fills most of its bounding
box). Painted things are textured; snow, sky and white dresses carry gradients and folds.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image

FLAT_STD = 4.0         # local luma std (7x7) below this = no texture
MAX_SAT = 0.10         # HSV saturation
MIN_LUMA = 150         # bright (the default blockout material is near-white)
MIN_AREA = 0.004       # of the frame (10-21 calibration: blocks 0.5-1.6 %)
MIN_FILL = 0.75       # area / bounding box: blocks 0.78-0.98; white fire 0.62, sky blobs lower


def _components(mask: np.ndarray):
    """4-connected components without scipy: (area, x0, y0, x1, y1) per component, via a union-find over runs."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nxt = 1
    for y in range(h):
        row = mask[y]
        for x in np.flatnonzero(row):
            up = lab[y - 1, x] if y else 0
            left = lab[y, x - 1] if x else 0
            if up and left:
                a, b = find(up), find(left)
                lab[y, x] = min(a, b)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            elif up or left:
                lab[y, x] = up or left
            else:
                parent.append(nxt)
                lab[y, x] = nxt
                nxt += 1
    roots = np.array([find(i) for i in range(len(parent))], np.int32)
    lab = roots[lab]
    out = []
    for r in np.unique(lab[lab > 0]):
        ys, xs = np.nonzero(lab == r)
        out.append((len(xs), int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return out


def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, float64 integral image (edges clamped)."""
    p = np.pad(x.astype(np.float64), r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k))[: x.shape[0], : x.shape[1]]


def _local_std(x: np.ndarray, r: int) -> np.ndarray:
    m = _box(x, r)
    return np.sqrt(np.maximum(_box(x * x, r) - m * m, 0))


def check(path: str, work_w: int = 384) -> dict:
    im = Image.open(path).convert("RGB")
    s = work_w / im.width
    im = im.resize((work_w, max(1, round(im.height * s))), Image.BILINEAR)
    a = np.asarray(im).astype(np.float32)
    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    std = _local_std(luma, 3)
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    mask = (std < FLAT_STD) & (sat < MAX_SAT) & (luma > MIN_LUMA)
    H, W = mask.shape
    best = {"placeholder": False, "area": 0.0, "fill": 0.0, "box": None}
    for area, x0, y0, x1, y1 in _components(mask):
        fa = area / (H * W)
        fill = area / max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        if fa > best["area"]:
            best.update(area=round(fa, 4), fill=round(fill, 2), box=[round(v / s) for v in (x0, y0, x1, y1)])
        if fa >= MIN_AREA and fill >= MIN_FILL:
            best.update(placeholder=True, area=round(fa, 4), fill=round(fill, 2),
                        box=[round(v / s) for v in (x0, y0, x1, y1)])
            break
    return best


if __name__ == "__main__":
    print(json.dumps({p: check(p) for p in sys.argv[1:]}, indent=1))
