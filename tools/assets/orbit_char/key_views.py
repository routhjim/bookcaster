"""Key the subject out of studio frames (plain, smoothly shaded grey background) on a GPU, for the image-to-3D model
and the frame projection. torch only; run on one XTX under its lock (or the iGPU with a torch built for it).

    python key_views.py out_dir frame.png [frame.png ...] [--thresh 0.07] [--flip] [--crop-y1 N] [--neutral] [--rows] [--poly N]

Per frame: the background is modelled as a quadratic in x, y per channel, fitted (least squares) to the frame's
border pixels; a pixel is subject when its colour is far from that model; the background is then grown in from the
border (iterated dilation inside the not-subject mask) so dark shading inside the figure is never keyed out, and
the subject's holes are filled. Writes <name>.png (RGBA, straight alpha, slightly feathered), and keys.json with
each frame's subject bounding box [x0, y0, x1, y1] (pixels, top-left origin) and whether it touches the frame edge.
--flip mirrors every frame left-right first (frames of a clip made from a mirrored key).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def key(img, thresh, dev, neutral=False, rows=False, poly=0):
    a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).to(dev)        # H W 3
    h, w, _ = a.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=dev), torch.linspace(-1, 1, w, device=dev),
                            indexing="ij")
    basis = torch.stack([torch.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy], -1)  # H W 6
    b = 6
    border = torch.zeros(h, w, dtype=torch.bool, device=dev)
    border[:b] = True
    border[:, :b] = True
    border[:, -b:] = True
    fit = border.clone()          # the fit uses top and side edges only (the bottom holds feet or shoulders)
    border[-b:] = True
    keep = fit
    for _ in range(3):            # robust: refit without the edge pixels the model explains worst
        coef = torch.linalg.lstsq(basis[keep], a[keep]).solution                      # 6 x 3
        bg = basis @ coef
        d = (a - bg).abs().sum(-1)
        lim = torch.quantile(d[fit], 0.8) * 2 + 0.01
        keep = fit & (d < lim)
    if rows:                      # backdrop shaded by row (floor, horizon): each row lerps between its two edge medians
        e = max(8, w // 14)
        L = a[:, :e].median(1).values
        R = a[:, -e:].median(1).values
        L = F.avg_pool1d(L.T[None], 9, 1, 4, count_include_pad=False)[0].T
        R = F.avg_pool1d(R.T[None], 9, 1, 4, count_include_pad=False)[0].T
        t = torch.linspace(0, 1, w, device=dev)[None, :, None]
        bg = L[:, None, :] * (1 - t) + R[:, None, :] * t
        d = (a - bg).abs().sum(-1)
    if poly:                      # smooth 2D backdrop (vignette, floor): robust polynomial fit over the whole frame
        terms = [xx ** i * yy ** j for i in range(poly + 1) for j in range(poly + 1 - i)]
        P = torch.stack(terms, -1)
        mx, mn = a.max(-1).values, a.min(-1).values
        keep = ((mx - mn) < 0.08) & (mx > 0.2)
        for _ in range(5):
            sub = keep.nonzero()[::7]
            coef = torch.linalg.lstsq(P[sub[:, 0], sub[:, 1]], a[sub[:, 0], sub[:, 1]]).solution
            bg = P @ coef
            d = (a - bg).abs().sum(-1)
            keep = keep & (d < thresh * 0.7)
    far = d > thresh
    if neutral:                   # any mid grey of low saturation is background (a shaded backdrop the model misses)
        mx, mn = a.max(-1).values, a.min(-1).values
        far &= ~(((mx - mn) < 0.06) & (mx > 0.22) & (mx < 0.8))
    # background = connected to the border through not-far pixels
    m = border & ~far
    free = (~far).float()[None, None]
    m = m.float()[None, None]
    for _ in range(max(h, w)):
        m2 = F.max_pool2d(m, 3, 1, 1) * free
        if torch.equal(m2, m):
            break
        m = m2
    alpha = 1.0 - m[0, 0]
    # drop specks: open with a 7 px square
    alpha = F.max_pool2d(-F.max_pool2d(-alpha[None, None], 7, 1, 3), 7, 1, 3)[0, 0]
    alpha = F.avg_pool2d(alpha[None, None], 3, 1, 1)[0, 0]
    ys, xs = torch.nonzero(alpha > 0.5, as_tuple=True)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    rgba = torch.cat([a, alpha[..., None]], -1)
    out = (rgba.clamp(0, 1) * 255).round().byte().cpu().numpy()
    edges = {"top": bbox[1] <= 1, "bottom": bbox[3] >= h - 1, "left": bbox[0] <= 1, "right": bbox[2] >= w - 1}
    return out, bbox, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--thresh", type=float, default=0.07)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--neutral", action="store_true", help="also key out every mid, unsaturated grey (head frames)")
    ap.add_argument("--poly", type=int, default=0, help="robust whole-frame polynomial backdrop of this degree")
    ap.add_argument("--rows", action="store_true", help="backdrop model per row from the frame's side edges")
    ap.add_argument("--crop-y1", type=int, default=0, help="keep rows above this one only (head views: cut at the neck)")
    a = ap.parse_args()
    dev = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: refusing to key on the CPU")
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for p in a.frames:
        im = Image.open(p).convert("RGB")
        if a.flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if a.crop_y1:
            im = im.crop((0, 0, im.size[0], a.crop_y1))
        rgba, bbox, edges = key(im, a.thresh, dev, a.neutral, a.rows, a.poly)
        name = os.path.splitext(os.path.basename(p))[0]
        Image.fromarray(rgba, "RGBA").save(os.path.join(a.out, f"{name}.png"))
        res[name] = {"src": p, "flip": a.flip, "crop_y1": a.crop_y1, "bbox": bbox, "edges": edges, "size": list(im.size)}
    kp = os.path.join(a.out, "keys.json")
    old = json.load(open(kp)) if os.path.exists(kp) else {}
    old.update(res)
    json.dump(old, open(kp, "w"), indent=1)
    print(json.dumps({"keyed": len(res), "out": a.out}))


if __name__ == "__main__":
    main()
