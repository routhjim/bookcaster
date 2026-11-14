"""What each part of a generated figure IS: hair, face, bare skin, or a garment, from a trained human parser.

    python tools/assets/img3d/parse_figure.py FIGURE.glb OUT.npz [--sheet SHEET.jpg]

The figure fit (bl_combo.py) has to know which of the generated figure's surface is the person's own head, hair and
skin (the MetaHuman supplies those) and which is clothing (kept). Told by colour it broke per character: brown hair
passed as skin, a ponytail and a beard counted as a collar, a tan garment as skin. Here the figure is looked at the
way the field does it for clothed-human meshes (render the mesh from several sides, label each picture with a 2D
human parser, carry the labels back onto the surface by visibility, the views voting):

  views    front, back, left, right, straight on (orthographic), the textured surface splatted through a depth buffer;
  parser   fashn-ai/fashn-human-parser (SegFormer-B4, 18 classes: face, hair, top, dress, skirt, pants, belt, bag,
           hat, scarf, glasses, arms, hands, legs, feet, torso, jewelry), on the GPU;
  vote     a vertex takes each view's label where it is the nearest surface in that view, weighted by how squarely
           it faces the view; a vertex no view sees takes its nearest labelled neighbour's.

OUT.npz: "points" (N, 3: the glb's own vertex coordinates), "labels" (N: class ids), "names" (the class names),
"groups" (N: 0 other, 1 hair, 2 face, 3 bare skin, 4 garment), and "facts" (json: the shares, the views' agreement).
The fit reads it by nearest point. Runs under the image-to-3D interpreter (torch, trimesh, transformers)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ["background", "face", "hair", "top", "dress", "skirt", "pants", "belt", "bag", "hat", "scarf", "glasses",
         "arms", "hands", "legs", "feet", "torso", "jewelry"]
HAIR, FACE = 2, 1
SKIN = (12, 13, 14, 15, 16)                 # arms, hands, legs, feet, torso (bare)
GARMENT = (3, 4, 5, 6, 7, 10)
GROUP = np.zeros(len(NAMES), np.uint8)
GROUP[HAIR], GROUP[FACE] = 1, 2
GROUP[9] = GROUP[11] = 1                    # a hat and glasses go with the head (the MetaHuman's), like its hair
GROUP[list(SKIN)] = 3
GROUP[list(GARMENT)] = 4
PAL = np.array([[20, 20, 20], [240, 200, 80], [255, 40, 40], [60, 110, 255], [160, 80, 255], [200, 120, 255],
                [40, 200, 120], [255, 255, 255], [120, 120, 120], [255, 140, 0], [0, 255, 255], [255, 255, 0],
                [255, 170, 120], [255, 120, 170], [200, 160, 100], [150, 110, 70], [255, 225, 190], [255, 0, 255]],
               np.uint8)
RES = (768, 1152)                           # a view's picture, wide by high
BG = 0.72                                   # the views' plain light grey ground (the drawings')
PARSER_IN = (384, 576)                     # the parser's trained input, wide by high
N_SAMPLES = 4_000_000
# name: (degrees about the up axis, vote weight, the share of the height in the picture from the top down)
VIEWS = {"front": (0.0, 1.0, 1.0), "back": (180.0, 1.0, 1.0), "left": (90.0, 0.6, 1.0), "right": (-90.0, 0.6, 1.0),
         "front_head": (0.0, 2.0, 0.34), "back_head": (180.0, 2.0, 0.34), "left_head": (90.0, 1.2, 0.34),
         "right_head": (-90.0, 1.2, 0.34),
         # quarter views: a parser reads a head from straight behind poorly (hair as a dress or as ground); seen
         # from a back quarter the profile is in the picture and the hair reads as hair
         "fl_head": (45.0, 1.5, 0.34), "fr_head": (-45.0, 1.5, 0.34), "bl_head": (135.0, 2.0, 0.34),
         "br_head": (-135.0, 2.0, 0.34)}   # head and shoulders large: in a whole spread figure the head is 40 px
OCCLUDE = 0.03                              # hidden: this far (of the figure's height) behind the nearest surface
DRAW_IOU = 0.7                              # the drawing is used when its silhouette lies on the mesh's this well
DRAW_WEIGHT = 6.0                           # the drawing's vote for what the front sees
HEAD_PRIOR = 4.0
GAMMA = 1.2                                 # the baked colours are darker than the drawing they came from


def front_sign(v: np.ndarray, col: np.ndarray) -> float:
    """+1 when the figure faces the glb's +z, -1 when -z: the face is the warm side of the head's middle."""
    y = v[:, 1]
    hi = y > y.min() + 0.89 * np.ptp(y)
    mid = hi & (y < y.min() + 0.96 * np.ptp(y)) & (np.abs(v[:, 0] - np.median(v[:, 0])) < 0.05 * np.ptp(y))
    if mid.sum() < 50:
        return 1.0
    warm, zm = col[:, 0] - col[:, 2], np.median(v[mid, 2])
    a, b = warm[mid & (v[:, 2] > zm)], warm[mid & (v[:, 2] <= zm)]
    return 1.0 if (a.mean() if len(a) else 0) >= (b.mean() if len(b) else 0) else -1.0


def view_axes(deg: float, fs: float):
    """(right, toward-the-viewer) unit vectors in the glb's x-z plane for a view turned deg about the up axis."""
    a = np.radians(deg)
    toward = np.array([np.sin(a), 0.0, np.cos(a) * fs])
    toward[0] *= fs
    right = np.array([toward[2], 0.0, -toward[0]])
    return right, toward


def splat(torch, pts, cols, right, toward, lo, scale, dev):
    """The surface points drawn straight on through a depth buffer: (picture H x W x 3 floats, depth H x W,
    each point's pixel index, each point's depth)."""
    W, H = RES
    u = ((pts @ torch.tensor(right, device=dev, dtype=pts.dtype)) - lo[0]) * scale
    vv = (pts[:, 1] - lo[1]) * scale
    fx, fy = u + W / 2, H - 1 - vv - 0.04 * H
    ok = (fx >= 0) & (fx <= W - 1) & (fy >= 0) & (fy <= H - 1)       # (a zoomed view leaves most of the figure out)
    px = fx.round().long().clamp(0, W - 1)
    py = fy.round().long().clamp(0, H - 1)
    d = pts @ torch.tensor(toward, device=dev, dtype=pts.dtype)
    idx = py * W + px
    zb = torch.full((H * W,), -1e9, device=dev, dtype=pts.dtype)
    zb = zb.scatter_reduce(0, idx[ok], d[ok], reduce="amax", include_self=True)
    # a point is drawn when nothing within a pixel of it stands nearer the viewer (by its own pixel alone, the back
    # of the head showed through the gaps between the face's points)
    import torch.nn.functional as Fn
    zb = Fn.max_pool2d(zb.reshape(1, 1, H, W), 3, 1, 1).reshape(-1)
    near = (d >= zb[idx] - OCCLUDE) & ok
    img = torch.full((H * W, 3), BG, device=dev, dtype=pts.dtype)
    for dx_, dy_ in ((1, 1), (1, 0), (0, 1), (0, 0)):               # each point two pixels square, its own pixel last
        j = (py + dy_).clamp(0, H - 1) * W + (px + dx_).clamp(0, W - 1)
        img[j[near]] = cols[near]
    return img.reshape(H, W, 3), zb.reshape(H, W), idx, d


def fill_holes(torch, img, zb):
    """A splatted picture's pinholes closed (a 3 x 3 dilation of the drawn pixels over the undrawn ones, twice)."""
    import torch.nn.functional as Fn
    drawn = (zb > -1e8).float()[None, None]
    x = img.permute(2, 0, 1)[None]
    for _ in range(2):
        s = Fn.avg_pool2d(x * drawn, 3, 1, 1)
        c = Fn.avg_pool2d(drawn, 3, 1, 1)
        fill = (drawn < 0.5) & (c > 0.2)
        x = torch.where(fill, s / c.clamp(min=1e-6), x)
        drawn = torch.where(fill, torch.ones_like(drawn), drawn)
    return x[0].permute(1, 2, 0)


def kmeans(X, k, it=25):
    C = X[np.random.default_rng(0).choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
    return C


def from_drawing(lab_d: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Each vertex's pixel in the drawing the figure was made from (the drawing is the figure seen from the front:
    its figure's box laid on the mesh's own front box; the side the mesh's right is on is the one whose
    silhouettes overlap more). Returns (px, py, overlap of the two silhouettes 0..1)."""
    ys, xs = np.nonzero(lab_d > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    best = None
    for flip in (False, True):
        t = (u - u.min()) / max(1e-6, np.ptp(u))
        px = np.round(x0 + ((1 - t) if flip else t) * (x1 - x0)).astype(int).clip(0, lab_d.shape[1] - 1)
        py = np.round(y1 - (v - v.min()) / max(1e-6, np.ptp(v)) * (y1 - y0)).astype(int).clip(0, lab_d.shape[0] - 1)
        k = 4                                             # silhouettes compared on a coarse grid
        a = np.zeros((lab_d.shape[0] // k + 1, lab_d.shape[1] // k + 1), bool)
        a[py // k, px // k] = True
        b = np.zeros_like(a)
        b[ys // k, xs // k] = True
        iou = float((a & b).sum()) / max(1, int((a | b).sum()))
        if best is None or iou > best[2]:
            best = (px, py, iou)
    return best


def parse(glb: Path, out: Path, sheet: Path | None = None, drawing: Path | None = None) -> dict:
    import torch
    import trimesh
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation
    if not torch.cuda.is_available():
        raise SystemExit("parse_figure: no GPU (the parser never runs on the CPU)")
    dev = "cuda"
    m = trimesh.load(str(glb), force="mesh")
    V = np.asarray(m.vertices, np.float32)
    VC = np.asarray(m.visual.to_color().vertex_colors)[:, :3].astype(np.float32) / 255.0
    VN = np.asarray(m.vertex_normals, np.float32)
    Fc = np.asarray(m.faces)
    fs = front_sign(V, VC)
    # the surface, sampled densely with its colours (the vertices alone leave a splatted picture full of holes)
    rng = np.random.default_rng(0)
    area = m.area_faces
    fi = rng.choice(len(Fc), N_SAMPLES, p=area / area.sum())
    bw = rng.dirichlet([1, 1, 1], N_SAMPLES).astype(np.float32)
    P = (V[Fc[fi]] * bw[:, :, None]).sum(1)
    PC = (VC[Fc[fi]] * bw[:, :, None]).sum(1)
    pts = torch.tensor(np.vstack([P, V]), device=dev)
    cols = torch.tensor(np.vstack([PC, VC]), device=dev)
    Hf = float(np.ptp(V[:, 1]))
    cols = cols.clamp(0, 1) ** (1.0 / GAMMA)
    hx = float(np.median(V[V[:, 1] > V[:, 1].min() + 0.9 * Hf, 0]))          # the head's middle, across
    hz = float(np.median(V[V[:, 1] > V[:, 1].min() + 0.9 * Hf, 2]))
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]

    def prep(pic_):                          # the parser's own input: 384 x 576, ImageNet-normalised
        a_ = np.asarray(pic_.resize(PARSER_IN, Image.BICUBIC), np.float32) / 255.0
        return (torch.tensor(a_, device=dev).permute(2, 0, 1)[None] - mean) / std
    model = SegformerForSemanticSegmentation.from_pretrained("fashn-ai/fashn-human-parser").to(dev).eval()
    votes = torch.zeros((len(V), len(NAMES)), device=dev)
    tiles, shares = [], {}
    vn = torch.tensor(VN, device=dev)
    seen_front, lab_d, dpx, dpy, d_iou = None, None, None, None, None
    if drawing:
        dim = Image.open(drawing).convert("RGB")
        with torch.no_grad():
            lg = model(pixel_values=prep(dim)).logits
            lab_d = torch.nn.functional.interpolate(lg, size=dim.size[::-1], mode="bilinear",
                                                    align_corners=False).argmax(1)[0].cpu().numpy()
    for name, (deg, wt, part) in VIEWS.items():
        right, toward = view_axes(deg, fs)
        scale = RES[1] * 0.92 / (Hf * part)
        lo = (float(np.array([hx, 0.0, hz]) @ right), float(V[:, 1].min() + (1.0 - part) * Hf))
        img, zb, idx, d = splat(torch, pts, cols, right, toward, lo, scale, dev)
        img = fill_holes(torch, img, zb)
        pic = Image.fromarray((img.clamp(0, 1) * 255).byte().cpu().numpy())
        with torch.no_grad():
            lg = model(pixel_values=prep(pic)).logits
            lab = torch.nn.functional.interpolate(lg, size=(RES[1], RES[0]), mode="bilinear",
                                                  align_corners=False).argmax(1)[0]
        vi = idx[len(P):]                                        # the vertices' own pixels
        vd = d[len(P):]
        uu = ((torch.tensor(V, device=dev) @ torch.tensor(right, device=dev, dtype=pts.dtype)) - lo[0]) * scale + RES[0] / 2
        hh_ = RES[1] - 1 - (torch.tensor(V[:, 1], device=dev) - lo[1]) * scale - 0.04 * RES[1]
        inside = (uu >= 0) & (uu <= RES[0] - 1) & (hh_ >= 0) & (hh_ <= RES[1] - 1)
        seen = (vd >= zb.reshape(-1)[vi] - OCCLUDE) & inside
        facing = (vn @ torch.tensor(toward, device=dev, dtype=vn.dtype)).clamp(min=0.0)
        if name == "front":
            seen_front = (seen & (facing > 0.15)).cpu().numpy()
            if lab_d is not None and (lab_d > 0).sum() > 1000:
                # the drawing itself, labelled: the sharpest picture there is of the figure's front (a parser reads
                # grey hair and a beard on the mesh's own rendering as cloth, and on the drawing as hair and face)
                dpx, dpy, d_iou = from_drawing(lab_d, V @ right.astype(np.float32), V[:, 1])
                if d_iou >= DRAW_IOU:
                    ld = torch.tensor(lab_d[dpy, dpx], device=dev).long()
                    wd = DRAW_WEIGHT * (0.2 + facing) * seen.float() * (ld > 0).float()
                    votes.scatter_add_(1, ld[:, None], wd[:, None])
        w = wt * (0.2 + facing) * seen.float()
        ll = lab.reshape(-1)[vi]
        w = w * (ll > 0).float()                                  # the ground is no one's label
        votes.scatter_add_(1, ll[:, None], w[:, None])
        shares[name] = {NAMES[i]: round(float((lab == i).float().mean()), 4) for i in range(len(NAMES))
                        if float((lab == i).float().mean()) > 0.0005}
        a = np.asarray(pic)
        tiles.append(np.hstack([a, (0.35 * a + 0.65 * PAL[lab.cpu().numpy()]).astype(np.uint8)]))
    head_prior = 0
    if dpx is not None and d_iou >= DRAW_IOU and (lab_d == FACE).any():
        # behind the drawing's head is the back of the head: a vertex the front does not see, above the drawing's
        # chin, whose place in the drawing is the head's (hair, face or a hat), is the head's own
        chin_row = int(np.nonzero((lab_d == FACE).any(1))[0].max())
        ld = lab_d[dpy, dpx]
        back = (~seen_front) & (dpy < chin_row) & np.isin(ld, (HAIR, FACE, 9))
        votes[torch.tensor(np.nonzero(back)[0], device=dev), HAIR] += HEAD_PRIOR
        head_prior = int(back.sum())
    got = votes.sum(1) > 0
    labels = votes.argmax(1)
    if (~got).any() and got.any():                               # unseen by every view: the nearest labelled vertex's
        src, dst = torch.tensor(V, device=dev)[got], torch.tensor(V, device=dev)[~got]
        near = torch.cat([torch.cdist(dst[i:i + 4096], src).argmin(1) for i in range(0, len(dst), 4096)])
        labels[~got] = labels[got][near]
    labels = labels.cpu().numpy().astype(np.uint8)
    # one pass of the neighbours' majority (a lone vertex of another class inside a region is noise)
    E = m.edges_unique
    cnt = np.zeros((len(V), len(NAMES)), np.float32)
    np.add.at(cnt, (E[:, 0], labels[E[:, 1]]), 1.0)
    np.add.at(cnt, (E[:, 1], labels[E[:, 0]]), 1.0)
    cnt[np.arange(len(V)), labels] += 1.5
    labels = cnt.argmax(1).astype(np.uint8)
    recol, col_facts = 0, {}
    if dpx is not None and d_iou >= DRAW_IOU:
        # the hair the drawing does not show (a plait down the back): told by the figure's own colours, learned from
        # what the drawing's labels say is its hair and its garments. Only a vertex the front does not see, from the
        # shoulder blades up, now labelled a garment, whose colour is the hair's and clearly not any garment's
        ld = lab_d[dpy, dpx]
        hs = seen_front & (ld == HAIR)
        gs = seen_front & np.isin(ld, GARMENT)
        if hs.sum() >= 200 and gs.sum() >= 500:
            dmin = lambda C_: np.sqrt(((VC[:, None] - C_[None]) ** 2).sum(-1)).min(1)       # noqa: E731

            def main_colours(X, k):          # a model's clusters holding a twentieth of it or more (a few hair
                C = kmeans(X, k)             # pixels at a garment's edge made "a garment colour" of the hair's)
                n = np.bincount(((X[:, None] - C[None]) ** 2).sum(-1).argmin(1), minlength=k)
                return C[n >= 0.05 * len(X)]
            hc, gc = main_colours(VC[hs], 2), main_colours(VC[gs], 4)
            dh, dg = dmin(hc), dmin(gc)
            apart = float(np.sqrt(((hc[:, None] - gc[None]) ** 2).sum(-1)).min())
            up = V[:, 1] > V[:, 1].min() + 0.62 * Hf
            cand = (~seen_front) & up & np.isin(labels, GARMENT)
            col_facts = {"hair": hc.round(3).tolist(), "garment": gc.round(3).tolist(), "apart": round(apart, 3),
                         "unseen_upper_garment": int(cand.sum())}
            # (vertex by vertex: one whose colour sits between the hair's and a garment's stays what it was; the
            # models as wholes may share a mid-tone, a highlight in dark hair and a shadow in a grey shirt)
            m_ = cand & (dh < 0.5 * dg)
            labels[m_] = HAIR
            recol = int(m_.sum())
    groups = GROUP[labels]
    facts = {"drawing": str(drawing) if drawing else None, "drawing_overlap": d_iou, "head_prior": head_prior,
             "hair_by_colour": recol, "colours": col_facts,
             "figure": str(glb), "front_sign": fs, "vertices": int(len(V)), "unseen": int((~got).sum().item()),
             "share": {NAMES[i]: round(float((labels == i).mean()), 4) for i in range(len(NAMES)) if (labels == i).any()},
             "views": shares}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, points=V, labels=labels, groups=groups, names=np.array(NAMES), facts=json.dumps(facts))
    if sheet:
        # the four views with their labels, and the surface's own voted labels seen from the front and the back
        def voted(deg):
            right, toward = view_axes(deg, fs)
            scale = RES[1] * 0.92 / (Hf * 0.34)
            lo = (float(np.array([hx, 0.0, hz]) @ right), float(V[:, 1].min() + 0.66 * Hf))
            lc = torch.tensor(PAL[np.concatenate([labels[Fc[fi]][np.arange(N_SAMPLES), bw.argmax(1)], labels])]
                              .astype(np.float32) / 255.0, device=dev)
            img, zb, _, _ = splat(torch, pts, lc, right, toward, lo, scale, dev)
            return (fill_holes(torch, img, zb).clamp(0, 1) * 255).byte().cpu().numpy()
        row = np.hstack(tiles[4:6] + [voted(0.0), voted(180.0)])          # the head views, front and back, and the vote
        Image.fromarray(row).resize((row.shape[1] // 2, row.shape[0] // 2)).save(sheet, quality=88)
    return facts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("glb")
    ap.add_argument("out")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--drawing", default=None, help="the drawing the figure was made from (its front view)")
    a = ap.parse_args()
    f = parse(Path(a.glb), Path(a.out), Path(a.sheet) if a.sheet else None, Path(a.drawing) if a.drawing else None)
    print(json.dumps({k: f[k] for k in ("share", "unseen", "front_sign", "drawing_overlap", "head_prior", "hair_by_colour", "colours")}))


if __name__ == "__main__":
    main()
