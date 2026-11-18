"""A cabin people sit inside, as a shell seen from within: one panorama picture of its inside (equirectangular, taken
from the head of whoever sits in it) -> its depth (MoGe-2, a panorama split into views and merged) -> a mesh of
everything the picture shows, facing inward, with the panes of its windows made glass (CLIPSeg finds them), the
surface behind each near thing carried on behind it, and the picture as its texture.

    python pano_shell.py PICTURE OUT.glb [--grid 512] [--device cuda] [--outdoor]

--outdoor: a place in the open, the panorama taken from a standing eye (outdoor() below): the land, a sky dome in the
picture's own sky, and a flat patch of ground under the eye where people stand; OUT.json: the eye, the point under it
(stand_m), the sky's light.

Why a shell and not an object: a picture-to-3D model makes a thing seen from outside, so a cabin drawn from inside
came out as a pod cut open where the drawing's camera stood, with no rear wall and a closed nose (11-14). A panorama
shows every wall once. The faces look inward and have no back: a camera outside the shell sees through the wall
nearest it into the cabin, as a film set's walls are taken out for the camera.

Asset frame of the .glb (as the library's contract, written y-up for glTF): z up, ahead -y, the lowest floor at
z 0, the middle of its footprint at the origin. OUT.json beside it: the eye, the size, and the seat under the eye (seat frame as
scenes/seatbox.py).

Runs on the card (the depth and the segmentation models); the mesh itself is array arithmetic.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EYE_UP_M = 0.78             # a sitter's eyes above the seat
SEAT_W_M = 0.56             # the width a person takes
PANE_MIN = 0.004            # a pane is at least this share of the picture
PANE_SURE = 0.3             # ... and at least this sure, against the surest pane (a dark screen is not glass)
GLASS_KEY = (0, 0, 255)     # the panes' colour in the shell's picture
GLASS_REGION = {"name": "pane", "material": "glass", "hsv": [[0.62, 0.70], [0.9, 1.0], [0.9, 1.0]],
                "color": [0.95, 0.97, 1.0], "rough": 0.02, "ior": 1.0}   # ... and what the asset chain makes of it (flat, clear)
VIEW_PX = 512               # the side of each view the panorama is split into for the depth model


def depth_of(image: np.ndarray, device: str, with_mask: bool = False):
    """The panorama's distance map in metres (H x W); with_mask: (distance, where the depth model found a
    surface)."""
    import cv2
    import torch
    import utils3d
    from moge.model.v2 import MoGeModel
    from moge.utils.panorama import get_panorama_cameras, merge_panorama_depth, split_panorama_image
    h, w = image.shape[:2]
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    ext, intr = get_panorama_cameras()
    views = split_panorama_image(image, ext, intr, VIEW_PX)
    dist, masks = [], []
    for i in range(0, len(views), 4):
        t = torch.tensor(np.stack(views[i:i + 4]) / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        fx, _ = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(intr[i:i + 4])))
        out = model.infer(t, fov_x=torch.tensor(fx, dtype=torch.float32, device=device), apply_mask=False)
        dist.extend(list(out["points"].norm(dim=-1).cpu().numpy()))
        masks.extend(list(out["mask"].cpu().numpy()))
    del model
    torch.cuda.empty_cache()
    d, m = merge_panorama_depth(min(1920, w), min(960, h), dist, masks, ext, intr)
    d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if with_mask:
        return d, cv2.resize(np.asarray(m, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return d


def seg_prob(im, device: str, texts: list, first: int = 1) -> np.ndarray:
    """How sure the segmenter (CLIPSeg) is, pixel by pixel, that the panorama shows texts[0] (H x W, 0..1; the other
    texts are asked beside it; first > 1: the surest of the first that many texts): asked on square views half over
    one another round the panorama, averaged."""
    import torch
    from PIL import Image
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    w, h = im.size
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
    acc, cnt = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    wide = Image.new("RGB", (w + h, h))
    wide.paste(im, (0, 0))
    wide.paste(im.crop((0, 0, h, h)), (w, 0))                       # (the seam wraps)
    for x0 in range(0, w, h // 2):
        tile = wide.crop((x0, 0, x0 + h, h))
        with torch.no_grad():
            o = model(**proc(text=list(texts), images=[tile] * len(texts), return_tensors="pt", padding=True).to(device))
        p = torch.sigmoid(o.logits[:first]).amax(0).cpu().numpy()
        xs = np.arange(x0, x0 + h) % w
        acc[:, xs] += np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize((h, h), Image.BILINEAR)) / 255.0
        cnt[:, xs] += 1
    del model
    return acc / np.maximum(cnt, 1)


def glass_of(im, device: str) -> np.ndarray:
    """Where the picture shows a window's pane (H x W bool): the segmenter asked for "a window" on square views
    half over one another round the panorama, then whole panes kept or dropped by how sure it is of each."""
    import scipy.ndimage as ndi
    w, h = im.size
    prob = seg_prob(im, device, ["a window", "a wall"])
    # a pane is a dark even patch (the picture is drawn with a plain even darkness outside) that the segmenter takes
    # for a window. The segmenter alone marks a window whole, its bars and frame with it, and misses the smaller
    # panes; darkness alone would take a dark screen. So: how dark glass is, is learnt from where the segmenter is
    # surest; every dark patch is then a pane or not by how sure the segmenter is of it, against the surest.
    lum = ndi.uniform_filter(np.asarray(im.convert("L"), np.float32), 5, mode="wrap")
    core = prob > 0.6
    if core.sum() < 100:
        return np.zeros((h, w), bool)
    g0 = np.asarray(im.convert("L"), np.float32)
    rough = np.sqrt(np.maximum(ndi.uniform_filter(g0 * g0, 7, mode="wrap") - ndi.uniform_filter(g0, 7, mode="wrap") ** 2, 0))
    dark = ndi.binary_opening((lum <= float(np.percentile(lum[core], 90)) * 1.15 + 2.0)
                              & (rough <= float(np.percentile(rough[core], 90)) * 1.5 + 0.5), iterations=2)
    raw = np.concatenate([dark, dark[:, :1]], 1)                    # a pane over the seam is one pane
    lab, n = ndi.label(raw)
    for i in np.unique(lab[:, -1][lab[:, -1] > 0]):
        j = lab[:, 0][lab[:, -1] == i]
        if j.size and j.max() > 0:
            lab[lab == i] = j.max()
    lab = lab[:, :-1]
    ids = [i for i in np.unique(lab) if i and (lab == i).mean() > PANE_MIN]
    sure = {i: float(prob[lab == i].mean()) for i in ids}
    print("panes", sorted(((round(float((lab == i).mean()), 4), round(v, 2)) for i, v in sure.items()), reverse=True))
    if not ids:
        return np.zeros((h, w), bool)
    panes = np.isin(lab, [i for i in ids if sure[i] >= PANE_SURE * max(sure.values())])
    return ndi.binary_fill_holes(ndi.binary_closing(panes, iterations=6))       # (a pane is whole: no specks in it)


SHELL_BUILD = 2             # the shell's build: 2 has the backing hull (a library shell of an older build is rebuilt)
HULL_CELLS = 15             # the backing hull follows the walls' farthest reach over this many grid cells round each
HULL_BACK = 1.03            # ... and stands this much farther out than anything the picture shows in that direction
STEEP_DEG = 78.0            # a face turned further than this from facing the eye is an edge seen end on, not a surface
JUMP = 1.15                 # neighbours whose distances differ by more than this are two things, one behind the other


def behind(d: np.ndarray, g: np.ndarray):
    """What stands behind the edges of near things: at a depth jump the far side's surface is carried on behind the
    near thing, cell by cell, until it meets itself from the other side: the wall round a seat closes behind the
    seat. -> (distance or 0, the row and column it was carried from, the far sides). The picture shows nothing there;
    without it the gap round every near thing, and the whole wall behind the seat the picture is taken from, showed
    the dark outside (11-15)."""
    gh, gw = d.shape
    rr, cc = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    steps = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]

    def moved(x, a, b):
        y = np.roll(x, (a, b), (0, 1))                               # (columns wrap round the panorama; rows do not)
        if a:
            y[0 if a > 0 else -1, :] = 0
        return y
    seed = np.zeros_like(d, bool)
    for a, b in steps:
        n = moved(d, a, b)
        seed |= (n > 0) & (n * JUMP < d)                             # a neighbour much nearer: this is the far side
    seed &= ~g                         # (what is seen through a pane is far outside: no wall is carried on from it)
    fd, sr, sc = np.where(seed, d, 0.0), rr.copy(), cc.copy()
    for _ in range(max(gh, gw)):
        moved_on = False
        for a, b in steps:
            n = moved(fd, a, b)
            take = (n > d * JUMP) & (fd == 0) & ~g                   # (the first to arrive: the nearest far side)
            moved_on = moved_on or bool(take.any())
            fd = np.where(take, n, fd)
            sr = np.where(take, moved(sr, a, b), sr)
            sc = np.where(take, moved(sc, a, b), sc)
        if not moved_on:
            break
    return fd, sr, sc, seed


def shell(depth: np.ndarray, glass: np.ndarray, grid: int, steep_deg: float = STEEP_DEG, lowest: bool = False):
    """-> vertices (asset frame, z up, the middle of the footprint at the origin), faces (wound to face the eye), uv,
    the eye's place. lowest: the asset's z 0 is its lowest point (outdoors: land may lie below where people stand);
    else the floor, the lowest hundredth, with nothing under it."""
    import scipy.ndimage as ndi
    import utils3d
    from moge.utils.panorama import spherical_uv_to_directions
    H, W = depth.shape
    gw, gh = grid, grid // 2
    ys, xs = ((np.arange(gh) + 0.5) * H / gh).astype(int), ((np.arange(gw) + 0.5) * W / gw).astype(int)
    # (a pixel beside a depth jump holds a mix of near and far: the grid takes the picture's median round each point)
    d = ndi.median_filter(depth, size=max(3, (W // gw) | 1), mode="wrap")[np.ix_(ys, xs)].astype(np.float64)
    g = glass[np.ix_(ys, xs)]
    dirs = spherical_uv_to_directions(utils3d.numpy.image_uv(width=gw, height=gh))
    dirs = np.stack([-dirs[..., 1], dirs[..., 0], dirs[..., 2]], -1)  # the picture's middle (-x there) is ahead: -y
    u = (np.arange(gw + 1) + 0.5) / gw                                # one more column past the seam, u past the edge
    v = 1 - (np.arange(gh) + 0.5) / gh
    fd, sr, sc, seed = behind(d, g)
    # each pane's own surface: the distance carried in from the frame round it (through the glass the depth model
    # saw the dark outside, at no distance one could use), smoothed across the pane
    gd = d.copy()
    if g.any() and not g.all():
        near = ndi.distance_transform_edt(g, return_distances=False, return_indices=True)
        gd = d[near[0], near[1]]
        for _ in range(60):
            sm = (np.roll(gd, 1, 1) + np.roll(gd, -1, 1) + np.vstack([gd[:1], gd[:-1]]) + np.vstack([gd[1:], gd[-1:]])) / 4
            gd = np.where(g, sm, gd)

    def wrap(x):                                                      # the first column again after the last
        return np.concatenate([x, x[:, :1]], 1)
    idx = np.arange(gh * (gw + 1)).reshape(gh, gw + 1)
    a, b, b2, a2 = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    quads = np.stack([a, b, b2, a2], -1).reshape(-1, 4)

    def flat(x):
        return wrap(x).reshape(-1)[quads]
    # the surfaces the picture shows: no glass, and no skin stretched from a near thing to what is behind it (seen
    # from anywhere but the eye such a skin is a wall across the cabin)
    # (nor a sliver along such an edge: seen from the side it hung in the air as a shard, 11-15)
    steep = 1.0 + (2 * np.pi / gw) * np.tan(np.radians(steep_deg))
    q1 = quads[~flat(g).any(1) & (flat(d).max(1) < steep * flat(d).min(1))]
    qg = quads[flat(g).any(1)]                                        # the panes (their own smooth surface)
    # what is behind the near things: carried on from the far side of each jump, in the far side's own colour; where
    # the far side is glass it stays open
    fq = flat(fd)
    su = (sc + 0.5) / gw
    uq = flat(su)
    q2 = quads[(fq > 0).all(1) & (fq.max(1) < steep * fq.min(1)) & ~flat(g[sr, sc]).any(1) & ~flat(seed).all(1)
               & (uq.max(1) - uq.min(1) < 0.5)]
    # the backing hull: one closed smooth surface just beyond everything the picture shows, in the picture's own
    # colours, open only where a pane is. The surfaces above leave tears wherever one thing hid another from the eye
    # (the side of a seat, the wall behind a console); seen from a camera away from the eye a tear showed the stage
    # beyond the cabin, white patches round the canopy in every cockpit clip (11-15). Through a tear the hull shows
    # wall, in the colour the picture has in that direction.
    wall = np.where(g, 0.0, np.maximum(d, fd))
    # (outdoors the rows do not wrap: the sky's dome at the top of the picture pushed the hull under the ground
    # 150 m down; a cabin's shell keeps the build it was made with)
    hm = ("nearest", "wrap") if lowest else "wrap"
    hd = ndi.maximum_filter(wall, size=HULL_CELLS, mode=hm)
    hd = np.where(hd > 0, hd, ndi.maximum_filter(wall, size=HULL_CELLS * 4, mode=hm))
    for _ in range(3):
        hd = ndi.uniform_filter(hd, size=HULL_CELLS, mode=hm)
    hd = np.maximum(hd, wall) * HULL_BACK
    q3 = quads[~flat(g).any(1)]
    n1 = idx.size
    vtx = np.concatenate([(wrap(dirs[..., k]) * wrap(gd)).reshape(-1) for k in range(3)]
                         + [(wrap(dirs[..., k]) * wrap(fd)).reshape(-1) for k in range(3)]
                         + [(wrap(dirs[..., k]) * wrap(hd)).reshape(-1) for k in range(3)]).reshape(3, 3, -1)
    vtx = np.concatenate([vtx[0].T, vtx[1].T, vtx[2].T])
    uv1 = np.stack(np.meshgrid(np.clip(u, 0, 1), v), -1).reshape(-1, 2)
    uv2 = np.stack([wrap(su), wrap(1 - (sr + 0.5) / gh)], -1).reshape(-1, 2)
    uv = np.concatenate([uv1, uv2, uv1])
    quads = np.concatenate([q1, qg, q2 + n1, q3 + 2 * n1])
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    c = vtx[faces]
    n = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    if (np.einsum("ij,ij->i", n, c.mean(1)) > 0).mean() > 0.5:       # the faces look at the eye (the origin)
        faces = faces[:, ::-1]
    used = np.unique(faces)                                           # (only the points a face stands on)
    remap = np.full(len(vtx), -1)
    remap[used] = np.arange(len(used))
    vtx, uv, faces = vtx[used], uv[used], remap[faces]
    floor = float(vtx[:, 2].min() if lowest else np.percentile(vtx[:, 2], 1))
    vtx[:, 2] = np.maximum(vtx[:, 2] - floor, 0.0)
    mid = (vtx[:, :2].min(0) + vtx[:, :2].max(0)) / 2                # (the asset chain puts the origin there)
    vtx[:, :2] -= mid
    return vtx, faces, uv, [float(-mid[0]), float(-mid[1]), -floor]


def seat_under(eye: list) -> dict:
    """The seat of whoever the picture was taken from: under the eye (seat frame: front -y, as scenes/seatbox.py)."""
    x, y, z = eye
    back = y + 0.15                                                   # the back's plane behind the eyes
    return {"seat_z_m": round(max(0.05, z - EYE_UP_M), 3), "front_y_m": round(back - 0.5, 3), "back_y_m": round(back, 3),
            "inner_x_m": [round(x - SEAT_W_M / 2, 3), round(x + SEAT_W_M / 2, 3)], "arm_z_m": None, "width_m": SEAT_W_M,
            "depth_m": 0.5, "places": [round(x, 3)], "front_asset": [0.0, -1.0], "inside": True, "open_walls": True, "seats_abreast": 1,
            "source": "img3d/pano_shell.py: under the eye the panorama was taken from"}


# ------------------------------------------------------------------ outdoors: a place in the open, as a world
# A place in the open is the same shell made from a panorama taken at a standing person's eye: the land the picture
# shows round the eye, open to a sky dome in the picture's own sky, and a flat patch of ground under the eye where the
# people stand (film previs: terrain, sky, blocked figures; the panorama -> sky / land -> depth -> mesh layering of
# panorama world builders). The depth model's scale on a painted panorama runs small, so the land is scaled to put
# the ground straight below at a standing eye's height (as tools/world.py's ground fit, "nadir").
EYE_STAND_M = 1.6           # a standing person's eyes above the ground: the outdoor panorama is drawn from there
FLAT_R_M = 9.0              # the ground people stand on is flat this far round the point under the eye
FLAT_KEEP_M = 1.0           # a point higher than this over the ground stands on it (a fence, a wall): not flattened
#                             (low stones by the feet, kept at 0.5 m, were sheets smeared toward every other camera)
RAMP_R_M = 25.0             # beyond the patch the land rises or falls back to its own height by this distance
SKY_R_M = 150.0             # the sky's dome: this far from the eye, the land inside it
LAND_MAX_K = 0.9            # ... no land farther out than this share of the dome
OUT_STEEP_DEG = 89.0        # far ground is seen nearly edge on from the eye: kept (at 78 degrees it was all dropped)
SKY_SURE = 0.5              # the segmenter's least sureness for sky
WORLD_BUILD = 4             # the outdoor shell's build (a library world of an older build is made again)
#                             4: the land's own things the plan sends people to are found in it (marks_m)


def pano_dirs(h: int, w: int) -> np.ndarray:
    """Unit directions of an equirectangular picture's pixels in the asset frame of shell() (z up, the picture's
    middle ahead at -y), H x W x 3."""
    u, v = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    th, ph = (1 - u) * 2 * np.pi, v * np.pi
    d = np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], -1)
    return np.stack([-d[..., 1], d[..., 0], d[..., 2]], -1)


SKY_PLAIN = 7.0             # a stretch of picture this even (the spread of its grey levels round each point) ...
SKY_NEAR = 36.0             # ... and this near the colour of the top of the picture (summed over red, green, blue) is
#                             sky too: a painted night sky is a plain dark the segmenter does not call sky, and the
#                             depth model put a lid over the eye there (as tools/world.py found with its sky model)


def plain_top(rgb: np.ndarray) -> np.ndarray:
    """Where the picture is as even as, and near the colour of, the very top of the picture (H x W bool)."""
    import scipy.ndimage as ndi
    h = rgb.shape[0]
    f = rgb[..., :3].astype(np.float32)
    g = f @ np.array([0.299, 0.587, 0.114], np.float32)
    m = ("nearest", "wrap")                                           # (rows do not wrap: the top is not the bottom)
    sd = np.sqrt(np.maximum(ndi.uniform_filter(g * g, 7, mode=m) - ndi.uniform_filter(g, 7, mode=m) ** 2, 0))
    top = np.median(f[:max(2, h // 20)].reshape(-1, 3), 0)
    return (np.abs(f - top).sum(-1) <= SKY_NEAR) & (sd <= SKY_PLAIN)


def sky_mask(prob: np.ndarray, valid: np.ndarray, rgb: np.ndarray | None = None) -> np.ndarray:
    """The sky (H x W bool): above the horizon, where the segmenter is sure of sky, the depth model found no
    surface, or the picture is as plain as its very top (rgb given), and joined to the top of the picture (a pale
    wall in a sky's colour is not sky)."""
    import scipy.ndimage as ndi
    h, w = prob.shape
    up = (np.cos(np.pi * (np.arange(h) + 0.5) / h) > 0.0)[:, None]
    raw = up & ((prob >= SKY_SURE) | ~valid | (plain_top(rgb) if rgb is not None else False))
    pad = np.pad(raw, ((4, 4), (4, 4)), mode="wrap")                  # (columns wrap; rows go on as they are)
    pad[:4], pad[-4:] = pad[4], pad[-5]
    raw = ndi.binary_opening(pad, iterations=2)[4:-4, 4:-4]
    lab, n = ndi.label(np.concatenate([raw, raw[:, :1]], 1))
    band = lab[:max(1, h // 50)]                                      # (the top rows: one point is one row there)
    top = set(np.unique(band[band > 0]).tolist())
    for i in np.unique(lab[:, -1][lab[:, -1] > 0]):                   # (a patch over the seam is one patch)
        j = lab[:, 0][lab[:, -1] == i]
        if j.size and j.max() > 0 and (i in top or j.max() in top):
            top |= {int(i), int(j.max())}
    sky = np.isin(lab[:, :-1], sorted(top)) if top else np.zeros((h, w), bool)
    return ndi.binary_fill_holes(sky) & up                            # (the moon, a star: sky)


GROUND_BLOCK = 16           # the land's own ground height is read in blocks of this many pixels ...
GROUND_PCT = 25             # ... as their low quarter (a thing standing on the ground is above it), then smoothed


def ground_of(over: np.ndarray, down: np.ndarray) -> np.ndarray:
    """The depth model's ground height over the plane round each point below the horizon (H x W, metres): each
    block's low quarter of the heights in it, smoothed across blocks (columns wrap), 0 where no block has ground."""
    h, w = over.shape
    b = GROUND_BLOCK
    hb, wb = -(-h // b), -(-w // b)
    pad = np.full((hb * b, wb * b), np.nan)
    pad[:h, :w] = np.where(down, over, np.nan)
    blk = pad.reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb, wb, b * b)
    has = np.isfinite(blk).sum(-1) >= b
    import warnings
    with warnings.catch_warnings():                                   # (a block with no ground: all empty)
        warnings.simplefilter("ignore", RuntimeWarning)
        g = np.where(has, np.nanpercentile(blk, GROUND_PCT, axis=-1), 0.0)
    wgt = has.astype(np.float64)
    g = g * wgt
    for _ in range(4):                       # (3 x 3 means: columns wrap, rows end where they are)
        def sm(x):
            y = x + np.roll(x, 1, 1) + np.roll(x, -1, 1)
            return y + np.vstack([y[:1], y[:-1]]) + np.vstack([y[1:], y[-1:]])
        g, wgt = sm(g), sm(wgt)
        g = np.where(wgt > 0, g / np.maximum(wgt, 1e-9), 0.0)
        wgt = (wgt > 0).astype(np.float64)
        g = g * wgt
    return np.repeat(np.repeat(g, b, 0), b, 1)[:h, :w]


OUT_GRID = 288             # the outdoor shell's grid: under the asset chain's triangle budget, so it is not thinned
#                             (thinned, the flat ground was folded into a few great triangles, and the panorama, which is
#                             not linear across them, swirled over the ground in every still, 11-16)
SKY_HIGH_DEG = 12.0         # high in the picture, what the depth model puts this far out is sky too (a cloud's
SKY_FAR_M = 30.0            # bright edge the segmenter did not take hung as shards over the land)


def high_far(d: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """Sky by place: higher than SKY_HIGH_DEG over the horizon and farther than SKY_FAR_M (d scaled to metres), joined
    to the sky found already."""
    h = d.shape[0]
    el = np.degrees(np.arcsin(np.clip(np.cos(np.pi * (np.arange(h) + 0.5) / h), -1, 1)))[:, None]
    return sky | ((el > SKY_HIGH_DEG) & (d > SKY_FAR_M))


def land(depth: np.ndarray, sky: np.ndarray, eye: float = EYE_STAND_M) -> tuple:
    """The panorama's distances (H x W, metres) made into the place: scaled so the ground straight below is a
    standing eye's height down; near the eye the depth model's own ground height (ground_of: it stood 0.5 to 1.3 m
    over the plane a few metres out, a step) taken off everything, so its ground is the plane and what stands on it
    stands on the plane, fading out between FLAT_R_M and RAMP_R_M; the ground round the point under the eye one flat
    plane (FLAT_R_M; a thing standing on it is kept); no land below the plane (the picture is drawn from a level eye,
    and what the depth model put below it was a slope that is not in the picture: the land fell 10 m within 40 m
    round a flat field); the sky at the dome; no land beyond the dome. Every point moves along its own ray, so the
    eye sees it where the picture has it. -> (the distances, facts)."""
    h, w = depth.shape
    dz = np.broadcast_to(np.cos(np.pi * (np.arange(h) + 0.5) / h)[:, None], (h, w))
    below = ~sky & (dz < -0.87)                                       # within 30 degrees of straight down
    raw = float(np.median(depth[below] * -dz[below])) if below.sum() > 200 else eye
    k = eye / max(raw, 1e-3)
    d = depth.astype(np.float64) * k
    sky = high_far(d, sky)
    down = (dz < -1e-3) & ~sky
    t = np.where(down, eye / np.maximum(-dz, 1e-3), np.inf)          # along the ray to the ground plane
    over = d * dz + eye                                               # each point's height over the plane
    g = ground_of(over, down)
    rr = d * np.sqrt(np.maximum(1 - dz * dz, 0))                      # how far out each point is
    keep = 1.0 - np.clip((rr - FLAT_R_M) / (RAMP_R_M - FLAT_R_M), 0.0, 1.0)
    over = np.where(down, over - keep * g, over)
    flat = down & (over < FLAT_KEEP_M) & (rr < FLAT_R_M)
    over = np.where(flat, 0.0, over)
    d = np.where(down, (over - eye) / np.minimum(dz, -1e-3), d)
    d = np.where(down, np.minimum(d, t), d)                           # (nothing under the ground)
    d = np.where(sky, SKY_R_M, np.minimum(d, LAND_MAX_K * SKY_R_M))
    return d, {"scale": round(k, 4), "eye_raw_m": round(raw, 4), "flat_share": round(float(flat.mean()), 4),
               "sky_share": round(float(sky.mean()), 4), "ground_near_m": round(float(np.median(g[below])) if below.any() else 0.0, 3)}


def sun_of(rgb: np.ndarray, sky: np.ndarray) -> dict | None:
    """Where the picture's light comes from: the brightest part of its sky (the sun, the moon, the bright side of
    the clouds), its direction (asset frame), its colour and how bright the sky is (0..1). None without a sky."""
    if sky.mean() < 0.01:
        return None
    lum = rgb[..., :3].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    cut = max(float(np.percentile(lum[sky], 99.7)), 0.85 * float(lum[sky].max()))
    hot = sky & (lum >= cut)
    d = pano_dirs(*sky.shape)[hot].mean(0)
    d = d / max(np.linalg.norm(d), 1e-9)
    c = rgb[hot][:, :3].astype(np.float64).mean(0)
    c = c * 255.0 / max(c.max(), 1.0)
    return {"dir": [round(float(x), 4) for x in d], "color": "#%02x%02x%02x" % tuple(int(x) for x in c),
            "peak": round(float(lum[hot].mean() / 255.0), 4), "level": round(float(lum[sky].mean() / 255.0), 4)}


SPOT_BACK_M = 0.8           # people sent to a part of the land stand this far short of it, on the ground before it
SPOT_SURE = 0.35            # the segmenter's least sureness for that part of the land


def spot_of(prob: np.ndarray, d: np.ndarray, sky: np.ndarray) -> list | None:
    """Where people sent to a part of the land stand (eye frame, metres: the point under the eye at 0, 0): the
    nearest point of the ground the segmenter takes for that part (prob, H x W), SPOT_BACK_M short of it toward the
    eye, at most FLAT_R_M less a step out (on the flat ground). None when the picture shows no such part below
    the horizon."""
    h, w = prob.shape
    dz = np.cos(np.pi * (np.arange(h) + 0.5) / h)[:, None] * np.ones((1, w))
    at = (dz < -0.02) & ~sky & (prob >= max(SPOT_SURE, 0.6 * float(prob[(dz < -0.02) & ~sky].max(initial=0.0))))
    if at.sum() < 20:
        return None
    p = pano_dirs(h, w)[at] * d[at][:, None]
    r = np.hypot(p[:, 0], p[:, 1])
    i = int(np.argmin(r))
    u = p[i, :2] / max(r[i], 1e-6)
    far = min(max(r[i] - SPOT_BACK_M, 0.0), FLAT_R_M - 1.5)
    return [round(float(u[0] * far), 3), round(float(u[1] * far), 3)]


MARK_MIN_PX = 150           # a thing of the land the segmenter finds is at least this many of the panorama's points


def mark_of(prob: np.ndarray, d: np.ndarray, sky: np.ndarray) -> dict | None:
    """A thing that is part of the land (a cave's mouth, a ledge, a ford) as the panorama shows it (prob, H x W, the
    segmenter's sureness of it): which way it lies from the eye, how far off its near side is, and where people sent
    to it stand: on the flat ground toward it, SPOT_BACK_M short of it (eye frame, metres). None when the picture
    does not show it: then it is a piece made on its own and stood on the ground."""
    h, w = prob.shape
    land_ = ~sky
    top = float(prob[land_].max(initial=0.0))
    at = land_ & (prob >= max(SPOT_SURE, 0.6 * top))
    if at.sum() < MARK_MIN_PX:
        return None
    dirs = pano_dirs(h, w)[at]
    p = dirs * d[at][:, None]
    r = np.hypot(p[:, 0], p[:, 1])
    u = (dirs[:, :2] * prob[at][:, None]).sum(0)
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        return None
    u = u / n
    near = float(np.percentile(r, 5))
    far = min(max(near - SPOT_BACK_M, 0.0), FLAT_R_M - 1.5)
    return {"stand": [round(float(u[0] * far), 3), round(float(u[1] * far), 3)],
            "toward": [round(float(u[0]), 4), round(float(u[1]), 4)], "near_m": round(near, 2),
            "sure": round(top, 3), "share": round(float(at.mean()), 5)}


def outdoor(picture: str, out: str, grid: int, device: str, spot: str = "", marks: dict | None = None) -> int:
    """The outdoor place's shell: OUT.glb and OUT.json (the eye, where people stand, the sky's light)."""
    import trimesh
    from PIL import Image
    im = Image.open(picture).convert("RGB")
    rgb = np.asarray(im)
    depth, valid = depth_of(rgb, device, with_mask=True)
    # (a dark painted sky is "a night sky" to the segmenter, and not "the sky": both are asked)
    sky = sky_mask(seg_prob(im, device, ["the sky", "a night sky", "the ground", "a building", "a tree"], first=2),
                   valid, rgb)
    d, info = land(depth, sky)
    sp = spot_of(seg_prob(im, device, [spot, "the sky", "the ground"]), d, sky) if spot else None
    found = {}
    for mid, words in (marks or {}).items():
        mk = mark_of(seg_prob(im, device, [words, "the sky", "the ground"]), d, sky)
        if mk:
            found[mid] = dict(mk, words=words)
    vtx, faces, uv, eye = shell(d, np.zeros(d.shape, bool), grid, steep_deg=OUT_STEEP_DEG, lowest=True)
    for mk in found.values():                # (asset frame, as stand_m and spot_m are)
        mk["stand"] = [round(eye[0] + mk["stand"][0], 3), round(eye[1] + mk["stand"][1], 3), round(eye[2] - EYE_STAND_M, 3)]
    lo, hi = vtx.min(0), vtx.max(0)
    o = Path(out)
    o.parent.mkdir(parents=True, exist_ok=True)
    yup = np.stack([vtx[:, 0], vtx[:, 2], -vtx[:, 1]], -1)
    trimesh.Trimesh(yup, faces, visual=trimesh.visual.TextureVisuals(uv=uv, image=im), process=False).export(o)
    Image.fromarray((sky * 255).astype(np.uint8)).save(o.with_name(o.stem + "_sky.png"))
    facts = {"eye_m": [round(v, 3) for v in eye], "stand_m": [round(eye[0], 3), round(eye[1], 3),
                                                               round(eye[2] - EYE_STAND_M, 3)],
             "bounds_m": [lo.round(3).tolist(), hi.round(3).tolist()], "size_m": (hi - lo).round(3).tolist(),
             "faces": int(len(faces)), "flat_r_m": FLAT_R_M, "sky_r_m": SKY_R_M, "land": info, "spot": spot or None,
             "spot_m": [round(eye[0] + sp[0], 3), round(eye[1] + sp[1], 3), round(eye[2] - EYE_STAND_M, 3)] if sp else None,
             "marks_m": found, "sun": sun_of(rgb, sky), "world_build": WORLD_BUILD}
    o.with_suffix(".json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picture")
    ap.add_argument("out")
    ap.add_argument("--grid", type=int, default=0, help="the mesh grid's columns (512; outdoors OUT_GRID)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdoor", action="store_true", help="a place in the open, from a standing eye (outdoor())")
    ap.add_argument("--spot", default="", help="outdoors: words for the part of the land people are sent to")
    ap.add_argument("--mark", action="append", default=[], help="outdoors: ID=WORDS, a thing of the land to find")
    a = ap.parse_args()
    if a.outdoor:
        return outdoor(a.picture, a.out, a.grid or OUT_GRID, a.device, a.spot,
                       dict(m.split("=", 1) for m in a.mark if "=" in m))
    a.grid = a.grid or 512
    import trimesh
    from PIL import Image
    im = Image.open(a.picture).convert("RGB")
    depth = depth_of(np.asarray(im), a.device)
    glass = glass_of(im, a.device)
    vtx, faces, uv, eye = shell(depth, glass, a.grid)
    # the panes are painted one flat key colour in the shell's own picture: the asset chain makes what has that
    # colour glass (model_img3d "regions": GLASS_REGION)
    tex = np.asarray(im).copy()
    tex[glass] = GLASS_KEY
    im_tex = Image.fromarray(tex)
    lo, hi = vtx.min(0), vtx.max(0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    yup = np.stack([vtx[:, 0], vtx[:, 2], -vtx[:, 1]], -1)            # glTF is y up
    trimesh.Trimesh(yup, faces, visual=trimesh.visual.TextureVisuals(uv=uv, image=im_tex), process=False).export(out)
    Image.fromarray((glass * 255).astype(np.uint8)).save(out.with_name(out.stem + "_glass.png"))
    facts = {"eye_m": [round(v, 3) for v in eye], "bounds_m": [lo.round(3).tolist(), hi.round(3).tolist()],
             "size_m": (hi - lo).round(3).tolist(), "faces": int(len(faces)), "glass_share": round(float(glass.mean()), 3),
             "seat": seat_under(eye), "regions": [GLASS_REGION], "shell_build": SHELL_BUILD}
    out.with_suffix(".json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts))
    if facts["glass_share"] < 0.02:
        print("no window found in the picture", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
