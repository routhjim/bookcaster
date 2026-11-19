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
FLAT_R_M = 40.0             # the ground people stand on is flat this far round the point under the eye: the set's
#                             own real ground (ground_disk), with the panorama only as what lies beyond it. (At 9 m
#                             the land round the people was the panorama's depth seen from the side: smeared ground
#                             and things, another background in every key, 11-18)
FLAT_KEEP_M = 1.0           # a point higher than this over the ground stands on it (a fence, a wall): not flattened
#                             (low stones by the feet, kept at 0.5 m, were sheets smeared toward every other camera)
RAMP_R_M = 90.0             # beyond the patch the land rises or falls back to its own height by this distance
GROUND_UNDER_M = 0.05       # the panorama's own ground lies this far under the set's real ground, out of sight
TILE_M = 3.0                # the real ground's picture: a square of the panorama's ground this wide, near the eye,
TILE_PX = 256               # ... seen from straight above at this many points a side, laid mirrored edge to edge
TILE_NEAR_M = 2.5           # ... its middle this far from the point under the eye
MAP_R_M = 12.0              # round the point under the eye the real ground shows the panorama's own ground, seen
MAP_PX = 1024               # from straight above (a square this far each way, this many points a side): its path, its
MAP_IN_M = 6.0              # water's edge. Beyond MAP_IN_M it fades into the tile, all tile at MAP_R_M (far from the
#                             eye one point of the panorama is a long streak of ground)
SPOT_MAX_M = 7.5            # people sent to a part of the land stand at most this far from the point under the eye
SKY_R_M = 150.0             # the sky's dome: this far from the eye, the land inside it
LAND_MAX_K = 0.9            # ... no land farther out than this share of the dome
OUT_STEEP_DEG = 89.0        # far ground is seen nearly edge on from the eye: kept (at 78 degrees it was all dropped)
SKY_SURE = 0.5              # the segmenter's least sureness for sky
WORLD_BUILD = 6             # the outdoor shell's build (a library world of an older build is made again)
#                             4: the land's own things the plan sends people to are found in it (marks_m)
#                             6: the ground map's disk in merged blocks and the whole world under the chain's budget,
#                             its water from the edit model's key (a world of 5 was thinned: streaked ground, 11-18)


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


def backdrop(d: np.ndarray, sky: np.ndarray, eye: float = EYE_STAND_M) -> np.ndarray:
    """The land's distances with nothing of the panorama nearer than the edge of the set's real ground (FLAT_R_M):
    what the picture shows standing nearer is moved out along its own ray to that distance (the eye sees it where
    the picture has it, as part of what lies beyond; what really stands on the set is built there as a piece), and
    the picture's own ground inside it is put just under the real ground."""
    h, w = d.shape
    dz = np.broadcast_to(np.cos(np.pi * (np.arange(h) + 0.5) / h)[:, None], (h, w))
    hz = np.sqrt(np.maximum(1 - dz * dz, 1e-6))
    rr, over = d * hz, d * dz + eye
    near = ~sky & (rr < FLAT_R_M)
    stands = near & (over > 0.02)
    out = np.where(stands, np.minimum(FLAT_R_M / hz, LAND_MAX_K * SKY_R_M), d)
    under = near & ~stands & (dz < -1e-3)
    return np.where(under, (eye + GROUND_UNDER_M) / np.maximum(-dz, 1e-3), out)


def ground_tile(rgb: np.ndarray, d: np.ndarray, sky: np.ndarray, eye: float = EYE_STAND_M) -> np.ndarray:
    """The real ground's picture (2 TILE_PX square, uint8): the panorama's ground seen from straight above, a square
    TILE_M wide near the point under the eye, on the side where most of it is bare ground, laid mirrored beside and
    below itself so its edges join when it is repeated."""
    import scipy.ndimage as ndi
    import utils3d
    from moge.utils.panorama import spherical_uv_to_directions
    h, w = d.shape
    dirs = spherical_uv_to_directions(utils3d.numpy.image_uv(width=w, height=h))
    dirs = np.stack([-dirs[..., 1], dirs[..., 0], dirs[..., 2]], -1)
    down = dirs[..., 2] < -0.2
    t = eye / np.maximum(-dirs[..., 2], 1e-3)
    gx, gy = dirs[..., 0] * t, dirs[..., 1] * t                       # where each downward ray meets the ground
    bare = down & ~sky & (np.abs(d * dirs[..., 2] + eye) < 0.12)      # ... and whether the picture has ground there
    best = None
    for k in range(8):
        cx, cy = TILE_NEAR_M * np.sin(k * np.pi / 4), -TILE_NEAR_M * np.cos(k * np.pi / 4)
        ix = np.floor((gx - cx + TILE_M / 2) / TILE_M * TILE_PX).astype(int)
        iy = np.floor((gy - cy + TILE_M / 2) / TILE_M * TILE_PX).astype(int)
        m = down & (ix >= 0) & (ix < TILE_PX) & (iy >= 0) & (iy < TILE_PX)
        share = float(bare[m].mean()) if m.any() else 0.0
        if best is None or share > best[0]:
            best = (share, ix, iy, m)
    _, ix, iy, m = best
    tile = np.zeros((TILE_PX, TILE_PX, 3), np.float64)
    n = np.zeros((TILE_PX, TILE_PX), np.float64)
    m = m & bare if (m & bare).sum() > TILE_PX * TILE_PX // 8 else m
    np.add.at(tile, (iy[m], ix[m]), rgb[m].astype(np.float64))
    np.add.at(n, (iy[m], ix[m]), 1.0)
    if not n.any():
        tile[:] = rgb[down & ~sky].mean(0) if (down & ~sky).any() else 96.0
    else:
        got = n > 0
        tile[got] /= n[got][:, None]
        near = ndi.distance_transform_edt(~got, return_distances=False, return_indices=True)
        tile = tile[near[0], near[1]]
    tile = tile.clip(0, 255).astype(np.uint8)
    top = np.concatenate([tile, tile[:, ::-1]], 1)
    return np.concatenate([top, top[::-1]], 0)


def ground_map(rgb: np.ndarray, d: np.ndarray, sky: np.ndarray, tile: np.ndarray, eye: float = EYE_STAND_M) -> np.ndarray:
    """The real ground's picture round the point under the eye (MAP_PX square, uint8; row 0 at -y, ahead): the
    panorama's own ground seen from straight above where the picture has bare ground there, fading into the tile
    with distance; the tile where the picture shows something standing."""
    h, w = d.shape
    c = (np.arange(MAP_PX) + 0.5) / MAP_PX * 2 * MAP_R_M - MAP_R_M
    X, Y = np.meshgrid(c, c)
    n = np.sqrt(X * X + Y * Y + eye * eye)
    az = -eye / n
    u = (1 - np.arctan2(-X / n, Y / n) / (2 * np.pi)) % 1.0
    v = np.arccos(np.clip(az, -1, 1)) / np.pi
    px, py = np.clip((u * w).astype(int), 0, w - 1), np.clip((v * h).astype(int), 0, h - 1)
    own = rgb[py, px].astype(np.float64)
    bare = ~sky[py, px] & (np.abs(d[py, px] * az + eye) < 0.12)
    T, k = 2 * TILE_M, tile.shape[0]
    tl = tile[np.clip(((1 - (Y / T) % 1.0) * k).astype(int), 0, k - 1), np.clip((((X / T) % 1.0) * k).astype(int), 0, k - 1)]
    r = np.sqrt(X * X + Y * Y)
    wt = np.where(bare, np.clip((r - MAP_IN_M) / (MAP_R_M - MAP_IN_M), 0.0, 1.0), 1.0)[..., None]
    return (own * (1 - wt) + tl * wt).clip(0, 255).astype(np.uint8)


def ground_disk(eye: list, n0: int) -> tuple:
    """The set's real ground: a flat round FLAT_R_M out from the point under the eye; its middle one square showing
    the ground map (MAP_R_M each way), round it squares each showing the ground tile once. -> (vertices in the asset
    frame, faces looking up, uv into the lower half of the shell's picture: the map at its left, the tile right of
    it)."""
    T = 2 * TILE_M
    k = int(np.ceil(FLAT_R_M / T))
    m = int(round(MAP_R_M / T))
    z = eye[2] - EYE_STAND_M
    e = 2.0 / 2048
    vtx, uv, faces = [], [], []

    def quad(x0, y0, x1, y1, u0, v0, u1, v1):
        b = n0 + len(vtx)
        for x, y, u, v in ((x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1), (x0, y1, u0, v1)):
            vtx.append([eye[0] + x, eye[1] + y, z])
            uv.append([u, v])
        faces.extend([[b, b + 1, b + 2], [b, b + 2, b + 3]])
    quad(-MAP_R_M, -MAP_R_M, MAP_R_M, MAP_R_M, e, 0.5 - e, 0.5 - e, e)   # (the map's row 0, at -y, is its top)
    for i in range(-k, k):
        for j in range(-k, k):
            if -m <= i < m and -m <= j < m:
                continue
            if np.hypot((i + 0.5) * T, (j + 0.5) * T) > FLAT_R_M + T / 2:
                continue
            quad(i * T, j * T, (i + 1) * T, (j + 1) * T, 0.5 + e, 0.25 + e, 0.75 - e, 0.5 - e)
    return np.array(vtx), np.array(faces), np.array(uv)


# ------------------------------------------------------------------ the ground's own map (places/ground_map.py)
# With a map of the place's ground drawn from straight above, the whole real ground is that one picture (its land's
# water, banks and paths where the people are), fading at its edge into the panorama's ground colour; its water is
# its own faces a little under the ground, which the asset chain and the engine can give a water material.
MAP_FADE_M = 6.0            # the map fades into the panorama's own ground over this many metres inside the disk's edge
WATER_UNDER_M = 0.04        # the water's faces lie this far under the real ground (its banks stand over it)
WATER_UV = [0.5, 1.0, 0.0, 0.5]   # the water's faces show the map's copy in the lower right of the shell's picture:
#                                   the faces in this part of the picture (u0, u1, v0, v1) are the water's
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "places"))


def map_texture(map_rgb: np.ndarray, rgb: np.ndarray, d: np.ndarray, sky: np.ndarray, px: int = 1024) -> np.ndarray:
    """The real ground's picture from the drawn map (any size, row 0 at +y): resized to px a side, fading into the
    panorama's own ground at the disk's edge (ground_map.edge_ring and blend)."""
    import ground_map as GM
    from PIL import Image
    m = np.asarray(Image.fromarray(np.asarray(map_rgb, np.uint8)).convert("RGB").resize((px, px), Image.LANCZOS))
    return GM.blend(m, GM.edge_ring(rgb, d, sky, FLAT_R_M, EYE_STAND_M), MAP_FADE_M, FLAT_R_M)


MAP_BLOCK = 32              # the disk's cells are merged in square blocks of at most this many cells a side (16 m):
#                             a block all land or all water, wholly inside the disk, is one quad; only the edges of
#                             the water and of the disk keep the 0.5 m cells. (One quad a cell, the disk was 40k faces,
#                             the world went over the asset chain's budget and was thinned, and the thinned ground
#                             showed chevrons and streaks of the wrong part of its picture in every frame, 11-18)


def disk_blocks(wet: np.ndarray, inside: np.ndarray, top: int = MAP_BLOCK) -> list:
    """The disk's cells as square blocks [(row, column, side, wet)] (row 0 at the top of the map): each block of
    `top` cells split in four until it is all inside the disk and all of one kind (land or water), or one cell; a
    cell outside the disk is left out."""
    n = wet.shape[0]
    out = []

    def go(i, j, s):
        ins = inside[i:i + s, j:j + s]
        if not ins.any():
            return
        w = wet[i:i + s, j:j + s]
        if s == 1 or (ins.all() and (w.all() or not w.any())):
            out.append((i, j, s, bool(w[0, 0])))
            return
        h_ = s // 2
        for di, dj in ((0, 0), (0, h_), (h_, 0), (h_, h_)):
            go(i + di, j + dj, h_)
    for i in range(0, n, top):
        for j in range(0, n, top):
            go(i, j, min(top, n - i, n - j))
    return out


def map_disk(eye: list, n0: int, water: dict | None = None) -> tuple:
    """The set's real ground with a map: the disk within FLAT_R_M of the point under the eye as blocks of
    ground_map.WATER_CELL_M cells (disk_blocks), each showing its own part of the map (one picture over the whole disk,
    in the lower left of the shell's picture: a point's uv is its place on the map whatever block it is in); the
    map's water (ground_map) WATER_UNDER_M lower and showing the map's copy in the lower right (WATER_UV), faces of
    their own. -> (vertices in the asset frame, faces looking up, uv, how many faces are water)."""
    import ground_map as GM
    h, c = FLAT_R_M, GM.WATER_CELL_M
    n = GM.cells(h, c)
    wet = GM.mask_of(water) if water and water.get("runs") else np.zeros((n, n), bool)
    if wet.shape != (n, n):
        wet = np.zeros((n, n), bool)
    mid = (np.arange(n) + 0.5) * c - h
    inside = np.hypot(mid[None, :], mid[::-1][:, None]) <= h        # (row 0 is the top of the map: +y)
    z = eye[2] - EYE_STAND_M
    e = 2.0 / 2048
    vtx, uv, faces, wf = [], [], [], []
    at = {}                                          # (blocks share their corners: the land, and each water, is one
    #                                                   piece; a lone water quad was a "speck" the asset chain drops)

    def corner(gi, gj, w_):
        key = (gi, gj, w_)
        if key not in at:
            x, y = -h + gj * c, h - gi * c
            at[key] = n0 + len(vtx)
            vtx.append([eye[0] + x, eye[1] + y, z - (WATER_UNDER_M if w_ else 0.0)])
            uv.append([(0.5 if w_ else 0.0) + min(0.5 - e, max(e, (x / (2 * h) + 0.5) * 0.5)),
                       min(0.5 - e, max(e, (y / (2 * h) + 0.5) * 0.5))])
        return at[key]
    for i, j, s_, w_ in disk_blocks(wet, inside):
        q = [corner(i + s_, j, w_), corner(i + s_, j + s_, w_), corner(i, j + s_, w_), corner(i, j, w_)]
        (wf if w_ else faces).extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
    return np.array(vtx), np.array(faces + wf, dtype=int).reshape(-1, 3), np.array(uv), len(wf)


# the asset chain thins a mesh over its tier's budget (img3d/model_img3d.py TIER_TRIS, times its own factor): a world
# is made under BUDGET_SHARE of it by construction, so it is never thinned
WORLD_TIER = "hero"         # the tier img3d/shell_one.py gives a world
BUDGET_SHARE = 0.9


def tier_limit(tier: str = WORLD_TIER, src: Path | None = None) -> int:
    """The asset chain's triangle limit for a tier, read from model_img3d.py's own table and factor (that file runs
    inside Blender and cannot be imported here)."""
    import ast
    import re
    t = (src or Path(__file__).resolve().parent / "model_img3d.py").read_text()
    table = ast.literal_eval(re.search(r"TIER_TRIS\s*=\s*(\{[^}]*\})", t).group(1))
    m = re.search(r"TIER_TRIS\[[^\]]*\]\s*\*\s*([0-9.]+)", t)
    return int(table[tier] * (float(m.group(1)) if m else 1.0))


def fit_shell(build, grid: int, room: int, tries: int = 6) -> tuple:
    """The shell at the finest grid (at most `grid`) whose faces fit in `room`: build(grid) -> (vertices, faces, uv,
    eye); made again at a coarser grid (faces go as its square) until they fit. -> (the shell, the grid)."""
    g = int(grid)
    for _ in range(tries):
        res = build(g)
        n = len(res[1])
        if n <= room:
            return res, g
        g = max(16, int(g * (room / n) ** 0.5 * 0.97) // 2 * 2)
    return res, g


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
    far = min(max(r[i] - SPOT_BACK_M, 0.0), SPOT_MAX_M)
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
    far = min(max(near - SPOT_BACK_M, 0.0), SPOT_MAX_M)
    return {"stand": [round(float(u[0] * far), 3), round(float(u[1] * far), 3)],
            "toward": [round(float(u[0]), 4), round(float(u[1]), 4)], "near_m": round(near, 2),
            "sure": round(top, 3), "share": round(float(at.mean()), 5)}


def outdoor(picture: str, out: str, grid: int, device: str, spot: str = "", marks: dict | None = None,
            map_pic: str = "", map_facts: str = "") -> int:
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
    # the set's real ground (a flat round FLAT_R_M out, in the panorama's own ground seen from above) with the
    # panorama as what lies beyond it: nothing of the picture is built nearer than the ground's edge
    tile = ground_tile(rgb, d, sky)
    mf = json.loads(Path(map_facts).read_text()) if map_facts and Path(map_facts).exists() else {}
    water = mf.get("water") if mf.get("water") and float(mf.get("half_m") or 0) == FLAT_R_M else None
    with_map = bool(map_pic and Path(map_pic).exists())
    # the shell's grid is chosen so the whole world, ground included, stays under the asset chain's budget
    limit = tier_limit()
    n_ground = len((map_disk([0.0, 0.0, 0.0], 0, water) if with_map else ground_disk([0.0, 0.0, 0.0], 0))[1])
    bd = backdrop(d, sky)
    (vtx, faces, uv, eye), grid = fit_shell(lambda g: shell(bd, np.zeros(d.shape, bool), g, steep_deg=OUT_STEEP_DEG,
                                                            lowest=True), grid, int(BUDGET_SHARE * limit) - n_ground)
    uv = np.stack([uv[:, 0], 0.5 + uv[:, 1] / 2], -1)                 # (the panorama: the upper half of the picture)
    n_wet = 0
    if with_map:
        # the place's own ground drawn from above, over the whole disk (and its water a little under it)
        gv, gf, guv, n_wet = map_disk(eye, len(vtx), water)
        mt = map_texture(np.asarray(Image.open(map_pic).convert("RGB")), rgb, d, sky)
        low = np.concatenate([mt, mt], 1)
    else:
        # (no map: the panorama's own ground near the eye and its bare tile repeated, as before)
        gv, gf, guv = ground_disk(eye, len(vtx))
        gm = ground_map(rgb, d, sky, tile)
        t4 = np.concatenate([np.concatenate([tile, tile], 1)] * 2, 0)
        low = np.concatenate([np.asarray(Image.fromarray(gm).resize((1024, 1024), Image.LANCZOS)),
                              np.asarray(Image.fromarray(t4).resize((1024, 1024), Image.LANCZOS))], 1)
    vtx, faces, uv = np.concatenate([vtx, gv]), np.concatenate([faces, gf]), np.concatenate([uv, guv])
    im = Image.fromarray(np.concatenate([np.asarray(im.resize((2048, 1024), Image.LANCZOS)), low], 0))
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
             "faces": int(len(faces)), "faces_limit": limit, "under_budget": bool(len(faces) <= BUDGET_SHARE * limit),
             "shell_grid": int(grid), "ground_faces": int(len(gf)), "flat_r_m": FLAT_R_M, "sky_r_m": SKY_R_M, "land": info, "spot": spot or None,
             "spot_m": [round(eye[0] + sp[0], 3), round(eye[1] + sp[1], 3), round(eye[2] - EYE_STAND_M, 3)] if sp else None,
             "marks_m": found, "sun": sun_of(rgb, sky), "world_build": WORLD_BUILD,
             # the ground's map, its water and its land places: in the set's frame (the point under the eye at 0, 0)
             "ground_map": {"picture": str(map_pic), "half_m": FLAT_R_M, "fade_m": MAP_FADE_M} if map_pic and Path(map_pic).exists() else None,
             "water": water if n_wet else None, "places_m": mf.get("places") or {},
             "water_region": {"name": "water", "material": "water", "uv": WATER_UV, "under_m": WATER_UNDER_M,
                              "faces": int(n_wet)} if n_wet else None}
    o.with_suffix(".json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts))
    if not facts["under_budget"]:
        print(f"the world has {len(faces)} faces, over {BUDGET_SHARE} of the chain's {limit}: it will be thinned",
              file=sys.stderr)
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
    ap.add_argument("--map", default="", help="outdoors: the place's ground drawn from straight above (ground_map)")
    ap.add_argument("--map-facts", default="", help="outdoors: the vision judge's reading of that map (map.json)")
    a = ap.parse_args()
    if a.outdoor:
        return outdoor(a.picture, a.out, a.grid or OUT_GRID, a.device, a.spot,
                       dict(m.split("=", 1) for m in a.mark if "=" in m), a.map, a.map_facts)
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
