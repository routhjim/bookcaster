"""A cabin people sit inside, as a shell seen from within: one panorama picture of its inside (equirectangular, taken
from the head of whoever sits in it) -> its depth (MoGe-2, a panorama split into views and merged) -> a mesh of
everything the picture shows, facing inward, with the panes of its windows made glass (CLIPSeg finds them), the
surface behind each near thing carried on behind it, and the picture as its texture.

    python pano_shell.py PICTURE OUT.glb [--grid 512] [--device cuda]

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


def depth_of(image: np.ndarray, device: str) -> np.ndarray:
    """The panorama's distance map in metres (H x W)."""
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
    d, _ = merge_panorama_depth(min(1920, w), min(960, h), dist, masks, ext, intr)
    return cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def glass_of(im, device: str) -> np.ndarray:
    """Where the picture shows a window's pane (H x W bool): the segmenter asked for "a window" on square views
    half over one another round the panorama, then whole panes kept or dropped by how sure it is of each."""
    import scipy.ndimage as ndi
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
            o = model(**proc(text=["a window", "a wall"], images=[tile, tile], return_tensors="pt", padding=True).to(device))
        p = torch.sigmoid(o.logits[0]).cpu().numpy()
        xs = np.arange(x0, x0 + h) % w
        acc[:, xs] += np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize((h, h), Image.BILINEAR)) / 255.0
        cnt[:, xs] += 1
    prob = acc / np.maximum(cnt, 1)
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


def shell(depth: np.ndarray, glass: np.ndarray, grid: int):
    """-> vertices (asset frame, z up, the middle of the footprint at the origin), faces (wound to face the eye), uv,
    the eye's place."""
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
    steep = 1.0 + (2 * np.pi / gw) * np.tan(np.radians(STEEP_DEG))
    q1 = quads[~flat(g).any(1) & (flat(d).max(1) < steep * flat(d).min(1))]
    qg = quads[flat(g).any(1)]                                        # the panes (their own smooth surface)
    # what is behind the near things: carried on from the far side of each jump, in the far side's own colour; where
    # the far side is glass it stays open
    fq = flat(fd)
    su = (sc + 0.5) / gw
    uq = flat(su)
    q2 = quads[(fq > 0).all(1) & (fq.max(1) < steep * fq.min(1)) & ~flat(g[sr, sc]).any(1) & ~flat(seed).all(1)
               & (uq.max(1) - uq.min(1) < 0.5)]
    n1 = idx.size
    vtx = np.concatenate([(wrap(dirs[..., k]) * wrap(gd)).reshape(-1) for k in range(3)]
                         + [(wrap(dirs[..., k]) * wrap(fd)).reshape(-1) for k in range(3)]).reshape(2, 3, -1)
    vtx = np.concatenate([vtx[0].T, vtx[1].T])
    uv1 = np.stack(np.meshgrid(np.clip(u, 0, 1), v), -1).reshape(-1, 2)
    uv2 = np.stack([wrap(su), wrap(1 - (sr + 0.5) / gh)], -1).reshape(-1, 2)
    uv = np.concatenate([uv1, uv2])
    quads = np.concatenate([q1, qg, q2 + n1])
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    c = vtx[faces]
    n = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    if (np.einsum("ij,ij->i", n, c.mean(1)) > 0).mean() > 0.5:       # the faces look at the eye (the origin)
        faces = faces[:, ::-1]
    used = np.unique(faces)                                           # (only the points a face stands on)
    remap = np.full(len(vtx), -1)
    remap[used] = np.arange(len(used))
    vtx, uv, faces = vtx[used], uv[used], remap[faces]
    floor = float(np.percentile(vtx[:, 2], 1))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picture")
    ap.add_argument("out")
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
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
             "seat": seat_under(eye), "regions": [GLASS_REGION]}
    out.with_suffix(".json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts))
    if facts["glass_share"] < 0.02:
        print("no window found in the picture", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
