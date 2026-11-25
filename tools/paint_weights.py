"""How much each drawn view paints each point of a figure (numpy only: Blender's script and the tests share it).

11-24: the views were blended by cos^2 of the surface normal, so a cheek at 45 degrees was half the front drawing and
half the side drawing. Two drawings of one face never register to the millimetre: the blend was a double exposure
(two noses, two mouths), on every head. Now each point belongs almost wholly to ONE view: a high power makes the
change-over a narrow seam, and the front drawing - the one that carries the face - is weighted up, so it owns
everything it sees within about 42 degrees and the seam falls on the cheek's edge, in front of the ear.
"""

from __future__ import annotations

import numpy as np

VIEW_DIRS = {"front": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0), "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0)}
POWER = 8            # cos^8: 9:1 to 1:9 between two views within about 15 degrees
FRONT_BIAS = 4.0     # the front drawing's weight against the others


def view_weights(normals, names) -> np.ndarray:
    """(n, len(names)) un-normalised weights of the views `names` (keys of VIEW_DIRS) for unit `normals` (n, 3)."""
    nrm = np.asarray(normals, float)
    W = np.stack([np.clip(nrm @ np.array(VIEW_DIRS[v]), 0, None) ** POWER for v in names], 1)
    if "front" in names:
        W[:, list(names).index("front")] *= FRONT_BIAS
    return W


FACE_SPLIT = 0.8     # along the head's depth (0 = back of the head, 1 = tip of the nose): the side drawings stop here
                     # (the nose, the eye sockets, the mouth; the cheeks stay the side drawings' - painted from the
                     # front alone, a head seen in profile is the front drawing stretched)
FACE_SEAM = 0.12     # ...fading out over this share of the depth behind it


def face_from_front(W, names, co, head_frac: float) -> np.ndarray:
    """The side drawings take no part in the FACE (11-24: with one view per point the flank of the nose and the eye
    sockets - surfaces that look sideways - still took the side drawing: a stripe of profile down the middle of every
    woman's face). In the head band (the top `head_frac` of the figure, which faces +X) the side views' weights fall
    to nothing over the front of the face; what is left there is the front drawing. `co` = (n, 3) rest positions."""
    co = np.asarray(co, float)
    W = np.array(W, float)
    side = [i for i, v in enumerate(names) if v in ("left", "right")]
    if not side or "front" not in names or len(co) == 0:
        return W
    z0, z1 = co[:, 2].min(), co[:, 2].max()
    head = co[:, 2] >= z1 - head_frac * (z1 - z0)
    if head.sum() < 10:
        return W
    x0, x1 = np.percentile(co[head, 0], 1), np.percentile(co[head, 0], 99)
    depth = max(x1 - x0, 1e-6)
    t = (co[:, 0] - x0) / depth
    keep = np.clip((FACE_SPLIT - t) / FACE_SEAM, 0.0, 1.0)
    keep[~head] = 1.0
    W[:, side] *= keep[:, None]
    return W


LEG_BAND = (0.10, 0.40)   # of the figure's height: the legs, whose width across is the torso's
ARM_FROM = 0.45           # the arms are above this height ...
ARM_OUT = 1.15            # ... and farther out to the side than the legs' half-width x this
ARM_SEAM = 0.15           # the side drawings fade out over this share of the half-width beyond it


def arms_from_front_back(W, names, co) -> np.ndarray:
    """The side drawings take no part in the ARMS (11-25: the figure is shaped with its arms held out, the side
    drawings show the near arm hanging down, so on the arm the side drawing projected the wall behind the figure:
    a grey, streaked sleeve). Out past the torso's width (measured on the legs, which never reach out), above the
    hips, the side views' weights fall to nothing; the front and back drawings, which show the arms where the shape
    has them, paint them. `co` = (n, 3) rest positions, the figure facing +X, +Z up."""
    co = np.asarray(co, float)
    W = np.array(W, float)
    side = [i for i, v in enumerate(names) if v in ("left", "right")]
    if not side or len(co) == 0:
        return W
    z0, z1 = co[:, 2].min(), co[:, 2].max()
    h = max(z1 - z0, 1e-6)
    zr = (co[:, 2] - z0) / h
    legs = (zr >= LEG_BAND[0]) & (zr <= LEG_BAND[1])
    if legs.sum() < 10:
        return W
    mid = float(np.median(co[legs, 1]))
    half = float(np.percentile(np.abs(co[legs, 1] - mid), 98)) * ARM_OUT
    out = (np.abs(co[:, 1] - mid) - half) / max(ARM_SEAM * half, 1e-6)
    keep = 1.0 - np.clip(out, 0.0, 1.0)
    keep[zr < ARM_FROM] = 1.0
    W[:, side] *= keep[:, None]
    return W


EDGE_FRAC = 0.002    # of the picture's longer side: a cut-out's rim this deep is not trusted for colour
SOLID = 0.98         # alpha from which a cut-out pixel is the figure and nothing of the card behind it


def _shifted(a, dy, dx):
    """`a` moved by (dy, dx) with nothing wrapping round the picture's edges (zeros come in)."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def erode(mask, r: int) -> np.ndarray:
    m = np.asarray(mask, bool)
    for _ in range(int(r)):
        m = m & _shifted(m, 1, 0) & _shifted(m, -1, 0) & _shifted(m, 0, 1) & _shifted(m, 0, -1)
    return m


def dilate(mask, r: int) -> np.ndarray:
    m = np.asarray(mask, bool)
    for _ in range(int(r)):
        m = m | _shifted(m, 1, 0) | _shifted(m, -1, 0) | _shifted(m, 0, 1) | _shifted(m, 0, -1)
    return m


def bleed_seed(alpha) -> np.ndarray:
    """The pixels of a cut-out drawing whose colour is the figure's own (11-25: the rim of a cut-out is the figure
    mixed with the card it was drawn on, and darkened by the cut; the fill that carries colour out past the outline
    started from that rim, so every surface the drawing does not show squarely - the tops of the shoulders, the crown
    of the head, the edge of a sleeve - was painted the card's grey, in streaks). The seed is the solid figure less a
    rim of EDGE_FRAC of the picture; parts thinner than that rim (a lock of hair, a finger) keep their solid
    pixels, so they are not painted with their neighbour's colour. An image with no usable solid part
    falls back to alpha > 0.5."""
    a = np.asarray(alpha, np.float32)
    solid = a >= SOLID
    r = max(2, int(round(max(a.shape) * EDGE_FRAC)))
    deep = erode(solid, r)
    thin = solid & ~dilate(deep, 2 * r)
    seed = deep | thin
    return seed if seed.any() else a > 0.5


NEAR_FRAC = 0.01     # of the picture's longer side: this far out of the seed the colour is carried pixel by pixel
COARSE = 4           # ... beyond it, on a picture this many times smaller (the same reach in a sixtieth of the time)


def _spread(col, filled, iters):
    h, w = filled.shape
    for _ in range(int(iters)):
        if filled.all():
            break
        acc = np.zeros_like(col)
        cnt = np.zeros((h, w), np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh_f = _shifted(filled, dy, dx)
            acc += _shifted(col, dy, dx) * sh_f[..., None]
            cnt += sh_f
        new = (~filled) & (cnt > 0)
        if not new.any():
            break
        col[new] = acc[new] / cnt[new][:, None]
        filled |= new
    return col, filled


def bleed(col, seed) -> np.ndarray:
    """`col` (h, w, 3) with every pixel outside `seed` given the colour carried outward from the seed's edge
    (iterated neighbour mean: pixel by pixel near the seed, on a smaller picture farther out)."""
    col = np.array(col, np.float32)
    seed = np.array(seed, bool)
    h, w = seed.shape
    col[~seed] = 0
    if not seed.any():
        return col
    col, filled = _spread(col, seed.copy(), max(4, int(max(h, w) * NEAR_FRAC)))
    if not filled.all():
        c = COARSE
        H, Wd = -(-h // c) * c, -(-w // c) * c
        pc = np.zeros((H, Wd, 3), np.float32); pc[:h, :w] = col
        pf = np.zeros((H, Wd), np.float32); pf[:h, :w] = filled
        sc = pc.reshape(H // c, c, Wd // c, c, 3).sum((1, 3))
        sf = pf.reshape(H // c, c, Wd // c, c).sum((1, 3))
        cf = sf > 0
        cc = np.where(cf[..., None], sc / np.maximum(sf, 1)[..., None], 0).astype(np.float32)
        cc, cf = _spread(cc, cf, H // c + Wd // c)
        cc[~cf] = col[seed].mean(0)
        up = np.repeat(np.repeat(cc, c, 0), c, 1)[:h, :w]
        col[~filled] = up[~filled]
    return col


OFF_BAR = 0.50       # more of a figure than this painted from outside its drawings: the shape and the drawings disagree
                     # (11-25, one ordinary figure with four views: 0.26)


def off_drawing(W, inside) -> float:
    """The share of a figure's points whose colour comes from OUTSIDE the drawings: `W` (n, views) the normalised
    view weights, `inside` (n, views) whether the point lands on that drawing's seed. 0 = every point is painted by
    what a drawing shows of it."""
    W = np.asarray(W, float)
    if W.size == 0:
        return 0.0
    return float((W * (1.0 - np.asarray(inside, float))).sum() / max(W.sum(), 1e-9))
