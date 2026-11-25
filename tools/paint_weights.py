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
