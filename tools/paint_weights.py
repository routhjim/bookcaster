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
