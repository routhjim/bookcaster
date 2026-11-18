"""Camera path and per-frame consistency as one picture (no GPU): top view of the camera centres round the subject
(asset frame, metres), coloured by clip, and per-frame curves against azimuth: mask agreement, depth error, and
(after splat.py) photometric error and camera correction.

    python -I path_sheet.py FUSE_DIR OUT.png [--fit FIT.json]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fuse")
ap.add_argument("out")
ap.add_argument("--fit")
a = ap.parse_args()
fj = json.loads((Path(a.fuse) / "frame.json").read_text())
cj = json.loads((Path(a.fuse) / "consistency.json").read_text())
fr = fj["frames"]
c = np.array([f["c2w"] for f in fr])[:, :3, 3]
clip = np.array([Path(f["path"]).parent.name for f in fr])
az = np.array([x["az"] for x in cj["frames"]])
kept = np.array([x["kept"] for x in cj["frames"]])
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 2, 1)
for name, colr in (("L", "tab:blue"), ("R", "tab:orange")):
    s = clip == name
    ax.plot(c[s, 0], c[s, 1], ".", color=colr, label=f"clip {name} ({s.sum()})")
s = ~np.isin(clip, ["L", "R"])
ax.plot(c[s, 0], c[s, 1], "*", color="k", ms=14, label="drawing")
ax.plot(c[~kept, 0], c[~kept, 1], "x", color="r", ms=8, label="dropped")
ex = cj["extent_m"]
ax.add_patch(matplotlib.patches.Rectangle((-ex[0] / 2, -ex[1] / 2), ex[0], ex[1], color="g", alpha=0.5,
                                          label=f"subject box {ex[0]:.1f} x {ex[1]:.1f} m (head at -Y)"))
ax.set_aspect("equal")
ax.legend()
ax.set_title("camera centres, top view (m)")
for k, (key, lab) in enumerate((("mask_agree", "fraction of the frame's points inside the other frames' masks"),
                                ("depth_err_rel", "depth disagreement with the 4 nearest views (relative)"))):
    ax = fig.add_subplot(4, 2, 2 + 2 * k)
    y = np.array([x[key] for x in cj["frames"]])
    for name, colr in (("L", "tab:blue"), ("R", "tab:orange")):
        s = clip == name
        ax.plot(az[s], y[s], ".", color=colr)
    ax.plot(az[~kept], y[~kept], "x", color="r")
    ax.set_title(lab, fontsize=9)
if a.fit:
    fit = json.loads(Path(a.fit).read_text())["frames"]
    faz = np.array([x["az"] for x in fit])
    fcl = np.array([Path(fr[x["i"]]["path"]).parent.name for x in fit])
    for k, (key, lab) in enumerate((("l1_in_mask", "splat fit: colour error inside the mask (0-1)"),
                                    ("rot_fix_deg", "splat fit: camera rotation correction (deg)"))):
        ax = fig.add_subplot(4, 2, 6 + 2 * k)
        y = np.array([x[key] for x in fit])
        for name, colr in (("L", "tab:blue"), ("R", "tab:orange")):
            s = fcl == name
            ax.plot(faz[s], y[s], ".", color=colr)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("azimuth (deg, drawing = 0)")
fig.tight_layout()
fig.savefig(a.out, dpi=80)
