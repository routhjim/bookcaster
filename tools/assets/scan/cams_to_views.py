"""OpenCV cameras (asset frame) -> the orbit skin stage's Blender camera list (orbit/bl_views.py, orbit/project.py).

    python -I cams_to_views.py CAMS.json MASK_DIR OUT.json [--size 1152 640] [--every 1] [--frame-from-path]

CAMS.json: {"frames": [{"i", "path", "c2w" (OpenCV camera-to-world), "K" (pixels of --size), "az"}]}, e.g. splat.py's
render/cams.json. MASK_DIR/m_###.png is indexed by position in that list. The view's "frame" is renamed
<index>_<clip><frame>.png in the clip-tag form project.py reads (clip = the frame's folder name, lower case).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cams")
ap.add_argument("masks")
ap.add_argument("out")
ap.add_argument("--size", type=int, nargs=2, default=[1152, 640])
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--link-dir", help="folder for clip-tagged links to the frames (project.py reads the tag)")
a = ap.parse_args()
W, H = a.size
fr = json.loads(Path(a.cams).read_text())["frames"]
flip = np.diag([1.0, -1.0, -1.0, 1.0])
views = []
ld = Path(a.link_dir) if a.link_dir else None
if ld:
    ld.mkdir(parents=True, exist_ok=True)
for k, f in enumerate(fr):
    if k % a.every and k != 0:
        continue
    K = np.array(f["K"])
    M = np.array(f["c2w"]) @ flip
    src = Path(f["path"])
    clip = src.parent.name.lower() if src.parent.name in ("L", "R") else "d"
    frame = str(src)
    if ld:
        link = ld / f"{k:03d}_{clip}{src.stem}.png"
        if not link.exists():
            link.symlink_to(src)
        frame = str(link)
    views.append({"name": f"v{k:03d}", "frame": frame, "mask": str(Path(a.masks) / f"m_{k:03d}.png"),
                  "az": f.get("az", 0.0), "size": [W, H], "matrix_world": M.tolist(), "sensor_width": 36.0,
                  "lens": float(K[0, 0] / W * 36.0), "shift_x": float((K[0, 2] - W / 2) / W),
                  "shift_y": float((H / 2 - K[1, 2]) / W), "fy_over_fx": float(K[1, 1] / K[0, 0])})
Path(a.out).write_text(json.dumps({"views": views}, indent=1))
print(f"{len(views)} views")
