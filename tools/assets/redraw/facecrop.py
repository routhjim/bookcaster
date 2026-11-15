"""A person's face cut from an accepted redraw, as the cast's "face_ref" for every later shot (redraw.py faces()).

    python tools/assets/redraw/facecrop.py SCENE_OUT CAMERA KEY REDRAW.png OUT.png [--pad 0.9]

Where the face is comes from the scene tool's own frame check for that still (result.json: head top and chin of the
subject or of the others in frame, as fractions of the frame), so no detector runs: a square around head top to
chin, widened by --pad head heights on each side, cut out of the redraw (a slice of the picture, no resampling).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def head_points(res: dict, cam: str, key: str):
    for sh in res.get("shots", []):
        if sh.get("name") != cam:
            continue
        oc = sh.get("others_check") or {}        # the camera's subject is in frame_check, everyone else here
        fc = oc.get(key) or (sh.get("frame_check") if key in (sh.get("truth") or {}) else None) or {}
        if fc.get("head_top") and fc.get("chin"):
            return fc["head_top"], fc["chin"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("cam")
    ap.add_argument("key")
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--pad", type=float, default=0.9)
    a = ap.parse_args()
    res = json.loads((Path(a.scene) / "result.json").read_text())
    pts = head_points(res, a.cam, a.key)
    if not pts:
        raise SystemExit(f"no head points for {a.key} in {a.cam}")
    (tx, ty, _), (cx, cy, _) = pts
    im = Image.open(a.image)
    w, h = im.size
    hh = max(8.0, (cy - ty) * h)                      # head top to chin in pixels
    side = hh * (1 + 2 * a.pad)
    mx, my = (tx + cx) / 2 * w, (ty + cy) / 2 * h
    box = [int(max(0, mx - side / 2)), int(max(0, my - side / 2)), int(min(w, mx + side / 2)),
           int(min(h, my + side / 2))]
    im.crop(box).save(a.out)
    print(a.out, box)


if __name__ == "__main__":
    main()
