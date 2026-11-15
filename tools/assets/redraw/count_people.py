"""The people a drawn picture shows (GPU: Grounding DINO "person" boxes), for pictures with no staged scene to
say it.

    DETECT_DEVICE=cuda python -I tools/assets/redraw/count_people.py COUNT.json     (writes COUNT["out"])

COUNT: {"out": "/abs/people.json", "items": [{"id", "path": "/abs/x.png"}]}
-> {id: {"boxes": [[left, top, right, bottom], ...]}}: each 0-1 of the picture, left to right. A box inside a
bigger one (a head found again in its own body's box) is not a second person.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DETECT_DEVICE", "cuda")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # tools/ (detect.py)

MIN_AREA = 0.01             # as scenes/shot_people.py: a figure this small is the crowd's, not a person of the shot


def distinct(boxes: list) -> list:
    """The boxes that are people of their own, left to right."""
    def inside(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix > 0.8 * (a[2] - a[0]) * (a[3] - a[1])
    big = sorted((b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_AREA),
                 key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept = []
    for b in big:
        if not any(inside(b, k) for k in kept):
            kept.append(b)
    return sorted(kept, key=lambda b: (b[0] + b[2]) / 2)


def main():
    import torch
    from PIL import Image
    import detect as D
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: counting runs on a card, never on the CPU")
    cfg = json.loads(Path(sys.argv[1]).read_text())
    dp, dm, _, _ = D.load()
    out = {}
    for it in cfg["items"]:
        if not Path(it["path"]).exists():
            continue
        det = D.detect(dp, dm, Image.open(it["path"]).convert("RGB"), ["person"])
        out[it["id"]] = {"boxes": distinct(det.get("person", []))}
    Path(cfg["out"]).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
