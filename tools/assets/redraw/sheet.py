"""Contact sheets for the redraw test (GPU resize; Pillow only draws the labels and boxes and writes the file).

    python -I tools/assets/redraw/sheet.py SPEC.json
    SPEC: {"out": "/abs/sheet.jpg", "cols": 4, "tile_w": 384, "title": "optional line(s) across the top",
           "tiles": [{"image": "/abs/a.png", "label": "text", "boxes": [[x0, y0, x1, y1, "label", "#rrggbb"], ...]}]}
A tile with "image": null is left blank (its label still shows). Device: cuda only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def load_resized(path: str, w: int, h: int, dev) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"))
    t = torch.from_numpy(a).to(dev).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
    return t[0].permute(1, 2, 0).clamp(0, 255).round().byte().cpu().numpy()


def make(spec: dict, dev) -> str:
    cols, tw = int(spec.get("cols", 4)), int(spec.get("tile_w", 384))
    th = round(tw * 640 / 1152)
    lab = 30
    tiles = spec["tiles"]
    rows = (len(tiles) + cols - 1) // cols
    title = [x for x in str(spec.get("title") or "").split("\n") if x]
    top = 14 * len(title) + (8 if title else 0)
    W, H = cols * tw + (cols - 1) * 4, top + rows * (th + lab) + (rows - 1) * 4
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    for k, line in enumerate(title):
        d.text((4, 4 + 14 * k), line, fill=(255, 212, 0))
    for i, t in enumerate(tiles):
        x, y = (i % cols) * (tw + 4), top + (i // cols) * (th + lab + 4)
        if t.get("image") and Path(t["image"]).exists():
            canvas.paste(Image.fromarray(load_resized(t["image"], tw, th, dev)), (x, y + lab))
        for b in t.get("boxes", []):
            x0, y0, x1, y1 = b[:4]
            col = b[5] if len(b) > 5 else "#ffd400"
            d.rectangle([x + x0 * tw, y + lab + y0 * th, x + x1 * tw - 1, y + lab + y1 * th - 1], outline=col, width=2)
            if len(b) > 4 and b[4]:
                d.text((x + x0 * tw + 3, y + lab + y0 * th + 2), str(b[4]), fill=col)
        for k, line in enumerate(str(t.get("label", "")).split("\n")[:2]):
            d.text((x + 3, y + 2 + 13 * k), line, fill=(235, 235, 230))
    out = Path(spec["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)
    return str(out)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: sheets resize on a card, never on the CPU")
    for a in sys.argv[1:]:
        print(make(json.loads(Path(a).read_text()), torch.device("cuda")), flush=True)


if __name__ == "__main__":
    main()
