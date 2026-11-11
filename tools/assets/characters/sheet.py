"""Side-by-side sheet: reference pictures next to renders, each scaled to a common height, with captions.

    python tools/assets/characters/sheet.py out.png --height 640 img1.png:caption img2.png:caption ...

Pasting and scaling only (Pillow), a few milliseconds per sheet.
"""

import argparse

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("items", nargs="+")
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--crop", action="append", default=[], help="index:x0,y0,x1,y1 in source pixels")
    a = ap.parse_args()
    crops = {int(c.split(":")[0]): tuple(int(v) for v in c.split(":")[1].split(",")) for c in a.crop}
    tiles = []
    for i, it in enumerate(a.items):
        path, _, cap = it.partition(":")
        im = Image.open(path).convert("RGB")
        if i in crops:
            im = im.crop(crops[i])
        w = round(im.width * a.height / im.height)
        tiles.append((im.resize((w, a.height), Image.LANCZOS), cap))
    W = sum(t.width for t, _ in tiles) + 8 * (len(tiles) - 1)
    sheet = Image.new("RGB", (W, a.height + 26), (245, 245, 240))
    d = ImageDraw.Draw(sheet)
    x = 0
    for t, cap in tiles:
        sheet.paste(t, (x, 26))
        d.text((x + 6, 6), cap, fill=(20, 20, 20))
        x += t.width + 8
    sheet.save(a.out)


if __name__ == "__main__":
    main()
