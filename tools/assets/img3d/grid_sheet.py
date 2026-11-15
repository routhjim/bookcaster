"""Grid sheet: pictures in rows with captions, each tile letterboxed into a common cell (Pillow, milliseconds).

    python tools/assets/img3d/grid_sheet.py OUT.jpg --cols 4 [--cell 576x320] [--title TEXT] img.png:caption ... [/ ...]

A lone "/" ends a row early (the rest of the row stays empty). "-" is an empty cell.
"""

import argparse

from PIL import Image, ImageDraw, ImageFont


def font(px):
    for p in ("/usr/share/fonts/google-noto/NotoSans-Regular.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
              "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(p, px)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("items", nargs="+")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", default="576x320")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    cw, ch = (int(v) for v in a.cell.split("x"))
    rows, cur = [], []
    for it in a.items:
        if it == "/":
            rows.append(cur)
            cur = []
            continue
        cur.append(it)
        if len(cur) == a.cols:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    cap_h, top = 24, (34 if a.title else 0)
    W, H = a.cols * cw + (a.cols - 1) * 4, top + len(rows) * (ch + cap_h + 4)
    sheet = Image.new("RGB", (W, H), (28, 28, 30))
    d = ImageDraw.Draw(sheet)
    f, ft = font(15), font(20)
    if a.title:
        d.text((8, 6), a.title, fill=(240, 240, 240), font=ft)
    for r, row in enumerate(rows):
        for c, it in enumerate(row):
            if it == "-":
                continue
            path, _, cap = it.partition(":")
            im = Image.open(path).convert("RGB")
            s = min(cw / im.width, ch / im.height)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            x, y = c * (cw + 4), top + r * (ch + cap_h + 4)
            sheet.paste((60, 60, 62), (x, y, x + cw, y + ch))
            sheet.paste(im, (x + (cw - im.width) // 2, y + (ch - im.height) // 2))
            d.text((x + 4, y + ch + 3), cap, fill=(235, 235, 235), font=f)
    sheet.save(a.out, quality=90)


if __name__ == "__main__":
    main()
