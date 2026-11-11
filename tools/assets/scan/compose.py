"""Grid of pictures with labels on a grey ground (no GPU; a few small images).

    python -I compose.py OUT.jpg COLS "ROW LABEL=img1,img2,..." ... [--width 2400]
"""
import sys

from PIL import Image, ImageDraw

out, cols = sys.argv[1], int(sys.argv[2])
args = [x for x in sys.argv[3:] if not x.startswith("--width")]
width = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 2400
args = [x for x in args if not x.isdigit()]
rows = [x.split("=", 1) for x in args]
cw = width // cols
cells = []
for label, files in rows:
    ims = []
    for f in files.split(","):
        im = Image.open(f).convert("RGBA")
        ims.append(im)
    h = max(int(cw * im.height / im.width) for im in ims)
    cells.append((label, ims, h))
H = sum(h + 28 for _, _, h in cells)
sheet = Image.new("RGB", (width, H), (96, 96, 96))
d = ImageDraw.Draw(sheet)
y = 0
for label, ims, h in cells:
    d.text((8, y + 6), label, fill=(255, 255, 255))
    for k, im in enumerate(ims):
        r = min(cw / im.width, h / im.height)
        im2 = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
        bg = Image.new("RGBA", im2.size, (96, 96, 96, 255))
        bg.alpha_composite(im2)
        sheet.paste(bg.convert("RGB"), (k * cw + (cw - im2.width) // 2, y + 28 + (h - im2.height) // 2))
    y += h + 28
sheet.save(out, quality=90)
