"""Cut the subject out of a reference picture (a character sheet's figure, a creature, a prop drawn on
white) and set it on a plain mid-grey card, for drawers that copy a reference's background into the
scene: Qwen-Image put the knight on blank white beside a lone table, the sword floating, after the
prop and creature sheets drawn on white (10-19). rembg (U2-Net, CPU); the model downloads on first use.

    python tools/cutout.py in.png out.png [--model isnet-general-use]
"""

from __future__ import annotations

import argparse
import sys

from PIL import Image

GREY = (128, 128, 128)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--model", default="isnet-general-use")
    a = ap.parse_args()
    from rembg import new_session, remove

    im = Image.open(a.src).convert("RGB")
    cut = remove(im, session=new_session(a.model))            # RGBA, the subject kept
    card = Image.new("RGB", im.size, GREY)
    card.paste(cut, mask=cut.split()[-1])
    bbox = cut.split()[-1].point(lambda v: 255 if v > 24 else 0).getbbox()
    if bbox:                                                   # the subject with a margin, not the empty card
        pad = int(0.04 * max(im.size))
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(im.width, bbox[2] + pad),
                min(im.height, bbox[3] + pad))
        card = card.crop(bbox)
    card.save(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
