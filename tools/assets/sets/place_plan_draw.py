"""Draw a solved room layout in plan (PIL only): walls, openings, pieces with their fronts, named places and the
camera pockets found for them. A picture to look at a layout with, before any engine build.

    python tools/assets/sets/place_plan_draw.py LAYOUT.json OUT.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PX = 110            # pixels per metre
PAD = 70
COL = {"seat": (196, 120, 88), "wall": (120, 140, 170), "free": (150, 170, 120), "flat": (225, 205, 170),
       "fire": (200, 80, 50), "on": (240, 220, 90)}


def draw(layout: dict, out, title: str = "") -> str:
    W, D, _ = layout["size_m"]
    w, h = int(W * PX) + 2 * PAD, int(D * PX) + 2 * PAD + 60
    im = Image.new("RGB", (w, h), (250, 248, 243))
    d = ImageDraw.Draw(im)

    def P(x, y):                                    # north (y up) at the top of the picture
        return (PAD + (x + W / 2) * PX, PAD + (D / 2 - y) * PX)

    def box(r, **kw):
        a, b = P(r[0], r[3]), P(r[2], r[1])
        d.rectangle([a[0], a[1], b[0], b[1]], **kw)

    box((-W / 2, -D / 2, W / 2, D / 2), outline=(40, 40, 40), width=5)
    for o in layout["openings"]:
        if o["kind"] == "fireplace":
            box(o["breast"], fill=(170, 160, 150), outline=(60, 60, 60))
        horiz = o["wall"] in ("north", "south")
        u0, u1 = o["from_m"], o["to_m"]
        y = D / 2 if o["wall"] == "north" else -D / 2
        x = W / 2 if o["wall"] == "east" else -W / 2
        a, b = (P(u0, y), P(u1, y)) if horiz else (P(x, u0), P(x, u1))
        c = {"window": (70, 150, 230), "door": (150, 100, 50), "fireplace": (200, 80, 50)}[o["kind"]]
        d.line([a, b], fill=c, width=11)
        if o["kind"] == "door" and o.get("swing"):
            box(o["swing"], outline=(190, 160, 120))
        m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        d.text((m[0] - 20, m[1] - 26 if o["wall"] == "north" else m[1] + 8), o["name"], fill=c)
    for p in sorted(layout["pieces"], key=lambda p: (not p.get("flat"), bool(p.get("on")))):
        kind = "flat" if p.get("flat") else "on" if p.get("on") else "fire" if p["kind"] == "fireplace" \
            else "seat" if p.get("seat") else "wall" if p.get("wall") else "free"
        r = p["rect"]
        if p.get("on"):
            c = P(*p["position"])
            d.ellipse([c[0] - 7, c[1] - 7, c[0] + 7, c[1] + 7], fill=COL["on"], outline=(90, 80, 20))
            continue
        box(r, fill=COL[kind], outline=(50, 50, 50), width=2)
        c, f = P(*p["position"]), p["facing"]
        if not p.get("flat"):                       # a tick where its front points
            ext = min(r[2] - r[0], r[3] - r[1]) / 2
            e = P(p["position"][0] + f[0] * ext, p["position"][1] + f[1] * ext)
            d.line([c, e], fill=(20, 20, 20), width=3)
            d.ellipse([e[0] - 4, e[1] - 4, e[0] + 4, e[1] + 4], fill=(20, 20, 20))
        d.text((P(r[0], r[3])[0] + 3, P(r[0], r[3])[1] + 2), p["name"], fill=(15, 15, 15))
    for k in layout.get("camera_pockets", []):
        c = P(*k["position"])
        ok = all(s["valid"] > 0 for s in k["shots"].values())
        for name, s in k["shots"].items():
            col = (30, 150, 60) if name == "medium" else (150, 60, 170)
            for pt in s["points"]:
                q = P(*pt)
                d.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=col)
        d.ellipse([c[0] - 8, c[1] - 8, c[0] + 8, c[1] + 8], outline=(20, 110, 40) if ok else (220, 0, 0), width=3)
        e = (c[0] + k["facing"][0] * 22, c[1] - k["facing"][1] * 22)
        d.line([c, e], fill=(20, 110, 40) if ok else (220, 0, 0), width=2)
        d.text((c[0] + 10, c[1] + 6), k["place"], fill=(0, 90, 30) if ok else (200, 0, 0))
    y = h - 56
    d.text((PAD, 12), title or f"{layout.get('kind')}  {W} x {D} m  search: {layout.get('search')}", fill=(0, 0, 0))
    st = layout.get("status") or ("ok" if layout.get("ok") else "NOT ok")
    d.text((PAD, y), f"{st}   cost {layout.get('cost', {}).get('total')}   "
                     f"north is up; a tick shows a front; rings are named places with camera points "
                     f"(green medium, purple close)", fill=(0, 0, 0))
    if layout.get("unmet"):
        d.text((PAD, y + 16), "unmet: " + "; ".join(layout["unmet"])[:160], fill=(200, 0, 0))
    terms = layout.get("cost", {}).get("terms", {})
    d.text((PAD, y + 32), "  ".join(f"{k} {v}" for k, v in terms.items())[:190], fill=(70, 70, 70))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return str(out)


def main():
    lay = json.loads(Path(sys.argv[1]).read_text())
    print(draw(lay.get("layout", lay), sys.argv[2]))


if __name__ == "__main__":
    main()
