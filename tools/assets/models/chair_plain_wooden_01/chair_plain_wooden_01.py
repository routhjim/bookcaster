"""Model script: plain oak chair, front -Y: solid dished seat, tapered square legs, stretchers, curved back slats."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

SZ, ST = 0.45, 0.032
oz = K.wood("oak", "Z", wear=0.7, gloss=0.25, grime=0.8)
ox = K.wood("oak", "X", wear=0.7, gloss=0.25, grime=0.8, offset=0.2)
oy = K.wood("oak", "Y", wear=0.6, gloss=0.25, grime=0.8, offset=0.5)
parts = []
LX, LYF, LYB = 0.185, -0.185, 0.175

# seat: rounded slab, dished, front corners rounded
seat = K.rbox("seat", (0.44, 0.42, ST), r=0.012, spacing=0.02, loc=(0, -0.015, SZ - ST / 2), mat=ox)
K.warp(seat, lambda c: (c.x, c.y, c.z - 0.008 * max(0.0, 1 - (c.x / 0.2) ** 2) * max(0.0, 1 - ((c.y + 0.02) / 0.19) ** 2)
                        * (1 if c.z > 0 else 0)))
parts.append(seat)


def leg(name, x, y, h, top_w=0.036, bot_w=0.028, rake=0.0):
    b = K.box(name, (top_w, top_w, h), (x, y, h / 2), bev=0.003, mat=oz)
    K.warp(b, lambda c: (c.x * K.lerp(bot_w / top_w, 1.0, (c.z + h / 2) / h), c.y * K.lerp(bot_w / top_w, 1.0, (c.z + h / 2) / h)
                         + rake * (0.5 - (c.z + h / 2) / h), c.z))
    return b


for s in (-1, 1):
    parts.append(leg(f"fleg{s}", s * LX, LYF, SZ - ST + 0.001))
    # back legs continue up as the back posts, raked back above the seat
    bl = leg(f"bleg{s}", s * LX, LYB, SZ - ST + 0.001, rake=0.02)
    parts.append(bl)
    post = []
    for k in range(13):
        t = k / 12
        post.append((s * LX * K.lerp(1.0, 1.04, t), LYB + 0.01 + 0.07 * t ** 1.3, K.lerp(SZ - ST - 0.01, 0.88, t)))
    secs = [K.superellipse2d(K.lerp(0.017, 0.014, k / 12), 0.016, 3.5, 0.0, 12) for k in range(13)]
    parts.append(K.sweep(f"post{s}", post, secs, up=(0, -1, 0), mat=oz, angle=40))
    parts.append(K.box(f"sstr{s}", (0.022, LYB - LYF, 0.026), (s * LX, (LYF + LYB) / 2, 0.16), bev=0.003, mat=oy))
parts.append(K.box("fstr", (2 * LX, 0.022, 0.026), (0, LYF, 0.22), bev=0.003, mat=ox))
parts.append(K.box("bstr", (2 * LX, 0.022, 0.026), (0, LYB, 0.22), bev=0.003, mat=ox))
# seat rails under the seat
parts.append(K.box("frail", (2 * LX, 0.02, 0.06), (0, LYF, SZ - ST - 0.03), bev=0.002, mat=ox))
parts.append(K.box("brail", (2 * LX, 0.02, 0.06), (0, LYB, SZ - ST - 0.03), bev=0.002, mat=ox))
for s in (-1, 1):
    parts.append(K.box(f"srail{s}", (0.02, LYB - LYF, 0.06), (s * LX, (LYF + LYB) / 2, SZ - ST - 0.03), bev=0.002, mat=oy))


def slat(name, z, h, t=0.018, bow=0.03):
    pts, secs = [], []
    for k in range(21):
        u = k / 20
        x = K.lerp(-LX * 1.03, LX * 1.03, u)
        zz = z + (z - SZ) / (0.88 - SZ)
        frac = (z - SZ + ST) / (0.88 - SZ + ST)
        y = LYB + 0.01 + 0.07 * frac ** 1.3 + bow * math.sin(math.pi * u)
        pts.append((x, y, z))
        secs.append(K.superellipse2d(t / 2, h / 2, 4.0, 0.0, 12))
    return K.sweep(name, pts, secs, up=(0, 0, 1), mat=ox, angle=40)


parts.append(slat("toprail", 0.84, 0.07, 0.022))
parts.append(slat("slat1", 0.73, 0.045))
parts.append(slat("slat2", 0.63, 0.045))
