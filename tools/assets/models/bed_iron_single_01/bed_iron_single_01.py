"""Model script: Victorian iron single bedstead with mattress, sheet, pillow and a draped blanket. Foot to -Y.

Iron: lathe posts, swept tube rails and bars. Blanket: a flat grid mapped over the mattress box (the overhang hangs
straight down, corners fall diagonally), wrinkled with sines, given thickness by Solidify."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bmesh  # noqa: E402

import period_kit as K  # noqa: E402

PX, PY = 0.47, 0.99                 # post centres (x +-PX, y +-PY)
HEAD_H, FOOT_H = 1.12, 0.84
iron = K.metal("cast_iron", tarnish=0.4, polish=0.45, rust=0.25)
brass = K.metal("brass", tarnish=0.8, polish=0.5)
linen = K.fabric("linen", (0.50, 0.48, 0.43), wear=0.5)
wool = K.fabric("wool", (0.24, 0.19, 0.12), wear=0.4, scale=1.5)
parts = []

# ---- posts with collars and finials, castors
for sy, top in ((1, HEAD_H), (-1, FOOT_H)):
    for sx in (-1, 1):
        ks = [(0.0, 0.0001, "c"), (0.0, 0.014, "c"), (0.03, 0.016), (0.045, 0.0175, "l"), (top - 0.02, 0.0175, "l"),
              (top - 0.01, 0.021), (top, 0.0001, "c")]
        p = K.lathe(f"post{sx}{sy}", K.keys(ks, 0.004), seg=16, mat=iron)
        p.location = (sx * PX, sy * PY, 0.0)
        cas = K.lathe(f"castor{sx}{sy}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.016, "c"), (0.008, 0.019), (0.03, 0.019),
                                                  (0.032, 0.0001, "c")], 0.002), seg=16, mat=brass)
        cas.location = (sx * PX, sy * PY, 0.0)
        fin = K.lathe(f"finial{sx}{sy}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.012, "c"), (0.006, 0.014),
                                                  *K.bead(0.012, 0.022, 0.008), (0.02, 0.011), (0.032, 0.030),
                                                  (0.050, 0.030), (0.066, 0.012), (0.072, 0.0001)], 0.002), seg=20,
                        mat=brass)
        fin.location = (sx * PX, sy * PY, top - 0.005)
        for zc in (0.40, top * 0.72):
            col = K.lathe(f"collar{sx}{sy}{zc:.2f}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.021, "c"), *K.bead(0.012, 0.025, 0.016),
                                                             (0.024, 0.021), (0.024, 0.0001, "c")], 0.002), seg=16, mat=brass)
            col.location = (sx * PX, sy * PY, zc - 0.012)
            parts.append(col)
        parts += [p, cas, fin]

# ---- head and foot frames: arched top rail, cross rails, bars
for sy, top in ((1, HEAD_H), (-1, FOOT_H)):
    y = sy * PY
    arch = [(K.lerp(-PX, PX, k / 30), y, top - 0.06 + 0.07 * math.sin(math.pi * k / 30)) for k in range(31)]
    parts.append(K.tube(f"arch{sy}", arch, 0.011, seg=12, mat=iron))
    rails = (0.42, top * 0.72) if sy > 0 else (0.42, top * 0.72)
    for z in rails:
        parts.append(K.tube(f"rail{sy}{z:.2f}", [(-PX, y, z), (PX, y, z)], 0.0095, seg=12, mat=iron))
    nb = 9
    for k in range(1, nb):
        x = K.lerp(-PX, PX, k / nb)
        ztop = top - 0.06 + 0.07 * math.sin(math.pi * (x + PX) / (2 * PX))
        parts.append(K.tube(f"bar{sy}{k}", [(x, y, rails[0]), (x, y, ztop)], 0.0055, seg=8, mat=iron))
    # small scrolls between the upper rail and the arch
    for k in (2, 6):
        x0 = K.lerp(-PX, PX, (k + 0.5) / nb)
        zc = (rails[1] + top - 0.04) / 2
        sc = [(x0 + 0.03 * math.cos(a) * (1 - a / 12), y, zc + 0.03 * math.sin(a) * (1 - a / 12)) for a in [i * 0.35 for i in range(28)]]
        parts.append(K.tube(f"scroll{sy}{k}", sc, 0.004, seg=6, mat=iron))

# ---- side rails (angle iron) and slats
for sx in (-1, 1):
    parts.append(K.box(f"side{sx}", (0.035, 2 * PY, 0.005), (sx * (PX - 0.02), 0, 0.395), bev=0.0, mat=iron))
    parts.append(K.box(f"sidev{sx}", (0.005, 2 * PY, 0.035), (sx * (PX - 0.0025 - 0.005), 0, 0.378), bev=0.0, mat=iron))

# ---- mattress, pillow
MT = 0.57
mat_ = K.rbox("mattress", (0.90, 1.94, 0.17), r=0.04, spacing=0.04, bulge=(0.01, 0.0, 0.012), loc=(0, 0.0, 0.40 + 0.085),
              mat=linen)
parts.append(mat_)
pil = K.rbox("pillow", (0.70, 0.40, 0.14), r=0.06, spacing=0.02, bulge=(0.02, 0.05, 0.05), sag=0.02,
             loc=(0, PY - 0.25, MT + 0.05), rot=(0.25, 0, 0.03), mat=linen)
K.warp(pil, lambda c: (c.x, c.y, c.z * (1 - 0.5 * (abs(c.x) / 0.35) ** 2)))
parts.append(pil)

# ---- blanket draped over the lower two thirds
A, Y0, Y1 = 0.455, -0.97, 0.30           # half width on top, foot edge, head edge (on the mattress)
nu, nv = 70, 110
U = 0.80                                  # flat half width -> 0.345 hangs over each side
bm = bmesh.new()
grid = []
for j in range(nv + 1):
    v = K.lerp(Y0 - 0.12, Y1, j / nv)
    row = []
    for i in range(nu + 1):
        u = K.lerp(-U, U, i / nu)
        dx = max(abs(u) - A, 0.0)
        dy = max(Y0 - v, 0.0)
        hang = math.hypot(dx, dy)
        x = math.copysign(min(abs(u), A) + 0.012 + 0.03 * K.smooth(hang / 0.2) * (dx > 0), u)
        y = max(v, Y0) - 0.012 * (dy > 0)
        z = MT + 0.012 - hang
        # wrinkles: long folds across the top, deeper near the head edge where it is turned back
        w = 0.010 * math.sin(u * 17 + v * 3) + 0.007 * math.sin(v * 23 + u * 5) + 0.012 * math.sin(u * 9 - v * 6) ** 3
        if hang > 0:
            x += math.copysign(0.012 * math.sin(v * 31) ** 2, u) * (dx > 0)
            y += 0.0
        z += w * (1.0 if hang == 0 else 0.4) + (0.025 * K.smooth((v - Y1 + 0.15) / 0.15) if hang == 0 else 0.0)
        row.append(bm.verts.new((x, y, max(z, MT - 0.33))))
    grid.append(row)
for j in range(nv):
    for i in range(nu):
        bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
bl = K.from_bm("blanket", bm)
so = bl.modifiers.new("solid", "SOLIDIFY")
so.thickness, so.offset = 0.008, 1.0
K.shade(bl, 70)
K.assign(bl, wool)
parts.append(bl)
