"""Model script: Victorian mahogany chest of drawers (2 short over 4 graduated long), front -Y.

Carcass boards, drawer fronts with cock beads, bail handles and escutcheons (period_kit.bail_handle), a moulded
top (period_kit.moulding round three sides), plinth and bracket feet cut from outlines (period_kit.plate)."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

W, D, H = 0.94, 0.46, 1.08              # carcass
Z0 = 0.10                               # carcass bottom (feet + plinth below)
FY = -D / 2
wv = K.wood("mahogany", "Z", wear=0.5, gloss=0.55, grime=0.6)
wx = K.wood("mahogany", "X", wear=0.55, gloss=0.6, grime=0.6, offset=0.3)
wy = K.wood("mahogany", "Y", wear=0.4, gloss=0.5, grime=0.6, offset=0.6)
brass = K.metal("brass", tarnish=0.75, polish=0.55)
parts = []
for s in (-1, 1):
    parts.append(K.box(f"side{s}", (0.02, D, H - Z0), (s * (W / 2 - 0.01), 0, (H + Z0) / 2), bev=0.002, mat=wv))
parts.append(K.box("back", (W - 0.04, 0.012, H - Z0), (0, D / 2 - 0.006, (H + Z0) / 2), bev=0.001, mat=wv))
# top: slab with a moulded nosing on front and sides
parts.append(K.box("top", (W + 0.04, D + 0.02, 0.025), (0, -0.01, H + 0.0125), bev=0.002, mat=wx))
nose = [(0.0, 0.0), (-0.004, 0.0), (-0.010, 0.004), (-0.013, 0.010), (-0.014, 0.016), (-0.012, 0.021), (-0.008, 0.024),
        (0.0, 0.025)]
f = K.moulding("nose_f", nose, -(W + 0.04) / 2, (W + 0.04) / 2, 1, 1, wx)
f.location = (0, -0.01 - (D + 0.02) / 2, H)
parts.append(f)
for s in (-1, 1):
    m = K.moulding(f"nose_s{s}", nose, -(D + 0.02) / 2, (D + 0.02) / 2, 1 if s > 0 else 0, 0 if s > 0 else 1, wy)
    m.rotation_euler, m.location = (0, 0, s * math.pi / 2), (s * (W + 0.04) / 2, -0.01, H)
    parts.append(m)
# plinth moulding and bracket feet
parts.append(K.box("plinth", (W + 0.02, D + 0.01, 0.04), (0, -0.005, Z0 + 0.02 - 0.02), bev=0.004, seg=2, mat=wx))
foot = [(0.0, 0.0), (0.03, 0.0), (0.035, 0.03), (0.05, 0.05), (0.09, 0.06), (0.10, 0.08), (0.0, 0.08)]
for sx in (-1, 1):
    for sy in (-1, 1):
        xc, yc = sx * (W / 2 + 0.01), sy * (D / 2 + 0.005)            # outer corner
        a = K.plate(f"footf{sx}{sy}", [(xc - sx * px, pz) for px, pz in foot], 0.02, y=yc - sy * 0.01, mat=wx, bev=0.002)
        b = K.plate(f"foots{sx}{sy}", [(-sy * px, pz) for px, pz in foot], 0.02, y=0.0, mat=wy, bev=0.002)
        b.rotation_euler, b.location = (0, 0, math.pi / 2), (xc - sx * 0.01, yc, 0.0)
        parts += [a, b]
# drawers: 2 short over 4 graduated long, rails between
rows = [0.15, 0.17, 0.19, 0.21, 0.135]   # bottom -> top heights of the long drawers, then the short row
gap = 0.022
z = Z0 + 0.03
fronts = []
for i, h in enumerate(rows):
    if i < 4:
        fronts.append((0.0, W - 0.06, z, h))
    else:
        dw = (W - 0.06 - gap) / 2
        fronts += [(-(dw + gap) / 2, dw, z, h), ((dw + gap) / 2, dw, z, h)]
    parts.append(K.box(f"rail{i}", (W - 0.04, 0.02, gap), (0, FY + 0.01, z - gap / 2), bev=0.002, mat=wx))
    z += h + gap
parts.append(K.box("rail_top", (W - 0.04, 0.02, H - z + gap), (0, FY + 0.01, (H + z - gap) / 2), bev=0.002, mat=wx))
if len(fronts) == 6:
    parts.append(K.box("muntin", (gap, 0.02, rows[-1]), (0, FY + 0.01, fronts[-1][2] + rows[-1] / 2), bev=0.002, mat=wv))
for i, (cx, w, zb, h) in enumerate(fronts):
    fr = K.box(f"drawer{i}", (w, 0.022, h), (cx, FY - 0.001, zb + h / 2), bev=0.003, seg=2,
               mat=K.wood("mahogany", "X", 0.6, 0.6, 0.5, 0.1 * i))
    parts.append(fr)
    # cock bead round the front
    for (bx, bz, bw, bh) in ((cx, zb + h + 0.002, w + 0.006, 0.005), (cx, zb - 0.002, w + 0.006, 0.005),
                             (cx - w / 2 - 0.002, zb + h / 2, 0.005, h), (cx + w / 2 + 0.002, zb + h / 2, 0.005, h)):
        parts.append(K.box(f"bead{i}{bx:.2f}{bz:.2f}", (bw, 0.006, bh), (bx, FY - 0.010, bz), bev=0.0024, seg=2, mat=wx))
    hz = zb + h * 0.55
    if w > 0.6:
        for s in (-1, 1):
            parts += K.bail_handle(f"h{i}{s}", cx + s * w * 0.28, FY - 0.012, hz, 0.075, brass)
    else:
        parts += K.bail_handle(f"h{i}", cx, FY - 0.012, hz, 0.07, brass)
    esc = K.lathe(f"esc{i}", [(0.0001, 0.0), (0.009, 0.0), (0.008, 0.0015), (0.0001, 0.002)], seg=16, mat=brass)
    esc.data.transform(K.Matrix.Diagonal((1.0, 1.6, 1.0, 1.0)))
    esc.location, esc.rotation_euler = (cx, FY - 0.012, zb + h * 0.55 if w <= 0.6 else zb + h * 0.55), (math.pi / 2, 0, 0)
    if w <= 0.6:
        esc.location = (cx, FY - 0.012, zb + h * 0.80)
    parts.append(esc)
