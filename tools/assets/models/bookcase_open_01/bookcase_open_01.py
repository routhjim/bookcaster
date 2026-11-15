"""Model script: Victorian mahogany open bookcase on a two-door cupboard base, filled with books, front -Y.

Carcass from boxes (wood grain along each board's long side), ogee cornice and plinth mouldings mitred round three
sides, framed doors with fielded panels (period_kit.frame_panel), turned knobs, books from period_kit.book_row."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

W, D, H = 0.96, 0.34, 2.0          # carcass (cornice and ledge project beyond)
T = 0.022                           # board thickness
YF, YB = -D / 2, D / 2              # front / back faces
CUP = 0.82                          # cupboard top
SHELVES = [0.84, 1.11, 1.38, 1.65]
TOP = 1.90
wv = K.wood("mahogany", "Z", wear=0.45, gloss=0.55, grime=0.6)
wx = K.wood("mahogany", "X", wear=0.45, gloss=0.55, grime=0.6, offset=0.2)
wy = K.wood("mahogany", "Y", wear=0.4, gloss=0.5, grime=0.6, offset=0.4)
back_m = K.wood("mahogany", "Z", wear=0.1, gloss=0.3, grime=0.8, offset=0.6)
brass = K.metal("brass", tarnish=0.7, polish=0.5)
parts = []

# ---- carcass
for s in (-1, 1):
    parts.append(K.box(f"side{s}", (T, D, TOP - 0.10), (s * (W / 2 - T / 2), 0, 0.10 + (TOP - 0.10) / 2), bev=0.002, mat=wv))
parts.append(K.box("back", (W - 2 * T, 0.012, TOP - 0.10), (0, YB - 0.006, 0.10 + (TOP - 0.10) / 2), bev=0.001, mat=back_m))
# back panelling: vertical boards with V-joints shown as thin dark grooves
for k in range(1, 6):
    x = -W / 2 + T + k * (W - 2 * T) / 6
    parts.append(K.box(f"groove{k}", (0.004, 0.003, TOP - 0.11), (x, YB - 0.0125, 0.10 + (TOP - 0.10) / 2), bev=0.0, mat=back_m))
parts.append(K.box("bottom", (W - 2 * T, D - 0.012, T), (0, -0.006, 0.10 + T / 2), bev=0.001, mat=wx))
parts.append(K.box("top", (W - 2 * T, D - 0.012, T), (0, -0.006, TOP - T / 2), bev=0.001, mat=wx))
# cupboard: ledge projecting at the front and sides
parts.append(K.box("ledge", (W + 0.04, D + 0.03, 0.028), (0, -0.015, CUP + 0.014), bev=0.004, seg=3, mat=wx))
for i, z in enumerate(SHELVES[1:]):
    parts.append(K.box(f"shelf{i}", (W - 2 * T, D - 0.03, T), (0, 0.0, z - T / 2), bev=0.002, mat=wx))
    # shelf lip (thin moulded front edge)
    parts.append(K.box(f"lip{i}", (W - 2 * T, 0.008, 0.03), (0, YF + 0.019, z - 0.012), bev=0.003, seg=2, mat=wx))
# face frame stiles of the upper case
for s in (-1, 1):
    parts.append(K.box(f"fstile{s}", (0.04, 0.02, TOP - CUP - 0.028), (s * (W / 2 - 0.02), YF - 0.008, (TOP + CUP + 0.028) / 2), bev=0.002, mat=wv))
parts.append(K.box("frail_top", (W - 0.08, 0.02, 0.06), (0, YF - 0.008, TOP - 0.03), bev=0.002, mat=wx))

# ---- cupboard doors (two, framed, fielded panel), with a meeting stile bead and knobs
DW = (W - 0.04) / 2
for s in (-1, 1):
    parts += K.frame_panel(f"door{s}", DW - 0.003, CUP - 0.115, 0.022, (s * DW / 2, YF - 0.006, 0.112), wv, wx, wy,
                           stile=0.055, rail=0.065, raise_w=0.035)
    kn = K.lathe(f"knob{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.007, "c"), (0.012, 0.006), (0.02, 0.013),
                                      (0.028, 0.014), (0.034, 0.0001)], 0.0015), seg=20, mat=wv)
    kn.location, kn.rotation_euler = (s * 0.05, YF - 0.017, 0.52), (math.pi / 2, 0, 0)
    parts.append(kn)
    esc = K.box(f"escutcheon{s}", (0.012, 0.003, 0.022), (s * 0.05, YF - 0.0175, 0.46), bev=0.001, seg=1, mat=brass)
    parts.append(esc)
parts.append(K.box("rail_under_ledge", (W - 0.02, 0.02, 0.04), (0, YF - 0.006, CUP - 0.02), bev=0.002, mat=wx))
parts.append(K.box("rail_low", (W - 0.02, 0.02, 0.015), (0, YF - 0.006, 0.1045), bev=0.002, mat=wx))

# ---- plinth with a moulded top, cornice with an ogee, both mitred round the front and sides
pl = [(0.0, 0.0), (-0.012, 0.0), (-0.012, 0.085), (-0.004, 0.085), (-0.004, 0.10), (0.0, 0.10)]
parts.append(K.moulding("plinth_f", pl, -W / 2, W / 2, 1, 1, wx))
for s in (-1, 1):
    m = K.moulding(f"plinth_s{s}", pl, -D / 2 if s > 0 else -D / 2, D / 2, 1 if s > 0 else 0, 0 if s > 0 else 1, wy)
    m.rotation_euler = (0, 0, s * math.pi / 2)
    m.location = (s * W / 2, 0.0, 0.0)
    parts.append(m)
pl_f = parts[-3]
pl_f.location = (0, YF, 0)
cor = K.ogee(0.05, 0.10, 12, base=0.0)
cf = K.moulding("cornice_f", cor, -W / 2, W / 2, 1, 1, wx)
cf.location = (0, YF, TOP)
parts.append(cf)
for s in (-1, 1):
    m = K.moulding(f"cornice_s{s}", cor, -D / 2, D / 2, 1 if s > 0 else 0, 0 if s > 0 else 1, wy)
    m.rotation_euler = (0, 0, s * math.pi / 2)
    m.location = (s * W / 2, 0.0, TOP)
    parts.append(m)
parts.append(K.box("cornice_cap", (W + 0.10, D + 0.05, 0.012), (0, -0.025, TOP + 0.094), bev=0.003, mat=wx))
parts.append(K.box("cornice_back", (W, 0.012, 0.088), (0, YB - 0.006, TOP + 0.044), bev=0.001, mat=back_m))

# ---- books on the four open shelves
for i, z in enumerate(SHELVES):
    zz = z if i else CUP + 0.028
    parts += K.book_row(f"bk{i}", -W / 2 + T + 0.005, W / 2 - T - 0.005, YB - 0.014, zz, D - 0.06,
                        h_range=(0.19, 0.25) if i < 2 else (0.17, 0.24), seed=11 + i * 7, stacks=1 if i % 2 else 0)
