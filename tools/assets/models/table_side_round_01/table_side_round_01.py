"""Model script: Victorian round tripod side table, mahogany, 0.60 m top at 0.72 m, front -Y.

Moulded round top (dished rim, rounded nosing, cove under), square bearer block, turned baluster column (vase knop,
beads, ring collar), three snake-curved splayed legs (one to the back) on pad feet."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

H, R_TOP, T_TOP = 0.72, 0.30, 0.022
mah = K.wood("mahogany", "Z", wear=0.55, gloss=0.65, grime=0.6)
mah_top = K.wood("mahogany", "X", wear=0.5, gloss=0.7, grime=0.4, offset=0.3)

# top: profile (r, z) from the centre of the underside round the edge to the centre of the top
z0 = H - T_TOP
prof = [(0.0, z0), (R_TOP - 0.035, z0), (R_TOP - 0.012, z0 + 0.003)]
for k in range(7):                      # cove under the nosing
    a = math.pi / 2 * k / 6
    prof.append((R_TOP - 0.012 + 0.008 * math.sin(a), z0 + 0.003 + 0.008 * (1 - math.cos(a))))
for k in range(9):                      # rounded nosing
    a = -math.pi / 2 + math.pi * k / 8
    prof.append((R_TOP - 0.0055 + 0.0055 * math.cos(a), z0 + 0.0165 + 0.0055 * math.sin(a)))
prof += [(R_TOP - 0.010, H), (R_TOP - 0.016, H - 0.003), (R_TOP - 0.022, H - 0.0035), (0.0, H - 0.0035)]
top = K.lathe("top", prof, seg=96, mat=mah_top)

# bearer block under the top
blk = K.box("bearer", (0.15, 0.15, 0.028), (0, 0, z0 - 0.014), bev=0.003, mat=mah_top)

# column: baluster from the leg block to the bearer
col_keys = [
    (0.10, 0.0001, "c"), (0.10, 0.050, "c"), (0.215, 0.050, "l"), (0.215, 0.044, "c"),
    *K.bead(0.226, 0.047, 0.018),
    (0.245, 0.034), (0.275, 0.030), (0.33, 0.060), (0.385, 0.052), (0.43, 0.030),  # vase knop
    *K.bead(0.443, 0.034, 0.014),
    (0.46, 0.024), (0.58, 0.021), (0.60, 0.026),
    *K.bead(0.615, 0.034, 0.016),
    (0.632, 0.028), (0.645, 0.040), (0.668, 0.040, "l"), (0.672, 0.025, "c"), (z0 - 0.026, 0.024, "l"),
    (z0 - 0.026, 0.0001, "c"),
]
col = K.lathe("column", K.keys(col_keys, 0.003), seg=40, mat=mah)

# three snake legs, one to the back (+Y)
legs = []
for i, ang in enumerate((90, 210, 330)):
    pts = K.bezier((0.030, 0.0, 0.205), (0.11, 0.0, 0.19), (0.20, 0.0, 0.10), (0.262, 0.0, 0.020), 36)
    pts.append((0.276, 0.0, 0.016))
    secs = []
    for j in range(len(pts)):
        t = j / (len(pts) - 1)
        hh = K.lerp(0.044, 0.018, K.smooth(t ** 0.8))          # vertical depth tapers
        hw = K.lerp(0.020, 0.014, t)                             # sideways thickness
        secs.append(K.superellipse2d(hw, hh, 2.8, 0.0, 16))
    leg = K.sweep(f"leg{i}", pts, secs, up=(0, 0, 1), mat=K.wood("mahogany", "X", 0.6, 0.65, 0.6, 0.1 * i))
    foot = K.lathe(f"foot{i}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.021, "c"), (0.006, 0.022), (0.014, 0.017),
                                        (0.018, 0.012), (0.018, 0.0001, "c")], 0.002), seg=24,
                   mat=K.wood("mahogany", "Z", 0.7, 0.6, 0.6, 0.2 * i))
    foot.location = (0.272, 0.0, 0.0)
    for o in (leg, foot):
        o.data.transform(K.Matrix.Translation(o.location))
        o.location = (0, 0, 0)
        o.rotation_euler = (0, 0, math.radians(ang))
    legs += [leg, foot]
