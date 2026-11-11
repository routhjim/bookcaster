"""Model script: Victorian balloon-back dining chair, mahogany with a leather stuffed-over seat, front -Y.

Front legs turned (lathe); each back leg + stile + half of the balloon hoop is one sweep, mirrored for the other
side; shaped waist rail; seat rails; leather seat (rounded box, front wider than back, domed); brass nail heads."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

SEAT_Z = 0.47
RAIL_Z0, RAIL_Z1 = 0.33, 0.405
FX, FY = 0.195, -0.205            # front leg centres
BY = 0.205                        # back leg y at the seat
leg_m = K.wood("mahogany", "Z", wear=0.6, gloss=0.6, grime=0.6)
rail_x = K.wood("mahogany", "X", wear=0.5, gloss=0.6, grime=0.5)
rail_y = K.wood("mahogany", "Y", wear=0.5, gloss=0.6, grime=0.5)
leather = K.fabric("leather", (0.040, 0.011, 0.008), wear=0.7)
brass = K.metal("brass", tarnish=0.7, polish=0.5)

parts = []
# ---- front legs: square block at the rail, then turned
fl_keys = [(0.0, 0.0001, "c"), (0.0, 0.014, "c"), (0.004, 0.016), (0.012, 0.013), (0.030, 0.012), (0.060, 0.014),
           (0.20, 0.0165), *K.bead(0.215, 0.020, 0.010), (0.23, 0.017), (0.29, 0.019),
           *K.bead(0.302, 0.022, 0.010), (0.312, 0.019), (0.322, 0.0001, "c")]
for s in (-1, 1):
    lg = K.lathe(f"fleg{s}", K.keys(fl_keys, 0.003), seg=24, mat=leg_m)
    lg.location = (s * FX, FY, 0)
    blk = K.box(f"fblock{s}", (0.036, 0.036, RAIL_Z1 - 0.318), (s * FX, FY, (0.318 + RAIL_Z1) / 2), bev=0.002, mat=leg_m)
    parts += [lg, blk]

# ---- back legs + stiles + balloon hoop (one sweep per side, the halves meet at the top centre)
def side_path(s):
    pts = []
    # sabre back leg: foot splays back, comes forward to the seat
    for (y, z) in K.bezier((BY + 0.05, 0.0), (BY + 0.03, 0.15), (BY, 0.30), (BY, RAIL_Z1), 10)[:-1]:
        pts.append((s * 0.185, y, z))
    # stile: rakes back, pinches in at the waist, swells out into the balloon, over the top to the centre
    up = K.bezier((s * 0.185, BY, RAIL_Z1), (s * 0.185, BY + 0.01, 0.50), (s * 0.150, BY + 0.03, 0.56),
                  (s * 0.155, BY + 0.04, 0.60), 10)
    loop = K.bezier((s * 0.155, BY + 0.04, 0.60), (s * 0.225, BY + 0.055, 0.70), (s * 0.215, BY + 0.07, 0.86),
                    (0.0, BY + 0.075, 0.875), 18)
    return pts + up[:-1] + loop


def side_secs(n):
    out = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.33:                       # leg: tapered square, rounded corners
            hw = K.lerp(0.014, 0.017, t / 0.33)
            out.append(K.superellipse2d(hw, hw, 3.5, 0.0, 16))
        else:                              # stile / hoop: flat moulded section, wider than thick
            tt = (t - 0.33) / 0.67
            out.append(K.superellipse2d(0.017 - 0.002 * tt, 0.013, 2.8, 0.0, 16))
    return out


for s in (-1, 1):
    p = side_path(s)
    sw = K.sweep(f"stile{s}", p, side_secs(len(p)), up=(1, 0, 0), mat=leg_m, angle=60)
    parts.append(sw)

# ---- waist rail: a curved bar between the stiles with a shaped dip in the middle
wp = []
for k in range(25):
    t = k / 24
    x = K.lerp(-0.152, 0.152, t)
    wp.append((x, BY + 0.035 + 0.012 * math.sin(math.pi * t), 0.585 - 0.012 * math.sin(math.pi * t) ** 2))
ws = [K.superellipse2d(0.010, 0.020 + 0.012 * math.sin(math.pi * k / 24) ** 4, 3.0, 0.0, 16) for k in range(25)]
parts.append(K.sweep("waist", wp, ws, up=(0, 0, 1), mat=K.wood("mahogany", "X", 0.4, 0.6, 0.5, 0.2)))

# ---- seat rails
parts.append(K.box("rail_front", (2 * FX, 0.022, RAIL_Z1 - RAIL_Z0), (0, FY - 0.004, (RAIL_Z0 + RAIL_Z1) / 2), bev=0.002, mat=rail_x))
parts.append(K.box("rail_back", (2 * 0.185, 0.022, RAIL_Z1 - RAIL_Z0 - 0.01), (0, BY, (RAIL_Z0 + RAIL_Z1) / 2 + 0.005), bev=0.002, mat=rail_x))
for s in (-1, 1):
    r = K.box(f"rail_side{s}", (0.022, BY - FY, RAIL_Z1 - RAIL_Z0), (s * 0.188, 0.0, (RAIL_Z0 + RAIL_Z1) / 2), bev=0.002, mat=rail_y)
    r.rotation_euler = (0, 0, s * -0.035)
    parts.append(r)

# ---- seat: stuffed over the rails, front wider than back, domed
seat = K.rbox("seat", (0.46, 0.45, SEAT_Z - RAIL_Z1 + 0.012), r=0.022, spacing=0.02, bulge=(0.004, 0.004, 0.018),
              mat=leather)
K.warp(seat, lambda c: (c.x * K.lerp(1.0, 0.86, (c.y + 0.225) / 0.45), c.y, c.z))
seat.location = (0, -0.005, (SEAT_Z + RAIL_Z1 - 0.012) / 2 - 0.009)
parts.append(seat)

# ---- brass nails along the bottom edge of the seat cover (front and sides)
zN = RAIL_Z1 - 0.004
pts, nrm = [], []
for k in range(15):
    x = K.lerp(-0.21, 0.21, k / 14)
    pts.append((x, -0.229, zN)); nrm.append((0, -1, 0))
for s in (-1, 1):
    for k in range(1, 12):
        y = K.lerp(-0.225, 0.19, k / 11)
        hw = 0.23 * K.lerp(1.0, 0.86, (y + 0.225) / 0.45) + 0.004
        pts.append((s * hw, y, zN)); nrm.append((s, 0, 0))
parts += K.nails("nail", pts, 0.0055, brass, nrm)
