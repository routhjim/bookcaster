"""Model script: a hand-forged iron horseshoe lying flat, ~0.13 m across (X) and 0.14 m long (toe to -Y), web
~22 mm wide and 9 mm thick, a fullered groove with four nail holes on each branch, a raised toe clip, heels a little
thicker. Black scale, worn bright on the ground face edges, rust in the holes.
"""

import math
import os
import sys

import bmesh
from mathutils import Vector

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
N = 80
RX, RY = 0.054, 0.060                   # centreline half axes
path, secs = [], []
for i in range(N + 1):
    # from one heel round the toe to the other: angle from -200 to +20 deg about the centre, shoe opens to +Y
    a = math.radians(K.lerp(-25.0, 205.0, i / N))
    x = RX * math.cos(a)
    y = -RY * math.sin(a) * (1.0 if math.sin(a) > 0 else 0.55)
    path.append((x, y + 0.01, 0.0045))
    t = abs(i / N - 0.5) * 2               # 0 at the toe, 1 at the heels
    w = K.lerp(0.0135, 0.0115, t)
    th = K.lerp(0.0046, 0.0052, t ** 2)
    secs.append(K.superellipse2d(th, w, 6.0, 0.0, 16))
shoe = K.sweep("shoe", path, secs, up=(0, 0, 1), angle=40)
# nail holes: square punches through the fuller line, four per branch
cutters = []
for side in (-1, 1):
    for k in range(4):
        u = K.lerp(0.20, 0.38, k / 3)
        a = math.radians(K.lerp(-25.0, 205.0, 0.5 + side * u))
        x, y = RX * math.cos(a), -RY * math.sin(a) + 0.01
        nx, ny = math.cos(a) * 0.004, -math.sin(a) * 0.004
        c = K.box("hole", (0.004, 0.004, 0.03), loc=(x + nx, y + ny, 0.0045), rot=(0, 0, -a), bev=0.0)
        c.hide_set(True)
        m = shoe.modifiers.new("hole", "BOOLEAN")
        m.operation, m.object, m.solver = "DIFFERENCE", c, "EXACT"
# fullering: a groove along each branch on the ground face (top), the nail holes punched in it
for side in (-1, 1):
    gp, gs = [], []
    for i in range(25):
        u = K.lerp(0.10, 0.47, i / 24)
        a = math.radians(K.lerp(-25.0, 205.0, 0.5 + side * u))
        rr = 1.0 + 0.004 / RY
        gp.append((RX * rr * math.cos(a), -RY * rr * math.sin(a) * (1.0 if math.sin(a) > 0 else 0.55) + 0.01, 0.0093))
        gs.append(K.superellipse2d(0.0022, 0.0022, 2.5, 0.0, 10))
    g = K.sweep(f"fuller{side}", gp, gs, up=(0, 0, 1))
    g.hide_set(True)
    m = shoe.modifiers.new("fuller", "BOOLEAN")
    m.operation, m.object, m.solver = "DIFFERENCE", g, "EXACT"
# toe clip: a small raised triangle at the front outer edge
bm = bmesh.new()
yo = -RY + 0.01 - 0.011
pts = [(-0.012, yo, 0.008), (0.012, yo, 0.008), (0.0, yo + 0.002, 0.022)]
f = [bm.verts.new(p) for p in pts]
b = [bm.verts.new((p[0], p[1] + 0.0035, p[2])) for p in pts]
bm.faces.new(f)
bm.faces.new(list(reversed(b)))
for i in range(3):
    bm.faces.new((f[i], b[i], b[(i + 1) % 3], f[(i + 1) % 3]))
clip = K.from_bm("clip", bm)
M = F.forged_iron("shoe_iron", scale_amt=0.8, rust=0.45, polish=0.3)
K.assign(shoe, M)
K.assign(clip, M)
