"""Model script: small wicker chair: woven seat, horseshoe back/arms band, rolled cane rims, cane legs, front -Y.

The weave is a material (two crossed band sets in object space, bump + dark gaps); the shape is a swept band for
the back (height falls from the back centre to the arm fronts), a rounded seat cushion, tubes for rims and legs."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

k = K.NK("wicker")
v = k.vec(k.obj, (1.0, 1.0, 1.0))
sx = k.N.new("ShaderNodeSeparateXYZ")
k.L.new(v, sx.inputs[0])
ang = k.m("ARCTAN2", sx.outputs["Y"], sx.outputs["X"])
a = k.m("SINE", k.m("MULTIPLY", ang, 70.0))                       # vertical stakes round the chair
b = k.m("SINE", k.m("MULTIPLY", sx.outputs["Z"], 420.0))         # horizontal weavers
c = k.m("SINE", k.m("ADD", k.m("MULTIPLY", sx.outputs["X"], 300.0), k.m("MULTIPLY", sx.outputs["Y"], 300.0)))  # seat weave
weave = k.m("MULTIPLY", k.m("ADD", k.m("MULTIPLY", a, 0.6), b), 0.5)
weave = k.m("ADD", weave, k.m("MULTIPLY", c, 0.2))
gap = k.mr(weave, -0.6, -0.2, 1.0, 0.0)
n = k.noise(k.obj, 12.0, 3.0, 0.5)
col = k.mix(k.mr(n, 0.3, 0.7), (0.24, 0.17, 0.09, 1), (0.33, 0.25, 0.14, 1))
col = k.mix(k.m("MULTIPLY", gap, 0.8), col, (0.05, 0.035, 0.02, 1))
cv = k.cavity(0.03)
col = k.mix(k.m("MULTIPLY", cv, 0.6), col, (0.06, 0.045, 0.03, 1))
e = k.edge(0.004)
col = k.mix(k.m("MULTIPLY", e, 0.5), col, (0.42, 0.34, 0.22, 1))
wick = k.out(col, k.mixf(gap, 0.65, 0.9), 0.0, k.bump(weave, 0.0015, 0.6))
cane = K.fabric("rope", (0.30, 0.22, 0.12))

SZ, R = 0.42, 0.25
parts = []
seat = K.rbox("seat", (0.50, 0.46, 0.05), r=0.022, spacing=0.02, bulge=(0, 0, 0.006), loc=(0, 0, SZ - 0.025), mat=wick)
K.warp(seat, lambda c: (c.x * (1 - 0.18 * max(0.0, c.y / 0.23) ** 2), c.y, c.z))
parts.append(seat)
# back band: horseshoe from one arm front round the back to the other
N = 61
path, secs = [], []
for i in range(N):
    t = i / (N - 1)
    th = math.radians(-30 + 240 * t)                 # -30 .. 210 degrees (opening at the front)
    x, y = 0.26 * math.cos(th), 0.235 * math.sin(th) + 0.01
    h = 0.17 + 0.21 * K.smooth(math.sin(math.pi * t) ** 1.5)
    path.append((x, y, SZ - 0.01 + h / 2))
    secs.append(K.superellipse2d(0.012, h / 2, 6.0, 0.0, 12))
band = K.sweep("band", path, secs, up=(0, 0, 1), mat=wick, angle=60)
parts.append(band)
rim = [(p[0], p[1], p[2] + (secs[i][0][1] if False else 0)) for i, p in enumerate(path)]
toprim = []
for i in range(N):
    t = i / (N - 1)
    h = 0.17 + 0.21 * K.smooth(math.sin(math.pi * t) ** 1.5)
    toprim.append((path[i][0], path[i][1], SZ - 0.01 + h))
parts.append(K.tube("toprim", toprim, 0.016, seg=12, mat=cane))
seatrim = [(0.25 * math.cos(a), 0.23 * math.sin(a), SZ - 0.01) for a in [math.tau * i / 48 for i in range(48)]]
parts.append(K.tube("seatrim", seatrim, 0.013, seg=10, closed=True, mat=cane))
# legs, skirt band, stretcher hoop
for i, a in enumerate((45, 135, 225, 315)):
    ar = math.radians(a)
    x, y = 0.21 * math.cos(ar), 0.19 * math.sin(ar)
    parts.append(K.tube(f"leg{i}", [(x * 1.06, y * 1.06, 0.0), (x, y, SZ - 0.01)], 0.014, seg=10, mat=cane))
skirt = [(0.235 * math.cos(a), 0.215 * math.sin(a), SZ - 0.06) for a in [math.tau * i / 48 for i in range(48)]]
sk = K.sweep("skirt", skirt, [K.superellipse2d(0.008, 0.04, 6.0, 0.0, 8) for _ in skirt], closed=True, up=(0, 0, 1), mat=wick)
parts.append(sk)
hoop = [(0.215 * math.cos(a), 0.195 * math.sin(a), 0.12) for a in [math.tau * i / 48 for i in range(48)]]
parts.append(K.tube("hoop", hoop, 0.008, seg=8, closed=True, mat=cane))
