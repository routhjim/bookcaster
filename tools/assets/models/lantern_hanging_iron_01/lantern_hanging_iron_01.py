"""Model script: an early medieval hanging lantern, lit: a small round black iron cage ~0.16 m across and ~0.36 m
tall to the top of its hanging ring: a dished base, six upright straps and three bands, thin amber horn panes glowing
from the candle inside, a door pane standing ajar at the front (-Y) so the candle and its flame show, a conical
pierced top with a ring. Origin at the base (hang it by placing its base at the wanted height).
"""

import math
import os
import random
import sys

import bmesh
from mathutils import Vector

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
rng = random.Random(8)
R = 0.075
Z0, Z1 = 0.022, 0.245
DOOR = (math.radians(-90 - 32), math.radians(-90 + 32))   # front opening (angles about Z; -90 deg = -Y)
irons, horn = [], []

irons.append(K.lathe("base", [(0.0001, 0.0), (0.05, 0.0), (0.07, 0.006), (R + 0.006, 0.012), (R + 0.008, Z0),
                              (R - 0.004, Z0), (R - 0.006, 0.014), (0.0, 0.014)], seg=48, cap1=False))
for z in (Z0 + 0.006, 0.135, Z1 - 0.004):
    irons.append(K.lathe("band", [(R - 0.002, z - 0.006), (R + 0.004, z - 0.006), (R + 0.004, z + 0.006),
                                  (R - 0.002, z + 0.006), (R - 0.002, z - 0.006)], seg=48, cap0=False, cap1=False))
for i in range(6):
    a = math.radians(-90 + 30 + 60 * i)
    s = K.box("strap", (0.012, 0.004, Z1 - Z0 + 0.01), loc=(math.cos(a) * (R + 0.003), math.sin(a) * (R + 0.003),
                                                             (Z0 + Z1) / 2), rot=(0, 0, a + math.pi / 2), bev=0.001)
    irons.append(s)


def pane_ring(name, a0, a1, r, z0, z1, seg=24):
    bm = bmesh.new()
    lo, hi = [], []
    for k in range(seg + 1):
        a = K.lerp(a0, a1, k / seg)
        lo.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z0)))
        hi.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z1)))
    for k in range(seg):
        bm.faces.new((lo[k], lo[k + 1], hi[k + 1], hi[k]))
    ob = K.from_bm(name, bm, recalc=False)
    m = ob.modifiers.new("t", "SOLIDIFY")
    m.thickness, m.offset = 0.0018, 0.0
    K.shade(ob, 30)
    return ob


horn.append(pane_ring("horn", DOOR[1], DOOR[0] + 2 * math.pi, R - 0.001, Z0 + 0.002, Z1 - 0.002, seg=40))
# the door: one curved pane in its own frame, hinged on the right edge of the opening and swung out ~70 deg
hx, hy = math.cos(DOOR[1]) * R, math.sin(DOOR[1]) * R
door = pane_ring("door", DOOR[0], DOOR[1], R - 0.001, Z0 + 0.012, Z1 - 0.012, seg=12)
dframe = []
for z in (Z0 + 0.014, Z1 - 0.014):
    dframe.append(K.lathe("door_bar", [(R - 0.003, z - 0.004), (R + 0.003, z - 0.004), (R + 0.003, z + 0.004),
                                       (R - 0.003, z + 0.004), (R - 0.003, z - 0.004)], seg=48, cap0=False, cap1=False))
for o in [door] + dframe:
    o.location = (-hx, -hy, 0)
for o in [door] + dframe:
    # clip the door bars to the opening (they are full rings): keep only verts inside the door arc by scaling the
    # ring's other verts onto the hinge (cheap clip)
    if o.name.startswith("door_bar"):
        for v in o.data.vertices:
            a = math.atan2(v.co.y, v.co.x)
            if not (DOOR[0] - 0.02 <= a <= DOOR[1] + 0.02):
                v.co.x, v.co.y = math.cos(DOOR[1]) * (R - 0.003 if math.hypot(v.co.x, v.co.y) < R else R + 0.003), \
                    math.sin(DOOR[1]) * (R - 0.003 if math.hypot(v.co.x, v.co.y) < R else R + 0.003)
        o.data.update()
# rotate the door group about the hinge (vertical axis through the hinge point)
for o in [door] + dframe:
    for v in o.data.vertices:
        p = Vector((v.co.x - hx, v.co.y - hy, v.co.z))
        c, s = math.cos(math.radians(-75)), math.sin(math.radians(-75))
        v.co = Vector((hx + c * p.x - s * p.y, hy + s * p.x + c * p.y, p.z))
    o.data.update()
    o.location = (0, 0, 0)
horn.append(door)
irons += dframe
irons.append(K.lathe("hinge", [(0.0001, Z0 + 0.01), (0.004, Z0 + 0.01), (0.004, Z1 - 0.01), (0.0, Z1 - 0.01)], seg=10))
irons[-1].location = (hx * 1.04, hy * 1.04, 0)

# conical top, pierced with a ring of holes, and the hanging ring
top = K.lathe("cone", [(R + 0.012, Z1), (R + 0.012, Z1 + 0.006), (0.026, Z1 + 0.075), (0.012, Z1 + 0.082),
                       (0.0, Z1 + 0.084)], seg=48, cap0=False)
sol = top.modifiers.new("t", "SOLIDIFY")
sol.thickness, sol.offset = 0.0015, -1.0
for i in range(10):
    a = 2 * math.pi * (i + 0.5) / 10
    rr = 0.055
    zc = Z1 + 0.006 + (R + 0.012 - rr) / (R + 0.012 - 0.026) * 0.069
    c = K.lathe("hole", [(0.0001, -0.03), (0.0065, -0.03), (0.0065, 0.03), (0.0, 0.03)], seg=12)
    c.location = (rr * math.cos(a), rr * math.sin(a), zc)
    c.rotation_euler = (0, math.radians(45), a)
    c.hide_set(True)
    m = top.modifiers.new("hole", "BOOLEAN")
    m.operation, m.object, m.solver = "DIFFERENCE", c, "EXACT"
irons.append(top)
ring = K.tube("ring", [(0.022 * math.cos(2 * math.pi * i / 24), 0.0, Z1 + 0.084 + 0.022 + 0.022 * math.sin(2 * math.pi * i / 24))
                       for i in range(24)], 0.0035, seg=8, closed=True)
irons.append(ring)

# candle on a pricket, and its flame
candle = K.lathe("candle", [(0.0001, 0.014), (0.013, 0.014), (0.013, 0.09), (0.011, 0.094), (0.004, 0.092),
                            (0.0, 0.093)], seg=24)
wick = K.lathe("wick", [(0.0001, 0.092), (0.0012, 0.092), (0.001, 0.102), (0.0, 0.103)], seg=6)
flame = K.lathe("flame", [(0.0001, 0.097), (0.0045, 0.101), (0.0055, 0.108), (0.0035, 0.118), (0.0, 0.13)], seg=16)
drips = []
for i in range(4):
    a = rng.uniform(0, 6.28)
    d = K.lathe("drip", [(0.0001, 0.0), (0.003, 0.0), (0.0025, 0.03), (0.0, 0.032)], seg=8)
    d.location = (0.0128 * math.cos(a), 0.0128 * math.sin(a), rng.uniform(0.05, 0.06))
    drips.append(d)

M_IRON = F.forged_iron("lantern_iron", scale_amt=0.6, rust=0.35, polish=0.2, soot=0.4)
kh = K.NK("lantern_horn_lit")
mot = kh.noise(kh.obj, 60.0, 4.0, 0.6)
streak = kh.noise(kh.obj, 1.0, 4.0, 0.6, stretch=(40.0, 40.0, 4.0))
f = kh.m("ADD", kh.m("MULTIPLY", mot, 0.6), kh.m("MULTIPLY", streak, 0.6))
base = kh.mix(kh.mr(f, 0.3, 0.9), (0.55, 0.32, 0.10), (0.30, 0.14, 0.04))
emit = kh.mix(kh.mr(f, 0.3, 0.9), (1.0, 0.62, 0.26), (0.55, 0.22, 0.05))
M_HORN = kh.out(base, 0.45, 0.0, kh.bump(mot, 0.0002, 0.2), emit, 4.0)
M_WAX = K.emissive((1.0, 0.8, 0.55), 0.6, base=(0.80, 0.74, 0.60), name="lantern_wax")
M_FLAME = K.emissive((1.0, 0.66, 0.26), 40.0, name="lantern_flame")
M_WICK = K.metal("iron", tarnish=1.0, name="lantern_wick")
for o in irons:
    K.assign(o, M_IRON)
for o in horn:
    K.assign(o, M_HORN)
for o in [candle] + drips:
    K.assign(o, M_WAX)
K.assign(flame, M_FLAME)
K.assign(wick, M_WICK)
