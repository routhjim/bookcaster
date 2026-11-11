"""Model script: a blacksmith's wall tool rack (back against a wall at +Y): a long oak board ~1.3 m x 0.16 m with
iron pegs; two hammers hang from pegs by their heads, handles down; a row of punches and cold chisels hang through a
drilled oak strip by their mushroomed heads; a pair of tongs hangs from a peg on the right. The origin is at the
bottom of the lowest hanging tool: place it on the wall with its base ~0.9 m up (the board is then ~1.2-1.4 m).
"""

import math
import os
import random
import sys

from mathutils import Vector

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
rng = random.Random(31)
W, BH, BT = 1.3, 0.16, 0.035
BZ = 0.36                          # board bottom above the asset base (handles hang below it)
BY = 0.0                           # board front face plane (back at BY + BT)
wood, irons, handles = [], [], []

wood.append(K.box("board", (W, BT, BH), loc=(0, BY + BT / 2, BZ + BH / 2), bev=0.006))
STRIP_Y = BY - 0.022
wood.append(K.box("strip", (0.46, 0.044, 0.03), loc=(0.02, STRIP_Y, BZ + BH - 0.03), bev=0.004))
for x in (-0.19, 0.25):           # strip brackets
    wood.append(K.box("strip_block", (0.03, 0.044, 0.05), loc=(x, STRIP_Y, BZ + BH - 0.065), bev=0.003))


def peg(x, z, length=0.06, r=0.0065):
    p = K.lathe("peg", [(0.0001, 0.0), (r, 0.0), (r * 0.9, length - 0.006), (r * 1.4, length - 0.004),
                        (r * 1.4, length), (0.0, length)], seg=12)
    p.location, p.rotation_euler = (x, BY + 0.01, z), (math.pi / 2, 0, 0)       # local +Z -> world -Y
    irons.append(p)


def hammer(x, size, peen=1):
    """Head lying along X on a peg, handle hanging straight down in front of the board."""
    hl, hw = 0.105 * size, 0.028 * size
    zt = BZ + BH - 0.035
    peg(x, zt - hw - 0.004, 0.07)
    head = K.rbox("hhead", (hl, hw, hw), r=0.004, spacing=0.02, loc=(x, BY - 0.035, zt - hw / 2 + 0.002))
    K.warp(head, lambda v: (v.x, v.y, v.z) if v.x * peen < hl * 0.1 else
           (v.x, v.y, v.z * K.lerp(1.0, 0.35, (v.x * peen - hl * 0.1) / (hl * 0.4))))
    irons.append(head)
    L = 0.36 * size
    z0 = zt - hw / 2
    path = [(x, BY - 0.035, z0 - L * t / 10) for t in range(11)]
    r = 0.0125 * size                    # K.tube r_fn returns the radius itself, not a factor
    handles.append(K.tube("handle", path, r, seg=12, r_fn=lambda t: r if t < 0.75 else r * 1.1))


hammer(-0.52, 1.1, 1)
hammer(-0.36, 1.0, -1)


def punch(x, kind, L):
    top = BZ + BH - 0.015 + 0.01
    path = [(x, STRIP_Y, top - L * t / 16) for t in range(17)]

    def wf(t):
        if t < 0.06:
            return 1.4 - 0.4 * t / 0.06
        if t > 0.65:
            return K.lerp(1.0, 0.2 if kind != "chisel" else 0.85, (t - 0.65) / 0.35)
        return 1.0
    ob = F.bar(kind, path, 0.015, 0.015, rnd=0.6 if kind == "round" else 0.15, w_fn=wf)
    if kind == "chisel":            # flatten the cutting end into an edge across X
        lo = top - 0.65 * L
        K.warp(ob, lambda v: (x + (v.x - x) * K.lerp(1.0, 1.35, K.smooth((lo - v.z) / (0.35 * L))),
                              STRIP_Y + (v.y - STRIP_Y) * K.lerp(1.0, 0.3, K.smooth((lo - v.z) / (0.35 * L))), v.z))
    irons.append(ob)


for k, x in enumerate([-0.15, -0.09, -0.03, 0.03, 0.09, 0.15, 0.21]):
    punch(x, ["punch", "chisel", "round", "punch", "chisel", "round", "punch"][k], rng.uniform(0.17, 0.23))

# tongs on a peg: reins over the peg, arms crossing at the rivet, jaws at the bottom
TX = 0.47
peg(TX, BZ + BH - 0.03, 0.08)
zt = BZ + BH - 0.03
TY = BY - 0.05
piv_z = zt - 0.30
for s in (-1, 1):
    pts = [Vector((TX + s * 0.006, TY, zt - 0.006))]
    for t in range(1, 11):
        u = t / 10
        pts.append(Vector((TX + s * K.lerp(0.006, 0.02, u ** 0.7) * (1 - K.smooth((u - 0.8) / 0.2)), TY,
                           zt - 0.006 - (zt - 0.006 - piv_z) * u)))
    for t in range(1, 7):
        u = t / 6
        pts.append(Vector((TX - s * 0.008 * K.smooth(u), TY, piv_z - 0.09 * u)))
    irons.append(K.tube("tong_arm", pts, 0.0055, seg=10))
boss = K.lathe("tong_boss", [(0.0001, -0.009), (0.016, -0.009), (0.016, 0.009), (0.0, 0.009)], seg=16)
boss.location, boss.rotation_euler = (TX, TY, piv_z), (math.pi / 2, 0, 0)
irons.append(boss)
for x in (-0.6, 0.6):              # spikes fixing the board to the wall
    irons += K.nails("spike", [(x, BY - 0.002, BZ + BH / 2)], r=0.009, normal=(0, -1, 0))

M_W = F.smithy_wood("rack_oak", axis="X", base=(0.13, 0.08, 0.045), grey=0.35, soot=0.5, wear=0.4)
M_HANDLE = F.smithy_wood("rack_handles", axis="Z", base=(0.22, 0.15, 0.08), grey=0.1, soot=0.2, wear=0.8)
M_IRON = F.forged_iron("rack_iron", scale_amt=0.7, rust=0.35, polish=0.4)
for o in wood:
    K.assign(o, M_W)
for o in handles:
    K.assign(o, M_HANDLE)
for o in irons:
    K.assign(o, M_IRON)
