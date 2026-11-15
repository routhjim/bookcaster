"""Model script: Victorian rococo mantel clock, 0.30 m tall, front -Y, after the reference drawing (a scrolled gilt
case with a shell crest and scroll feet round a drum movement; a white enamel dial with black Roman numerals and a
minute track, blued-steel spade hands at ten past ten, a domed glass in a gilt bezel).

The relief (C-scrolls, volutes, the shell's ribs, leaf scrolls) is geometry swept along curves on the case front;
the numerals, track and hands are thin raised parts on the dial (they hold at a close shot without a texture).

SPEC["parts"] (optional): only these groups ("case", "dial", "glass") for a hybrid.
"""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

PARTS = set(SPEC.get("parts") or ("case", "dial", "glass"))  # noqa: F821
gilt = K.gilt("clock_gilt")
enamel = K.ceramic((0.86, 0.85, 0.80), glaze=0.9, craze=0.25, name="dial_enamel")
ink = K.ceramic((0.015, 0.015, 0.015), glaze=0.7, craze=0.0, name="dial_ink")
blued = K.metal("blued", tarnish=0.2, polish=0.9, name="hands_blued")
ZC, RD = 0.150, 0.058                   # dial centre height, dial radius
YF = -0.058                             # case front face


def catmull(pts, n=6, closed=False):
    """A smooth curve through the points (Catmull-Rom)."""
    out = []
    m = len(pts)
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        p0 = pts[(i - 1) % m] if closed or i > 0 else pts[0]
        p1, p2 = pts[i], pts[(i + 1) % m]
        p3 = pts[(i + 2) % m] if closed or i + 2 < m else pts[-1]
        for k in range(n):
            t = k / n
            out.append(tuple(0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t
                                    + (-a + 3 * b - 3 * c + d) * t ** 3) for a, b, c, d in zip(p0, p1, p2, p3)))
    if not closed:
        out.append(pts[-1])
    return out


def spiral(cx, cz, r0, a0, turns, shrink, n=24, s=1):
    """A volute: points of a spiral that winds in from radius r0 at angle a0."""
    return [(cx + s * r0 * math.exp(-shrink * t) * math.cos(a0 + s * turns * math.tau * t),
             cz + r0 * math.exp(-shrink * t) * math.sin(a0 + s * turns * math.tau * t)) for t in [k / n for k in range(n + 1)]]


if "case" in PARTS:
    half = [(0.0, 0.040), (0.040, 0.034), (0.075, 0.022), (0.100, 0.003), (0.122, 0.002), (0.131, 0.016),
            (0.122, 0.030), (0.104, 0.046), (0.100, 0.072), (0.111, 0.102), (0.112, 0.132), (0.101, 0.163),
            (0.091, 0.190), (0.101, 0.213), (0.096, 0.234), (0.076, 0.240), (0.060, 0.254), (0.046, 0.276),
            (0.030, 0.294), (0.012, 0.301)]
    # top centre, down the left side, along the feet, up the right side: counter-clockwise seen from the front
    ring = [(0.0, 0.302)] + [(-x, z) for x, z in reversed(half[1:])] + [(0.0, 0.040)] + half[1:]
    outline = []
    for p in catmull(ring, n=6, closed=True):               # drop points closer than 1.5 mm (bevel slivers)
        if not outline or math.hypot(p[0] - outline[-1][0], p[1] - outline[-1][1]) > 0.0015:
            outline.append(p)
    K.plate("case_front", outline, 0.045, y=YF + 0.0225, mat=gilt, bev=0.0025, seg=2)
    # the drum that holds the movement, front to back, with a back door
    drum = K.lathe("drum", K.keys([(0.0, 0.0001, "c"), (0.0, 0.066, "c"), (0.004, 0.068), (0.056, 0.068, "l"),
                                   (0.060, 0.065), (0.060, 0.0001, "c")], 0.003), seg=64, mat=gilt)
    drum.location, drum.rotation_euler = (0, -0.016, ZC), (-math.pi / 2, 0, 0)
    door = K.lathe("back_door", [(0.0001, 0.0), (0.058, 0.0), (0.058, 0.003), (0.050, 0.005), (0.0, 0.006)], seg=48,
                   mat=K.metal("brass", tarnish=0.7, polish=0.4, name="door_brass"))
    door.location, door.rotation_euler = (0, 0.043, ZC), (-math.pi / 2, 0, 0)
    # a base bar joining the feet front to back, so the clock stands on four scroll feet
    for sx in (-1, 1):
        K.box("foot_bar", (0.016, 0.075, 0.012), (sx * 0.108, 0.005, 0.006), bev=0.003, mat=gilt)

    # relief on the case front: C-scrolls up the sides with volutes, the shell's ribs, leaf scrolls under the dial
    Y = YF - 0.001
    for s in (-1, 1):
        side = [(s * 0.098, 0.050), (s * 0.090, 0.080), (s * 0.097, 0.110), (s * 0.096, 0.140), (s * 0.084, 0.170),
                (s * 0.083, 0.195), (s * 0.090, 0.215)]
        path = catmull(side, 5) + spiral(s * 0.080, 0.218, 0.010, 0.0 if s > 0 else math.pi, 1.1, 1.4, 18, s)[1:]
        K.tube("c_scroll", [(x, Y, z) for x, z in path], 0.0042, seg=10,
               r_fn=lambda t: 0.0045 * (1 - 0.55 * t), mat=gilt)
        foot = spiral(s * 0.112, 0.020, 0.012, math.pi / 2, 1.0, 1.2, 20, -s)
        K.tube("volute", [(x, Y, z) for x, z in [(s * 0.098, 0.050), (s * 0.106, 0.036)] + foot], 0.004, seg=10,
               r_fn=lambda t: 0.0042 * (1 - 0.5 * t), mat=gilt)
        leaf = catmull([(s * 0.004, 0.046), (s * 0.030, 0.050), (s * 0.052, 0.068), (s * 0.050, 0.086),
                        (s * 0.038, 0.084)], 5)
        K.tube("leaf", [(x, Y, z) for x, z in leaf], 0.0035, seg=8, r_fn=lambda t: 0.0036 * (1 - 0.6 * t), mat=gilt)
        for k in range(3):                                      # small leaf fronds off the C-scroll
            z0 = 0.085 + 0.035 * k
            fr = catmull([(s * 0.093, z0), (s * 0.104, z0 + 0.010), (s * 0.106, z0 + 0.022)], 4)
            K.tube("frond", [(x, Y, z) for x, z in fr], 0.0024, seg=8, r_fn=lambda t: 0.0025 * (1 - 0.7 * t), mat=gilt)
    for k in range(9):                                          # the shell crest's ribs, fanning up from its hinge
        a = math.radians(25 + 130 * k / 8)
        pts = [(0.0 + 0.008 * math.cos(a), 0.226 + 0.008 * math.sin(a)),
               (0.052 * math.cos(a) * 0.98, 0.226 + 0.066 * math.sin(a))]
        K.tube("rib", [(x, Y - 0.001, z) for x, z in catmull(pts, 6)], 0.0034, seg=8,
               r_fn=lambda t: 0.0022 + 0.0016 * t, mat=gilt)
    hin = K.lathe("shell_hinge", [(0.0001, 0.0), (0.009, 0.0), (0.008, 0.004), (0.0, 0.006)], seg=24, mat=gilt)
    hin.location, hin.rotation_euler = (0, Y + 0.002, 0.226), (math.pi / 2, 0, 0)

if "dial" in PARTS:
    YD = YF - 0.006
    dial = K.lathe("dial", [(0.0001, 0.0), (RD, 0.0), (RD, 0.002), (RD * 0.6, 0.0035), (0.0, 0.004)], seg=96, mat=enamel)
    dial.location, dial.rotation_euler = (0, YD, ZC), (math.pi / 2, 0, 0)
    face = YD - 0.0042                                          # the dial's front
    ring = [((RD + 0.003) * math.cos(math.tau * i / 96), face + 0.001, ZC + (RD + 0.003) * math.sin(math.tau * i / 96))
            for i in range(96)]
    K.tube("bezel", ring, 0.0055, seg=14, closed=True, mat=gilt)
    beads = [((RD + 0.0085) * math.cos(math.tau * i / 72), face + 0.003, ZC + (RD + 0.0085) * math.sin(math.tau * i / 72))
             for i in range(72)]
    K.tube("bezel_bead", beads, 0.0022, seg=8, closed=True, mat=gilt)
    for rr in (0.049, 0.0525):                                  # the minute track
        K.tube("track", [(rr * math.cos(math.tau * i / 128), face - 0.0002, ZC + rr * math.sin(math.tau * i / 128))
                         for i in range(128)], 0.00035, seg=6, closed=True, mat=ink)

    def stroke(name, p0, p1, w, mat=ink, y=face - 0.0003):
        dx, dz = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dz)
        K.box(name, (w, 0.0006, L), ((p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2), (0, math.atan2(dx, dz), 0),
              bev=0.0001, seg=1, mat=mat)

    for i in range(60):                                         # minute ticks
        th = math.tau * i / 60
        r0, r1 = 0.049, 0.0525 if i % 5 else 0.0555
        stroke("tick", (r0 * math.sin(th), ZC + r0 * math.cos(th)), (r1 * math.sin(th), ZC + r1 * math.cos(th)),
               0.0006 if i % 5 else 0.0012)
    GL = {"I": [((0, 0), (0, 1))], "V": [((-0.5, 1), (0, 0)), ((0, 0), (0.5, 1))],
          "X": [((-0.5, 0), (0.5, 1)), ((-0.5, 1), (0.5, 0))]}
    GW = {"I": 0.25, "V": 1.0, "X": 1.0}
    NUM = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    GH, RN = 0.0085, 0.0405                                     # glyph height, numeral ring radius (glyph foot)
    for h, txt in enumerate(NUM):
        th = math.tau * h / 12
        up = (math.sin(th), math.cos(th))
        tg = (math.cos(th), -math.sin(th))
        width = sum(GW[c] for c in txt) * GH * 0.55 + (len(txt) - 1) * GH * 0.18
        u = -width / 2
        for c in txt:
            gw = GW[c] * GH * 0.55
            for (a, b) in GL[c]:
                pa = (u + gw / 2 + a[0] * gw, a[1] * GH)
                pb = (u + gw / 2 + b[0] * gw, b[1] * GH)
                w2 = lambda p: (tg[0] * p[0] + up[0] * (RN + p[1]), ZC + tg[1] * p[0] + up[1] * (RN + p[1]))
                stroke("numeral", w2(pa), w2(pb), 0.0011 if c == "I" else 0.0010)
            for z in (0.0, GH):                                 # serifs top and bottom across the glyph
                p = (u - 0.0006, z)
                q = (u + gw + 0.0006, z)
                stroke("serif", (tg[0] * p[0] + up[0] * (RN + p[1]), ZC + tg[1] * p[0] + up[1] * (RN + p[1])),
                       (tg[0] * q[0] + up[0] * (RN + q[1]), ZC + tg[1] * q[0] + up[1] * (RN + q[1])), 0.0005)
            u += gw + GH * 0.18

    def hand(name, deg, L, wide, tail, y):
        th = math.radians(deg)
        up = (math.sin(th), math.cos(th))
        tg = (math.cos(th), -math.sin(th))
        loc = [(0.0, -tail), (0.0016, -tail * 0.6), (0.0013, L * 0.55), (wide, L * 0.66), (0.0006, L * 0.83),
               (0.0003, L)]
        pts = loc + [(-u, v) for u, v in reversed(loc[1:-1])]
        pts = [(tg[0] * u + up[0] * v, ZC + tg[1] * u + up[1] * v) for u, v in pts]
        K.plate(name, pts, 0.0008, y=y, mat=blued, bev=0.0002, seg=1)

    hand("hour_hand", 305, 0.030, 0.0042, 0.006, face - 0.0012)
    hand("minute_hand", 60, 0.044, 0.0032, 0.008, face - 0.0024)
    arb = K.lathe("arbor", [(0.0001, 0.0), (0.0022, 0.0), (0.0018, 0.002), (0.0, 0.0028)], seg=16, mat=blued)
    arb.location, arb.rotation_euler = (0, face - 0.0024, ZC), (math.pi / 2, 0, 0)

if "glass" in PARTS:
    face = YF - 0.006 - 0.0042
    dome = K.lathe("dome_glass", [(RD + 0.002, 0.0), (RD * 0.8, 0.004), (RD * 0.45, 0.0075), (0.0001, 0.0085),
                                  (0.0001, 0.0071), (RD * 0.45, 0.0061), (RD * 0.8, 0.0028), (RD + 0.0005, -0.0011),
                                  (RD + 0.002, 0.0)], seg=96, cap0=False, cap1=False,
                   mat=K.glass((0.96, 0.97, 0.96), 0.01, name="clock_glass"))
    dome.location, dome.rotation_euler = (0, face - 0.0005, ZC), (math.pi / 2, 0, 0)
