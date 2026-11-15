"""Model script: blacksmith's tongs lying flat, ~0.5 m long along X (jaws to -X, reins to +X), jaws a little open,
reins spread, ball ends.

Two hand-forged arms half-lapped at a peened rivet: each arm is one sweep whose section changes along it (deep,
thin jaw; wide, flat boss round the rivet; round rein tapering to a rounded end). SPEC["variant"]:
  "flat" (default)  flat jaws ~22 mm deep meeting face to face, for flat stock;
  "wolf"            each jaw bowed outward with a V groove (an eye between the tips) for round bar.
"""

import math
import os
import sys

from mathutils import Vector

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
WOLF = SPEC.get("variant") == "wolf"  # noqa: F821
X_TIP, X_PIV, X_END = -0.25, -0.165, 0.25
Z0 = 0.012                                  # centre height of jaws and reins (lying on the floor)
N = 90


def arm(s):
    """s = +1: the arm whose jaw is on +Y and rein on -Y (they cross at the pivot); s = -1 the other."""
    path, secs = [], []
    for i in range(N + 1):
        t = i / N
        x = K.lerp(X_TIP, X_END, t)
        # plan offset y: jaw beside the centre line, crossing at the pivot, rein spreading out
        if x < X_PIV - 0.02:
            u = (x - X_TIP) / (X_PIV - 0.02 - X_TIP)
            y = s * (0.0115 if not WOLF else 0.0035)
            if WOLF:
                y += s * 0.017 * math.sin(math.pi * min(1.0, u * 1.15)) * (1 - K.smooth((u - 0.85) / 0.15))
        elif x < X_PIV + 0.03:
            u = (x - (X_PIV - 0.02)) / 0.05
            y = s * (0.0115 if not WOLF else 0.0035) * (1 - K.smooth(u)) - s * 0.004 * K.smooth(u)
        else:
            u = (x - (X_PIV + 0.03)) / (X_END - X_PIV - 0.03)
            y = -s * (0.004 + 0.046 * K.smooth(min(1.0, u * 1.25)) ** 0.85)
        # height: half-lapped at the boss (this arm above or below), back to Z0 elsewhere
        boss = 1 - K.smooth(abs(x - X_PIV) / 0.035)
        z = Z0 + s * 0.0042 * boss
        # section: (half depth along Z, half width along Y)
        if x < X_PIV - 0.03:
            u = (x - X_TIP) / (X_PIV - 0.03 - X_TIP)
            hz = (0.011 if not WOLF else 0.008) * (0.9 + 0.1 * u)
            hy = 0.0045 if not WOLF else 0.0042
        elif x < X_PIV + 0.035:
            hz = K.lerp(0.0105, 0.0058, boss)
            hy = K.lerp(0.0055, 0.0145, boss)
        else:
            u = (x - (X_PIV + 0.035)) / (X_END - X_PIV - 0.035)
            hz, hy = K.lerp(0.0055, 0.0045, u), K.lerp(0.0075, 0.0052, u)
        nexp = 3.0 if x > X_PIV + 0.035 else 5.0
        path.append((x, y, z))
        secs.append(K.superellipse2d(hz, hy, nexp, 0.0, 16))
    ob = K.sweep(f"arm{'a' if s > 0 else 'b'}", path, secs, up=(0, 0, 1), angle=50)
    if WOLF:                                  # wolf tongs: square-cut rein ends
        return [ob]
    ball = K.lathe(f"ball{s}", [(0.0001, -0.0085), (0.006, -0.007), (0.0085, 0.0), (0.006, 0.007), (0.0, 0.0085)],
                   seg=16)
    ball.location = (X_END + 0.004, path[-1][1], Z0)
    return [ob, ball]


arms = arm(1) + arm(-1)
riv = []
for zt, sgn in ((Z0 + 0.0042 + 0.0058, 1), (Z0 - 0.0042 - 0.0058, -1)):
    # peened rivet heads above and below the boss
    h = K.lathe("rivet", [(0.0001, 0.0), (0.0075, 0.0), (0.0068, 0.0022), (0.004, 0.0034), (0.0, 0.0038)], seg=20,
                cap1=False)
    h.location = (X_PIV, 0.0, zt)
    if sgn < 0:
        h.rotation_euler = (math.pi, 0, 0)
    riv.append(h)
pin = K.lathe("pin", [(0.0001, Z0 - 0.0101), (0.004, Z0 - 0.0101), (0.004, Z0 + 0.0101), (0.0, Z0 + 0.0101)], seg=16)
pin.location = (X_PIV, 0.0, 0.0)
M = F.forged_iron("tongs_iron", scale_amt=0.75, rust=0.25, polish=0.45, soot=0.25)
for o in arms + riv + [pin]:
    K.assign(o, M)
