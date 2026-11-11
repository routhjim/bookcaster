"""Model script: small earthenware jug: lathe shell (outer, lip, inner), lip pulled into a spout at +X, swept
loop handle at -X."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

blue = (0.035, 0.07, 0.22)
cer = K.ceramic((0.62, 0.58, 0.48), glaze=0.85, craze=0.35, pattern=(blue, [(0.004, 0.010), (0.088, 0.094)]))
outer = [(0.0001, 0.003), (0.026, 0.003), (0.028, 0.0), (0.031, 0.0), (0.032, 0.006), (0.040, 0.018), (0.045, 0.035),
         (0.044, 0.055), (0.038, 0.072), (0.033, 0.085), (0.034, 0.095), (0.037, 0.103), (0.038, 0.106)]
inner = [(0.0355, 0.106), (0.034, 0.100), (0.031, 0.092), (0.031, 0.084), (0.036, 0.071), (0.0415, 0.055),
         (0.042, 0.035), (0.037, 0.019), (0.028, 0.009), (0.0001, 0.007)]
jug = K.lathe("jug", outer + inner, seg=64, cap0=False, cap1=False, mat=cer)


def spout(c):
    r = math.hypot(c.x, c.y)
    a = math.atan2(c.y, c.x)
    w = max(0.0, math.cos(a)) ** 6 * K.smooth((c.z - 0.080) / 0.026)
    nr = r + 0.020 * w
    return (nr * math.cos(a), nr * math.sin(a) * (1 - 0.25 * w), c.z + 0.004 * w)


K.warp(jug, spout)
hp = K.bezier((-0.034, 0.0, 0.088), (-0.075, 0.0, 0.095), (-0.072, 0.0, 0.030), (-0.040, 0.0, 0.032), 24)
secs = [K.superellipse2d(0.0055, 0.0042, 2.4, 0.0, 12) for _ in hp]
K.sweep("handle", hp, secs, up=(0, 1, 0), mat=cer)
