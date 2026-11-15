"""Model script: brass candlestick with a half-burnt candle (SPEC["variant"] "lit" adds a flame)."""

import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

LIT = SPEC.get("variant") == "lit"  # noqa: F821
brass = K.metal("brass", tarnish=0.7, polish=0.65)
wax = K.ceramic((0.62, 0.56, 0.42), glaze=0.35, craze=0.0, name="tallow")
if LIT:
    wax = K.emissive((0.9, 0.55, 0.25), 0.15, base=(0.64, 0.57, 0.42), name="tallow_lit")
wick = K.metal("iron", tarnish=0.9, polish=0.0, soot=1.0)
body = [(0.0, 0.0001, "c"), (0.0, 0.064, "c"), (0.004, 0.064, "l"), *K.bead(0.009, 0.064, 0.008), (0.014, 0.058),
        (0.030, 0.045), (0.042, 0.026), (0.050, 0.016), *K.bead(0.057, 0.019, 0.010), (0.066, 0.012), (0.10, 0.0105),
        (0.115, 0.018), (0.13, 0.0105), (0.165, 0.011), *K.bead(0.172, 0.016, 0.009), (0.178, 0.012), (0.19, 0.016),
        (0.196, 0.050), (0.200, 0.052), (0.204, 0.050), (0.205, 0.016), (0.225, 0.016), (0.228, 0.0001, "c")]
stick = K.lathe("stick", K.keys(body, 0.002), seg=48, mat=brass)
CZ0, CH, CR = 0.205, 0.105, 0.0115
ctop = CZ0 + CH
candle = K.lathe("candle", K.keys([(CZ0, 0.0001, "c"), (CZ0, CR, "c"), (ctop - 0.006, CR, "l"), (ctop - 0.001, CR * 0.97),
                                   (ctop, CR * 0.85), (ctop - 0.004, CR * 0.6), (ctop - 0.006, 0.0001)], 0.0015),
                  seg=32, mat=wax)
R = random.Random(3)
for k in range(4):                       # wax runs down one side
    a = -0.6 + k * 0.35 + R.uniform(-0.1, 0.1)
    L = R.uniform(0.02, 0.07)
    p = [((CR - 0.0005) * math.cos(a), (CR - 0.0005) * math.sin(a), ctop - 0.002 - L * t / 8) for t in range(9)]
    K.tube(f"run{k}", p, 0.0022, seg=8, mat=wax, r_fn=lambda t: 0.0022 * (1 - 0.35 * t) + 0.0012 * (t > 0.9))
puddle = K.lathe("puddle", [(0.0001, 0.2045), (0.022, 0.2045), (0.024, 0.2058), (0.018, 0.2068), (0.0001, 0.207)],
                 seg=24, mat=wax, cap0=False, cap1=False)
puddle.location = (0.004, -0.006, 0.0)
w = K.tube("wick", [(0, 0, ctop - 0.007), (0.0, 0.0, ctop + 0.004), (0.002, 0.0, ctop + 0.009)], 0.0009, seg=6, mat=wick)
if LIT:
    fl = K.lathe("flame", K.keys([(ctop + 0.003, 0.0001, "c"), (ctop + 0.007, 0.0045), (ctop + 0.014, 0.0055),
                                  (ctop + 0.028, 0.0025), (ctop + 0.038, 0.0001)], 0.001), seg=16,
                 mat=K.emissive((1.0, 0.60, 0.20), 40.0, name="flame"))
