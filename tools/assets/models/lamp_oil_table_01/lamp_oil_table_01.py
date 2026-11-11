"""Model script: Victorian paraffin table lamp, 0.60 m, front -Y (SPEC["variant"]: "unlit" | "lit").

Turned brass foot and column, round font, burner with a pierced gallery and a wick-winder knob to the front, opal
globe shade, clear glass chimney (a real shell, so it refracts). Lit: a flame inside the chimney and a warm glow in
the globe (emissive)."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

LIT = SPEC.get("variant") == "lit"  # noqa: F821
brass = K.metal("brass", tarnish=0.85, polish=0.45)
brass_dark = K.metal("brass", tarnish=0.8, polish=0.3, soot=0.5, soot_z=(0.33, 0.36))

# foot + column + font + collar, one turning
body = [
    (0.0, 0.0001, "c"), (0.0, 0.072, "c"), (0.006, 0.072, "l"), (0.009, 0.068),
    *K.bead(0.016, 0.068, 0.012),
    (0.024, 0.062), (0.040, 0.050), (0.052, 0.034), (0.060, 0.026),
    *K.bead(0.068, 0.030, 0.010),
    (0.078, 0.019), (0.110, 0.016), (0.125, 0.022), (0.140, 0.016),
    *K.bead(0.150, 0.022, 0.010),
    (0.160, 0.020), (0.170, 0.034),
]
for k in range(1, 13):                       # the font: a squat sphere
    a = -math.pi / 2 + math.pi * k / 13
    body.append((0.222 + 0.050 * math.sin(a), 0.068 * math.cos(a) + 0.0001))
body += [(0.272, 0.026), (0.276, 0.026, "l"), *K.bead(0.283, 0.029, 0.010), (0.290, 0.0001, "c")]
font = K.lathe("body", K.keys(body, 0.003), seg=48, mat=brass)

# burner: body, pierced gallery flaring to hold the globe
burner = K.lathe("burner", K.keys([(0.288, 0.0001, "c"), (0.288, 0.026, "c"), (0.315, 0.026, "l"),
                                     (0.322, 0.034), (0.338, 0.046), (0.343, 0.047, "l"), (0.345, 0.040, "c"),
                                     (0.345, 0.0001, "c")], 0.002), seg=40, mat=brass_dark)
# gallery prongs that clip the globe
prongs = []
for i in range(8):
    a = math.tau * i / 8
    p = K.box(f"prong{i}", (0.004, 0.0015, 0.016), (0.0465 * math.cos(a), 0.0465 * math.sin(a), 0.348),
              (0, 0, a + math.pi / 2), bev=0.0004, seg=1, mat=brass_dark)
    prongs.append(p)
# wick-winder knob to the front
shaft = K.lathe("knob_shaft", [(0.0001, 0.0), (0.003, 0.0), (0.003, 0.03), (0.0001, 0.03)], seg=12, mat=brass)
shaft.location, shaft.rotation_euler = (0, -0.018, 0.305), (math.pi / 2, 0, 0)
knob = K.lathe("knob", K.keys([(0.0, 0.0001, "c"), (0.0, 0.010, "c"), (0.002, 0.012), (0.005, 0.012, "l"),
                                (0.007, 0.009), (0.007, 0.0001, "c")], 0.001), seg=24, mat=brass)
knob.location, knob.rotation_euler = (0, -0.046, 0.305), (math.pi / 2, 0, 0)

# opal globe: a shell open at top and bottom
opal = K.ceramic((0.80, 0.78, 0.70), glaze=0.85, craze=0.0, name="opal_glass")
if LIT:
    opal = K.emissive((1.0, 0.66, 0.34), 0.9, base=(0.80, 0.74, 0.62), name="opal_glass_lit")
g_out = []
for k in range(25):
    t = k / 24
    z = K.lerp(0.346, 0.505, t)
    r = 0.044 + 0.056 * math.sin(math.pi * min(1, t * 1.08)) ** 0.8
    if t > 0.85:
        r = K.lerp(r, 0.048, K.smooth((t - 0.85) / 0.15))
    g_out.append((r, z))
g_in = [(r - 0.0025, z) for r, z in reversed(g_out)]
globe = K.lathe("globe", g_out + [(g_out[-1][0] - 0.0025, g_out[-1][1])] + g_in[1:] + [g_out[0]], seg=64, cap0=False,
                cap1=False, mat=opal)

# chimney: clear glass shell, bulb at the bottom
cg = K.glass((0.93, 0.95, 0.92), 0.02, name="chimney_glass")
c_out = [(0.030, 0.340), (0.031, 0.350), (0.036, 0.372), (0.034, 0.395), (0.024, 0.420), (0.021, 0.445),
         (0.021, 0.598), (0.0225, 0.600)]
c_in = [(r - 0.0018, z) for r, z in reversed(c_out)]
c_in[0] = (0.0195, 0.600)
chim = K.lathe("chimney", c_out + c_in + [c_out[0]], seg=48, cap0=False, cap1=False, mat=cg)

parts = [font, burner, shaft, knob, globe, chim] + prongs
if LIT:
    flame_m = K.emissive((1.0, 0.62, 0.22), 30.0, name="flame")
    fl = K.lathe("flame", K.keys([(0.343, 0.0001, "c"), (0.346, 0.010), (0.352, 0.013), (0.366, 0.009), (0.385, 0.0001)],
                                 0.002), seg=20, mat=flame_m)
    fl.scale = (1.0, 0.45, 1.0)
    fl.data.transform(K.Matrix.Diagonal((1.0, 0.45, 1.0, 1.0)))
    fl.scale = (1, 1, 1)
