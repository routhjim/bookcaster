"""Model script: Victorian paraffin table lamp, 0.60 m, front -Y (SPEC["variant"]: "unlit" | "lit").

Turned brass foot, column and font (beads, coves, fine lathe rings, a seam where the font is spun onto its collar),
a burner with a pierced gallery, a flat wick in its tube and a wick-winder knob to the front, an opal globe of
frosted glass (a thin transmissive shell: light passes through it), a clear glass chimney with real thickness and
a rolled rim. Lit: a flame on the wick (hot core, orange body, blue foot) inside the chimney, and the globe glows.

SPEC["parts"] (optional): only these groups ("brass", "burner", "globe", "chimney", "flame") for a hybrid.
"""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

LIT = SPEC.get("variant") == "lit"  # noqa: F821
PARTS = set(SPEC.get("parts") or ("brass", "burner", "globe", "chimney", "flame"))  # noqa: F821
brass = K.metal("brass", tarnish=0.85, polish=0.45)
brass_dark = K.metal("brass", tarnish=0.8, polish=0.3, soot=0.5, soot_z=(0.33, 0.36))


def rings(z0, z1, r, n, depth=0.0006):
    """Fine lathe rings: a run of shallow grooves along a straight stretch of a turning."""
    out = []
    for k in range(n):
        za = z0 + (z1 - z0) * k / n
        zb = za + (z1 - z0) / n
        out += [(za, r), (za + (zb - za) * 0.5, r - depth, "l"), (zb, r, "l")]
    return out


if "brass" in PARTS:
    body = [
        (0.0, 0.0001, "c"), (0.0, 0.074, "c"), (0.004, 0.074, "l"), (0.006, 0.071),
        *K.bead(0.012, 0.071, 0.008),
        (0.017, 0.069), (0.019, 0.066, "c"), (0.022, 0.066, "l"),
        *K.bead(0.026, 0.064, 0.006),
        (0.030, 0.060), (0.042, 0.048), (0.052, 0.034), (0.060, 0.026),
        *K.bead(0.068, 0.030, 0.010),
        (0.074, 0.021), (0.078, 0.019, "c"),
        *rings(0.080, 0.104, 0.0168, 6),
        (0.110, 0.016), (0.125, 0.023), (0.140, 0.016),
        *K.bead(0.150, 0.022, 0.010),
        (0.157, 0.019), (0.160, 0.020, "c"), (0.163, 0.024, "l"), (0.167, 0.024, "l"), (0.170, 0.034),
    ]
    for k in range(1, 13):                       # the font: a squat sphere
        a = -math.pi / 2 + math.pi * k / 13
        r = 0.068 * math.cos(a) + 0.0001
        z = 0.222 + 0.050 * math.sin(a)
        body.append((z, r))
        if k == 6:                               # the spun seam round the font's equator
            body += [(z + 0.0015, r + 0.0012, "l"), (z + 0.003, r, "l")]
    body += [(0.272, 0.026), (0.276, 0.026, "l"), *K.bead(0.283, 0.029, 0.010), (0.290, 0.0001, "c")]
    K.lathe("body", K.keys(body, 0.002), seg=64, mat=brass)
    # the wick-winder: a shaft to the front and a knurled knob
    shaft = K.lathe("knob_shaft", [(0.0001, 0.0), (0.003, 0.0), (0.003, 0.03), (0.0001, 0.03)], seg=12, mat=brass)
    shaft.location, shaft.rotation_euler = (0, -0.018, 0.305), (math.pi / 2, 0, 0)
    kp = [(0.0, 0.0001, "c"), (0.0, 0.010, "c")]
    for i in range(10):                          # milled edge
        kp += [(0.0007 * i + 0.0007, 0.0122 if i % 2 == 0 else 0.0112, "l")]
    kp += [(0.005, 0.012, "l"), (0.007, 0.009), (0.007, 0.0001, "c")]
    knob = K.lathe("knob", K.keys(kp, 0.001), seg=32, mat=brass)
    knob.location, knob.rotation_euler = (0, -0.046, 0.305), (math.pi / 2, 0, 0)

if "burner" in PARTS:
    K.lathe("burner", K.keys([(0.288, 0.0001, "c"), (0.288, 0.026, "c"), (0.300, 0.026, "l"),
                              *K.bead(0.304, 0.028, 0.006), (0.315, 0.026, "l"), (0.320, 0.030)], 0.002), seg=48,
            mat=brass_dark)
    # the pierced gallery: a flared band with a ring of round holes, the globe sits in it
    gal = K.lathe("gallery", [(0.030, 0.320), (0.034, 0.322), (0.046, 0.338), (0.047, 0.343), (0.045, 0.344),
                              (0.044, 0.339), (0.032, 0.323), (0.028, 0.321), (0.030, 0.320)], seg=96, cap0=False,
                  cap1=False, mat=brass_dark)
    for i in range(18):
        a = math.tau * (i + 0.37) / 18                # off the lathe seam (slivers at x = 0)
        c = K.lathe("hole", [(0.0001, -0.02), (0.0026, -0.02), (0.0026, 0.02), (0.0, 0.02)], seg=10)
        c.location = (0.040 * math.cos(a), 0.040 * math.sin(a), 0.331)
        c.rotation_euler = (0, math.radians(127), a)          # along the flare's outward normal
        c.rotation_mode = "XYZ"
        c.hide_set(True)
        m = gal.modifiers.new("hole", "BOOLEAN")
        m.operation, m.object, m.solver = "DIFFERENCE", c, "EXACT"
    for i in range(8):                           # prongs that clip the globe
        a = math.tau * i / 8
        K.box(f"prong{i}", (0.004, 0.0015, 0.016), (0.0465 * math.cos(a), 0.0465 * math.sin(a), 0.349),
              (0, 0, a + math.pi / 2), bev=0.0004, seg=1, mat=brass_dark)
    # the dome over the wick (the flame comes up through its slot) and the flat wick in its tube
    K.lathe("dome", K.keys([(0.330, 0.0001, "c"), (0.330, 0.022, "c"), (0.336, 0.020), (0.343, 0.012),
                            (0.345, 0.0001, "c")], 0.001), seg=40, mat=brass_dark)
    K.box("wick_tube", (0.016, 0.0035, 0.006), (0, 0, 0.346), bev=0.0004, seg=1, mat=brass_dark)
    K.box("wick", (0.013, 0.0015, 0.004), (0, 0, 0.3505), bev=0.0003, seg=1,
          mat=K.fabric("felt", (0.03, 0.025, 0.02), wear=0.2, name="wick_char"))

if "globe" in PARTS:
    # opal globe: frosted milk glass, a thin transmissive shell open at top and bottom
    opal = K.glass((0.93, 0.91, 0.86), 0.5, ior=1.5, name="opal_glass" + ("_lit" if LIT else ""))
    if LIT:
        q = opal.node_tree.nodes["Principled BSDF"]
        q.inputs["Emission Color"].default_value = (1.0, 0.78, 0.52, 1)
        q.inputs["Emission Strength"].default_value = 1.6
    g_out = []
    for k in range(25):
        t = k / 24
        z = K.lerp(0.346, 0.505, t)
        r = 0.044 + 0.056 * math.sin(math.pi * min(1, t * 1.08)) ** 0.8
        if t > 0.85:
            r = K.lerp(r, 0.048, K.smooth((t - 0.85) / 0.15))
        g_out.append((r, z))
    g_in = [(r - 0.003, z) for r, z in reversed(g_out)]
    K.lathe("globe", g_out + [(g_out[-1][0] - 0.0015, g_out[-1][1] + 0.002)] + g_in + [g_out[0]], seg=96, cap0=False,
            cap1=False, mat=opal)

if "chimney" in PARTS:
    # chimney: clear glass with a 1.8 mm wall, a bulb at the bottom and a rolled rim at the top
    cg = K.glass((0.95, 0.96, 0.94), 0.02, name="chimney_glass")
    c_out = [(0.030, 0.340), (0.031, 0.350), (0.036, 0.372), (0.034, 0.395), (0.024, 0.420), (0.021, 0.445),
             (0.021, 0.592), (0.0235, 0.595), (0.024, 0.598), (0.023, 0.6005)]
    c_in = [(0.0195, 0.6005), (0.0192, 0.596)] + [(r - 0.0018, z) for r, z in reversed(c_out[:7])]
    K.lathe("chimney", c_out + c_in + [c_out[0]], seg=64, cap0=False, cap1=False, mat=cg)

if LIT and "flame" in PARTS:
    kf = K.NK("lamp_flame")
    sz = kf.N.new("ShaderNodeSeparateXYZ")
    kf.L.new(kf.obj, sz.inputs[0])
    col = kf.mix(kf.mr(sz.outputs["Z"], 0.352, 0.362), (0.35, 0.45, 1.0), (1.0, 0.56, 0.18))
    col = kf.mix(kf.mr(sz.outputs["Z"], 0.356, 0.368, 0.0, 1.0), col, (1.0, 0.9, 0.66))
    M_FLAME = kf.out(col, 0.5, 0.0, None, col, 60.0)
    fl = K.lathe("flame", K.keys([(0.352, 0.0001, "c"), (0.355, 0.0085), (0.362, 0.0115), (0.378, 0.008),
                                  (0.398, 0.0001)], 0.0015), seg=24, mat=M_FLAME)
    fl.data.transform(K.Matrix.Diagonal((1.0, 0.4, 1.0, 1.0)))    # a flat-wick flame: wide, thin
