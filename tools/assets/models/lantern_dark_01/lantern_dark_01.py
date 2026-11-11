"""Model script: bull's-eye dark lantern, front -Y (SPEC["variant"] "lit": glowing lens interior and vents).

Body, hood and chimney are lathe turnings about Z; the lens tube is a lathe along -Y with a glass plano-convex lens;
handles are swept tubes / a bent strap; vent holes are dark slots on the chimney cap."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

LIT = SPEC.get("variant") == "lit"  # noqa: F821
tin = K.metal("iron", tarnish=0.4, polish=0.75, rust=0.2, soot=0.6, soot_z=(0.20, 0.26))
brass = K.metal("brass", tarnish=0.7, polish=0.6)
lensg = K.glass((0.92, 0.93, 0.88), 0.01, name="lens_glass")
inside = K.emissive((1.0, 0.62, 0.25), 6.0, base=(0.3, 0.2, 0.1), name="lamp_glow") if LIT else \
    K.metal("nickel", tarnish=0.5, polish=0.2, soot=0.4)
dark = K.emissive((1.0, 0.55, 0.2), 2.0, base=(0.05, 0.03, 0.02), name="vent_glow") if LIT else \
    K.fabric("linen", (0.006, 0.005, 0.005), wear=0.0)
R = 0.055
body = K.lathe("body", K.keys([(0.0, 0.0001, "c"), (0.0, R - 0.002, "c"), (0.003, R, "l"), *K.bead(0.008, R + 0.003, 0.006),
                               (0.014, R), (0.150, R, "l"), *K.bead(0.155, R + 0.003, 0.006), (0.160, R + 0.001),
                               (0.165, R + 0.006), (0.168, R + 0.006, "l"), (0.205, 0.028), (0.21, 0.022, "l"),
                               (0.21, 0.0001, "c")], 0.003), seg=48, mat=tin)
chim = K.lathe("chimney", K.keys([(0.205, 0.0001, "c"), (0.205, 0.022, "c"), (0.24, 0.022, "l"), (0.243, 0.030),
                                  (0.252, 0.030, "l"), (0.258, 0.016), (0.262, 0.0001, "c")], 0.002), seg=32, mat=tin)
for k in range(8):                                    # vent slots on the chimney
    a = math.tau * k / 8
    K.box(f"vent{k}", (0.008, 0.004, 0.022), (0.0215 * math.cos(a), 0.0215 * math.sin(a), 0.225), (0, 0, a + math.pi / 2),
          bev=0.0, mat=dark)
# lens tube along -Y
LZ = 0.085
tube = K.lathe("lens_tube", K.keys([(0.0, 0.0001, "c"), (0.0, 0.040, "c"), (0.035, 0.040, "l"), (0.038, 0.044),
                                    (0.045, 0.044, "l"), (0.046, 0.036, "c"), (0.046, 0.0001, "c")], 0.002), seg=48, mat=brass)
tube.location, tube.rotation_euler = (0, -R + 0.006, LZ), (math.pi / 2, 0, 0)
lens = K.lathe("lens", [(0.0001, 0.0), (0.036, 0.0), (0.034, 0.006), (0.026, 0.012), (0.014, 0.016), (0.0001, 0.017)],
               seg=48, mat=lensg)
lens.location, lens.rotation_euler = (0, -R - 0.040, LZ), (math.pi / 2, 0, 0)
# the inside seen through the lens: reflector / lamp glow disc
glow = K.lathe("reflector", [(0.0001, 0.0), (0.034, 0.0), (0.034, 0.002), (0.0001, 0.002)], seg=32, mat=inside)
glow.location, glow.rotation_euler = (0, -R - 0.002, LZ), (math.pi / 2, 0, 0)
# ring handle on top, strap handle at the back
ring = K.tube("ring", [(0.0, 0.03 * math.cos(a), 0.262 + 0.022 + 0.022 * math.sin(a)) for a in [math.tau * i / 24 for i in range(24)]],
              0.0035, seg=8, closed=True, mat=tin)
strap = K.sweep("strap", K.bezier((0, R - 0.002, 0.14), (0, R + 0.06, 0.13), (0, R + 0.06, 0.05), (0, R - 0.002, 0.04), 20),
                [K.superellipse2d(0.010, 0.0016, 6.0, 0.0, 12) for _ in range(21)], up=(1, 0, 0), mat=tin)
# shutter knob on the side
kn = K.lathe("shutter_knob", K.keys([(0.0, 0.0001, "c"), (0.0, 0.006, "c"), (0.012, 0.005), (0.016, 0.009), (0.02, 0.0001)],
                                    0.0015), seg=16, mat=brass)
kn.location, kn.rotation_euler = (R - 0.002, 0, 0.18), (0, math.pi / 2, 0)
