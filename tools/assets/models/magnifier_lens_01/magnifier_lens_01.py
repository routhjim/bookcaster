"""Model script: magnifying glass lying flat along X: glass lens, brass rim and ferrule, turned ebony handle."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

brass = K.metal("brass", tarnish=0.6, polish=0.8)
ebony = K.wood("ebonised", "Z", wear=0.5, gloss=0.7, grime=0.5)
lensg = K.glass((0.95, 0.96, 0.94), 0.005, name="lens_glass")
R, Z = 0.0375, 0.008
rim = K.lathe("rim", K.keys([(-0.006, R - 0.001, "c"), (-0.006, R + 0.004, "c"), (-0.004, R + 0.006), (0.004, R + 0.006),
                             (0.006, R + 0.004, "c"), (0.006, R - 0.001, "c"), (-0.006, R - 0.001, "c")], 0.001), seg=64, cap0=False,
              cap1=False, mat=brass)   # a closed ring section: caps here would be brass discs over the lens
rim.location = (0.08, 0.0, Z)
lens = K.lathe("lens", [(0.0001, -0.0045), (R * 0.5, -0.004), (R, -0.0015), (R, 0.0015), (R * 0.5, 0.004), (0.0001, 0.0045)],
               seg=64, mat=lensg, cap0=False, cap1=False)
lens.location = (0.08, 0.0, Z)
fer = K.lathe("ferrule", K.keys([(0.0, 0.0001, "c"), (0.0, 0.006, "c"), (0.022, 0.006, "l"), *K.bead(0.026, 0.0085, 0.006),
                                 (0.030, 0.0001, "c")], 0.001), seg=20, mat=brass)
fer.location, fer.rotation_euler = (0.08 - R - 0.003, 0.0, Z), (0, -math.pi / 2, 0)
han = K.lathe("handle", K.keys([(0.0, 0.0001, "c"), (0.0, 0.0075, "c"), (0.01, 0.0085), (0.06, 0.0080), (0.095, 0.0095),
                                (0.11, 0.0080), *K.bead(0.115, 0.0085, 0.006), (0.12, 0.0001, "c")], 0.002), seg=24, mat=ebony)
han.location, han.rotation_euler = (0.08 - R - 0.030, 0.0, Z), (0, -math.pi / 2, 0)
