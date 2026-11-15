"""Model script: small baluster porcelain vase (lathe shell with an inner wall), blue glaze, gilt bands.
(No separate reserve panel: unpainted it showed as a bare white patch; decoration is left to the paint pass.)"""

import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

gold = (0.45, 0.32, 0.10)
por = K.ceramic((0.020, 0.035, 0.12), glaze=0.92, craze=0.1, pattern=(gold, [(0.0, 0.006), (0.196, 0.21)]), name="blue_porcelain")
outer = [(0.0001, 0.003), (0.030, 0.003), (0.031, 0.0), (0.034, 0.0), (0.036, 0.008), (0.044, 0.030), (0.048, 0.060),
         (0.045, 0.095), (0.035, 0.125), (0.022, 0.150), (0.020, 0.165), (0.026, 0.188), (0.036, 0.205), (0.038, 0.209)]
inner = [(0.035, 0.209), (0.033, 0.204), (0.024, 0.189), (0.017, 0.166), (0.017, 0.150), (0.0001, 0.145)]
K.lathe("vase", outer + inner, seg=64, cap0=False, cap1=False, mat=por)
