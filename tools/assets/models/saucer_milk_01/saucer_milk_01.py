"""Model script: earthenware saucer (lathe, glaze with a blue rim band) holding a shallow pool of milk."""

import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

blue = (0.035, 0.07, 0.22)
cer = K.ceramic((0.62, 0.58, 0.48), glaze=0.85, craze=0.35, pattern=(blue, [(0.0175, 0.0215)]))
milk = K.ceramic((0.70, 0.69, 0.64), glaze=0.6, craze=0.0, name="milk")
# outer surface bottom -> rim -> inner surface back to the centre (a closed solid via the lathe point ends)
prof = [(0.0001, 0.0045), (0.040, 0.0045), (0.042, 0.0), (0.046, 0.0), (0.048, 0.004), (0.058, 0.0075),
        (0.068, 0.014), (0.0725, 0.0225), (0.0718, 0.024), (0.069, 0.0225), (0.063, 0.017), (0.052, 0.0105),
        (0.042, 0.0075), (0.0001, 0.0068)]
sau = K.lathe("saucer", prof, seg=72, cap0=False, cap1=False, mat=cer)
pool = K.lathe("milk", [(0.0001, 0.0066), (0.050, 0.0066), (0.051, 0.0095), (0.0001, 0.0098)], seg=48, cap0=False,
               cap1=False, mat=milk)
