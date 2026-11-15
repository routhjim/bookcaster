"""Model script: a flat painted backdrop seen through a set's windows (the far side of a street), one picture over
a plane. The picture is _refs/<id>/picture.png in the asset library (drawn by the image model). Built flat (picture
in XY, so the image material's generated coordinates span it), then stood upright: front -Y, bottom edge on Z = 0.
Spec: "backdrop": {"w": metres, "h": metres}."""

import math
import os
import sys

from mathutils import Matrix

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

B = SPEC["backdrop"]  # noqa: F821
LIB = os.environ.get("ASSETS_LIBRARY") or os.path.join(os.path.expanduser("~"), "ue", "library")
w, h = float(B["w"]), float(B["h"])
pic = K.box(f"{SPEC['id']}_pic", (w, h, 0.02), (0, 0, 0.01), bev=0.0,  # noqa: F821
            mat=K.image_material(os.path.join(LIB, "_refs", SPEC["id"], "picture.png"), f"{SPEC['id']}_print"))  # noqa: F821
pic.matrix_world = Matrix.Translation((0, 0, h / 2)) @ Matrix.Rotation(math.pi / 2, 4, "X") @ pic.matrix_basis
