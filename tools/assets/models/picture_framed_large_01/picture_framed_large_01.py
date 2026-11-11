"""Model script: Victorian pastoral oil landscape (cattle by a stream, cloudy sky), yellowed varnish, in a deep gilt frame, 0.85 x 0.65 m, for over a mantelpiece. Frame, mount and picture from period_kit.framed_picture; the picture is
_refs/<id>/picture.png in the asset library (drawn by the image model), mapped by the picture part's coordinates."""

import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

P = SPEC["picture"]  # noqa: F821
LIB = os.environ.get("ASSETS_LIBRARY") or os.path.join(os.path.expanduser("~"), "ue", "library")
K.framed_picture(SPEC["id"], P["w"], P["h"], P["frame_w"], P["mount_w"], 0.045,  # noqa: F821
                 os.path.join(LIB, "_refs", SPEC["id"], "picture.png"), P["oil"])  # noqa: F821
