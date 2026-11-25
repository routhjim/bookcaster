"""11-24: each point of a figure is painted almost wholly by one view; the front drawing owns the face."""
import math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from paint_weights import view_weights
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)
names = ["front", "back", "left", "right"]
def share(deg):          # a normal turned `deg` from the front toward the left side
    a = math.radians(deg)
    W = view_weights([[math.cos(a), math.sin(a), 0.0]], names)[0]
    return W / W.sum()
check("the face (to 40 degrees) is the front drawing", share(0)[0] == 1.0 and share(40)[0] > 0.9, share(40))
check("the side of the head is the side drawing", share(70)[2] > 0.95, share(70))
check("the change-over is a narrow seam", share(42)[0] > 0.85 and share(60)[2] > 0.85, (share(42), share(60)))
check("never two views in near-equal parts outside the seam",
      all(max(share(d)) > 0.85 for d in list(range(0, 43)) + list(range(60, 91))))
check("the far side takes nothing", share(30)[3] == 0.0 and share(30)[1] == 0.0)
check("front and back only: still one view each", view_weights([[1, 0, 0], [-1, 0, 0]], ["front", "back"]).tolist() == [[4.0, 0.0], [0.0, 1.0]])
src = (HERE.parent / "blockout_model_bl.py").read_text()
check("the painter uses these weights", "W = view_weights(nrm, names)" in src)
print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
