"""11-25: the arms are painted by the front and back drawings alone: the shape holds them out, the side drawings
show them hanging, so a side drawing projected onto an arm paints the wall behind the figure."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paint_weights import arms_from_front_back, view_weights

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

names = ["front", "back", "left", "right"]
# a figure 2 m tall facing +X: legs 0.2 m either side of the middle, then single points to test
legs = [[0.0, y, z] for z in np.linspace(0.2, 0.8, 30) for y in (-0.2, -0.1, 0.1, 0.2)]
pts = [[0.0, 0.6, 1.3],     # 0 the forearm, far out
       [0.0, -0.6, 1.3],    # 1 the other forearm
       [0.0, 0.18, 1.3],    # 2 the flank of the chest
       [0.0, 0.6, 0.85],    # 3 far out but below the hips (a wide skirt)
       [0.0, 0.245, 1.3],   # 4 in the seam
       [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
co = np.array(pts + legs)
W = arms_from_front_back(np.ones((len(co), 4)), names, co)
check("the arm takes nothing from the side drawings", W[0, 2] == 0 and W[0, 3] == 0 and W[1, 2] == 0, W[:2])
check("it keeps the front and the back", W[0, 0] == 1 and W[0, 1] == 1)
check("the flank of the chest keeps its side paint", W[2, 2] == 1 and W[2, 3] == 1, W[2])
check("a wide skirt keeps its side paint", W[3, 2] == 1, W[3])
check("a soft seam between", 0 < W[4, 2] < 1, W[4])
check("the legs are untouched", (W[len(pts):] == 1).all())
check("front and back only: untouched", (arms_from_front_back(np.ones((len(co), 2)), ["front", "back"], co) == 1).all())
off = co + np.array([0.0, 3.0, 0.0])
check("a figure standing off the middle is measured about itself", np.allclose(arms_from_front_back(np.ones((len(co), 4)), names, off), W, atol=1e-6))
src = (HERE.parent / "blockout_model_bl.py").read_text()
check("the painter applies it before the weights are summed", src.index("arms_from_front_back(W, names") < src.index("W /= W.sum(1, keepdims=True)"))
print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
