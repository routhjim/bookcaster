"""11-25: a cut-out drawing's rim (the figure mixed with the card behind it) never colours the mesh: the colour carried
out past the outline starts inside the figure; thin parts keep their own colour; the paint step's measure of how much
of a figure no drawing shows blocks a model above its bar."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
from paint_weights import bleed, bleed_seed, erode, dilate, off_drawing, OFF_BAR, EDGE_FRAC
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# a red figure (a block with a thin green lock beside it) cut out of a grey card: soft, grey-mixed rim
h, w = 400, 300
RED, GREEN, GREY = np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.7, 0.1]), np.array([0.5, 0.5, 0.5])
body = np.zeros((h, w), bool); body[100:300, 80:200] = True
lock = np.zeros((h, w), bool); lock[120:280, 230:236] = True            # 6 px wide: thinner than twice the rim
alpha = np.zeros((h, w), np.float32)
col = np.zeros((h, w, 3), np.float32)
for m, c in ((body, RED), (lock, GREEN)):
    alpha[m] = 1.0; col[m] = c
    rim = dilate(m, 3) & ~erode(m, 2)                                   # 5 px of rim: half transparent, card-grey
    alpha[rim] = 0.7; col[rim] = (0.5 * c + 0.5 * GREY) * 0.7
seed = bleed_seed(alpha)
r = max(2, round(max(h, w) * EDGE_FRAC))
check("the seed is inside the solid figure", not (seed & (alpha < 0.98)).any())
check("on the body no pixel beside the rim seeds the fill", not (seed & body & dilate(alpha < 0.98, 1)).any())
check("the thin part keeps a seed of its own", (seed & lock).sum() > 100 and not (seed & lock & (alpha < 0.98)).any(), int((seed & lock).sum()))
out = bleed(col, seed)
grey = lambda c: float(np.abs(c - c.mean(-1, keepdims=True)).max())     # 0 = a grey
check("above the figure's top edge: the figure's colour, not the card's", np.allclose(out[96, 140], RED, atol=0.02), out[96, 140])
check("the rim itself is repainted", np.allclose(out[100, 140], RED, atol=0.02), out[100, 140])
check("beside the thin part: its colour", np.allclose(out[200, 238], GREEN, atol=0.05), out[200, 238])
check("no pixel is left near grey", min(grey(out[y, x]) for y in range(0, h, 7) for x in range(0, w, 7)) > 0.25)
check("inside is untouched", np.allclose(out[200, 140], RED))
old = bleed(col, alpha > 0.5)                                            # the fill as it was: seeded from the rim
check("(the old seed did carry the card's grey out)", grey(old[96, 140]) < 0.25, old[96, 140])
check("an image with nothing solid still has a seed", bleed_seed(np.full((20, 20), 0.6, np.float32)).all())
check("nothing wraps round the picture's edge",
      not dilate(np.pad(np.ones((1, 4), bool), ((0, 9), (0, 0))), 1)[-1].any())

W = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
ins = np.array([[1, 1], [1, 0], [0, 0], [1, 0]], bool)
check("off_drawing = weighted share landing outside", abs(off_drawing(W, ins) - 1.5 / 4) < 1e-9, off_drawing(W, ins))
check("all inside = 0", off_drawing(W, np.ones_like(ins)) == 0.0)

import characters as C
C.log = lambda *a, **k: None
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    check("no record: nothing said", C.paint_qa(d) == {})
    (d / "paint.json").write_text(json.dumps({"off_drawing": OFF_BAR - 0.05, "bar": OFF_BAR}))
    check("under the bar: recorded, not blocking", "blocking" not in C.paint_qa(d) and "paint" in C.paint_qa(d))
    (d / "paint.json").write_text(json.dumps({"off_drawing": OFF_BAR + 0.05, "bar": OFF_BAR}))
    check("over the bar: blocking", len(C.paint_qa(d).get("blocking", [])) == 1)
src = (HERE.parent / "blockout_model_bl.py").read_text()
check("the projection samples the bled colour", "col, seed = _bled(image_path, img)" in src and "bleed_seed" in src)
check("older models are painted again", C.PAINT_VERSION == "clean-rim-1125")
print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
sys.exit(1 if fails else 0)
