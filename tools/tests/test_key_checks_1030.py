"""10-30: two key checks that failed every seed of good pictures. A pale sky along the top edge read as a white
sheet (Stoke Moran exteriors); a standing close-up beside a bed failed "a bed right behind or under him"."""
import sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import mannequin_flow as MF
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

d = Path(tempfile.mkdtemp(prefix="keychk_"))
def pic(name, fill):
    a = np.full((704, 1280), 90, np.uint8)
    fill(a)
    Image.fromarray(a).save(d / name)
    return d / name
def sky(a): a[:260] = 235                       # an overcast sky over a dark house
def sheet(a): a[:] = 240; a[100:600, 400:880] = 60  # a pose on a white sheet
def panel(a): a[:, 900:] = 245                  # a white text panel down the right side
check("a pale sky along the top is not a sheet", not MF._blank_edges(pic("sky.png", sky)))
check("a white sheet still is", MF._blank_edges(pic("sheet.png", sheet)))
check("a white side panel still is", MF._blank_edges(pic("panel.png", panel)))
check("a dark room is not", not MF._blank_edges(pic("room.png", lambda a: None)))

class Cx:
    book = Path(tempfile.mkdtemp())
S.RG.Book.look = lambda self, w: "a man of about 40, wearing a dark frock coat"
S.video_dress = lambda *a, **k: ""
q, _ = S._key_questions(Cx, {"id": "talk-001095_s0", "who": ["Sherlock Holmes"], "kind": "talk", "picture": "",
                            "framing": "close", "pose_held": {"Sherlock Holmes": "stand"}})
ch = [v for k, v in q.items() if k.startswith("chair")]
check("a standing close-up asks whether he SITS or LIES on the furniture, not whether a bed is behind him",
      ch and "sitting or lying on a chair, sofa or bed" in ch[0] and "behind" not in ch[0], q)
print({"fails": fails})
sys.exit(1 if fails else 0)
