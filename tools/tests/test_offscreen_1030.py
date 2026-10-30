"""10-30: quiet shots of one person whose action named another ("watches the man in the white nightshirt", "as the men
approach", "as the man in the dark frock coat speaks") grew a second Holmes, Watson or Helen walking in. Whoever is not
in the picture becomes "someone out of the picture", and the prompt says nobody else comes in."""
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

b = Path(tempfile.mkdtemp(prefix="offscreen_"))
(b / "work" / "video").mkdir(parents=True)
(b / "work" / "video" / "look.json").write_text(json.dumps({"characters": {
    "Sherlock Holmes": "a man of about 40, wearing a dark Victorian frock coat",
    "Doctor Watson": "a man of about 35, wearing a white nightshirt",
    "Helen Stoner": "a woman of about 30, wearing a black mourning dress"}, "places": {}}))
cx = SimpleNamespace(book=b, stem="01_x")
S.MF._dress = lambda *a, **k: {}
R_ = S.RG.Book(b)
holmes, watson, helen = (S.describe(R_, w) for w in ("Sherlock Holmes", "Doctor Watson", "Helen Stoner"))
a = S._offscreen(cx, {"id": "action-000759_s0", "who": ["Sherlock Holmes"]},
                 f"{holmes} watches {watson} with a steady expression.")
check("the one not in the picture is someone out of it", watson not in a and "watches someone out of the picture" in a, a)
check("the one in the picture keeps his description", a.startswith(holmes), a)
check("the prompt says nobody else comes in (one person)", "Only the one person already in the picture is ever in it" in a, a)
a = S._offscreen(cx, {"id": "action-001496_s0", "who": ["Helen Stoner"]}, f"{helen} rises as the men approach.")
check("'the men' are out of the picture too", "the men" not in a and "people out of the picture approach" in a, a)
a = S._offscreen(cx, {"id": "action-001146_s0", "who": ["Doctor Watson", "Sherlock Holmes"]},
                 f"{watson} nods as {holmes} speaks.")
check("two in the picture: both kept, said as 2 people", holmes in a and watson in a and "Only the 2 people" in a, a)
check("a crowd shot is left alone", S._offscreen(cx, {"who": ["Helen Stoner"], "crowd": True}, "x") == "x")
check("a picture with nobody in it is left alone", S._offscreen(cx, {"who": []}, "x") == "x")
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
