"""10-27: the LoRAs learned old faces from the book's older pictures (17 of Watson's 39 training pictures were drawn
from his clean-shaven 10-15 look) and other people's faces from crowd pictures; a knight-book sword pose gave Roylott
a sword; outfit sheets kept the base look's top hat. charlora-2 rules."""
import json, os, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import charlora as L
from tts_reader import video

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

d = Path(tempfile.mkdtemp(prefix="lorafresh_"))
ch = d / "01_x"; (ch / "img").mkdir(parents=True); (ch / "anim").mkdir()
def png(p, age):
    p.write_bytes(b"x"); t = time.time() - age; os.utime(p, (t, t)); return p
old_alone = png(ch / "img" / "p001.png", 3600)        # drawn before the current sheet
new_alone = png(ch / "img" / "p002.png", 10)          # after it, the person alone
new_crowd = png(ch / "img" / "p003.png", 10)          # after it, with someone else
sheet = png(d / "sheet.png", 60)
plans = {"01_x": {"pictures": [{"id": "p001", "who": ["Watson"]}, {"id": "p002", "who": ["Watson"]},
                               {"id": "p003", "who": ["Watson", "Holmes"]}]}}
R = SimpleNamespace(same=lambda a, b: str(a).split()[-1] == str(b).split()[-1])
c = SimpleNamespace(chapters=[ch], plans=plans, R=R, name="Doctor Watson", sheet=lambda oid: sheet)
cands = []
L.cand_book(c, cands)
got = sorted(Path(x["path"]).name for x in cands)
check("book pictures: only the person ALONE, drawn after their current sheet", got == ["p002.png"], got)
check("the recipe version changed: every LoRA rebuilds once", L.TOOL_VERSION == "charlora-2")
check("no sword pose for someone without a sword", all(x in L.PROPS for x in ("sword", "hammer", "shield")))
p = video.outfit_prompt("Roylott", "a man of about 60", {"dress": "a long grey dressing-gown, red slippers"}, "s")
check("an outfit sheet: these clothes in both views, a hat only if written", "in BOTH views" in p and
      "a hat or other headwear only if written here" in p, p)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
