"""10-28 user: "the director and the key frame have to not contradict each other". Each shot opens where the shot
before LEFT everyone (at_end / facing_end carried, like poses); the LTX action words say where a move ends."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

cast = ["Sherlock Holmes", "Helen Stoner"]
res = lambda v: next((w for w in cast if v and v.split()[-1] in w), None)
carry, sides = {}, {}
a = S.compose({"kind": "action", "people": [
        {"name": "Holmes", "side": "right", "pose": "standing", "at": "by the fire", "facing": "Helen Stoner",
         "at_end": "at the desk", "facing_end": "the desk"},
        {"name": "Helen Stoner", "side": "left", "pose": "sitting", "at": "in the armchair", "facing": "Holmes"}],
    "setting": "The desk on the right of the picture.", "action": "He turns to his right and crosses to the desk."},
    cast, [], sides, res, carry)
check("the opening picture is the moment before the move (by the fire, facing Helen)",
      "by the fire" in a["picture"] and "facing Helen Stoner" in a["picture"] and "desk," not in a["picture"].split(".")[0],
      a["picture"])
check("the end is carried: Holmes left at the desk, facing it", {k: v for k, v in carry.get("Sherlock Holmes", {}).items()
      if k != "pose"} == {"at": "at the desk", "facing": "the desk"}, carry)
b = S.compose({"kind": "action", "people": [
        {"name": "Holmes", "side": "right", "pose": "standing", "at": "by the window", "facing": "the door"}],
    "setting": "", "action": "He opens the drawer."}, cast, [], sides, res, carry)
check("the next shot OPENS where he was left, whatever the director re-wrote (at the desk, facing the desk)",
      "at the desk" in b["picture"] and "facing the desk" in b["picture"] and "window" not in b["picture"], b["picture"])
check("someone the scene hasn't moved keeps the director's own opening", "in the armchair" in a["picture"], a["picture"])
class CX: book = None
S.describe = lambda R, w, d: "the tall man in a dark frock coat"
S.RG.Book = lambda b: None
S._shot_dress = lambda cx, s: {}
e = S._ends(CX, a)
check("the LTX words say where the move ends", "ends at the desk, facing the desk" in e, e)
check("nobody who doesn't move gets an ending", S._ends(CX, b) == "", S._ends(CX, b))
# 10-29 user: "Watson should just remain in bed and sit up"
cast2 = ["Doctor Watson", "Sherlock Holmes"]
res2 = lambda v: next((w for w in cast2 if v and v.split()[-1] in w), None)
c2, s2 = {}, {}
S.compose({"kind": "quiet", "people": [{"name": "Watson", "side": "right", "pose": "lying", "at": "in bed",
                                        "facing": "the ceiling"}], "setting": "", "action": "He blinks."},
          cast2, [], s2, res2, c2)
up = S.compose({"kind": "quiet", "people": [{"name": "Watson", "side": "right", "pose": "sitting", "at": "in bed",
                                             "facing": "Holmes"}], "setting": "",
                "action": "He pushes himself up against the headboard."}, cast2, [], s2, res2, c2)
check("the sit-up shot opens him lying, as the shot before left him", up["pose"].get("Doctor Watson") == "lying"
      and "lies in bed" in up["picture"], (up["pose"], up["picture"]))
check("... and ends sitting: the action makes the change", up["pose_end"].get("Doctor Watson") == "sitting"
      and c2["Doctor Watson"]["pose"] == "sit", (up["pose_end"], c2))
nx = S.compose({"kind": "quiet", "people": [{"name": "Watson", "side": "right", "pose": "sitting", "at": "in bed",
                                             "facing": "Holmes"}], "setting": "", "action": "He frowns."},
               cast2, [], s2, res2, c2)
check("sitting in bed is sitting UP in bed under the covers (not on its edge or in a chair)",
      "sits up in bed, under the covers" in nx["picture"] and nx["pose"]["Doctor Watson"] == "sitting", nx["picture"])
check("someone not yet shown keeps the director's pose", S.compose({"kind": "quiet", "people": [
      {"name": "Holmes", "side": "left", "pose": "standing", "at": "beside the bed", "facing": "Watson"}],
      "setting": "", "action": "He waits."}, cast2, [], s2, res2, c2)["pose"]["Sherlock Holmes"] == "standing")
print({"fails": fails})
sys.exit(1 if fails else 0)
