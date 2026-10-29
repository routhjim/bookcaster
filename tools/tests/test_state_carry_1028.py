"""Poses carry from shot to shot within a scene (10-29); the 10-28 at_end/facing_end carry is gone."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# 10-29: the at_end/facing_end carry is gone (never a real move in 256 rows; postures carried instead)
check("the director is not asked for at_end/facing_end", "at_end" not in S.SHOT_RULES and "facing_end" not in S.SHOT_RULES)
check("no ending words for LTX", not hasattr(S, "_ends"))
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
c3, s3 = {}, {}
S.compose({"kind": "quiet", "people": [{"name": "Holmes", "side": "left", "pose": "standing", "at": "by the fire",
           "facing": "Watson"}], "setting": "", "action": "He waits."}, cast2, [], s3, res2, c3)
st2 = S.compose({"kind": "quiet", "people": [{"name": "Holmes", "side": "left", "pose": "standing", "at": "by the door",
                 "facing": "the window"}], "setting": "", "action": "He waits."}, cast2, [], s3, res2, c3)
check("each shot's place and facing are the director's own (only the pose carries)",
      "by the door" in st2["picture"] and "facing the window" in st2["picture"] and c3["Sherlock Holmes"] == {"pose": "stand"},
      (st2["picture"], c3))
print({"fails": fails})
sys.exit(1 if fails else 0)
