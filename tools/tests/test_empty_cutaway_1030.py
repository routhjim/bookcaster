"""10-30: Helen's telling cut to people-free pictures whose action had people in it ("the couple exchanges rings",
"people run away"), and LTX drew strangers walking in mid-clip. A people-free shot's action may not have a person (the
director is asked again, at plan time), and its LTX prompt says nobody comes in."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

def shot(act, who=(), crowd=False):
    return {"picture": "x", "action": act, "who": list(who), "dur": 3.0, "crowd": crowd,
            "pose": {w: "standing" for w in who}}
f = lambda *ss: S._faults({"shots": list(ss)}, [], 3.0 * len(ss))
check("a people-free shot whose action has people is a fault",
      any("nobody in" in x for x in f(shot("The couple exchanges rings. The guests smile politely."))))
check("... and a hand counts", any("nobody in" in x for x in f(shot("A hand places a stack of files on the desk."))))
check("a people-free shot of the place alone is fine",
      not f(shot("Wind blows through the overgrown garden, rustling the dry weeds.")), f(shot("Wind blows through the garden.")))
check("a crowd shot is not held to it", not any("nobody in" in x for x in f(shot("People walk past.", crowd=True))))
check("a shot with people may have people in its action",
      not any("nobody in" in x for x in S._faults({"shots": [shot("The man turns to the window.", ["Holmes"])]},
                                                   ["Holmes"], 3.0)))
check("the telling rule draws only the Names", "Only people from the Names are ever drawn" in S.RULE_TOLD)
check("the empty hold line is set", "nobody comes into it" in S.EMPTY_HOLD)
print({"fails": fails})
sys.exit(1 if fails else 0)
