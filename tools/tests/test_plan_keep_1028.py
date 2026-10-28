"""10-28: (1) a saved plan is kept across a render-side fix (the plan signature covers the planning code only);
(2) a described place keeps its description (no rewrite from each run's set sheet, no re-ask); (3) the one-person
close-up names nobody beside the viewer (it drew an over-the-shoulder listener)."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S
from tts_reader import animate

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

reach = S._plan_reach()
check("the plan's code covers the director, compose, keys and the shot templates",
      {"direct", "compose", "keys_judged", "describe_places", "TALK_PICTURE", "SHOT_RULES"} <= reach)
check("the render's code is not in the plan's signature",
      not reach & {"ltx", "_stitch_talk", "assemble", "judge_takes", "_render", "_judge_as_they_land"},
      reach & {"ltx", "_stitch_talk", "assemble", "judge_takes", "_render", "_judge_as_they_land"})
h0 = S._plan_code()
real_ltx = S.ltx
S.ltx = lambda *a, **k: None
S.ltx.__module__ = S.__name__
check("a render-side change keeps the plan", S._plan_code() == h0)
S.ltx = real_ltx
real = S.TALK_PICTURE
S.TALK_PICTURE = real + " x"
check("a template change plans again", S._plan_code() != h0)
S.TALK_PICTURE = real

b = Path(tempfile.mkdtemp(prefix="places_")); vd = b / "work" / "video"; vd.mkdir(parents=True)
look = {"characters": {}, "places": {"Baker Street Sitting Room": "Inside: a sitting room with two armchairs",
                                     "Stoke Moran Manor": "Outside: a grey stone manor"},
        "_described": {"Baker Street Sitting Room": "room:aaaa", "Stoke Moran Manor": S.MF._h(S.PLACE_ASK)}}
(vd / "look.json").write_text(json.dumps(look))
animate.igpu = lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked the language model"))
shots = [{"id": "m1_s0", "place": "Baker Street Sitting Room (inside)", "room": "a different room, a bay window"},
         {"id": "m2_s0", "place": "Stoke Moran Manor (outside)", "room": "an old house with ivy"}]
es = [{"id": "m1", "t": 0.0, "dur": 3.0}, {"id": "m2", "t": 3.0, "dur": 3.0}]
try:
    S.describe_places(b, "01_x", es, shots, log=lambda m: None)
    asked = False
except AssertionError:
    asked = True
after = json.loads((vd / "look.json").read_text())["places"]
check("a described place keeps its words (the set sheet's room doesn't rewrite it)", after == look["places"], after)
check("a place the set sheet described is not asked again", not asked)
check("the close-up names nobody beside the viewer", "beside the viewer" not in S.TALK_PICTURE and
      "toward someone" not in S.TALK_PICTURE and "alone" in S.TALK_PICTURE, S.TALK_PICTURE)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
