"""10-26 Acts 2 review (user): "'Tongues of Fire' needs to just be flames on top of people .. no weird tongue thing";
"the start of the chapter, everyone still has fire on their heads and that should only be at the specific scene".
Real director answers from that render (fixtures/fire_1026) against the new checks; CPU only.
    ~/tts_reader/.venv/bin/python3 tools/tests/test_fire_1026.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as S
FX = Path(__file__).parent / "fixtures" / "fire_1026"
tl = json.loads((FX / "timeline.json").read_text())
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)
def ans(m):
    return json.loads((FX / f"{m}.json").read_text())

t0 = S.timed(tl, 0.0, 5.0)
fire_at = [t for t, z in S.timed(tl, 0.0, 22.0) if "fire" in z]
check("timed: the fire sentence is well after the chapter's start", fire_at and fire_at[0] > 8.0, fire_at)
check("timed: the opening stretch's narration has no fire", not any("fire" in z for _, z in t0), t0)
f0 = S._passing(ans("action-000000"), t0, "")
check("0:00 shots with flames descending: faulted (before the narration has it)", any("fire" in x for x in f0), f0)
st = S._passing(ans("action-000213"), S.timed(tl, 21.3, 5.0), "a cobbled street between stone houses")
check("street shots with flames on their heads: faulted (not in this stretch's narration)", any("never has it" in x for x in st), st)
st2 = S._passing(ans("action-000413"), S.timed(tl, 41.3, 5.0), "")
check("Peter speaking in the street with flames on his hair: faulted", bool(st2), st2)
# the scene itself: shots at/after "divided tongues as of fire appeared" keep their flames
t_scene = S.timed(tl, fire_at[0] - 1.0, 6.0)
ok_scene = {"shots": [{"dur": 3.0, "picture": "three seated men", "action": "small flames descend from above and rest over their heads"},
                      {"dur": 3.0, "picture": "small flames above their heads", "action": "the flames glow steadily"}]}
check("the fire's own scene: no fault", S._passing(ok_scene, t_scene, "") == [], S._passing(ok_scene, t_scene, ""))
check("a forge's fire in the room: no fault", S._passing({"shots": [{"dur": 3, "picture": "a smith at the fire",
      "action": "sparks from the hammer; the forge fire glows"}]}, [(0.0, "He took up the hammer.")],
      "a stone forge with a glowing fire at the back") == [])
# figurative wording
check("figurative: tongues of fire -> small flames",
      S.figurative("Three distinct tongues of fire descend and rest on their heads") ==
      "Three distinct small flames descend and rest on their heads")
check("figurative: divided tongues as of fire", "tongue" not in S.figurative("divided tongues as of fire appeared"))
check("figurative: speaking in tongues -> talking eagerly",
      S.figurative("the men are speaking in tongues") == "the men are talking eagerly")
fa = S._faults(ans("action-000100"), ["Peter", "Disciple", "Early Believer"], 5.0)
check("'tongue' in a shot is a fault for the director", any("tongue" in x for x in fa), fa)
check("the rule text tells the director the times", "[seconds from the passage's start]" in S.PASSAGE_ASK and
      "WHEN:" in S.SHOT_RULES and any("tongues of fire" in r for _, r in S.RULE_EXTRA))
print(json.dumps({"fails": fails}))
