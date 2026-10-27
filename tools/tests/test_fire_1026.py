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
# 10-27 audit: the passing-effect check was this chapter's band-aid (it faulted Baker Street's lit hearth); the
# director now carries how things stand per scene and the flames belong to their scene's narration
check("no passing-effect check any more (a lit hearth is never a fault)", not hasattr(S, "_passing"))
# figurative wording
check("figurative: tongues of fire -> small flames",
      S.figurative("Three distinct tongues of fire descend and rest on their heads") ==
      "Three distinct small flames descend and rest on their heads")
check("figurative: divided tongues as of fire", "tongue" not in S.figurative("divided tongues as of fire appeared"))
check("figurative: speaking in tongues -> talking eagerly",
      S.figurative("the men are speaking in tongues") == "the men are talking eagerly")
fa = S._faults(ans("action-000100"), ["Peter", "Disciple", "Early Believer"], 5.0)
check("'tongue' wording is redrawn by figurative(), not a fault of one story", not any("tongue" in x for x in fa), fa)
check("the rule text tells the director the times; the figurative rule is generic", "[seconds from the passage's start]"
      in S.PASSAGE_ASK and "WHEN:" in S.SHOT_RULES and any("Figurative words" in r for _, r in S.RULE_EXTRA)
      and not any("tongues of fire" in r for _, r in S.RULE_EXTRA))
print(json.dumps({"fails": fails}))
