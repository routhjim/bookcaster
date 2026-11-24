"""11-23: the cast reply's room grows with the cast (a fixed cap cut a ten-part reply short, the JSON did not parse and
every voice stayed on its default); an unreadable reply is asked for again and then warned about; spoken titles
give a gender; everyone with 8+ lines gets their own voice, up to 20."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tts_reader import casting as C

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

check("spoken titles give a gender", (C._noun_gender("Missus Hale"), C._noun_gender("Mister Hale"),
                                      C._noun_gender("Hale")) == ("F", "M", ""))
check("ten parts by default", C.cast_size([50, 9, 9, 3, 2], 10) == 10)
check("everyone with 8+ lines gets a voice", C.cast_size([40] * 13 + [3] * 9, 10) == 13)
check("at most twenty", C.cast_size([40] * 30, 10) == 20)

ROSTER = [("nar", "Male", "narrator"), ("min", "Male", "minor"), ("va", "Female", "a"), ("vb", "Male", "b")]
def session(replies):
    s = C.CastSession(engine="f5", llm_url="http://x", llm_model="m", custom_roster=list(ROSTER))
    s.members = [C.CastMember(name="Missus Hale", quotes=9, samples=["x"], contexts=[]),
                 C.CastMember(name="Tam", quotes=8, samples=["y"], contexts=[])]
    s.narrator_voice, s.dialogue_voice = "nar", "min"
    for m, v in zip(s.members, ["vb", "va"]):
        m.voice = v
    seen = []
    def chat(prompt, max_tokens=900):
        seen.append(max_tokens)
        return replies.pop(0)
    s._chat = chat
    return s, seen
GOOD = json.dumps({"narrator": "nar", "dialogue": "min", "characters": {
    "Missus Hale": {"role": "r", "gender": "female", "voice": "va", "why": "w", "same_as": None},
    "Tam": {"role": "r", "gender": "male", "voice": "vb", "why": "w", "same_as": None}}})
logs = []
s, seen = session([GOOD[:120], GOOD])
s.describe_and_suggest("text", log=logs.append)
check("a cut-off reply is asked for again with more room", len(seen) == 2 and seen[1] > seen[0] > 600, seen)
check("the second reply sets the voices", [m.voice for m in s.members] == ["va", "vb"], [m.voice for m in s.members])
logs.clear()
s, seen = session(["nothing", "still nothing"])
s.describe_and_suggest("text", log=logs.append)
check("two unreadable replies are warned about, not passed over", any("defaults" in x for x in logs), logs)

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
