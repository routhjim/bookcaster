"""11-05: a story's rooms were listed as CHARACTERS as well as places (look.characters and look.json), so the shot
flow would train "creature" LoRAs for them and the director cast them. A place is never a character: the planning
ask says so, and its answer, the look registry it fills (keep_looks) and the registry's people all drop an entry
whose name is one of the places or whose description is written as a place's ("Inside:"/"Outside:"). CPU only, a
SYNTHETIC book, the language model faked (no service is called).
    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_cast_no_places_1105.py"""
import json
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import registry as RG                   # noqa: E402
from tts_reader import video as V       # noqa: E402
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


places = {"The Workshop": "Inside: a long low room with benches", "North Tower (outside)": "Outside: a grey stone tower"}
check("a name that is one of the places", RG.place_named("the workshop", "a man of 40", places))
check("a side-marked place name matches its place", RG.place_named("North Tower", "", places))
check("a description written as a place's", RG.place_named("The Yard", "Outside: a muddy yard by a wall", {}))
check("a person is not a place", not RG.place_named("Ada Brook", "a woman of about 30, Inside: no", places)
      and not RG.place_named("Ada Brook", "a woman of about 30", places))
check("a creature is not a place", not RG.place_named("Wyrm", "a great dragon with black scales", places))

chars = {"Ada Brook": "a woman of about 30", "Wyrm": "a great dragon",
         "The Workshop": "Inside: a long low room with benches", "Bell Tower": "Outside: a tower of red brick"}
with tempfile.TemporaryDirectory() as td:
    b = Path(td)
    (b / "work" / "video").mkdir(parents=True)
    (b / "work" / "video" / "look.json").write_text(json.dumps({"characters": chars, "places": places}))
    R = RG.Book(b)
    check("registry: the places listed as characters are not people", R.people() == ["Ada Brook", "Wyrm"], R.people())
    check("registry: a place's name is never a person", R.person("The Workshop") is None and R.person("Bell Tower") is None)
    check("registry: the place itself is still a place", R.place_look("The Workshop") == places["The Workshop"])

    # the planning ask: its answer's characters lose the places; the ask itself says so (no example)
    reply = {"look": {"characters": dict(chars), "places": dict(places)},
             "pictures": [{"start": 0, "who": ["Ada Brook"], "where": "The Workshop", "crowd": "", "prompt": "Ada Brook sits."}]}
    asked = []
    V._chat = lambda url, model, prompt, max_tokens=0: (asked.append(prompt), json.dumps(reply))[1]
    V.check_looks = lambda c, *a, **k: c
    p = V.plan_pictures(["Ada Brook sat in the workshop."], "http://x", "m", "", 1)
    check("plan: the answer's characters are the people and creatures only",
          sorted(p["look"]["characters"]) == ["Ada Brook", "Wyrm"] and sorted(p["look"]["places"]) == sorted(places),
          p["look"])
    check("plan: the ask says a place is never a character", asked and "is never a character" in asked[0])

    # keep_looks: a plan that still lists a place as a character never puts it in the registry
    reg = b / "work" / "video" / "look2.json"
    plan = {"look": {"characters": dict(chars), "places": dict(places)}}
    V.keep_looks(plan, reg)
    got = json.loads(reg.read_text())
    check("keep_looks: the registry's characters are the people and creatures only",
          sorted(got["characters"]) == ["Ada Brook", "Wyrm"] and sorted(plan["look"]["characters"]) == ["Ada Brook", "Wyrm"],
          got["characters"])
print({"fails": fails})
sys.exit(1 if fails else 0)
