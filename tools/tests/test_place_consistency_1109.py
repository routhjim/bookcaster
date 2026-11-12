"""11-09 (E34, Speckled Band 0-23:00 redo): the told house's inside was drawn on the lawn (its picture was drawn from
the house's outside words). A place named with its side ("X (inside)") takes only that side's description, never X's
unmarked or other-side words; describe_places writes its own side's entry and leaves X's alone; the breakdown's place
list stays the same. (Layouts by place AND side, director side: test_rooms_1029.)"""
import json, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import registry as RG
import mannequin_flow as MF
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# --- 1. a side's own description --------------------------------------------------------------------------------
b = Path(tempfile.mkdtemp(prefix="e34_"))
(b / "work" / "video").mkdir(parents=True)
OUTW = "a large old grey stone house with two wings. The lawn is overgrown."
(b / "work" / "video" / "look.json").write_text(json.dumps({"characters": {}, "places": {
    "Grey Manor": OUTW, "Grey Manor (outside)": "Outside: the stone house from the drive",
    "Mill": "Outside: a stone mill by a stream"}}))
R = RG.Book(b)
check("an inside never takes the house's unmarked words", R.place_look("Grey Manor (inside)") == "", R.place_look("Grey Manor (inside)"))
check("an inside never takes the other side's words (the old fallback)", R.place_look("Mill (inside)") == "")
check("the place itself and its outside keep theirs", R.place_look("Grey Manor") == OUTW
      and R.place_look("Grey Manor (outside)").startswith("Outside: the stone house"))
check("place_slot: an inside is stored under its own name, the house's words untouched",
      RG.place_slot(R.data["places"], "Grey Manor (inside)", "Inside: a corridor") == ("Grey Manor (inside)", False))
check("place_slot: a name with no side still finds its one entry",
      RG.place_slot(R.data["places"], "the grey manor", "Inside: x") == ("Grey Manor", True))

# describe_places: the set sheet's room becomes the inside's description; the house's words stay
(b / "work" / "video" / "01_x").mkdir()
S.subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="")
from tts_reader import animate
animate.igpu = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no language model: the room is known"))
shots = [{"id": "quiet-00m01s-000010_s0", "place": "Grey Manor (inside)", "room": "a narrow corridor with three doors"}]
S.describe_places(b, "01_x", [{"id": "quiet-00m01s-000010", "t": 1.0, "dur": 3.0}], shots, log=lambda m: None)
lk = json.loads((b / "work" / "video" / "look.json").read_text())
check("describe_places: the inside described as an inside, under its own name",
      lk["places"].get("Grey Manor (inside)") == "Inside: a narrow corridor with three doors", lk["places"])
check("describe_places: the house's outside words untouched", lk["places"]["Grey Manor"] == OUTW)
check("describe_places: marked under the side's own entry", "Grey Manor (inside)" in lk.get("_described", {})
      and "Grey Manor" not in lk.get("_described", {}), lk.get("_described"))
R = RG.Book(b)
check("the hero now reads the inside's words", R.place_look("Grey Manor (inside)").startswith("Inside: a narrow corridor"))
st = MF._hero_state(b, "Grey Manor (inside)", lambda m: None, ())
check("the hero prompt is the inside's (no lawn)", isinstance(st, dict) and "corridor" in st["prompt"]
      and "lawn" not in st["prompt"], st if not isinstance(st, dict) else st["prompt"])
# the breakdown's stable place list leaves out only places a run invented, not a side it described (k2 stays)
_REAL = (MF.paths, S.load_timeline, S.breakdown)
MF.paths = lambda book, stem: (b / "work" / "video" / stem, None, None)
(b / "work" / "video" / "01_x" / "plan.json").write_text(json.dumps({"look": {"characters": {}}}))
seen = {}
S.load_timeline = lambda book, stem, chars=None: [{"t": 0.0, "text": "x"}]
S.breakdown = lambda book, stem, tl, chars, places, nar, log=print, stable=None: seen.update(stable=stable) or []
S.scene_list(b, "01_x")
check("scene_list: the house stays in the breakdown's stable places", "Grey Manor" in (seen.get("stable") or []),
      seen)
MF.paths, S.load_timeline, S.breakdown = _REAL

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
