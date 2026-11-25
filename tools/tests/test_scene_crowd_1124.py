"""11-24: a 3D key names the scene's unnamed people: the keyframe's own crowd, else the nearest one of the same
scene and place; never another scene's or place's."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import remake_chapter as R
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)
F = {"a1": {"scene": "s1", "where": "Hall", "crowd": ""},
     "a2": {"scene": "s1", "where": "Hall", "crowd": "villagers at long tables"},
     "a3": {"scene": "s1", "where": "Hall", "crowd": ""},
     "a4": {"scene": "s1", "where": "Hall", "crowd": "pipers by the door"},
     "a5": {"scene": "s1", "where": "Yard", "crowd": ""},
     "a6": {"scene": "s2", "where": "Hall", "crowd": ""}}
check("its own crowd", R.scene_crowd(F, "a4") == "pipers by the door")
check("the nearest of the scene, before", R.scene_crowd(F, "a1") == "villagers at long tables")
check("the nearest of the scene, earlier wins a tie", R.scene_crowd(F, "a3") == "villagers at long tables")
check("another place has none", R.scene_crowd(F, "a5") == "")
check("another scene has none", R.scene_crowd(F, "a6") == "")
check("an unknown moment has none", R.scene_crowd(F, "zz") == "")
src = (HERE.parent / "remake_chapter.py").read_text()
check("the key prompt puts them far behind the named people", "in the foreground, each appearing once" in src and "Far behind" in src)
check("a talking single stays clean unless its still shows the guests", "if not shot_guests(e):" in src
      and '"alone in the foreground" if crowd else "the only person in the picture"' in src)
import json, tempfile
with tempfile.TemporaryDirectory() as td:
    a_, b_ = Path(td) / "a.json", Path(td) / "b.json"
    a_.write_text(json.dumps({"figures": [], "crowd": [{"who": "guest 1 #0"}]}))
    b_.write_text(json.dumps({"figures": []}))
    check("a shot with guests", R.shot_guests({"shot": str(a_)}))
    check("a shot without", not R.shot_guests({"shot": str(b_)}))
    check("no shot file", not R.shot_guests({"shot": str(Path(td) / "none.json")}) and not R.shot_guests({}))
print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
