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
check("a talking single stays clean", 'crowd = ""                                # the clean single stays clean' in src)
print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
