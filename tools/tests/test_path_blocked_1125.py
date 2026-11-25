"""11-25: a camera move that starts or passes behind the set is not used; the moment keeps a still camera, which
the renderer moves clear."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import scene_stage as SS

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

class Wall:                       # a wall along y = 0: blocks any line that crosses it
    dep = 1
    def blocks(self, a, b):
        return (a[1] < 0) != (b[1] < 0)
class NoSet:
    dep = None
    def blocks(self, a, b):
        return True

clear = [{"pos": [0, 1, 1.6], "look": [0, 4, 1]}, {"pos": [0, 0.5, 1.6], "look": [0, 4, 1]}]
behind = [{"pos": [0, 1, 1.6], "look": [0, 4, 1]}, {"pos": [0, -2, 1.6], "look": [0, 4, 1]}]
check("a move in the open is kept", not SS.path_blocked({}, clear, Wall()))
check("a move that passes behind the wall is not", SS.path_blocked({}, behind, Wall()))
check("no measured set: nothing to test", not SS.path_blocked({}, behind, NoSet()))
check("no path: nothing blocked", not SS.path_blocked({}, [], Wall()))
src = (HERE.parent / "scene_stage.py").read_text()
check("the staging asks before it takes a move", "if path_blocked(base, path):" in src and '"camera_path_blocked"' in src)
print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
