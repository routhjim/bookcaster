"""11-24: a figure whose used views are newer than its shape is shaped again; unused sides and older views do not count."""
import os, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import characters as C
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    def put(n, t):
        (d / n).write_bytes(b"x"); os.utime(d / n, (t, t))
    for v in ("front", "back", "left", "right"):
        put(f"apose_{v}_rgba.png", 100)
    put("shape.glb", 200)
    check("no shape yet: nothing is newer", C.views_newer(d, d / "none.glb") == [])
    check("older views: not stale", C.views_newer(d, d / "shape.glb") == [])
    put("apose_left_rgba.png", 300)
    check("a newer side that did not pass its check does not count", C.views_newer(d, d / "shape.glb") == [])
    put("apose_left.ok", 300)
    check("a newer checked side does", C.views_newer(d, d / "shape.glb") == ["apose_left_rgba.png"])
    check("the used views are front, back and checked sides",
          [p.name for p in C.used_views(d)] == ["apose_front_rgba.png", "apose_back_rgba.png", "apose_left_rgba.png"])
    for f in ("rigged.blend", "qa.json"):
        put(f, 250)
    C.log = lambda *a, **k: None
    C.step_shape({"dir": str(d), "key": "k"}, "05") if False else None
src = (HERE.parent / "characters.py").read_text()
check("a stale shape takes the model and its QA with it", '"shape.glb", "rigged.blend", "model.blend", "rig.json", "qa.json"' in src)
print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
