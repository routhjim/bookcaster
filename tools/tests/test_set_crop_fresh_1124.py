"""11-24: the cut of a place picture behind an action key is made again when the place picture is drawn again."""
import os, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as SF
from PIL import Image

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    hero = d / "hero.png"
    Image.new("RGB", (1152, 640), (250, 250, 250)).save(hero)
    s = {"framing": "medium", "who": ["Orla", "Tam"]}
    c = SF._set_crop(hero, d, s)
    check("a cut is made", c is not None and c.exists(), c)
    m0 = c.stat().st_mtime
    check("the same place picture keeps its cut", SF._set_crop(hero, d, s) == c and c.stat().st_mtime == m0)
    Image.new("RGB", (1152, 640), (10, 10, 10)).save(hero)
    t = time.time() + 5
    os.utime(hero, (t, t))
    c2 = SF._set_crop(hero, d, s)
    px = Image.open(c2).convert("RGB").getpixel((5, 5))
    check("a place picture drawn again is cut again", c2 == c and px[0] < 50, px)
    check("the new cut is not older than the place picture", c2.stat().st_mtime >= m0)

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
