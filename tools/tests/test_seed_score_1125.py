"""Picking between drawn key seeds (11-25): a staged person is found under any spelling of their name (boxes,
likeness refs), and only a glowing prop the staging put in view scores as "hot": a bright warm wall does not."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T))
import remake_chapter as R
import keyframe_likeness as KL

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)

ex = {"size": [320, 180], "figures": [{"who": "Mr. Tam", "bbox": [10, 20, 110, 170]},
                                      {"who": "Orla Venn", "bbox": [200, 30, 300, 170]}]}
F = R.fig_boxes(ex)
check("written-out title finds the staged figure", "Mister Tam" in F and F["Mister Tam"] == [10, 20, 110, 170])
check("exact spelling still works", F.get("Orla Venn") == [200, 30, 300, 170] and "Orla Venn" in F)
check("a stranger is not found", "Doctor Hale" not in F and F.get("Doctor Hale") is None)
check("staged spelling returned", F.name("mister tam") == "Mr. Tam" and F.name("Nobody") == "Nobody")
check("canon: title spellings agree", KL.canon("Mister Tam") == KL.canon("Mr. Tam") != KL.canon("Mrs. Tam"))

cam = {"pos": [0, -4, 1.5], "R": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "lens": 50.0}   # looks along +y
warm = np.zeros((180, 320, 3), np.uint8); warm[...] = (240, 190, 60)                  # a bright warm wall, all of it
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    still = td / "start_color.png"
    Image.fromarray(warm).save(still)
    def scene(prims):
        (td / "stills.json").write_text(json.dumps({"frames": [{"name": "start", "camera": cam, "prims": prims}]}))
    scene([{"t": "box", "c": [0, 0, 1.5], "M": [[.1, 0, 0], [0, .1, 0], [0, 0, .1]], "tag": "set"}])
    check("no glowing prop staged: no hot region", KL.hot_region(str(still), 320, 180) is None)
    scene([{"t": "capsule", "a": [0, -9, 1.5], "b": [0, -9, 1.6], "r": .01, "tag": "hot"}])
    check("glowing prop behind the camera: no hot region", KL.hot_region(str(still), 320, 180) is None)
    scene([{"t": "capsule", "a": [40, 0, 1.5], "b": [40, 0, 1.6], "r": .01, "tag": "hot"}])
    check("glowing prop out of frame: no hot region", KL.hot_region(str(still), 320, 180) is None)
    scene([{"t": "capsule", "a": [0, 0, 1.5], "b": [0, 0, 1.6], "r": .01, "tag": "hot"}])
    m = KL.hot_region(str(still), 320, 180)
    check("glowing prop in view: a small region round it", m is not None and m[90, 160] and not m[5, 5] and m.mean() < 0.1)

    KL.embed = lambda ims: np.ones((len(ims), 1))
    (td / "e.json").write_text(json.dumps(ex))
    key = td / "key.png"; Image.fromarray(warm).save(key)
    sheet = td / "sheet.png"; Image.fromarray(warm).save(sheet)
    def run():
        KL.main([str(key), str(td / "e.json"), "--ref", f"Mister Tam={sheet}", "--ref", f"Orla Venn={sheet}",
                 "--still", str(still), "--json", str(td / "o.json")])
        return json.loads((td / "o.json").read_text())
    scene([])
    o = run()
    check("both people scored, staged spelling", sorted(o["likeness"]) == ["Mr. Tam", "Orla Venn"])
    check("warm bright picture, no glowing prop: hot is None", o["hot"] is None)
    scene([{"t": "capsule", "a": [0, 0, 1.4], "b": [0, 0, 1.6], "r": .05, "tag": "hot"}])
    o = run()
    check("glowing prop in view and kept: hot scores", o["hot"] is not None and o["hot"] > 0.9)

src = (T / "remake_chapter.py").read_text()
check("one place builds the figure boxes", src.count('f["who"]: f["bbox"]') == 1)
print("all passed" if ok else "FAILED")
sys.exit(0 if ok else 1)
