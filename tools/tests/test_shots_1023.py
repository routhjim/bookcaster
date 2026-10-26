"""10-23 night: the shot-list flow (tools/shot_flow.py) - the director's lint, LTX lengths, the take tie, the cut's
frame counts, the clang finder. Run with system python: python3 tools/tests/test_shots_1023.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import shot_flow as SF          # noqa: E402

fails = []


def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + (f"  ({got})" if got and not ok else ""))
    if not ok:
        fails.append(what)


cast = ["Edric", "Vorgrath", "The Knight"]
good = {"shots": [
    {"dur": 3, "who": ["Edric"], "framing": "medium", "picture": "x", "kind": "action",
     "action": "The young man brings the hammer down; sparks fly from the hammer striking the blade. The blade's edge "
               "takes on a faint blue sheen."},
    {"dur": 2, "who": ["Vorgrath"], "framing": "wide", "picture": "y", "kind": "action",
     "action": "The huge black dragon breathes a stream of fire from its mouth."}]}
check(SF._faults(good, cast, 5.0) == [], "director: sourced effects, plain words, lengths that add up", SF._faults(good, cast, 5.0))
bad = json.loads(json.dumps(good))
bad["shots"][0]["action"] = "Edric swings his enchanted sword. Sparks fly. Light glows in his hands."
f = SF._faults(bad, cast, 5.0)
check(any("names Edric" in x for x in f) and any("enchant" in x for x in f) and any("hands" in x for x in f)
      and any("sparks" in x.lower() for x in f), "director: a name, magic, light at the hands, unsourced sparks", f)
check(any("add up" in x for x in SF._faults(good, cast, 12.0)), "director: lengths that miss the stretch")
k = json.loads(json.dumps(good))
k["shots"][1]["action"] = "A knight in plate armour raises his sword."
check(not any("names" in x for x in SF._faults(k, cast, 5.0)), "director: 'a knight' is not the name 'The Knight'")

# LTX lengths: 8k+1 frames covering the shot
check([SF._frames(s) for s in (2.0, 3.0, 5.0)] == [49, 73, 121], "ltx: 8k+1 frames per shot length",
      [SF._frames(s) for s in (2.0, 3.0, 5.0)])

# the clang finder: a motion that stops dead
import numpy as np              # noqa: E402
import subprocess               # noqa: E402
import tempfile                 # noqa: E402
with tempfile.TemporaryDirectory() as td:
    fr = []
    for i in range(48):         # a bright bar falls fast for 12 frames, stops, and holds; again at 30
        a = np.zeros((54, 96), np.uint8)
        y = min(40, (i % 30) * 4) if i % 30 < 12 else 40
        a[y:y + 8, 30:60] = 255
        fr.append(a)
    raw = b"".join(x.tobytes() for x in fr)
    mp4 = Path(td) / "t.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "96x54", "-r", "24",
                    "-i", "-", "-c:v", "libopenh264", "-b:v", "2M", str(mp4)], input=raw, check=True)
    h = SF._impacts_in_clip(mp4)
    check(len(h) >= 1 and any(9 <= x <= 13 for x in h), "clangs: a landing found in the picture", h)

# talk words: short, lips and jaw, never "mouth closed" (10-24: that sentence froze the speaker's lips, as on 10-22)
class _R:
    @staticmethod
    def gender_of(d):
        return "woman" if "woman" in d else "man"


class _Cx:
    @staticmethod
    def look(w, t):
        return "a woman of about 70, grey hair tied back, wearing a heavy wool dress"


tp = SF._talk_prompt(_R, _Cx, {"speaker": "Hulda", "t": 1.0, "emotion": "cold"}, "Outside Hulda's Forge")
check("lips and jaw moving clearly" in tp and "mouth closed" not in tp and "Gothic" not in tp and len(tp.split()) < 60,
      "talk: short words that move the lips", tp)

# armor that is being made is not worn (10-24: Edric wore the plates he was forging)
w = {"shots": [{"picture": "He stands wearing thin folded metal arm plates over his jerkin.", "action": "He flexes."},
               {"picture": "He holds a dark metal helm in his hands.", "action": "He turns the helm to look at the slit."}]}
f = SF._worn(w, "worn brown leather jerkin, trousers")
check(len(f) == 1 and f[0].startswith("shot 1"), "worn: armor on a body still in leathers is a fault, held armor is not", f)
check(SF._worn(w, "Star-iron breastplate, star-iron arm plates") == [], "worn: armor the wardrobe has on is fine")

# making is shown as the work (10-24: "he made the sword" came out as admiring a finished sword)
adm = {"shots": [{"picture": "He holds a finished sword.", "action": "He turns the sword in his hands to look at it."}]}
wrk = {"shots": [{"picture": "He holds a glowing blade on the anvil.", "action": "He brings the hammer down on the blade."}]}
said = "Last of all, he made the sword. He folded the star-iron over itself a hundred times."
check(SF._making(adm, said) and not SF._making(wrk, said) and not SF._making(adm, "He walked north."),
      "making: the work is shown when the narration makes something")

# groups: up to 6 named, one row to 3, two depth rows from 4; boxes inside the frame, left to right
ok_l = True
for n in range(1, SF.MAX_PEOPLE + 1):
    lay = SF.group_layout(n)
    ok_l &= len(lay) == n and all(0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1 for b, _ in lay)
    ok_l &= [b[0] + b[2] for b, _ in lay] == sorted(b[0] + b[2] for b, _ in lay)
    ok_l &= (n < 4) == all(not w.startswith("behind") for _, w in lay)
check(ok_l, "groups: a box and a place for each of 1-6 people, left to right, a back row from 4")
g = {"shots": [{"dur": 5, "who": ["Edric", "Vorgrath", "The Knight"], "framing": "medium", "picture": "x",
                "action": "A young man raises a sword.", "kind": "action"}]}
check(any("need a wide shot" in x for x in SF._faults(g, cast, 5.0)), "groups: three people in a medium shot is a fault")
g["shots"][0]["framing"] = "wide"
check(not any("people" in x for x in SF._faults(g, cast, 5.0)), "groups: three people in a wide shot is fine")

# the set holds no story things (10-24: a dark blue sword hung above the forge door)
r = SF._unpropped("A heavy wooden door with iron bands stands at the back. A dark blue sword hangs above the door.")
check(r == "A heavy wooden door with iron bands stands at the back.", "scene: weapons are cut from the room", r)
r = SF._unpropped("A vast stone hall. A massive black dragon silhouette fills the window.")
check(r == "A vast stone hall.", "scene: creatures are cut from the room", r)

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
