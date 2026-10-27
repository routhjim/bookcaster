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
     "pose": {"Edric": "standing"}, "pose_end": {"Edric": "standing"},
     "action": "The young man brings the hammer down; sparks fly from the hammer striking the blade. The blade's edge "
               "takes on a faint blue sheen."},
    {"dur": 2, "who": ["Vorgrath"], "framing": "wide", "picture": "y", "kind": "action",
     "pose": {"Vorgrath": "crouching"}, "pose_end": {"Vorgrath": "crouching"},
     "action": "The huge black dragon breathes a stream of fire from its mouth."}]}
check(SF._faults(good, cast, 5.0) == [], "director: sourced effects, plain words, lengths that add up", SF._faults(good, cast, 5.0))
bad = json.loads(json.dumps(good))
bad["shots"][0]["action"] = "Edric swings his enchanted sword. Sparks fly. Light glows in his hands."
f = SF._faults(bad, cast, 5.0)
check(any("names Edric" in x for x in f) and any("sparks" in x.lower() for x in f)
      and not any("enchant" in x or "hands" in x for x in f),
      "director: a name and unsourced sparks are faults; the forge's magic/hand-light words no longer are (10-27)", f)
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


sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakebook
_Cx = fakebook.book({"Hulda": "a woman of about 70, grey hair tied back, wearing a heavy wool dress"})


tp = SF._talk_prompt(_R, _Cx, {"speaker": "Hulda", "t": 1.0, "emotion": "cold"}, "Outside Hulda's Forge")
check("lips and jaw moving clearly" in tp and "mouth closed" not in tp and "Gothic" not in tp and len(tp.split()) < 60,
      "talk: short words that move the lips", tp)

# 10-27 audit: the worn-armor and making checks were the forge's band-aids; the wardrobe owns what is worn and the
# director's scene state owns what is being made
check(not hasattr(SF, "_worn") and not hasattr(SF, "_making"), "no story band-aid checks (armor, making) in shared code")

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

# the set holds no story things: asked of the set sheet itself (SET_ASK), not cut afterwards by a word list (10-27
# audit: STORY_THING dropped any room sentence naming a blade, shield or silhouette)
check("nothing the story's people carry" in SF.SET_ASK and not hasattr(SF, "_unpropped"),
      "scene: the set sheet is asked for fixed things only; no after-the-fact word filter")

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
