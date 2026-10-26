"""10-23 staging findings as checks (run with the pose venv): fighters face each other, the motion follows the
director's beats (never an idle), the dragon's ground moves are on all fours.

    ~/tts-ab/.venv-pose/bin/python tools/tests/test_staging_1023.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import motion_batch as MB        # noqa: E402
import motion_composite as MC    # noqa: E402
import scene_stage as SS         # noqa: E402

fails = []


def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + (f"  ({got})" if got and not ok else ""))
    if not ok:
        fails.append(what)


# fighters face each other
shot = {"figures": [{"who": "Edric", "kind": "human", "at": [0, 0], "facing": -45.0},
                    {"who": "Vorgrath", "kind": "creature", "at": [-1, 6], "facing": 0.0}]}
SS.confront_rule(shot, {"acts": [{"do": "block", "who": "Edric", "tool": "sword", "against": "Vorgrath:foreclaw"}],
                        "beats": ["Vorgrath: extends its foreclaw towards Edric"]})
f = {g["who"]: g["facing"] for g in shot["figures"]}
check(f == {"Edric": "Vorgrath", "Vorgrath": "Edric"}, "confront: the blocker and the creature face each other", f)
shot["figures"][0]["facing"] = 10.0
SS.confront_rule(shot, {"beats": ["Edric: runs away from the dragon toward the ledge"]})
check(shot["figures"][0]["facing"] == 10.0, "confront: someone running away keeps their own facing")

# motion from the beats
fr = {"beats": ["Vorgrath: sweeps its tail across the ground, knocking Edric flat", "Edric: rolls his body to the left "
                "away from the claws, right hand still gripping the sword."]}
b = MB.beats_of(fr, "Edric")
check(b == ["Edric rolls his body to the left away from the claws, right hand still gripping the sword."], "beats_of", b)
p = MB.beats_prompt(b, "Edric")
check(p and p.startswith("A person rolls") and "his" not in p, "beats_prompt: the beat as body words", p)
v = [x["verb"] for x in MB.creature_verbs(MB.beats_of(fr, "Vorgrath"), 5.0, {"t_impact": 2.7, "target": None})[0]]
check("tail" in v, "creature verbs: 'sweeps its tail' is the tail move", v)

# the dragon's ground moves on all fours
for k in ("idle", "strike", "fall", "tail"):
    check("all fours" in MC.COMPOSITE_ACTIONS[k]["prompt"], f"composite {k}: on all fours")
check("all fours" not in MC.COMPOSITE_ACTIONS["fly"]["prompt"], "composite fly: not on all fours")

# the fight camera: a medium on ONE person -> the solver's over-the-shoulder up at the creature (never a far wide)
base = {"figures": [{"who": "Edric", "kind": "human"}, {"who": "Vorgrath", "kind": "creature"}], "camera": {"framing": "medium"}}
cam = SS.camera_for({"kind": "action"}, {}, base, {"axis": [], "marks": {}})
check(cam.get("framing") == "medium" and cam.get("target") == "Edric", "fight camera: medium on Edric (over-the-shoulder)", cam)

# the fallen stay down
v = [x["verb"] for x in MB.creature_verbs(["Vorgrath lies dead on the ash, unmoving."], 5.0, {"t_impact": 2.7, "target": None})[0]]
check(v and v[0] == "dead", "creature verbs: 'lies dead' is the dead pose", v)
check("completely still" in MC.COMPOSITE_ACTIONS["dead"]["prompt"], "composite dead: lies still")

# a held prop's state is the same prop (the killing thrust's build failed on 'sword__finished' vs 'sword')
import blockout_contact as BC    # noqa: E402
check(BC.held_hand({"right": "sword"}, "sword__finished") == "right", "contacts: 'sword__finished' is the held 'sword'")
check(BC.held_hand({"right": "hammer"}, "sword") is None, "contacts: another prop is not held")

# the pose map's face points only where the face turns to the camera (Edric was drawn facing the viewer from behind)
import numpy as np               # noqa: E402
import blockout_rig as rig       # noqa: E402
Wf = {n: None for n in rig.KP_NAMES}
Wf.update(head=np.array([0, 0, 1.7]), nose=np.array([0, .1, 1.7]), r_eye=np.array([-.03, .09, 1.73]),
          l_eye=np.array([.03, .09, 1.73]), r_ear=np.array([-.08, 0, 1.7]), l_ear=np.array([.08, 0, 1.7]))
face = lambda cam: [n for n, q in zip(rig.KP_NAMES, rig.coco18(Wf, cam)) if q is not None and n != "head"]  # noqa: E731
check("nose" not in face([0, -5, 1.7]) and "r_ear" in face([0, -5, 1.7]), "pose map: from behind, ears only", face([0, -5, 1.7]))
check("nose" in face([0, 5, 1.7]), "pose map: from the front, the face")
import mannequin_flow as MFl     # noqa: E402
import inspect                   # noqa: E402
check("coco18(J, None)" not in inspect.getsource(__import__("blockout_video")), "pose map: drawn with the camera")

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
