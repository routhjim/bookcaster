"""11-24: staging reads the story's words only (never the standing instructions every clip carries for the video
model), and a title written as it is spoken names the same cast member."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import blockout_plan as BP
import scene_stage as SS
from tts_reader import animate

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

body = "Medium shot. Orla watches Tam across the hall"
motion = f"{body}. {animate.MOTION} plain ink drawing."
check("the standing instructions are in the fixture", animate.MOTION in motion)
st = BP.story_text(motion)
check("the story's words stay", body in st and "plain ink drawing" in st, st)
check("the standing instructions go", not any(s in st for s in animate.MOTION.split(". ") if len(s) > 20), st)
check("text without them is untouched", BP.story_text(body) == body)
respaced = motion.replace("  ", " ").replace("stays ONE object in the same", "stays ONE object in the  same")
check("a re-spaced copy loses the other sentences", "two arms" not in BP.story_text(respaced), BP.story_text(respaced))
f = {"shot": "a", "who": ["Orla", "Tam"], "prompt": body + ".", "motion": motion}
check("the frame is copied, not changed", BP.story_frame(f) is not f and f["motion"] == motion)
for w in ("Orla", "Tam"):
    pose, _ = BP.verb_pose(BP._about(BP.story_frame(f)["prompt"] + " " + BP.story_frame(f)["motion"], w, ["Orla", "Tam"]) or body)
    check(f"{w} holds no pose from the instructions", pose == "standing", pose)
shot = BP.frame_to_shot(dict(f), cast={})
check("nobody is handed an item the story never names", all(not g.get("holds") for g in shot["figures"]),
      [g.get("holds") for g in shot["figures"]])
check("and nobody takes a weapon pose", all(g["pose"] == "standing" for g in shot["figures"]), [g["pose"] for g in shot["figures"]])

cast = {"characters": {"Mr. Vane": {}, "Mrs. Vane": {}, "Dr. Holt": {}, "Orla Pike": {}}}
frames = {"a": {"who": ["Mister Vane", "Missus Vane", "Doctor Holt", "Orla"], "speaker": "Mister Vane"}}
SS.canon_names(frames, cast)
check("spoken titles become the cast's names", frames["a"]["who"] == ["Mr. Vane", "Mrs. Vane", "Dr. Holt", "Orla Pike"], frames["a"]["who"])
check("the speaker too", frames["a"]["speaker"] == "Mr. Vane", frames["a"]["speaker"])
frames = {"a": {"who": ["Mister Vane", "Mr. Vane"], "speaker": None}}
SS.canon_names(frames, cast)
check("two spellings in one frame are one person", frames["a"]["who"] == ["Mr. Vane"], frames["a"]["who"])

frames = {"a": {"who": ["Mister Vane"], "present": ["Mister Vane", "Orla"],
                "poses": [{"who": "Mister Vane", "pose": "standing", "box": [0, 0, 1, 1]}],
                "acts": [{"who": "Missus Vane", "do": "give", "target": "Mister Vane:hand_r"}],
                "beats": ["Mister Vane: turns away", "the door opens"], "dress": {"Doctor Holt": "coat"}}}
SS.canon_names(frames, cast)
f = frames["a"]
check("pose boxes name the cast member", f["poses"][0]["who"] == "Mr. Vane", f["poses"])
check("those present too", f["present"] == ["Mr. Vane", "Orla Pike"], f["present"])
check("an act's people, a part kept", f["acts"][0]["who"] == "Mrs. Vane" and f["acts"][0]["target"] == "Mr. Vane:hand_r", f["acts"])
check("a beat's person; other beats untouched", f["beats"] == ["Mr. Vane: turns away", "the door opens"], f["beats"])
check("what they wear is theirs", f["dress"] == {"Dr. Holt": "coat"}, f["dress"])

# the people of one moment stand together
names = ["Orla", "Tam", "Bress", "Ivo", "Wyn"]
scene = {"marks": {n: {"at": [1.4 * i - 2.8, 4.0], "facing": 180.0, "kind": "human"} for i, n in enumerate(names)},
         "reference_camera": {"pos": [0, -2, 1.5], "look": [0, 4, 1.5]}}
def figs(*who):
    return {"figures": [{"who": w, "kind": "human", "at": list(scene["marks"][w]["at"]), "facing": 180.0} for w in who]}
b = figs("Wyn", "Orla")
moved = SS.gather(b, scene, {})
at = {g["who"]: g["at"] for g in b["figures"]}
check("two with absent people between them close up", sorted(moved) == ["Orla", "Wyn"]
      and abs(abs(at["Wyn"][0] - at["Orla"][0]) - SS.TALK_GAP) < 1e-6, at)
check("about the centre of the scene", abs((at["Wyn"][0] + at["Orla"][0]) / 2) < 1e-6 and at["Orla"][1] == 4.0, at)
check("nobody changes side", at["Orla"][0] < at["Wyn"][0], at)
check("a pair turns to each other", {g["who"]: g["facing"] for g in b["figures"]} == {"Orla": "Wyn", "Wyn": "Orla"}, b["figures"])
b = figs("Tam", "Bress")
check("neighbours keep their marks", SS.gather(b, scene, {}) == [] and b["figures"][0]["at"] == scene["marks"]["Tam"]["at"])
b = figs("Orla", "Bress", "Wyn")
SS.gather(b, scene, {})
xs = [g["at"][0] for g in sorted(b["figures"], key=lambda g: names.index(g["who"]))]
check("three keep their order, evenly", xs == sorted(xs) and abs(xs[1] - xs[0] - SS.TALK_GAP) < 1e-6 and abs(xs[1]) < 1e-6, xs)
check("three keep their own facing", all(g["facing"] == 180.0 for g in b["figures"]))
b = figs("Wyn", "Orla")
check("an interaction program places its own people", SS.gather(b, scene, {"acts": [{"who": "Orla", "do": "give"}]}) == [])
b = figs("Wyn", "Orla")
b["figures"][0]["attach"] = {"to": "Orla"}
check("someone carried stays where they hang", SS.gather(b, scene, {}) == [])
b = figs("Orla")
check("one person keeps their mark", SS.gather(b, scene, {}) == [])
two = {"marks": {"Orla": {"at": [-1.5, 4.0], "facing": 180.0}, "Tam": {"at": [1.5, 4.0], "facing": 180.0}},
       "reference_camera": {"pos": [0, -2, 1.5], "look": [0, 4, 1.5]}}
mk = lambda: {"figures": [{"who": w, "kind": "human", "at": list(two["marks"][w]["at"]), "facing": 180.0} for w in ("Orla", "Tam")]}
b = mk()
check("two far apart who do not talk keep their marks", SS.gather(b, two, {}) == [])
b = mk()
SS.gather(b, two, {}, talk=True)
at = {g["who"]: g["at"] for g in b["figures"]}
check("two who talk from afar close up about their midpoint", abs(at["Tam"][0] - at["Orla"][0] - SS.TALK_GAP) < 1e-6
      and abs(at["Tam"][0] + at["Orla"][0]) < 1e-6, at)
check("and face each other", {g["who"]: g["facing"] for g in b["figures"]} == {"Orla": "Tam", "Tam": "Orla"})
near = {"marks": {"Orla": {"at": [-1.0, 4.0]}, "Tam": {"at": [1.0, 4.0]}}, "reference_camera": two["reference_camera"]}
b = {"figures": [{"who": w, "kind": "human", "at": list(near["marks"][w]["at"]), "facing": 180.0} for w in ("Orla", "Tam")]}
check("two who talk within a shared frame stay", SS.gather(b, near, {}, talk=True) == [])

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
