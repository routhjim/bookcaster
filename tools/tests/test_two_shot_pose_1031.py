"""10-31: a talk two-shot where one sits and the other stands says the posture in its opening clause (key words
only); uniform postures, other shots and the legacy prompt are left as they were.
Run with bookcaster's python: .venv/bin/python3 tools/tests/test_two_shot_pose_1031.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import shot_flow as SF  # noqa: E402

fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def pic(left, right, who, lis, held, at):
    hs = "; ".join(f"{w} {SF.POSE_VERB[held[w]]}{' ' + at[w] if at.get(w) else ''}" for w in (left, right) if held.get(w))
    return SF.TALK_TWO.format(left=left, right=right, who=who, listener=lis) + (". " + hs if hs else "")


H, Hel, W = "Sherlock Holmes", "Helen Stoner", "Dr. Watson"
s = {"who": [Hel, H], "listener": Hel, "pose_held": {Hel: "stand", H: "sit"}}
p = pic(Hel, H, H, Hel, s["pose_held"], {H: "in the armchair"})
out = SF._two_shot_posed(s, p)
check("mixed: the seated one in the opening clause, framing kept",
      out.startswith(f"{Hel} on the left and {H} on the right, {H} seated in the armchair, face lower in the picture, "
                     f"{Hel} standing beside and leaning in; seen from the waist up and close to the viewer"))
check("mixed: the held suffix is dropped (said in the clause)", out.endswith("listening"))
s2 = {"who": [W, H], "listener": W, "pose_held": {W: "sit", H: "stand"}}
out2 = SF._two_shot_posed(s2, pic(W, H, H, W, s2["pose_held"], {W: "in a chair"}))
check("a name with a full stop (Dr. Watson) keeps the whole template",
      f"{W} seated in a chair" in out2 and out2.endswith("listening"))
s3 = {"who": [Hel, H], "listener": Hel, "pose_held": {Hel: "sit", H: "sit"}}
p3 = pic(Hel, H, H, Hel, s3["pose_held"], {})
check("both seated: unchanged", SF._two_shot_posed(s3, p3) == p3)
s4 = {"who": [Hel, H], "listener": Hel, "pose_held": {}}
p4 = pic(Hel, H, H, Hel, {}, {})
check("no held poses: unchanged", SF._two_shot_posed(s4, p4) == p4)
s5 = {"who": [H], "pose_held": {H: "sit"}}
p5 = SF.TALK_PICTURE.format(who=H)
check("a close-up: unchanged", SF._two_shot_posed(s5, p5) == p5)
s6 = {"who": [Hel, H], "pose_held": {Hel: "stand", H: "sit"}}
check("not a talk two-shot (no listener): unchanged", SF._two_shot_posed(s6, p) == p)
s7 = {"who": ["Julia Stoner", Hel], "listener": Hel, "pose_held": {Hel: "sit"}}
p7 = SF.TALK_TWO_BED.format(left="Julia Stoner", right=Hel, abed=Hel, bedpose="sitting up", other="Julia Stoner",
                            who="Julia Stoner", listener=Hel)
check("the in-bed two-shot: unchanged (no second Helen in a chair)", SF._two_shot_posed(s7, p7) == p7)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
