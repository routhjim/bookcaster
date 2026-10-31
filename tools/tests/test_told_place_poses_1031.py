"""10-31: a pose in a told place (young Helen in bed in Helen's Bedroom, as she tells it) is not carried into the
scene's own place (the Helen telling it at Baker Street). carry_poses keeps each place's poses apart.
Run with bookcaster's python: .venv/bin/python3 tools/tests/test_told_place_poses_1031.py"""
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


Hel, H = "Helen Stoner", "Sherlock Holmes"
HOME, TOLD = "Baker Street Sitting Room", "Helen's Bedroom"
es = [{"id": f"m{i}", "t": float(i)} for i in range(4)]
out = {"m0": [{"kind": "quiet", "place": HOME, "who": [Hel, H], "pose": {Hel: "sitting", H: "sitting"}}],
       "m1": [{"kind": "quiet", "place": TOLD, "who": [Hel], "pose": {Hel: "lying"}}],
       "m2": [{"kind": "quiet", "place": HOME, "who": [Hel, H]}],       # no pose given: carried
       "m3": [{"kind": "quiet", "place": TOLD, "who": [Hel]}]}
SF.carry_poses(es, out)
check("home shot after a told one carries the HOME pose", out["m2"][0]["pose_held"] == {Hel: "sit", H: "sit"})
check("the told place keeps its own pose", out["m3"][0]["pose_held"] == {Hel: "lie"})
check("a new scene still clears everything",
      (SF.carry_poses(es, out, cuts=(2.0,)) or out["m2"][0]["pose_held"]) == {})
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
