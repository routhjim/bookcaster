"""11-12: the take checks in the flow: off by default (the judge alone, as before); on, a take's mechanical faults
join the judge's and a chain take they fault is not picked while another is clean; the review record carries the
reviewer against the judge."""
import json, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

S.MF.fight_clip_ok = lambda mp4, people=None: ["3 people, not 2, at 1.0 s"] if "crowd" in str(mp4) else []
seen = []
def fake_checks(mp4, kind=None, **kw):
    seen.append((Path(mp4).name, kind))
    return {"faults": ["the camera moves from 1.0 s (the background shifts 9% of the frame)"] if "pan" in str(mp4) else []}
S.TC.clip_faults = fake_checks

S.TAKE_CHECKS = False
check("off: the judge's faults alone", S._take_faults(Path("quiet-1_pan.mp4"), 2) == [] and seen == [], seen)
S.TAKE_CHECKS = True
f = S._take_faults(Path("quiet-1_pan.mp4"), 2)
check("on: a camera move is a fault", len(f) == 1 and "camera moves" in f[0], f)
check("the kind comes from the clip's name", seen[-1] == ("quiet-1_pan.mp4", "quiet"), seen)
f = S._take_faults(Path("action-2_crowd_pan.mp4"), 2)
check("on: the judge's faults and the checks' together", len(f) == 2, f)
check("a camera fault is not a head count (it is retaken as it was, not held)", not S._crowded(S._take_faults(Path("quiet-1_pan.mp4"), 2)), None)
j = {"name": "quiet-1", "seed": 1}
nx = S._next_take(j, "quiet-1", 2, ["the camera moves from 1.0 s"])
check("a faulted take gets its one retake", nx is not None and nx["name"] == "quiet-1_t2", nx)
def boom(mp4, kind=None, **kw):
    raise OSError("unreadable")
S.TC.clip_faults = boom
check("a check that cannot read the clip never fails the take", S._take_faults(Path("quiet-1.mp4"), 2) == [], None)
S.TC.clip_faults = fake_checks

# a chain line: the judge picks, the checks veto a faulted pick while another take is clean
J = Path(tempfile.mkdtemp(prefix="chain_"))
S._take_sheet = lambda t, out: out
fake = types.ModuleType("characters")
fake.vl_ask = lambda items: {"pick": {"best": "A", "next": "B"}}
sys.modules["characters"] = fake
st = {"id": "action-9", "ver": [0], "lines": ["she walks to the chair"], "director_lines": ["she walks to the chair"]}
import random
order = list(range(3)); random.Random("action-9|1|0").shuffle(order)
names = ["x", "y", "z"]
names[order[0]] = "pan"                     # the take the judge's "A" maps to pans
takes = [J / f"{n}.mp4" for n in names]
jd = S._judge_chain_line(st, 1, takes, J, lambda *a: None)
check("the judge's pick pans: a clean take is kept instead", takes[jd["pick"]].stem != "pan", jd["pick"])
check("... the judge's own next-best when it is clean", jd["pick"] == order[1], (jd["pick"], order))
check("the checks per take are recorded", jd["checks"] is not None and sum(bool(c) for c in jd["checks"]) == 1, jd["checks"])
S.TAKE_CHECKS = False
jd = S._judge_chain_line(st, 1, takes, J, lambda *a: None)
check("off: the judge's pick stands", jd["pick"] == order[0] and jd["checks"] is None, jd)

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
