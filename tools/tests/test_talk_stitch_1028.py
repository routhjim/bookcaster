"""10-28: pieced talk shots are stitched (the gate read the moment's old pieces, so none were all night), old joins
are dropped first, and a line left unjoined is a problem."""
import json, subprocess, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

L = Path(tempfile.mkdtemp(prefix="stitch_")); (L / "out").mkdir()
talk = [{"name": "talk-1_s0_c0", "chunk": [0, 2, 0, 97]}, {"name": "talk-1_s0_c1", "chunk": [1, 2, 90, 50]},
        {"name": "talk-2_s0", "chunk": [0, 1, 0, 60]}]
(L / "out" / "talk-1_s0.mp4").write_text("yesterday's join")
probs, ran = [], []
S._problem = lambda book, log, msg: probs.append(msg)
real = subprocess.run
def fake(cmd, *a, **k):
    ran.append(cmd)
    old = (L / "out" / "talk-1_s0.mp4").exists()
    (L / "out" / "talk-1_s0.mp4").write_text("old" if old else "tonight")
    return types.SimpleNamespace(returncode=0, stderr="")
S.subprocess.run = fake
R = types.SimpleNamespace(LTX_PY="python")
S._stitch_talk(Path("."), R, L, talk, log=lambda m: None)
check("the stitcher runs when a talk shot has pieces", len(ran) == 1, ran)
check("yesterday's join is dropped before stitching", (L / "out" / "talk-1_s0.mp4").read_text() == "tonight")
check("no problem when every line joined", probs == [], probs)
S.subprocess.run = lambda cmd, *a, **k: types.SimpleNamespace(returncode=1, stderr="Traceback\nFileNotFoundError: x")
S._stitch_talk(Path("."), R, L, talk, log=lambda m: None)
check("an unjoined line is a problem", probs and "1 of 1" in probs[0] and "FileNotFoundError" in probs[0], probs)
ran.clear(); S.subprocess.run = fake
S._stitch_talk(Path("."), R, L, talk[2:], log=lambda m: None)
check("no stitcher for single-piece talk", ran == [], ran)
src = (HERE.parent / "shot_flow.py").read_text()
check("render uses the checked helper (not the moment's old pieces)", "_stitch_talk(book, R, L, talk, log)" in src
      and 'byid[s["id"].rsplit("_s", 1)[0]].get("chunks")' not in src)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
