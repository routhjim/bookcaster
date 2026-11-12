"""E29 (11-08 user: "when the lips are supposed to be in synch, they need to be"): two lip-sync mechanics.
1. SyncNet's zone for an over-the-shoulder talk shot: all but the listener's foreground third (the side-by-side half
   began at x 0.42; Acts ch.1's Jesus sat at x 0.34-0.47 and no face was found). A kept clip takes the plan's zone and
   is measured again in it.
2. One shift per joined line: each piece of a line is shifted to its own voice inside the join (a ramp across each
   seam, frame count kept), the assembly adds none; the kept take's pieces travel with its join."""
import json, os, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE.parent / "ltx_runner"))
import numpy as np
import shot_flow as S
from stitch_talk import shift_map

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# --- 1. the over-the-shoulder zone ----------------------------------------------------------------------------------
who = ["Peter", "Jesus"]
zr = S._speaker_box({"who": who, "ots": "Peter", "kind": "talk"}, "Jesus")
zl = S._speaker_box({"who": who, "ots": "Jesus", "kind": "talk"}, "Peter")
check("over the shoulder, speaker on the right: the zone holds the measured face (x 0.34-0.47)",
      zr[0] <= 0.34 and zr[2] >= 0.47, zr)
check("...and not the listener's third (heads ended at x 0.30)", zr[0] > 0.30, zr)
check("speaker on the left: the mirror", zl[0] == 0.0 and abs(zl[2] - (1 - zr[0])) < 1e-9, zl)
check("a two-shot keeps its layout box", S._speaker_box({"who": who, "speaker": "Jesus", "kind": "talk"}, "Jesus") ==
      S.talk_layout({"who": who, "speaker": "Jesus", "kind": "talk"})[1][0])
check("the key's layout (LoRA boxes, sides) is untouched", S.talk_layout({"who": who, "ots": "Peter", "kind": "talk"})
      == S.group_layout(2))

d = Path(tempfile.mkdtemp(prefix="e29_")); (d / "out").mkdir()
model = d / "syncnet_src" / "data"; model.mkdir(parents=True); (model / "syncnet_v2.model").write_text("")
S.SYNCNET, S.SYNCNET_PY = d / "syncnet_src", Path(sys.executable)
ran = []
real_run = S.subprocess.run
def fake_run(cmd, *a, **k):
    ran.append(cmd)
    return types.SimpleNamespace(returncode=0, stdout="AV offset: 3\nConfidence: 6.5\n", stderr="")
S.subprocess.run = fake_run
c = d / "out" / "talk-1_s0.mp4"; c.write_bytes(b"x" * 10)
(d / "out" / "talk-1_s0.json").write_text(json.dumps({"speaker_box": [0.42, 0.0, 1.0, 1.0]}))
S._sync_offsets(Path("."), d, ["talk-1_s0"], log=lambda m: None)
m = json.loads(c.with_suffix(".sync.json").read_text())
check("a reading records the zone it was taken in", m.get("box") == [0.42, 0.0, 1.0, 1.0], m)
ran.clear(); S._sync_offsets(Path("."), d, ["talk-1_s0"], log=lambda m: None)
check("same clip, same zone: not measured again", not any("run_syncnet.py" in str(x) for x in ran), ran)
(d / "out" / "talk-1_s0.json").write_text(json.dumps({"speaker_box": zr}))
S._sync_offsets(Path("."), d, ["talk-1_s0"], log=lambda m: None)
check("a new zone: measured again in it", any("run_syncnet.py" in x for x in map(str, ran)) and
      json.loads(c.with_suffix(".sync.json").read_text())["box"] == zr, ran)
c.with_suffix(".sync.json").write_text(json.dumps({"sig": f"{c.stat().st_size}:{c.stat().st_mtime_ns}", "offset": 0,
                                                   "conf": 7.0}))
ran.clear(); S._sync_offsets(Path("."), d, ["talk-1_s0"], log=lambda m: None)
check("a reading from before zones were recorded, of a shot with a zone: measured once more", bool(ran), ran)
(d / "out" / "talk-1_s0.json").write_text(json.dumps({}))
# (a reading of the voiced-frames measure, 11-12: it records "lock"; one from before it is measured once more)
c.with_suffix(".sync.json").write_text(json.dumps({"sig": f"{c.stat().st_size}:{c.stat().st_mtime_ns}", "offset": 0,
                                                   "conf": 7.0, "lock": True}))
ran.clear(); S._sync_offsets(Path("."), d, ["talk-1_s0"], log=lambda m: None)
check("...but a close-up's (no zone) is kept", not any("run_syncnet.py" in str(x) for x in ran), ran)

# --- 2. a shift per piece -----------------------------------------------------------------------------------------
p = shift_map(400, [0, 192], [-2, 3])
check("frame count kept", len(p) == 400)
check("each piece's stretch reads its own shift outside the ramp", p[100] == 98 and p[300] == 303, (p[100], p[300]))
check("never backwards", bool((np.diff(p) >= 0).all()))
r = np.diff(p[170:215])
check("the move is spread over the seam (no jump of 5)", r.max() <= 2 and (r == 2).sum() == 5, r)
check("ends held, never outside the clip", p[0] == 0 and p[-1] == 399 and p.min() >= 0)
check("no shifts: the old join exactly", (shift_map(300, [0, 150], [0, 0]) == np.arange(300)).all())
q = shift_map(400, [0, 192], [4, -6])
check("a backwards move holds frames, never repeats a stretch in reverse", bool((np.diff(q) >= 0).all()) and
      q[150] == 154 and q[260] == 254, (q[150], q[260]))
check("an unread piece takes the nearest read one's shift", S._piece_shifts([None, -2, None, None, 1]) == [-2, -2, -2, 1, 1])
check("a tie goes to the piece before", S._piece_shifts([3, None, 1]) == [3, 3, 1])
check("none read: none", S._piece_shifts([None, None]) == [None, None])

j = d / "out" / "talk-2_s0.mp4"; j.write_bytes(b"y" * 10)
st = j.stat()
j.with_suffix(".sync.json").write_text(json.dumps({"sig": f"{st.st_size}:{st.st_mtime_ns}", "offset": -4, "conf": 6.0}))
check("a join with no piece shifts: its own reading moves it", S.sync_shift(j) == -4)
j.with_suffix(".join.json").write_text(json.dumps({"shifts": [-2, 3]}))
check("a join of pieces each on its own voice: the assembly moves it no more", S.sync_shift(j) == 0)
j.with_suffix(".join.json").write_text(json.dumps({"shifts": [None, None]}))
check("...unless no piece read confidently", S.sync_shift(j) == -4)

# the stitcher is told each piece's shift, and a join made with other shifts is made again
S.subprocess.run = real_run
L = Path(tempfile.mkdtemp(prefix="e29s_")); (L / "out").mkdir()
talk = [{"name": "talk-3_s0_c0", "chunk": [0, 2, 0, 193], "line": "talk-3"},
        {"name": "talk-3_s0_c1", "chunk": [1, 2, 192, 97], "line": "talk-3", "chain_from": "talk-3_s0_c0"}]
for x, off in (("talk-3_s0_c0", -2), ("talk-3_s0_c1", 3)):
    f = L / "out" / f"{x}.mp4"; f.write_bytes(b"z" * 10)
    f.with_suffix(".sync.json").write_text(json.dumps({"sig": f"{f.stat().st_size}:{f.stat().st_mtime_ns}",
                                                       "offset": off, "conf": 7.0, "box": None}))
S._sync_offsets = lambda book, L_, names, log=print, quiet=False: measured.append(tuple(names))
measured, ran = [], []
def fake_stitch(cmd, *a, **k):
    ran.append(json.loads((L / cmd[2]).read_text()))
    (L / "out" / "talk-3_s0.mp4").write_text("join")
    (L / "out" / "talk-3_s0.join.json").write_text(json.dumps({"shifts": [x.get("sync_shift") for x in ran[-1]]}))
    return types.SimpleNamespace(returncode=0, stderr="")
S.subprocess.run = fake_stitch
S._problem = lambda book, log, msg: print("problem:", msg)
R = types.SimpleNamespace(LTX_PY="python")
S._stitch_talk(Path("."), R, L, talk, log=lambda m: None)
check("the pieces are measured before the join", measured and measured[0] == ("talk-3_s0_c0", "talk-3_s0_c1"), measured)
check("each piece goes to the stitcher with its own shift (25 -> 24 fps)",
      ran and [x["sync_shift"] for x in ran[0]] == [-2, 3], ran)
ran.clear(); S._stitch_talk(Path("."), R, L, talk, log=lambda m: None)
check("a join with its pieces' shifts is kept", ran == [], ran)
f = L / "out" / "talk-3_s0_c1.sync.json"
f.write_text(json.dumps(dict(json.loads(f.read_text()), offset=-1)))
S._stitch_talk(Path("."), R, L, talk, log=lambda m: None)
check("a piece read again with another offset: joined again", ran and [x["sync_shift"] for x in ran[0]] == [-2, -1], ran)

# the kept take's pieces travel with its join; a line retaken before E29 gets its kept take's pieces back
K = Path(tempfile.mkdtemp(prefix="e29k_")); (K / "out").mkdir()
for t in ("", "_t2"):
    for x in ("", "_c0", "_c1"):
        (K / "out" / f"talk-4_s0{t}{x}.mp4").write_text(f"take{t or '_t1'}{x}")
        (K / "out" / f"talk-4_s0{t}{x}.sync.json").write_text(json.dumps({"conf": 7.0 if t else 1.0}))
S._keep_best_talk(K, "talk-4_s0", 2, lambda m: None, pieces=2)
check("the kept take's join and pieces are the line's", [(K / "out" / f"talk-4_s0{x}.mp4").read_text() for x in
      ("", "_c0", "_c1")] == ["take_t2", "take_t2_c0", "take_t2_c1"])
check("...and it is recorded", json.loads((K / "out" / "talk-4_s0.synced").read_text()).get("_pieces") == "talk-4_s0_t2")
M = Path(tempfile.mkdtemp(prefix="e29m_")); (M / "out").mkdir()
for t in ("", "_t2"):
    for x in ("_c0", "_c1"):
        (M / "out" / f"talk-5_s0{t}{x}.mp4").write_text(f"take{t or '_t1'}{x}")
(M / "out" / "talk-5_s0.mp4").write_text("take_t2")         # the old swap: the join only
(M / "out" / "talk-5_s0.synced").write_text(json.dumps({"talk-5_s0": 1.0, "talk-5_s0_t2": 6.4}))
S._kept_pieces(M, "talk-5_s0", 2)
check("an old retaken line: the kept take's pieces swapped in under its join",
      [(M / "out" / f"talk-5_s0_c{i}.mp4").read_text() for i in (0, 1)] == ["take_t2_c0", "take_t2_c1"])
S._kept_pieces(M, "talk-5_s0", 2)
check("...once", [(M / "out" / f"talk-5_s0_c{i}.mp4").read_text() for i in (0, 1)] == ["take_t2_c0", "take_t2_c1"])
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
