"""10-28 user: SyncNet runs on the iGPU as the XTXs' talk clips land - a one-piece clip at once, a pieced line joined
and measured when its last piece lands; a join from tonight's pieces is kept (not made again after the render)."""
import json, sys, tempfile, threading, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

L = Path(tempfile.mkdtemp(prefix="land_")); (L / "out").mkdir()
talk = [{"name": "talk-1_s0", "chunk": [0, 1, 0, 60], "line": "talk-1"},
        {"name": "talk-2_s0_c0", "chunk": [0, 2, 0, 97], "line": "talk-2"},
        {"name": "talk-2_s0_c1", "chunk": [1, 2, 96, 50], "line": "talk-2"}]
calls = []
real_stitch = S._stitch_talk
S._stitch_talk = lambda book, R, L_, t, log=print, only=None, quiet=False: (
    calls.append(("stitch", tuple(sorted(only or ())))), (L_ / "out" / "talk-2_s0.mp4").write_text("j"))
S._sync_offsets = lambda book, L_, names, log=print, quiet=False: calls.append(("sync", tuple(names)))
_sl = time.sleep
S.time = types.SimpleNamespace(sleep=lambda s: _sl(0.05), time=time.time)
stop = threading.Event()
th = threading.Thread(target=S._sync_as_they_land, args=(Path("."), None, L, talk, stop, lambda m: None))
th.start()
(L / "out" / "talk-1_s0.mp4").write_text("x"); time.sleep(0.3)
check("a one-piece clip is measured as it lands", ("sync", ("talk-1_s0",)) in calls, calls)
(L / "out" / "talk-2_s0_c0.mp4").write_text("x"); time.sleep(0.3)
check("a line with a piece still rendering waits", not any(c[1] == ("talk-2_s0",) for c in calls), calls)
(L / "out" / "talk-2_s0_c1.mp4").write_text("x"); time.sleep(0.3)
check("its last piece landed: joined, then measured", calls[-2:] == [("stitch", ("talk-2_s0",)), ("sync", ("talk-2_s0",))],
      calls)
stop.set(); th.join(5)
check("the watcher ends when the render does", not th.is_alive())
# a join made from tonight's pieces is kept by the pass after the render
S._stitch_talk = real_stitch
ran = []
S.subprocess.run = lambda cmd, *a, **k: (ran.append(cmd), types.SimpleNamespace(returncode=0, stderr=""))[1]
time.sleep(0.05); (L / "out" / "talk-2_s0.mp4").write_text("tonight's join")
S._stitch_talk(Path("."), types.SimpleNamespace(LTX_PY="python"), L, talk, lambda m: None)
check("a join newer than its pieces is kept (no stitch again)", ran == [] and (L / "out" / "talk-2_s0.mp4").exists(), ran)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
