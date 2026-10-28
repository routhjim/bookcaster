"""10-28 user ("version 3 is definitely an improvement ... let's add that to our flow"): a talk clip's picture is
shifted by SyncNet's offset only when the reading is confident; the sign follows the measured test (+ = the picture
lags -> shown earlier); capped; a stale measure (the clip changed) moves nothing."""
import json, os, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

d = Path(tempfile.mkdtemp(prefix="sync_"))
c = d / "talk-1_s0.mp4"
c.write_bytes(b"x" * 100)
def meas(off, conf, sig=None):
    st = c.stat()
    c.with_suffix(".sync.json").write_text(json.dumps({"sig": sig or f"{st.st_size}:{st.st_mtime_ns}", "offset": off,
                                                       "conf": conf}))
    return S.sync_shift(c)
check("a confident lag of 2 (25 fps) shows the picture 2 frames earlier", meas(2, 6.1) == 2)
check("a confident lead of -1 holds the first frame 1 frame", meas(-1, 4.0) == -1)
check("a weak reading moves nothing (the user's pick: version 3)", meas(-12, 1.6) == 0)
check("a shift is capped at half a second", meas(-15, 9.0) == -S.SYNC_MAX_SHIFT)
check("a measure of an older clip moves nothing", meas(3, 8.0, sig="1:2") == 0)
check("no measure: no shift", (c.with_suffix(".sync.json").unlink(), S.sync_shift(c))[1] == 0)
src = (HERE.parent / "shot_flow.py").read_text()
check("the assembly shifts talk clips, the render measures them after the stitch",
      "sh = sync_shift(c) if s.get(\"kind\") == \"talk\" else 0" in src and "_sync_offsets(book, L, sorted(" in src)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
