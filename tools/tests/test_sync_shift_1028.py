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
check("a shift is capped at half a second", meas(-13, 9.0) == -S.SYNC_MAX_SHIFT)
check("a reading at the edge of the search (-15 of 15) moves nothing", meas(-15, 4.0) == 0 and meas(14, 6.0) == 0)
check("a measure of an older clip moves nothing", meas(3, 8.0, sig="1:2") == 0)
check("no measure: no shift", (c.with_suffix(".sync.json").unlink(), S.sync_shift(c))[1] == 0)
c.with_suffix(".json").write_text(json.dumps({"pre": 12}))
check("a talk clip's lead-in is read from its job", S.talk_pre(c) == 12)
c.with_suffix(".json").unlink()
(d / "talk-1_s0_c0.json").write_text(json.dumps({"pre": 12}))
check("a stitched line's lead-in is its first piece's", S.talk_pre(c) == 12)
(d / "talk-1_s0_c0.json").unlink()
check("an old clip has no lead-in", S.talk_pre(c) == 0)
check("a two-shot's speaker zone is their layout box", S._speaker_box({"who": ["Helen Stoner", "Sherlock Holmes"]},
      "Sherlock Holmes") == S.group_layout(2)[1][0])
check("four people: the speaker's own box, not a half", S._speaker_box({"who": list("ABCD")}, "B") ==
      S.group_layout(4)[1][0] and S._speaker_box({"who": ["A"]}, "A") is None)
(d / "out").mkdir(exist_ok=True)
(d / "out" / "talk-3_s0_c0.json").write_text(json.dumps({"speaker_box": [0.02, 0.1, 0.5, 1.0]}))
check("a joined line's zone comes from its first piece", S._job_speaker_box(d, "talk-3_s0") == [0.02, 0.1, 0.5, 1.0])
check("10-28 talk defaults: 8 s pieces, 20 steps, STG off, framing lock 0.3", S.TALK_PIECE == 193 and
      S.TALK_STEPS == 20 and S.TALK_GUIDER == {"stg_scale": 0.0, "modality_scale": 1.0} and S.TALK_LOCK == 0.3 and (S.TALK_PIECE - 1) % 8 == 0)
src = (HERE.parent / "shot_flow.py").read_text()
check("the assembly drops the lead-in plus the sync shift", "st_ = talk_pre(c) + sync_shift(c)" in src)
check("talk jobs carry the settings, and a clip made with other settings is drawn again",
      '"steps": TALK_STEPS, "guider": dict(TALK_GUIDER), "lock_end": None if close else TALK_LOCK' in src and
      'for k in ("steps", "guider", "lock_end", "base")' in src)
check("talk audio starts TALK_PRE frames of silence early", "adelay={TALK_PRE / 24 * 1000:.0f}" in src and
      S.TALK_PRE >= S.SYNC_MAX_SHIFT)
check("the assembly shifts talk clips, the render measures them after the stitch",
      "sync_shift(c) if s.get(\"kind\") == \"talk\" else 0" in src and "_sync_offsets(book, L, sorted(" in src)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
