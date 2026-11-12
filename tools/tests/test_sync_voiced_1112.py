"""11-12: the lip-sync reading of a talk clip, on synthetic data (no SyncNet, no audio or clips).
1. Voiced frames only: a short line in a longer clip is scored over the frames where its voice is, read from the
   line's voice cut (the other voices and the narration are digital zero there); an energy cut is the fallback.
2. A best offset at the edge of SyncNet's search (the edge the assembly's shift refuses) is no lock: the bar sees 0.
3. The speaker's track: the face where the speaker is in their zone (told, else about its middle), not a listener's
   face cut by the zone's edge, not the largest, and not whichever track scores best.
4. A short voiced span is believed only when the whole clip confirms its offset, and any voiced reading only inside
   the believed offsets; otherwise the whole-clip reading stands (a few windows of an unsynced face read high by
   chance).
5. The assembly's shift goes by the whole-clip reading, as before the voiced measure.
6. A record of the whole-clip measure is used as it is unless BOOKCASTER_SYNC_REMEASURE=1.
7. Nothing unreadable raises or writes NaN: it is "not measured" with the reason.
   python3 tools/tests/test_sync_voiced_1112.py"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

V = S.SYNC_VSHIFT
rng = np.random.default_rng(0)


def dists(rows, voiced_off, silent_off=None, depth=4.0):
    """Per-window distances over 2V+1 offsets: voiced rows dip at voiced_off; silent rows dip at silent_off (or
    nowhere: a flat, noisy curve)."""
    d = 10 + rng.normal(0, 0.3, (len(rows), 2 * V + 1))
    for i, v in enumerate(rows):
        off = voiced_off if v else silent_off
        if off is not None:
            d[i, V - off] -= depth
    return d


# --- 1. voiced frames ------------------------------------------------------------------------------------------------
sr = 16000
cut = np.zeros(int(3.3 * sr), np.int16)
cut[: int(1.76 * sr)] = rng.integers(-3000, 3000, int(1.76 * sr))      # the line's voice, then silenced narration
v = S.voiced_frames((cut, sr), n=101, shift=-0.5)                       # a 0.5 s lead-in before the cut
check("the lead-in is not voiced", not any(v[:12]), v[:13])
check("the line's span is voiced", all(v[13:56]), v[13:56])
check("the silenced rest is not voiced", not any(v[57:]), v[57:])
pc = S.voiced_frames((cut, sr), n=50, shift=1.0 - 0.5)                  # a piece starting 1.0 s into the line
check("a piece's frames are read from where it starts in the line", all(pc[:30]) and not any(pc[33:]), pc)
quiet = rng.integers(-20, 20, sr * 4).astype(np.int16)
quiet[sr: 2 * sr] = rng.integers(-4000, 4000, sr)
ev = S.energy_frames(quiet, sr, 100)
check("energy fallback: the loud second is voiced, the hiss is not", all(ev[26:49]) and not any(ev[:24] + ev[51:]),
      ev)

rows = [False] * 13 + [True] * (S.SYNC_ROWS_SURE + 6) + [False] * 46      # under half voiced, as the 11-12 line
d = dists(rows, voiced_off=-3, silent_off=None)
d[np.array(rows) == False] += rng.normal(0, 2.0, (rows.count(False), 2 * V + 1))   # silence: a mouth moving at random
r_all, r_v = S.sync_score(d), S.sync_score(d, rows)
check("scored over the voiced windows, the in-sync line reads higher than over the whole clip",
      r_v["conf"] > r_all["conf"], (r_all, r_v))
check("...at the line's own offset", r_v["offset"] == -3, r_v)
few = [False] * 90 + [True] * 6
r_few = S.sync_score(dists(few, -2, -2), few)
check("too few voiced windows: the whole clip is scored", r_few["all_rows"] and r_few["rows"] == 96, r_few)

# --- 2. no lock at the search edge ----------------------------------------------------------------------------------
E = V - S.SYNC_LOCK_EDGE
edge = S.sync_score(dists([True] * 80, voiced_off=V))
check("a best offset at the edge of the search is no lock", not edge["lock"] and edge["conf"] == 0.0, edge)
check("...its raw confidence is kept for the record", edge["conf_raw"] > 0, edge)
near = S.sync_score(dists([True] * 80, voiced_off=-E))
check("the frame inside the edge too (the assembly's shift refuses both)", not near["lock"], near)
check("the measure's edge is the assembly's", S.SYNC_LOCK_EDGE == 1)
ok = S.sync_score(dists([True] * 80, voiced_off=-(E - 1), depth=8.0))
check("a strong, steady offset one frame inside it stands (the assembly shifts it, clamped)",
      ok["lock"] and ok["conf"] == ok["conf_raw"] > S.TALK_SYNC_BAR and ok["offset"] == -(E - 1), ok)
check("...read over the whole clip, since it is outside the believed offsets of a voiced reading",
      ok["basis"] == "whole" and not S.SYNC_BAND[0] <= ok["offset"] <= S.SYNC_BAND[1], ok)

# --- 3. the speaker's track -----------------------------------------------------------------------------------------
w = 691.0                                                               # a zone 0.6 of a 1152 px frame
listener = [[60, 200, 180, 335]] * 30                                   # further back, smaller, toward the zone's side
speaker = [[300, 150, 520, 370]] * 30                                   # about the zone's middle
check("the face about the zone's middle is the speaker's", S.speaker_track([listener, speaker], w) == 1)
big_near = [[40, 80, 330, 400]] * 30                                    # a listener nearer the viewer: the largest
check("...not the largest face (a listener nearer the viewer)", S.speaker_track([big_near, speaker], w) == 1)
check("told where the speaker is, the face there - even off the middle",
      S.speaker_track([speaker, listener], w, at=0.17) == 1 and S.speaker_track([speaker, listener], w, at=0.6) == 0)
sliced = [[-40, 100, 120, 400]] * 30                                    # a big face cut by the zone's edge
check("a face cut by the zone's edge is not taken for the speaker", S.speaker_track([sliced, listener], w) == 1)
check("...not even when it is where the speaker was said to be", S.speaker_track([sliced, listener], w, at=0.05) == 1)
check("one face: it", S.speaker_track([listener], w) == 0)
check("no face: none", S.speaker_track([], w) is None)

# --- 4. short spans and the believed offsets ------------------------------------------------------------------------
F = S.SYNC_ROWS_SURE
def clip(n_voiced, voiced_off, whole_off, n_silent=70, depth_silent=4.0):
    rows = [True] * n_voiced + [False] * n_silent
    d = dists(rows, voiced_off=voiced_off, silent_off=whole_off, depth=4.0)
    return d, rows
d, rows = clip(F - 5, voiced_off=-7, whole_off=0)                       # the 11-12 noise case: 3.5 at -7, clip at 0
r = S.sync_score(d, rows)
check("a short voiced span the whole clip does not confirm: the whole-clip reading stands",
      r["basis"] == "whole" and r["offset"] == 0 and "why" in r, r)
check("...the voiced reading is kept beside it", r["voiced"]["offset"] == -7 and r["whole"]["offset"] == 0, r)
d, rows = clip(F - 5, voiced_off=-3, whole_off=-2)
d[np.array(rows) == False, V + 2] -= 3.0                                # the rest of the clip dips firmly at -2 too
r = S.sync_score(d, rows)
check("a short span the whole clip confirms (within SYNC_AGREE, itself a lock): the voiced reading counts",
      r["basis"] == "voiced" and r["offset"] == -3 and "why" not in r and r["whole"]["conf"] >= S.SYNC_MIN_CONF, r)
rows = [True] * (F - 5) + [False] * 70                                  # the wrong-voice case (11-12 check): a chance
d = dists(rows, voiced_off=0, silent_off=None)                          # peak on a short span, the rest of the clip
d[np.array(rows) == False] += rng.normal(0, 1.5, (70, 2 * V + 1))       # noise that happens to sit near it
d[np.array(rows) == False, V - 1] -= 0.6
r = S.sync_score(d, rows)
check("a short span beside a whole clip that reads no lock of its own: the whole-clip reading stands",
      r["basis"] == "whole" and r["whole"]["conf"] < S.SYNC_MIN_CONF and "why" in r, r)
check("the floor is the wrong-voice check's (no chance pass at 44 windows, two of twenty at 30)", F == 44, F)
d, rows = clip(F + 5, voiced_off=-3, whole_off=6)
r = S.sync_score(d, rows)
check("a long enough span needs no confirmation", r["basis"] == "voiced" and r["offset"] == -3, r)
lo, hi = S.SYNC_BAND
rows = [True] * (F + 30) + [False] * 30
d = dists(rows, voiced_off=lo - 1, silent_off=None)
d[np.array(rows) == False] += rng.normal(0, 3.0, (rows.count(False), 2 * V + 1))
r = S.sync_score(d, rows)
check("a voiced reading outside the believed offsets: the whole-clip reading decides",
      r["basis"] == "whole" and r["conf"] == r["whole"]["conf"] < r["voiced"]["conf"], r)
check("the believed offsets hold every offset passing clips read (-5..+1), with room", lo <= -5 - 2 and hi >= 1 + 2,
      (lo, hi))
d, rows = clip(F + 20, voiced_off=V, whole_off=-3)                      # voiced at the edge, the clip locked at -3
r = S.sync_score(d, rows)
check("a voiced reading at the edge falls back to the clip's own lock, not to nothing",
      r["basis"] == "whole" and r["lock"] and r["offset"] == -3, r)
check("the floor is a setting", isinstance(F, int) and F >= 30)

# --- 5. the shift goes by the whole-clip reading --------------------------------------------------------------------
import json, tempfile
tmp = Path(tempfile.mkdtemp())
def rec_shift(rec):
    c = tmp / "c.mp4"; c.write_bytes(b"x" * 7)
    st = c.stat()
    c.with_suffix(".sync.json").write_text(json.dumps(dict(rec, sig=f"{st.st_size}:{st.st_mtime_ns}")))
    return S.measured_shift(c)
new = {"measure": S.SYNC_MEASURE, "lock": True, "conf": 4.9, "offset": -9, "old": {"conf": 1.9, "offset": -9}}
check("a weak whole-clip reading moves nothing, whatever the voiced one reads", rec_shift(new) is None)
new = {"measure": S.SYNC_MEASURE, "lock": True, "conf": 7.5, "offset": -2, "old": {"conf": 5.8, "offset": -3}}
check("a confident one shifts by the whole-clip offset, as before", rec_shift(new) == round(-3 * S.FPS / 25))
new = {"measure": S.SYNC_MEASURE, "lock": False, "conf": 0.0, "offset": 0, "old": {"conf": 4.0, "offset": -3}}
check("a voiced reading at the edge does not take a clip's shift away", rec_shift(new) == round(-3 * S.FPS / 25))
check("a record of the old measure shifts as it always did", rec_shift({"conf": 6.0, "offset": -4}) ==
      round(-4 * S.FPS / 25) and rec_shift({"conf": 2.9, "offset": -4}) is None)
check("at the search edge: no shift", rec_shift({"conf": 6.0, "offset": V - 1}) is None)
check("13 of 15 is shifted, clamped", rec_shift({"conf": 6.0, "offset": 13}) == S.SYNC_MAX_SHIFT)
check("no reading: no shift", rec_shift({"measure": S.SYNC_MEASURE, "conf": None, "offset": 0, "old": None}) is None)
S.SYNC_SHIFT_FROM = "voiced"
new = {"measure": S.SYNC_MEASURE, "lock": True, "conf": 7.5, "offset": -2, "old": {"conf": 2.6, "offset": -3}}
check("asked to, the voiced reading shifts - when it passes the bar", rec_shift(new) == round(-2 * S.FPS / 25))
check("...and not under it", rec_shift(dict(new, conf=4.9)) is None)
S.SYNC_SHIFT_FROM = "whole"

# --- 6. old records stand unless asked ------------------------------------------------------------------------------
sig = "7:1"
check("a record of the whole-clip measure is used as it is", S._sync_current({"sig": sig, "conf": 6.0, "offset": -2},
                                                                              sig, None))
S.SYNC_REMEASURE = True
check("...and measured again when asked", not S._sync_current({"sig": sig, "conf": 6.0, "offset": -2}, sig, None))
S.SYNC_REMEASURE = False
check("a record of this measure stands", S._sync_current({"sig": sig, "measure": S.SYNC_MEASURE, "conf": 6.0}, sig, None))
check("one of the first voiced version is measured again", not S._sync_current({"sig": sig, "lock": True}, sig, None))
check("a changed clip is measured", not S._sync_current({"sig": "8:1", "measure": S.SYNC_MEASURE}, sig, None))
check("a changed zone is measured", not S._sync_current({"sig": sig, "measure": S.SYNC_MEASURE, "box": [0, 0, .5, 1]},
                                                         sig, [0.4, 0, 1, 1]))

# --- 7. nothing unreadable raises or writes NaN ---------------------------------------------------------------------
import pickle
none = S.sync_score(np.zeros((0, 2 * V + 1)))
check("no windows: no reading, with the reason", none["conf"] is None and none["why"], none)
check("distances that are not finite: no reading", S.sync_score(np.full((20, 2 * V + 1), np.nan))["conf"] is None)
check("a misshapen block: no reading", S.sync_score(np.zeros((20, 7)))["conf"] is None)
short = S.sync_score(dists([True] * 60, -2), [True] * 10)              # a voiced list shorter than the windows
check("a voiced list shorter than the windows: the rest are not voiced", short["voiced_rows"] == 10, short)
rec = S._sync_record(sig, None, {"conf": float("nan"), "offset": 3, "lock": True, "conf_raw": float("inf"),
                                 "old": {"conf": float("nan"), "offset": 0}}, [True, False], "cut")
txt = json.dumps(rec, allow_nan=False)
check("a record never holds NaN: null and the reason", rec["conf"] is None and rec["conf_raw"] is None and
      rec["old"] is None and not rec["lock"] and rec["offset"] == 0 and rec["why"] and "NaN" not in txt, rec)
good = S._sync_record(sig, None, S.sync_score(dists([True] * 80, -2), [True] * 80), [True] * 84, "cut")
check("a reading's record carries the measure it was made by", good["measure"] == S.SYNC_MEASURE and good["lock"]
      and good["offset"] == -2 and good["basis"] == "voiced", good)
work = tmp / "w"; pw = work / "pywork" / "r"; pw.mkdir(parents=True)
check("no SyncNet output: not measured", S.sync_read(work, "r", [], 100.0)["conf"] is None)
(pw / "tracks.pckl").write_bytes(b"not a pickle"); (pw / "activesd.pckl").write_bytes(b"")
r = S.sync_read(work, "r", [], 100.0)
check("a broken pickle: not measured, with the reason, no exception", r["conf"] is None and "not measured" in r["why"], r)
tr = [{"track": {"bbox": np.array([[40, 10, 60, 30]] * 84), "frame": np.arange(84)}},
      {"track": {"bbox": np.array([[5, 10, 15, 20]] * 84), "frame": np.arange(84)}}]
voiced = [False] * 12 + [True] * 50 + [False] * 30
d0 = dists([10 <= i < 60 for i in range(80)], -2)                        # rows voiced where their middle frame is
pickle.dump(tr, open(pw / "tracks.pckl", "wb")); pickle.dump([d0], open(pw / "activesd.pckl", "wb"))
r = S.sync_read(work, "r", voiced, 100.0)
check("more tracks than distances: the pairs there are, no exception", r["conf"] is not None and r["faces"] == 1, r)
check("a window is voiced when its middle frame is (row i = frames i..i+4)", r["voiced_rows"] == 50, r)
pickle.dump([d0, dists([True] * 80, 0, depth=8.0)], open(pw / "activesd.pckl", "wb"))
r = S.sync_read(work, "r", voiced, 100.0)
check("the speaker's track is read by where it is, not by which scores best", r["track"] == 0 and r["offset"] == -2, r)
check("...the best track over the whole clip is kept beside it for the shift", r["old"]["track"] == 1 and
      r["old"]["offset"] == 0, r)
pickle.dump([{"track": {"bbox": [[1, 2, 3]], "frame": [0]}}], open(pw / "tracks.pckl", "wb"))
r = S.sync_read(work, "r", voiced, 100.0)
check("tracks and distances that do not fit: not measured, no exception", r["conf"] is None and r["why"], r)

print({"fails": fails})
sys.exit(1 if fails else 0)
