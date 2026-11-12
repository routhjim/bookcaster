"""11-12: the lip-sync reading of a talk clip, on synthetic data (no SyncNet, no audio or clips).
1. Voiced frames only: a short line in a longer clip is scored over the frames where its voice is, read from the
   line's voice cut (the other voices and the narration are digital zero there); an energy cut is the fallback.
2. A best offset at or near the edge of SyncNet's search is no lock: the bar sees 0.
3. The speaker's track: the largest face inside the speaker's zone, not a listener's face cut by the zone's edge,
   and not whichever track scores best.
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

rows = [False] * 13 + [True] * 43 + [False] * 40                       # 45% voiced, as the 11-12 line
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
edge = S.sync_score(dists([True] * 80, voiced_off=V))
check("a best offset at the edge of the search is no lock", not edge["lock"] and edge["conf"] == 0.0, edge)
check("...its raw confidence is kept for the record", edge["conf_raw"] > 0, edge)
near = S.sync_score(dists([True] * 80, voiced_off=-(V - S.SYNC_LOCK_EDGE)))
check("near the edge too", not near["lock"], near)
ok = S.sync_score(dists([True] * 80, voiced_off=-(V - S.SYNC_LOCK_EDGE - 1)))
check("inside it, the reading stands", ok["lock"] and ok["conf"] == ok["conf_raw"] > 0, ok)

# --- 3. the speaker's track -----------------------------------------------------------------------------------------
w = 691.0                                                               # a zone 0.6 of a 1152 px frame
listener = [[230, 200, 350, 335]] * 30                                  # further back, smaller
speaker = [[460, 150, 680, 370]] * 30                                   # nearest the viewer, larger
check("the largest face in the zone is the speaker's", S.speaker_track([listener, speaker], w) == 1)
sliced = [[-40, 100, 120, 400]] * 30                                    # a big face cut by the zone's edge
check("a face cut by the zone's edge is not taken for the speaker", S.speaker_track([sliced, listener], w) == 1)
check("one face: it", S.speaker_track([listener], w) == 0)
check("no face: none", S.speaker_track([], w) is None)

print({"fails": fails})
sys.exit(1 if fails else 0)
