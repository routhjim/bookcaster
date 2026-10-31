"""10-31: the shots flow plans, places and mixes sound effects (they had silently dropped out). Spots move onto
their cue words; an impact inside an action shot moves onto the clip's motion peak (within SNAP_S), never a soft
sound, never in a talk shot; the planner keeps the cue words and line end; effect files are named by content.
No models, GPUs or servers: the language model reply, word times and motion peaks are faked.
Run with bookcaster's python: .venv/bin/python3 tools/tests/test_shots_sfx_1031.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import shot_flow as SF  # noqa: E402
from tts_reader import score, sfx as SX  # noqa: E402

fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


# --- the planner: cue words and the line's end kept; version 2
tl = [{"t": 0.0, "speaker": "narrator", "text": "It was a quiet night."},
      {"t": 10.0, "speaker": "narrator", "text": "Then the door was suddenly dashed open by a tall man."},
      {"t": 20.0, "speaker": "narrator", "text": "He stood there."}]
score._chat = lambda *a, **k: json.dumps({"beds": [{"from": 0, "to": 2, "sound": "coal fire crackling, clock"}],
                                          "spots": [{"at": 1, "words": "dashed open", "sound": "a door slammed open",
                                                     "seconds": 1.5, "loud": "loud"}]})
plan = SX.plan_sfx(tl, 30.0, "", "")
sp = plan["spots"][0]
check("plan: version 2", plan["version"] == 2)
check("plan: cue words and line end kept", sp["words"] == "dashed open" and sp["t"] == 10.0 and sp["t_end"] == 20.0)

# --- spot_times: onto the first cue word inside the line
book = Path(tempfile.mkdtemp())
(book / "chapters").mkdir()
text = "It was a quiet night. Then the door was suddenly dashed open by a tall man. He stood there."
(book / "chapters" / "01_x.txt").write_text(text)
words = text.split()
times = [0.0 + 0.4 * i if i < 5 else 10.0 + 0.4 * (i - 5) for i in range(len(words))]
SF.word_times = lambda *a, **k: times
spots = [dict(sp)]
n = SF.spot_times(book, "01_x", spots)
want = times[words.index("dashed")]
check("words: the spot sits on 'dashed', not the line start", n == 1 and abs(spots[0]["t"] - want) < 1e-6)
spots2 = [dict(sp, words="nowhere in it")]
check("words: no match -> stays at the line start", SF.spot_times(book, "01_x", spots2) == 0 and spots2[0]["t"] == 10.0)

# --- snap_impacts
clip = book / "a.mp4"
clip.write_bytes(b"")
SF._impacts_in_clip = lambda p, fps=24: [30]                 # a peak 1.25 s into the clip
spans = [(10.0, 15.0, str(clip), "action"), (15.0, 20.0, str(clip), "talk")]
s1 = [{"t": 11.6, "sound": "a door slammed open"}]
check("impact in an action shot: onto the peak", SF.snap_impacts(s1, spans) == 1 and s1[0]["t"] == 11.25)
s2 = [{"t": 13.5, "sound": "a door slammed open"}]
check("impact too far from the peak: stays", SF.snap_impacts(s2, spans) == 0 and s2[0]["t"] == 13.5)
s3 = [{"t": 11.6, "sound": "soft footsteps on a carpet"}]
check("not an impact: stays", SF.snap_impacts(s3, spans) == 0 and s3[0]["t"] == 11.6)
s4 = [{"t": 15.5, "sound": "a knock at the door"}]
check("in a talk shot: stays", SF.snap_impacts(s4, spans) == 0 and s4[0]["t"] == 15.5)

# --- files named by content
a, b = SX._fx_name("spot", 3, "a knock", 1.5), SX._fx_name("spot", 3, "a clang", 1.5)
check("effect files: another sound at the same index is another file", a != b and a.startswith("spot003_"))
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
