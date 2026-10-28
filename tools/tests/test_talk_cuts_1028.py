"""10-28 user: a line chopped into ~2 s close-ups of one speaker, cut mid-phrase; "Watson" got a sliver before a cut.
Back-to-back talk shots of one speaker in one framing are one shot; a cut inside a line lands in the voice's pause;
no pause near = the cut stays."""
import json, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

d = Path(tempfile.mkdtemp(prefix="cuts_"))
wav = d / "v.wav"      # speech-like: tone 0-4.6 s, a 0.3 s pause at 4.6-4.9, tone to 10 s
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=200:d=10", "-af",
                "volume='if(between(t,4.6,4.9),0,1)':eval=frame", str(wav)], check=True)
spans = {"Holmes": [(0.0, 10.0, {})]}
T = lambda f: {"kind": "talk", "speaker": "Holmes", "framing": f}
sh, st = S._talk_cuts([T("close"), T("close"), T("close"), T("two")], [0.0, 2.0, 3.5, 4.3], 10.0, 0.0, spans, wav,
                      lambda v: v)
check("back-to-back close-ups of one speaker are one shot", [x["framing"] for x in sh] == ["close", "two"], sh)
check("the cut to the two-shot lands in the pause (4.6-4.9 s)", 4.6 <= st[1] <= 4.9, st)
sh, st = S._talk_cuts([T("close"), T("two")], [0.0, 7.5], 10.0, 0.0, spans, wav, lambda v: v)
check("no pause near: the cut stays where it was", abs(st[1] - 7.5) < 1e-6, st)
sh, st = S._talk_cuts([T("close"), {"kind": "talk", "speaker": "Helen", "framing": "close"}], [0.0, 4.3], 10.0, 0.0,
                      spans, wav, lambda v: v)
check("a change of speaker stays a cut", len(sh) == 2, sh)
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "Holmes turned to his desk and drew out a small case-book, which he consulted."},
      {"t": 10.0, "kind": "quote", "speaker": "Holmes", "text": "Farintosh,"},
      {"t": 11.0, "kind": "narration", "speaker": "narrator", "text": "said he."},
      {"t": 11.8, "kind": "quote", "speaker": "Holmes", "text": "Ah yes, I recall the case."},
      {"t": 30.0, "kind": "quote", "speaker": "Helen", "text": "It is easy for me to be so."},
      {"t": 40.0, "kind": "narration", "speaker": "narrator", "text": "She clasped her hands and looked down at the fire for a long while before she spoke again."}]
tc = S._turn_cuts(tl)
check("no cut inside one speaker's turn ('said he' and the line after it)", {11.0, 11.8} <= tc["inside"], tc)
check("a change of speaker and a narration paragraph are good cuts", {10.0, 30.0, 40.0} <= tc["change"], tc)
ps = S.passages(tl, 0.0, 50.0, mx=12.0)
check("passages never break inside the turn", all(abs(a_ - c) > 1e-6 for a_, _ in ps for c in (11.0, 11.8)), ps)
f = S._shot_faults({"shots": [{"start": 0.0, "kind": "quiet", "who": ["Sherlock Holmes"], "dur": 3.0,
                               "pose": {"Sherlock Holmes": "standing"}, "action": "He looks up.",
                               "picture": "Sherlock Holmes stands at the desk. A small case-book sits on the desk."}]},
                   ["Sherlock Holmes"], 3.0, [], [])
check("a thing that 'sits' is no one's posture", not any("someone sitting" in x for x in f), f)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
