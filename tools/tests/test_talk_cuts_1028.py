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
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
