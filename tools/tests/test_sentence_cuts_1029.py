"""10-29 user: "the audio clip starts in the middle of a word... cut up the talking by sentences with actual space in
between them" - cuts inside speech fall at sentence breaks (else between words); passages too."""
import json, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

pts = ([4.2, 9.8], [1.1, 2.0, 3.1, 4.2, 5.3, 6.6, 7.4, 8.9, 9.8])
check("a cut inside speech goes to the nearest sentence break", S.snap_cut(5.0, 2.0, 9.0, pts) == 4.2)
check("...none inside [lo, hi] (both shots need room): the nearest word break", S.snap_cut(5.0, 4.5, 9.0, pts) == 5.3)
check("no sentence break near: the nearest break between words", S.snap_cut(7.2, 6.0, 9.0, pts, reach=1.0) == 7.4)
check("nothing timed and no voice: the cut stays", S.snap_cut(7.2, 6.0, 9.0, None) == 7.2)
# a voice track: 3 sentences of tone with 0.4 s silences between them; a scripted Whisper
b = Path(tempfile.mkdtemp(prefix="cuts_"))
(b / "chapters").mkdir(); (b / "work").mkdir()
stem = "01_x"
(b / "chapters" / f"{stem}.txt").write_text("One two three. Four five six. Seven eight nine.")
v = b / "work" / f"{stem}.mp3"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "aevalsrc='if(between(t,0,1.6)+between(t,2.0,3.6)+between(t,4.0,5.6),sin(2*PI*300*t),0)':d=6",
                "-ac", "1", str(v)], check=True)
heard = [["One", 0.0], ["two", 0.5], ["three.", 1.0], ["Four", 2.0], ["five", 2.5], ["six.", 3.0], ["Seven", 4.0],
         ["eight", 4.5], ["nine.", 5.0]]
real = S.subprocess.run
S.subprocess.run = lambda cmd, *a, **k: (Path(cmd[5]).write_text(json.dumps(heard)), type("R", (), {"returncode": 0})())[1] \
    if cmd and cmd[0] == str(S.WHISPER_PY) else real(cmd, *a, **k)
sent, words = S.cut_points(b, stem, v)
S.subprocess.run = real
check("two sentence breaks, each in the silence (1.6-2.0, 3.6-4.0 s)", len(sent) == 2 and 1.6 <= sent[0] <= 2.0 and
      3.6 <= sent[1] <= 4.0, sent)
check("word breaks include them", set(sent) <= set(words) and len(words) == 8, words)
tl = [{"t": 0.0, "kind": "quote", "speaker": "Stoner", "text": "One two three. Four five six. Seven eight nine."}]
ps = S.passages(tl, 0.0, 6.0, mx=4.0, sent=sent)
check("a passage inside one long line ends at a sentence break", all(any(abs(b_ - c) < 1e-6 for c in sent) for _, b_ in ps[:-1]), ps)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
