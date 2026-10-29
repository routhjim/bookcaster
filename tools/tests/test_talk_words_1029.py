"""10-29: a talk prompt quotes the line it says, in the BOOK's words (the timeline holds respellings and only each
line's first ~150 characters), picked by when each word is said (forced alignment: Whisper's word times on the voice
track matched to the book's words; here a scripted Whisper)."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

b = Path(tempfile.mkdtemp(prefix="words_"))
(b / "chapters").mkdir(); (b / "work").mkdir()
stem = "01_x"
book_txt = ('I looked up. “Farintosh,” said he. “Ah yes, I recall the case; it was concerned with an opal tiara. '
            'I think it was before your time, Watson.” She sat down. “It was a terrible night,” said she.')
(b / "chapters" / f"{stem}.txt").write_text(book_txt)
# the timeline: respelled ("tee-ar-uh") and cut short (" Watson." missing), times per line
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "I looked up."},
      {"t": 1.0, "kind": "quote", "speaker": "Holmes", "text": "Farintosh,"},
      {"t": 2.0, "kind": "narration", "speaker": "narrator", "text": "said he."},
      {"t": 3.0, "kind": "quote", "speaker": "Holmes", "text": "Ah yes, I recall the case; it was concerned with an opal tee-ar-uh. I think it was before your ti"},
      {"t": 9.0, "kind": "narration", "speaker": "narrator", "text": "She sat down."},
      {"t": 10.0, "kind": "quote", "speaker": "Stoner", "text": "It was a terrible night,"},
      {"t": 12.0, "kind": "narration", "speaker": "narrator", "text": "said she."}]
bw = S._WORD.findall(book_txt)
heard = []                                  # what a scripted Whisper hears: the spoken words, 0.5 s apart from 3.0
t_ = {"Farintosh,": 1.1}
k = 3.0
for w in "Ah yes, I recall the case; it was concerned with an opal tiara. I think it was before your time, Watson.".split():
    heard.append([w, round(k, 2)]); k += 0.5
heard = [["I", 0.1], ["looked", 0.3], ["up.", 0.5], ["Farintosh,", 1.1], ["said", 2.0], ["he.", 2.2]] + heard
track = b / "work" / f"{stem}.mp3"
track.write_bytes(b"x")
def fake_run(cmd, *a, **kw):
    Path(cmd[5]).write_text(json.dumps(heard))
    return type("R", (), {"returncode": 0})()
real = S.subprocess.run
S.subprocess.run = fake_run
times = S.word_times(b, stem, log=lambda m: None)
S.subprocess.run = real
check("every heard book word has a time", times[bw.index("“Farintosh,”")] == 1.1 and times[bw.index("tiara.")] is not None,
      list(zip(bw, times)))
check("cached beside the track", (b / "work" / f"{stem}.mp3.wordtimes.json").exists())
lines = lambda who: [(float(x["t"]), float(y["t"]), x) for x, y in zip(tl, tl[1:] + [{"t": 14.0}])
                     if x["kind"] == "quote" and x["speaker"] == who]
w = S.spoken_words(b, stem, tl, lines("Holmes"), 0.9, 8.3, times)
check("a whole clip: the book's words, its spelling (tiara, not tee-ar-uh), past the timeline's cut, no quote marks",
      w == "Farintosh, Ah yes, I recall the case; it was concerned with an opal tiara. I think it was before your time, Watson.", w)
w = S.spoken_words(b, stem, tl, lines("Holmes"), 3.0, 3.6, times)
check("a clip cut inside a long line: the words said in it", w == "Ah yes, I recall the case; it was", w)
w = S.spoken_words(b, stem, tl, lines("Stoner"), 9.9, 2.2, times)
check("a line no word of which was heard (untimed): whole, by its span", w == "It was a terrible night,", w)
check("nobody's line: nothing", S.spoken_words(b, stem, tl, [], 0, 5, times) == "")
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
