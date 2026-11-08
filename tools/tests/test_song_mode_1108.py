"""11-08: song mode (tts_reader/song.py) - a song filmed by the shot flow as a music video. The lyrics' sections and
lines, each line's span from its words' times, the section starts the director's passages are cut at, the song
marker, the song-only prompt text (and that a book's prompts are untouched), and the song's own audio in the cut."""
import inspect, json, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
from tts_reader import song as S
import shot_flow as SF

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    if not ok:
        fails.append(name)

LY = "[Verse 1]\nOne two three\nFour five six\n\n[Chorus]\nSing it out\n[Verse 2]\nSeven eight\n[Chorus]\nSing it out\n"
ls = S.parse_lyrics(LY)
check("every lyric line, with its section", [(x["section"], x["text"]) for x in ls] ==
      [("Verse 1", "One two three"), ("Verse 1", "Four five six"), ("Chorus", "Sing it out"), ("Verse 2", "Seven eight"),
       ("Chorus", "Sing it out")], ls)
check("a repeated label is a section of its own", [(a, b) for a, b in S.sections(ls)] ==
      [("Verse 1", [0, 1]), ("Chorus", [2]), ("Verse 2", [3]), ("Chorus", [4])], S.sections(ls))

# line spans: heard words give the start; a line ends when the voice falls silent (never into the next line); an
# unheard line sits between its neighbours
env = np.zeros(2000)
env[100:300] = 1.0          # line 1 sung 1.0-3.0 s
env[500:700] = 1.0          # line 2 5.0-7.0 s
env[900:1100] = 1.0         # line 3 (unheard) 9-11 s
env[1300:1500] = 1.0        # line 4 13-15 s
wt = [1.0, 1.5, 2.0, 5.0, 5.5, 6.0, None, None, 13.0, 14.0]
lw = [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
sp = S.line_times(wt, lw, env, 20.0)
check("a heard line starts at its first word", sp[0][0] == 1.0 and sp[1][0] == 5.0 and sp[3][0] == 13.0, sp)
check("... and ends where the voice falls silent", 2.9 <= sp[0][1] <= 3.2 and 6.9 <= sp[1][1] <= 7.2, sp)
check("an unheard line lands between its neighbours", 6.0 < sp[2][0] < 13.0 and sp[2][1] <= 13.0, sp)
check("no line runs into the next", all(b <= n[0] + 1e-6 for (_, b), n in zip(sp, sp[1:])), sp)

tl = [{"t": 0, "kind": "music", "section": "Verse 1"}, {"t": 1, "kind": "quote", "section": "Verse 1"},
      {"t": 5, "kind": "quote", "section": "Verse 1"}, {"t": 7.5, "kind": "music", "section": "Verse 1"},
      {"t": 9, "kind": "quote", "section": "Chorus"}, {"t": 13, "kind": "quote", "section": "Verse 2"}]
check("the director's passages cut at each new section's first line", SF.section_starts(tl) == [9.0, 13.0],
      SF.section_starts(tl))

with tempfile.TemporaryDirectory() as d:
    check("a book is no song", SF.song_of(Path(d)) is None)
    (Path(d) / "song.json").write_text(json.dumps({"singer": {"name": "X"}}))
    check("song.json marks a song", (SF.song_of(Path(d)) or {}).get("singer") == {"name": "X"})

# the song's ask: the passage header swapped only in song mode (the replaced text must still be in the book's ask)
check("the book's passage header is what song mode replaces",
      'Spoken lines say\n"SAYS ALOUD" with their start and end.' in SF.PASSAGE_ASK)
src = inspect.getsource(SF.direct)
check("song mode adds its rules to the director's", "SONG_RULES.format(singer=" in src and "SINGS" in src)
check("the song rules are general (no quoted examples)", '"' not in SF.SONG_RULES.replace('"talk"', ""),
      SF.SONG_RULES)
check("a song gets no sound effects and no narration pauses",
      "if not song:" in inspect.getsource(SF.run) and "if not song else []" in inspect.getsource(SF.run))
check("the cut is muxed on the song's own recording", "song_mux(Path(song[\"audio\"])" in inspect.getsource(SF.assemble))
with tempfile.TemporaryDirectory() as d:     # word timing: a book as before; a song only matches its sung words
    b = Path(d)
    (b / "chapters").mkdir()
    (b / "work").mkdir()
    (b / "chapters" / "x.txt").write_text("The song.\n\n[Verse 1]\nThe bell rang out\n")
    tr = b / "work" / "x.mp3"
    tr.write_bytes(b"0")
    seen = []
    def fake(cmd, **k):
        seen.append(cmd)
        Path(cmd[5]).write_text(json.dumps([["The", 1.0], ["bell", 1.4], ["rang", 1.8], ["out", 2.2]]))
        return type("R", (), {"returncode": 0})()
    real = SF.subprocess.run
    SF.subprocess.run = fake
    try:
        t_book = SF.word_times(b, "x", track=tr, log=lambda *a: None)
        (b / "song.json").write_text(json.dumps({"align": {"model": "large-v3-turbo", "prompt": "The bell rang out",
                                                            "from_word": 2}}))
        t_song = SF.word_times(b, "x", track=tr, log=lambda *a: None)
    finally:
        SF.subprocess.run = real
    check("a book's word timing: its model, the chapter's own words", seen[0][6] == SF.WHISPER_MODEL and
          t_book[4:] == [1.0, 1.4, 1.8, 2.2], (seen[0][6:], t_book))
    check("a song's: its model and prompt, its opening line and labels left out", seen[1][6:] ==
          ["large-v3-turbo", "The bell rang out"] and t_song == [None, None, None, None, 1.0, 1.4, 1.8, 2.2],
          (seen[1][6:], t_song))
    # a song's timeline lines are whole: the next [section] label is never appended to a section's last line
    from tts_reader import bookcaster as BC
    (b / "chapters" / "x.txt").write_text("[Verse 1]\nThe bell rang\n\n[Chorus]\nSing out\n")
    tl_ = [{"t": 0, "kind": "quote", "text": "The bell rang"}, {"t": 2, "kind": "quote", "text": "Sing out"}]
    (b / "work" / "x.timeline.json").write_text(json.dumps(tl_))
    BC.whole_timeline(b, "x", log=lambda *a: None)
    check("a song's timeline is left whole", json.loads((b / "work" / "x.timeline.json").read_text()) == tl_)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
