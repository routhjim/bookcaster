"""11-08: song mode (tts_reader/song.py) - a song filmed by the shot flow as a music video. The lyrics' sections and
lines, each line's span from its words' times, the section starts the director's passages are cut at, the song
marker, the song-only prompt text (and that a book's prompts are untouched), and the song's own audio in the cut."""
import inspect, json, re, sys, tempfile
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
check("the song rules are general (no quoted examples: only the answer's own field names)",
      '"' not in re.sub(r'"(talk|people|setting|group|beat)"', "", SF.SONG_RULES), SF.SONG_RULES)
check("a song's talk shots are staged by the director, varied across the song",
      'gives "people" and "setting"' in SF.SONG_RULES and "never the same picture twice running" in SF.SONG_RULES)
src_d = inspect.getsource(SF.direct)
check("... and their staging is what the talk picture holds (not one carried pose)", "if song:" in src_d and
      'lp[w_] = pose_cat(pl.get("pose"))' in src_d)
check("a line sung by many is marked ALL SING, and its group shot sings together", "ALL SING" in SF.SONG_RULES and
      "chorus=True" in src_d)
check("a chorus clip has everyone singing, no head count, and is measured but not retaken on SyncNet",
      'if s.get("chorus"):' in inspect.getsource(SF._talk_prompt) and
      '"people": None if s.get("chorus")' in inspect.getsource(SF.ltx) and
      "not gated" in inspect.getsource(SF._sync_as_they_land))
# a fixture is never held in a key: a hand rests on it; a hand prop stays held
check("the helm the captain stands at is not held", "holding" not in SF.fixed_holds(
    "The Captain stands behind the helm, holding helm, on the left", {}))
check("a set fixture is not held", "resting on the ship's rail" in SF.fixed_holds(
    "Elsie stands by the boat, holding the ship's rail", {"fixtures": ["ship's rail"]}))
check("a hand prop stays held", SF.fixed_holds("Tom sits on a crate, holding the fiddle", {"fixtures": ["helm"]}) ==
      "Tom sits on a crate, holding the fiddle")
check("a talk clip of a song never quotes a [section] label", "\\[[^\\]]*\\]" in inspect.getsource(SF.spoken_words))
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

# 11-08 (the user's notes on the smoke clip): the author's direction, verbatim, reaches the planning prompts
import registry as RGm
from tts_reader import video as V, song as S2
with tempfile.TemporaryDirectory() as d:
    b = Path(d)
    (b / "work" / "video").mkdir(parents=True)
    check("a book has no direction", RGm.Book(b).direction() == "")
    (b / "song.json").write_text(json.dumps({"direction": "make it merry", "tags": "Lively, cozy"}))
    dr = RGm.Book(b).direction()
    check("a song's direction is its author's words, verbatim, with the recording's notes",
          '"make it merry"' in dr and "Lively, cozy" in dr, dr)
    asked = []
    V._chat = lambda url, model, prompt, max_tokens=0: asked.append(prompt) or "{}"
    V.plan_wardrobe(["p"], {"A": "a man"}, "", "", direction=dr)
    check("the wardrobe planner reads it", dr in asked[-1])
    try:
        V.plan_pictures(["p"], "", "", "", 3, direction=dr)
    except ValueError:
        pass
    check("the look planner reads it", dr in asked[-1])
    try:
        V.plan_pictures(["p"], "", "", "", 3)
    except ValueError:
        pass
    check("... and a book's look ask has no direction in it", "AUTHOR'S DIRECTION" not in asked[-1])
    check("every person the text shows acting gets a look, not only recurring ones",
          "INDIVIDUAL living person" in asked[-1] and "recurring character" not in asked[-1])
    check("... and a group is never a character (it is the picture's crowd)",
          "never a character" in asked[-1] and "never added" in V.MISSING_ASK)
    try:
        V.plan_pictures(["p"], "", "", "", 3, names=["Elsie"])
    except ValueError:
        pass
    check("a re-plan keeps the people's names", "already uses for its people: Elsie" in asked[-1])
    import plan_review as PR
    check("the plan review never makes a crowd mention a cast member", "never put in 'who' as a cast member" in
          inspect.getsource(PR))
    check("a book's wardrobe key is as before", V.wardrobe_inputs_key({"look": {}}, "t") ==
          V.wardrobe_inputs_key({"look": {}}, "t", direction=""))
    check("a direction is part of the wardrobe's key",
          V.wardrobe_inputs_key({"look": {}}, "t") != V.wardrobe_inputs_key({"look": {}}, "t", direction=dr))
    # a changed direction plans the looks, places, style and picture plan again (pins kept; LoRAs left to their checks)
    (b / "art_style.txt").write_text("old style\n")
    (b / "work" / "video" / "look.json").write_text(json.dumps({"characters": {"A": "a", "B": "b"},
                                                                 "places": {"P": "Outside: p"}, "_pinned": {"B": "b"}}))
    (b / "work" / "video" / S2.STEM).mkdir()
    (b / "work" / "video" / S2.STEM / "plan.json").write_text("{}")
    S2.replan_for_direction(b, log=lambda *a: None)
    lk = json.loads((b / "work" / "video" / "look.json").read_text())
    check("a new direction plans looks again, a pinned look kept", lk["characters"] == {"B": "b"} and not lk["places"]
          and not (b / "art_style.txt").exists() and not (b / "work" / "video" / S2.STEM / "plan.json").exists(), lk)
    check("... and the names it had are kept for the re-plan", lk.get("_names") == ["A", "B"], lk.get("_names"))
    # a person the chapter shows acting with no planned look gets one (asked once more after the picture plan)
    V._chat = lambda url, model, prompt, max_tokens=0: (asked.append(prompt) or
                                                         '{"characters": {"The Cook": "a man of about 40", '
                                                         '"Elsie": "dup", "The Galley": "Inside: a galley"}}')
    got = V.missing_people(["The cook hid almonds."], {"Elsie": "a girl"}, {"The Galley": "Inside: a galley"}, "", "",
                           direction=dr)
    check("a person acting without a look gets one; the planned and the places are not added",
          got == {"The Cook": "a man of about 40"} and dr in asked[-1], got)
    check("a new direction re-chooses the art style (the chapter's style.txt goes too)",
          '"style.txt"' in inspect.getsource(S2.replan_for_direction))
src_bd = inspect.getsource(SF.breakdown) + inspect.getsource(SF.direct) + inspect.getsource(SF.place_sheets) + \
    inspect.getsource(SF.describe_places)
check("the breakdown, director, set sheets and place descriptions read the direction",
      src_bd.count("direction()") >= 4, src_bd.count("direction()"))
check("'(inside)'/'(outside)' are one place's two sides; another room is a place of its own",
      "a different room" in SF.SIDES_RULE and "SIDES_RULE" in inspect.getsource(SF.breakdown)
      and "a different room" not in SF.BREAKDOWN_ASK)     # (song only: no book's cached breakdown changes)

# a place named for one side never takes the other side's description (the cabin drawn as the open deck)
with tempfile.TemporaryDirectory() as d:
    b = Path(d)
    (b / "work" / "video").mkdir(parents=True)
    (b / "work" / "video" / "look.json").write_text(json.dumps({"places": {"Ship Deck": "Outside: an open deck"}}))
    R2 = RGm.Book(b)
    check("'X (inside)' with only an outside description is undescribed", R2.place_look("Ship Deck (inside)") == "",
          R2.place_look("Ship Deck (inside)"))
    check("... 'X (outside)' and 'X' keep it", R2.place_look("Ship Deck (outside)").startswith("Outside:")
          and R2.place_look("Ship Deck").startswith("Outside:"))
    (b / "work" / "video" / "look.json").write_text(json.dumps({"places": {"Mill": "A mill by the river"}}))
    check("a description with no side serves either side", RGm.Book(b).place_look("Mill (inside)") == "A mill by the river")

# a song's clap/stomp shots follow its beat: the percussion stem, its onsets, an A2Vid clip driven by it (never for a book)
with tempfile.TemporaryDirectory() as d:
    w = Path(d) / "beat.wav"
    SF.subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                       "aevalsrc='if(lt(mod(t,0.5),0.02),0.9*sin(2*PI*900*t),0)+0.02*(random(0)-0.5)':s=44100:d=4", str(w)], check=True)
    bt = SF.beat_times(w)
    check("the percussion's onsets are found on its beat", len(bt) >= 6 and all(abs((b % 0.5) - 0) < 0.04 or
                                                                            abs((b % 0.5) - 0.5) < 0.04 for b in bt), bt)
check("an impact on a beat counts, one off it does not", SF.on_beat([24, 37], [1.0]) == (1, 2))
src_l = inspect.getsource(SF.ltx)
check("a song's beat shot is an A2Vid clip driven by the drums stem", 'elif BEAT_DRIVE and s.get("beat") and key.exists() and '
      '(BEAT_DRUMS(book)).exists()' in src_l and not SF.BEAT_DRIVE and '"beat": True' in src_l)
check("the lip-sync checker skips beat clips; the cut trims their lead-in", 'not j.get("beat")' in
      inspect.getsource(SF._sync_as_they_land) and 'if s.get("beat") and s.get("kind") != "talk"' in inspect.getsource(SF.assemble))
check("the separation keeps the percussion stem beside the vocals", '"drums"' in S2._SEP_RUN and
      'drums.wav' in inspect.getsource(S2.separate))
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
