"""10-29 user: "keep her at 1x and everyone else at 1.25x" - a per-voice pace: the voice's quoted lines at its own
pace, the rest at the book's, the timeline and the planned moments moved by the same map."""
import json, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
from tts_reader import bookcaster as BC

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

b = Path(tempfile.mkdtemp(prefix="pace_"))
(b / "work" / "video").mkdir(parents=True)
(b / "chapters").mkdir()
(b / "chapters" / "01_x.txt").write_text("x")
(b / "book.cast.json").write_text(json.dumps({"characters": {"Holmes": {}, "Stoner": {}, "Watson": {}}}))
(b / "work" / "video" / "look.json").write_text(json.dumps({"characters": {"Helen Stoner": "", "Sherlock Holmes": ""}}))
check("--who Helen -> the cast's Stoner", BC.pace_voice(b, "Helen") == "Stoner")
check("--who Helen Stoner -> Stoner", BC.pace_voice(b, "Helen Stoner") == "Stoner")
# a 7 s raw track: narration 0-2, Stoner 2-5, narration 5-7
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=7", "-ac", "2", "-b:a", "192k",
                str(b / "work" / "01_x.mp3")], check=True)
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "a"},
      {"t": 2.0, "kind": "quote", "speaker": "Stoner", "text": "b"},
      {"t": 5.0, "kind": "narration", "speaker": "narrator", "text": "c"}]
(b / "work" / "01_x.timeline.json").write_text(json.dumps(tl))
man = b / "work" / "video" / "01_x" / "anim" / "manifest.json"
BC._video_work = lambda book, f: b / "work" / "video" / f.stem
man.parent.mkdir(parents=True)
man.write_text(json.dumps([{"t": 3.0, "dur": 1.0}]))
f = b / "chapters" / "01_x.txt"
BC.set_pace(b, 1.25)
BC.apply_pace(b, f, lambda m: None, fresh=True)
dur = lambda p: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                      str(p)], capture_output=True, text=True).stdout)
check("the book at 1.25: 7 s -> 5.6 s", abs(dur(b / "work" / "01_x.mp3") - 5.6) < 0.08, dur(b / "work" / "01_x.mp3"))
m = json.loads(man.read_text())[0]
check("its moment at 3.0 s raw -> 2.4 s", abs(m["t"] - 2.4) < 0.01, m)
BC.set_pace(b, 1.25, {"Stoner": 1.0})
check("pace.txt keeps both", (b / "pace.txt").read_text() == "1.25\nStoner = 1\n", (b / "pace.txt").read_text())
check("the shown pace says both", BC.pace_text(b).endswith("pace 1.25x, Stoner 1x"), BC.pace_text(b))
changed = BC.apply_pace(b, f, lambda m: None)
check("re-paced", changed)
t = [x["t"] for x in json.loads((b / "work" / "01_x.timeline.json").read_text())]
check("Helen's line at 1.0, the narration around it at 1.25: starts 0, 1.6, 4.6", all(abs(a - e) < 0.03 for a, e in
      zip(t, [0.0, 1.6, 4.6])), t)
check("the track: 1.6 + 3 + 1.6 = 6.2 s", abs(dur(b / "work" / "01_x.mp3") - 6.2) < 0.08, dur(b / "work" / "01_x.mp3"))
m = json.loads(man.read_text())[0]
check("the moment in her line (raw 3-4 s) -> 2.6 s, 1.0 s long", abs(m["t"] - 2.6) < 0.03 and abs(m["dur"] - 1.0) < 0.03, m)
check("asked again: nothing to do", not BC.apply_pace(b, f, lambda m: None))
BC.set_pace(b, 1.25, {})
BC.apply_pace(b, f, lambda m: None)
check("her own pace dropped: back to one 1.25 stretch", abs(dur(b / "work" / "01_x.mp3") - 5.6) < 0.08 and
      (b / "work" / "01_x.pace").read_text() == "1.25", dur(b / "work" / "01_x.mp3"))
m = json.loads(man.read_text())[0]
check("...and the moment back at 2.4 s", abs(m["t"] - 2.4) < 0.03, m)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
