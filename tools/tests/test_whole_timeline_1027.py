"""10-27: timelines recorded by older code kept only 160 characters of each line (the Speckled Band's director saw
4,952 of 9,812 words). bookcaster.whole_timeline restores them from the chapter."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from tts_reader import bookcaster as BC

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

b = Path(tempfile.mkdtemp(prefix="whole_tl_"))
(b / "chapters").mkdir(); (b / "work").mkdir()
para = ("It was early in April that I woke one morning to find my friend standing, fully dressed, by the side of my "
        "bed. He was a late riser, as a rule, and as the clock on the mantelpiece showed me that it was only a "
        "quarter-past seven, I blinked up at him in some surprise.")
(b / "chapters" / "01_x.txt").write_text(f"{para}\n\n“Very sorry to knock you up,” said he. “A client.”\n")
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": para[:160]},
      {"t": 20.0, "kind": "quote", "speaker": "Holmes", "text": "Very sorry to knock you up,"},
      {"t": 22.0, "kind": "narration", "speaker": "narrator", "text": "said he."},
      {"t": 23.0, "kind": "quote", "speaker": "Holmes", "text": "A client."}]
(b / "work" / "01_x.timeline.json").write_text(json.dumps(tl))
n = BC.whole_timeline(b, "01_x", log=lambda m: None)
out = json.loads((b / "work" / "01_x.timeline.json").read_text())
check("the cut paragraph is whole again (the clock is back)", n == 1 and out[0]["text"] == para, (n, out[0]["text"]))
check("whole lines are untouched", [x["text"] for x in out[1:]] == ["Very sorry to knock you up,", "said he.", "A client."],
      [x["text"] for x in out[1:]])
check("times and speakers are kept", [(x["t"], x["speaker"]) for x in out] == [(x["t"], x["speaker"]) for x in tl])
check("a second run changes nothing", BC.whole_timeline(b, "01_x", log=lambda m: None) == 0)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
