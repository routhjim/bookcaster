"""11-24: a place whose pictures are by night is drawn by night; a daytime place and one without evidence keep
their prompt."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
from tts_reader import video
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)
night = {"pictures": [{"where": "Hall", "prompt": "Orla waits under the stars at midnight"},
                      {"where": "Hall", "prompt": "Tam enters by moonlight"},
                      {"where": "Mill", "prompt": "Tam grinds corn in the afternoon"},
                      {"where": "Barn", "prompt": "Vane sits"}]}
n_in = video.place_light(night, "Hall", "Show the INSIDE of Hall: ")
check("a night interior", n_in.startswith("Night:") and "window" in n_in, n_in)
n_out = video.place_light(night, "Hall", "Show the OUTSIDE of Hall, ")
check("a night exterior", n_out.startswith("Night under"), n_out)
check("a daytime place keeps its prompt", video.place_light(night, "Mill", "") == "")
check("no evidence: no lighting claim", video.place_light(night, "Barn", "") == "")
check("an explicit when wins", video.place_light({"pictures": [{"where": "Mill", "when": "dusk", "prompt": "at noon"}]},
                                                 "Mill", "").startswith("Dusk"))
print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
