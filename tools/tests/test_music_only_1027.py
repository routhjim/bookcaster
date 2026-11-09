"""10-27: `make --music-only` with the music server or language model down "mixed" the voices alone and reported
done (the Speckled Band's 1.25x mix had no music). It now brings the music server up, and fails if it can't."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from tts_reader import bookcaster as BC

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    if not ok:
        fails.append(name)

calls, status = [], {}
BC.http_ok = lambda url: False
BC.subprocess.run = lambda cmd, **k: calls.append(cmd) or type("R", (), {"returncode": 0})()
BC.time.sleep = lambda s: None
BC.chapter_files = lambda book: [Path("x.txt")]
BC._set_status = lambda book, **k: status.update(k)
with tempfile.TemporaryDirectory() as d:
    rc = BC.run_job(Path(d), level="music", music_only=True)
check("music-only with no music server: tried to bring it up", any("--music" in c for c in calls if isinstance(c, list)), calls)
check("... and fails loudly instead of saving the voices as the mix", rc == 1 and status.get("state") == "failed", (rc, status))
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
