"""A key redrawn after its shot record was written is not set aside for that old record (11-25)."""
import hashlib, os, sys, tempfile, time, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flow3d as F

ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)

R = types.SimpleNamespace(char_lora=lambda cx, w: None, TALK_SINGLE=False)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    cx = types.SimpleNamespace(OUT=td)
    shot = td / "shot.json"
    d = td / "stage" / "action-1"
    d.mkdir(parents=True)
    P = {"entries": [{"id": "action-1", "route": "action", "shot": str(shot), "who": ["Orla"]}]}
    now = time.time()
    def put(f, text, age):
        f.write_text(text)
        os.utime(f, (now - age, now - age))
    sha = lambda: hashlib.sha1(shot.read_bytes()).hexdigest()[:16]

    put(shot, "new shot", 300)
    put(d / "key.shot.sha", "0123456789abcdef", 600)            # the record of the shot before
    put(d / "key.png", "k", 100)                                # the key redrawn after the new shot
    check("a key newer than its record and its shot stays", F.refresh_stale_keys(R, cx, P) == 0)
    check("its record is brought up to date", (d / "key.shot.sha").read_text() == sha())
    check("and it stays on the next run", F.refresh_stale_keys(R, cx, P) == 0)

    shot.write_text("the shot changed")                         # a current record, then the shot changes
    check("a changed shot sets the key aside", F.refresh_stale_keys(R, cx, P) == 1 and not (d / "key.png").exists())

    put(shot, "newer shot", 50)
    put(d / "key.png", "k", 100)                                # no record, the shot newer than the key
    check("a key older than its shot is set aside", F.refresh_stale_keys(R, cx, P) == 1)

print("all passed" if ok else "FAILED")
sys.exit(0 if ok else 1)
