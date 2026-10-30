"""10-30 user: "narrow the prompt and redo the shot until we get rid of the dupes". A take with more people than its
shot is taken again with the HELD prompt (people only hold still) and a new seed, up to DUPE_TAKES more, until one is
clean; the best take is kept. Other faults keep the one plain retake."""
import json, sys, tempfile, threading
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

b = Path(tempfile.mkdtemp(prefix="dupes_"))
(b / "work" / "video").mkdir(parents=True)
(b / "work" / "video" / "look.json").write_text(json.dumps({"characters": {
    "Sherlock Holmes": "a man of about 40, wearing a dark Victorian frock coat",
    "Helen Stoner": "a woman of about 30, wearing a black mourning dress"}, "places": {}}))
cx = SimpleNamespace(book=b, stem="01_x")
S.MF._dress = lambda *a, **k: {}
h = S._held(cx, {"id": "action-000100_s0", "who": ["Sherlock Holmes"]})
check("the held prompt: he only holds still, nobody else comes in",
      h.startswith("The man in the dark") and "stays still" in h and "nobody else comes into the picture" in h, h)

job = {"name": "action-000100_s0", "emb": "qd_x", "prompt": "p", "seed": 7, "people": 1,
       "held": {"emb": "qh_x", "prompt": h}}
crowd = ["2 people, not 1, at 1.5 s"]
t2 = S._next_take(job, job["name"], 2, crowd)
check("an extra person: take 2 is held, a new seed", t2["name"].endswith("_t2") and t2["emb"] == "qh_x" and
      t2["prompt"] == h and t2["seed"] != 7, t2)
t3 = S._next_take(job, job["name"], 3, crowd)
check("take 3 again held, with yet another seed", t3["emb"] == "qh_x" and t3["seed"] not in (7, t2["seed"]), t3)
check(f"no more than 1 + {S.DUPE_TAKES} takes", S._next_take(job, job["name"], 2 + S.DUPE_TAKES, crowd) is None)
o2 = S._next_take(job, job["name"], 2, ["deformed body at 1.5 s"])
check("another fault: one plain retake (the prompt as it was)", o2["emb"] == "qd_x", o2)
check("... and only one", S._next_take(job, job["name"], 3, ["deformed body at 1.5 s"]) is None)

# the judge as clips land: take 1 and 2 crowded, take 3 clean -> takes queued until clean
out_ = b / "out"; out_.mkdir()
more, done = b / "more.json", b / "done"
more.write_text("[]")
verdict = {"action-000100_s0": crowd, "action-000100_s0_t2": crowd, "action-000100_s0_t3": []}
S.MF.fight_clip_ok = lambda mp4, people=None: verdict[mp4.stem]
_real_sleep = S.time.sleep
S.time.sleep = lambda s: _real_sleep(0.01)
stop = threading.Event()
th = threading.Thread(target=S._judge_as_they_land, args=(out_, [job], more, done, stop, lambda m: None), daemon=True)
(out_ / "action-000100_s0.mp4").write_text("x")
th.start()
import time as _t
for _ in range(200):                # the "runner": each queued take lands
    q = json.loads(more.read_text())
    for j in q:
        (out_ / f"{j['name']}.mp4").write_text("x")
    if done.exists():
        break
    _real_sleep(0.02)
th.join(5)
q = json.loads(more.read_text())
check("takes 2 and 3 queued, then it stopped at the clean one", [j["name"] for j in q] ==
      ["action-000100_s0_t2", "action-000100_s0_t3"] and done.exists(), [j["name"] for j in q])

# after the render: the clean take 3 becomes the shot's clip
S.shot_dir = lambda book, stem: b
(b / "ltx").mkdir(); (b / "ltx" / "out").mkdir()
for k, n in (("1", "action-000100_s0"), ("2", "action-000100_s0_t2"), ("3", "action-000100_s0_t3")):
    (b / "ltx" / "out" / f"{n}.mp4").write_text(k)
S.subprocess.run = lambda *a, **k: SimpleNamespace(returncode=0, stdout="")
S.MF.fight_clip_ok = lambda mp4, people=None: [] if mp4.read_text() == "3" else crowd
again = S.judge_takes(b, "01_x", [{"id": "action-000100_s0", "kind": "quiet", "who": ["Sherlock Holmes"]}],
                      log=lambda m: None)
check("the clean take is kept under the shot's name", (b / "ltx" / "out" / "action-000100_s0.mp4").read_text() == "3",
      (b / "ltx" / "out" / "action-000100_s0.mp4").read_text())
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
