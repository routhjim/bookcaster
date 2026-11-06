"""11-06: a retake flagged exactly like the take before it (the same faults at the same times) can't win: no more
takes are queued for that shot, and the best take is kept as before. Takes that fail differently go on as they did."""
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

job = {"name": "quiet-000100_s0", "emb": "qd_x", "prompt": "p", "seed": 7, "people": 2,
       "held": {"emb": "qh_x", "prompt": "held"}}
c1 = ["3 people, not 2, at 0.8 s", "3 people, not 2, at 2.8 s"]
check("take 3 after a take 2 that failed differently", S._next_take(job, job["name"], 3, c1, c1[:1]) is not None)
check("no take 3 after a take 2 that failed the same (order aside)",
      S._next_take(job, job["name"], 3, c1, c1[::-1]) is None)
check("take 2 still follows take 1 (nothing to compare)", S._next_take(job, job["name"], 2, c1) is not None)

# the judge as clips land: take 1 and take 2 crowded alike -> only take 2 queued, the judge finishes
b = Path(tempfile.mkdtemp(prefix="samefaults_"))
out_ = b / "out"; out_.mkdir()
more, done = b / "more.json", b / "done"
more.write_text("[]")
verdict = {"quiet-000100_s0": c1, "quiet-000100_s0_t2": list(c1), "quiet-000100_s0_t3": []}
S.MF.fight_clip_ok = lambda mp4, people=None: verdict[mp4.stem]
_real_sleep = S.time.sleep
S.time.sleep = lambda s: _real_sleep(0.01)
_real_run = S.subprocess.run
S.subprocess.run = lambda cmd, *a, **k: (SimpleNamespace(returncode=0, stdout="") if "systemctl" in map(str, cmd)
                                         else _real_run(cmd, *a, **k))
logs = []
th = threading.Thread(target=S._judge_as_they_land, args=(out_, [job], more, done, threading.Event(), logs.append),
                      daemon=True)
(out_ / "quiet-000100_s0.mp4").write_text("x")
th.start()
for _ in range(200):                # the "runner": each queued take lands
    for j in json.loads(more.read_text()):
        (out_ / f"{j['name']}.mp4").write_text("x")
    if done.exists():
        break
    _real_sleep(0.02)
th.join(5)
q = [j["name"] for j in json.loads(more.read_text())]
check("only take 2 queued, then the judge is done", q == ["quiet-000100_s0_t2"] and done.exists(), q)
check("the log says no more takes", any("take 2" in m and "no more takes" in m for m in logs), logs)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
