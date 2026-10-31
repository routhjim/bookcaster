"""10-30 user ("re-roll the seed and keep the take with the best SyncNet" - "yeah we should do that"): a one-piece
talk shot under TALK_SYNC_BAR gets a new-seed take on the loaded runner's queue, up to TALK_TAKES in all, and the
best-scoring take becomes the shot's clip. A fake runner makes the clips; SyncNet is scripted."""
import json, sys, tempfile, threading, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

L = Path(tempfile.mkdtemp(prefix="retake_")); (L / "out").mkdir()
CONF = {"talk-a_s0": 7.0,                                                  # good first time: no retake
        "talk-b_s0": 1.0, "talk-b_s0_t2": 6.7,                              # second take passes: stop there
        "talk-c_s0": 2.0, "talk-c_s0_t2": 3.5, "talk-c_s0_t3": 1.0,         # never passes: best of 3 (take 2)
        "talk-d_s0_c0": 0.5, "talk-d_s0_c1": 0.5}                          # pieced line: not retaken
def fake_sync(book, L_, names, log=print, quiet=False):
    for n in names:
        c = L_ / "out" / f"{n}.mp4"
        st = c.stat()
        (L_ / "out" / f"{n}.sync.json").write_text(json.dumps({"sig": f"{st.st_size}:{st.st_mtime_ns}", "offset": 0,
                                                              "conf": CONF.get(n, CONF.get(n + "_c0"))}))
S._sync_offsets = fake_sync
S._stitch_talk = lambda book, R, L_, talk, log=print, only=None, quiet=False: [
    (L_ / "out" / f"{n}.mp4").write_text(n) for n in only or ()]
S.time = types.SimpleNamespace(sleep=lambda s: time.sleep(0.02))
talk = [{"name": "talk-a_s0", "seed": 100, "chunk": [0, 1, 0, 97]}, {"name": "talk-b_s0", "seed": 200, "chunk": [0, 1, 0, 97]},
        {"name": "talk-c_s0", "seed": 300, "chunk": [0, 1, 0, 97]},
        {"name": "talk-d_s0_c0", "seed": 400, "chunk": [0, 2, 0, 193]}, {"name": "talk-d_s0_c1", "seed": 400, "chunk": [1, 2, 192, 9]}]
more = L / "more.json"; more.write_text("[]")
stop, seen = threading.Event(), set()
def runner():                           # renders the first jobs, then whatever lands on --more
    for j in talk:
        (L / "out" / f"{j['name']}.mp4").write_text(j["name"])
    while not stop.is_set():
        for j in json.loads(more.read_text()):
            if j["name"] not in seen:
                seen.add(j["name"]); time.sleep(0.05)
                (L / "out" / f"{j['name']}.mp4").write_text(j["name"] + f" seed {j['seed']}")
        time.sleep(0.02)
th = threading.Thread(target=runner, daemon=True); th.start()
logs = []
S._sync_as_they_land(None, None, L, talk, threading.Event(), logs.append, False, more)
stop.set()
q = {j["name"]: j["seed"] for j in json.loads(more.read_text())}
check("a good first take is not retaken", not any(n.startswith("talk-a") for n in q), q)
check("a miss gets take 2 with a new seed and stops once it passes", "talk-b_s0_t2" in q and "talk-b_s0_t3" not in q
      and q["talk-b_s0_t2"] != 200, q)
check("never passing: TALK_TAKES takes in all, no more", {n for n in q if n.startswith("talk-c")} ==
      {f"talk-c_s0_t{k}" for k in range(2, S.TALK_TAKES + 1)}, q)
check("the passing take becomes the shot's clip", (L / "out" / "talk-b_s0.mp4").read_text().startswith("talk-b_s0_t2"),
      (L / "out" / "talk-b_s0.mp4").read_text())
check("...with its sync record", json.loads((L / "out" / "talk-b_s0.sync.json").read_text())["conf"] == 6.7)
check("best of three kept when none passes (take 2, 3.5)",
      (L / "out" / "talk-c_s0.mp4").read_text().startswith("talk-c_s0_t2"), (L / "out" / "talk-c_s0.mp4").read_text())
check("a pieced line is not retaken (its pieces chain)", not any(n.startswith("talk-d") for n in q), q)
check("the shot is marked so a later run doesn't retake it", (L / "out" / "talk-c_s0.synced").exists())
check("kept takes are logged", any("take 2 of 3 kept" in m for m in logs), logs)
print({"fails": fails})
