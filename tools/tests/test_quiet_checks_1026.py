"""10-26 user: quiet clips get the same as-they-land visual checks as action (judged on the iGPU while the XTXs
render, a faulted one's retake queued on the loaded runner, the better take kept); a short clip is sampled at 30% and
75% of its length. CPU only (stand-in judge).
    ~/tts_reader/.venv/bin/python3 tools/tests/test_quiet_checks_1026.py"""
import inspect, json, sys, tempfile, threading, time
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import characters as CH
import mannequin_flow as MF
import shot_flow as SF
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# sampling: a 2.4 s quiet clip -> two frames inside it; a 5 s action clip -> 1.5/3.0/4.5 as before
asked = []
real_run, real_ask = MF.subprocess.run, CH.vl_ask
def fake_run(a, *k, **kw):
    if a[0] == "ffprobe":
        return SimpleNamespace(stdout=fake_run.dur, returncode=0)
    Path(a[a.index("-frames:v") + 2]).write_text("png")
    return SimpleNamespace(stdout="", returncode=0)
MF.subprocess.run = fake_run
CH.vl_ask = lambda items: (asked.append([it["id"] for it in items]),
                           {it["id"]: {"whole": "yes", "hands": "no"} for it in items})[1]
with tempfile.TemporaryDirectory() as td:
    fake_run.dur = "2.4\n"
    MF.fight_clip_ok(Path(td) / "q.mp4")
    fake_run.dur = "5.04\n"
    MF.fight_clip_ok(Path(td) / "a.mp4")
MF.subprocess.run, CH.vl_ask = real_run, real_ask
check("short quiet clip sampled at 30% and 75% (0.7 s, 1.8 s)", asked[0] == ["0.7", "1.8"], asked)
check("action clip still sampled at 1.5 / 3.0 / 4.5 s", asked[1] == ["1.5", "3.0", "4.5"], asked)

# as they land: quiet jobs go through the same judge, the faulted one's retake queued
with tempfile.TemporaryDirectory() as d:
    td = Path(d); o = td / "out"; o.mkdir()
    more, done = td / "quiet_more.json", td / "quiet_more.done"
    more.write_text("[]")
    MF.fight_clip_ok = lambda p: ["deformed body at 0.7 s"] if p.name.startswith("Q2") else []
    SF.subprocess.run = lambda *a, **k: None
    stop = threading.Event()
    th = threading.Thread(target=SF._judge_as_they_land, args=(o, [{"name": "Q1_s0", "seed": SF.QUIET_SEED},
                          {"name": "Q2_s0", "seed": SF.QUIET_SEED}], more, done, stop, print))
    th.start()
    (o / "Q1_s0.mp4").write_text("x"); (o / "Q2_s0.mp4").write_text("x")
    th.join(40)
    check("quiet: judged as they land, the faulted one's retake queued (seed RETAKE)",
          json.loads(more.read_text()) == [{"name": "Q2_s0_t2", "seed": SF.RETAKE}] and (o / "Q1_s0.judged").exists(),
          more.read_text())
    # after the render: quiet takes compared, the cleaner kept; a quiet shot with one faulted take comes back
    SF.shot_dir = lambda b, s: td
    lo = td / "ltx" / "out"; lo.mkdir(parents=True)
    (lo / "Q2_s0.mp4").write_text("take1"); (lo / "Q2_s0_t2.mp4").write_text("take2"); (lo / "Q3_s0.mp4").write_text("x")
    MF.fight_clip_ok = lambda p: [] if p.name == "Q2_s0_t2.mp4" else ["deformed body at 0.7 s"]
    again = SF.judge_takes(td, "s", [{"id": "Q2_s0", "kind": "quiet"}, {"id": "Q3_s0", "kind": "quiet"},
                                     {"id": "T_s0", "kind": "talk"}], print)
    check("quiet: the cleaner retake is kept under the shot's name", (lo / "Q2_s0.mp4").read_text() == "take2")
    check("quiet: a single faulted take is taken once more; talk is never judged", again == {"Q3_s0"}, again)
# the render wiring: quiet gen gets the as-they-land judge + --more queue; a retake builds a second quiet job
src = inspect.getsource(SF)
q = src[src.index("    if quiet:\n        # 10-26"):src.index("def judge_talk")]
check("quiet gen runs _judge_as_they_land with --more/--more-done", "_judge_as_they_land" in q and "--more" in q
      and "--more-done" in q, q[:200])
check("quiet job builder adds the RETAKE take for a shot in retake", "(QUIET_SEED,) + ((RETAKE,) if s[\"id\"] in (retake or ())" in src)
print(json.dumps({"fails": fails}))
