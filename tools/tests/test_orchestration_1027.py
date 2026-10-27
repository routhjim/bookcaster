"""10-27 orchestration audit: one box lease for GPU jobs, `stop` gives back only what its job took, a voices-only
fallback is marked and never passes the video's mix guard, `assets` lists unless --do, one writer of a job's outcome
(failures / problems), `pace` refuses under a running job, start_job keeps the old status when it can't start, the
Jellyfin copy compares content. No real services: systemctl / systemd-run / the stack script are faked."""
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="bc-orch-"))
os.environ["BOOKCASTER_BOX_LOCK"] = str(TMP / "box.lock")
os.environ["BOOKCASTER_CARD_LOCK"] = str(TMP / "gpu-{bus}.lock")
os.environ["BOOKCASTER_STRATA_ROOM_WAIT"] = "0"
os.environ["BOOKCASTER_JELLYFIN_SENT"] = str(TMP / "sent.json")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from tts_reader import animate, boxlease, bookcaster as BC, jellyfin  # noqa: E402

fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    if not ok:
        fails.append(name)


REAL_RUN, REAL_POPEN = subprocess.run, subprocess.Popen
PY = sys.executable
calls, active = [], set()


def fake_run(cmd, **k):
    calls.append(list(map(str, cmd)))
    rc = 0
    if cmd[:4] == ["systemctl", "--user", "is-active", "-q"]:
        rc = 0 if cmd[4] in active else 3
    return types.SimpleNamespace(returncode=rc, stdout="", stderr="")


def hold(path: Path, info: dict, secs: float) -> subprocess.Popen:
    """Another process holding a flock on *path* (writing *info* in it) for *secs*."""
    code = ("import fcntl,json,sys,time; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX); f.seek(0); "
            "f.truncate(); f.write(sys.argv[2]); f.flush(); print('held',flush=True); time.sleep(float(sys.argv[3]))")
    p = REAL_POPEN([PY, "-I", "-c", code, str(path), json.dumps(info), str(secs)], stdout=subprocess.PIPE, text=True)
    p.stdout.readline()
    return p


def book(name: str) -> Path:
    d = TMP / name
    (d / "chapters").mkdir(parents=True, exist_ok=True)
    (d / "work" / "final").mkdir(parents=True, exist_ok=True)
    (d / "book.txt").write_text("x")
    (d / "chapters" / "01_one.txt").write_text("One\n\nIt was late.\n")
    return d


subprocess.run = fake_run
animate.time.sleep = lambda s: None

# --- the box lease -----------------------------------------------------------------------------------------------
other = hold(boxlease.LOCK, {"book": "other_book", "what": "job", "since": "10-27 09:00"}, 3)
check("lease: someone else's job is seen as the holder", boxlease.holder().get("book") == "other_book", boxlease.holder())
try:
    with boxlease.lease("mine", wait=False):
        check("lease: a second lease without waiting is refused", False)
except BlockingIOError as exc:
    check("lease: a second lease without waiting is refused, naming the holder", "other_book" in str(exc), exc)
calls.clear()
got = animate._servers("up", print, ["fish", "ace"])
check("servers up: refused while another job holds the box", got == [] and not any("up" in c for c in calls), calls)
try:
    animate.igpu("strata", print)
    check("igpu(strata): refused while another job holds the box", False)
except RuntimeError as exc:
    check("igpu(strata): refused while another job holds the box", "holds the box" in str(exc), exc)
b1 = book("waiter")
seen = []
real_set = BC._set_status
BC._set_status = lambda b, **k: (seen.append(dict(k)), real_set(b, **k))[1]
t = time.time()
rc = BC.run_held(b1, lambda: (seen.append({"ran": boxlease.held()}), 0)[1])
BC._set_status = real_set
other.wait()
check("run_held: the second job waited, saying for whom",
      any("waiting for other_book's job" in str(s.get("step")) for s in seen) and seen[0].get("state") == "waiting", seen)
check("run_held: then ran holding the lease", {"ran": True} in seen and time.time() - t > 1, seen)
check("run_held: box_before recorded", "box_before" in BC.read_status(b1), BC.read_status(b1))
check("lease: released after the job", boxlease.holder() == {} and not boxlease.held(), boxlease.holder())
card = hold(Path(str(TMP / "gpu-65.lock")), {}, 30)
calls.clear()
got = animate._servers("up", print, ["fish"])
check("servers up: refused onto a card whose gpu lock is held", got == [] and not any("up" in c for c in calls), calls)
card.kill()
card.wait()
calls.clear()
got = animate._servers("up", print, ["fish"])
check("servers up: only what was asked (no --music for the voice server alone)",
      got == ["fish"] and any(c[-1] == "up" for c in calls) and not any("--music" in c for c in calls), calls)
animate.mem_avail_gb = lambda: 40
animate._vae_server = lambda action, log: False
try:
    animate.igpu("strata", print)
    check("igpu(strata): refused without room for it", False)
except RuntimeError as exc:
    check("igpu(strata): refused without room for it (memory floor)", "40 GiB free" in str(exc), exc)

# --- stop: only this job's processes, only what it paused ---------------------------------------------------------
unit = "bookcaster-test-orch"
tagged = REAL_POPEN(["sleep", "30"], env=dict(os.environ, BOOKCASTER_JOB=unit))
untagged = REAL_POPEN(["sleep", "30"], env=dict(os.environ, BOOKCASTER_JOB="bookcaster-someone-else"))
time.sleep(0.2)
pids = BC.job_pids(unit)
check("job_pids: finds the job's tagged process, not another job's", tagged.pid in pids and untagged.pid not in pids, pids)
BC.time.sleep = lambda s: None
BC.end_pids(pids, print)
try:
    tagged.wait(timeout=5)
except subprocess.TimeoutExpired:
    pass
check("end_pids: the job's process ended", tagged.poll() is not None)
check("end_pids: the other job's process untouched", untagged.poll() is None)
untagged.kill()

b2 = book("stopper")
BC._status_path(b2).write_text(json.dumps({"state": "running", "step": "shots flow", "box_before": ["fish", "strata"]}))
active.clear()
active.add(BC.unit_name(b2))
igpu_calls = []
animate.igpu = lambda tenant, log=print: igpu_calls.append(tenant)
BC.job_pids = lambda u: []
calls.clear()
BC.stop_job(b2)
stack_ups = [c for c in calls if c and c[0].endswith("fish_stack.sh") and c[1] == "up"]
check("stop: the voice server it paused comes back (not the music server it never had)",
      len(stack_ups) == 1 and "--music" not in stack_ups[0], stack_ups)
check("stop: the language model it stopped comes back", igpu_calls == ["strata"], igpu_calls)
check("stop: no box-wide pgrep/kill by renderer name", not any(c[0] in ("pgrep", "pkill") for c in calls), calls)
check("stop: the status says stopped", BC.read_status(b2).get("state") == "stopped", BC.read_status(b2))
BC._status_path(b2).write_text(json.dumps({"state": "running", "step": "shots flow", "box_before": ["fish"]}))
igpu_calls.clear()
BC.stop_job(b2)
check("stop: a shots job that never had the language model doesn't start it", igpu_calls == [], igpu_calls)
active.clear()
BC._status_path(b2).write_text(json.dumps({"state": "done", "step": "finished"}))
calls.clear()
BC.stop_job(b2)
check("stop: an inactive unit is left alone - status kept, nothing started or stopped",
      BC.read_status(b2).get("state") == "done" and all(c[:4] == ["systemctl", "--user", "is-active", "-q"] for c in calls),
      (BC.read_status(b2), calls))

# --- one writer of the outcome: failures and problems -------------------------------------------------------------
b3 = book("writer")


@BC._runner
def inner(b):
    BC._set_status(b, state="running", step="shots flow", outputs=[])
    BC._set_status(b, state="failed", step="shots flow: LTX failed")
    return 1


@BC._runner
def outer(b, fail_inner):
    BC._set_status(b, state="running", step="recording", outputs=["a.mp3"])
    if fail_inner:
        inner(b)
    BC.degrade(b, "01_one WITHOUT sound effects: boom")
    BC._set_status(b, state="done", step="finished", outputs=["a.mp3"])


BC._status_path(b3).write_text("{}")
outer(b3, True)
st = BC.read_status(b3)
check("one writer: a nested runner's failure isn't overwritten by the outer 'done'",
      st["state"] == "failed" and "LTX failed" in st["step"] and "sound effects" in st["step"], st)
check("one writer: the nested runner didn't clobber the outer outputs", st.get("outputs") == ["a.mp3"], st)
BC._status_path(b3).write_text("{}")
outer(b3, False)
st = BC.read_status(b3)
check("one writer: problems only -> 'done with problems', listed", st["state"] == "done with problems"
      and "WITHOUT sound effects" in st["step"] and st["problems"], st)
check("degrade: the API shot_flow can call is bookcaster.degrade(book, text)", callable(BC.degrade))

# --- the voices-only fallback is marked; the video guard remixes first ---------------------------------------------
b4 = book("nomusic")
BC.set_pace(b4, 1.0)
(b4 / "work" / "01_one.mp3").write_bytes(b"v")
(b4 / "work" / "01_one.timeline.json").write_text("[]")
real_tts, real_vm = BC.tts, BC.voice_map


def fake_tts(*args, capture=False, log=None):
    a = list(map(str, args))
    if a[0] == "convert" and "-o" in a:
        Path(a[a.index("-o") + 1]).write_bytes(b"v")
    if a[0] == "score" and "-o" in a:
        return types.SimpleNamespace(returncode=1)             # the mix fails
    return types.SimpleNamespace(returncode=0)


BC.tts, BC.voice_map = fake_tts, lambda b: "Narrator=bob"
BC.http_ok = lambda url: True
BC._status_path(b4).write_text("{}")
rc = BC.run_job(b4, level="music")
st = BC.read_status(b4)
check("mix failed: the voices are kept as the final, marked .nomusic",
      (b4 / "work" / "final" / "01_one.mp3").exists() and BC.has_nomusic(b4, "01_one"), list((b4 / "work" / "final").iterdir()))
check("mix failed: the job says done WITH problems, WITHOUT music", st["state"] == "done with problems"
      and "WITHOUT music" in st["step"], st)
import tts_reader.score as SC  # noqa: E402
SC.audio_seconds = lambda p: 100.0
check("video guard: a .nomusic mix never matches the voices (same length or not)", not BC.mix_matches_voices(b4, "01_one"))
fake_flow3d = types.SimpleNamespace(chapter_stem=lambda b, c: "01_one")
remixed = []
sys.modules["flow3d"] = fake_flow3d
sys.modules["mannequin_flow"] = types.SimpleNamespace()
sys.modules["shot_flow"] = types.SimpleNamespace(run=lambda *a, **k: remixed.append("RENDERED"))
real_run_job = BC.run_job
BC.run_job = lambda b, level, music_only=False, sfx=False: remixed.append((level, music_only)) or 1
BC._status_path(b4).write_text("{}")
rc = BC.run_mannequin_job(b4, "1", None, flow="shots")
st = BC.read_status(b4)
check("video guard: remix first, and fail loudly when music still can't be made",
      remixed == [("music", True)] and rc == 1 and st["state"] == "failed" and "still has no music" in st["step"],
      (remixed, st))
BC.mark_nomusic(b4, "01_one", False)
check("video guard: a real mix of the voices' length passes", BC.mix_matches_voices(b4, "01_one"))
BC.run_job = real_run_job

# --- assets lists unless --do; refused for a shots-flow book --------------------------------------------------------
b5 = book("assets")
started = []
real_start = BC.start_job
BC.start_job = lambda b, **k: started.append(k)
listed = []
sys.modules["flow3d"] = types.SimpleNamespace(assets=lambda b, c, plan_only=False, **k: listed.append(plan_only) or
                                              {"pending": [], "rc": {}})
BC.main(["assets", str(b5)])
check("assets: no --do = the listing (plan_only), no job", listed == [True] and started == [], (listed, started))
try:
    BC.main(["assets", str(b5), "--do"])
    check("assets --do: refused for a shots-flow book", False)
except SystemExit as exc:
    check("assets --do: refused for a shots-flow book, saying why", "shots flow" in str(exc) and started == [], exc)
BC.main(["assets", str(b5), "--do", "--flow", "3d"])
check("assets --do --flow 3d: the job starts", started and started[-1].get("flow3d") == "assets", started)
active.add(BC.unit_name(b5))
try:
    BC.main(["keyframes", str(b5)])
    check("keyframes: refused while the book's job runs", False)
except SystemExit as exc:
    check("keyframes: refused while the book's job runs", "job is running" in str(exc), exc)
active.clear()
blocker = hold(boxlease.LOCK, {"book": "busy_book", "what": "job"}, 30)
try:
    BC.main(["audition", str(b5), "Holmes"])
    check("audition: refused while another job holds the box", False)
except SystemExit as exc:
    check("audition: refused while another job holds the box", "busy_book" in str(exc), exc)
blocker.kill()
blocker.wait()

# --- pace: refused under a running job; the shots plan goes, the moment list stays (re-timed) -----------------------
b6 = book("pace")
BC.set_pace(b6, 1.0)
w = b6 / "work"
(w / "01_one.mp3").write_bytes(b"v")
(w / "01_one.timeline.json").write_text(json.dumps([{"t": 10.0}]))
(w / "final" / "01_one.mp3").write_bytes(b"m")
(w / "score" / "01_one").mkdir(parents=True)
vd = w / "video" / "01_one"
(vd / "anim").mkdir(parents=True)
(vd / "shots").mkdir(parents=True)
(vd / "anim" / "manifest.json").write_text(json.dumps([{"id": "a", "t": 12.5, "dur": 5.0}]))
(vd / "anim" / "a.mp4").write_bytes(b"clip")
(vd / "shots" / "plan_01_one_shots.json").write_text("{}")
active.add(BC.unit_name(b6))
try:
    BC.change_pace(b6, 1.25)
    check("pace: refused while the book's job runs", False)
except SystemExit as exc:
    check("pace: refused while the book's job runs, nothing changed",
          "job is running" in str(exc) and BC.book_pace(b6) == 1.0 and (vd / "shots" / "plan_01_one_shots.json").exists(), exc)
active.clear()


def ffmpeg_run(cmd, **k):
    if cmd and cmd[0] == "ffmpeg":
        Path(cmd[-1]).write_bytes(b"fast")
    return fake_run(cmd, **k)


subprocess.run = ffmpeg_run
started.clear()
BC.change_pace(b6, 1.25)
subprocess.run = fake_run
man = json.loads((vd / "anim" / "manifest.json").read_text())
check("pace: the saved shots plan is deleted", not list((vd / "shots").glob("plan_*.json")))
check("pace: the moment list is kept, re-timed to the new pace", man == [{"id": "a", "t": 10.0, "dur": 4.0}], man)
check("pace: the old clips go", not (vd / "anim" / "a.mp4").exists())
check("pace: the remix job is started (the chapter had music)", started == [{"level": "music", "music_only": True}], started)
BC.start_job = real_start

# --- start_job: a failed systemd-run leaves the old status --------------------------------------------------------
b7 = book("starter")
BC._status_path(b7).write_text(json.dumps({"state": "done", "step": "finished"}))
subprocess.run = lambda cmd, **k: (fake_run(cmd, **k) if cmd[0] != "systemd-run"
                                   else types.SimpleNamespace(returncode=1, stdout="", stderr="Unit exists"))
try:
    BC.start_job(b7)
    check("start_job: fails loudly when systemd-run fails", False)
except SystemExit as exc:
    check("start_job: fails loudly, the old status back", "Unit exists" in str(exc)
          and BC.read_status(b7) == {"state": "done", "step": "finished"}, (exc, BC.read_status(b7)))
cmd_seen = []
subprocess.run = lambda cmd, **k: cmd_seen.append(cmd) or fake_run(cmd, **k)
BC.start_job(b7)
sd = next(c for c in cmd_seen if c[0] == "systemd-run")
check("start_job: tags the job's processes (BOOKCASTER_JOB)", f"--setenv=BOOKCASTER_JOB={BC.unit_name(b7)}" in sd, sd)
check("start_job: a fresh status with empty problems", BC.read_status(b7).get("problems") == [], BC.read_status(b7))
subprocess.run = fake_run

# --- Jellyfin: same size is not the same file ---------------------------------------------------------------------
remote = {}
jellyfin.config = lambda: {"SMB_SHARE": "//x/y", "SMB_USER": "u"}
jellyfin.remote_size = lambda cfg, path: remote.get(path)
puts = []


class FakeSmb:
    def __init__(self, cmds, stdin=None):
        self.cmds, self.buf = cmds, bytearray()
        self.stdin = types.SimpleNamespace(write=self.buf.extend, close=lambda: None)
        self.stdout = types.SimpleNamespace(read=lambda: b"")

    def wait(self):
        tmp = self.cmds.split('"')[1]
        remote[tmp] = len(self.buf)
        puts.append(tmp)
        return 0

    def communicate(self):
        if "rename" in self.cmds:
            parts = self.cmds.split('"')
            tmp, dest = parts[3], parts[5]
            remote[dest] = remote.pop(tmp)
        return ("", None)


jellyfin._smb = lambda cfg, cmds, stdin=None: FakeSmb(cmds, stdin)
src = TMP / "v.mp4"
src.write_bytes(b"A" * 1000)
check("jellyfin: first send uploads", jellyfin.send(src, "Book", "01 One.mp4", log=print) and len(puts) == 1, puts)
check("jellyfin: the same file again is skipped", jellyfin.send(src, "Book", "01 One.mp4", log=print) and len(puts) == 1, puts)
src.write_bytes(b"B" * 1000)                                   # remade: same size, new content
check("jellyfin: a remade video of the SAME size is sent again", jellyfin.send(src, "Book", "01 One.mp4", log=print)
      and len(puts) == 2, puts)
remote["Book\\.01 One.mp4.part"] = 1000                        # a stale .part of another file, same size
src.write_bytes(b"C" * 1000)
jellyfin.send(src, "Book", "01 One.mp4", log=print)
check("jellyfin: a stale same-size .part is not reused", len(puts) == 3, puts)

subprocess.run = REAL_RUN
import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
