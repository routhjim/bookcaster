"""Asset chain scheduler: one per box; owns the chain's work on the GPUs and takes jobs from any number of clients.

    python tools/assets/assetd.py --serve      # started by run_asset.py on first use (detached); exits when idle
    python tools/assets/run_asset.py --status # what each device is doing, the queue, recent jobs

`run_asset.py SPEC MODEL [...]` is a thin client: it sends its command line to this daemon over a Unix socket
(<ASSETS_SCHED_DIR>/sched.sock), prints the job's log lines and exits with the job's code. A job is a chain of
stages, each dispatched the moment its device is free, so stages of different assets overlap:

    preview:  blender_preview (GPU)
    final:    blender_final (GPU: bake, export; then a small sheet) -> [paint] -> ue (an editor: import + stills)
              -> finalize (manifest). The UE stage starts as soon as the glTF exists (Blender writes
              blender_report.json with status "exported"), while Blender still renders its sheet.

Device policy (GPU only: nothing is ever baked or rendered on the CPU; with no GPU slot free a stage waits):
- Blender: up to ASSETS_BLENDER_PER_CARD jobs per XTX, cards without a UE editor first, then the editor's card
  (a bake and the idle editor share 24 GB), then the iGPU (one job, only when its busy % is low); previews try the
  iGPU first. A card is used only when this daemon holds its lock (/tmp/gpu-<bus>.lock, flock -n) and its free
  VRAM (sysfs) covers the job.
- UE: one editor per card at most. The first editor starts on ASSETS_UE_CARDS[0] (else the next card whose lock is
  free) as soon as a final job arrives (not when its glTF is ready); a second editor on another card starts when
  ASSETS_UE_EDITORS=2 and the UE backlog is at least ASSETS_UE_SECOND_AT. Every job imports its own asset in the
  editor that runs it (each asset id has its own content folder; one id never runs twice at once), so two editors
  never write the same package; autosave is off in both.

Locks: every lock is taken with a non-blocking flock and never held while waiting for another. A card lock is held
while a job of ours runs on the card, or while one of our editors lives there (the editor inherits it). The first
editor is the one declared tenancy (/tmp/gpu-<bus>.tenant names it): it quits after ASSETS_UE_IDLE_S without work,
a second editor after ASSETS_UE_IDLE2_S. The project lock (<project>/Saved/AssetServer/project.lock) is held
while any editor of ours is alive, so a --local run never starts a second editor writer beside them.
A client that disconnects cancels its job: queued stages are dropped, a running Blender is stopped.
"""

import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import line as L  # noqa: E402
import run_asset as R  # noqa: E402

SOCK = C.SCHED_DIR / "sched.sock"
CODE_FILES = ("assetd.py", "run_asset.py", "config.py", "line.py")
UE_FILES = ("ue/ue_render.py", "ue/ue_import.py", "ue/init_unreal.py")


def code_hash(files=CODE_FILES):
    h = hashlib.sha1()
    for f in files:
        try:
            h.update((HERE / f).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


def code_stamp(files=CODE_FILES):
    """Cheap change detector for the line's own code (size and mtime of each file)."""
    out = []
    for f in files:
        try:
            st = (HERE / f).stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def proc_start(pid):
    """A process's start time in clock ticks (/proc/<pid>/stat field 22), or None when it is gone; with the pid it
    names one process for good (a pid can be reused, the pair cannot)."""
    try:
        txt = Path(f"/proc/{pid}/stat").read_text()
        f = txt.rsplit(")", 1)[1].split()
        return None if f[0] == "Z" else int(f[19])
    except (OSError, ValueError, IndexError):
        return None


# the line reloads itself onto newer code on disk: running jobs finish, nothing new starts, then it re-executes
RELOAD = os.environ.get("ASSETS_LINE_RELOAD", "1") == "1"
# a borrow command waits at most this long (or BATCH editor jobs) while editor work keeps arriving
BORROW_MAX_WAIT_S = float(os.environ.get("ASSETS_LINE_BORROW_MAX_WAIT_S", "600"))
# a station whose next line is full holds its job only while that line moves; a stuck next line is reported, and the
# hold is relaxed to twice the line length so the station upstream does not idle behind it
HOLD_GRACE_S = float(os.environ.get("ASSETS_LINE_HOLD_GRACE_S", "30"))
STALL_REPORT_S = float(os.environ.get("ASSETS_LINE_STALL_REPORT_S", "60"))
INHIBIT = os.environ.get("ASSETS_LINE_INHIBIT", "1") == "1"      # 0: no sleep inhibitor (dry-run lines in tests)


def now():
    return time.time()


def try_flock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        return None


def lock_holders(path, deep=False):
    """[(pid, cmdline)] holding a flock on path (from /proc/locks); deep: name the processes that still hold it
    when the locker exited."""
    try:
        ino = Path(path).stat().st_ino
        lines = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        f = ln.split()
        if len(f) > 5 and f[1] == "FLOCK" and f[5].endswith(f":{ino}"):
            try:
                cmd = Path(f"/proc/{f[4]}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                cmd = "(exited; held by " + (", ".join(fd_holders(path)) or "?") + ")" if deep else "(exited)"
            out.append((int(f[4]), cmd.strip()[:160]))
    return out


def fd_holders(path):
    """'pid name' of every process with path open (the lock lives on in children that inherited the fd)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if any(os.readlink(x) == path for x in (d / "fd").iterdir()):
                out.append(f"{d.name} {(d / 'comm').read_text().strip()}")
        except OSError:
            continue
    return out


def other_project_editors(project=None):
    """Pids of UE editors on a project (default: the chain's) that this scheduler did not start (two writers
    corrupt a project)."""
    mine = {e.upid for e in getattr(SCHED, "editors", []) if getattr(e, "upid", None)}
    proj = str(project or C.UE_PROJECT)
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if cmd and cmd[0].endswith(b"UnrealEditor") and proj.encode() in cmd and int(d.name) not in mine:
            out.append(int(d.name))
    return out


SCHED = None


def sid_of(pid):
    """Session id of a process (line commands start their own session: sid = the command's pid)."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[3])
    except (OSError, ValueError, IndexError):
        return None


def busy_pct(bus):
    try:
        return int(Path(f"/sys/bus/pci/devices/0000:{bus}:00.0/gpu_busy_percent").read_text())
    except (OSError, ValueError):
        return None


def vram(bus):
    """(used GB, total GB) from sysfs."""
    d = Path(f"/sys/bus/pci/devices/0000:{bus}:00.0")
    try:
        return int((d / "mem_info_vram_used").read_text()) / 2**30, int((d / "mem_info_vram_total").read_text()) / 2**30
    except (OSError, ValueError):
        return None, None


# ---------------------------------------------------------------- devices
class Card:
    """One XTX. fd: our hold on its lock (None = not ours right now)."""

    def __init__(self, bus, role):
        self.bus, self.role, self.fd, self.blender, self.editor = bus, role, None, [], None
        self.lock_path = f"/tmp/gpu-{bus}.lock"
        self.tenant_path = Path(f"/tmp/gpu-{bus}.tenant")
        self.held_since = None
        self.outside_since = None         # since when an outside user holds the lock (we wanted it)
        self.released_until = 0.0         # --release-card: not retaken before this time
        self.draining = False             # --release-card: no new work; let go when ours ends
        self.last_used = 0.0
        self.lent = False                 # given away with --release-card: an outside holder is expected
        self.spares = []                  # warm Blender processes (bl_spare.py) waiting for a job on this card
        self.cmd_busy = None              # a line command runs here and holds the card's lock itself

    def stop_spares(self):
        for sp in self.spares:
            if sp.proc.poll() is None and not sp.taken:
                (sp.dir / "quit").touch()
                if now() - sp.quit_t > 5 if sp.quit_t else False:
                    sp.proc.kill()
                sp.quit_t = sp.quit_t or now()

    def ready_spare(self):
        for sp in self.spares:
            if not sp.taken and sp.proc.poll() is None and (sp.dir / "ready").exists():
                return sp
        return None

    def hold(self):
        if self.fd is None:
            if self.draining or now() < self.released_until or self.cmd_busy is not None:
                return False
            self.fd = try_flock(self.lock_path)
            if self.fd is not None:
                self.held_since, self.outside_since, self.lent = now(), None, False
            elif self.outside_since is None:
                self.outside_since = now()
        return self.fd is not None

    def release_if_unused(self, linger=0.0):
        if self.blender or self.editor is not None:
            self.last_used = now()
            return
        if self.fd is not None and (self.draining or now() - self.last_used >= linger):
            self.stop_spares()                         # they inherited the lock: it frees when they are gone
            if any(sp.proc.poll() is None for sp in self.spares):
                return
            self.spares = []
            os.close(self.fd)
            self.fd, self.held_since = None, None
            self.tenant_path.unlink(missing_ok=True)
            if self.draining:
                self.draining = False
                self.released_until = now() + C.RELEASE_GRACE_S
                self.lent = True
                print(f"[sched] card {self.bus} released (--release-card); not retaken for "
                      f"{C.RELEASE_GRACE_S:.0f}s, then when its lock is free", flush=True)

    def free_gb(self):
        u, t = vram(self.bus)
        return (t - u) if u is not None else 0.0


class Spare:
    N = 0

    def __init__(self, card):
        Spare.N += 1
        self.card, self.taken, self.quit_t = card, False, None
        self.dir = C.SCHED_DIR / "spares" / f"{card.bus}_{os.getpid()}_{Spare.N}"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.log = self.dir / "spare.log"
        with open(self.log, "w") as f:
            self.proc = subprocess.Popen([str(c) for c in (C.BLENDER, "-b", "--factory-startup", "-t", C.BLENDER_THREADS,
                                                           "--python", HERE / "bl_spare.py", "--", self.dir, card.bus,
                                                           C.HIPRT)],
                                         stdout=f, stderr=subprocess.STDOUT, pass_fds=(card.fd,),
                                         start_new_session=True)


class Igpu:
    def __init__(self):
        self.bus, self.fd, self.blender = C.IGPU_BUS, None, []
        self.lock_path = f"/tmp/gpu-{C.IGPU_LOCK_NAME}.lock"

    def hold(self):
        if self.fd is None:
            self.fd = try_flock(self.lock_path)
        return self.fd is not None

    def release_if_unused(self):
        if self.fd is not None and not self.blender:
            os.close(self.fd)
            self.fd = None


def proj_dir(project):
    """<project>/Saved/AssetServer: the queue folders of the scheduler's editor on that project."""
    d = Path(project).parent / "Saved" / "AssetServer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def asset_project(project):
    return Path(project).resolve() == Path(C.UE_PROJECT).resolve()


class Editor:
    def __init__(self, card, primary, project=None):
        self.card, self.primary = card, primary
        self.project = str(project or C.UE_PROJECT)
        self.dir = proj_dir(self.project) / f"ed-{card.bus}"
        self.qdir = self.dir / "queue"
        self.state = "starting"          # starting / ready / busy / quitting / dead
        self.t_start, self.t_ready, self.idle_since = now(), None, None
        self.proc, self.pid, self.log, self.job, self.n_jobs = None, None, None, None, 0
        self.code = code_hash(UE_FILES)
        self.error = None

    def alive(self):
        if self.proc is not None:
            return self.proc.poll() is None
        return False

    def quit(self):
        if self.state not in ("dead", "quitting"):
            self.state = "quitting"
            self.quit_t = now()
            self.qdir.mkdir(parents=True, exist_ok=True)
            (self.qdir / "quit").touch()


# ---------------------------------------------------------------- jobs
class Stage:
    def __init__(self, job, name, kind, deps=(), ready=None):
        self.job, self.name, self.kind, self.deps = job, name, kind, list(deps)
        self.ready_fn = ready
        self.state = "wait"               # wait / ready / run / done / fail / cancel
        self.t_ready = self.t_start = self.t_end = None
        self.device = self.device_class = None
        self.info = {}

    def deps_done(self):
        if self.ready_fn is not None:
            return self.ready_fn()
        return all(d.state == "done" for d in self.deps)

    def record(self):
        return {"name": self.name, "state": self.state, "device": self.device, "device_class": self.device_class,
                "wait_s": round((self.t_start or self.t_end or now()) - self.t_ready, 1) if self.t_ready else None,
                "run_s": round((self.t_end or now()) - self.t_start, 1) if self.t_start else None,
                "start": self.t_start, "end": self.t_end, **self.info}


class Job:
    N = 0

    def __init__(self, args, spec, conn, client_pid, argv):
        Job.N += 1
        self.n, self.args, self.spec, self.conn, self.client_pid, self.argv = Job.N, args, spec, conn, client_pid, argv
        self.id = spec["id"]
        self.adir = C.LIBRARY / spec["class"] / spec["id"]
        self.mode = "preview" if args.preview else "final"
        self.t_submit = now()
        self.state = "queued"
        self.cancelled = False
        self.exported = False             # the glTF exists (Blender may still render its sheet)
        self.blend = None                 # blender report
        self.result = None
        self.stages = []
        self.times = {}
        self.send_lock = threading.Lock()
        self.pending = None               # (state, result) once a stage failed
        self.entry = None                 # the work-list entry (line.py), saved on every state change
        self.seq = None
        self.logf = None

    def log(self, msg):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
        if self.logf is not None:
            try:
                with open(self.logf, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        if self.conn is None:
            return
        try:
            with self.send_lock:
                self.conn.sendall((json.dumps({"log": line}) + "\n").encode())
        except OSError:
            pass                          # the client's reader thread notices the disconnect

    def blender_stage(self):
        return next((s for s in self.stages if s.kind == "blender"), None)


# ---------------------------------------------------------------- the scheduler
class Sched:
    def __init__(self):
        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        self.ue_bus, self.bl_bus = C.UE_CARD, C.BLENDER_CARD
        self.cards = {self.ue_bus: Card(self.ue_bus, "ue"), self.bl_bus: Card(self.bl_bus, "blender")}
        self.spills = 0                   # Blender bakes run on the UE card (logged decisions)
        self.released = None              # None = asset mode; else {"reason", "manual", "t", "done_t", ...}
        self.last_release = None
        self.warm_wanted = False
        self.outside, self.free_since, self.t_probe = {}, {}, 0.0
        try:                              # a release asked of an earlier scheduler still stands
            r = json.loads((C.SCHED_DIR / "released.json").read_text())
            if r.get("manual") and now() - r["t"] < 24 * 3600:
                self.released = dict(r, done_t=None)
        except (OSError, ValueError, KeyError):
            pass
        self.igpu = Igpu()
        self.editors = []
        self.jobs = []                    # live jobs, submit order
        self.recent = collections.deque(maxlen=40)
        self.t0 = now()
        self.code = code_hash()
        self.code_stamp, self.t_code, self.bad_code = code_stamp(), 0.0, None
        self.reload = None                # {"t", "code"}: newer code on disk; running jobs finish, then re-exec
        self.drain = False                # exit when idle (asked with the drain command)
        self.srv = self.lock_fd = None    # the listening socket and the daemon lock (kept across a reload)
        self.engine_turn = "editor"       # the engine card: "editor" work, or a "borrow" command's turn
        self.editor_runs = 0              # editor jobs started since the last borrow
        self.why = {}                     # line -> [reasons] its ready entries were not started (last dispatch)
        self.idle_since = {}              # station -> since when it runs nothing
        self.stall_said = {}
        self.project_fd = None
        self.last_busy = now()
        self.util = collections.deque(maxlen=7200)        # (t, {bus: busy%}) at 2 Hz
        self.est = self.load_est()
        self.mannequin = None
        self.stopping = False
        # the line (line.py): work list on disk, batches, power / hardware guard
        self.wl = L.WorkList(C.SCHED_DIR / "line")
        self.batcher = L.Batcher()
        self.guard = L.PowerGuard(cards=[self.ue_bus, self.bl_bus])
        self.t_guard, self.verdict, self.said_why = 0.0, {"cards": True, "pause": False, "why": []}, None
        self.state_written = False
        self.child_sids = set()
        self.inhibit = None
        self.window = None                # the video window: {"t", "n"} while open
        self.peak_load = 0.0
        self.kind_cost = collections.defaultdict(list)    # kind -> [(seconds, Wh used)]

    # ------------------------------------------------------------ estimates (seconds), for waits and --status
    def load_est(self):
        p = C.SCHED_DIR / "estimates.json"
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {}

    def est_get(self, key, default):
        return self.est.get(key, default)

    def est_put(self, key, v):
        old = self.est.get(key)
        self.est[key] = round(v if old is None else 0.7 * old + 0.3 * v, 2)
        try:
            tmp = C.SCHED_DIR / f".estimates.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(self.est, indent=1))
            tmp.replace(C.SCHED_DIR / "estimates.json")
        except OSError:
            pass

    # ------------------------------------------------------------ submissions
    # ------------------------------------------------------------ the line: entries, stations, admission
    def make_job(self, e, conn=None):
        """A Job from a work-list entry (a new submission, or one reloaded after a restart)."""
        plain = argparse.Namespace(preview=False, paint=False, skip_render=False, rebuild_stage=False, device=None,
                                   ue_timeout=float(e.get("timeout_s") or C.UE_JOB_TIMEOUT_S), model=None,
                                   ue_only=False)
        if e["type"] == "asset":
            args = R.build_parser().parse_args(e["argv"])
            args.spec, args.model = e["spec"], e["model"]
            spec = json.loads(Path(e["spec"]).read_text())
            job = Job(args, spec, conn, e.get("client_pid"), e["argv"])
        elif e["type"] == "ue_job":
            jd = e["ue"]
            job = Job(plain, {"id": e["id"], "class": "ue_job"}, conn, e.get("client_pid"), [])
            job.mode, job.ue = "ue_job", jd
            job.project = str(Path(jd["project"]).resolve())
            job.adir = Path(jd.get("log_dir") or (C.SCHED_DIR / "jobs" / f"{e['id']}_{e['seq']}"))
            job.adir.mkdir(parents=True, exist_ok=True)
        else:                                               # "cmd": run this command at a station
            job = Job(plain, {"id": e["id"], "class": "cmd"}, conn, e.get("client_pid"), [])
            job.mode = "cmd"
            job.adir = C.SCHED_DIR / "line" / "jobs" / f"{e['seq']:06d}_{e['id']}"
            job.adir.mkdir(parents=True, exist_ok=True)
        job.entry, job.seq = e, e["seq"]
        job.t_submit = e.get("submitted") or now()
        job.logf = C.SCHED_DIR / "line" / "logs" / f"{e['seq']:06d}.log"
        job.logf.parent.mkdir(parents=True, exist_ok=True)
        return job

    def first_station(self, e):
        if e["type"] == "asset":
            if "--ue-only" in e["argv"]:
                return f"card{self.ue_bus}"
            return f"card{self.bl_bus}"
        if e["type"] == "ue_job":
            return f"card{self.ue_bus}"
        if e["station"] == f"card{self.ue_bus}":
            return self.borrow_line()                      # borrows wait in their own line beside the editor's
        return e["station"]

    def borrow_line(self):
        return f"card{self.ue_bus}:borrow"

    def line_of(self, st):
        """The waiting line a stage queues in: its station, except that commands borrowing the engine card have their
        own line beside the editor's work, so a line full of borrows never holds a bake whose next stage is the editor
        (and the editor's short imports and stills never fill the borrows' line)."""
        b = self.station(st)
        if b == f"card{self.ue_bus}" and st.kind == "cmd":
            return self.borrow_line()
        return b

    def add_entry(self, e, conn=None, resume=False):
        """Admit an entry to the line (or say the line is full); returns (job, None) or (None, why)."""
        if e.get("type") == "cmd":
            e["station"] = STATION_ALIASES.get(str(e.get("station")), e.get("station"))
        with self.lock:
            if not resume:
                dup = self.duplicate_of(e)
                if dup is not None:
                    return None, (f"already queued as entry {dup.seq} ({dup.id}, {self.job_where(dup)}); "
                                  f"not queued again")
                why = L.line_full(self.heading(), self.first_station(e))
                if why:
                    return None, why
                e.update(seq=self.wl.next_seq(), state="queued", submitted=now(), stages={})
                self.wl.save(e)
            job = self.make_job(e, conn)
            self.submit(job)
            self.persist(job)
            if resume:
                self.restore(job)
            return job, None

    def duplicate_of(self, e):
        """A live job this submission repeats (two submitters for one job): a command with the same id at the same
        station, an asset with the same id and kind, an engine job with the same id and the same job file content."""
        for j in self.jobs:
            x = j.entry
            if (x is None or j.cancelled or j.state in ("done", "failed", "cancelled") or x.get("type") != e.get("type")
                    or x.get("id") != e.get("id")):
                continue
            if e["type"] == "cmd" and x.get("station") == e.get("station"):
                return j
            if e["type"] == "asset" and x.get("kind") == e.get("kind"):
                return j
            if e["type"] == "ue_job" and x.get("ue") == e.get("ue"):
                return j
        return None

    def job_where(self, j):
        run = [st.name for st in j.stages if st.state == "run"]
        return f"running {', '.join(run)}" if run else "waiting"

    def restore(self, job):
        """After a restart: a command that ended while no line watched it is finished from its record (never run
        again); one still running is adopted (watched to its end, not killed and not started twice)."""
        for st in job.stages:
            if job.mode == "cmd" and st.state in ("done", "fail"):
                print(f"[line] entry {job.seq} {job.id}: {st.name} ended while the line was down "
                      f"(rc {st.info.get('rc')}); finished from its record, not run again", flush=True)
                self.after_stage(st)
                return
        for name, a in (getattr(job, "adopt", None) or {}).items():
            st = next((x for x in job.stages if x.name == name), None)
            if st is not None and st.state == "wait":
                st.t_ready = now()
                self.launch(st, self.run_cmd, a)

    def persist(self, job):
        e = job.entry
        if e is None:
            return
        e["stages"] = {s.name: s.state for s in job.stages}
        e["pids"] = {s.name: s.info["pid"] for s in job.stages if s.state == "run" and s.info.get("pid")}
        e["pid_start"] = {s.name: s.info.get("pid_start") for s in job.stages if s.state == "run" and s.info.get("pid")}
        rcf = {s.name: s.info["rc_file"] for s in job.stages if s.info.get("rc_file")}
        if rcf:
            e["rc_files"] = rcf
        e["state"] = {"queued": "queued", "running": "running"}.get(job.state, job.state)
        e["n"] = job.n
        try:
            self.wl.save(e)
        except OSError as ex:
            print(f"[line] could not save entry {e.get('seq')}: {ex}", flush=True)

    def station(self, st):
        """The station a stage runs at (None: CPU bookkeeping such as finalize, which runs at once)."""
        if st.kind == "blender":
            return f"card{self.bl_bus}"
        if st.kind == "ue":
            return f"card{self.ue_bus}"
        if st.kind == "paint":
            return C.IMAGE_STATION
        if st.kind == "cmd":
            return st.job.entry["station"]
        return None

    def skind(self, st):
        """The batch kind of a stage: a station runs a batch of one kind before it switches (no reloads)."""
        if st.kind == "blender":
            return "blender"
        if st.kind == "ue":
            return "ue_script" if st.name == "ue_script" else "ue_asset"
        if st.kind == "cmd":
            return st.job.entry.get("kind") or "cmd"
        return st.kind

    def heading(self):
        """station -> jobs in line for it: a job whose next station stage waits there, or a job running elsewhere
        whose next station it is (so the short line also holds back the station upstream)."""
        out = collections.Counter()
        for j in self.jobs:
            if j.cancelled or j.state in ("done", "failed", "cancelled"):
                continue
            todo = [(st, self.line_of(st)) for st in j.stages if self.station(st) and st.state in ("wait", "ready", "run")]
            if not todo:
                continue
            st0, b0 = todo[0]
            if st0.state != "run":
                out[b0] += 1
                continue
            nxt = next((b for st, b in todo[1:] if b != b0 and st.state != "run"), None)
            if nxt:
                out[nxt] += 1
        return out

    def next_station(self, st):
        """The next line this job joins after st (None: it ends here or stays at this station)."""
        sts = st.job.stages
        b0 = self.line_of(st)
        for x in sts[sts.index(st) + 1:]:
            b = self.line_of(x)
            if b and b != b0:
                return b
        return None

    def video_left(self):
        return any(st.kind == "cmd" and st.job.entry["station"] == "video" and st.state in ("wait", "ready", "run")
                   for j in self.jobs for st in j.stages)

    def submit(self, job):
        a = job.args
        s = []
        if job.mode == "cmd":
            s.append(Stage(job, job.entry.get("kind") or "cmd", "cmd"))
        elif job.mode == "ue_job":
            s.append(Stage(job, "ue_script", "ue"))
        elif getattr(a, "ue_only", False):
            # baked and exported before: the engine stage alone (import + stills), from the asset's Blender report
            job.blend = json.loads((job.adir / "blender_report.json").read_text())
            job.exported = True
            ue = Stage(job, "ue", "ue")
            s += [ue, Stage(job, "finalize", "cpu", [ue])]
        elif job.mode == "preview":
            s.append(Stage(job, "blender_preview", "blender"))
        else:
            bl = Stage(job, "blender_final", "blender")
            s.append(bl)
            prev = None
            if a.paint:
                prev = Stage(job, "paint", "paint", [bl])
                s.append(prev)
            if a.skip_render:
                ue = Stage(job, "ue_import", "ue", ready=lambda: (job.exported or bl.state == "done") and
                           (prev is None or prev.state == "done"))
            else:
                ue = Stage(job, "ue", "ue", ready=lambda: (job.exported or bl.state == "done") and
                           (prev is None or prev.state == "done"))
            s.append(ue)
            s.append(Stage(job, "finalize", "cpu", [bl, ue] + ([prev] if prev else [])))
        job.stages = s
        # after a restart: a finished bake is kept (its report says ok); everything after it runs again
        was = (job.entry or {}).get("stages") or {}
        ended = (job.entry or {}).pop("ended", None) or {}
        job.adopt = (job.entry or {}).pop("adopt", None) or {}
        if job.mode == "cmd":
            for st in s:
                if st.name in ended or was.get(st.name) in ("done", "fail"):
                    rc = ended.get(st.name, (job.entry or {}).get("rc", 0 if was.get(st.name) == "done" else 1))
                    st.state, st.info["rc"] = ("done" if rc == 0 else "fail"), rc
                    if rc != 0:
                        st.info["error"] = f"command failed (rc {rc}), recorded while the line was down"
                    job.result = {"rc": rc, "log": (job.entry or {}).get("log")}
        bl = job.blender_stage()
        if job.mode == "final" and bl is not None and was.get(bl.name) == "done":
            try:
                rep = json.loads((job.adir / "blender_report.json").read_text())
                if rep.get("status") == "ok":
                    bl.state, job.blend, job.exported = "done", rep, True
            except (OSError, ValueError):
                pass
        with self.lock:
            self.jobs.append(job)
            self.last_busy = now()
            self.wake.notify_all()

    def cancel(self, job, why):
        with self.lock:
            if job.state in ("done", "failed", "cancelled"):
                return
            job.cancelled = True
            job.cancel_why = why
            for st in job.stages:
                if st.state in ("wait", "ready"):
                    st.state = "cancel"
            self.wake.notify_all()
        print(f"[sched] job {job.n} {job.id} cancelled: {why}", flush=True)

    # ------------------------------------------------------------ main loop
    def loop(self):
        while True:
            with self.lock:
                self.wake.wait(0.25)
                try:
                    self.tick()
                except Exception:
                    traceback.print_exc()
                if self.should_exit():
                    return

    # ------------------------------------------------------------ asset mode / released (video runs)
    def outside_holders(self, c):
        """Pids other than ours holding a card's lock (our editor inherits our lock: /proc/locks shows our pid)."""
        return [(p, cmd) for p, cmd in lock_holders(c.lock_path)
                if p != os.getpid() and p not in self.child_sids and sid_of(p) not in self.child_sids]

    def card_work_queued(self):
        for j in self.jobs:
            for s in j.stages:
                if s.state not in ("wait", "ready"):
                    continue
                if s.kind == "ue" or (s.kind == "blender" and j.mode == "final"):
                    return True
                # a command waiting on a card station is card work too: without this, a release for a video
                # window is never taken back while only such commands wait, and they wait for the reclaim
                if s.kind == "cmd" and str(j.entry.get("station", "")).startswith("card"):
                    return True
        return False

    def release(self, reason, manual, card=None):
        if self.released:
            return
        self.released = {"reason": reason, "manual": manual, "t": now(), "done_t": None, "card": card}
        print(f"[sched] releasing both cards: {reason}", flush=True)
        try:
            (C.SCHED_DIR / "released.json").write_text(json.dumps(self.released))
        except OSError:
            pass
        self.wake.notify_all()

    def reclaim(self, why):
        if not self.released:
            return
        r = self.released
        self.last_release = dict(r, reclaim_t=now(), reclaim_why=why)
        self.released = None
        self.warm_wanted = True                            # load the editor now, even before UE work arrives
        self.outside, self.free_since = {}, {}             # fresh windows: no stale "held for N s" after a reclaim
        for c in self.cards.values():
            c.outside_since = None
        (C.SCHED_DIR / "released.json").unlink(missing_ok=True)
        print(f"[sched] reclaiming the cards: {why}", flush=True)

    def update_mode(self, t):
        if t - self.t_probe >= 1.0:
            self.t_probe = t
            for c in self.cards.values():
                if c.fd is None and self.outside_holders(c):
                    self.outside[c.bus] = self.outside.get(c.bus) or t
                else:
                    self.outside.pop(c.bus, None)
        if not self.released:
            ours = any(c.fd is not None for c in self.cards.values())
            for bus, since in self.outside.items():
                if self.cards[bus].lent:
                    continue
                lim = C.OUTSIDE_RELEASE_S if self.cards[bus].role == "ue" else C.OUTSIDE_RELEASE_BL_S
                if self.cards[bus].role != "ue" and t - since >= lim:
                    # a job working on the Blender card (busy) is not a video run waiting for the UE card: no
                    # release, bakes spill instead; a holder that leaves the card idle is waiting for the other
                    busy = self.load_between(bus, since, t).get("busy_mean")
                    if busy is None or busy >= 10:
                        continue
                if t - since >= lim and (ours or self.card_work_queued()):
                    who = "; ".join(f"{p} {cmd[:60]}" for p, cmd in self.outside_holders(self.cards[bus]))
                    self.release(f"an outside job holds card {bus} ({who or '?'}) for {t - since:.0f}s", False, bus)
                    break
            return
        r = self.released
        if r["done_t"] is None and not any(e.state != "dead" for e in self.editors) and all(
                c.fd is None for c in self.cards.values()):
            r["done_t"] = t
            r["release_s"] = round(t - r["t"], 1)
            r["vram_used_gb"] = {b: round(vram(b)[0] or 0, 1) for b in self.cards}
            print(f"[sched] both cards released in {r['release_s']}s; VRAM used {r['vram_used_gb']}", flush=True)
        if r["done_t"] is None or not self.card_work_queued() or self.video_left():
            return
        for b in self.cards:
            if b in self.outside:
                self.free_since.pop(b, None)
            else:
                self.free_since.setdefault(b, t)
        free_for = min((t - self.free_since.get(b, t)) for b in self.cards)
        grace = C.RELEASE_GRACE_S if r["manual"] else 0.0
        if t - r["t"] >= grace and free_for >= C.RECLAIM_FREE_S and all(b in self.free_since for b in self.cards):
            self.reclaim(f"both card locks free for {free_for:.0f}s and asset work is queued")

    def tick(self):
        t = now()
        self.reap_editors()
        self.update_mode(t)
        self.advance(t)
        self.check_guard(t)
        self.check_code(t)
        self.dispatch()
        self.report_stalls(t)
        self.keep_awake()
        self.manage_editors()
        # the Blender card stays ours while Blender work is queued, and BLENDER_LINGER_S after it (no thrash)
        bl_queued = any(st.kind == "blender" and st.state in ("wait", "ready") for j in self.jobs for st in j.stages)
        for c in self.cards.values():
            if c.role == "blender" and bl_queued and c.fd is not None and not c.draining and not self.reload:
                c.last_used = t
            self.keep_spares(c, t)
            c.release_if_unused(C.BLENDER_LINGER_S if c.role == "blender" and not self.released else 0.0)
        self.igpu.release_if_unused()
        if self.jobs or any(e.state != "dead" for e in self.editors):
            self.last_busy = t
        self.write_tenancy()
        self.maybe_reload(t)

    def advance(self, t):
        """Finish cancelled jobs; a stage whose dependencies are done becomes ready."""
        for job in list(self.jobs):
            if job.cancelled and not any(st.state == "run" for st in job.stages):
                self.finish_job(job, "cancelled", {"rc": 130, "error": "cancelled (client gone)"})
                continue
            # the same asset id never runs twice at once: later jobs of an id wait for the earlier one
            if any(j.id == job.id and j.n < job.n for j in self.jobs):
                continue
            if job.state == "queued":
                job.state = "running"
            for st in job.stages:
                if st.state == "wait" and st.deps_done():
                    st.state, st.t_ready = "ready", t
                if any(d.state in ("fail", "cancel") for d in st.deps) and st.state in ("wait", "ready"):
                    st.state = "cancel"

    # ------------------------------------------------------------ the line: dispatch, guard, wake lock
    def station_names(self):
        return ["igpu", f"card{self.bl_bus}", f"card{self.ue_bus}", "video"]

    def line_moving(self, line, t, running_lines):
        """A line whose station works through it now (or was idle only a moment, or its editor is loading)."""
        if line in running_lines:
            return True
        if line == f"card{self.ue_bus}" and any(e.state == "starting" for e in self.editors):
            return True
        since = self.idle_since.get(line.split(":")[0])
        return since is not None and t - since < HOLD_GRACE_S

    def dispatch(self):
        """Each idle station takes its next job: same kind as its current batch while one waits (up to the batch
        size), else the oldest. A job starts only when the guard allows it (battery floor; card jobs: charger,
        both cards and ASPM, the hold file), no reload is pending, and the line at its next station is not full
        (while that line moves; a stuck next line holds at twice its length, so it never idles this station). Card
        stations hold new work while a video window is wanted. Every hold is recorded in self.why (--status)."""
        t = now()
        ready = [st for j in self.jobs if not j.cancelled for st in j.stages if st.state == "ready"]
        for st in ready:
            if self.station(st) is None:
                self.launch(st, self.run_finalize, None)
        running = [st for j in self.jobs for st in j.stages if st.state == "run" and self.station(st)]
        busy = {self.station(st) for st in running}
        running_lines = {self.line_of(st) for st in running}
        for b in self.station_names():
            if b in busy:
                self.idle_since.pop(b, None)
            else:
                self.idle_since.setdefault(b, t)
        why = collections.defaultdict(list)

        def hold(st, text):
            why[self.line_of(st)].append(text)

        v, head = self.verdict, self.heading()
        ue_st = f"card{self.ue_bus}"
        cards = (ue_st, f"card{self.bl_bus}")
        video_wanted = self.video_left()
        if video_wanted and not self.released and not (busy & set(cards)) and not self.reload:
            n = sum(1 for j in self.jobs if j.mode == "cmd" and j.entry["station"] == "video" and not j.cancelled)
            self.window = {"t": now(), "n": n}
            self.release(f"video window: {n} clip job(s) waiting", False)
        self.engine_turn_update(ready, t)
        by = collections.defaultdict(list)
        for st in ready:
            b = self.station(st)
            if b is None or b in busy:
                continue
            if self.reload:
                hold(st, "reloading onto newer code: running jobs finish first, nothing new starts")
                continue
            if v["pause"] or (b in cards and not v["cards"]):
                hold(st, "guard: " + "; ".join(v["why"]))
                continue
            if b == "video":
                if not (self.released and self.released.get("done_t")):
                    hold(st, "video window: waiting for both cards to be released")
                    continue
            elif b in cards and (video_wanted or self.released):
                hold(st, "video window: card work waits until the clips are done" if video_wanted else
                     f"cards released ({(self.released or {}).get('reason')})")
                continue
            if b == ue_st:
                if st.kind == "cmd":
                    ok, text = self.lend_engine_card()
                    if not ok:
                        hold(st, text)
                        continue
                elif st.kind == "ue" and self.engine_turn == "borrow":
                    hold(st, "a borrow command's turn on the engine card (the editor unloads for it)")
                    continue
            nb = self.next_station(st)
            if nb:
                lim = L.LINE_MAX if self.line_moving(nb, t, running_lines) else 2 * L.LINE_MAX
                if head.get(nb, 0) >= lim:
                    hold(st, f"held for the next line: {nb} is full ({head.get(nb, 0)}/{lim})")
                    continue
            by[b].append((st.job.seq or 0, self.skind(st), st))
        for b, cand in by.items():
            prev = dict(self.batcher.cur)
            st = self.batcher.pick(b, cand)
            if st is None:
                continue
            if st.kind == "blender":
                pl = self.place_blender(st)
                ok = bool(pl) and self.launch(st, self.run_blender, pl)
                if not pl:
                    hold(st, self.blender_wait_why())
            elif st.kind == "ue":
                ed = self.place_ue(st)
                ok = bool(ed) and self.launch(st, self.run_ue, ed)
                if not ed:
                    hold(st, "waiting for the editor: " + self.editor_brief())
                elif b == ue_st:
                    self.editor_runs += 1
            else:
                card = self.cards.get(b[4:]) if b.startswith("card") else None
                if card is not None:
                    card.cmd_busy = st                      # at once: no editor or bake is placed on it meanwhile
                ok = self.launch(st, self.run_paint if st.kind == "paint" else self.run_cmd, None)
                if ok and b == ue_st:
                    self.engine_turn, self.editor_runs = "editor", 0
            if not ok:
                self.batcher.cur = prev                     # the station could not take it yet: no batch count
        self.why = dict(why)

    def engine_turn_update(self, ready, t):
        """Whose turn the engine card is: the editor's own work (imports, stills, scripts: seconds each) or a borrow
        command (image-to-3D: minutes). A borrow takes the card when no editor work can start, or once the editor
        has run BATCH jobs (or the borrow waited BORROW_MAX_WAIT_S) while it waited; after a borrow, the editor's
        waiting work goes first. A turn given to a borrow stands until it starts (no unload / reload ping-pong)."""
        ue_st = f"card{self.ue_bus}"
        borrows = [st for st in ready if st.kind == "cmd" and self.station(st) == ue_st]
        if not borrows:
            self.engine_turn, self.editor_runs = "editor", 0
            return
        if self.engine_turn == "borrow":
            return
        live = any(s.kind == "ue" and s.state in ("ready", "run") for j in self.jobs if not j.cancelled
                   for s in j.stages)
        oldest = max(t - (st.t_ready or t) for st in borrows)
        if not live or self.editor_runs >= L.BATCH or oldest >= BORROW_MAX_WAIT_S:
            if live:
                print(f"[line] {time.strftime('%H:%M:%S')} engine card: a borrow's turn after {self.editor_runs} "
                      f"editor job(s), oldest borrow waited {oldest:.0f}s", flush=True)
            self.engine_turn = "borrow"

    def lend_engine_card(self):
        """(ok, why): a command may borrow the engine card on its turn (engine_turn_update): nothing of the
        editor's runs, no Blender job spilled onto that card, and the resident editor has unloaded (an idle one is
        asked to quit here). While the command runs, the card's cmd_busy keeps the editor from loading; the next
        engine job loads it again afterwards."""
        c = self.cards[self.ue_bus]
        if c.blender:
            return False, "a Blender job runs on the engine card"
        if any(s.kind == "ue" and s.state == "run" for j in self.jobs if not j.cancelled for s in j.stages):
            return False, "editor work runs on the engine card"
        if self.engine_turn != "borrow":
            n = sum(1 for j in self.jobs if not j.cancelled for s in j.stages if s.kind == "ue" and s.state == "ready")
            return False, (f"editor work goes first ({n} ready; a borrow's turn after {L.BATCH} editor jobs or "
                           f"{BORROW_MAX_WAIT_S:.0f}s)")
        live = [e for e in self.editors if e.state != "dead" and e.card is c]
        for e in live:
            if e.state == "ready":
                print(f"[sched] editor on {c.bus}: unloading so a line command can borrow the card", flush=True)
                e.quit()
        self.warm_wanted = False                            # no warm reload until engine work arrives
        if live:
            return False, "waiting for the editor to unload (" + ", ".join(
                f"pid {getattr(e, 'pid', '?')} {e.state}" for e in live) + ")"
        return True, None

    def blender_wait_why(self):
        c = self.cards[self.bl_bus]
        if self.released:
            return "cards released"
        if c.draining:
            return f"card {c.bus} is being released (--release-card)"
        if c.cmd_busy is not None:
            return f"card {c.bus} is handed to a command"
        if c.fd is None:
            who = "; ".join(f"{p} {cmd[:50]}" for p, cmd in lock_holders(c.lock_path))
            return f"card {c.bus} lock not ours" + (f" (held by {who})" if who else "")
        return f"card {c.bus}: free VRAM {c.free_gb():.1f} GB below what the bake needs"

    def editor_brief(self):
        live = [e for e in self.editors if e.state != "dead"]
        if not live:
            c = self.cards[self.ue_bus]
            return ("none loaded" + (" (the card is lent to a command)" if c.cmd_busy is not None else "")
                    + (" (reload pending)" if self.reload else ""))
        return ", ".join(f"{e.state} on {e.card.bus}" + (f" for {Path(e.project).stem}" if e.project else "")
                         for e in live)

    def report_stalls(self, t):
        """A station idle STALL_REPORT_S with ready work is logged with the reasons (again when they change, or
        every ten minutes)."""
        for line, reasons in self.why.items():
            since = self.idle_since.get(line.split(":")[0])
            if since is None or t - since < STALL_REPORT_S:
                continue
            text = "; ".join(sorted(set(reasons)))
            said_t, said = self.stall_said.get(line, (0.0, None))
            if said == text and t - said_t < 600:
                continue
            self.stall_said[line] = (t, text)
            print(f"[line] {time.strftime('%H:%M:%S')} {line} idle {t - since:.0f}s with ready work: {text}",
                  flush=True)

    # ------------------------------------------------------------ reload onto newer code
    def check_code(self, t):
        """Newer line code on disk: finish what runs, start nothing new, then re-execute (maybe_reload)."""
        if not RELOAD or t - self.t_code < 2.0:
            return
        self.t_code = t
        stamp = code_stamp()
        if stamp == self.code_stamp:
            return
        self.code_stamp = stamp
        h = code_hash()
        if h == self.code:
            if self.reload:
                print("[line] the code on disk is the running code again: reload called off", flush=True)
                self.reload = None
            return
        if h == self.bad_code:
            return
        n = sum(1 for j in self.jobs for st in j.stages if st.state == "run")
        if self.reload is None:
            print(f"[line] {time.strftime('%H:%M:%S')} newer code on disk ({self.code} -> {h}): {n} running job(s) "
                  f"finish, nothing new starts, then the line reloads and resumes its list", flush=True)
            self.reload = {"t": t, "code": h}
        else:
            self.reload["code"] = h

    def maybe_reload(self, t):
        if not self.reload or any(st.state == "run" for j in self.jobs for st in j.stages):
            return
        live = [e for e in self.editors if e.state != "dead"]
        for e in live:
            if e.state in ("ready", "starting"):
                e.quit()
            elif e.state == "quitting" and t - (getattr(e, "quit_t", None) or t) > 60 and e.proc is not None:
                e.proc.terminate()
        for c in self.cards.values():
            c.stop_spares()
        if live or any(sp.proc.poll() is None for c in self.cards.values() for sp in c.spares):
            return
        try:
            r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import assetd",
                                str(HERE)], cwd=str(HERE), capture_output=True, text=True, timeout=120)
            bad = r.returncode != 0 and (r.stderr.strip().splitlines() or ["?"])[-1]
        except (OSError, subprocess.TimeoutExpired) as ex:
            bad = str(ex)
        if bad:
            print(f"[line] newer code {self.reload['code']} does not import ({bad}); staying on {self.code} until it "
                  f"changes again", flush=True)
            self.bad_code, self.reload = self.reload["code"], None
            return
        self.exec_reload(t)

    def exec_reload(self, t):
        """Re-execute this process on the code on disk. The pid, the daemon lock and the listening socket are kept
        (clients that connect meanwhile wait in its backlog); the new code resumes the work list from disk."""
        if self.srv is None or self.lock_fd is None:
            return
        print(f"[line] {time.strftime('%H:%M:%S')} reloading onto {self.reload['code']} "
              f"{t - self.reload['t']:.0f}s after it appeared; pid {os.getpid()} stays", flush=True)
        if self.inhibit is not None and self.inhibit.poll() is None:
            self.inhibit.terminate()                        # the new code takes its own at once
        env = dict(os.environ, ASSETS_LINE_LISTEN_FD=str(self.srv.fileno()), ASSETS_LINE_LOCK_FD=str(self.lock_fd),
                   ASSETS_LINE_RELOADED=json.dumps({"from": self.code, "t": self.reload["t"]}))
        os.set_inheritable(self.srv.fileno(), True)
        os.set_inheritable(self.lock_fd, True)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            os.execve(sys.executable, [sys.executable, str(HERE / "assetd.py"), "--serve"], env)
        except OSError as ex:
            print(f"[line] reload failed ({ex}); staying on {self.code}", flush=True)
            self.bad_code, self.reload = self.reload["code"], None
            os.set_inheritable(self.srv.fileno(), False)
            os.set_inheritable(self.lock_fd, False)

    def power_note(self):
        g = self.guard.update()
        return {"t": round(now(), 1), "capacity": g.get("capacity"), "status": g.get("status"),
                "energy_wh": g.get("energy_wh"), "load1": round(os.getloadavg()[0], 2)}

    def check_guard(self, t):
        if t - self.t_guard < 5:
            return
        self.t_guard = t
        self.guard.update()
        self.verdict = self.guard.verdict()
        hold = Path(os.environ.get("ASSETS_GPU_HOLD", str(C.UE_HOME / "GPU_HOLD")))
        if hold.exists():                                   # the machine's hold file: no new card job
            self.verdict["cards"] = False
            self.verdict["why"].append(f"{hold} exists: no new card job")
        self.peak_load = max(self.peak_load, os.getloadavg()[0])
        why = "; ".join(self.verdict["why"])
        if why != self.said_why:
            print(f"[line] {time.strftime('%H:%M:%S')} guard: {why or 'all clear'} ({self.guard.brief()})", flush=True)
            self.said_why = why
        if self.verdict["pause"] and not self.state_written:
            self.wl.write_state([j.entry for j in self.jobs if j.entry], why)
            self.state_written = True
            print(f"[line] paused; list state written to {self.wl.root / 'state.json'}", flush=True)
        elif not self.verdict["pause"]:
            self.state_written = False

    def keep_awake(self):
        """An idle/sleep inhibitor the whole time the list has work, also while paused at the battery floor (the
        owner: "we want to block sleep"); released when the list is empty. It exits by itself when this scheduler
        dies."""
        want = bool(self.jobs) and INHIBIT
        alive = self.inhibit is not None and self.inhibit.poll() is None
        if want and not alive and shutil.which("systemd-inhibit"):
            watch = ("import os, sys, time\npid = int(sys.argv[1])\n"
                     "while os.path.exists(f'/proc/{pid}'):\n    time.sleep(5)\n")
            self.inhibit = subprocess.Popen(["systemd-inhibit", "--what=idle:sleep", "--who=asset line",
                                             "--why=asset jobs on the line", "--mode=block", sys.executable, "-c",
                                             watch, str(os.getpid())],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[line] holding the idle/sleep inhibitor (pid {self.inhibit.pid})", flush=True)
        elif not want and alive:
            self.inhibit.terminate()
            self.inhibit = None
            print("[line] inhibitor released", flush=True)

    def hand_over(self, b, st):
        """Give the station's card to a tool that takes its lock itself: stop the warm Blender spares and let go of
        the line's own hold on it (the tool's session counts as ours). Returns the card (None for the iGPU)."""
        card = self.cards.get(b[4:]) if b.startswith("card") else None
        if card is not None:
            with self.lock:
                card.cmd_busy = st
                card.stop_spares()
            t0 = now()
            while any(sp.proc.poll() is None for sp in card.spares) and now() - t0 < 20:
                time.sleep(0.2)
            with self.lock:
                card.spares = [sp for sp in card.spares if sp.proc.poll() is None]
                if card.fd is not None and not card.blender:
                    os.close(card.fd)
                    card.fd, card.held_since = None, None
        return card

    def start_cmd(self, st):
        """Start a command entry through the recording runner (line.py --run RC_FILE -- CMD), in its own session.
        Its pid, the pid's start time and the record file are on disk before this returns."""
        job, e = st.job, st.job.entry
        b = e["station"]
        log = Path(e.get("log") or (job.adir / "cmd.log"))
        log.parent.mkdir(parents=True, exist_ok=True)
        rc_file = job.adir / f"{st.name}.rc.json"
        rc_file.unlink(missing_ok=True)
        env = dict(os.environ, **{k: str(v) for k, v in (e.get("env") or {}).items()},
                   ASSETS_LINE_STATION=b, ASSETS_LINE_SEQ=str(job.seq), ASSETS_IMAGE_DEVICE=station_device(b))
        job.log(f"cmd at {b} ({e.get('kind')}): {' '.join(e['cmd'])[:200]}")
        with open(log, "a") as f:
            p = subprocess.Popen([sys.executable, str(HERE / "line.py"), "--run", str(rc_file), "--"] + list(e["cmd"]),
                                 cwd=e.get("cwd") or None, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        with self.lock:
            self.child_sids.add(p.pid)
            st.info.update(pid=p.pid, pid_start=proc_start(p.pid), rc_file=str(rc_file), log=str(log))
            self.persist(job)
        return p

    def run_cmd(self, st, adopt=None):
        """A generic line entry: run its command at its station. The command takes the station's lock itself
        (existing tools do), so the line lets go of that lock first and counts the command's session as its own.
        The runner records the exit code the moment the command ends, and the line writes the stage's end to the
        entry before anything else, so a restart at any point never runs a finished command again. adopt
        ({"pid", "pid_start", "rc_file"}): a command a previous line started that still runs; watch it to its end."""
        job, e = st.job, st.job.entry
        b = e["station"]
        st.device, st.device_class = b, "cmd"
        card = self.hand_over(b, st)
        try:
            p = None
            if adopt:
                pid, start = adopt["pid"], adopt["pid_start"]
                with self.lock:
                    self.child_sids.add(pid)
                    st.info.update(pid=pid, pid_start=start, rc_file=adopt["rc_file"], adopted=True,
                                   log=str(e.get("log") or (job.adir / "cmd.log")))
                    self.persist(job)
                job.log(f"the line restarted while the command ran: watching pid {pid} to its end")
                alive = lambda: proc_start(pid) == start
            else:
                p = self.start_cmd(st)
                pid = p.pid
                alive = lambda: p.poll() is None
            while alive():
                if job.cancelled:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except OSError:
                        pass
                    t0 = now()
                    while alive() and now() - t0 < 15:
                        time.sleep(0.2)
                    if alive():
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except OSError:
                            pass
                    if p is not None:
                        p.wait()
                    st.state = "cancel"
                    return
                time.sleep(0.2)
            rec = L.read_rc(st.info["rc_file"])
            rc = rec["rc"] if rec else None
            if rc is None and p is not None:
                rc = p.returncode
            log = st.info.get("log")
            with self.lock:                                 # the end is on disk first, then the rest
                st.info["rc"] = rc
                job.result = {"rc": rc if rc is not None else 1, "log": log}
                if rc != 0:
                    st.info["error"] = (f"command failed (rc {rc}); see {log}" if rc is not None else
                                        f"the command ended without a recorded exit code (killed?); see {log}")
                    st.state = "fail"
                else:
                    st.state = "done"
                self.persist(job)
            job.log(f"cmd rc={rc}; log {log}")
        finally:
            with self.lock:
                self.child_sids.discard(st.info.get("pid"))
                if card is not None and card.cmd_busy is st:
                    card.cmd_busy = None

    def keep_spares(self, c, t):
        """Warm Blender spares on the Blender card while it is ours and in use (within its linger)."""
        for sp in c.spares:
            if sp.proc.poll() is not None:
                shutil.rmtree(sp.dir, ignore_errors=True)
        c.spares = [sp for sp in c.spares if sp.proc.poll() is None]
        want = 0
        if (c.role == "blender" and c.fd is not None and not c.draining and not self.released and not self.reload
                and t - c.last_used < C.BLENDER_LINGER_S):
            want = C.BLENDER_SPARES
        idle = [sp for sp in c.spares if not sp.taken]
        if len(idle) < want and c.free_gb() >= C.BAKE_VRAM_GB and R.mem_avail_gb() > C.MIN_FREE_GB + 8:
            c.spares.append(Spare(c))
        elif len(idle) > want:
            for sp in idle[want:]:
                (sp.dir / "quit").touch()

    def should_exit(self):
        if self.jobs or any(e.state != "dead" for e in self.editors):
            return False
        return self.stopping or self.drain or now() - self.last_busy > C.SCHED_IDLE_S

    def launch(self, st, fn, arg):
        st.state, st.t_start = "run", now()
        b = self.station(st)
        if b:
            st.info["power_start"] = self.power_note()
            ps = st.info["power_start"]
            print(f"[line] {time.strftime('%H:%M:%S')} start {b} {self.skind(st)} #{st.job.seq} {st.job.id} | "
                  f"bat {ps['capacity']}% {ps['status']} {ps['energy_wh']}Wh load {ps['load1']}", flush=True)
            st.job.log(f"start {st.name} at {b} | battery {ps['capacity']}% {ps['status']} {ps['energy_wh']} Wh")
        self.persist(st.job)

        def go():
            try:
                fn(st, arg)
            except BaseException as ex:                     # SystemExit from a chain helper too
                err = traceback.format_exc()
                print(err, flush=True)
                st.info["error"] = str(ex) if isinstance(ex, SystemExit) else err.strip().splitlines()[-1]
                st.state = "fail"
            finally:
                with self.lock:
                    if st.state == "run":
                        st.state = "fail"
                    st.t_end = now()
                    if b:
                        self.note_cost(st)
                    self.after_stage(st)
                    self.persist(st.job)
                    self.wake.notify_all()

        threading.Thread(target=go, daemon=True).start()
        return True

    def note_cost(self, st):
        """Battery energy over a job (start and end logged): what each kind of job costs."""
        a, z = st.info.get("power_start") or {}, self.power_note()
        st.info["power_end"] = z
        dt = (st.t_end or now()) - (st.t_start or now())
        d = None
        if a.get("energy_wh") is not None and z.get("energy_wh") is not None:
            d = round(z["energy_wh"] - a["energy_wh"], 3)
        k = self.skind(st)
        self.kind_cost[k].append((round(dt, 1), d))
        rec = {"seq": st.job.seq, "id": st.job.id, "station": self.station(st), "kind": k, "stage": st.name,
               "state": st.state, "s": round(dt, 1), "start": a, "end": z, "energy_delta_wh": d,
               "w_avg": round(d * 3600 / dt, 1) if d is not None and dt > 0 else None}
        try:
            with open(self.wl.root / "power.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        print(f"[line] {time.strftime('%H:%M:%S')} end {rec['station']} {k} #{st.job.seq} {st.job.id} {st.state} "
              f"{rec['s']}s | bat {z['capacity']}% {z['status']} {z['energy_wh']}Wh (delta {d} Wh) load {z['load1']}",
              flush=True)

    def after_stage(self, st):
        """A stage ended: report the job when nothing of it runs any more (a failure cancels what is queued)."""
        job = st.job
        for c in self.cards.values():
            if c.cmd_busy is st and st.state != "run":
                c.cmd_busy = None
        if job.state in ("done", "failed", "cancelled"):
            return
        if st.state == "fail" and job.pending is None:
            job.pending = ("failed", {"rc": 1, "error": st.info.get("error") or f"{st.name} failed"})
            for s in job.stages:
                if s.state in ("wait", "ready"):
                    s.state = "cancel"
        if any(s.state == "run" for s in job.stages):
            return
        if job.pending:
            self.finish_job(job, *job.pending)
        elif job.cancelled:
            self.finish_job(job, "cancelled", {"rc": 130, "error": getattr(job, "cancel_why", "cancelled")})
        elif all(s.state in ("done", "cancel") for s in job.stages):
            self.finish_job(job, "done", job.result or {"rc": 0})

    def finish_job(self, job, state, result):
        job.state = state
        rec = self.job_record(job)
        result = dict(result, sched=rec)
        try:
            (job.adir / "logs").mkdir(parents=True, exist_ok=True)
            (job.adir / "logs" / "sched.json").write_text(json.dumps(rec, indent=1, default=str))
        except OSError:
            pass
        if job.conn is not None:
            try:
                with job.send_lock:
                    job.conn.sendall((json.dumps({"result": result}, default=str) + "\n").encode())
            except OSError:
                pass
        self.recent.appendleft(rec)
        if job in self.jobs:
            self.jobs.remove(job)
        if job.entry is not None:
            job.entry["result"] = {k: v for k, v in result.items() if k != "sched"}
        self.persist(job)
        print(f"[sched] job {job.n} {job.id} {job.mode} {state} in {rec['total_s']} s: "
              + ", ".join(f"{s['name']} {s.get('device_class') or ''} w{s['wait_s']} r{s['run_s']}"
                          for s in rec["stages"]), flush=True)

    def job_record(self, job):
        return {"job": job.n, "id": job.id, "mode": job.mode, "state": job.state, "client_pid": job.client_pid,
                "submitted": datetime.datetime.fromtimestamp(job.t_submit).isoformat(timespec="seconds"),
                "total_s": round(now() - job.t_submit, 1),
                "wait_s": round(sum((s.record()["wait_s"] or 0) for s in job.stages), 1),
                "stages": [s.record() for s in job.stages]}

    # ------------------------------------------------------------ Blender placement (GPU only)
    def vram_need(self, st):
        a = st.job.args
        if st.job.mode == "preview":
            return C.BAKE_VRAM_GB / 2
        px = int(st.job.spec.get("texture_px") or C.TIERS.get(st.job.spec.get("tier"), {}).get("tex", 1024))
        return C.BAKE_VRAM_GB * (1.5 if px >= 4096 else 1.0) + (0 if a.no_sheet else 0)

    def card_ok(self, c, need):
        if len(c.blender) >= C.BLENDER_PER_CARD or c.draining:
            return False
        if c.fd is None and not c.hold():
            return False                                   # an outside user holds the card (or it was released)
        return c.free_gb() >= need

    def place_blender(self, st):
        """The Blender card, or None (the stage waits). Bakes, sheets and previews all run on the Blender card, one
        job at a time; never the iGPU (it has its own station) and never the CPU unless --device cpu was asked for
        by hand."""
        want = st.job.args.device or C.BLENDER_DEVICE
        if want == "cpu":
            return ("cpu", None)
        if self.released:
            return None
        home = self.cards[self.bl_bus]
        if self.card_ok(home, self.vram_need(st)):
            return ("card", home)
        return None

    def run_blender(self, st, pl):
        job, a = st.job, st.job.args
        kind, dev = pl
        with self.lock:
            if dev is not None:
                dev.blender.append(st)
        bus = dev.bus if dev is not None else None
        st.device = f"card {bus}" if kind == "card" else ("igpu" if kind == "igpu" else "cpu (asked)")
        st.device_class = {"card": "xtx", "igpu": "igpu", "cpu": "cpu"}[kind]
        mode = job.mode
        try:
            job.log(f"blender {mode}: {st.device} after {round(st.t_start - st.t_ready, 1)}s")
            bj = R.make_blender_job(job.spec, a.model, job.adir, a, mode, ue_follows=not a.skip_render)
            bj.update(device="HIP" if bus else "CPU", hip_bus=bus or "")
            jp = job.adir / "logs" / f"blender_{mode}_job.json"
            jp.write_text(json.dumps(bj, indent=1))
            rp = job.adir / "blender_report.json"
            rp.unlink(missing_ok=True)
            fds = (dev.fd,) if dev is not None and dev.fd is not None else ()
            t = now()
            with self.lock:
                sp = dev.ready_spare() if kind == "card" and bj.get("hiprt") == C.HIPRT else None
                if sp is not None:
                    sp.taken = True
            if sp is not None:                              # a warm Blender on this card takes the job
                tmp = sp.dir / "job.tmp"
                tmp.write_text(json.dumps({"job": str(jp), "log": str(job.adir / "logs" / f"blender_{mode}.log")}))
                tmp.replace(sp.dir / "job.json")
                p = sp.proc
                st.info["warm_spare"] = True
            else:
                with open(job.adir / "logs" / f"blender_{mode}.log", "w") as f:
                    p = subprocess.Popen([str(c) for c in (C.BLENDER, "-b", "--factory-startup", "-t",
                                                           C.BLENDER_THREADS, "--python", HERE / "bl_harness.py",
                                                           "--", jp)],
                                         stdout=f, stderr=subprocess.STDOUT, pass_fds=fds, start_new_session=True)
                st.info["warm_spare"] = False
            st.info["pid"] = p.pid
            with self.lock:
                self.persist(job)
            while p.poll() is None:
                if job.cancelled:
                    p.terminate()
                    try:
                        p.wait(10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    break
                if not job.exported and mode == "final" and rp.exists():
                    try:
                        r0 = json.loads(rp.read_text())
                    except ValueError:
                        r0 = {}
                    if r0.get("status") == "exported":
                        with self.lock:
                            job.blend, job.exported = r0, True
                            st.info["export_s"] = round(now() - t, 1)
                            self.wake.notify_all()
                        job.log(f"glTF exported after {st.info['export_s']}s; UE may start")
                time.sleep(0.1)
            dt = round(now() - t, 1)
            rep = json.loads(rp.read_text()) if rp.exists() else {}
            st.info.update(rc=p.returncode, status=rep.get("status"), harness_device=rep.get("device"))
            job.log(f"blender {mode} on {st.device}: rc={p.returncode} {dt}s status={rep.get('status')}")
            if job.cancelled:
                st.state = "cancel"
                return
            if p.returncode != 0 or rep.get("status") != "ok":
                st.info["error"] = (f"blender step failed; see {job.adir / 'logs' / f'blender_{mode}.log'}\n"
                                    f"{rep.get('error', '')}")
                st.state = "fail"
                return
            rep["wall_s"] = dt
            rep["startup_s"] = round(dt - rep["times_s"].get("total", 0), 1)
            rep["lock_wait_s"] = round(st.t_start - st.t_ready, 1)
            rep["device_choice"] = st.device
            rep["gpu_load"] = self.load_between(bus, t, now()) if bus else None
            rp.write_text(json.dumps(rep, indent=1))
            st.info["gpu_busy_mean"] = (rep["gpu_load"] or {}).get("busy_mean")
            with self.lock:
                job.blend = rep
                job.exported = True
            self.est_put(f"{mode}:{job.spec.get('tier')}:{st.device_class}", dt)
            job.times["blender"] = dt
            if mode == "preview":
                failed = [c["name"] for c in rep["checks"] if not c["ok"]]
                job.log(f"preview {dt}s; sheet {rep['files'].get('preview_sheet')}; failed checks: {failed or 'none'}")
                job.result = {"rc": 0, "preview_sheet": rep["files"].get("preview_sheet"), "failed_checks": failed}
            else:
                failed = [c["name"] for c in rep["checks"] if not c["ok"]]
                if failed:
                    job.log(f"checks failed: {failed} (continuing; recorded in asset.json)")
            st.state = "done"
        finally:
            with self.lock:
                if dev is not None and st in dev.blender:
                    dev.blender.remove(st)

    def load_between(self, bus, t0, t1):
        v = [u.get(bus) for t, u in self.util if t0 <= t <= t1 and u.get(bus) is not None]
        return {"samples": len(v), "busy_mean": round(sum(v) / len(v), 1) if v else None,
                "busy_peak": max(v, default=None)}

    # ------------------------------------------------------------ paint (self-managed devices)
    def run_paint(self, st, _):
        job = st.job
        job.log("paint: guide views -> image model -> projection")
        b = C.IMAGE_STATION
        st.device, st.device_class = f"paint.py at {b}", "paint"
        card = self.hand_over(b, st)
        env = dict(os.environ, ASSETS_LINE_STATION=b, ASSETS_IMAGE_DEVICE=station_device(b))
        try:
            with open(job.adir / "logs" / "paint_stage.log", "w") as f:
                p = subprocess.Popen([sys.executable, str(HERE / "paint.py"), str(job.adir)], stdout=f,
                                     stderr=subprocess.STDOUT, start_new_session=True, env=env)
                with self.lock:
                    self.child_sids.add(p.pid)
                while p.poll() is None:
                    if job.cancelled:
                        p.terminate()
                        p.wait()
                        st.state = "cancel"
                        return
                    time.sleep(0.5)
        finally:
            with self.lock:
                self.child_sids.discard(p.pid if "p" in locals() else None)
                if card is not None:
                    card.cmd_busy = None
        pj = job.adir / "paint" / "paint.json"
        if p.returncode != 0 or not pj.exists():
            st.info["error"] = f"paint failed; see {job.adir / 'logs' / 'paint_stage.log'}"
            st.state = "fail"
            return
        job.times["paint"] = json.loads(pj.read_text()).get("times_s")
        job.painted = True
        st.state = "done"

    # ------------------------------------------------------------ UE
    def place_ue(self, st):
        if self.released:
            return None
        if st.job.args.rebuild_stage:
            live = [e for e in self.editors if e.state != "dead"]
            for e in live:
                if e.state == "ready":
                    e.quit()
            return "rebuild" if not live else None
        cur = code_hash(UE_FILES)
        want = getattr(st.job, "project", None) or str(C.UE_PROJECT)
        idle = [e for e in self.editors if e.state == "ready" and e.code == cur and e.project == want]
        # fairness between projects: once another project's oldest job has waited past the swap limit and this
        # editor has had its minimum stay, leave it idle so manage_editors swaps it (else a busy project starves
        # the other for as long as it keeps submitting)
        need = self.next_project()
        if need and need != want and self.oldest_wait(need) > C.UE_SWAP_AFTER_S:
            idle = [e for e in idle if not self.stay_over(e)]
        for e in self.editors:
            if e.state == "ready" and e.code != cur:
                print(f"[sched] editor {e.card.bus}: ue code changed; restarting it", flush=True)
                e.quit()
        if not idle:
            return None
        idle.sort(key=lambda e: (not e.primary, e.n_jobs))
        e = idle[0]
        e.state, e.job = "busy", st
        return e

    def ue_backlog(self, project=None):
        """UE stages queued (for one project, or all)."""
        n = 0
        for j in self.jobs:
            if j.cancelled or j.mode not in ("final", "ue_job"):
                continue
            if project is not None and (getattr(j, "project", None) or str(C.UE_PROJECT)) != project:
                continue
            for s in j.stages:
                if s.kind == "ue" and s.state in ("wait", "ready"):
                    n += 1
        return n

    def stay_over(self, e):
        """True once an editor has served its project for the minimum stay (so swaps cannot ping-pong)."""
        t_ready = getattr(e, "t_ready", None)
        return t_ready is None or now() - t_ready > C.UE_STAY_S

    def oldest_wait(self, project):
        """Seconds the oldest queued UE stage of a project has been ready."""
        t, w = now(), 0.0
        for j in self.jobs:
            if (getattr(j, "project", None) or str(C.UE_PROJECT)) != project:
                continue
            for s in j.stages:
                if s.kind == "ue" and s.state == "ready" and s.t_ready:
                    w = max(w, t - s.t_ready)
        return w

    def next_project(self):
        """The project of the oldest queued UE stage (the editor serves one project at a time)."""
        for j in self.jobs:
            if j.cancelled or j.mode not in ("final", "ue_job"):
                continue
            if any(s.kind == "ue" and s.state in ("wait", "ready") for s in j.stages):
                return getattr(j, "project", None) or str(C.UE_PROJECT)
        return str(C.UE_PROJECT) if self.warm_wanted else None

    def manage_editors(self):
        live = [e for e in self.editors if e.state != "dead"]
        blocked = any(j.args.rebuild_stage for j in self.jobs if j.mode == "final")
        backlog = self.ue_backlog()
        need = self.next_project()
        t = now()
        if self.released:
            for e in live:
                if e.state in ("ready", "starting"):
                    e.quit()                                  # in-flight UE work finishes first (state busy)
            return
        # idle quits (and all idle editors when draining for newer code with nothing queued)
        for e in live:
            if e.state == "ready" and self.drain and not self.jobs:
                print("[sched] newer chain code: unloading the idle editor so the next scheduler loads it", flush=True)
                e.quit()
                continue
            if e.state == "ready" and need and e.project != need and (
                    self.ue_backlog(e.project) == 0
                    or (self.oldest_wait(need) > C.UE_SWAP_AFTER_S and self.stay_over(e))):
                print(f"[sched] editor on {e.card.bus}: swapping project {Path(e.project).stem} -> "
                      f"{Path(need).stem}", flush=True)
                self.swap_t = t
                e.quit()
                continue
            if e.state == "ready":
                lim = C.UE_IDLE_S if e.primary else C.UE_IDLE2_S
                if (e.idle_since and t - e.idle_since > lim and backlog == 0) or self.stopping:
                    print(f"[sched] editor on {e.card.bus} idle {t - e.idle_since:.0f}s: quitting", flush=True)
                    e.quit()
        if blocked or self.stopping or need is None:
            return
        # no editor while a reload is pending, while a command borrows the engine card or has the turn for it (an
        # editor pre-warmed for a bake still running would only be unloaded again), or before the borrow's thread
        # has taken the card
        if self.reload or self.engine_turn == "borrow" or self.cards[self.ue_bus].cmd_busy is not None:
            return
        if live and not (C.UE_EDITORS >= 2 and len(live) == 1 and live[0].state in ("ready", "busy")
                         and live[0].project == need and backlog >= C.UE_SECOND_AT):
            return                                            # the resident editor, plus a second one under load
        other = other_project_editors(need) if not live else []
        if other:                                             # an editor not started by us writes this project
            if asset_project(need) and R.server_dir().joinpath("queue").exists():
                (R.server_dir() / "queue" / "quit").touch()   # the pre-scheduler warm editor: ask it to go
            if t - getattr(self, "said_other", 0) > 30:
                self.said_other = t
                print(f"[sched] waiting for editor(s) {other} on {Path(need).stem} to quit", flush=True)
            return
        for b in ([self.ue_bus] if not live else [self.bl_bus]):
            c = self.cards.get(b)
            if c.lent or c.draining or c.editor is not None:
                continue
            had = c.fd is not None
            if not c.hold():
                continue
            if c.free_gb() < C.UE_VRAM_GB:
                if not had:
                    c.release_if_unused()
                continue
            if self.project_fd is None:
                self.project_fd = try_flock(proj_dir(need) / "project.lock")
                if self.project_fd is None:
                    if not had:
                        c.release_if_unused()
                    return                                    # a --local run owns the project right now
            self.warm_wanted = False
            e = Editor(c, primary=not live, project=need)
            c.editor = e
            self.editors.append(e)
            threading.Thread(target=self.start_editor, args=(e,), daemon=True).start()
            return

    def start_editor(self, e):
        c = e.card
        try:
            ours = asset_project(e.project)
            shim = Path(e.project).parent / "Content" / "Python" / "init_unreal.py"
            if ours:
                R.install_shim()
                if R.ensure_rt_ini():
                    print("[sched] project renderer settings written; this start compiles shaders", flush=True)
                if not any(x.state in ("ready", "busy") for x in self.editors if x is not e):
                    R.ensure_stage(C.LOGS)
                self.mannequin = R.ensure_mannequin()
            elif not shim.exists():                           # another project's own shim is never replaced
                shim.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(HERE / "ue" / "init_unreal.py", shim)
            if e.dir.exists():
                shutil.rmtree(e.dir, ignore_errors=True)
            e.qdir.mkdir(parents=True, exist_ok=True)
            first = {"script": str(HERE / "ue" / "ue_render.py"), "map": C.UE_STAGE_MAP if ours else None,
                     "serve_dir": str(e.qdir), "serve_idle_s": (C.UE_IDLE_S if e.primary else C.UE_IDLE2_S) + 120,
                     "ready_file": str(e.dir / "ready.json"), "res": [1152, 640], "warm_frames": 90,
                     "settle_frames": 30, "ref_box_material": C.UE_STAGE_MAP.rsplit("/", 1)[0] + "/M_StageRef",
                     "mannequin": self.mannequin if ours else None, "idle_fps": C.UE_IDLE_FPS}
            jp = e.dir / "first_job.json"
            jp.write_text(json.dumps(first, indent=1))
            cmd, env = R.editor_cmd(c.bus)
            if not ours:                                      # the same flags on the other project, no stage map
                cmd = [cmd[0], e.project] + [x for x in cmd[3:]]
            env["ASSETS_UE_JOB"] = str(jp)
            env["MH_JOB"] = str(jp)                           # the character tools' shim reads this name
            e.log = C.LOGS / f"assets_ue_ed{c.bus}_{time.strftime('%m%d_%H%M%S')}.log"
            fds = tuple(x for x in (c.fd, self.project_fd) if x is not None)
            with open(e.log, "w") as f:
                # the locks stay with a small holder process; the editor itself (and the helpers it forks, such
                # as zenserver and UnrealTraceServer, which outlive it) never inherit them
                e.proc = subprocess.Popen([sys.executable, str(HERE / "assetd.py"), "--hold", "--"] + cmd, stdout=f,
                                          stderr=subprocess.STDOUT, env=env, pass_fds=fds, start_new_session=True)
            e.pid = e.proc.pid
            (e.dir / "editor.json").write_text(json.dumps({"pid": e.pid, "card": c.bus, "log": str(e.log),
                                                           "started": time.ctime(), "sched_pid": os.getpid()}))
            print(f"[sched] editor pid {e.pid} starting on card {c.bus} for {Path(e.project).stem} "
                  f"({'primary' if e.primary else 'second'})",
                  flush=True)
            while e.alive():
                if (e.dir / "ready.json").exists():
                    try:
                        e.upid = json.loads((e.dir / "ready.json").read_text())["pid"]
                    except (OSError, ValueError, KeyError):
                        pass
                    with self.lock:
                        e.t_ready, e.idle_since = now(), now()
                        if e.state == "starting":
                            e.state = "ready"
                        self.wake.notify_all()
                    self.est_put("editor_start", e.t_ready - e.t_start)
                    sw = getattr(self, "swap_t", None)
                    print(f"[sched] editor on {c.bus} ready for {Path(e.project).stem} after "
                          f"{e.t_ready - e.t_start:.1f}s" + (f" (project swap: {e.t_ready - sw:.1f}s since the "
                                                             f"swap began)" if sw else ""), flush=True)
                    self.swap_t = None
                    return
                if R.mem_avail_gb() < C.MIN_FREE_GB:
                    e.proc.terminate()
                time.sleep(0.2)
            e.error = f"editor exited during start (rc {e.proc.returncode}); see {e.log}"
            print(f"[sched] {e.error}", flush=True)
        except Exception:
            e.error = traceback.format_exc()
            print(e.error, flush=True)
            if e.proc is not None and e.proc.poll() is None:
                e.proc.terminate()

    def reap_editors(self):
        for e in self.editors:
            if e.state == "dead":
                continue
            gone = (e.proc is None and e.error) or (e.proc is not None and e.proc.poll() is not None)
            if gone:
                rc = e.proc.returncode if e.proc else None
                print(f"[sched] editor on {e.card.bus} ended (rc {rc}, {e.n_jobs} jobs)", flush=True)
                e.state = "dead"
                if e.card.editor is e:
                    e.card.editor = None
        self.editors = [e for e in self.editors if e.state != "dead" or now() - e.t_start < 3600][-8:]
        if self.project_fd is not None and not any(e.state != "dead" for e in self.editors):
            os.close(self.project_fd)
            self.project_fd = None
        for e in self.editors:
            if e.state != "dead" and R.mem_avail_gb() < C.MIN_FREE_GB and e.proc is not None:
                print(f"[sched] MemAvailable {R.mem_avail_gb():.1f} GiB: stopping editor {e.pid}", flush=True)
                e.proc.terminate()

    def write_tenancy(self):
        for c in self.cards.values():
            e = c.editor
            if c.fd is not None and e is not None and e.state != "dead":
                txt = json.dumps({"holder": "asset chain scheduler (UE editor)", "sched_pid": os.getpid(),
                                  "editor_pid": e.pid, "since": time.ctime(e.t_start), "state": e.state,
                                  "releases_after_idle_s": C.UE_IDLE_S if e.primary else C.UE_IDLE2_S,
                                  "idle_since": time.ctime(e.idle_since) if e.idle_since else None})
                try:
                    if not c.tenant_path.exists() or c.tenant_path.read_text() != txt:
                        c.tenant_path.write_text(txt)
                except OSError:
                    pass
            else:
                c.tenant_path.unlink(missing_ok=True)

    def run_ue(self, st, ed):
        if st.name == "ue_script":
            try:
                return self._run_ue_script(st, ed)
            finally:
                self.editor_free(ed)
        job, a = st.job, st.job.args
        blend = job.blend or {}
        glb = (blend.get("files") or {}).get("export", {}).get("glb")
        painted = job.adir / "export" / "painted" / f"{job.id}.glb"     # as run_asset.py (paint.py writes it)
        tset = job.spec.get("texture_set") or ("painted" if a.paint and (job.spec.get("paint") or {}).get("use", True)
                                               else "procedural")
        if tset == "painted":
            if not painted.exists():
                st.info["error"] = "texture_set painted but no painted maps: run with --paint"
                st.state = "fail"
                if isinstance(ed, Editor):
                    self.editor_free(ed)
                return
            glb = str(painted)
        job.tset = tset
        job.log(f"texture set for UE: {tset}")
        dest = f"{C.UE_ASSET_ROOT}/{job.spec['class'].capitalize()}s/{job.id}"
        imp_job = {"file": glb, "dest": dest, "id": job.id, "nanite": not blend.get("glass_materials"),
                   "result": str(job.adir / "logs" / "ue_import.json")}
        Path(imp_job["result"]).unlink(missing_ok=True)
        job.imp_job = imp_job
        if ed in ("cmdlet", "rebuild"):
            st.device, st.device_class = "commandlet (no GPU)", "none"
            with self.lock:
                if self.project_fd is None:
                    self.project_fd = try_flock(R.server_dir() / "project.lock")
                pf = self.project_fd
            if pf is None:
                st.info["error"] = "the project lock is held by a --local run; try again"
                st.state = "fail"
                return
            try:
                if ed == "rebuild":
                    R.ensure_stage(job.adir / "logs", True)
                    job.args.rebuild_stage = False          # the render goes through an editor now
                    st.state = "ready"
                    st.t_ready = now()
                    return
                job.log("UE import (commandlet)")
                rc, job.times["ue_import"], imp = R.ue_cmdlet(HERE / "ue" / "ue_import.py", imp_job,
                                                              job.adir / "logs" / "ue_import.log")
            finally:
                with self.lock:
                    if not any(e.state != "dead" for e in self.editors) and self.project_fd is not None:
                        os.close(self.project_fd)
                        self.project_fd = None
            job.imp = imp
            st.state = "done" if imp.get("ok") else "fail"
            if not imp.get("ok"):
                st.info["error"] = f"UE import failed: {imp.get('error')}"
            return
        st.device, st.device_class = f"editor {ed.card.bus}", "xtx"
        try:
            self._run_ue_editor(st, ed, imp_job)
        finally:
            self.editor_free(ed)

    def _run_ue_script(self, st, ed):
        """A generic editor job (run_asset.py --ue-job): the script runs in the resident editor of its project."""
        job, jd = st.job, st.job.ue
        st.device, st.device_class = f"editor {ed.card.bus}", "xtx"
        res_p = job.adir / "ue_result.json"
        res_p.unlink(missing_ok=True)
        Path(str(res_p) + ".cancel").unlink(missing_ok=True)
        ed.n_jobs += 1
        q = ed.qdir / f"{job.id}.{job.n}.job.json"
        tmp = ed.qdir / f".{job.id}.{job.n}.tmp"
        tmp.write_text(json.dumps({"ue_script": jd["script"], "job": jd.get("job", {}), "id": job.id,
                                   "result": str(res_p)}, indent=1))
        tmp.replace(q)
        job.log(f"UE script {Path(jd['script']).name} on {Path(ed.project).stem}: editor on card {ed.card.bus} "
                f"after {round(st.t_start - st.t_ready, 1)}s")
        t0, limit = now(), float(jd.get("timeout_s", C.UE_JOB_TIMEOUT_S))
        outs = [Path(o) for o in jd.get("outputs", [])]
        res, why = None, None
        while True:
            if res_p.exists():
                try:
                    res = json.loads(res_p.read_text())
                    break
                except ValueError:
                    pass
            if job.cancelled:                               # --cancel, or the client went away: stop it in the editor
                Path(str(res_p) + ".cancel").touch()
                t1 = now()
                while not res_p.exists() and ed.alive() and now() - t1 < 15:
                    time.sleep(0.1)
                why = "cancelled"
                break
            if not ed.alive():
                why = "the editor died"
                break
            if now() - t0 > limit:
                why = f"timed out after {limit:.0f}s (timeout_s)"
                Path(str(res_p) + ".cancel").touch()        # a generator script stops at its next tick
                break
            time.sleep(0.05)
        Path(str(res_p) + ".cancel").unlink(missing_ok=True)
        missing = []
        if res is not None and res.get("ok") and not why:
            t1 = now()                                      # outputs: checked once, with a short grace
            while now() - t1 < C.UE_JOB_OUTPUT_GRACE_S and not all(o.exists() for o in outs):
                time.sleep(0.1)
            missing = [str(o) for o in outs if not o.exists()]
        dt = round(now() - t0, 1)
        ok = bool(res and res.get("ok")) and not missing and not why
        err = None if ok else (why or (res or {}).get("error") or (f"outputs missing: {missing}" if missing else
                                                                    "the script failed"))
        if why == "cancelled":
            st.info["error"] = err
            job.log(f"UE script cancelled after {dt}s")
            job.result = {"rc": 130, "error": err, "result_file": str(res_p), "seconds": dt}
            st.state = "cancel"
            return
        job.result = {"rc": 0 if ok else 1, "result": (res or {}).get("result"), "result_file": str(res_p),
                      "seconds": dt, "error": err}
        job.log(f"UE script done in {dt}s ok={ok}" + (f": {err}" if err else ""))
        if ok:
            st.state = "done"
        else:
            st.info["error"] = err
            st.state = "fail"

    def editor_free(self, ed):
        with self.lock:
            if ed.state == "busy":
                ed.state = "ready"
            ed.job, ed.idle_since = None, now()
            self.wake.notify_all()

    def _run_ue_editor(self, st, ed, imp_job):
        job, a = st.job, st.job.args
        out = job.adir / "ue"
        out.mkdir(exist_ok=True)
        uj = R.ue_render_job(job.spec, imp_job, job.adir, a, self.mannequin)
        if st.name == "ue_import":
            uj["stills"] = False
        rp = Path(uj["result"])
        rp.unlink(missing_ok=True)
        ed.n_jobs += 1
        q = ed.qdir / f"{job.id}.{job.n}.job.json"
        tmp = ed.qdir / f".{job.id}.{job.n}.tmp"
        tmp.write_text(json.dumps(uj, indent=1))
        tmp.replace(q)
        job.log(f"UE import + render: editor on card {ed.card.bus} (pid {ed.pid}) after "
                f"{round(st.t_start - st.t_ready, 1)}s")
        t0 = now()
        res = None
        while now() - t0 < a.ue_timeout:
            if rp.exists():
                try:
                    res = json.loads(rp.read_text())
                    break
                except ValueError:
                    pass
            if not ed.alive():
                break
            time.sleep(0.05)
        dt = round(now() - t0, 1)
        if res is None:
            died = not ed.alive()
            st.info["error"] = (f"editor on card {ed.card.bus} {'died' if died else 'timed out'}; see {ed.log}")
            if died and not st.info.get("retried"):
                st.info["retried"] = True                    # once, on whichever editor is next free
                job.log(st.info["error"] + "; retrying the UE stage")
                with self.lock:
                    st.state, st.t_ready = "ready", now()
                return
            if not died:
                ed.proc.terminate()
            res = {"ok": False, "error": st.info["error"]}
        res.update({"time_s": dt, "card": ed.card.bus, "editor": "warm" if ed.n_jobs > 1 else "started",
                    "editor_pid": ed.pid, "log": str(ed.log), "lock_wait_s": round(st.t_start - st.t_ready, 1),
                    "gpu_load": self.load_between(ed.card.bus, t0, now())})
        st.info["gpu_busy_mean"] = res["gpu_load"].get("busy_mean")
        job.render = res
        job.times["ue_render"] = dt
        ip = Path(imp_job["result"])
        imp = json.loads(ip.read_text()) if ip.exists() else {"ok": False, "error": res.get("error")}
        job.imp = imp
        job.times["ue_import_in_editor"] = imp.get("time_s")
        self.est_put("ue", dt)
        job.log(f"render {dt}s ok={res.get('ok')} card={ed.card.bus} editor={res['editor']}")
        if not imp.get("ok"):
            st.info["error"] = f"UE import failed: {imp.get('error')}"
            st.state = "fail"
            return
        st.state = "done"

    # ------------------------------------------------------------ finalize (manifest)
    def run_finalize(self, st, _):
        job = st.job
        st.device, st.device_class = "cpu", "cpu-io"
        imp = getattr(job, "imp", {}) or {}
        render = getattr(job, "render", None)
        job.log(f"import ok={imp.get('ok')} -> {imp.get('static_mesh')} problems={imp.get('problems')}")
        job.times["total"] = round(now() - job.t_submit, 1)
        job.times["waited"] = round(sum((s.record()["wait_s"] or 0) for s in job.stages), 1)
        entry = R.write_manifest(job.spec, job.adir, job.blend, imp, render, job.args.model, job.times)
        entry["texture_set"] = getattr(job, "tset", "procedural")
        pj = job.adir / "paint" / "paint.json"
        if pj.exists():
            pr = json.loads(pj.read_text())
            entry["textures_painted"] = (pr.get("files") or {}).get("textures")
            entry["export_painted"] = (pr.get("files") or {}).get("glb")
        entry["sched"] = {s.name: {"device": s.device, "wait_s": s.record()["wait_s"], "run_s": s.record()["run_s"]}
                          for s in job.stages if s is not st}
        (job.adir / "asset.json").write_text(json.dumps(entry, indent=1))
        job.log(f"done in {job.times['total']}s; checks_passed={entry['checks_passed']}; "
                f"render ok={(render or {}).get('ok')}")
        job.log(f"asset folder: {job.adir}")
        job.result = {"rc": 0 if (render is None or render.get("ok")) else 3}
        st.state = "done"

    # ------------------------------------------------------------ status
    def status(self):
        t = now()
        out = {"pid": os.getpid(), "up_s": round(t - self.t0), "code": self.code, "drain": self.drain,
               "reload": dict(self.reload, waiting_for=[f"#{j.seq} {j.id} {st.name}" for j in self.jobs
                                                        for st in j.stages if st.state == "run"])
               if self.reload else None,
               "devices": [], "queue": [], "recent": list(self.recent)[:12], "estimates": self.est,
               "spills": self.spills, "released": self.released, "last_release": self.last_release,
               "outside": {b: round(t - v) for b, v in self.outside.items()}}
        for c in self.cards.values():
            u, tot = vram(c.bus)
            d = {"device": f"card {c.bus} ({c.role}{', draining' if c.draining else ''}"
                           f"{', released' if t < c.released_until else ''})", "busy": busy_pct(c.bus),
                 "vram_gb": f"{u:.1f}/{tot:.0f}" if u is not None else None,
                 "lock": "ours" if c.fd is not None else (
                     "outside: " + "; ".join(f"{p} {cmd}" for p, cmd in lock_holders(c.lock_path, True))),
                 "blender": [f"{s.job.id} {s.name} {t - s.t_start:.0f}s (client {s.job.client_pid})" for s in c.blender]}
            if d["lock"] == "outside: ":
                d["lock"] = "free"
            e = c.editor
            if e is not None:
                d["editor"] = (f"pid {getattr(e, 'upid', None) or e.pid} {e.state} ({'tenant' if e.primary else 'second'}), up {t - e.t_start:.0f}s, "
                               f"{e.n_jobs} jobs" + (f", now {e.job.job.id} {t - e.job.t_start:.0f}s" if e.job else "")
                               + (f", idle {t - e.idle_since:.0f}s" if e.state == "ready" and e.idle_since else ""))
            out["devices"].append(d)
        g = self.igpu
        hold = lock_holders(g.lock_path, True)
        out["devices"].append({"device": f"igpu {g.bus}", "busy": busy_pct(g.bus),
                               "lock": "ours" if g.fd is not None else (
                                   ("outside: " + "; ".join(f"{p} {cmd}" for p, cmd in hold)) if hold else "free"),
                               "blender": [f"{s.job.id} {s.name}" for s in g.blender]})
        for j in self.jobs:
            for s in j.stages:
                if s.state in ("ready", "run", "wait"):
                    out["queue"].append({"job": j.n, "seq": j.seq, "id": j.id, "mode": j.mode, "client": j.client_pid,
                                         "stage": s.name, "state": s.state, "device": s.device,
                                         "since_s": round(t - (s.t_start or s.t_ready or j.t_submit), 1)})
        out["line"] = self.line_status(t)
        return out

    def line_status(self, t):
        """The one-screen view: per station what runs and how long, what waits and the oldest wait; the guard."""
        head = self.heading()
        names = ["igpu", f"card{self.bl_bus}", f"card{self.ue_bus}", self.borrow_line(), "video"]
        stations = {b: {"running": [], "waiting": head.get(b, 0), "oldest_wait_s": 0.0,
                        "batch": self.batcher.state().get(b.split(":")[0]), "why": []} for b in names}
        busy = set()
        notes = collections.defaultdict(set)
        for j in self.jobs:
            todo = [st for st in j.stages if self.station(st) and st.state in ("wait", "ready", "run")]
            for st in j.stages:
                b = self.line_of(st)
                if b and st.state == "run":
                    busy.add(self.station(st))
                    stations[b]["running"].append(f"#{j.seq} {j.id} {self.skind(st)} {t - st.t_start:.0f}s")
            if todo and todo[0].state != "run":
                b = self.line_of(todo[0])
                stations[b]["oldest_wait_s"] = max(stations[b]["oldest_wait_s"], round(t - j.t_submit))
                if todo[0].state == "wait":
                    notes[b].add("an entry waits for its earlier stage" if not j.cancelled else "cancelling")
            elif todo:
                nxt = next((x for x in todo[1:] if self.line_of(x) != self.line_of(todo[0]) and x.state != "run"), None)
                if nxt is not None:
                    notes[self.line_of(nxt)].add(f"next after #{j.seq} {todo[0].name} at {self.station(todo[0])}")
        for b, d in stations.items():
            # why a line with waiting entries starts none: the holds of the last dispatch, else what it waits for
            if d["waiting"] and b.split(":")[0] not in busy:
                d["why"] = sorted(set(self.why.get(b, []))) or sorted(notes.get(b, ())) or ["(no hold recorded)"]
            if d["waiting"] and b.split(":")[0] in busy and not d["running"]:
                other = [x for x in stations if x != b and x.split(":")[0] == b.split(":")[0] and stations[x]["running"]]
                d["why"] = [f"the card is busy with {', '.join(other)}"] + sorted(set(self.why.get(b, [])))
            idle = self.idle_since.get(b.split(":")[0])
            d["idle_s"] = round(t - idle) if idle and b.split(":")[0] not in busy else 0
        try:
            hour = self.wl.recent(t - 3600)
        except OSError:
            hour = []
        cost = {k: {"n": len(v), "s_mean": round(sum(a for a, _ in v) / len(v), 1),
                    "wh_mean": (round(sum(d for _, d in v if d is not None) / max(1, sum(1 for _, d in v if d is not None)), 3))}
                for k, v in self.kind_cost.items() if v}
        return {"stations": stations, "line_max": L.LINE_MAX, "batch": L.BATCH,
                "guard": self.verdict, "power": self.guard.last,
                "power_events": [(time.strftime("%H:%M:%S", time.localtime(a)), b) for a, b in self.guard.events],
                "load": [round(x, 2) for x in os.getloadavg()], "peak_load": round(self.peak_load, 2),
                "mem_avail_gb": round(R.mem_avail_gb(), 1),
                "inhibitor": self.inhibit is not None and self.inhibit.poll() is None,
                "window": self.window if self.released else None,
                "done_last_hour": {"n": len(hour), "done": sum(1 for e in hour if e.get("state") == "done"),
                                   "failed": [f"#{e['seq']} {e['id']}" for e in hour if e.get("state") == "failed"]},
                "cost_by_kind": cost}


# ---------------------------------------------------------------- socket server
def resume(S):
    """Starting the line again: every unfinished entry goes back on the line in list order. A command that ended
    while no line watched it is finished from its runner's record (never run again); one that still runs is
    adopted (watched to its end); anything else that was in flight runs again (a process of ours still running from
    before the restart is stopped first; an orphan editor's holder is not an outside user of the card while it
    quits). A process is ours only if its pid and start time match the entry (pids are reused)."""
    for p in R.server_dir().glob("ed-*/editor.json"):
        try:
            pid = json.loads(p.read_text())["pid"]
            if Path(f"/proc/{pid}").exists():
                S.child_sids.add(pid)
        except (OSError, ValueError, KeyError):
            pass
    entries = S.wl.load()
    Job.N = max([Job.N] + [int(e.get("n") or 0) for e in entries])   # job numbers go on across a reload
    for e in entries:
        pids, starts = e.pop("pids", None) or {}, e.pop("pid_start", None) or {}
        rcf = e.get("rc_files") or {}
        for name in list(e.get("resumed") or []):
            rec = L.read_rc(rcf[name]) if rcf.get(name) else None
            pid = pids.get(name)
            if rec is not None:
                e.setdefault("ended", {})[name] = rec.get("rc")
                continue
            if pid and starts.get(name):
                if proc_start(pid) != starts[name]:
                    continue                                   # gone (or the pid now names another process)
                if e["type"] == "cmd" and rcf.get(name):
                    e.setdefault("adopt", {})[name] = {"pid": pid, "pid_start": starts[name], "rc_file": rcf[name]}
                    print(f"[line] entry {e['seq']} {e['id']}: {name} still runs (pid {pid}); adopting it", flush=True)
                    continue
            elif pid:                                          # written by older code: no start time recorded
                try:
                    cmdl = Path(f"/proc/{pid}/cmdline").read_bytes()
                except OSError:
                    continue
                if not (sid_of(pid) == pid and (b"blender" in cmdl or e["type"] == "cmd")):
                    continue
            else:
                continue
            print(f"[line] stopping {pid} left running by the previous scheduler ({e['id']} {name})", flush=True)
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
        e["resumed"] = [n for n in (e.get("resumed") or []) if n not in (e.get("ended") or {})
                        and n not in (e.get("adopt") or {})]
        try:
            job, _ = S.add_entry(e, None, resume=True)
            print(f"[line] resumed entry {e['seq']} {e['id']} ({e['type']}"
                  + (f"; re-running {', '.join(e['resumed'])}" if e.get("resumed") else "") + ")", flush=True)
        except Exception as ex:                                # a broken entry must not stop the line
            e["state"], e["error"] = "failed", f"could not resume: {ex}"
            S.wl.save(e)
            print(f"[line] entry {e.get('seq')} could not resume: {ex}", flush=True)
    if entries:
        print(f"[line] {len(entries)} entries back on the line", flush=True)


def serve():
    C.SCHED_DIR.mkdir(parents=True, exist_ok=True)
    # a reload re-executes this process: it hands over the daemon lock and the listening socket (fds kept open)
    lock_env, listen_env = os.environ.pop("ASSETS_LINE_LOCK_FD", None), os.environ.pop("ASSETS_LINE_LISTEN_FD", None)
    reloaded = os.environ.pop("ASSETS_LINE_RELOADED", None)
    me = None
    if lock_env:
        me = int(lock_env)
        os.set_inheritable(me, False)
        try:
            fcntl.flock(me, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the same open file: still ours
        except OSError:
            me = None
    if me is None:
        me = try_flock(C.SCHED_DIR / "daemon.lock")
    if me is None:
        print("another scheduler is running", flush=True)
        return 0
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(SystemExit(0)))
    global SCHED
    S = SCHED = Sched()
    # an editor left by an earlier scheduler (or the pre-scheduler chain) is asked to quit
    for p in R.server_dir().glob("ed-*/editor.json"):
        try:
            pid = json.loads(p.read_text())["pid"]
            if "UnrealEditor" in Path(f"/proc/{pid}/cmdline").read_text(errors="replace"):
                (p.parent / "queue").mkdir(parents=True, exist_ok=True)
                (p.parent / "queue" / "quit").touch()
                print(f"[sched] asked orphan editor {pid} to quit", flush=True)
        except (OSError, ValueError, KeyError):
            pass
    if R.live_server():
        (R.server_dir() / "queue" / "quit").touch()
        print("[sched] asked the pre-scheduler warm editor to quit", flush=True)
    resume(S)

    def sampler():
        buses = list(S.cards) + [C.IGPU_BUS]
        while True:
            S.util.append((now(), {b: busy_pct(b) for b in buses}))
            time.sleep(0.5)

    threading.Thread(target=sampler, daemon=True).start()
    if listen_env:
        srv = socket.socket(fileno=int(listen_env))
        os.set_inheritable(srv.fileno(), False)
    else:
        SOCK.unlink(missing_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(SOCK))
        os.chmod(SOCK, 0o600)
        srv.listen(64)
    S.srv, S.lock_fd = srv, me
    how = ""
    if reloaded:
        try:
            r = json.loads(reloaded)
            how = f"; reloaded from {r['from']}, {now() - r['t']:.0f}s after the newer code appeared"
        except (ValueError, KeyError):
            pass
    print(f"[sched] pid {os.getpid()} serving on {SOCK} (code {S.code}{how})", flush=True)

    def accept():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(S, conn), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    try:
        S.loop()
    finally:
        with S.lock:
            try:
                SOCK.unlink(missing_ok=True)
            except OSError:
                pass
            srv.close()
        for c in S.cards.values():
            c.tenant_path.unlink(missing_ok=True)
        print(f"[sched] exiting after {now() - S.t0:.0f}s", flush=True)
    return 0


def station_device(b):
    """The device a tool at station b uses: "igpu" or the card's bus ("05")."""
    return b[4:] if b.startswith("card") else b


STATION_ALIASES = {"igpu": "igpu", "video": "video", "blender": f"card{C.BLENDER_CARD}",
                   C.BLENDER_CARD: f"card{C.BLENDER_CARD}", f"card{C.BLENDER_CARD}": f"card{C.BLENDER_CARD}"}
STATION_ALIASES["image"] = C.IMAGE_STATION         # the image model's station (config ASSETS_IMAGE_STATION)
# a command may borrow the engine card while the engine has no work (Sched.lend_engine_card)
STATION_ALIASES.update({"engine": f"card{C.UE_CARD}", C.UE_CARD: f"card{C.UE_CARD}",
                        f"card{C.UE_CARD}": f"card{C.UE_CARD}"})


def entry_from_msg(S, msg, cmd, conn):
    """The work-list entry for a submission (None after an answer was sent)."""
    def no(text):
        conn.sendall((json.dumps({"result": {"rc": 2, "error": text}}) + "\n").encode())

    if cmd == "ue_job":
        jd = msg["ue_job"]
        for k in ("project", "script"):
            if not jd.get(k) or not Path(jd[k]).exists():
                no(f"ue job: '{k}' must name an existing file ({jd.get(k)})")
                return None
        return {"type": "ue_job", "id": jd.get("id") or Path(jd["script"]).stem, "ue": jd,
                "timeout_s": jd.get("timeout_s"), "client_pid": msg.get("pid")}
    if cmd == "cmd":
        b = STATION_ALIASES.get(str(msg.get("station")))
        if b is None:
            no(f"station must be one of igpu, blender ({C.BLENDER_CARD}), image ({C.IMAGE_STATION}), engine "
               f"({C.UE_CARD}: borrowed while the engine has no work) or video; engine work goes through "
               f"--ue-job (got {msg.get('station')})")
            return None
        if not msg.get("argv"):
            no("no command given")
            return None
        return {"type": "cmd", "id": msg.get("id") or Path(msg["argv"][0]).name, "station": b,
                "kind": msg.get("kind") or "cmd", "cmd": msg["argv"], "cwd": msg.get("cwd"),
                "env": msg.get("env_extra") or {}, "log": msg.get("log"), "client_pid": msg.get("pid")}
    if msg.get("env") != R.sched_env():
        conn.sendall((json.dumps({"local": "the scheduler runs with other ASSETS_* settings"}) + "\n").encode())
        return None
    if msg.get("code") != S.code:
        S.t_code = 0.0                                        # look at the code on disk now (it reloads if newer)
    args = R.build_parser().parse_args(msg["argv"])
    spec_p = Path(msg["cwd"], args.spec).resolve()
    model_p = Path(msg["cwd"], args.model).resolve()
    spec = json.loads(spec_p.read_text())
    R.validate_spec(spec)
    adir = C.LIBRARY / spec["class"] / spec["id"]
    if args.ue_only:
        try:
            ok = json.loads((adir / "blender_report.json").read_text()).get("status") == "ok"
        except (OSError, ValueError):
            ok = False
        if not ok:
            no(f"--ue-only: {adir / 'blender_report.json'} is missing or not ok; bake first")
            return None
    for d in ("source", "logs", "ue"):
        (adir / d).mkdir(parents=True, exist_ok=True)
    dst_m = adir / "source" / model_p.name
    if dst_m.resolve() != model_p:
        shutil.copy(model_p, dst_m)
    (adir / "source" / "spec.json").write_text(json.dumps(spec, indent=1))
    # the model runs from where it was given (model scripts import their kits from beside their folder)
    return {"type": "asset", "id": spec["id"], "argv": msg["argv"], "spec": str(adir / "source" / "spec.json"),
            "model": str(model_p), "client_pid": msg.get("pid"),
            "kind": "preview" if args.preview else ("ue_only" if args.ue_only else "final")}


def handle(S, conn):
    f = conn.makefile("rb")
    try:
        line = f.readline()
        if not line:
            return
        msg = json.loads(line)
        cmd = msg.get("cmd")
        if cmd == "status":
            with S.lock:
                st = S.status()
            conn.sendall((json.dumps(st, default=str) + "\n").encode())
            return
        if cmd == "stop_editors":
            with S.lock:
                for e in S.editors:
                    if e.state in ("ready", "starting"):
                        e.quit()
                    elif e.state == "busy":
                        e.primary = False                    # quits after this job (short idle)
                n = len([e for e in S.editors if e.state != "dead"])
            conn.sendall((json.dumps({"ok": True, "editors": n}) + "\n").encode())
            return
        if cmd == "release_card":
            with S.lock:
                c = S.cards.get(msg.get("bus"))
                if c is None:
                    rep = {"ok": False, "error": f"not one of the scheduler's cards: {list(S.cards)}"}
                elif c.role == "ue":
                    rep = {"ok": False, "error": f"card {c.bus} is the UE editor's home; take the Blender card "
                                                 f"{S.bl_bus} (or stop the editor with --stop-editor)"}
                else:
                    c.draining = True
                    S.wake.notify_all()
                    rep = {"ok": True, "running": [f"{s.job.id} {s.name}" for s in c.blender],
                           "grace_s": C.RELEASE_GRACE_S}
            conn.sendall((json.dumps(rep) + "\n").encode())
            return
        if cmd in ("release_cards", "reclaim_cards"):
            with S.lock:
                if cmd == "release_cards":
                    S.release(f"asked by client pid {msg.get('pid')}", True)
                else:
                    if S.released:
                        S.reclaim(f"asked by client pid {msg.get('pid')}")
                    else:
                        S.warm_wanted = True
                S.wake.notify_all()
            conn.sendall(b'{"ok": true}\n')
            return
        if cmd == "cancel":
            want = str(msg.get("job"))
            with S.lock:
                hit = [j for j in S.jobs if str(j.n) == want or j.id == want]
                for j in hit:
                    S.cancel(j, f"cancelled by client pid {msg.get('pid')} (--cancel)")
            conn.sendall((json.dumps({"ok": bool(hit), "jobs": [f"{j.n} {j.id}" for j in hit]}) + "\n").encode())
            return
        if cmd == "drain":
            with S.lock:
                S.drain = True
            conn.sendall(b'{"ok": true}\n')
            return
        if cmd == "watch":                                    # a client re-attaching to its entry (after a reload)
            seq = int(msg.get("seq"))
            with S.lock:
                job = next((j for j in S.jobs if j.seq == seq), None)
                if job is not None:
                    job.conn = conn
            if job is None:
                e = S.wl.get(seq) or {}
                res = e.get("result") or ({"rc": 0} if e.get("state") == "done" else
                                          {"rc": 1, "error": f"line entry {seq}: {e.get('state') or 'not found'}"})
                conn.sendall((json.dumps({"result": res}, default=str) + "\n").encode())
                return
            conn.sendall((json.dumps({"watching": seq}) + "\n").encode())
            while f.readline():
                pass
            with S.lock:
                if job.conn is conn:
                    job.conn = None
            return
        if cmd in ("ue_job", "cmd", "submit"):
            with S.lock:
                if S.stopping or S.should_exit():
                    conn.sendall(b'{"retry": true}\n')
                    return
            e = entry_from_msg(S, msg, cmd, conn)
            if e is None:
                return
            job, why = S.add_entry(e, conn)
            if why and why.startswith("already queued as entry "):
                seq = int(why.split()[4])
                conn.sendall((json.dumps({"duplicate": why, "seq": seq}) + "\n").encode())
                print(f"[line] {time.strftime('%H:%M:%S')} refused {e.get('id')}: {why}", flush=True)
                return
            if why:
                conn.sendall((json.dumps({"line_full": why}) + "\n").encode())
                print(f"[line] {time.strftime('%H:%M:%S')} refused {e.get('id')}: {why}", flush=True)
                return
            job.log(f"{job.id}: line entry {job.seq} (job {job.n}) queued with the scheduler (pid {os.getpid()})")
            conn.sendall((json.dumps({"queued": job.seq, "job": job.n}) + "\n").encode())
            if msg.get("detach"):
                job.conn = None
                return
            while True:                                       # EOF = the client went away; the job stays on the line
                if not f.readline():
                    break
            job.conn = None
            return
        conn.sendall(b'{"error": "unknown command"}\n')
    except SystemExit as e:
        try:
            conn.sendall((json.dumps({"result": {"rc": 2, "error": str(e)}}) + "\n").encode())
        except OSError:
            pass
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        try:
            conn.sendall((json.dumps({"result": {"rc": 1, "error": err.strip().splitlines()[-1]}}) + "\n").encode())
        except OSError:
            pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


# ---------------------------------------------------------------- client side
def connect(start=True):
    """A connected socket to the scheduler; starts one (detached) when none answers."""
    C.SCHED_DIR.mkdir(parents=True, exist_ok=True)

    def try_conn():
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(str(SOCK))
            return s
        except OSError:
            s.close()
            return None

    s = try_conn()
    if s or not start:
        return s
    t0 = now()
    while now() - t0 < 60:
        fd = try_flock(C.SCHED_DIR / "start.lock")
        if fd is None:
            time.sleep(0.2)
            s = try_conn()
            if s:
                return s
            continue
        try:
            s = try_conn()
            if s:
                return s
            C.LOGS.mkdir(parents=True, exist_ok=True)
            logf = open(C.LOGS / f"assets_sched_{time.strftime('%m%d')}.log", "a")
            subprocess.Popen([sys.executable, str(HERE / "assetd.py"), "--serve"], stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(HERE))
            logf.close()
            t1 = now()
            while now() - t1 < 20:
                s = try_conn()
                if s:
                    return s
                time.sleep(0.1)
        finally:
            os.close(fd)
    raise SystemExit("could not reach or start the asset scheduler; see its log in ASSETS_LOGS")


LINE_FULL_RC = 75                                         # EX_TEMPFAIL: the line is full, retry later
DUPLICATE_RC = 76                                         # the same job is already on the line (not queued again)


def _client(msg, on_result, detach=False):
    """Send one submission; relay its log. Returns an exit code (None = run locally). When the connection drops
    before the result (the line reloading onto newer code, or a restart), the client re-attaches to its entry and
    keeps relaying; the entry itself is on disk and runs either way."""
    seq, lost, retries = None, 0.0, 0
    while True:
        s = connect()
        s.sendall((json.dumps(msg if seq is None else {"cmd": "watch", "seq": seq, "pid": os.getpid()}) + "\n")
                  .encode())
        retry = False
        for line in s.makefile("rb"):
            m = json.loads(line)
            if "log" in m:
                print(m["log"], flush=True)
            elif m.get("retry"):
                retry = True
                break
            elif "line_full" in m:
                print(f"LINE FULL: {m['line_full']}", file=sys.stderr, flush=True)
                s.close()
                return LINE_FULL_RC
            elif "duplicate" in m:                          # the same job is on the line already: not queued again
                print(f"ALREADY QUEUED: {m['duplicate']}", file=sys.stderr, flush=True)
                s.close()
                if detach:
                    return DUPLICATE_RC
                seq = m["seq"]                              # an attached client follows the entry that is there
                break
            elif "queued" in m:
                seq = m["queued"]
                if detach:
                    print(f"queued: line entry {m['queued']} (see run_asset.py --status)", flush=True)
                    s.close()
                    return 0
            elif "watching" in m:
                lost = 0.0
            elif "local" in m:
                if os.environ.get("ASSETS_LINE_SEQ"):       # inside a line entry: never a private run on a card
                    print(f"scheduler: {m['local']}; refused: this runs inside line entry "
                          f"{os.environ['ASSETS_LINE_SEQ']} and must not start its own GPU work", file=sys.stderr,
                          flush=True)
                    s.close()
                    return 3
                print(f"scheduler: {m['local']}; running locally", flush=True)
                return None
            elif "result" in m:
                return on_result(m["result"])
        s.close()
        if retry and seq is None and retries < 5:
            retries += 1
            time.sleep(0.5)
            continue
        if seq is not None:                                 # the entry is on the line: re-attach to it
            lost = lost or now()
            if now() - lost < 300:
                time.sleep(0.5)
                continue
        print("the line went away before the job finished; the job stays on the work list and runs when the "
              "line starts again (run_asset.py --status)", file=sys.stderr)
        return 4


def _plain_result(r):
    if r.get("error") and r.get("rc"):
        print(r["error"], file=sys.stderr, flush=True)
    return int(r.get("rc", 1))


def client_submit(argv):
    """Send this run to the line and relay its log; returns the exit code (None = run locally instead).
    --detach: return as soon as the entry is on the line. A full line answers LINE FULL (exit 75): stop submitting."""
    detach = "--detach" in argv
    msg = {"cmd": "submit", "argv": argv, "cwd": os.getcwd(), "pid": os.getpid(), "env": R.sched_env(),
           "code": code_hash(), "detach": detach}
    return _client(msg, _plain_result, detach)


def client_ue_job(path, detach=False):
    """run_asset.py --ue-job JOB.json: run an editor-side script in the scheduler's resident editor of its project.
    JOB: {"project": .uproject, "script": editor-side .py, "job": {...} (the script's JOB), "outputs": [paths to wait
    for], "timeout_s": 1800, "id": name, "log_dir": folder for sched.json / ue_result.json}. Prints the result."""
    jd = json.loads(Path(path).read_text())
    for k in ("project", "script"):
        if jd.get(k):
            jd[k] = str(Path(jd[k]).resolve())
    jd["outputs"] = [str(Path(o).resolve()) for o in jd.get("outputs", [])]

    def res(r):
        r.pop("sched", None)
        print(json.dumps(r, default=str), flush=True)
        return int(r.get("rc", 1))
    return _client({"cmd": "ue_job", "ue_job": jd, "pid": os.getpid(), "detach": detach}, res, detach)


def client_line_cmd(argv):
    """run_asset.py --line-cmd STATION KIND [--id NAME] [--log FILE] [--detach] -- COMMAND...: run an existing tool
    at a station (igpu, blender, image, engine, video; engine borrows the engine card while it has no work: the
    editor unloads first and loads again for the next engine job). The tool takes the station's lock itself, as it does today; the line
    starts it only when the station is free and the guard allows. KIND names the batch (e.g. refs, paint, cloth,
    shape, video): a station finishes a batch of one kind before it switches."""
    if "--" not in argv or len(argv) < 4:
        raise SystemExit("usage: run_asset.py --line-cmd STATION KIND [--id NAME] [--log FILE] [--detach] -- CMD...")
    k = argv.index("--")
    head, cmd = argv[:k], argv[k + 1:]
    station, kind, opts = head[0], head[1], head[2:]
    detach = "--detach" in opts
    get = lambda o: opts[opts.index(o) + 1] if o in opts else None
    msg = {"cmd": "cmd", "station": station, "kind": kind, "argv": cmd, "cwd": os.getcwd(), "pid": os.getpid(),
           "id": get("--id"), "log": str(Path(get("--log")).resolve()) if get("--log") else None, "detach": detach}
    return _client(msg, _plain_result, detach)


def feed(list_path):
    """run_asset.py --feed LIST: one submitter for a backlog. LIST is a text file, one run_asset.py argument line
    per entry (e.g. "SPEC MODEL --ue-only", or "--line-cmd igpu refs --id X -- CMD..."), in order. Each line is submitted detached; on LINE FULL the feeder waits
    until the line has room (it watches the status, it does not resubmit blindly). Progress is kept beside the
    list (<LIST>.fed), so running it again skips what was already put on the line."""
    import shlex
    lp = Path(list_path)
    fed_p = lp.with_name(lp.name + ".fed")
    fed = set(fed_p.read_text().splitlines()) if fed_p.exists() else set()
    lines = [ln.strip() for ln in lp.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines:
        if ln in fed:
            continue
        t0 = now()                                       # leave one place in each line for other builders
        while now() - t0 < 600:
            ln_st = (client_cmd("status") or {}).get("line") or {}
            if not ln_st or all(v["waiting"] < ln_st["line_max"] - 1 for v in ln_st["stations"].values()):
                break
            time.sleep(2)
        while True:
            argv = shlex.split(ln)
            rc = (client_line_cmd(argv[1:k] + ["--detach"] + argv[k:]) if argv[0] == "--line-cmd" and
                  (k := argv.index("--")) else client_submit(argv + ["--detach"]))
            if rc == DUPLICATE_RC:                       # already on the line: count it as fed
                rc = 0
            if rc != LINE_FULL_RC:
                break
            t0 = now()
            while now() - t0 < 600:                      # wait for room: the station's line shrinks as jobs finish
                st = client_cmd("status") or {}
                ln_st = (st.get("line") or {})
                if not ln_st:
                    break
                if all(v["waiting"] < ln_st["line_max"] for v in ln_st["stations"].values()):
                    break
                time.sleep(2)
        if rc != 0:
            print(f"feed: '{ln}' failed to queue (rc {rc}); stopping", file=sys.stderr)
            return rc
        fed.add(ln)
        with open(fed_p, "a") as f:
            f.write(ln + "\n")
    print(f"feed: all {len(lines)} entries are on the line")
    return 0


def client_cmd(cmd, **kw):
    s = connect(start=False)
    if s is None:
        return None
    s.sendall((json.dumps({"cmd": cmd, **kw}) + "\n").encode())
    line = s.makefile("rb").readline()
    s.close()
    return json.loads(line) if line else None


def cancel_job(which):
    """run_asset.py --cancel N|ID: drop its queued stages; a running Blender is stopped, a running editor script is
    told to stop (at its next tick), a running asset import/render finishes first."""
    r = client_cmd("cancel", job=which, pid=os.getpid())
    if r is None:
        print("no scheduler running")
        return 1
    print(f"cancelled: {r['jobs']}" if r.get("ok") else f"no queued or running job {which}")
    return 0 if r.get("ok") else 1


def release_cards():
    """Asset work makes way for a video run: in-flight stages finish, the queue holds, the editor stops, both
    card locks are freed. Waits until both cards are free and prints how long that took."""
    t0 = now()
    if client_cmd("release_cards", pid=os.getpid()) is None:
        print("no scheduler running: the asset chain holds no card")
        return 0
    while True:
        st = client_cmd("status")
        r = (st or {}).get("released") or {}
        if st is None or r.get("done_t"):
            break
        time.sleep(0.5)
    vr = {b: f"{vram(b)[0]:.1f} GB used" for b in (C.UE_CARD, C.BLENDER_CARD)}
    print(f"both cards released in {now() - t0:.1f}s ({vr}); queued asset jobs wait. Take /tmp/gpu-{C.UE_CARD}.lock "
          f"and /tmp/gpu-{C.BLENDER_CARD}.lock with flock -n now; asset work resumes by itself once both locks have "
          f"been free for {C.RECLAIM_FREE_S:.0f}s (and {C.RELEASE_GRACE_S:.0f}s have passed), or at --reclaim-cards")
    return 0


def reclaim_cards():
    """Back to asset mode: reload the editor on its card and resume the queue; prints how long the reload took."""
    t0 = now()
    if client_cmd("reclaim_cards", pid=os.getpid()) is None:
        connect()                                      # start a scheduler: it starts in asset mode
        client_cmd("reclaim_cards", pid=os.getpid())
    while True:
        st = client_cmd("status") or {}
        if st.get("released"):
            print("released: " + st["released"].get("reason", ""))
        ed = next((d.get("editor") for d in st.get("devices", []) if d.get("editor")), "")
        if " ready " in f" {ed} " or " busy " in f" {ed} ":
            break
        if any(f"lock: outside" in d.get("lock", "") for d in st.get("devices", []) if "card" in d["device"]):
            print("an outside job still holds a card: the editor starts when its lock is free")
        time.sleep(1.0)
    print(f"cards reclaimed; editor ready after {now() - t0:.1f}s")
    return 0


def release_card(bus):
    """Ask the scheduler to let go of a card for an outside heavy job; it waits for ours to finish there."""
    r = client_cmd("release_card", bus=bus)
    if r is None:
        print(f"no scheduler running: card {bus} is not held by the asset chain")
        return 0
    if not r.get("ok"):
        print(r.get("error"))
        return 1
    print(f"card {bus}: draining ({len(r['running'])} job(s) still running: {r['running'] or 'none'}); the lock "
          f"/tmp/gpu-{bus}.lock frees when they end; the scheduler retakes it after {r['grace_s']:.0f}s once free")
    return 0


def print_status():
    st = client_cmd("status")
    if st is None:
        wl = L.WorkList(C.SCHED_DIR / "line")
        n = len(list(wl.edir.glob("*.json")))
        print(f"the line is not running; {n} unfinished entries on disk (they resume when it starts: "
              f"run_asset.py --start)")
        for name in [*C.GPU_CARDS, C.IGPU_LOCK_NAME]:
            h = lock_holders(f"/tmp/gpu-{name}.lock", True)
            print(f"  lock gpu-{name}: " + ("; ".join(f"{p} {c}" for p, c in h) if h else "free"))
        return 0
    ln = st["line"]
    pw, g = ln["power"] or {}, ln["guard"] or {}
    print(f"asset line pid {st['pid']} up {st['up_s']}s code {st.get('code')}"
          f"{' (exits when idle: drain asked)' if st['drain'] else ''} | "
          f"line max {ln['line_max']} per line, batch {ln['batch']}")
    if st.get("reload"):
        r = st["reload"]
        print(f"RELOADING onto newer code {r.get('code')} (seen {now() - r['t']:.0f}s ago): nothing new starts until "
              f"these finish: {', '.join(r.get('waiting_for') or []) or 'none (unloading the editor)'}")
    print(f"power: battery {pw.get('capacity')}% {pw.get('status')} {pw.get('energy_wh')} Wh, "
          f"charger {'on' if pw.get('ac') else 'OFF'} (USB-C {pw.get('usb_online')}/{pw.get('usb_n')}), "
          f"ASPM {(pw.get('aspm') or '?').split('[')[-1].split(']')[0]} | load {ln['load']} (peak {ln['peak_load']}) | "
          f"mem free {ln['mem_avail_gb']} GiB | awake lock {'held' if ln['inhibitor'] else 'off'}")
    if g.get("why"):
        print(("PAUSED: " if g.get("pause") else "HELD: ") + "; ".join(g["why"]))
    for a, b in ln.get("power_events") or []:
        print(f"  power event {a}: {b}")
    if st.get("released"):
        r = st["released"]
        print(f"cards released: {r.get('reason')}" + (f" (both free after {r.get('release_s')}s)" if r.get("done_t") else ""))
    devs = {d["device"].split(" (")[0]: d for d in st["devices"]}
    for b, s in ln["stations"].items():
        d = devs.get(f"card {b[4:]}") if b.startswith("card") else devs.get(f"igpu {C.IGPU_BUS}") if b == "igpu" else None
        extra = ""
        if d:
            extra = f"  [busy {d.get('busy')}%{' vram ' + d['vram_gb'] if d.get('vram_gb') else ''}, lock {d['lock'][:60]}]"
        bt = s.get("batch") or {}
        print(f"{b:13} run: {', '.join(s['running']) or 'idle':55} wait {s['waiting']}"
              f"{' (oldest ' + str(int(s['oldest_wait_s'])) + 's)' if s['waiting'] else ''}"
              f"{' batch ' + str(bt.get('kind')) + ' x' + str(bt.get('n')) if bt and ':' not in b else ''}{extra}")
        if d and d.get("editor") and ":" not in b:
            print(f"              editor {d['editor']}")
        if s.get("why"):
            print(f"              why not running" + (f" (idle {s['idle_s']}s)" if s.get("idle_s") else "")
                  + ": " + "; ".join(s["why"]))
    h = ln["done_last_hour"]
    print(f"last hour: {h['done']} done of {h['n']} finished" + (f"; failed {h['failed']}" if h["failed"] else ""))
    if ln.get("cost_by_kind"):
        print("cost by kind: " + "; ".join(f"{k} n{v['n']} {v['s_mean']}s {v['wh_mean']:+} Wh"
                                           for k, v in ln["cost_by_kind"].items()))
    return 0


def start_line():
    """Start the line (it resumes the work list on disk) without submitting anything."""
    s = connect()
    s.close()
    return print_status()


def hold_run(cmd):
    """Run cmd while this process keeps the lock fds it inherited; cmd gets none of them. SIGTERM is passed on."""
    p = subprocess.Popen(cmd, close_fds=True)
    signal.signal(signal.SIGTERM, lambda *a: p.terminate())
    signal.signal(signal.SIGINT, lambda *a: p.terminate())
    while True:
        try:
            return p.wait()
        except InterruptedError:
            continue


if __name__ == "__main__":
    if sys.argv[1:3] == ["--hold", "--"]:
        sys.exit(hold_run(sys.argv[3:]))
    if sys.argv[1:] == ["--serve"]:
        sys.exit(serve())
    if sys.argv[1:] == ["--status"]:
        sys.exit(print_status())
    print(__doc__)
