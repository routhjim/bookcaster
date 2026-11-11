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
import run_asset as R  # noqa: E402

SOCK = C.SCHED_DIR / "sched.sock"
CODE_FILES = ("assetd.py", "run_asset.py", "config.py")
UE_FILES = ("ue/ue_render.py", "ue/ue_import.py", "ue/init_unreal.py")


def code_hash(files=CODE_FILES):
    h = hashlib.sha1()
    for f in files:
        try:
            h.update((HERE / f).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


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
            if self.draining or now() < self.released_until:
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

    def log(self, msg):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
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
        self.drain = False                # exit when idle (code changed)
        self.project_fd = None
        self.last_busy = now()
        self.util = collections.deque(maxlen=7200)        # (t, {bus: busy%}) at 2 Hz
        self.est = self.load_est()
        self.mannequin = None
        self.stopping = False

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
    def submit(self, job):
        a = job.args
        s = []
        if job.mode == "ue_job":
            s.append(Stage(job, "ue_script", "ue"))
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
        with self.lock:
            self.jobs.append(job)
            self.last_busy = now()
            self.wake.notify_all()

    def cancel(self, job, why):
        with self.lock:
            if job.state in ("done", "failed", "cancelled"):
                return
            job.cancelled = True
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
        return [(p, cmd) for p, cmd in lock_holders(c.lock_path) if p != os.getpid()]

    def card_work_queued(self):
        for j in self.jobs:
            for s in j.stages:
                if s.state in ("wait", "ready") and (s.kind == "ue" or (s.kind == "blender" and j.mode == "final")):
                    return True
        return False

    def release(self, reason, manual):
        if self.released:
            return
        self.released = {"reason": reason, "manual": manual, "t": now(), "done_t": None}
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
                if t - since >= C.OUTSIDE_RELEASE_S and (ours or self.card_work_queued()):
                    who = "; ".join(f"{p} {cmd[:60]}" for p, cmd in self.outside_holders(self.cards[bus]))
                    self.release(f"an outside job holds card {bus} ({who or '?'}) for {t - since:.0f}s", False)
                    break
            return
        r = self.released
        if r["done_t"] is None and not any(e.state != "dead" for e in self.editors) and all(
                c.fd is None for c in self.cards.values()):
            r["done_t"] = t
            r["release_s"] = round(t - r["t"], 1)
            r["vram_used_gb"] = {b: round(vram(b)[0] or 0, 1) for b in self.cards}
            print(f"[sched] both cards released in {r['release_s']}s; VRAM used {r['vram_used_gb']}", flush=True)
        if r["done_t"] is None or not self.card_work_queued():
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
        ready = [st for j in self.jobs if not j.cancelled for st in j.stages if st.state == "ready"]
        ready.sort(key=lambda s: (s.job.n, s.t_ready))
        for st in ready:
            if st.kind == "blender":
                pl = self.place_blender(st)
                if pl:
                    self.launch(st, self.run_blender, pl)
            elif st.kind == "ue":
                ed = self.place_ue(st)
                if ed:
                    self.launch(st, self.run_ue, ed)
            elif st.kind == "paint":
                self.launch(st, self.run_paint, None)
            elif st.kind == "cpu":
                self.launch(st, self.run_finalize, None)
        self.manage_editors()
        # the Blender card stays ours while Blender work is queued, and BLENDER_LINGER_S after it (no thrash)
        bl_queued = any(st.kind == "blender" and st.state in ("wait", "ready") for j in self.jobs for st in j.stages)
        for c in self.cards.values():
            if c.role == "blender" and bl_queued and c.fd is not None and not c.draining:
                c.last_used = t
            self.keep_spares(c, t)
            c.release_if_unused(C.BLENDER_LINGER_S if c.role == "blender" and not self.released else 0.0)
        self.igpu.release_if_unused()
        if self.jobs or any(e.state != "dead" for e in self.editors):
            self.last_busy = t
        self.write_tenancy()

    def keep_spares(self, c, t):
        """Warm Blender spares on the Blender card while it is ours and in use (within its linger)."""
        for sp in c.spares:
            if sp.proc.poll() is not None:
                shutil.rmtree(sp.dir, ignore_errors=True)
        c.spares = [sp for sp in c.spares if sp.proc.poll() is None]
        want = 0
        if (c.role == "blender" and c.fd is not None and not c.draining and not self.released
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
                    self.after_stage(st)
                    self.wake.notify_all()

        threading.Thread(target=go, daemon=True).start()

    def after_stage(self, st):
        """A stage ended: report the job when nothing of it runs any more (a failure cancels what is queued)."""
        job = st.job
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
            self.finish_job(job, "cancelled", {"rc": 130, "error": "cancelled (client gone)"})
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
        """The home device of a Blender stage, or None: the stage waits. Homes are fixed: bakes and sheets on the
        Blender card, previews on the iGPU (or the Blender card when the iGPU is taken). The one exception is a
        logged spill of a bake to the UE card when an outside job has held the Blender card for BAKE_SPILL_S and
        the UE card has the VRAM. Never the CPU, unless --device cpu was asked for by hand."""
        want = st.job.args.device or C.BLENDER_DEVICE
        if want == "cpu":
            return ("cpu", None)
        need = self.vram_need(st)
        home = self.cards[self.bl_bus]
        if self.released:                                  # cards given to a video run: previews on the iGPU only
            if st.job.mode == "preview" and want in ("auto", "igpu") and not self.igpu.blender and R.igpu_free() \
                    and self.igpu.hold():
                return ("igpu", self.igpu)
            return None
        if st.job.mode == "preview" and want in ("auto", "igpu"):
            if not self.igpu.blender and R.igpu_free() and self.igpu.hold():
                return ("igpu", self.igpu)
            if want == "igpu":
                return None
        if want == "igpu":
            return ("igpu", self.igpu) if not self.igpu.blender and R.igpu_free() and self.igpu.hold() else None
        if self.card_ok(home, need):
            return ("card", home)
        ue = self.cards[self.ue_bus]
        out_s = now() - home.outside_since if home.outside_since and home.fd is None else 0.0
        if (st.job.mode == "final" and C.BAKE_SPILL_S >= 0 and not self.released and (out_s >= C.BAKE_SPILL_S or home.draining
                                                                 or now() < home.released_until)
                and ue.fd is not None and len(ue.blender) < 1 and ue.free_gb() >= need + 2):
            self.spills += 1
            st.info["spill"] = (f"Blender card {home.bus} held by an outside job for {out_s:.0f}s; "
                                f"baking on the UE card {ue.bus} ({ue.free_gb():.1f} GB free)")
            print(f"[sched] spill: {st.job.id}: {st.info['spill']}", flush=True)
            return ("card", ue)
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
        st.device, st.device_class = "paint.py (own locks)", "paint"
        with open(job.adir / "logs" / "paint_stage.log", "w") as f:
            p = subprocess.Popen([sys.executable, str(HERE / "paint.py"), str(job.adir)], stdout=f,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            while p.poll() is None:
                if job.cancelled:
                    p.terminate()
                    p.wait()
                    st.state = "cancel"
                    return
                time.sleep(0.5)
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
            if e.state == "ready" and need and e.project != need and self.ue_backlog(e.project) == 0:
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
        ed.n_jobs += 1
        q = ed.qdir / f"{job.id}.{job.n}.job.json"
        tmp = ed.qdir / f".{job.id}.{job.n}.tmp"
        tmp.write_text(json.dumps({"ue_script": jd["script"], "job": jd.get("job", {}), "id": job.id,
                                   "result": str(res_p)}, indent=1))
        tmp.replace(q)
        job.log(f"UE script {Path(jd['script']).name} on {Path(ed.project).stem}: editor on card {ed.card.bus} "
                f"after {round(st.t_start - st.t_ready, 1)}s")
        t0, limit = now(), float(jd.get("timeout_s", 1800))
        outs = [Path(o) for o in jd.get("outputs", [])]
        res = None
        while now() - t0 < limit:
            if res is None and res_p.exists():
                try:
                    res = json.loads(res_p.read_text())
                except ValueError:
                    pass
            if res is not None and (not res.get("ok") or all(o.exists() for o in outs)):
                break
            if not ed.alive():
                break
            time.sleep(0.05)
        dt = round(now() - t0, 1)
        missing = [str(o) for o in outs if not o.exists()]
        ok = bool(res and res.get("ok")) and not missing
        err = None if ok else ((res or {}).get("error") or (f"outputs missing: {missing}" if res else
                               f"no result ({'editor died' if not ed.alive() else 'timed out'} after {dt}s); "
                               f"see {ed.log}"))
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
                    out["queue"].append({"job": j.n, "id": j.id, "mode": j.mode, "client": j.client_pid,
                                         "stage": s.name, "state": s.state, "device": s.device,
                                         "since_s": round(t - (s.t_start or s.t_ready or j.t_submit), 1)})
        return out


# ---------------------------------------------------------------- socket server
def serve():
    C.SCHED_DIR.mkdir(parents=True, exist_ok=True)
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

    def sampler():
        buses = list(S.cards) + [C.IGPU_BUS]
        while True:
            S.util.append((now(), {b: busy_pct(b) for b in buses}))
            time.sleep(0.5)

    threading.Thread(target=sampler, daemon=True).start()
    SOCK.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK))
    os.chmod(SOCK, 0o600)
    srv.listen(64)
    print(f"[sched] pid {os.getpid()} serving on {SOCK} (code {S.code})", flush=True)

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
        if cmd == "drain":
            with S.lock:
                S.drain = True
            conn.sendall(b'{"ok": true}\n')
            return
        if cmd == "ue_job":
            with S.lock:
                if S.should_exit() or S.stopping:
                    conn.sendall(b'{"retry": true}\n')
                    return
            jd = msg["ue_job"]
            for k in ("project", "script"):
                if not jd.get(k) or not Path(jd[k]).exists():
                    raise SystemExit(f"ue job: '{k}' must name an existing file ({jd.get(k)})")
            args = argparse.Namespace(preview=False, paint=False, skip_render=False, rebuild_stage=False,
                                      device=None, ue_timeout=float(jd.get("timeout_s", 1800)), model=None)
            jid = jd.get("id") or Path(jd["script"]).stem
            job = Job(args, {"id": jid, "class": "ue_job"}, conn, msg.get("pid"), [])
            job.mode, job.ue = "ue_job", jd
            job.project = str(Path(jd["project"]).resolve())
            job.adir = Path(jd.get("log_dir") or (C.SCHED_DIR / "jobs" / f"{jid}_{job.n}"))
            job.adir.mkdir(parents=True, exist_ok=True)
            job.log(f"{jid}: UE job {job.n} for {Path(jd['project']).stem} queued with the scheduler (pid {os.getpid()})")
            S.submit(job)
            while True:
                if not f.readline():
                    break
            S.cancel(job, "client disconnected")
            return
        if cmd != "submit":
            conn.sendall(b'{"error": "unknown command"}\n')
            return
        with S.lock:
            if S.should_exit() or S.stopping:
                conn.sendall(b'{"retry": true}\n')
                return
        if msg.get("env") != R.sched_env():
            conn.sendall((json.dumps({"local": "the scheduler runs with other ASSETS_* settings"}) + "\n").encode())
            return
        if msg.get("code") != S.code:
            S.drain = True                                    # newer chain code: exit when idle, next run restarts
        args = R.build_parser().parse_args(msg["argv"])
        args.spec = str(Path(msg["cwd"], args.spec).resolve())
        args.model = str(Path(msg["cwd"], args.model).resolve())
        spec = json.loads(Path(args.spec).read_text())
        R.validate_spec(spec)
        adir = C.LIBRARY / spec["class"] / spec["id"]
        for d in ("source", "logs", "ue"):
            (adir / d).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.model, adir / "source" / Path(args.model).name)
        (adir / "source" / "spec.json").write_text(json.dumps(spec, indent=1))
        job = Job(args, spec, conn, msg.get("pid"), msg["argv"])
        job.log(f"{spec['id']}: job {job.n} queued with the scheduler (pid {os.getpid()})")
        S.submit(job)
        while True:                                           # EOF = the client went away
            if not f.readline():
                break
        S.cancel(job, "client disconnected")
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


def client_submit(argv):
    """Send this run to the scheduler and relay its log; returns the exit code (None = run locally instead)."""
    for _ in range(5):
        s = connect()
        msg = {"cmd": "submit", "argv": argv, "cwd": os.getcwd(), "pid": os.getpid(), "env": R.sched_env(),
               "code": code_hash()}
        s.sendall((json.dumps(msg) + "\n").encode())
        f = s.makefile("rb")
        retry = False
        for line in f:
            m = json.loads(line)
            if "log" in m:
                print(m["log"], flush=True)
            elif m.get("retry"):
                retry = True
                break
            elif "local" in m:
                print(f"scheduler: {m['local']}; running locally", flush=True)
                return None
            elif "result" in m:
                r = m["result"]
                if r.get("error") and r.get("rc"):
                    print(r["error"], file=sys.stderr, flush=True)
                return int(r.get("rc", 1))
        s.close()
        if not retry:
            print("the scheduler went away before the job finished; see its log in ASSETS_LOGS", file=sys.stderr)
            return 4
        time.sleep(0.5)
    return 4


def client_ue_job(path):
    """run_asset.py --ue-job JOB.json: run an editor-side script in the scheduler's resident editor of its project.
    JOB: {"project": .uproject, "script": editor-side .py, "job": {...} (the script's JOB), "outputs": [paths to wait
    for], "timeout_s": 1800, "id": name, "log_dir": folder for sched.json / ue_result.json}. Prints the result."""
    jd = json.loads(Path(path).read_text())
    for k in ("project", "script"):
        if jd.get(k):
            jd[k] = str(Path(jd[k]).resolve())
    jd["outputs"] = [str(Path(o).resolve()) for o in jd.get("outputs", [])]
    for _ in range(5):
        s = connect()
        s.sendall((json.dumps({"cmd": "ue_job", "ue_job": jd, "pid": os.getpid()}) + "\n").encode())
        retry = False
        for line in s.makefile("rb"):
            m = json.loads(line)
            if "log" in m:
                print(m["log"], flush=True)
            elif m.get("retry"):
                retry = True
                break
            elif "result" in m:
                r = m["result"]
                r.pop("sched", None)
                print(json.dumps(r, default=str), flush=True)
                return int(r.get("rc", 1))
        s.close()
        if not retry:
            print("the scheduler went away before the job finished; see its log in ASSETS_LOGS", file=sys.stderr)
            return 4
        time.sleep(0.5)
    return 4


def client_cmd(cmd, **kw):
    s = connect(start=False)
    if s is None:
        return None
    s.sendall((json.dumps({"cmd": cmd, **kw}) + "\n").encode())
    line = s.makefile("rb").readline()
    s.close()
    return json.loads(line) if line else None


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
        print("no scheduler running (it starts with the next run_asset.py job)")
        for name in [*C.GPU_CARDS, C.IGPU_LOCK_NAME]:
            h = lock_holders(f"/tmp/gpu-{name}.lock", True)
            print(f"  lock gpu-{name}: " + ("; ".join(f"{p} {c}" for p, c in h) if h else "free"))
        return 0
    print(f"scheduler pid {st['pid']}, up {st['up_s']}s, code {st['code']}{' (exits when idle)' if st['drain'] else ''}"
          f", bakes spilled to the UE card: {st.get('spills', 0)}")
    print("devices:")
    for d in st["devices"]:
        print(f"  {d['device']:9} busy {d.get('busy')}%  vram {d.get('vram_gb')}  lock: {d['lock']}")
        if d.get("editor"):
            print(f"            editor {d['editor']}")
        for b in d.get("blender") or []:
            print(f"            blender {b}")
    print("queue:" + ("" if st["queue"] else " empty"))
    for q in st["queue"]:
        print(f"  job {q['job']:>3} {q['id']:32} {q['mode']:7} {q['stage']:15} {q['state']:5} "
              f"{q['since_s']:>6}s {q.get('device') or ''} (client {q['client']})")
    print("recent:")
    for r in st["recent"]:
        parts = ", ".join(f"{s['name']} {s.get('device') or '-'} wait {s['wait_s']} run {s['run_s']}"
                          for s in r["stages"])
        print(f"  job {r['job']:>3} {r['id']:32} {r['mode']:7} {r['state']:9} {r['total_s']:>6}s  {parts}")
    return 0


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
