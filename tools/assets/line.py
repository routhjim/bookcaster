"""The serial asset line: the work list on disk, admission (short lines), batching, and the power / hardware guard.

Pure logic with no GPU or UE calls, so it can be unit-tested (tools/tests/test_asset_line.py); assetd.py drives it.

Stations, one job each (nothing falls back to the CPU; a job waits for its station):
    igpu       reference drawing, paint, cloth drape (image model / solver on the iGPU)
    card<B>    the Blender card (ASSETS_BLENDER_CARD, default 05): Blender bakes and previews, image-to-3D
    card<U>    the engine card (ASSETS_UE_CARD, default 65): the one resident UE editor; a command job ("engine")
               may borrow it while the engine has no work (the editor unloads, and loads for the next engine job)
    video      a booked window: the card stations finish, both cards are handed over, every waiting clip runs,
               then the cards come back

Settings (environment):
    ASSETS_LINE_MAX         jobs that may wait in front of one station           default 3
    ASSETS_LINE_BATCH       jobs of one kind a station runs before it may switch default 8
    ASSETS_POWER_ROOT       sysfs power_supply folder                             default /sys/class/power_supply
    ASSETS_POWER_PAUSE_PCT  at or below this battery %, start nothing new (running jobs finish), write the list's
                            state (the sleep inhibitor stays held)                default 20
    ASSETS_POWER_RESUME_PCT resume once the battery is back at this %             default 40

Memory: MemBudget below (settings in config.py, ASSETS_MEM_*); a command that only found a resource busy is
run again (busy_exit, ASSETS_LINE_BUSY_*, ASSETS_LINE_RETRIES).

The load is bounded by the stations themselves (one job each), the short lines and the batches; the battery is
watched over time (percent and energy_now logged at every job start and end, cost per job kind) and only the floor
above stops work. A lost charger or card is logged with its time and holds card jobs until both are back.
    ASSETS_ASPM_POLICY      file that must read [performance] for XTX jobs        default /sys/module/pcie_aspm/parameters/policy
"""

import json
import os
import time
from pathlib import Path


def _env_i(k, d):
    return int(os.environ.get(k, d))


def _env_f(k, d):
    return float(os.environ.get(k, d))


LINE_MAX = _env_i("ASSETS_LINE_MAX", 3)
BATCH = _env_i("ASSETS_LINE_BATCH", 8)


# ---------------------------------------------------------------- the work list on disk
class WorkList:
    """One JSON file per entry (<dir>/entries/<seq>.json), written atomically. An entry is a dict with at least
    seq, type ("asset" | "ue_job" | "cmd"), state ("queued" | "running" | "done" | "failed" | "cancelled"),
    stages ({name: state}) and submitted (epoch s). Finished entries move to <dir>/done/."""

    def __init__(self, root):
        self.root = Path(root)
        self.edir, self.ddir = self.root / "entries", self.root / "done"
        self.edir.mkdir(parents=True, exist_ok=True)
        self.ddir.mkdir(parents=True, exist_ok=True)

    def next_seq(self):
        p = self.root / "seq"
        try:
            n = int(p.read_text()) + 1
        except (OSError, ValueError):
            n = 1 + max([int(f.stem) for d in (self.edir, self.ddir) for f in d.glob("*.json")
                         if f.stem.isdigit()], default=0)
        tmp = self.root / f".seq.{os.getpid()}"
        tmp.write_text(str(n))
        tmp.replace(p)
        return n

    def save(self, e):
        """Atomic and durable (fsync before the rename): a state written here survives a crash or power loss."""
        p = self.edir / f"{e['seq']:06d}.json"
        tmp = self.edir / f".{e['seq']:06d}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(e, indent=1, default=str))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)
        if e.get("state") in ("done", "failed", "cancelled"):
            p.replace(self.ddir / p.name)

    def get(self, seq):
        """One entry by its number (unfinished or done), or None."""
        for d in (self.edir, self.ddir):
            try:
                return json.loads((d / f"{int(seq):06d}.json").read_text())
            except (OSError, ValueError):
                continue
        return None

    def load(self):
        """Unfinished entries in list order; what was running goes back to the line (it is re-run)."""
        out = []
        for f in sorted(self.edir.glob("*.json")):
            try:
                e = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if e.get("state") in ("done", "failed", "cancelled"):
                f.replace(self.ddir / f.name)
                continue
            e["resumed"] = []
            for k, v in (e.get("stages") or {}).items():
                if v == "run":
                    e["stages"][k] = "wait"
                    e.setdefault("resumed", []).append(k)
            if e.get("state") == "running":
                e["state"] = "queued"
            out.append(e)
        return out

    def recent(self, since):
        out = []
        for f in self.ddir.glob("*.json"):
            if f.stat().st_mtime >= since:
                try:
                    out.append(json.loads(f.read_text()))
                except (OSError, ValueError):
                    pass
        return sorted(out, key=lambda e: e["seq"])

    def write_state(self, entries, why):
        """The list's state in one file (power pause): what waits, what was running."""
        tmp = self.root / f".state.{os.getpid()}"
        tmp.write_text(json.dumps({"t": time.time(), "time": time.ctime(), "why": why,
                                   "entries": [{k: e.get(k) for k in ("seq", "type", "id", "kind", "state", "stages")}
                                               for e in entries]}, indent=1))
        tmp.replace(self.root / "state.json")


# ---------------------------------------------------------------- admission and batching
def line_full(waiting_at, station, line_max=None):
    """None when one more job may wait at station, else the 'line full' answer for the client."""
    n, lim = waiting_at.get(station, 0), LINE_MAX if line_max is None else line_max
    if n < lim:
        return None
    return f"line full at {station} ({n} waiting, max {lim}); retry later, do not submit more now"


class Batcher:
    """Per station: keep one kind going until BATCH jobs of it ran or none of it waits, then the oldest other."""

    def __init__(self, batch=None):
        self.batch = BATCH if batch is None else batch
        self.cur = {}                     # station -> [kind, n run in this batch]

    def pick(self, station, ready):
        """ready: [(seq, kind, item)] for this station. Returns the item to run next (or None)."""
        if not ready:
            return None
        ready = sorted(ready, key=lambda r: r[0])
        kind, n = self.cur.get(station, (None, 0))
        same = [r for r in ready if r[1] == kind]
        other = [r for r in ready if r[1] != kind]
        if same and (n < self.batch or not other):
            choice = same[0]
        else:
            choice = (other or same)[0]
        if choice[1] == kind:
            self.cur[station] = [kind, n + 1]
        else:
            self.cur[station] = [choice[1], 1]
        return choice[2]

    def state(self):
        return {s: {"kind": k, "n": n} for s, (k, n) in self.cur.items()}


# ---------------------------------------------------------------- memory budget (pure arithmetic; assetd.py feeds it)
class MemBudget:
    """What may start, by system memory. A job kind declares what it needs (GiB); a job starts only when
    MemAvailable, less what jobs that started within the ramp have declared (they are still loading: the memory
    is promised, not yet taken), less the reserve, covers it. Two jobs that cannot both fit therefore never
    overlap, whichever starts first, without either naming the other."""

    def __init__(self, kind_gb, default_gb, reserve_gb, ramp_s, total_gb=None):
        self.kind_gb, self.default_gb = dict(kind_gb), float(default_gb)
        self.reserve_gb, self.ramp_s, self.total_gb = float(reserve_gb), float(ramp_s), total_gb

    def need(self, kind, declared=None):
        """GiB a job needs: what its entry declares, else its kind's figure, else the default."""
        if declared is not None:
            return max(0.0, float(declared))
        return max(0.0, float(self.kind_gb.get(kind, self.default_gb)))

    def pending(self, running, t):
        """GiB promised to jobs still loading. running: [(need GiB, start time)]."""
        return sum(n for n, t0 in running if t0 is not None and t - t0 < self.ramp_s)

    def room(self, avail, running, t):
        """GiB a new job may take now."""
        return avail - self.pending(running, t) - self.reserve_gb

    def fits(self, need, avail, running, t, reclaim=0.0):
        """(ok, text). reclaim: GiB that will come back before the job loads (an editor that unloads for it)."""
        if need <= 0:
            return True, None
        if self.total_gb is not None and need + self.reserve_gb > self.total_gb:
            return False, (f"memory: it needs {need:.0f} GiB and {self.reserve_gb:.0f} kept free, more than the "
                           f"box has ({self.total_gb:.0f}); it can never start (declare less, or cancel it)")
        pend = self.pending(running, t)
        room = avail + reclaim - pend - self.reserve_gb
        if room >= need:
            return True, None
        return False, (f"memory: it needs {need:.0f} GiB; {avail:.0f} available"
                       + (f", {pend:.0f} promised to jobs still loading" if pend else "")
                       + f", {self.reserve_gb:.0f} kept free")

    def never(self, need):
        return self.total_gb is not None and need + self.reserve_gb > self.total_gb


# ---------------------------------------------------------------- retries: a resource was busy, not a failure
def busy_exit(rc, log_tail, rcs, log_re):
    """Why a failed command only found a resource busy (text), or None for a real failure: its exit code is one
    of rcs, or its last log lines match log_re (tools that exit 1 after waiting for memory or a lock)."""
    if rc is None or rc == 0:
        return None
    if rc in rcs:
        return f"exit code {rc} (resource busy)"
    if log_re and log_tail:
        import re
        m = re.search(log_re, log_tail, re.I)
        if m:
            return f"its log says: {m.group(0)[:80]}"
    return None


def retry_delay(n, base_s):
    """Seconds before retry number n (1-based): base, doubled each time."""
    return base_s * (2 ** max(0, n - 1))


def log_tail(path, lines=12, max_bytes=8192):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return "\n".join(f.read().decode(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# ---------------------------------------------------------------- the standing fill list
def parse_cmd_line(text):
    """One "--line-cmd STATION KIND [--id NAME] [--log FILE] [--mem-gb N] [--vram-gb N] [--est-s N] [--fill]
    [--detach] -- CMD..." line (the --feed format) as a dict {station, kind, id, log, mem_gb, vram_gb, est_s, cmd},
    or (None, why) when it is not one. Returns (dict, None) or (None, why)."""
    import shlex
    try:
        argv = shlex.split(text)
    except ValueError as ex:
        return None, f"cannot be split into arguments ({ex})"
    if not argv or argv[0] != "--line-cmd":
        return None, "not a --line-cmd line (only commands can be fill work)"
    if "--" not in argv or argv.index("--") < 3 or argv.index("--") == len(argv) - 1:
        return None, "needs STATION KIND [options] -- COMMAND"
    k = argv.index("--")
    head, cmd = argv[1:k], argv[k + 1:]
    opts = head[2:]
    get = lambda o: opts[opts.index(o) + 1] if o in opts and opts.index(o) + 1 < len(opts) else None
    out = {"station": head[0], "kind": head[1], "id": get("--id"), "log": get("--log"), "cmd": cmd}
    for o, key in (("--mem-gb", "mem_gb"), ("--vram-gb", "vram_gb"), ("--est-s", "est_s")):
        v = get(o)
        if v is not None:
            try:
                out[key] = float(v)
            except ValueError:
                return None, f"{o} is not a number ({v})"
    return out, None


def check_cmd_line(text, aliases):
    """(entry, [problems]) for one list line: the station must resolve (aliases: name -> station) and not be the
    video window; a script or binary the command names by absolute path must exist. Problems are plain words."""
    d, why = parse_cmd_line(text)
    if d is None:
        return None, [why]
    probs = []
    b = aliases.get(str(d["station"]))
    if b is None:
        probs.append(f"station '{d['station']}' is not a station of this line ({', '.join(sorted(set(aliases)))})")
    elif b == "video":
        probs.append("the video window is not a fill station")
    for a in d["cmd"]:
        if "=" in a and not a.startswith("/"):
            a = a.split("=", 1)[1]                      # VAR=/path in an "env VAR=... CMD" line
        if a.startswith("/") and not Path(a).exists():
            probs.append(f"{a} does not exist")
    d["station_resolved"] = b
    return d, probs


class FillList:
    """The standing fill list: a text file, one --line-cmd line per job, top first. take() removes a line (to
    <list>.done with the time), reject() moves a line that cannot run to <list>.bad with the reason; both rewrite
    the list atomically, so a line is never drawn twice and never lost."""

    def __init__(self, path):
        self.path = Path(path)

    def lines(self):
        try:
            return [ln.strip() for ln in self.path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        except OSError:
            return []

    def _remove(self, line, to, note):
        try:
            raw = self.path.read_text().splitlines()
        except OSError:
            return
        done, kept = False, []
        for ln in raw:
            if not done and ln.strip() == line:
                done = True
                continue
            kept.append(ln)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""))
        tmp.replace(self.path)
        with open(self.path.with_name(self.path.name + to), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{note}\t{line}\n")

    def take(self, line, note=""):
        self._remove(line, ".done", note)

    def reject(self, line, why):
        self._remove(line, ".bad", why)

    def add(self, lines):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        have = set(self.lines())
        new = [ln for ln in lines if ln not in have]
        with open(self.path, "a") as f:
            for ln in new:
                f.write(ln + "\n")
        return len(new)


# ---------------------------------------------------------------- power and hardware guard
def _read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


class PowerGuard:
    """Battery floor, charger and card presence, ASPM. update() samples; verdict() says what may start."""

    def __init__(self, root=None, cards=(), aspm=None, clock=time.time):
        self.root = Path(root or os.environ.get("ASSETS_POWER_ROOT", "/sys/class/power_supply"))
        self.cards = list(cards)
        self.aspm = aspm or os.environ.get("ASSETS_ASPM_POLICY", "/sys/module/pcie_aspm/parameters/policy")
        self.pci = Path(os.environ.get("ASSETS_PCI_ROOT", "/sys/bus/pci/devices"))
        self.clock = clock
        self.pause_pct = _env_f("ASSETS_POWER_PAUSE_PCT", 20)
        self.resume_pct = _env_f("ASSETS_POWER_RESUME_PCT", 40)
        self.paused_since = None          # battery floor reached (until back at resume_pct)
        self.last = {}
        self.events = []                  # (t, text): battery floor, charger / card lost and back
        self.lost_since = None

    def sample(self):
        b = self.root / "BAT0"
        e = _read(b / "energy_now")
        cap = _read(b / "capacity")
        ac = _read(self.root / "AC0" / "online")
        usb = [_read(p / "online") for p in sorted(self.root.glob("ucsi-source-psy-*"))]
        return {"status": _read(b / "status"), "capacity": int(cap) if cap and cap.isdigit() else None,
                "energy_wh": int(e) / 1e6 if e and e.isdigit() else None,
                "ac": ac == "1" if ac is not None else None,
                "usb_online": sum(1 for u in usb if u == "1"), "usb_n": len(usb),
                "cards": {c: (self.pci / f"0000:{c}:00.0").exists() for c in self.cards},
                "aspm": _read(self.aspm)}

    def update(self, s=None):
        t = self.clock()
        s = s or self.sample()
        self.last = dict(s, t=t)
        cap = s.get("capacity")
        if cap is not None:
            if self.paused_since is None and cap <= self.pause_pct:
                self.paused_since = t
                self.events.append((t, f"battery {cap}%: paused (resumes at {self.resume_pct:.0f}%)"))
            elif self.paused_since is not None and cap >= self.resume_pct:
                self.events.append((t, f"battery {cap}%: resumed after {t - self.paused_since:.0f}s"))
                self.paused_since = None
        lost = self._lost(s)
        if lost and self.lost_since is None:
            self.lost_since = t
            self.events.append((t, "lost: " + ", ".join(lost)))
        elif not lost and self.lost_since is not None:
            self.events.append((t, f"charger and cards back after {t - self.lost_since:.0f}s"))
            self.lost_since = None
        self.events = self.events[-20:]
        return self.last

    def _lost(self, s):
        out = []
        if s.get("ac") is False:
            out.append("charger (AC0 offline)")
        elif s.get("usb_n") and s.get("usb_online", 0) == 0:
            out.append("charger (no USB-C source online)")
        out += [f"card {c}" for c, ok in (s.get("cards") or {}).items() if not ok]
        return out

    def verdict(self):
        """{"pause": start nothing (battery floor), "cards": may start XTX jobs, "why": [reasons]}."""
        s = self.last
        why, v = [], {"pause": False, "cards": True}
        if self.paused_since is not None:
            v.update(pause=True, cards=False)
            why.append(f"paused: battery {s.get('capacity')}% (floor {self.pause_pct:.0f}%, resumes at "
                       f"{self.resume_pct:.0f}%); running jobs finish, list state written")
        lost = self._lost(s)
        if lost:
            v["cards"] = False
            when = time.strftime('%H:%M:%S', time.localtime(self.lost_since)) if self.lost_since else "?"
            why.append(f"lost {', '.join(lost)} since {when}: card jobs held until the charger and both cards are back")
        aspm = s.get("aspm")
        if aspm is not None and "[performance]" not in aspm:
            v["cards"] = False
            why.append(f"PCIe ASPM policy is not [performance] ({aspm.strip()}): XTX jobs refused (a reboot resets "
                       f"it; set it back as root)")
        v["why"] = why
        return v

    def brief(self):
        s = self.last
        return (f"bat {s.get('capacity')}% {s.get('status')} {s.get('energy_wh') or 0:.2f}Wh "
                f"ac={'on' if s.get('ac') else 'off'} usb={s.get('usb_online')}/{s.get('usb_n')}")


# ---------------------------------------------------------------- the recording runner for command entries
def read_rc(path):
    """The exit code a command's runner recorded ({"rc", "t"}), or None when it has not ended (or never ran)."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def run_recorded(rc_file, cmd):
    """Run cmd and write its exit code to rc_file (atomic, fsynced) the moment it ends, before anything else.

    The line starts every command entry through this (python line.py --run RC_FILE -- CMD...), in the command's own
    session. The record does not depend on the line: if the line restarts while the command runs, or between the
    command's end and the line's own bookkeeping, the next line reads the record and never runs a finished entry
    again. SIGTERM / SIGINT to the runner are passed on to the command."""
    import signal
    import subprocess
    p = subprocess.Popen(cmd)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda n, f: p.send_signal(n))
    while True:
        try:
            rc = p.wait()
            break
        except InterruptedError:
            continue
    rc_file = Path(rc_file)
    tmp = rc_file.with_name(f".{rc_file.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps({"rc": rc, "t": time.time()}))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(rc_file)
    return rc


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 4 and sys.argv[1] == "--run" and sys.argv[3] == "--":
        rc = run_recorded(sys.argv[2], sys.argv[4:])
        sys.exit(rc if rc >= 0 else 128 - rc)
    sys.exit("usage: line.py --run RC_FILE -- CMD...")
