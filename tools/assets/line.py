"""The serial asset line: the work list on disk, admission (short lines), batching, and the power / hardware guard.

Pure logic with no GPU or UE calls, so it can be unit-tested (tools/tests/test_asset_line.py); assetd.py drives it.

Stations, one job each (nothing falls back to the CPU; a job waits for its station):
    igpu       reference drawing, paint, cloth drape (image model / solver on the iGPU)
    card<B>    the Blender card (ASSETS_BLENDER_CARD, default 05): Blender bakes and previews, image-to-3D
    card<U>    the engine card (ASSETS_UE_CARD, default 65): the one resident UE editor
    video      a booked window: the card stations finish, both cards are handed over, every waiting clip runs,
               then the cards come back

Settings (environment):
    ASSETS_LINE_MAX         jobs that may wait in front of one station           default 3
    ASSETS_LINE_BATCH       jobs of one kind a station runs before it may switch default 8
    ASSETS_POWER_ROOT       sysfs power_supply folder                             default /sys/class/power_supply
    ASSETS_POWER_PAUSE_PCT  at or below this battery %, start nothing new (running jobs finish), write the list's
                            state (the sleep inhibitor stays held)                default 20
    ASSETS_POWER_RESUME_PCT resume once the battery is back at this %             default 40

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
        p = self.edir / f"{e['seq']:06d}.json"
        tmp = self.edir / f".{e['seq']:06d}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(e, indent=1, default=str))
        tmp.replace(p)
        if e.get("state") in ("done", "failed", "cancelled"):
            p.replace(self.ddir / p.name)

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
