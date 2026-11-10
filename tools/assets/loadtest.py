"""Load test for the asset chain: the library's example assets, run by several clients at once.

    python tools/assets/loadtest.py --clients 2,4,8 --modes preview,final [--n 8] [--tier mid] [--tag NAME]
    python tools/assets/loadtest.py --clean            # remove the test copies (library folders + manifest entries)

Each asset (spec + model script under examples/ and models/) is copied under a test id prefix (default "lt_"), so
the real library entries are never overwritten; with --n larger than the number of assets they repeat under new
ids. For every (mode, clients) cell, `clients` workers each run `run_asset.py SPEC MODEL [--preview]` (the runner
given by --runner, default this folder's) one asset after another from a shared list until all --n are done.

Measured per cell: wall time of the batch; per asset wall time; seconds waited for devices (the scheduler's
logs/sched.json when present, else parsed from the run's log for the pre-scheduler chain); the bake device (XTX /
iGPU / CPU from blender_report.json); mean gpu_busy_percent of both cards and the iGPU (sampled at 4 Hz); and who
held each card's lock when the cell started. Output: <out>/<tag>/results.json and a table on stdout.
Outputs go under ASSETS_LOADTEST_OUT (default $ASSETS_UE_HOME/lab/sched).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

ASSETS = ["examples/anvil", "models/poker_wrought_01", "models/hammer_cross_peen_01", "models/bellows_forge_01",
          "models/wafer_cassette_200mm_01", "models/cleanroom_tool_bay_01", "models/starfighter"]
OUT = Path(os.environ.get("ASSETS_LOADTEST_OUT") or (C.UE_HOME / "lab" / "sched"))


def busy(bus):
    try:
        return int(Path(f"/sys/bus/pci/devices/0000:{bus}:00.0/gpu_busy_percent").read_text())
    except (OSError, ValueError):
        return None


def vram_gb(bus):
    try:
        return round(int(Path(f"/sys/bus/pci/devices/0000:{bus}:00.0/mem_info_vram_used").read_text()) / 2**30, 1)
    except (OSError, ValueError):
        return None


def lock_holders():
    """{lock name: [pid cmdline]} for the GPU lock files, from /proc/locks."""
    out = {}
    try:
        locks = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return out
    for name in ("05", "65", "igpu"):
        p = Path(f"/tmp/gpu-{name}.lock")
        if not p.exists():
            continue
        ino = p.stat().st_ino
        who = []
        for ln in locks:
            f = ln.split()
            if len(f) > 5 and f[1] == "FLOCK" and f[5].endswith(f":{ino}"):
                pid = f[4]
                try:
                    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                except OSError:
                    cmd = "?"
                who.append(f"{pid} {cmd[:120]}")
        out[name] = who
    return out


class Sampler:
    def __init__(self, buses=("05", "65", C.IGPU_BUS)):
        self.buses, self.s, self.stop = buses, {b: [] for b in buses}, False
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        while not self.stop:
            for b in self.buses:
                v = busy(b)
                if v is not None:
                    self.s[b].append(v)
            time.sleep(0.25)

    def done(self):
        self.stop = True
        self.th.join()
        return {b: round(sum(v) / len(v), 1) if v else None for b, v in self.s.items()}


def make_copies(n, prefix, tier, work):
    """n (spec copy, model) pairs with ids prefix+k+original id, written under work/specs."""
    sd = work / "specs"
    sd.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(n):
        d = HERE / ASSETS[k % len(ASSETS)]
        spec_p = next(iter(sorted(d.glob("*.json"))))
        model = d / (spec_p.stem + ".py")
        if not model.exists():
            model = next(iter(sorted(d.glob("*.py"))))
        spec = json.loads(spec_p.read_text())
        spec["id"] = f"{prefix}{k:02d}_{spec['id']}"[:64]
        if tier:
            spec["tier"] = tier
            spec.pop("texture_px", None)
        p = sd / f"{spec['id']}.json"
        p.write_text(json.dumps(spec, indent=1))
        out.append((p, model, spec))
    return out


def asset_dir(spec):
    return C.LIBRARY / spec["class"] / spec["id"]


def dev_class(dev):
    d = str(dev or "")
    if d.startswith("HIP"):
        return "igpu" if f"{C.IGPU_BUS}:00" in d else "xtx"
    return "cpu" if d else None


def measure(spec, mode, logp, wall, rc):
    """Per-asset numbers from its folder (and its run log for the pre-scheduler chain)."""
    adir = asset_dir(spec)
    r = {"id": spec["id"], "wall_s": wall, "rc": rc}
    try:
        rep = json.loads((adir / "blender_report.json").read_text())
    except (OSError, ValueError):
        rep = {}
    r["bake_device"] = dev_class(rep.get("device"))
    r["blender_s"] = rep.get("wall_s") or (rep.get("times_s") or {}).get("total")
    sj = adir / "logs" / "sched.json"
    text = logp.read_text(errors="replace") if logp.exists() else ""
    if sj.exists() and sj.stat().st_mtime > time.time() - wall - 5:
        s = json.loads(sj.read_text())
        r["wait_s"] = round(sum(st.get("wait_s") or 0 for st in s.get("stages", [])), 1)
        r["stages"] = {st["name"]: {k: st.get(k) for k in ("device", "wait_s", "run_s")} for st in s.get("stages", [])}
        bl = next((st for st in s.get("stages", []) if st["name"].startswith("blender")), {})
        r["bake_device"] = {"cpu": "cpu", "igpu": "igpu"}.get(bl.get("device_class"), r["bake_device"])
        r["scheduler"] = True
        return r
    w = 0.0
    m = re.search(r"blender \w+: .* after ([\d.]+)s", text)
    if m:
        w += float(m.group(1))
    if mode == "final":
        try:
            ue = json.loads((adir / "logs" / "ue_render.json").read_text())
            m2 = re.search(r"render ([\d.]+)s ok=", text)
            if m2 and ue.get("elapsed_s") is not None:
                r["ue_s"] = float(m2.group(1))
                w += max(0.0, float(m2.group(1)) - float(ue["elapsed_s"]))     # queue + lock + editor start
        except (OSError, ValueError):
            pass
    r["wait_s"] = round(w, 1)
    r["scheduler"] = False
    return r


def run_cell(copies, mode, clients, runner, work, extra):
    todo = list(range(len(copies)))
    lk = threading.Lock()
    res = []
    logd = work / f"{mode}_c{clients}"
    logd.mkdir(parents=True, exist_ok=True)

    def worker():
        while True:
            with lk:
                if not todo:
                    return
                i = todo.pop(0)
            sp, model, spec = copies[i]
            cmd = [sys.executable, str(runner), str(sp), str(model)] + (["--preview"] if mode == "preview" else []) \
                + extra
            logp = logd / f"{spec['id']}.log"
            t = time.time()
            with open(logp, "w") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            wall = round(time.time() - t, 1)
            m = measure(spec, mode, logp, wall, rc)
            with lk:
                res.append(m)

    held = lock_holders()
    vram0 = {b: vram_gb(b) for b in ("05", "65")}
    smp = Sampler()
    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(clients)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = round(time.time() - t0, 1)
    util = smp.done()
    walls = [r["wall_s"] for r in res]
    devs = [r["bake_device"] for r in res if r["bake_device"]]
    sched = None
    try:
        import assetd
        st = assetd.client_cmd("status")
        if st:
            sched = {"spills": st.get("spills"), "released": st.get("released"),
                     "last_release": st.get("last_release"), "estimates": st.get("estimates")}
    except Exception as e:                                  # the pre-scheduler chain: nothing to ask
        sched = {"error": str(e)}
    cell = {"sched": sched, "mode": mode, "clients": clients, "n": len(res), "batch_wall_s": wall,
            "asset_mean_s": round(sum(walls) / len(walls), 1), "asset_worst_s": max(walls),
            "wait_sum_s": round(sum(r.get("wait_s") or 0 for r in res), 1),
            "wait_worst_s": max((r.get("wait_s") or 0) for r in res),
            "bakes": {k: devs.count(k) for k in ("xtx", "igpu", "cpu")},
            "busy_mean": util, "failed": [r["id"] for r in res if r["rc"] != 0],
            "locks_at_start": held, "vram_used_at_start_gb": vram0, "assets": res}
    return cell


def fmt(c):
    b = c["bakes"]
    u = c["busy_mean"]
    return (f"| {c['mode']} | {c['clients']} | {c['n']} | {c['batch_wall_s']} | {c['asset_mean_s']} | "
            f"{c['asset_worst_s']} | {c['wait_sum_s']} / {c['wait_worst_s']} | {b['xtx']}/{b['igpu']}/{b['cpu']} | "
            f"{u.get('05')} / {u.get('65')} / {u.get(C.IGPU_BUS)} | {len(c['failed'])} |")


HEAD = ("| mode | clients | assets | batch wall s | mean s | worst s | waited sum / worst s | bakes xtx/igpu/cpu | "
        "busy % 05 / 65 / igpu | failed |\n|---|---|---|---|---|---|---|---|---|---|")


def clean(prefix):
    mp = C.LIBRARY / "manifest.json"
    n = 0
    for cls in C.LIBRARY.iterdir():
        if cls.is_dir():
            for d in cls.glob(f"{prefix}*"):
                shutil.rmtree(d)
                n += 1
    if mp.exists():
        man = json.loads(mp.read_text())
        man["assets"] = {k: v for k, v in man["assets"].items() if not k.startswith(prefix)}
        tmp = mp.with_suffix(f".lt{os.getpid()}.tmp")
        tmp.write_text(json.dumps(man, indent=1))
        tmp.replace(mp)
    print(f"removed {n} test asset folders and their manifest entries (UE content left in the project)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clients", default="2,4,8")
    ap.add_argument("--modes", default="preview,final")
    ap.add_argument("--n", type=int, default=8, help="assets per cell")
    ap.add_argument("--prefix", default="lt_")
    ap.add_argument("--tier", default="mid", help="force this tier on the copies ('' keeps each spec's own)")
    ap.add_argument("--runner", default=str(HERE / "run_asset.py"))
    ap.add_argument("--tag", default=time.strftime("run_%m%d_%H%M%S"))
    ap.add_argument("--extra", default="", help="extra run_asset.py arguments, space separated")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    if a.clean:
        clean(a.prefix)
        return 0
    work = OUT / a.tag
    work.mkdir(parents=True, exist_ok=True)
    copies = make_copies(a.n, a.prefix, a.tier, work)
    cells = []
    for mode in a.modes.split(","):
        for k in [int(x) for x in a.clients.split(",")]:
            print(f"== {mode} x{k} clients, {a.n} assets", flush=True)
            c = run_cell(copies, mode, k, Path(a.runner), work, a.extra.split())
            cells.append(c)
            print(HEAD + "\n" + fmt(c), flush=True)
            (work / "results.json").write_text(json.dumps({"runner": a.runner, "tier": a.tier, "cells": cells},
                                                          indent=1))
    print(HEAD)
    for c in cells:
        print(fmt(c))
    print(f"results: {work / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
