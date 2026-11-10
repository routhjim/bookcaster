"""Run one MetaHuman lab step in UE: a no-GPU commandlet or a full offscreen editor on one locked XTX.

    python tools/assets/metahuman/mh_run.py cmdlet <ue_script.py> [--job job.json]
    python tools/assets/metahuman/mh_run.py render <job.json>        (job["script"] runs inside the editor)

Settings (environment; defaults follow tools/assets/config.py):
    MH_UE_HOME      UE root            default $ASSETS_UE_HOME or ~/ue
    MH_UE_ENGINE    engine folder      default $MH_UE_HOME/UE_5.8.1
    MH_UE_PROJECT   lab project        default $MH_UE_HOME/proj/MetaLab/MetaLab.uproject
    MH_LAB          scratch + stills   default $MH_UE_HOME/lab/metahuman
    MH_GPU_CARDS    cards to try       default "65,05" (lock files /tmp/gpu-<bus>.lock, non-blocking flock polled)
    MH_LOCK_WAIT_S  give up on cards   default 1800
    MH_MIN_FREE_GB  memory guard       default 12 (the editor is stopped below this MemAvailable)
    MH_EXTRA_ARGS   extra editor args  default "" (e.g. -NoMetaHumanAccountPortalLoginFallback)

The render editor is started with the project's Content/Python/init_unreal.py shim (copied from this folder),
which runs job["script"] when MH_JOB names a job. Peak RSS of the editor and the lowest MemAvailable are recorded.
Stdlib only.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _p(var, default):
    v = os.environ.get(var)
    return Path(os.path.expanduser(v or str(default)))


UE_HOME = _p("MH_UE_HOME", os.environ.get("ASSETS_UE_HOME") or "~/ue")
UE_ENGINE = _p("MH_UE_ENGINE", UE_HOME / "UE_5.8.1")
UE_PROJECT = _p("MH_UE_PROJECT", UE_HOME / "proj" / "MetaLab" / "MetaLab.uproject")
LAB = _p("MH_LAB", UE_HOME / "lab" / "metahuman")
CARDS = [c.strip() for c in os.environ.get("MH_GPU_CARDS", "65,05").split(",") if c.strip()]
LOCK_WAIT_S = float(os.environ.get("MH_LOCK_WAIT_S", "1800"))
MIN_FREE_GB = float(os.environ.get("MH_MIN_FREE_GB", "12"))
EXTRA = os.environ.get("MH_EXTRA_ARGS", "").split()   # e.g. -NoMetaHumanAccountPortalLoginFallback
UE_BIN = UE_ENGINE / "Engine" / "Binaries" / "Linux"


def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def rss_gb(pid):
    """RSS of a process and its children (the editor spawns shader workers)."""
    total = 0
    todo = [pid]
    while todo:
        p = todo.pop()
        try:
            for line in open(f"/proc/{p}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
            kids = Path(f"/proc/{p}/task/{p}/children").read_text().split()
            todo += [int(k) for k in kids]
        except OSError:
            pass
    return total / 2**20


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def take_card(wait_s):
    t0 = time.time()
    while True:
        for bus in CARDS:
            fd = os.open(f"/tmp/gpu-{bus}.lock", os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return bus, fd, round(time.time() - t0, 1)
            except BlockingIOError:
                os.close(fd)
        if time.time() - t0 > wait_s:
            return None, None, round(time.time() - t0, 1)
        time.sleep(15)


def supervise(cmd, logf, env, timeout, pass_fds=()):
    """Run the editor, stop it (by its own PID) if memory runs low; return rc, seconds, memory stats."""
    st = {"rss_peak_gb": 0.0, "avail_min_gb": mem_avail_gb(), "guard": None}
    t0 = time.time()
    with open(logf, "w") as f:
        p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, pass_fds=pass_fds)
        done = threading.Event()

        def watch():
            while not done.is_set() and p.poll() is None:
                a = mem_avail_gb()
                st["avail_min_gb"] = min(st["avail_min_gb"], a)
                st["rss_peak_gb"] = max(st["rss_peak_gb"], rss_gb(p.pid))
                if a < MIN_FREE_GB:
                    st["guard"] = f"MemAvailable {a:.1f} GiB at {time.time() - t0:.0f}s; editor stopped"
                    p.terminate()
                done.wait(1)

        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                rc = p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait()
        done.set()
    st.update(rc=rc, time_s=round(time.time() - t0, 1), log=str(logf))
    st["rss_peak_gb"] = round(st["rss_peak_gb"], 2)
    st["avail_min_gb"] = round(st["avail_min_gb"], 2)
    return st


def cmdlet(script, job, name, timeout):
    LAB.joinpath("logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if job:
        env["MH_JOB"] = str(job)
    if mem_avail_gb() < MIN_FREE_GB + 6:
        raise SystemExit(f"MemAvailable {mem_avail_gb():.1f} GiB: too little for a UE commandlet")
    cmd = [UE_BIN / "UnrealEditor-Cmd", UE_PROJECT, "-run=pythonscript", f"-script={script}", "-nullrhi",
           "-unattended", "-nosplash", "-nosound", "-stdout"] + EXTRA
    return supervise(cmd, LAB / "logs" / f"{name}.log", env, timeout)


def render(job_path, name, timeout):
    job = json.loads(Path(job_path).read_text())
    shim = UE_PROJECT.parent / "Content" / "Python" / "init_unreal.py"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "init_unreal.py", shim)
    while mem_avail_gb() < MIN_FREE_GB + 12:
        log(f"waiting for memory: MemAvailable {mem_avail_gb():.1f} GiB")
        time.sleep(30)
    bus, fd, waited = take_card(LOCK_WAIT_S)
    if bus is None:
        return {"ok": False, "error": f"no card free after {waited}s"}
    log(f"card {bus} locked after {waited}s")
    env = dict(os.environ, MH_JOB=str(Path(job_path).resolve()), DRI_PRIME=f"pci-0000_{bus}_00_0")
    env.pop("MESA_VK_DEVICE_SELECT", None)
    cmd = [UE_BIN / "UnrealEditor", UE_PROJECT, job.get("map", "/Engine/Maps/Templates/Template_Default"),
           "-RenderOffscreen", "-graphicsadapter=0", "-vulkan", "-sm6", "-windowed", "-resx=1152", "-resy=640",
           "-unattended", "-nosplash", "-nosound", "-stdout"] + EXTRA
    try:
        st = supervise(cmd, LAB / "logs" / f"{name}.log", env, timeout, pass_fds=(fd,))
    finally:
        os.close(fd)
    st.update(card=bus, lock_wait_s=waited)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cmdlet", "render"])
    ap.add_argument("path", help="UE script (cmdlet) or job JSON (render)")
    ap.add_argument("--job", help="job JSON for a commandlet (env MH_JOB)")
    ap.add_argument("--name", help="log name")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    name = a.name or Path(a.path).stem
    log(f"{a.mode} {name}: MemAvailable {mem_avail_gb():.1f} GiB")
    st = cmdlet(Path(a.path).resolve(), a.job, name, a.timeout) if a.mode == "cmdlet" else \
        render(a.path, name, a.timeout)
    log(json.dumps(st))
    return 0 if st.get("rc") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
