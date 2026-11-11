"""Run one character editor job (char_render.py) in a full offscreen UE editor on one locked XTX card.

    python tools/assets/characters/char_job.py job.json [--name NAME] [--timeout S]

Waits (polling) while the hold file exists ($CHAR_GPU_HOLD, default ~/ue/GPU_HOLD), takes the first free card lock
(/tmp/gpu-<bus>.lock, flock -n polled), and starts the editor through the MetaHuman lab's runner functions
(metahuman/mh_run.py: memory guard, logs, peak RSS) with one difference: this process keeps the lock and the editor
does NOT inherit its fd, so helpers the editor forks (zenserver, trace server), which can outlive it, never hold the
card. Defaults:
    MH_UE_PROJECT  $MH_UE_HOME/proj/CharLab/CharLab.uproject     MH_LAB  $MH_UE_HOME/lab/character
    MH_GPU_CARDS   "65,05"
    MH_EXTRA_ARGS  adds -NoMetaHumanAccountPortalLoginFallback (cloud steps fail instead of opening a sign-in page)
                   and -notraceserver
Stdlib only.
"""

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UE_HOME = Path(os.path.expanduser(os.environ.get("MH_UE_HOME") or os.environ.get("ASSETS_UE_HOME") or "~/ue"))
HOLD = Path(os.environ.get("CHAR_GPU_HOLD", str(UE_HOME / "GPU_HOLD")))
os.environ.setdefault("MH_UE_PROJECT", str(UE_HOME / "proj" / "CharLab" / "CharLab.uproject"))
os.environ.setdefault("MH_LAB", str(UE_HOME / "lab" / "character"))
os.environ.setdefault("MH_GPU_CARDS", "65,05")
_extra = os.environ.get("MH_EXTRA_ARGS", "").split()
for _f in ("-NoMetaHumanAccountPortalLoginFallback", "-notraceserver"):
    if _f not in _extra:
        _extra.append(_f)
os.environ["MH_EXTRA_ARGS"] = " ".join(_extra)
sys.path.insert(0, str(HERE.parent / "metahuman"))
import mh_run as R  # noqa: E402


def take_card(cards):
    t0 = time.time()
    while True:
        if not HOLD.exists():
            for bus in cards:
                fd = os.open(f"/tmp/gpu-{bus}.lock", os.O_RDWR | os.O_CREAT, 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return bus, fd, round(time.time() - t0, 1)
                except BlockingIOError:
                    os.close(fd)
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--name")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    job_path = Path(a.job).resolve()
    job = json.loads(job_path.read_text())
    name = a.name or job_path.stem
    shim = R.UE_PROJECT.parent / "Content" / "Python" / "init_unreal.py"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE.parent / "metahuman" / "init_unreal.py", shim)
    while R.mem_avail_gb() < R.MIN_FREE_GB + 12:
        R.log(f"waiting for memory: MemAvailable {R.mem_avail_gb():.1f} GiB")
        time.sleep(30)
    bus, fd, waited = take_card(R.CARDS)
    R.log(f"card {bus} locked after {waited}s (hold file checked)")
    env = dict(os.environ, MH_JOB=str(job_path), DRI_PRIME=f"pci-0000_{bus}_00_0")
    env.pop("MESA_VK_DEVICE_SELECT", None)
    cmd = [R.UE_BIN / "UnrealEditor", R.UE_PROJECT, job.get("map", "/Engine/Maps/Templates/Template_Default"),
           "-RenderOffscreen", "-graphicsadapter=0", "-vulkan", "-sm6", "-windowed", "-resx=1152", "-resy=640",
           "-unattended", "-nosplash", "-nosound", "-stdout"] + R.EXTRA
    R.LAB.joinpath("logs").mkdir(parents=True, exist_ok=True)
    try:
        st = R.supervise(cmd, R.LAB / "logs" / f"{name}.log", env, a.timeout, pass_fds=())
    finally:
        os.close(fd)
    st.update(card=bus, lock_wait_s=waited)
    R.log(json.dumps(st))
    return 0 if st.get("rc") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
