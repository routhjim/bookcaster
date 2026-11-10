"""Run bl_outfit.py in Blender on one locked XTX card (never the CPU for bake or render).

    python tools/assets/characters/char_outfit.py job.json [--cards 05,65] [--geometry-only]

Waits while the GPU hold file exists ($CHAR_GPU_HOLD, default ~/ue/GPU_HOLD), then takes the first free card lock
(/tmp/gpu-<bus>.lock, non-blocking flock polled every 5 s), runs Blender with that card as the only Cycles device
(job["hip_bus"]) and 4 CPU threads, and releases the lock when Blender exits. --geometry-only runs the build, skinning
and export steps without a card (no bake, no preview). Settings: CHAR_BLENDER (default $ASSETS_BLENDER, else
~/opt/blender/current/blender). Stdlib only.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UE_HOME = Path(os.path.expanduser(os.environ.get("MH_UE_HOME") or os.environ.get("ASSETS_UE_HOME") or "~/ue"))
HOLD = Path(os.environ.get("CHAR_GPU_HOLD", str(UE_HOME / "GPU_HOLD")))
BLENDER = os.path.expanduser(os.environ.get("CHAR_BLENDER") or os.environ.get("ASSETS_BLENDER")
                             or "~/opt/blender/current/blender")


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def take(cards):
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
    ap.add_argument("--cards", default=os.environ.get("CHAR_BLENDER_CARDS", "05"))
    ap.add_argument("--geometry-only", action="store_true")
    a = ap.parse_args()
    job = json.loads(Path(a.job).read_text())
    fd = None
    if a.geometry_only:
        job["steps"] = [s for s in job.get("steps", []) if s in ("build", "export")]
        bus = None
    else:
        bus, fd, waited = take([c.strip() for c in a.cards.split(",") if c.strip()])
        log(f"card {bus} locked after {waited}s")
        job["hip_bus"] = bus
    run_job = Path(a.job).with_suffix(".run.json")
    run_job.write_text(json.dumps(job, indent=1))
    t0 = time.time()
    cmd = [BLENDER, "-b", "--factory-startup", "-t", "4", "--python", str(HERE / "bl_outfit.py"), "--", str(run_job)]
    try:
        rc = subprocess.call(cmd, pass_fds=())
    finally:
        if fd is not None:
            os.close(fd)
    log(f"blender rc={rc} in {time.time() - t0:.1f}s (card {bus})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
