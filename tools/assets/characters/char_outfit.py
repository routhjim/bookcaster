"""Run bl_outfit.py in Blender; GPU locks are taken inside Blender only for the seconds a bake or render runs.

    python tools/assets/characters/char_outfit.py job.json [--cards 05,65] [--geometry-only]

Waits while the GPU hold file exists ($CHAR_GPU_HOLD, default ~/ue/GPU_HOLD), then takes the first free card lock
(/tmp/gpu-<bus>.lock, non-blocking flock polled every 5 s), runs Blender with that card as the only Cycles device
(job["hip_bus"]) and 4 CPU threads, and releases the lock when Blender exits. --geometry-only runs the build, skinning
and export steps without a card (no bake, no preview).

Drape (job["drape"] = {"parts": ["trousers", ...], "params": {"default": {...}, "<part>": {...}}}): layers are
settled innermost first with the contact solver on the iGPU. For each part a geometry-only Blender pass writes its
panels, stitches, pins and the fixed collider (body + layers already settled) and stops; drape_run.py settles it under
the iGPU lock (/tmp/gpu-igpu.lock, polled; HIP_VISIBLE_DEVICES from `hip-index igpu`); the next pass reads every
result so far. The final pass is the normal job with all results. Settings: CHAR_BLENDER (default $ASSETS_BLENDER,
else ~/opt/blender/current/blender); CHAR_DRAPE_SOLVER (the solver's folder holding the `ppf-contact-solver`
launcher; default the newest under $MH_UE_HOME/tools/ppf-contact-solver/). Stdlib only.
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


def solver_dir():
    d = os.environ.get("CHAR_DRAPE_SOLVER")
    if d:
        return Path(d)
    cands = sorted((UE_HOME / "tools" / "ppf-contact-solver").glob("*/ppf-contact-solver"))
    if not cands:
        raise SystemExit("no contact solver found; set CHAR_DRAPE_SOLVER")
    return cands[-1].parent


def run_drape(inp, out, log_path):
    """drape_run.py on the iGPU under its lock; returns wall seconds and the lock wait."""
    t0 = time.time()
    fd = os.open("/tmp/gpu-igpu.lock", os.O_RDWR | os.O_CREAT, 0o666)
    while True:
        if not HOLD.exists():
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
        time.sleep(2)
    waited = time.time() - t0
    try:
        env = dict(os.environ)
        try:
            env["HIP_VISIBLE_DEVICES"] = subprocess.check_output(["hip-index", "igpu"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit("hip-index igpu failed: refusing to run the drape without the iGPU selected")
        sd = solver_dir()
        t1 = time.time()
        with open(log_path, "w") as lf:
            rc = subprocess.call([str(sd / "ppf-contact-solver"), "python", str(HERE / "drape_run.py"), str(inp),
                                  str(out)], cwd=str(sd), env=env, stdout=lf, stderr=subprocess.STDOUT)
        log_txt = Path(log_path).read_text(errors="replace")
        if "target/cpu" in log_txt:
            raise SystemExit("drape ran on the CPU backend: failed run")
        if rc != 0 or not Path(out).exists():
            raise SystemExit(f"drape failed rc={rc}; see {log_path}")
        return round(time.time() - t1, 1), round(waited, 1)
    finally:
        os.close(fd)


def drape_passes(job, jp):
    dj = job["drape"]
    results = dict(dj.get("result") or {})
    times = {}
    for key in dj.get("parts", []):
        if key in results:
            continue
        inp = Path(job["out"]) / f"drape_{key}_in.npz"
        out = Path(job["out"]) / f"drape_{key}_out.npz"
        pj = dict(job, steps=["build"], drape=dict(dj, export=str(inp), export_part=key, result=results))
        pj["out"] = str(Path(job["out"]) / f"drape_{key}_pass")
        pp = jp.with_name(jp.stem + f".drape_{key}.json")
        pp.write_text(json.dumps(pj, indent=1))
        t0 = time.time()
        rc = subprocess.call([BLENDER, "-b", "--factory-startup", "-t", "4", "--python", str(HERE / "bl_outfit.py"),
                              "--", str(pp)])
        if rc != 0 or not inp.exists():
            raise SystemExit(f"drape export for {key} failed (rc={rc})")
        t_exp = round(time.time() - t0, 1)
        t_drape, waited = run_drape(inp, out, Path(job["out"]) / f"drape_{key}.log")
        results[key] = str(out)
        times[key] = {"export_s": t_exp, "drape_s": t_drape, "igpu_wait_s": waited}
        log(f"draped {key}: {times[key]}")
    job["drape"] = dict(dj, result=results)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--cards", default=os.environ.get("CHAR_BLENDER_CARDS", "05"))
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--lock-whole-run", action="store_true",
                    help="old behaviour: hold one card for the whole Blender run (default: bl_outfit.py takes a "
                         "lock only around each render or bake: previews iGPU then 05, bakes 05)")
    a = ap.parse_args()
    job = json.loads(Path(a.job).read_text())
    fd = None
    bus = None
    if a.geometry_only:
        job["steps"] = [s for s in job.get("steps", []) if s in ("build", "export")]
    elif a.lock_whole_run:
        bus, fd, waited = take([c.strip() for c in a.cards.split(",") if c.strip()])
        log(f"card {bus} locked after {waited}s")
        job["bake_cards"] = job["preview_cards"] = [bus]
    if job.get("drape"):
        os.makedirs(job["out"], exist_ok=True)
        job["drape_times"] = drape_passes(job, Path(a.job))
    run_job = Path(a.job).with_suffix(".run.json")
    run_job.write_text(json.dumps(job, indent=1))
    t0 = time.time()
    cmd = [BLENDER, "-b", "--factory-startup", "-t", "4", "--python", str(HERE / "bl_outfit.py"), "--", str(run_job)]
    env = dict(os.environ)
    # show Blender only the GPUs this job may use: HIP sets up every visible device, and a reset of one it never
    # uses (the shared iGPU) otherwise aborts the bake
    cards = sorted(set(job.get("bake_cards", ["05"]) + job.get("preview_cards", ["c4", "05"])))
    try:
        idx = [subprocess.check_output(["hip-index", "igpu" if c == "c4" else c], text=True).strip() for c in cards]
        env["HIP_VISIBLE_DEVICES"] = ",".join(idx)
    except (OSError, subprocess.CalledProcessError):
        log("hip-index unavailable: Blender sees every GPU")
    try:
        rc = subprocess.call(cmd, pass_fds=(), env=env)
    finally:
        if fd is not None:
            os.close(fd)
    log(f"blender rc={rc} in {time.time() - t0:.1f}s (card {bus})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
