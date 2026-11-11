"""Picture to 3D: one drawing in, a textured .glb out, with TRELLIS.2 (MIT) through the trellis.cpp port (MIT).

    python tools/assets/img3d/img3d.py DRAWING.png OUT.glb [--res 1024|1536] [--seed N] [--atlas 4096]
                                       [--build vulkan|rocm] [--device 65|05|igpu] [-- EXTRA trellis-cli ARGS]

Run it through the line (it takes the device's lock itself, as the other tools do):

    python tools/assets/run_asset.py --line-cmd engine shape --id NAME --detach -- \
        python tools/assets/img3d/img3d.py DRAWING.png OUT.glb

The engine station borrows the engine card while the engine has no work (measured 11-11: the image model keeps the
Blender card busy, the iGPU is ~2.6x slower). The device comes from --device, else the line's station
(ASSETS_LINE_STATION); there is no CPU fallback. Writes beside OUT.glb: OUT.ply (the raw point cloud the port also
writes), OUT_base.png (the base-colour atlas), OUT_cutout.png (the background cut-out it worked from), OUT.log (the
port's verbose log, each line stamped with seconds since start) and OUT.json (device, build, settings, seconds per
stage, mesh size). The port gives base colour only (no normal or roughness maps); derive those downstream.

Environment:
    ASSETS_TRELLIS_HOME    the port's install (bin-vulkan/, bin-rocm/)     [$ASSETS_UE_HOME/tools/trellis]
    ASSETS_TRELLIS_MODELS  the GGUF weights                                [~/models/trellis2-gguf]
    ASSETS_TRELLIS_OMP     folder holding libomp.so (ROCm build only)      [unset: the ROCm build will not start]
    ASSETS_HIP_INDEX       helper mapping a device to its HIP index        [~/bin/hip-index]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

HOME_T = Path(os.path.expanduser(os.environ.get("ASSETS_TRELLIS_HOME", str(C.UE_HOME / "tools" / "trellis"))))
MODELS = Path(os.path.expanduser(os.environ.get("ASSETS_TRELLIS_MODELS", "~/models/trellis2-gguf")))
OMP = os.environ.get("ASSETS_TRELLIS_OMP")
HIP_INDEX = Path(os.path.expanduser(os.environ.get("ASSETS_HIP_INDEX", "~/bin/hip-index")))
STAGES = {"1": "preprocess", "2": "condition", "3": "structure", "4": "shape_flow", "5": "shape_decode",
          "6": "texture", "7": "write_mesh"}


def log(msg):
    print(f"[img3d] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def take_lock(dev, poll=2.0):
    path = f"/tmp/gpu-{dev}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    t, said = time.time(), False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not said:
                log(f"{path} is held: polling")
                said = True
            time.sleep(poll)
    if said:
        log(f"got {path} after {time.time() - t:.0f} s")
    return fd


def device_env(dev, build):
    """Environment that shows the port exactly one device: dev (a card's PCI bus or "igpu")."""
    bus = C.IGPU_BUS if dev == "igpu" else dev
    bindir = HOME_T / f"bin-{build}"
    env = dict(os.environ)
    if build == "vulkan":
        env.update(LD_LIBRARY_PATH=str(bindir), DRI_PRIME=f"pci-0000_{bus}_00_0!",
                   MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE="1")
        env.pop("MESA_VK_DEVICE_SELECT", None)
    else:
        if not OMP:
            raise SystemExit("the ROCm build needs libomp.so: set ASSETS_TRELLIS_OMP to its folder")
        idx = subprocess.run([str(HIP_INDEX), dev], capture_output=True, text=True, check=True).stdout.strip()
        env.update(LD_LIBRARY_PATH=f"{bindir}:{OMP}", HIP_VISIBLE_DEVICES=idx)
        if dev == "igpu":
            env["HSA_CU_MASK"] = f"{idx}:0-38"              # one CU left for the desktop
    return env, bindir


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--res", type=int, default=1024, choices=(512, 1024, 1536))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atlas", type=int, default=4096)
    ap.add_argument("--build", default="vulkan", choices=("vulkan", "rocm"))
    ap.add_argument("--device", default=None, help="65, 05 or igpu (default: the line's station)")
    a = ap.parse_args(argv)
    dev = a.device
    if dev is None:
        st = os.environ.get("ASSETS_LINE_STATION", "")
        dev = st[4:] if st.startswith("card") else (st or None)
    if dev not in ("igpu",) and not (dev and re.fullmatch(r"[0-9a-f]{2}", dev)):
        raise SystemExit("no device: pass --device 65|05|igpu or run through the line (never the CPU)")
    image, out = Path(a.image).resolve(), Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    env, bindir = device_env(dev, a.build)
    cmd = [str(bindir / "trellis-cli"), str(image), str(out), "--models", str(MODELS), "--res", str(a.res),
           "--atlas", str(a.atlas), "--seed", str(a.seed), "--webp", "off", "--dump-bg", "-v", "--gpu", "0"] + extra
    fd = take_lock(dev)
    t0 = time.time()
    marks, stage_t, cur = [], {}, None
    logp = out.with_suffix(".log")
    log(f"device {dev} build {a.build} res {a.res}: {' '.join(cmd[1:])}")
    with open(logp, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        devname = None
        for ln in p.stdout:
            t = time.time() - t0
            f.write(f"{t:8.1f} {ln}")
            f.flush()
            m = re.match(r"\[(\d)/\d\]", ln)
            if m:
                if cur:
                    stage_t[cur] = round(t - marks[-1], 1)
                cur = STAGES.get(m.group(1), m.group(1))
                marks.append(t)
                log(f"{t:6.0f}s {ln.strip()[:120]}")
            if devname is None and "using " in ln and "MB)" in ln:
                devname = ln.strip()
            if ln.startswith("ggml_vulkan: 0 =") or "ROCm" in ln and "Device 0" in ln:
                log(ln.strip()[:160])
        rc = p.wait()
    t_all = time.time() - t0
    if cur and marks:
        stage_t[cur] = round(t_all - marks[-1], 1)
    os.close(fd)
    rep = {"image": str(image), "glb": str(out), "device": dev, "build": a.build, "res": a.res, "seed": a.seed,
           "atlas": a.atlas, "extra": extra, "rc": rc, "seconds": round(t_all, 1), "stage_s": stage_t,
           "backend": devname, "log": str(logp), "time": time.ctime()}
    txt = logp.read_text()
    m = re.search(r"decimated postproc:.*F=(\d+)", txt) or re.search(r"decimate_qem_vk.*F \d+->(\d+)", txt)
    rep["faces"] = int(m.group(1)) if m else None
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    log(f"rc {rc} in {t_all:.0f}s; stages {stage_t}")
    if rc != 0 or not out.exists():
        raise SystemExit(rc or 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
