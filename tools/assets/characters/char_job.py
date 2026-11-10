"""Run one character editor job through the MetaHuman lab runner, respecting the box's GPU hold.

    python tools/assets/characters/char_job.py job.json [--name NAME] [--timeout S]

Waits (polling) while the hold file exists ($CHAR_GPU_HOLD, default $MH_UE_HOME/GPU_HOLD, i.e. ~/ue/GPU_HOLD), then
starts `metahuman/mh_run.py render job.json` with the character lab's defaults:
    MH_UE_PROJECT  default $MH_UE_HOME/proj/CharLab/CharLab.uproject
    MH_LAB         default $MH_UE_HOME/lab/character
    MH_GPU_CARDS   default "65" (UE's card on this box)
    MH_EXTRA_ARGS  adds -NoMetaHumanAccountPortalLoginFallback (cloud steps fail instead of opening a sign-in page)
                   and -notraceserver (a forked trace server would inherit and keep the card's lock fd)
Stdlib only.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UE_HOME = Path(os.path.expanduser(os.environ.get("MH_UE_HOME") or os.environ.get("ASSETS_UE_HOME") or "~/ue"))
HOLD = Path(os.environ.get("CHAR_GPU_HOLD", str(UE_HOME / "GPU_HOLD")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--name")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    t0 = time.time()
    while HOLD.exists():
        if time.time() - t0 < 1:
            print(time.strftime("%H:%M:%S"), f"GPU hold present ({HOLD}); waiting", flush=True)
        time.sleep(10)
    if time.time() - t0 > 1:
        print(time.strftime("%H:%M:%S"), f"hold cleared after {time.time() - t0:.0f}s", flush=True)
    env = dict(os.environ)
    env.setdefault("MH_UE_PROJECT", str(UE_HOME / "proj" / "CharLab" / "CharLab.uproject"))
    env.setdefault("MH_LAB", str(UE_HOME / "lab" / "character"))
    env.setdefault("MH_GPU_CARDS", "65")
    extra = env.get("MH_EXTRA_ARGS", "").split()
    for flag in ("-NoMetaHumanAccountPortalLoginFallback", "-notraceserver"):
        if flag not in extra:
            extra.append(flag)
    env["MH_EXTRA_ARGS"] = " ".join(extra)
    cmd = [sys.executable, str(HERE.parent / "metahuman" / "mh_run.py"), "render", str(Path(a.job).resolve()),
           "--timeout", str(a.timeout)]
    if a.name:
        cmd += ["--name", a.name]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
