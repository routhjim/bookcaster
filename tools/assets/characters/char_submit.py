"""Submit a character editor job (char_render.py) to the asset scheduler's resident UE editor (run_asset.py --ue-job).

    python tools/assets/characters/char_submit.py job.json [--timeout S] [--id NAME]

No private editor and no card lock of our own: the scheduler places the job on its editor's card (swapping the
editor to our project when its queue allows) and returns {"rc", "result", "seconds", "error"}. The character job's
own result dict is also written to job["result"]. Settings:
    CHAR_UE_PROJECT    default $MH_UE_HOME/proj/CharLab/CharLab.uproject (MH_UE_HOME default ~/ue)
    CHAR_ASSET_CHAIN   the asset chain's tools/assets folder (run_asset.py); default: the exp-ue-assets worktree
                       beside this one
Stdlib only.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UE_HOME = Path(os.path.expanduser(os.environ.get("MH_UE_HOME") or os.environ.get("ASSETS_UE_HOME") or "~/ue"))
PROJECT = Path(os.environ.get("CHAR_UE_PROJECT", str(UE_HOME / "proj" / "CharLab" / "CharLab.uproject")))
CHAIN = Path(os.environ.get("CHAR_ASSET_CHAIN",
                            str(HERE.parents[3] / "exp-ue-assets" / "tools" / "assets")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--id")
    a = ap.parse_args()
    jp = Path(a.job).resolve()
    job = json.loads(jp.read_text())
    outs = [os.path.join(job["out"], f"{s['name']}.png") for s in job.get("shots", [])] if job.get("out") else []
    if job.get("result"):
        outs.append(job["result"])
        if os.path.exists(job["result"]):
            os.remove(job["result"])
    log_dir = Path(job.get("out") or jp.parent) / "sched"
    log_dir.mkdir(parents=True, exist_ok=True)
    sj = {"project": str(PROJECT), "script": str(HERE / "char_render.py"), "job": job, "outputs": outs,
          "timeout_s": a.timeout, "id": a.id or jp.stem, "log_dir": str(log_dir)}
    sp = jp.with_suffix(".sched.json")
    sp.write_text(json.dumps(sj, indent=1))
    t0 = time.time()
    rc = subprocess.call([sys.executable, str(CHAIN / "run_asset.py"), "--ue-job", str(sp)], cwd=str(CHAIN.parents[1]))
    print(time.strftime("%H:%M:%S"), f"scheduler rc={rc} in {time.time() - t0:.1f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
