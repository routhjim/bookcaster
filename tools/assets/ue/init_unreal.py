"""Start-up shim, copied by run_asset.py into <project>/Content/Python/init_unreal.py.

One shim for the one engine project (props, sets, characters, scenes): the character tools' shim reads MH_JOB, the
chain's ASSETS_UE_JOB; this one reads either (the scheduler sets both to the same file).
Does nothing unless ASSETS_UE_JOB (or MH_JOB) names a job JSON whose "script" key is a Python file to run (the render job,
which needs a live editor with ticking frames: -ExecutePythonScript quits a few frames after its script returns).
Runs once per process even if the start-up script is imported again.
"""

import json
import os
import runpy

import unreal

_job = os.environ.get("ASSETS_UE_JOB") or os.environ.get("MH_JOB")
_spec = json.load(open(_job)) if _job and os.path.exists(_job) else {}
if _spec.get("script") and not getattr(unreal, "_assets_job_started", False) \
        and not getattr(unreal, "_mh_job_started", False):
    unreal._assets_job_started = unreal._mh_job_started = True     # one run per process
    unreal.log_warning(f"ASSETS shim: running {_spec['script']}")
    runpy.run_path(_spec["script"], run_name="__assets_job__")
