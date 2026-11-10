"""Start-up shim, copied by mh_run.py into <project>/Content/Python/init_unreal.py.

Does nothing unless MH_JOB names a job JSON whose "script" key is a Python file to run inside the live editor.
Runs once per process.
"""

import json
import os
import runpy

import unreal

_job = os.environ.get("MH_JOB")
_spec = json.load(open(_job)) if _job and os.path.exists(_job) else {}
if _spec.get("script") and not getattr(unreal, "_mh_job_started", False):
    unreal._mh_job_started = True
    unreal.log_warning(f"MH shim: running {_spec['script']}")
    runpy.run_path(_spec["script"], run_name="__mh_job__")
