"""10-31: `bookcaster stop` stops the language model when the stopped job started it (box_before had no "strata"),
and leaves it alone when it was up before the job. No real servers are touched (systemctl and the lease are faked).
Run with bookcaster's python: .venv/bin/python3 tools/tests/test_stop_started_llm_1031.py"""
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tts_reader import animate, bookcaster as BC, boxlease  # noqa: E402

fails, calls = [], []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


class _R:
    returncode, stdout, stderr = 0, "", ""


BC.subprocess.run = lambda cmd, **k: calls.append(cmd) or _R()
boxlease.lease = lambda *a, **k: contextlib.nullcontext()
boxlease.busy_cards = lambda: []
BC._unit_active = lambda u: False
animate._servers = lambda *a, **k: True
animate.igpu = lambda *a, **k: calls.append(["igpu"] + list(a[:1]))


def run(before, now):
    calls.clear()
    BC.box_state = lambda: list(now)
    logs = []
    BC.restore_idle(Path("/nonexistent/book"), {"box_before": before}, logs.append)
    return [c for c in calls if c[:3] == ["systemctl", "--user", "stop"] and c[3] == animate.STRATA_UNIT], logs


stops, logs = run([], ["strata"])
check("the job started the language model: stopped", len(stops) == 1)
check("and said so, not 'nothing to restart'", any("language model stopped" in x for x in logs)
      and not any("nothing to restart" in x for x in logs))
stops, logs = run(["strata"], ["strata"])
check("it was up before the job: left up", not stops and any("nothing to restart" in x for x in logs))
stops, logs = run(["strata"], [])
check("the job paused it: brought back, not stopped", not stops and ["igpu", "strata"] in calls)
stops, logs = run([], [])
check("never up: nothing", not stops)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
