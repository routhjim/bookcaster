"""10-28 user: action and talk share one LTX load. The shots flow makes ONE gen call for both (one jobs file, both
route flags); the runner builds both pipelines over the one DiT and routes each job by its fields; talk clips are
body-checked as they land as counted defects (lean: no retake path). Source-level: the GPU run is itest_15s."""
import re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    ok or fails.append(name)
sf = (HERE.parent / "shot_flow.py").read_text()
rn = (HERE.parent / "ltx_runner" / "ltx_split_next.py").read_text()
check("one gen call for action + talk with both route flags",
      'kind = "two_talk" if two and talk' in sf and '["--ti2v"] * bool(two) + ["--a2vid"] * bool(talk)' in sf
      and 'gen("two", two,' not in sf and 'gen("talk", talk,' not in sf)
check("the combined jobs file is written", '("two_talk_jobs", two + talk)' in sf)
check("the runner builds both pipelines over one DiT in a mixed hold", "mixed = a2v and ti2v" in rn and
      'def make_pipe(dev, kind=None)' in rn and '"ti2v": prepare(make_pipe(dev, "ti2v")[0], dev)' in rn)
check("each job takes its route from its fields (talk has audio)",
      'kind = "a2vid" if j.get("audio") else "ti2v"' in rn and "if j_a2v:" in rn and "elif j_ti2v:" in rn)
check("talk faults are counted defects, not retakes", "talk clip defect" in sf and '_t2", seed=RETAKE' in sf
      and sf.count('_t2", seed=RETAKE') == 1)
print({"fails": fails}); sys.exit(1 if fails else 0)
