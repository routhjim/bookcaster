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
# (11-01: the mixed hold is any 2+ routes - quiet's "distilled" too; behaviour in ltx_runner/test_one_hold.py)
check("the runner builds both pipelines over one DiT in a mixed hold", "mixed = len(routes) > 1" in rn and
      'def make_pipe(dev, kind=None)' in rn and "else make_pipe(dev, k)[0], dev) for k in routes}" in rn)
check("each job takes its route from its fields (talk has audio)",
      "kind = job_route(j, routes)" in rn and 'if j.get("audio") and "a2vid" in routes:' in rn
      and "if j_a2v:" in rn and "elif j_ti2v:" in rn)
check("11-01: quiet clips in the action/talk hold (one gen call, --with-distilled)",
      'gen("hold" if held else kind, two + talk + held,' in sf and '"--with-distilled"' in sf
      and '("hold_jobs", two + talk + quiet)' in sf)
check("talk faults are counted defects, not retakes", "talk clip defect" in sf and '_t2", seed=RETAKE' in sf
      and sf.count('_t2", seed=RETAKE') == 1)
check("10-28: quiet clips on the dev DiT + distilled LoRA (no distilled model, no distilled encode)",
      '"--base", "dev", "--dlora-strength", str(QUIET_DLORA)' in sf and 'emb = f"qd_' in sf and
      'enc_dist.append' not in sf and '"base": "dev"' in sf)
print({"fails": fails}); sys.exit(1 if fails else 0)
