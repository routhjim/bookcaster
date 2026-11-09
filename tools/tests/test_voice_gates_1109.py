"""The 11-09 voice-mining liveliness gates (CPU only, a few seconds).

    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_voice_gates_1109.py

User 11-09: keep robotic, smith-like voices out of the mined reference clips. tools/mine_voice.py gates candidates on
phrasing, "flat on every channel" (pitch AND loudness), pace variation and shape (10-15 s, sentence end), and ranks
survivors by a liveliness score. Checks: the verdict logic on measure dicts; the clip trimming and ranking; the miner
asks Strata at the pipeline's URL; and, on the REAL library clips (read only, skipped when absent or when the miner's
venv is missing), smith.wav fails while the liked kilmer/weiss/melin staging clips pass.
Prints PASS/FAIL per check, exit 1 on any failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import mine_voice as M  # noqa: E402
from tts_reader import bookcaster  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# measure dicts shaped like the 11-09 analysis's reference clips
SMITH = dict(duration_s=9.0, pauses=2, pauses_per_10s=2.2, phrase_mean_s=2.57, f0_sd_st=2.81, f0_range_st=9.2,
             syll_loud_sd_db=2.68, swell_db_s=1.74, pace_cv=0.05)
KILMER = dict(duration_s=10.6, pauses=4, pauses_per_10s=3.8, phrase_mean_s=1.78, f0_sd_st=3.29, f0_range_st=10.8,
              syll_loud_sd_db=3.25, swell_db_s=3.90, pace_cv=0.14)
WEISS = dict(duration_s=12.1, pauses=6, pauses_per_10s=5.0, phrase_mean_s=1.40, f0_sd_st=4.42, f0_range_st=14.0,
             syll_loud_sd_db=2.45, swell_db_s=1.41, pace_cv=0.24)


def verdict_logic():
    fails, s_smith = M.live_verdict(SMITH, "and they")
    check("smith-like measures fail every gate", len(fails) == 5, "; ".join(fails))
    for name, m in (("kilmer", KILMER), ("weiss", WEISS)):
        f, _ = M.live_verdict(m, "It ended there.")
        check(f"{name}-like measures pass", not f, "; ".join(f))
    # one live channel is enough: weiss is loudness-flat but pitch-live
    flat_both = dict(WEISS, f0_sd_st=2.5, f0_range_st=8.0)
    f, _ = M.live_verdict(flat_both, "Done.")
    check("pitch AND loudness low -> flat on every channel", any("flat on every channel" in x for x in f), f)
    f, _ = M.live_verdict(dict(WEISS, pace_cv=float("nan")), "Done.")
    check("unmeasurable pace variation fails", any("unmeasurable" in x for x in f), f)
    f, _ = M.live_verdict(dict(KILMER, pauses_per_10s=2.0, phrase_mean_s=2.1), "Done.")
    check("short phrases alone satisfy phrasing", not f, f)
    f, _ = M.live_verdict(KILMER, None)
    check("no transcript -> ending not judged", not f, f)
    ranked = M.rank_survivors([{"score": s, "id": i} for i, s in enumerate([1.1, 1.6, 1.3])])
    check("survivors ranked liveliest first (cand1 = best)", [c["id"] for c in ranked] == [1, 2, 0])
    _, s_k = M.live_verdict(KILMER, None)
    _, s_w = M.live_verdict(WEISS, None)
    check("liked clips outscore smith", min(s_k, s_w) > s_smith, f"{s_smith:.2f} vs {s_k:.2f}/{s_w:.2f}")


def trimming():
    seg = lambda i, a, b, t: {"i": i, "start": a, "end": b, "text": t}  # noqa: E731
    span = [seg(0, 0, 4, "One, two"), seg(1, 4, 8, "three."), seg(2, 8, 11, "four five"), seg(3, 11, 12.5, "six.")]
    got = M.trim_span(span)
    check("trim runs on to the first sentence end past 10 s", got is not None and len(got) == 4)
    long = [seg(0, 0, 9, "a"), seg(1, 9, 16, "b."), seg(2, 16, 17, "c.")]
    check("trim gives up past 15.5 s without a sentence end", M.trim_span(long) is None)
    check("ends_sentence: quote after full stop", M.ends_sentence("He went home.”"))
    check("ends_sentence: comma is mid-sentence", not M.ends_sentence("and they,"))


def strata_url():
    check("miner asks Strata at the pipeline's URL", M.LLM_URL == bookcaster.LLM_URL and ":8080" not in M.LLM_URL,
          M.LLM_URL)
    check("miner uses the pipeline's model name", M.LLM_MODEL == bookcaster.LLM_MODEL)


LIB = Path(os.environ.get("F5_VOICES_DIR") or Path.home() / ".local/share/tts_reader/f5_voices").expanduser()
MINER_PY = Path(os.environ.get("MINE_VOICE_PY", Path.home() / "tts-ab/.venv-qwen/bin/python"))
REAL = {"smith.wav": False, "staging/kilmer.neutral.cand2.wav": True, "staging/weiss.neutral.cand1.wav": True,
        "staging/melin.neutral.cand1.wav": True}


def real_clips():
    clips = [LIB / c for c in REAL]
    if not MINER_PY.exists() or not all(c.exists() for c in clips):
        print("SKIP real clips (library clips or the miner's venv absent)")
        return
    code = ("import json, sys; from pathlib import Path; sys.path.insert(0, 'tools'); import mine_voice as M; "
            "print(json.dumps({p: M.check_clip(Path(p))[1:] for p in sys.argv[1:]}))")
    out = subprocess.run([str(MINER_PY), "-c", code, *map(str, clips)], cwd=HERE.parent, capture_output=True,
                         text=True, timeout=300)
    if out.returncode:
        check("real-clip measures ran", False, out.stderr[-500:])
        return
    res = json.loads(out.stdout.strip().splitlines()[-1])
    for rel, should_pass in REAL.items():
        fails, score = res[str(LIB / rel)]
        check(f"{rel} {'passes' if should_pass else 'fails'}", (not fails) == should_pass,
              f"score {score:.2f}; " + "; ".join(fails))
    smith_fails = res[str(LIB / "smith.wav")][0]
    check("smith fails on liveliness, not only on shape",
          any("flat on every channel" in f for f in smith_fails) and any("long phrases" in f for f in smith_fails))


verdict_logic()
trimming()
strata_url()
real_clips()
print("FAILED: " + ", ".join(FAILS) if FAILS else "ALL PASS")
sys.exit(1 if FAILS else 0)
