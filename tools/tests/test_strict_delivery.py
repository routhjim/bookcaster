"""Strict all-new delivery (user 10-21: "no fallbacks ... it needs to be all new"), synthetic, CPU only, seconds.

    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_strict_delivery.py

Checks: strict is the full-3D default and REMAKE_ALLOW_FALLBACK turns it off; a clip counts only when it is newer
than its moment's current key (a redraw since, or no key = an I2V fallback, does not). Prints PASS/FAIL per check, exit 1 on any failure.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import remake_chapter as R  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


os.environ["REMAKE_FULL3D"] = "1"
os.environ.pop("REMAKE_ALLOW_FALLBACK", None)
check("strict by default in full 3D", R.strict())
os.environ["REMAKE_ALLOW_FALLBACK"] = "1"
check("REMAKE_ALLOW_FALLBACK turns it off", not R.strict())
os.environ.pop("REMAKE_ALLOW_FALLBACK")

with tempfile.TemporaryDirectory() as td:
    cx = SimpleNamespace(OUT=Path(td))
    lo = Path(td) / "ltx" / "out"
    lo.mkdir(parents=True)
    now = time.time()
    for mid, key_age, clip_age in (("a", 600, 3600), ("b", 3600, 60), ("c", None, 60)):   # seconds ago
        if key_age is not None:
            k = Path(td) / "stage" / mid / "key.png"
            k.parent.mkdir(parents=True)
            k.write_bytes(b"k")
            os.utime(k, (now - key_age, now - key_age))
        c = lo / f"r_{mid}.mp4"
        c.write_bytes(b"x")
        os.utime(c, (now - clip_age, now - clip_age))
    check("a clip older than its (redrawn) key is not new", R.new_clip(cx, {"id": "a"}) is None)
    check("a clip made from the current key is new", R.new_clip(cx, {"id": "b"}) == lo / "r_b.mp4")
    check("a clip without a key (an I2V fallback) is not new", R.new_clip(cx, {"id": "c"}) is None)
    check("a missing clip is missing", R.new_clip(cx, {"id": "d"}) is None)

print(f"{len(FAILS)} failure(s)" if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
