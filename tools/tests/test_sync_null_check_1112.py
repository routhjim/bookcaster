"""11-12: the synthetic lip-sync checks' own mechanics (no SyncNet, no clips): the voiced frames of a clip whose
sound is moved later, a voiced span cut to its first windows, and the span buckets.
   python3 tools/tests/test_sync_null_check_1112.py"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import sync_null_check as N

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

v = [False] * 3 + [True] * 5 + [False] * 4
check("sound 2 frames later: its voiced frames are 2 later, the length kept",
      N.shifted(v, 2) == [False] * 5 + [True] * 5 + [False] * 2, N.shifted(v, 2))
check("moved past the end: what is left", N.shifted(v, 6) == [False] * 9 + [True] * 3, N.shifted(v, 6))
check("no shift: unchanged", N.shifted(v, 0) == v)
w = [True, False, True, True, False, True]
check("the first 2 voiced windows of a span", N.first_span(w, 2) == [True, False, True, False, False, False],
      N.first_span(w, 2))
check("more than there are: all of them", N.first_span(w, 9) == w)
check("another clip's voiced frames fitted to this clip's length", N.fit([True] * 3, 5) == [True] * 3 + [False] * 2
      and N.fit([True] * 9, 4) == [True] * 4)
check("span buckets", [N.bucket(x) for x in (0, 29, 30, 59, 60, 500, None)] ==
      ["0-29", "0-29", "30-59", "30-59", "60+", "60+", "0-29"], [N.bucket(x) for x in (0, 29, 30, 59, 60, 500, None)])
check("the search cannot see the longest shift (so it must not pass)", max(N.SHIFTS) >= N.S.SYNC_VSHIFT)
check("the cut spans straddle the floor", min(N.SPANS) < N.S.SYNC_ROWS_SURE < max(N.SPANS))

print({"fails": fails})
sys.exit(1 if fails else 0)
