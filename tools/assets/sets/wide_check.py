"""A built set's wide pictures, looked at: too dark, or nearly empty (11-27).

    python tools/assets/sets/wide_check.py SET_DIR/ue        # the measures and the verdict, printed

The set builder renders a few wide pictures of every set it builds (ue/<id>_wide_<n>.png). Nothing read them: sets
that came out nearly black, or as bare ground with nothing to see on it, were recorded "ok" and shots were staged in
them. This measures each picture's pixels (numpy and PIL, no model) and says of the whole set "ok", "too dark" or
"nearly empty". It only says so: the caller records it and logs it, the run goes on.

Per picture, on its brightness (0-255) at 256 x 144:
  median, p90     the middle and the 90th percentile brightness
  near_black      the share of the picture under NEAR_BLACK
  seen            the share at or over it (1 - near_black): what there is to see at all
  flat_low        the share of the lower half (the ground and what stands on it) in 8 x 8 blocks with almost no
                  contrast inside them: plain black, plain fog, an unlit or untextured plane

The thresholds, from the wide pictures of some twenty built sets (rooms, and places in the open by day and by night):
  SEEN_MIN 0.35   rooms and good places in the open showed 0.42-1.00 of the frame over near black (one camera of
                  one good room 0.30); the pictures of the sets too dark by eye 0.08-0.33. The same at night: a night set may be dark overall (a
                  median under 16 is usual there, and is no fault here), but a third of the frame must still be
                  seen. So the place's time of day is not asked for.
  FLAT_MAX 0.45   good pictures had 0.00-0.36 of the lower half flat (the highest: clean rooms with plain floors);
                  the bad ones 0.47-0.86.
A set fails a measure when at least half of its pictures do: one camera facing a near wall or a dark corner is no
fault of the set.

What this cannot see: ground that is lit and textured but bare (a plain with a few small things far off) measures
like a room's floor. That wants the pieces counted in the frame, not the pixels.
"""

import json
import sys
from pathlib import Path

SIZE = (256, 144)            # the picture as measured (16:9, a multiple of BLOCK both ways)
BLOCK = 8                    # the side of a block whose inner contrast is measured
NEAR_BLACK = 16.0            # brightness (of 255) under this: near black, nothing to see there
FLAT_SD = 1.5                # a block whose brightness spreads less than this: no contrast in it
SEEN_MIN = 0.35              # a picture showing less of its frame over near black: too dark
FLAT_MAX = 0.45              # a picture with more of its lower half flat: nearly empty
REASONS = {"dark": "too dark", "empty": "nearly empty"}


def measure(path) -> dict:
    """One picture's measures (see the module's words)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L").resize(SIZE, Image.BILINEAR), dtype=np.float32)
    low = a[SIZE[1] // 2:]
    sd = low.reshape(low.shape[0] // BLOCK, BLOCK, low.shape[1] // BLOCK, BLOCK).std(axis=(1, 3))
    black = float((a < NEAR_BLACK).mean())
    m = {"median": round(float(np.median(a)), 1), "p90": round(float(np.percentile(a, 90)), 1),
         "near_black": round(black, 3), "seen": round(1.0 - black, 3), "flat_low": round(float((sd < FLAT_SD).mean()), 3)}
    m["faults"] = [k for k, bad in (("dark", m["seen"] < SEEN_MIN), ("empty", m["flat_low"] > FLAT_MAX)) if bad]
    return m


def verdict(pictures: dict) -> dict:
    """{picture name: its measures} -> {"verdict": "ok" | "too dark" | "nearly empty" | "too dark, nearly empty" |
    "no pictures", "ok", "faults", "pictures"}. A fault is the set's when at least half of its pictures have it."""
    n = len(pictures)
    faults = [k for k in REASONS if n and 2 * sum(k in m["faults"] for m in pictures.values()) >= n]
    said = ", ".join(REASONS[k] for k in faults) if n else "no pictures"
    return {"verdict": said or "ok", "ok": bool(n) and not faults, "faults": faults,
            "limits": {"seen_min": SEEN_MIN, "flat_low_max": FLAT_MAX, "near_black": NEAR_BLACK}, "pictures": pictures}


def check_set(ue_dir) -> dict:
    """The verdict on a set's wide pictures (every *_wide_*.png in its ue/ folder)."""
    return verdict({q.name: measure(q) for q in sorted(Path(ue_dir).glob("*_wide_*.png"))})


def fault_line(place: str, res: dict) -> str:
    """The one loud line for the job log when the set failed ("" when it did not, or when it has no pictures)."""
    if res["ok"] or not res["pictures"]:
        return ""
    ms = list(res["pictures"].values())
    bits = []
    if "dark" in res["faults"]:
        n = sum("dark" in m["faults"] for m in ms)
        bits.append(f"too dark ({n} of {len(ms)} show under {SEEN_MIN:.0%} of the frame over near black, "
                    f"least {min(m['seen'] for m in ms):.0%})")
    if "empty" in res["faults"]:
        n = sum("empty" in m["faults"] for m in ms)
        bits.append(f"nearly empty ({n} of {len(ms)} have over {FLAT_MAX:.0%} of the lower half with nothing in it, "
                    f"most {max(m['flat_low'] for m in ms):.0%})")
    return f"!!! {place}: the set's wide pictures are " + " and ".join(bits) + ": shots staged in it will show it"


if __name__ == "__main__":
    r = check_set(sys.argv[1])
    print(json.dumps(r, indent=1))
    print(fault_line(Path(sys.argv[1]).resolve().parent.name, r) or r["verdict"])
