"""pp_fit.py [--decode-b] <out>/pp_timing.jsonl [...] : fit per-block DiT time on each card + card-05 overhead per route, and
print the --split that minimises max(card A, card B) busy time (the 2-lane pipeline's throughput limit).

Model (per job j, from ltx_split_next.py --pp records):
  A(s) = s * kA_j + a_fixed_j          kA_j = a_dit_s / split        a_fixed_j = a_other_s + a_copy_s
  B(s) = (48 - s) * kB_j + b_fixed_j   kB_j = b_dit_s / (48 - split)  b_fixed_j = b_copy_s + b_decode_s
(b_decode_s = VAE decode moved to card B by --decode-b, 10-24; a route that has such records is fitted on those
only, since decoding on A or B moves the balance point)
Summed over a route's jobs, best s = argmin_s max(sum A(s), sum B(s)); jobs/min at 2 lanes ~ 60 / max(...) per job.
With --decode-b (10-26: decode on card B is the default for every route) a record made with decode on card A has an
estimated decode moved from A to B: frames x the per-frame b_decode_s of the records that did decode on B (any route,
same 1152x640 frames; else DEC_PER_FRAME).
"""
import json, sys
from collections import defaultdict

NB = 48
DEC_PER_FRAME = 0.094            # s/frame on card B (10-26: ch2 quiet clips, 5.35 s for 57 frames)
argv = [a for a in sys.argv[1:] if a != "--decode-b"]
decode_b = "--decode-b" in sys.argv[1:]
recs = [json.loads(l) for f in argv for l in open(f) if l.strip()]
recs = [r for r in recs if r.get("pp") and r.get("b_dit_s", 0) > 0 and 0 < r["split"] < NB]
if not recs:
    sys.exit("no --pp records with timings")
measured = [r for r in recs if r.get("b_decode_s") and r.get("frames")]
per_frame = sum(r["b_decode_s"] for r in measured) / sum(r["frames"] for r in measured) if measured else DEC_PER_FRAME
by = defaultdict(list)
for r in recs:
    by[r["route"]].append(r)
for route, rs in sorted(by.items()):
    if any("b_decode_s" in r for r in rs):
        rs = [r for r in rs if "b_decode_s" in r]
    elif decode_b:                   # decoded on A when measured: move the decode to B
        rs = [dict(r, a_other_s=max(0.0, r.get("a_other_s", 0) - per_frame * r.get("frames", 0)),
                   b_decode_s=per_frame * r.get("frames", 0)) for r in rs]
    kA = sum(r["a_dit_s"] / r["split"] for r in rs)
    kB = sum(r["b_dit_s"] / (NB - r["split"]) for r in rs)
    aF = sum(r.get("a_other_s", 0) + r["a_copy_s"] for r in rs)
    bF = sum(r["b_copy_s"] + r.get("b_decode_s", 0) for r in rs)
    best = min(range(4, NB - 3), key=lambda s: max(s * kA + aF, (NB - s) * kB + bF))
    A, B = best * kA + aF, (NB - best) * kB + bF
    cur = sorted({r["split"] for r in rs})
    n = len(rs)
    print(f"[{route}] {n} jobs, splits seen {cur}: per block A {kA / n:.3f}s B {kB / n:.3f}s per job "
          f"(B/A speed ratio {kB / kA:.2f}); card-A fixed {aF / n:.1f}s/job, B fixed {bF / n:.1f}s/job")
    for s in sorted(set(cur + [best])):
        a, b = s * kA + aF, (NB - s) * kB + bF
        print(f"   split {s:2d}: A {a / n:6.1f}s  B {b / n:6.1f}s per job -> bottleneck {max(a, b) / n:6.1f}s "
              f"= {60 * n / max(a, b):.2f} jobs/min at 2 lanes")
    print(f"   BEST --split {best}: A {A / n:.1f}s B {B / n:.1f}s per job")
