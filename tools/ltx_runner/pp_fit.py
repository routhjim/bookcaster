"""pp_fit.py <out>/pp_timing.jsonl [...] : fit per-block DiT time on each card + card-05 overhead per route, and
print the --split that minimises max(card A, card B) busy time (the 2-lane pipeline's throughput limit).

Model (per job j, from ltx_split_next.py --pp records):
  A(s) = s * kA_j + a_fixed_j          kA_j = a_dit_s / split        a_fixed_j = a_other_s + a_copy_s
  B(s) = (48 - s) * kB_j + b_fixed_j   kB_j = b_dit_s / (48 - split)  b_fixed_j = b_copy_s + b_decode_s
(b_decode_s = VAE decode moved to card B by --decode-b, 10-24; a route that has such records is fitted on those
only, since decoding on A or B moves the balance point)
Summed over a route's jobs, best s = argmin_s max(sum A(s), sum B(s)); jobs/min at 2 lanes ~ 60 / max(...) per job.
"""
import json, sys
from collections import defaultdict

NB = 48
recs = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
recs = [r for r in recs if r.get("pp") and r.get("b_dit_s", 0) > 0 and 0 < r["split"] < NB]
if not recs:
    sys.exit("no --pp records with timings")
by = defaultdict(list)
for r in recs:
    by[r["route"]].append(r)
for route, rs in sorted(by.items()):
    if any("b_decode_s" in r for r in rs):
        rs = [r for r in rs if "b_decode_s" in r]
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
