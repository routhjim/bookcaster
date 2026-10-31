"""timeline_report.py <out>/timeline.jsonl [--skip S]: per-card occupancy of a --pp run from the per-call timeline
ltx_split_next.py writes (union of busy intervals, so two lanes overlapping on a card count once), and what both lanes
were doing while card A sat idle. --skip drops the first S seconds after both lanes are running (compile warm-up)."""
import json
import sys


def union(iv):
    tot, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            tot += (cur[1] - cur[0]) if cur else 0.0
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + ((cur[1] - cur[0]) if cur else 0.0)


def main():
    args = sys.argv[1:]
    skip = float(args[args.index("--skip") + 1]) if "--skip" in args else 0.0
    E = [json.loads(ln) for ln in open(args[0]) if ln.strip()]
    lanes = {}
    for c, ln, k, a, b in E:
        lo, hi = lanes.get(ln, (a, b))
        lanes[ln] = (min(lo, a), max(hi, b))
    lo = max(a for a, _ in lanes.values()) + skip
    hi = min(b for _, b in lanes.values())
    W = hi - lo
    if W <= 0:
        sys.exit("no window with every lane running")

    def busy(card, kinds):
        return union([(max(a, lo), min(b, hi)) for c, _, k, a, b in E if c == card and k in kinds and b > lo and a < hi])
    print(f"window {W:.0f} s with {len(lanes)} lanes running")
    print(f"card A: DiT {busy('A', {'dit'}) / W:.0%}, with copies {busy('A', {'dit', 'copy', 'copyback'}) / W:.0%}")
    print(f"card B: DiT {busy('B', {'dit'}) / W:.0%}, decode {busy('B', {'decode'}) / W:.0%}, "
          f"any work {busy('B', {'dit', 'decode', 'copyin', 'copyout'}) / W:.0%}")
    states, step = {}, 0.05
    for i in range(int(W / step)):
        t = lo + i * step
        if any(c == "A" and k == "dit" and a <= t < b for c, _, k, a, b in E):
            continue
        now = []
        for ln in lanes:
            s = next((f"{c}:{k}" for c, l2, k, a, b in E if l2 == ln and a <= t < b), "host")
            now.append(s)
        key = " | ".join(sorted(now))
        states[key] = states.get(key, 0.0) + step
    print(f"card A idle {sum(states.values()):.0f} s; the lanes meanwhile:")
    for k, v in sorted(states.items(), key=lambda x: -x[1])[:8]:
        print(f"  {v:6.1f} s  {k}")


if __name__ == "__main__":
    main()
