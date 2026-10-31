"""10-30 user ("a decent chunk in all segments, but also divide in natural breaks"): a long line's chained talk pieces
are cut where the voice pauses, every piece at least TALK_PIECE_MIN frames (talk-002477 was 8 s + 9 frames)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

def voice(total, pauses, seed=1):       # loud speech with syllable flicker, silent at each (start, length) pause
    r = np.random.default_rng(seed)
    e = 0.1 + 0.05 * r.random(total + 193)
    for a, n in pauses:
        e[a:a + n] = 0.001
    return e

def valid(ps, total, mx=193, ov=1):
    return (ps[0][0] == 0 and all((n - 1) % 8 == 0 and S.TALK_PIECE_MIN <= n <= mx for _, n in ps[:-1])
            and all(ps[i + 1][0] == ps[i][0] + ps[i][1] - ov for i in range(len(ps) - 1))
            and ps[-1][0] + ps[-1][1] >= total and ps[-1][1] <= mx)

ps = S.talk_pieces(202, voice(202, [(110, 10)]))          # 002477: a pause after the first sentence
check("8.4 s line with a pause at 4.6 s: two pieces cut in the pause, no 9-frame tail",
      valid(ps, 202) and len(ps) == 2 and 110 <= ps[0][1] - 1 <= 120 and ps[1][1] >= S.TALK_PIECE_MIN, ps)
ps = S.talk_pieces(150, voice(150, [(60, 8)]))
check("a line that fits in one piece stays one piece", ps == [(0, 153)], ps)
ps = S.talk_pieces(930, voice(930, [(170, 9), (330, 12), (480, 8), (610, 10), (790, 9)]))
cuts = [s + n - 1 for s, n in ps[:-1]]
check("38 s line: every cut lands in one of its pauses",
      valid(ps, 930) and all(any(a - 2 <= c <= a + n + 1 for a, n in [(170, 9), (330, 12), (480, 8), (610, 10),
                                                                    (790, 9)]) for c in cuts), ps)
ps = S.talk_pieces(380, voice(380, []))
check("no pause anywhere: the fewest pieces, near even", valid(ps, 380) and len(ps) == 2
      and abs(ps[0][1] - ps[1][1]) <= 16, ps)
ps = S.talk_pieces(353, voice(353, [(96, 10), (168, 10)]))
check("an extra piece isn't bought when a clean two-piece cut exists", valid(ps, 353) and len(ps) == 2, ps)
print({"fails": fails})
