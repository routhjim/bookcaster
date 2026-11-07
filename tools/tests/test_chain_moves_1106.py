"""11-06 (CHANGES 73, loop E15): a chain only when its shot moves - one yes/no ask per chain, cached; "no" leaves the
director's shots single; an answer that is neither keeps the chain. CPU only (the ask stubbed).
    ~/tts_reader/.venv/bin/python3 tools/tests/test_chain_moves_1106.py"""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import shot_flow as S
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


calls = []


def ask(msgs):
    calls.append(msgs[0]["content"])
    t = msgs[0]["content"]
    return "No." if "stands still" in t else ("Yes, it moves." if "runs" in t else "Hard to say.")


still = {"lines": ["the room stands still.", "the light stands still."], "shots": [2, 3]}
moves = {"lines": [{"line": "the man runs.", "bridge": False}, "the man stops."], "shots": [5, 6]}
odd = {"lines": ["a thing.", "another thing."], "shots": [8, 9]}
with tempfile.TemporaryDirectory() as td:
    c = Path(td) / "p.motion.json"
    logs = []
    got = [S.chain_moves(x, ask, c, logs.append) for x in (still, moves, odd)]
    check("no -> single shots; yes -> chain; neither -> chain kept", got == [False, True, True], got)
    check("one short ask per chain, the question and the lines in it", len(calls) == 3 and
          "actually moving or acting" in calls[0] and "1. the room stands still.\n2. the light" in calls[0]
          and "1. the man runs." in calls[1], calls)
    check("no concrete example in the ask", "{lines}" in S.MOTION_ASK and S.MOTION_ASK.count("\n") == 2)
    check("the drop is logged", any("no motion" in m for m in logs), logs)
    again = [S.chain_moves(x, ask, c, logs.append) for x in (still, moves, odd)]
    check("cached: no second ask, same answers", len(calls) == 3 and again == got, (len(calls), again))
    d = json.loads(c.read_text())
    check("cache keeps the reply and the lines", len(d) == 3 and all({"lines", "reply", "moves"} <= set(v) for v in d.values()))
src = (HERE / "shot_flow.py").read_text()
check("wired after the chain ask, before the chain tags", src.index("chain_moves(c, ask_") > src.index("chs = ask_chains(")
      and src.index("chain_moves(c, ask_") < src.index("for j_, (nums, head) in enumerate(chain_heads("))
print("ALL PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
