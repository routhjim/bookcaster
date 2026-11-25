"""One paint + rig at a time when memory is short; several when there is room."""
import sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import characters as C

ok = True
def check(what, cond):
    global ok
    ok &= bool(cond)
    print("PASS" if cond else "FAIL", what)

check("room for several: no lock is held", C.rig_turn(lambda: C.RIG_ROOM_GB + 1) is None and not C._RIG_TURN.locked())
t = C.rig_turn(lambda: 5.0)
check("short of memory: the lock is held", t is C._RIG_TURN and t.locked())
live, peak = [0], [0]
def worker():
    k = C.rig_turn(lambda: 5.0)
    live[0] += 1; peak[0] = max(peak[0], live[0]); time.sleep(0.05); live[0] -= 1
    k.release()
th = [threading.Thread(target=worker) for _ in range(3)]
[x.start() for x in th]
time.sleep(0.1)
check("others wait while it is held", peak[0] == 0)
t.release()
[x.join() for x in th]
check("then they go one at a time", peak[0] == 1 and not C._RIG_TURN.locked())
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)
