"""The language model gives the iGPU back before anything else loads a model on it, and a command that
started it stops it again."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tts_reader import animate, bookcaster

ok = True
def check(what, cond):
    global ok
    ok &= bool(cond)
    print("PASS" if cond else "FAIL", what)

state = {"up": True, "stops": 0, "flush": 0, "starts": 0}
animate.strata_up = lambda: state["up"]
animate.STRATA_STOP_WAIT_S = 0
animate._flush = lambda: state.__setitem__("flush", state["flush"] + 1)
class R:
    returncode = 0
def fake_run(cmd, **k):
    if cmd[:3] == ["systemctl", "--user", "stop"]:
        state["stops"] += 1
        state["up"] = False
    return R()
animate.subprocess.run = fake_run
said = []
check("up: it is stopped and its memory flushed", animate.strata_down(said.append, "why") and state == {"up": False, "stops": 1, "flush": 1, "starts": 0})
check("down: nothing is done", not animate.strata_down(said.append) and state["stops"] == 1 and state["flush"] == 1)

def fake_igpu(tenant, log=print):
    state["starts"] += not state["up"]
    state["up"] = True
animate.igpu = fake_igpu
with bookcaster._llm_lent(said.append):
    check("a command gets the language model", state["up"] and state["starts"] == 1)
check("... and stops the one it started", not state["up"] and state["stops"] == 2)
state["up"] = True
with bookcaster._llm_lent(said.append):
    pass
check("one that was already up is left up", state["up"] and state["stops"] == 2)

import characters as C
state["up"] = True
C.igpu_room()
check("the vision judge takes the iGPU from it", not state["up"] and state["stops"] == 3)
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)
