"""11-06 user: "fill them with LoRA training" - the cast's LoRAs train on both XTX cards WHILE the director plans
(they sat idle ~30 min under it), each build judging on its own card while the language model holds the iGPU.
  - charlora: a card's build asks the iGPU helper only when the language model is down (a fake systemctl on PATH);
  - shot_flow.run: the language model up first, then the builds, the director running beside them, the keys
    started after it with the builds' gate (11-08: drawn as each LoRA lands, test_keys_as_loras_land_1108), the
    job's wait on every build after them; a person's missing LoRA stops the run; nothing to train = as before.
CPU only, every step faked (no card, no service: systemctl is a fake, nothing is really started or stopped).
    PATH=<fakebin>:$PATH ~/tts_reader/.venv/bin/python3 tools/tests/test_lora_beside_director_1106.py"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


TMP = Path(tempfile.mkdtemp(prefix="lora-beside-"))
# a fake systemctl first on PATH: `is-active -q strata-eval` answers from a file; anything else is refused loudly
fb = TMP / "fakebin"
fb.mkdir()
state = TMP / "strata_active"
(fb / "systemctl").write_text(f"""#!/bin/sh
[ "$1 $2 $3 $4" = "--user is-active -q strata-eval" ] || {{ echo "fake systemctl: $*" >&2; exit 9; }}
[ -e {state} ] && exit 0 || exit 3
""")
(fb / "systemctl").chmod(0o755)
os.environ["PATH"] = f"{fb}:{os.environ['PATH']}"
import charlora as L    # noqa: E402

check("charlora: the language model down -> the iGPU may judge beside a card", not L.igpu_taken())
state.touch()
check("charlora: the language model up -> the card judges itself (no iGPU helper)", L.igpu_taken())
src = Path(L.__file__).read_text()
check("charlora build: --igpu-judge only when the iGPU is free",
      'if card != "igpu" and not igpu_taken():\n                cmd += ["--igpu-judge"]' in src)

# the memory floors: the iGPU's (its model memory is host RAM) stay; on a card only the load's (~9-11.5 GB measured)
L.mem_avail_gb = lambda: 20.0                    # what the box has with the language model up
L.on_card = lambda: True


def refused(gb):
    try:
        L.need_mem(gb)
        return False
    except SystemExit:
        return True


check("on a card: the train's and the text encoder's loads go ahead with the language model up (20 GB free)",
      not refused(30) and not refused(25) and not refused(15))
L.mem_avail_gb = lambda: 8.0
check("on a card: still refused below the load's own need", refused(15))
L.on_card = lambda: False
L.mem_avail_gb = lambda: 20.0
check("on the iGPU: the floors are as before", refused(30) and refused(25) and not refused(15))

# --- shot_flow.run: the order of things ------------------------------------------------------------------------------
import flow3d                       # noqa: E402
import shot_flow as S               # noqa: E402
from tts_reader import animate, video as V    # noqa: E402

book = TMP / "book"
(book / "work" / "video" / "01_one").mkdir(parents=True)
(book / "work" / "video" / "01_one" / "plan.json").write_text(json.dumps({"look": {"characters": {"Ada": "x", "Bo": "y"}}}))
ev, lock = [], threading.Lock()


def note(e):
    with lock:
        ev.append(e)


real_run = subprocess.run
S.subprocess.run = lambda cmd, **k: (note("stop strata") if "stop" in cmd else None,
                                     real_run(["true"]))[1]
S.MF.entries = lambda *a: [{"id": "m1", "t": 0.0, "dur": 5.0}]
S.MF.cover = lambda *a: False
S.out_path = lambda b, s, t0, t1: TMP / "out" / "x.mp4"
S.shot_dir = lambda b, s: TMP / "shots"
S._plan_sig = lambda b, s: note("sig") or "sig"
S._step = lambda *a: None
S.wardrobe = lambda *a: note("wardrobe")
S.room_layouts = lambda *a: note("layouts") or {}
S.sfx_plan = lambda *a: note("sfx")
S.place_sheets = lambda *a, **k: note("sheets") or {}
S.apply_chains = lambda es, by, log: es
S._migrate_ids = lambda *a: None
S.describe_places = lambda *a: None
S.chain_pauses = lambda *a: []
S.sheet = lambda *a, **k: TMP / "keys.jpg"
S.RG.Book = lambda b: type("R", (), {"place_look": lambda self, w: ""})()
V.looks = lambda plan: (list((plan.get("look") or {}).get("characters", {})), {})
animate.igpu = lambda t, log=print: note(f"igpu {t}")


def direct(book, stem, es, log, **k):
    note("director start")
    time.sleep(0.3)
    note("director end")
    return es, {"m1": [{"id": "m1_s1", "kind": "quiet", "who": ["Ada"]}]}, None


def keys(book, stem, shots, log, gate=None):
    note("keys" + (" gated" if gate else ""))
    return []


S.direct, S.keys_judged = direct, keys
todo = {"Ada": "new", "Bo": "new"}
flow3d.lora_todo = lambda b, only=None: {n: w for n, w in todo.items() if not only or n in only}
seen_only = []


def ensure(b, log, only=None, status_cb=None, rank=None, landed=None):
    seen_only.append(set(only or ()))
    note("loras start")
    time.sleep(0.6)                     # longer than the director: the keys must wait for it
    todo.clear()
    note("loras end")
    return []


flow3d.ensure_loras = ensure
S.run(book, "01_one", 0, 10, None, TMP / "mix.mp3", log=lambda m: None, until="keys")
i = ev.index
check("the language model goes up before the builds (it can't start beside them: ~96 GB)",
      "igpu strata" in ev and i("igpu strata") < i("loras start"), ev)
check("the builds start after the room layouts and before the director, and run beside it",
      i("layouts") < i("loras start") < i("director start") and i("director end") < i("loras end"), ev)
check("the builds are the chapter's cast only", seen_only == [{"Ada", "Bo"}], seen_only)
check("the keys start after the director, gated on the builds still training",
      i("director end") < i("keys gated") < i("loras end"), ev)
check("the saved plan's signature is taken again after the LoRAs", ev[i("loras end"):].count("sig") == 1, ev)

ev.clear()
todo.update({"Ada": "new"})
flow3d.ensure_loras = lambda b, log, only=None, status_cb=None, rank=None, landed=None: (note("loras end"),
                                                                                          ["no LoRA: Ada (new)"])[1]
try:
    S.run(book, "01_one", 0, 10, None, TMP / "mix.mp3", log=lambda m: None, until="keys")
    check("a person's missing LoRA stops the run", False, ev)
except RuntimeError as exc:
    check("a person's missing LoRA stops the run (after the keys it does not block)",
          "Ada" in str(exc) and "keys gated" in ev and "sig" not in ev[i("keys gated"):], (exc, ev))

ev.clear()
todo.clear()
S.run(book, "01_one", 0, 10, None, TMP / "mix.mp3", log=lambda m: None, until="keys")
check("nothing to train: no build and no extra start of the language model",
      "loras start" not in ev and "igpu strata" not in ev and "keys" in ev, ev)
print({"fails": fails})
sys.exit(1 if fails else 0)
