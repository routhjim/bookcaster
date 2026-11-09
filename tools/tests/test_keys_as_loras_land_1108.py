# suite: slow
"""11-08 user: "the LoRA's in the most shots are made first and then have the iGPU get to work on keyframes as soon as
Strata/Director is done ... whenever the first XTX card is done with LoRA, kick that one off on key frames too".
CPU only: the LoRA builds (flow3d.run), qwen_image --serve, the key judge, the place pictures and key_tasks faked;
shot_flow.loras_beside / _Loras / keys_judged(gate=) and flow3d.ensure_loras are the real code.
  - the builds go the most moments first, one queue over both cards;
  - the iGPU draws from the start, each card once it has no LoRA left; nothing is drawn before its people's LoRAs;
  - an unsettled place picture is drawn on the first free card, before that card's drawer and before any key there;
  - every shot settled, each seed drawn once; a person's failed LoRA: its shots never drawn, the rest drawn, wait()
    raises "no LoRA for"; the mode off (BOOKCASTER_QUIET_IGPU=0): no iGPU drawer; a card held by someone else: polled.
    ~/tts_reader/.venv/bin/python3 tools/tests/test_keys_as_loras_land_1108.py"""
import json, queue, sys, tempfile, threading, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import flow3d as F3                 # noqa: E402
import shot_flow as S               # noqa: E402
from tts_reader import animate, video as V    # noqa: E402
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


ev, lock, T0 = [], threading.Lock(), [0.0]


def note(*e):
    with lock:
        ev.append((round(time.time() - T0[0], 3),) + e)


# ---- the LoRA builds: flow3d.run faked (each person takes DUR s on its card) ------------------------------------------
DUR, FAIL, built, BUSY = {}, set(), set(), set()


def fake_build(cmd, logf=None, *a, **k):
    n, c = cmd[cmd.index("build") + 2], cmd[cmd.index("--card") + 1]
    note("build start", n, c)
    time.sleep(DUR[n])
    if n not in FAIL:
        with lock:
            built.add(n)
    note("build end", n, c)
    return 0 if n not in FAIL else 1


F3.run = fake_build
F3.lora_todo = lambda b, only=None: {n: "no LoRA yet" for n in DUR if n not in built and (only is None or n in only)}
F3._is_person = lambda b, n: True
F3._log = lambda msg, log=print: log(msg)
animate.igpu = lambda *a, **k: note("igpu strata")
animate._flush = lambda: None
V.looks = lambda plan: (list(DUR), {})
S.RG.Book = lambda b: types.SimpleNamespace(same=lambda a, b_: a == b_)


# ---- the drawers: qwen_image --serve faked (a load line, then one line per job, its picture written) -----------------
class FakeProc:
    def __init__(self, cmd, **k):
        self.card = cmd[1].rsplit("-", 1)[1].split(".")[0] if cmd[0] == "flock" else "igpu"
        self.out = queue.Queue()
        self.stdin = self.stdout = self
        note("drawer start", self.card)
        self.out.put(json.dumps({"phase": "load"}) + "\n")

    def write(self, s):
        j = json.loads(s)
        time.sleep(0.03 if self.card == "igpu" else 0.01)      # (the iGPU is slower)
        Path(j["path"]).write_text(j["id"])
        sid = j["id"].rsplit("_key_", 1)[0]
        with lock:
            have = set(built)
        note("draw", self.card, sid, j["id"], tuple(sorted(have)))
        self.out.put(json.dumps({"id": j["id"]}) + "\n")

    def readline(self):
        v = self.out.get()
        if '"phase"' in v:              # the model's load: it takes a while, then its line
            time.sleep(0.05)
            note("loaded", self.card)
        if not v:
            self.out.put("")
        return v

    def flush(self):
        pass

    def close(self):
        self.out.put("")

    def wait(self, *a):
        return 0

    def kill(self):
        self.out.put("")


def fake_run(cmd, *a, **k):
    if cmd[:2] == ["flock", "-n"]:
        return types.SimpleNamespace(returncode=1 if cmd[2] in BUSY else 0, stdout="", stderr="")
    if cmd[0] == "systemctl":
        return types.SimpleNamespace(returncode=3, stdout="", stderr="")    # the language model is down
    return types.SimpleNamespace(returncode=0, stdout="1\n", stderr="")


S.subprocess = types.SimpleNamespace(run=fake_run, Popen=FakeProc, PIPE=-1, DEVNULL=-3, TimeoutExpired=Exception)
real_sleep = time.sleep
S.time = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_") and k != "sleep"},
                               sleep=lambda s_: real_sleep(0.01))
S._judge = lambda items, ans, first: [ans.update({it["id"]: {"ok": True}}) or note("judged", it["id"]) for it in items]
S._clean = lambda a, k, need: bool(a.get("ok"))

# ---- the place pictures: the Smithy settled, the Hall still to draw ----------------------------------------------------
HERO = {"Hall": False}
S.MF._hero_state = lambda book, w, log=print, must=(): Path(f"/hero/{w}.png") if w != "Hall" or HERO["Hall"] else {}


def heroes_for(book, places, log=print, cards=()):
    note("heroes", tuple(cards), tuple(sorted(w for w in places if w == "Hall" and not HERO["Hall"])))
    time.sleep(0.05)                    # (its own drawers load and draw)
    note("heroes end")
    HERO["Hall"] = True
    return {w: Path(f"/hero/{w}.png") for w in places}


S.MF.heroes_for = heroes_for
S.MF.HERO_CARDS = ("05", "65")
TD = [None]
fed = []


def key_tasks(book, stem, shots, log=print, heroes=None):
    out = []
    for s in shots:
        assert heroes is not None and (not s.get("place") or s["place"] in heroes), s
        d = Path(TD[0]) / s["id"]
        d.mkdir(parents=True, exist_ok=True)
        fed.append(s["id"])
        if not (d / "key.png").exists():
            out.append({"s": s, "d": d, "q": {}, "need": 1, "seeds": list(S.ALL_SEEDS), "tried": [],
                        "job": {"prompt": s["id"]}})
    return out


S.key_tasks = key_tasks


def sh(i, who, place="Smithy"):
    return {"id": f"quiet-{i:06d}_s0", "kind": "quiet", "who": who, "place": place}


SHOTS = [sh(0, []), sh(1, ["Ada"]), sh(2, ["Cy"]), sh(3, ["Ada", "Cy"]), sh(4, ["Bo"]), sh(5, ["Bo", "Cy"], "Hall"),
         sh(6, ["Ada"]), sh(7, []), sh(8, ["Cy"], "Hall")]
# the stretch's moments: Ada in 5, Cy in 3, Bo in 1 - the build order Ada, Cy, Bo
ES = [{"id": f"m{i}", "t": float(i), "dur": 1.0, "who": w} for i, w in enumerate([["Ada"], ["Ada", "Cy"], ["Ada"], ["Ada"], ["Ada", "Cy"],
                                                       ["Cy"], ["Bo"], []])]


def go(dur, fail=(), igpu=True, busy=(), director=0.1):
    ev.clear(); built.clear(); FAIL.clear(); FAIL.update(fail); BUSY.clear(); BUSY.update(busy); fed.clear()
    DUR.clear(); DUR.update(dur)
    HERO["Hall"] = False
    S.QUIET_IGPU = igpu
    TD[0] = tempfile.mkdtemp(prefix="keys-loras-")
    book = Path(TD[0]) / "book"
    (book / "work" / "video" / "01").mkdir(parents=True)
    (book / "work" / "video" / "01" / "plan.json").write_text("{}")
    logs = []
    T0[0] = time.time()
    gate = S.loras_beside(book, "01", logs.append, ES)
    real_sleep(director)                # the director, the set sheets, ... (the language model then stopped)
    note("keys start")
    dirty = S.keys_judged(book, "01", SHOTS, logs.append, gate=gate)
    note("keys end")
    err = None
    try:
        gate()
    except RuntimeError as e:
        err = str(e)
    if "-v" in sys.argv:
        print("\n".join(logs)); print("\n".join(map(str, ev)))
    return logs, dirty, err


def t(*e):
    return next((x[0] for x in ev if x[1:1 + len(e)] == e), None)


def draws():
    """(t, card, shot id, job id, the LoRAs built when drawn)"""
    return [(x[0],) + x[2:] for x in ev if x[1] == "draw"]


WHO = {s["id"]: s["who"] for s in SHOTS}


def loads():
    """[(start, end)] of every model load: each drawer's (start .. its load line) and the place pictures' drawing."""
    out, open_ = [], {}
    for x in ev:
        if x[1] in ("drawer start", "heroes"):
            open_[x[2] if x[1] == "drawer start" else "h"] = x[0]
        elif x[1] in ("loaded", "heroes end"):
            out.append((open_.pop(x[2] if x[1] == "loaded" else "h"), x[0]))
    return sorted(out)


def no_overlap():
    ls = loads()
    return len(ls) >= 2 and all(a[1] <= b[0] for a, b in zip(ls, ls[1:]))

# ---- 1. three people on two cards ----------------------------------------------------------------------------------
logs, dirty, err = go({"Ada": 0.8, "Cy": 0.3, "Bo": 0.3})
starts = [(x[2], x[3]) for x in ev if x[1] == "build start"]
check("the most moments first: Ada on 05, Cy on 65, then Bo on the first card free (65)",
      starts == [("Ada", "05"), ("Cy", "65"), ("Bo", "65")] or starts == [("Cy", "65"), ("Ada", "05"), ("Bo", "65")],
      starts)
check("the order is said", any("Ada (5), Cy (3), Bo (1)" in ln for ln in logs), logs[:3])
check("the iGPU drawer starts with the keys, before any LoRA lands",
      t("drawer start", "igpu") is not None and t("keys start") <= t("drawer start", "igpu") < t("build end", "Cy"),
      ev[:8])
check("nothing is drawn before every one of its people's LoRAs is built",
      all(set(WHO[x[2]]) <= set(x[4]) for x in draws()), [x for x in draws() if not set(WHO[x[2]]) <= set(x[4])])
first = draws()[0]
check("the first key drawn is one with nobody in it, on the iGPU", not WHO[first[2]] and first[1] == "igpu", first)
check("card 65 starts drawing once its last LoRA (Bo) is done, before Ada's lands on 05",
      t("build end", "Bo") <= t("drawer start", "65") < t("build end", "Ada"), ev)
check("card 05 starts drawing once Ada's is done", t("build end", "Ada") <= (t("drawer start", "05") or -1), ev)
hs = [x for x in ev if x[1] == "heroes"]
check("the Hall's picture is drawn once, on the first free card (65), before that card's drawer",
      len(hs) == 1 and hs[0][2] == ("65",) and hs[0][3] == ("Hall",) and hs[0][0] <= t("drawer start", "65"), hs)
check("no Hall key before its picture", all(x[0] >= hs[0][0] for x in draws() if x[2] in ("quiet-000005_s0",
                                                                                         "quiet-000008_s0")))
check("every shot settled (a key.png), none left, none lost",
      all((Path(TD[0]) / s["id"] / "key.png").exists() for s in SHOTS) and not dirty and err is None, (dirty, err))
ids = [x[3] for x in draws()]
check("each seed drawn once, each shot fed once", len(ids) == len(set(ids)) == len(SHOTS) and
      sorted(fed) == sorted(s["id"] for s in SHOTS), (ids, fed))
check("the iGPU and both cards all drew", {x[1] for x in draws()} >= {"igpu", "65"}, {x[1] for x in draws()})
check("the drawers load one at a time, never beside each other or the place pictures", no_overlap(), loads())

# ---- 2. a person's LoRA fails ----------------------------------------------------------------------------------------
logs, dirty, err = go({"Ada": 0.4, "Cy": 0.2, "Bo": 0.2}, fail={"Cy"})
cy = {s["id"] for s in SHOTS if "Cy" in s["who"]}
check("Cy's build fails (twice): no key with Cy in it is drawn", not any(x[2] in cy for x in draws()), draws())
check("the other shots' keys are all drawn",
      all((Path(TD[0]) / s["id"] / "key.png").exists() for s in SHOTS if s["id"] not in cy))
check("the job then stops as before: no LoRA for Cy", err and "no LoRA for" in err and "Cy" in err, err)

# ---- 3. the mode off: no iGPU drawer, the keys wait for a free card ----------------------------------------------------
logs, dirty, err = go({"Ada": 0.4, "Cy": 0.2, "Bo": 0.1}, igpu=False)
check("BOOKCASTER_QUIET_IGPU=0: no iGPU drawer, the first key after the first card is free",
      t("drawer start", "igpu") is None and draws()[0][0] >= t("build end", "Bo"), ev)
check("... every key still drawn", all((Path(TD[0]) / s["id"] / "key.png").exists() for s in SHOTS))
check("... and the cards load one at a time", no_overlap(), loads())

# ---- 4. a card someone else holds: polled (flock -n), never queued on --------------------------------------------------
logs, dirty, err = go({"Ada": 0.3, "Cy": 0.1, "Bo": 0.1}, busy={"/tmp/gpu-65.lock"})
check("a card whose lock is held: no drawer started on it, the others draw everything",
      t("drawer start", "65") is None and all((Path(TD[0]) / s["id"] / "key.png").exists() for s in SHOTS), ev)

# ---- 5. one person on two cards: the idle card draws from the start ----------------------------------------------------
logs, dirty, err = go({"Ada": 0.4})
check("one LoRA to train: card 65 has none and draws (after the place picture) from the keys' start",
      t("drawer start", "65") is not None and t("drawer start", "65") < t("build end", "Ada"), ev)
print({"fails": fails})
sys.exit(1 if fails else 0)
