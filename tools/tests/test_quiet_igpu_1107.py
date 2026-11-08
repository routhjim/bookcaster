"""11-07 (loop E24, E21's option 2): BOOKCASTER_QUIET_IGPU=1 - the main hold's keys on the cards before LTX (as main),
the quiet hold's keys on the iGPU while the main hold renders. CPU only (the LTX runner, qwen_image --serve and the
key judge faked):
  - quiet_later: only the quiet route, only with the mode on and a main hold to draw beside;
  - _render: ONE drawer on the iGPU (one CU free, no card lock), loaded before LTX loads, drawing while the main hold
    renders; the quiet hold renders from those keys; no card drawer;
  - the drawer dies part-way / fails to load / hangs: the quiet hold is not held back (a hang is killed after
    QUIET_KEY_GRACE), the wait is logged, the cards draw what is missing (the seeds drawn there judged, not redrawn);
  - a resume with the quiet keys and clips on disk: no drawer, nothing drawn, the clips kept;
  - no quiet shots: ltx without quiet_keys (main's own path).
    ~/tts_reader/.venv/bin/python3 tools/tests/test_quiet_igpu_1107.py"""
import json, queue, sys, tempfile, threading, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import shot_flow as S
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


ev, holds, lock, procs = [], [], threading.Lock(), []


def note(*e):
    with lock:
        ev.append(e)


class FakeProc:
    """qwen_image.py --serve: a load line, then one line per job (its picture written). igpu_mode: ok | die:N (EOF
    after N draws) | noload (EOF at once) | hang (never answers until killed)."""
    igpu_mode = "ok"

    def __init__(self, cmd, **k):
        self.card = cmd[1].rsplit("-", 1)[1].split(".")[0] if cmd[0] == "flock" else "igpu"
        self.cmd, self.env, self.out, self.n = cmd, k.get("env") or {}, queue.Queue(), 0
        self.stdin = self.stdout = self
        self.mode = FakeProc.igpu_mode if self.card == "igpu" else "ok"
        procs.append(self)
        note("drawer start", self.card)
        self.out.put("" if self.mode == "noload" else json.dumps({"phase": "load drawing model (kept)"}) + "\n")

    def write(self, s):
        j = json.loads(s)
        if self.mode in ("hang", "noload"):
            return
        if self.mode.startswith("die:") and self.n >= int(self.mode[4:]):
            self.out.put("")
            return
        self.n += 1
        time.sleep(0.02)
        Path(j["path"]).write_text(j["id"])
        note("draw", self.card, j["id"])
        self.out.put(json.dumps({"id": j["id"]}) + "\n")

    def readline(self):
        v = self.out.get()
        if "phase" in v:
            note("drawer loaded", self.card)
        if not v:
            self.out.put("")
        return v

    def flush(self):
        pass

    def close(self):
        pass

    def wait(self, *a):
        return 0

    def kill(self):
        note("drawer killed", self.card)
        self.out.put("")


S.subprocess = types.SimpleNamespace(run=lambda *a_, **k_: types.SimpleNamespace(stdout="2\n", returncode=1, stderr=""),
                                     Popen=FakeProc, PIPE=-1, DEVNULL=-3, TimeoutExpired=Exception)
real_sleep = time.sleep
S.time = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_") and k != "sleep"},
                               sleep=lambda s_: real_sleep(0.01))
S.voice_pitch = lambda w: None
S._offscreen = lambda cx, s, t, only=True: t
S._pose_hold = lambda cx, s: ""
S._held = lambda cx, s: "held"
S._talk_prompt = lambda R, cx, e, here, s=None: "talk " + e["id"]
S.talk_pieces = lambda total, env, mx, ov, mn=0: [(0, 49)]
S._talk_score = lambda *a_: 99.0
S._sync_offsets = lambda *a_, **k_: None
S._stitch_talk = lambda *a_, **k_: None
S.MF.fight_clip_ok = lambda p, n=None: []
S.judge_takes = lambda *a_, **k_: set()
S.sheet = lambda *a_, **k_: "sheet"
S.assemble = lambda *a_, **k_: "cut"
S.video_mix = lambda *a_, **k_: None
S._step = lambda *a_, **k_: None
S.REVIEW = "off"
S.QUIET_IGPU = True
S.QUIET_KEY_GRACE = 2
sys.modules["flow3d"] = types.SimpleNamespace(_pp_split=lambda *a_: 24)
MAIN_WAIT = {"s": 0.0}                  # how long the fake main hold stays open (the quiet keys are drawn meanwhile)


def fake_step(cxl, name, args):
    """The loaded runner: its file's jobs, then --more until --more-done (a clip on disk is skipped)."""
    L = cxl.OUT / "ltx"
    jobs = json.loads((L / args[1]).read_text())
    more, done = L / args[args.index("--more") + 1], L / args[args.index("--more-done") + 1]
    note("ltx", args[1])
    h = {"file": args[1], "made": []}
    holds.append(h)
    seen, t0 = set(), time.time()
    while True:
        for j in jobs + (json.loads(more.read_text()) if more.exists() else []):
            if j["name"] in seen:
                continue
            seen.add(j["name"])
            out = L / "out" / f"{j['name']}.mp4"
            if out.exists():
                continue
            out.write_text(json.dumps(j))
            out.with_suffix(".json").write_text(json.dumps(j))
            h["made"].append(j["name"])
        if done.exists() and (args[1] == "quiet_jobs.json" or time.time() - t0 > MAIN_WAIT["s"]):
            break
        real_sleep(0.01)
    note("ltx end", args[1])
    return 0


R = types.SimpleNamespace(LTX_PY="py", NEG_EXTRA="", audio_env=lambda *a_: None, HOLD_TIMEOUT=0, PP=False,
                          LTX_PP_ARGS="--lanes 3", ltx_step=fake_step,
                          encode_all=lambda cxl, f, name, extra: note("encode"))
S._ctx = lambda book, stem: (R, None)
BAD = set()


def fake_judge(items, ans, first):
    for it in items:
        sid, seed = it["id"].split("|")[0], int(it["id"].split("|")[1].split("_")[1])
        ans[it["id"]] = {"ok": (sid, seed) not in BAD}
        note("judged", sid, seed)


S._judge = fake_judge
S._clean = lambda a, k, need: bool(a.get("ok"))


def keydir(td, s):
    return Path(td) / s["id"].rsplit("_s", 1)[0] / s["id"]


def tasks_for(td, rows):
    """key_tasks' tasks (main's skip rule, simplified): every shot without a key.png."""
    out = []
    for s in rows:
        d = keydir(td, s)
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "key.png").exists():
            out.append({"s": s, "d": d, "q": {}, "need": 1, "seeds": list(S.ALL_SEEDS), "tried": [],
                        "job": {"prompt": s["id"]}})
    return out


def sh(i, kind, dur=3.0, **k):
    return dict(id=f"{kind}-{i:06d}_s0", kind=kind, place="Smithy", action=f"ROW-{i}", picture=f"P-{i}", dur=dur,
                who=["Hulda"], **k)


def setup(quiet=True):
    """A fresh book: the main hold's keys already drawn (as run() does on the cards), the quiet ones not."""
    td = tempfile.mkdtemp()
    S.shot_dir = lambda book, stem: Path(td)
    S.key_tasks = lambda book, stem, shots, log=print: tasks_for(td, shots)
    rows = [sh(0, "quiet", 2.0), sh(1, "action", 2.5), sh(2, "talk", 4.0), sh(3, "quiet", 4.5), sh(4, "quiet", 3.0)]
    if not quiet:
        rows = [r for r in rows if r["kind"] != "quiet"]
    es = [{"id": r["id"].rsplit("_s", 1)[0], "wav": "w.wav", "chunks": [(0, 49)], "speaker": "Hulda"} for r in rows]
    for r in rows:
        keydir(td, r).mkdir(parents=True, exist_ok=True)
        if r["kind"] != "quiet":
            (keydir(td, r) / "key.png").write_text("main key")
    ev.clear(); holds.clear(); procs.clear(); BAD.clear()
    return td, rows, es


def render(td, rows, es):
    logs = []
    fz = Path(td) / "plan.json"
    fz.write_text("{}")
    t0 = time.time()
    S._render(Path(td), "x", es, {}, rows, None, Path(td) / "o.mp4", fz, logs.append)
    if "-v" in sys.argv:
        print("\n".join(logs)); print(ev)
    return logs, time.time() - t0


def made(td, sid):
    return (Path(td) / "ltx" / "out" / f"{sid}.mp4").exists()


def quiet_ids(rows):
    return [r["id"] for r in rows if r["kind"] == "quiet"]


# ---- 0. which shots ---------------------------------------------------------------------------------------------------
td, rows, es = setup()
check("quiet_later: the quiet route only (not talk, action or a chain)",
      [s["id"] for s in S.quiet_later(rows + [sh(9, "quiet", chain={"lines": ["a"]})])] == quiet_ids(rows))
S.QUIET_IGPU = False
check("quiet_later: [] with the mode off", S.quiet_later(rows) == [])
S.QUIET_IGPU = True
check("quiet_later: [] with no main hold to draw beside (every shot quiet)",
      S.quiet_later([r for r in rows if r["kind"] == "quiet"]) == [])

# ---- 1. the quiet keys drawn on the iGPU during the main hold -----------------------------------------------------------
td, rows, es = setup()
q = quiet_ids(rows)
BAD.add((q[1], 7))                      # its first seed fails: seed 11 drawn there too
MAIN_WAIT["s"] = 1.5
FakeProc.igpu_mode = "ok"
logs, _ = render(td, rows, es)
kinds = [e[:2] for e in ev]
draws = [e for e in ev if e[0] == "draw"]
check("one drawer, on the iGPU; no card drawer", [p.card for p in procs] == ["igpu"], [p.card for p in procs])
ig = procs[0]
check("the iGPU drawer: its own torch, HSA_CU_MASK keeps one CU free, no card lock",
      ig.cmd[0] == str(S.IGPU_PY) and ig.env.get("HSA_CU_MASK") == "2:0-38" and ig.env.get("HIP_VISIBLE_DEVICES") == "2"
      and "flock" not in ig.cmd, (ig.cmd, ig.env.get("HSA_CU_MASK")))
check("the drawer is loaded before the main hold loads LTX",
      kinds.index(("drawer loaded", "igpu")) < kinds.index(("ltx", "two_talk_jobs.json")), ev)
i0, i1 = ev.index(("ltx", "two_talk_jobs.json")), ev.index(("ltx end", "two_talk_jobs.json"))
check("every quiet key drawn before the main hold ends (beside the encode and LTX; seed 11 after a failed 7)",
      sorted(e[2] for e in draws) == sorted([f"{q[0]}_key_7", f"{q[1]}_key_7", f"{q[1]}_key_11", f"{q[2]}_key_7"])
      and all(ev.index(e) < i1 for e in draws), (draws, i0, i1))
check("only quiet keys drawn (the main hold's were drawn before, on the cards)",
      all(e[2].startswith("quiet") for e in draws), draws)
check("two holds as on main: the main hold then the quiet hold, every clip made",
      [h["file"] for h in holds] == ["two_talk_jobs.json", "quiet_jobs.json"] and all(made(td, r["id"]) for r in rows)
      and set(holds[1]["made"]) == set(q), holds)
j1 = json.loads((Path(td) / "ltx" / "out" / f"{q[1]}.mp4").read_text())
check("a quiet clip starts from its iGPU key (the judge's pick, seed 11)",
      j1["images"][0]["path"].endswith(f"{q[1]}/key.png")
      and (keydir(td, rows[3]) / "key.png").read_text().endswith("_key_11"), j1["images"])
check("the wait at the quiet hold is logged", any("the quiet hold waited" in m for m in logs), logs)
check("quiet_jobs.json is the quiet hold's list", len(json.loads((Path(td) / "ltx" / "quiet_jobs.json").read_text())) == 3)

# ---- 2. a resume: quiet keys and clips on disk -> no drawer, nothing drawn, clips kept --------------------------------
mt = {s: (Path(td) / "ltx" / "out" / f"{s}.mp4").stat().st_mtime for s in q}
ev.clear(); holds.clear(); procs.clear()
logs, _ = render(td, rows, es)
check("a resume: no drawer, no draw, no hold, the quiet clips kept",
      procs == [] and not [e for e in ev if e[0] == "draw"] and holds == []
      and all((Path(td) / "ltx" / "out" / f"{s}.mp4").stat().st_mtime == mt[s] for s in q), (procs, holds))

# ---- 3. the drawer dies after one draw: the cards draw the rest, before the quiet hold ---------------------------------
td, rows, es = setup()
q = quiet_ids(rows)
MAIN_WAIT["s"] = 0.5
FakeProc.igpu_mode = "die:1"
logs, _ = render(td, rows, es)
draws = [e for e in ev if e[0] == "draw"]
check("the drawer died: said; the cards (both, under their locks) draw the 2 keys it left, the 1 it drew kept",
      any("igpu key drawer died" in m for m in logs) and {p.card for p in procs} - {"igpu"} <= {"05", "65"}
      and len(procs) > 1
      and [e[1] for e in draws].count("igpu") == 1 and len([e for e in draws if e[1] != "igpu"]) == 2
      and all(p.cmd[:2] == ["flock", f"/tmp/gpu-{p.card}.lock"] for p in procs if p.card != "igpu"), (draws, logs))
check("the card draws come after the main hold, before the quiet hold; every quiet clip made",
      all(ev.index(("ltx end", "two_talk_jobs.json")) < ev.index(e) < ev.index(("ltx", "quiet_jobs.json"))
          for e in draws if e[1] != "igpu") and all(made(td, s) for s in q), ev)

# ---- 4. the drawer fails to load: LTX does not wait out QUIET_LOAD_WAIT; the cards draw all ------------------------------
td, rows, es = setup()
q = quiet_ids(rows)
MAIN_WAIT["s"] = 0.0
FakeProc.igpu_mode = "noload"
logs, secs = render(td, rows, es)
draws = [e for e in ev if e[0] == "draw"]
check("the drawer did not load: no wait for it, the 3 quiet keys drawn on the cards, the quiet hold made",
      secs < 30 and not [e for e in draws if e[1] == "igpu"] and len(draws) == 3 and all(made(td, s) for s in q),
      (secs, draws))

# ---- 5. the drawer hangs: killed after QUIET_KEY_GRACE, the wait logged, the cards draw ----------------------------------
td, rows, es = setup()
q = quiet_ids(rows)
FakeProc.igpu_mode = "hang"
S.QUIET_LOAD_WAIT = 1
logs, secs = render(td, rows, es)
draws = [e for e in ev if e[0] == "draw"]
w = [m for m in logs if "the quiet hold waited" in m]
check("a hung drawer: killed after the grace, the wait logged, the cards draw the 3 keys, the quiet hold made",
      ("drawer killed", "igpu") in ev and w and len([e for e in draws if e[1] != "igpu"]) == 3
      and all(made(td, s) for s in q), (w, draws))
S.QUIET_LOAD_WAIT = 180

# ---- 6. a seed drawn on the iGPU but not judged when it stopped: judged on the cards' turn, not drawn again ------------
td, rows, es = setup()
q = quiet_ids(rows)
(keydir(td, rows[0]) / "key_7.png").write_text("drawn there")
FakeProc.igpu_mode = "noload"
logs, _ = render(td, rows, es)
check("a seed on disk is judged, not drawn again",
      not [e for e in ev if e[0] == "draw" and e[2] == f"{q[0]}_key_7"] and ("judged", q[0], 7) in ev
      and (keydir(td, rows[0]) / "key.png").read_text() == "drawn there", ev)

# ---- 7. no quiet shots: main's own path ----------------------------------------------------------------------------------
td, rows, es = setup(quiet=False)
FakeProc.igpu_mode = "ok"
seen = {}
real_ltx = S.ltx
S.ltx = lambda *a_, **k_: (seen.update(k_), real_ltx(*a_, **k_))
logs, _ = render(td, rows, es)
S.ltx = real_ltx
check("no quiet shots: no quiet_keys, no drawer, main's one hold",
      seen.get("quiet_keys") is None and procs == [] and [h["file"] for h in holds] == ["two_talk_jobs.json"], (seen, holds))

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
