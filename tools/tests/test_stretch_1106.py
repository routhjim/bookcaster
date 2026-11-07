"""11-06 (CHANGES 79, loop E17-P1): stretch chaining. CPU only (the pair ask stubbed, the LTX runner faked):
  - the pair ask: code cuts on a place change and at an action chain; 'continue' marks the next shot, an unclear
    answer cuts; cached; no concrete example in the ask;
  - stretch_links: only the shot right before; BOOKCASTER_STRETCH off -> none;
  - ltx(): a continued shot has no key and starts from its source's last job (chain_from, last frame, overlap 1, no
    seam re-renders), its own prompt and length; a quiet-only stretch in the quiet hold, a stretch with talk/action
    whole in the action/talk hold (its quiet shot on the two-stage route); talk's first piece chained, a talk
    source's last piece chained from; stretch jobs first in the queue, never retaken; off -> today's jobs.
    ~/tts_reader/.venv/bin/python3 tools/tests/test_stretch_1106.py"""
import json, sys, tempfile, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import shot_flow as S
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


# ---- the pair ask -----------------------------------------------------------------------------------------------
calls = []


def ask(msgs):
    t = msgs[0]["content"]
    calls.append(t)
    return "cut" if "ROW-C" in t.split("Shot 2:")[1] else "Hmm." if "ROW-E" in t.split("Shot 2:")[1] else \
        "It is the same view, so: continue."


def sh(i, place="Hangar", kind="quiet", **k):
    return dict(id=f"{kind}-{i:06d}_s0", kind=kind, place=place, action=f"ROW-{i}", dur=3.0, who=["Daven"], **k)


rows = [sh(0), sh(1), sh(2), sh(3, place="Cockpit"), sh(4, place="Cockpit", chain={"lines": ["x"]}),
        sh(5, place="Cockpit"), sh(6, place="Cockpit")]
rows[2]["action"] = "ROW-C"
rows[6]["action"] = "ROW-E"
with tempfile.TemporaryDirectory() as td:
    c = Path(td) / "stretch.json"
    logs = []
    n = S.mark_stretches(rows, lambda a, b: f"words {a['id']} {b['id']}", ask, c, logs.append)
    got = [r.get("cont_of") for r in rows]
    check("continue marks the next shot; cut, unclear, a place change and a chain cut",
          got == [None, rows[0]["id"], None, None, None, None, None] and n == 1, got)
    check("asked only where code does not cut (0-1, 1-2, 5-6)", len(calls) == 3, len(calls))
    check("the ask has both rows and the narration over them",
          '"action": "ROW-0"' in calls[0] and '"action": "ROW-1"' in calls[0] and
          f"words {rows[0]['id']} {rows[1]['id']}" in calls[0] and "Reply continue or cut." in calls[0], calls[0])
    check("no concrete example in the ask (placeholders and general words only)",
          all(x in S.PAIR_ASK for x in ("{a}", "{b}", "{words}")) and S.PAIR_ASK.count('"') == 0)
    S.mark_stretches(rows, lambda a, b: f"words {a['id']} {b['id']}", ask, c, logs.append)
    check("cached: no second ask, same marks", len(calls) == 3 and [r.get("cont_of") for r in rows] == got)
    check("reply parse: the last word counts, the question's words ignored",
          S._says_continue("Not a cut. continue") and not S._says_continue("continue or cut? cut")
          and not S._says_continue("") and not S._says_continue(None))

# ---- stretch_links ----------------------------------------------------------------------------------------------
a, b, x = {"id": "a"}, {"id": "b", "cont_of": "a"}, {"id": "x", "cont_of": "zz"}
check("links: only the shot right before", S.stretch_links([a, b, x]) == {"b": a} and S.stretch_links([b]) == {})
S.STRETCH = False
check("links: off -> none", S.stretch_links([a, b]) == {})
S.STRETCH = True

# ---- ltx() job building, the runner faked -----------------------------------------------------------------------
real_sleep = time.sleep
S.time = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_") and k != "sleep"},
                               sleep=lambda s_: real_sleep(0.02))
S.subprocess = types.SimpleNamespace(run=lambda *a_, **k_: types.SimpleNamespace(stdout="NEG", returncode=0, stderr=""))
S.voice_pitch = lambda w: None
S.talk_pieces = lambda n_all, env, mx, ov: [(0, 97), (96, 97)]
S._talk_prompt = lambda R, cx, e, here, s=None: "talk words"
S._offscreen = lambda cx, s, t, only=True: t
S._pose_hold = lambda cx, s: ""
S._held = lambda cx, s: "held"
S.MF.fight_clip_ok = lambda p, n=None: ["melted"]          # every clip faulted: nothing in a stretch is retaken
S._sync_offsets = lambda *a_, **k_: None
sys.modules["flow3d"] = types.SimpleNamespace(_pp_split=lambda *a_: 24)
holds = []


def fake_step(cxl, name, args):
    jobs = json.loads((cxl.OUT / "ltx" / args[1]).read_text())
    holds.append((args[1], jobs))
    for j in jobs:
        (cxl.OUT / "ltx" / "out" / f"{j['name']}.mp4").write_text("v")
    more = cxl.OUT / "ltx" / ("quiet_more.json" if args[1] == "quiet_jobs.json" else "two_more.json")
    for _ in range(60):                 # the judge threads: a retake queued while the hold is loaded
        if more.exists() and json.loads(more.read_text()):
            holds.append(("more", json.loads(more.read_text())))
            break
        real_sleep(0.05)
    return 0


R = types.SimpleNamespace(LTX_PY="py", NEG_EXTRA="", audio_env=lambda *a_: None, HOLD_TIMEOUT=0, PP=False,
                          LTX_PP_ARGS="--lanes 3", ltx_step=fake_step, encode_all=lambda *a_: None)
S._ctx = lambda book, stem: (R, None)


def stitch(book, R_, L, talk, log=print, only=None, quiet=False, jobs_file=""):
    for j in talk:
        (L / "out" / f"{j['name'].rsplit('_c', 1)[0]}.mp4").write_text("v")


S._stitch_talk = stitch


def render(td, shots, es):
    S.shot_dir = lambda book, stem: Path(td)
    holds.clear()
    S.ltx(Path(td), "x", es, shots, log=lambda m: None)
    return {h: {j["name"]: j for j in js} for h, js in holds}


def plan():
    q = [sh(i, place="Bunk") for i in range(3)]
    act = sh(10, place="Hangar", kind="action")
    tk = dict(sh(11, place="Hangar", kind="talk"), action="")
    q4 = sh(12, place="Hangar")
    for s_, src in ((q[1], q[0]), (q[2], q[1]), (tk, act), (q4, tk)):
        s_["cont_of"] = src["id"]
    es = [{"id": tk["id"].rsplit("_s", 1)[0], "wav": "w.wav", "speaker": "Daven", "chunks": [(0, 180)]}]
    return q + [act, tk, q4], es


with tempfile.TemporaryDirectory() as td:
    shots, es = plan()
    for s_ in (shots[0], shots[3], shots[4], shots[5]):     # keys only at the cuts (here also on the continued shots
        d = Path(td) / s_["id"].rsplit("_s", 1)[0] / s_["id"]  # of the off run, where they are today's keys)
        d.mkdir(parents=True)
        (d / "key.png").write_text("k")
    for s_ in (shots[4], shots[5]):                          # on: a continued shot has no key
        (Path(td) / s_["id"].rsplit("_s", 1)[0] / s_["id"] / "key.png").unlink()
    h = render(td, shots, es)
    q0, q1, q2, act, tk, q4 = [s_["id"] for s_ in shots]
    qj, mj = h.get("quiet_jobs.json", {}), h.get("hold_jobs.json") or h.get("two_talk_jobs.json") or {}
    check("quiet-only stretch: all in the quiet hold, in a chain", set(qj) == {q0, q1, q2} and
          qj[q1]["chain_from"] == q0 and qj[q2]["chain_from"] == q1 and "chain_from" not in qj[q0], qj.keys())
    check("continued: last frame, overlap 1, no seam re-renders, no key, its own prompt and length",
          qj[q1]["chain_frame"] == qj[q0]["frames"] - 1 and qj[q1]["overlap"] == 1 and qj[q1]["seam_tries"] == 0
          and qj[q1]["images"] == [] and qj[q1]["prompt"] == "ROW-1" and qj[q1]["frames"] == S._frames(3.0), qj[q1])
    check("the opener keeps its key", qj[q0]["images"][0]["path"].endswith("key.png"))
    check("stretch jobs first in the queue, never retaken", all(j.get("priority") == 1 and j.get("stretch")
                                                                 for j in qj.values()) and "more" not in h)
    check("a stretch with talk/action: whole in the one action/talk hold",
          {act, f"{tk}_c0", f"{tk}_c1", q4} <= set(mj) and q4 not in qj, (list(mj), list(qj)))
    check("talk: first piece from the action's last frame, the second piece from the first as today",
          mj[f"{tk}_c0"]["chain_from"] == act and mj[f"{tk}_c0"]["images"] == [] and
          mj[f"{tk}_c0"]["chain_frame"] == mj[act]["frames"] - 1 and mj[f"{tk}_c1"]["chain_from"] == f"{tk}_c0"
          and mj[f"{tk}_c1"]["overlap"] == 1 and "audio" in mj[f"{tk}_c0"], mj.get(f"{tk}_c0"))
    check("quiet after talk: from the talk's last piece, on the two-stage route with a negative",
          mj[q4]["chain_from"] == f"{tk}_c1" and mj[q4]["chain_frame"] == 96 and mj[q4]["route"] == "ti2v"
          and mj[q4]["neg_extra"] == "" and mj[q4]["prompt"] == "ROW-12", mj[q4])
    enc = {x["name"] for x in json.loads((Path(td) / "ltx" / "enc_dev.json").read_text())}
    check("the two-stage quiet's words encoded with a negative", {mj[q4]["emb"], mj[q4]["emb"] + "__neg"} <= enc)
    n_judged = sorted(p.stem for p in (Path(td) / "ltx" / "out").glob("*.judged"))
    check("faults recorded, not retaken", q1 in n_judged and q4 in n_judged and act in n_judged, n_judged)
    jt = S.judge_takes(Path(td), "x", shots, log=lambda m: None)
    check("judge_takes: no retake for a stretch's shot", jt == set(), jt)

with tempfile.TemporaryDirectory() as td:
    S.STRETCH = False
    shots, es = plan()
    for s_ in shots:
        d = Path(td) / s_["id"].rsplit("_s", 1)[0] / s_["id"]
        d.mkdir(parents=True)
        (d / "key.png").write_text("k")
    h = render(td, shots, es)
    allj = {n_: j for v in h.values() for n_, j in v.items()}
    check("off: no chained shot, every shot from its key, quiet on the distilled route",
          not any(j.get("stretch") or j.get("priority") for j in allj.values())
          and all(j.get("chain_from") in (None, f"{shots[4]['id']}_c0") for j in allj.values())
          and allj[shots[5]["id"]]["route"] == "i2v" and set(h["quiet_jobs.json"]) == {shots[i]["id"] for i in (0, 1, 2, 5)},
          list(allj))
    check("off: faulted clips retaken as today", "more" in h, list(h))
    S.STRETCH = True

src = (HERE / "shot_flow.py").read_text()
check("wired: the pair asks before the language model stops, keys skip continued shots",
      src.index("mark_stretches(win") < src.index('"stop", animate.STRATA_UNIT], capture_output=True)   # (11-06')
      and "keys_judged(book, stem, [x for x in shots if x[\"id\"] not in cont]" in src)
print("ALL PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
