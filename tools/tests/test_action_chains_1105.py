"""11-05 user: "lock in the best known setup" - action chains in the shots flow. CPU only (ffmpeg; no LLM, no GPU, the
vision judge faked):
  - the chain ask's format faults (C4) and the replaced shots made one chain shot (G5 route a: the first replaced
    row's own staging, `who` from it, the key from its picture);
  - the head key's text: each thing's look after its name, a hand gloss for the same word skipped;
  - the chain thread: takes judged as a set (shown in a shuffled order, mapped back), a review card per line and its
    choice.json (pick / override -> new words wait for an encode), the next line chained from the pick, the joined clip;
  - the narration gap: the pause at a sentence break inside the chain's slot, never inside a spoken line, the times
    after it moved, the silence sample-exact.
    ~/tts_reader/.venv/bin/python3 tools/tests/test_action_chains_1105.py"""
import json, subprocess, sys, tempfile, threading, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import shot_flow as S
import characters as CH
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


cast = ["Edric", "Vorgrath"]
looks = {"Edric": "the man in armour", "Vorgrath": "the dragon"}


def named_in(t, c):
    return [w for w in c if w in str(t or "")]


shots = [
    {"kind": "quiet", "start": 0.0, "who": ["Edric"], "people": [{"name": "Edric", "side": "left"}], "picture": "q."},
    {"kind": "action", "start": 3.0, "who": ["Edric", "Vorgrath"], "picture": "Edric stands on the left; Vorgrath "
     "crouches on the right.", "people": [{"name": "Edric", "side": "left", "pose": "standing", "holds": "a sword"},
                                          {"name": "Vorgrath", "side": "right", "pose": "crouching"}],
     "setting": "A cave.", "keyshot": "old moment", "pose_end": {"Edric": "standing"}},
    {"kind": "action", "start": 6.0, "who": ["Edric", "Vorgrath"], "picture": "x.", "people": [],
     "pose_end": {"Edric": "lying"}},
    {"kind": "talk", "start": 9.0, "speaker": "Edric"},
]
good = json.dumps({"chains": [{"lines": ["the man in armour runs toward the dragon.",
                                         "the man in armour drives the sword into the dragon's side."],
                               "head": {}, "things": {"sword": "a long straight steel blade"}, "shots": [2, 3]}]})
f, per, ch = S._chain_faults(good, shots, cast, named_in, looks)
check("faults: a good chain has none", f == [] and per == [[]], f)
bad = json.dumps({"chains": [{"lines": ["Edric runs.", "he swings", "x", "y", "z"], "shots": [3, 4]}]})
f, per, ch = S._chain_faults(bad, shots, cast, named_in, looks)
check("faults: names, pronouns, >4 lines, a talk shot", any("names Edric" in x for x in f) and
      any("pronoun he" in x for x in f) and any("5 lines" in x for x in f) and any("talk shots" in x for x in f), f)
f, per, ch = S._chain_faults("no json here", shots, cast, named_in, looks)
check("faults: no JSON", ch is None and f, f)
calls = []


def ask(msgs):
    calls.append(msgs)
    return bad if len(calls) == 1 else good


with tempfile.TemporaryDirectory() as td:
    cache = Path(td) / "p.chains.json"
    q0 = "THE PASSAGE\nAbout 3 to 5 shots. Places: a cave.\n{fix}"
    kept = S.ask_chains(q0, shots, cast, named_in, looks, ask, cache, log=lambda m: None)
    check("ask: one re-ask on faults, the mended chain kept", len(calls) == 2 and len(kept) == 1, (len(calls), kept))
    check("ask: the message has no shot-count line and no {fix}", "About 3 to 5" not in calls[0][0]["content"] and
          "{fix}" not in calls[0][0]["content"])
    check("ask: the direction rule is in the ask", "Never forward, back, left or right" in calls[0][0]["content"])
    S.ask_chains(q0, shots, cast, named_in, looks, ask, cache, log=lambda m: None)
    check("ask: cached by its message", len(calls) == 2, len(calls))

heads = S.chain_heads(shots, kept, cast, [], {}, None)
check("chain head: one per chain, over its replaced shots", len(heads) == 1 and heads[0][0] == [2, 3], heads)
h = heads[0][1]
check("chain head: the first replaced row (who, picture), keyshot from it, the old moment gone",
      h["who"] == ["Edric", "Vorgrath"] and h["keyshot"] == shots[1]["picture"] and "moment" not in h
      and h["chain"]["lines"][0].startswith("the man in armour runs"), h)
check("chain head: the pose it leaves = the last replaced shot's", h["pose_end"] == {"Edric": "lying"})
# the collapse after the set sheets: only the moments cut from the tagged shots (a split quiet piece too)
tag = {"key": "1:0", "shot": h}
es = [{"id": "m0", "t": 0.0, "dur": 3.0}, {"id": "m1", "t": 3.0, "dur": 2.0}, {"id": "m2", "t": 5.0, "dur": 2.0},
      {"id": "m3", "t": 7.0, "dur": 2.0}, {"id": "m4", "t": 9.0, "dur": 2.0}]
by = {"m0": [{"id": "m0_s0", "kind": "quiet", "place": "cave"}],
      "m1": [{"id": "m1_s0", "kind": "action", "place": "cave", "_chain": tag}],
      "m2": [{"id": "m2_s0", "kind": "action", "place": "cave", "_chain": tag}],
      "m3": [{"id": "m3_s0", "kind": "action", "place": "cave", "_chain": tag}],
      "m4": [{"id": "m4_s0", "kind": "talk", "place": "cave"}]}
keep = {k: json.dumps(by[k]) for k in ("m0", "m4")}
es2 = S.apply_chains(es, by, log=lambda m: None)
check("collapse: the tagged moments are one chain moment (the first id, their whole time), the others untouched",
      [e["id"] for e in es2] == ["m0", "m1", "m4"] and es2[1]["dur"] == 6.0 and by["m1"][0]["chain"]
      and by["m1"][0]["dur"] == 6.0 and by["m1"][0]["id"] == "m1_s0" and "m2" not in by and "m3" not in by
      and all(json.dumps(by[k]) == v for k, v in keep.items()), (es2, by))
check("collapse: no tag left", not any("_chain" in x for v in by.values() for x in v))

# the head key: the looks of the things, the gloss for the same word skipped
BOOK = Path.home() / "audiobooks" / "knight_forged_v3"         # (read only)
if BOOK.exists():
    s = {"id": "action-05m16s-003162_s0", "kind": "action", "framing": "medium", "place": "The Burnt Mountain",
         "who": ["Edric"], "picture": "Edric stands by the forge, holding a poker and the bellows",
         "keyshot": "Edric stands by the forge, holding a poker and the bellows",
         "things": {"poker": "a straight iron rod", "forge": "a stone hearth with glowing coals"}}
    got = S._key_prompt(BOOK, "01_chapter_1", "action-05m16s-003162", s)
    check("key text: each thing's look after its name, the hand gloss for that word skipped, others kept",
          "poker (a straight iron rod)" in got and "hooked tip" not in got and
          "forge (a stone hearth with glowing coals)" in got and "pleated leather" in got, got)
    s.pop("things")
    got = S._key_prompt(BOOK, "01_chapter_1", "action-05m16s-003162", s)
    check("key text without things: as before (the gloss)", "hooked tip" in got and "(a straight" not in got, got)

# the chain thread, end to end on fake clips (review: agent)
with tempfile.TemporaryDirectory() as td:
    sd = Path(td)
    L = sd / "ltx"
    (L / "out").mkdir(parents=True)
    (L / "emb").mkdir()
    key = sd / "key.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=192x108", "-frames:v", "1",
                    str(key)], check=True)
    shot = {"id": "action-000030_s0", "who": ["Edric", "Vorgrath"],
            "chain": {"lines": ["line one.", "line two."], "shots": [2, 3]}}
    st = S._chain_state(L, shot, key)
    for t_ in st["lines"]:
        (L / "emb" / f"{S._chain_emb(shot['id'], t_)}.pt").write_text("x")
    S.REVIEW_TIMEOUT = 30
    asked = []

    def vl(items):
        asked.append(items)
        return {"pick": {"best": "B", "next": "C"}}
    CH.vl_ask = vl
    more = L / "more.json"
    more.write_text("[]")
    firsts = S._chain_jobs(st, 1)
    stop = threading.Event()
    logs = []
    th_ = threading.Thread(target=S._chains_as_they_land, args=(sd, [shot["id"]], more, {j["name"] for j in firsts},
                                                                stop, logs.append, "agent"), daemon=True)
    S.CHAIN_POLL = 0.2
    th_.start()

    def render(j, color):
        mp4 = L / "out" / f"{j['name']}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=192x108:r=24",
                        "-frames:v", "97", "-c:v", "libopenh264", str(mp4)], check=True)
        (L / "out" / f"{j['name']}.json").write_text(json.dumps(j))
    for j, col in zip(firsts, ("red", "green", "blue")):
        render(j, col)
    card = sd / "review" / shot["id"] / "1" / "card.json"
    for _ in range(100):
        if card.exists():
            break
        time.sleep(0.2)
    cd = json.loads(card.read_text()) if card.exists() else {}
    o = asked[0][0]["images"] if asked else []
    order = json.loads((L / "chains" / f"{shot['id']}.json").read_text())["judge"]["1.0"]["order"] if asked else []
    check("judge: one ask, the 3 sheets as 3 images", len(asked) == 1 and len(o) == 3, asked)
    check("judge: 'B' maps back through the shown order", cd.get("judge_pick") == "ABC"[order[1]] if order else False,
          (cd.get("judge_pick"), order))
    check("card: line, 3 takes, a side-by-side sheet", cd.get("line") == "line one." and len(cd.get("takes", {})) == 3
          and Path(cd.get("sheet", "/x")).exists(), cd)
    check("card ready said in the log", any("REVIEW card ready" in m for m in logs), logs[-3:])
    (card.parent / "choice.json").write_text(json.dumps({"pick": "C"}))
    nxt = []
    for _ in range(100):
        nxt = [j for j in json.loads(more.read_text()) if j["name"].startswith(shot["id"] + "_x2")]
        if len(nxt) == 3:
            break
        time.sleep(0.2)
    check("line 2: 3 takes queued, chained from the reviewer's pick C, first in the queue",
          len(nxt) == 3 and all(j["chain_from"] == firsts[2]["name"] and j["chain_frame"] == 96 and j["overlap"] == 1
                                and j["chain_strength"] == 1.0 and j["priority"] == 1 for j in nxt), nxt)
    for j, col in zip(nxt, ("red", "green", "blue")):
        render(j, col)
    card2 = sd / "review" / shot["id"] / "2" / "card.json"
    for _ in range(100):
        if card2.exists():
            break
        time.sleep(0.2)
    (card2.parent / "choice.json").write_text(json.dumps({"override": "the man in armour falls back."}))
    for _ in range(100):
        if not th_.is_alive():
            break
        time.sleep(0.2)
    st = json.loads((L / "chains" / f"{shot['id']}.json").read_text())
    check("override: the line's words replaced, version 1, waiting for an encode (the next hold)",
          st["lines"][1] == "the man in armour falls back." and st["ver"][1] == 1 and st.get("blocked") == "encode"
          and not th_.is_alive(), st)
    lg = [json.loads(x) for x in (sd / "review" / "log.jsonl").read_text().splitlines()]
    check("log: every choice next to the judge's pick and the director's line",
          len(lg) == 2 and lg[0]["choice"] == {"pick": "C"} and lg[1]["director_line"] == "line two."
          and all("judge_pick" in x for x in lg), lg)
    # the next round: encoded, the new takes rendered, accepted -> joined
    (L / "emb" / f"{S._chain_emb(shot['id'], st['lines'][1])}.pt").write_text("x")
    st.pop("blocked")
    S._chain_save(L, st)
    j2 = S._chain_jobs(st, 2)
    check("override takes: new names (v1) and new seeds", all("_x2v1" in j["name"] for j in j2) and
          j2[0]["seed"] != nxt[0]["seed"], [j["name"] for j in j2])
    for j, col in zip(j2, ("red", "green", "blue")):
        render(j, col)
    (sd / "review" / shot["id"] / "2.1").mkdir(parents=True, exist_ok=True)
    (sd / "review" / shot["id"] / "2.1" / "choice.json").write_text(json.dumps({"accept": True}))
    S._chains_as_they_land(sd, [shot["id"]], None, set(), threading.Event(), logs.append, "agent", True)
    st = json.loads((L / "chains" / f"{shot['id']}.json").read_text())
    joined = L / "out" / f"{shot['id']}.mp4"
    nfr = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", str(joined)], capture_output=True,
                             text=True).stdout.strip() or 0) if joined.exists() else 0
    check("joined: 2 lines = 193 frames (the overlap frame dropped)", st["done"] and nfr == S.chain_frames(2) == 193,
          (st["done"], nfr))
    import os
    os.utime(key, (time.time() + 5, time.time() + 5))       # a new key: the chain starts again
    st = S._chain_state(L, shot, key)
    check("a new key: takes, joined clip and the old review cards set aside (no old choice answers a new card)",
          not st["done"] and not joined.exists() and not (sd / "review" / shot["id"]).exists()
          and any(p.name.startswith(shot["id"] + ".old-") for p in (sd / "review").iterdir()), st)

# the narration gap
pauses = [{"t": 10.0, "g": 2.5, "shot": "a"}]
check("map: before the break unchanged, at/after it moved", S.pause_map(9.99, pauses) == 9.99 and
      S.pause_map(10.0, pauses) == 12.5 and S.pause_map(20.0, pauses) == 22.5)
es = [{"id": "m0", "t": 5.0, "dur": 5.0}, {"id": "m1", "t": 10.0, "dur": 3.0}]
by = {"m0": [{"id": "m0_s0", "dur": 5.0, "chain": {}}], "m1": [{"id": "m1_s0", "dur": 3.0}]}
pauses = [{"t": 9.5, "g": 3.08, "shot": "m0_s0"}]
S.apply_pauses(es, by, pauses)
check("apply: the chain's slot grows by its pause, the next moment moves", abs(S._spans(es)["m0"] - 8.08) < 1e-6 and
      abs(by["m0"][0]["dur"] - 8.08) < 1e-6 and abs(es[1]["t"] - 13.08) < 1e-6 and by["m1"][0]["dur"] == 3.0, (es, by))
# where the narration waits (11-05 user: "you don't cut to fit.. you wait at the end of the line")
q = [(8.0, 16.0)]                                   # a spoken line 8-16 s
check("break: the sentence break in the slot nearest its end", S.pause_point(2.0, 10.0, [4.0, 7.5, 12.0], [], 30.0) == 7.5)
check("break: never inside a spoken line", S.pause_point(2.0, 10.0, [4.0, 9.0], q, 30.0) == 4.0)
check("break: none in the slot, the slot ends inside a line -> the first break after the line ends, however far",
      S.pause_point(2.0, 10.0, [9.0, 15.0, 22.5, 25.0], q, 30.0) == 22.5)
check("break: none in the slot, mid-sentence -> the next sentence break", S.pause_point(2.0, 6.0, [7.0], [], 30.0) == 7.0)
check("break: none after -> the stretch's end", S.pause_point(2.0, 10.0, [], q, 30.0) == 30.0)
es = [{"id": "c", "t": 2.0, "dur": 4.0}, {"id": "x", "t": 6.0, "dur": 3.0}, {"id": "y", "t": 9.0, "dur": 4.0}]
by = {"c": [{"id": "c_s0", "dur": 4.0, "chain": {}}], "x": [{"id": "x_s0", "dur": 3.0, "kind": "quiet"}],
      "y": [{"id": "y_s0", "dur": 4.0, "kind": "quiet"}]}
pz = [{"t": 9.0, "t_pic": 6.0, "g": 4.042, "shot": "c_s0"}]   # the slot ends at 6 inside a line; its break at 9
S.apply_pauses(es, by, pz)
check("past the slot: the chain plays its full length, never cut; nothing dropped: the later moments keep their "
      "shots, moved on by the pause",
      [e["id"] for e in es] == ["c", "x", "y"] and set(by) == {"c", "x", "y"} and abs(by["c"][0]["dur"] - 8.042) < 1e-6
      and abs(S._spans(es)["c"] - 8.042) < 1e-6 and abs(es[1]["t"] - 10.042) < 1e-6 and abs(es[2]["t"] - 13.042) < 1e-6
      and by["x"][0]["dur"] == 3.0, (es, by))
check("past the slot: the narration (audio map) waits at the break, the pictures at the slot's end",
      S.pause_map(7.0, pz) == 7.0 and S.pause_map(9.0, pz) == 13.042 and S.pause_map(7.0, pz, "t_pic") == 11.042)
with tempfile.TemporaryDirectory() as td:
    src = Path(td) / "v.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:r=16000:d=20", str(src)], check=True)
    o = S._with_silence(src, [{"t": 5.0, "g": 1.25}, {"t": 12.0, "g": 2.0}], Path(td) / "o.wav")
    import wave
    w = wave.open(str(o))
    n = w.getnframes()
    import numpy as np
    a = np.frombuffer(w.readframes(n), np.int16)
    check("silence: sample-exact length (20 + 3.25 s)", n == int(23.25 * 16000), n)
    check("silence: silent where inserted, sound around it",
          np.abs(a[int(5.2 * 16000):int(6.1 * 16000)]).max() == 0 and np.abs(a[int(4.5 * 16000):int(4.9 * 16000)]).max() > 1000
          and np.abs(a[int(6.4 * 16000):int(6.8 * 16000)]).max() > 1000)
print({"fails": fails})
sys.exit(1 if fails else 0)
