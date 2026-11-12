"""11-12: an independent review of the supplied-keys options (keys staged elsewhere, rendered by the flow) found that
a talk shot could render with its speaker not in the key (someone else mouthed the line), that the plan's side words
contradicted the key's order, that the run wrote into the book it was pointed at, that it validated after writing,
that chains kept old takes under changed options, that the lip-sync zone ignored where the key puts the speaker, that
the scene facts reached talk and action words, and that loose camera words dropped the hold sentence. Each rule here."""
import hashlib, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S
from PIL import Image

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

def raises(fn, *words):
    try:
        fn()
    except RuntimeError as x:
        return all(w in str(x) for w in words), str(x)
    return False, "no error"

LOOKS = {"Person A": "a man of about 40, wearing a dark frock coat",
         "Person B": "a woman of about 30, wearing a black mourning dress",
         "Person C": "a man of about 35, wearing a tweed jacket"}
A, B, C = LOOKS
dA, dB, dC = "the man in the dark frock coat", "the woman in the black mourning dress", "the man in the tweed jacket"
STEM = "01_x"


def make_book(clips=()):
    b = Path(tempfile.mkdtemp(prefix="supplied_"))
    vd = b / "work" / "video"
    (vd / STEM / "shots" / "ltx" / "out").mkdir(parents=True)
    (vd / "look.json").write_text(json.dumps({"characters": LOOKS, "places": {}}))
    (b / "book.cast.json").write_text(json.dumps({"characters": {k: {} for k in LOOKS}}))
    for sid in clips:               # a shot the book already has: its key and its clip
        d = vd / STEM / "shots" / sid.rsplit("_s", 1)[0] / sid
        d.mkdir(parents=True)
        Image.new("RGB", (S.W, S.H), (9, 9, 9)).save(d / "key.png")
        (vd / STEM / "shots" / "ltx" / "out" / f"{sid}.mp4").write_text("the book's clip")
        (vd / STEM / "shots" / "ltx" / "out" / f"{sid}.json").write_text("{}")
    return b


def make_keys(ids, facts=None, size=None, mode="RGB", colour=(40, 50, 60)):
    k = Path(tempfile.mkdtemp(prefix="keys_"))
    for i in ids:
        Image.new(mode, size or (S.W, S.H), colour if mode == "RGB" else colour + (255,)).save(k / f"{i}.png")
    for i, vis in (facts or {}).items():
        (k / f"{i}.shot_facts.json").write_text(json.dumps({"visible": vis}))
    return k


def make_plan(shots, es):
    f = Path(tempfile.mkdtemp(prefix="plan_")) / "plan.json"
    by = {"_es": es}
    for s in shots:
        by.setdefault(s["id"].rsplit("_s", 1)[0], []).append(s)
    f.write_text(json.dumps(by))
    return f


def tree(root):
    return {str(p.relative_to(root)): (hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else "dir")
            for p in sorted(Path(root).rglob("*")) if not p.is_symlink()}


S.MF._dress = lambda *a, **k: {}
S._ctx = lambda b, st: (None, SimpleNamespace(book=Path(b), stem=st))
ran = []
S._clips = lambda book, stem, es, shots, log, op=None: ran.append((Path(book), [dict(s) for s in shots]))
logs = []
log = logs.append

TALK = {"id": "talk-000100_s0", "kind": "talk", "who": [A, B], "speaker": A, "dur": 4.0}
ACT = {"id": "action-000200_s0", "kind": "action", "who": [B, A], "dur": 4.0,
       "action": f"{dB} on the left looks up at {dA} on the right."}
QUIET = {"id": "quiet-000300_s0", "kind": "quiet", "who": [B], "dur": 3.0, "action": f"{dB} looks down."}
ES = [{"id": "talk-000100", "kind": "talk", "speaker": "Person A", "t": 1.0, "dur": 4.0},
      {"id": "action-000200", "kind": "action", "t": 5.0, "dur": 4.0},
      {"id": "quiet-000300", "kind": "quiet", "t": 9.0, "dur": 3.0}]
vis = lambda *names, boxes=None: [dict({"name": n}, **({"box": boxes[i]} if boxes else {})) for i, n in enumerate(names)]

# --- 8: the camera-hold sentence is dropped only for a camera move ----------------------------------------------------
S.HOLD_CAMERA, S.SCENE_FACTS, S.KEY_PEOPLE = True, "", {}
HOLD = "The camera holds the opening picture's framing"
for words in ("A close-up of her hands on the table.", "The camera holds on her face.",
              "She pushes in the drawer.", "He pulls back to look at her."):
    check(f"hold kept: {words!r}", HOLD in S.with_facts("x.", {"id": "a", "kind": "action", "action": words}))
for words in ("The camera pans left across the room.", "The camera slowly pushes in on his face.",
              "A slow zoom in on the letter.", "A tracking shot along the wall.", "Dolly in toward the door."):
    check(f"hold dropped for a camera move: {words!r}",
          HOLD not in S.with_facts("x.", {"id": "a", "kind": "action", "action": words}))
check("a talk prompt that already holds the camera gets no second sentence",
      HOLD not in S.with_facts("Static camera, the framing does not change. x.", {"id": "a", "kind": "talk"}))

# --- 7: the scene's facts go on quiet shots only -----------------------------------------------------------------------
S.SCENE_FACTS, S.HOLD_CAMERA = "The fire is lit.", False
check("quiet words carry the facts", "The fire is lit." in S.with_facts("x.", {"id": "q", "kind": "quiet"}))
for kind in ("talk", "action", None):
    check(f"{kind} words carry no facts", "The fire is lit." not in S.with_facts("x.", {"id": "q", "kind": kind}))
check("a chain line carries no facts", "The fire is lit." not in S._chain_words({"id": "c"}, "She walks to the chair."))
check("the per-shot free text option is gone", not hasattr(S, "SHOT_FACTS") and not hasattr(S, "shot_facts_words"))
S.SCENE_FACTS = ""

# --- no option: the words, seeds and chain signature are the flow's own -------------------------------------------------
S.HOLD_CAMERA, S.KEY_PEOPLE, S.KEY_SIDES, S.KEY_BOXES, S.QUIET_SEEDS = False, {}, {}, {}, ()
for kind in ("talk", "action", "quiet", None):
    check(f"no option: {kind} words unchanged", S.with_facts("Some words.", {"id": "a", "kind": kind}) == "Some words.")
check("no option: a chain line unchanged", S._chain_words({"id": "c"}, "She sits.") == "She sits.")
L = Path(tempfile.mkdtemp(prefix="chain_"))
(L / "out").mkdir()
key = L / "key.png"; key.write_text("k")
ch = {"id": "action-000900_s0", "who": [A], "chain": {"lines": ["She walks.", "She sits."]}}
st0 = S._chain_state(L, ch, key)
check("no option: the chain's signature is lines + key time, as before",
      st0["sig"] == S.MF._h(json.dumps([ch["chain"]["lines"], key.stat().st_mtime_ns])), st0["sig"])
check("no option: side words from the flow's layout", S._side({"id": "a", "who": [A, B]}, B) == " on the right")
check("no option: the speaker's zone is the layout's", S._speaker_box({"id": "t", "kind": "talk", "who": [A, B],
      "speaker": A}, A) == S.talk_layout({"kind": "talk", "who": [A, B], "speaker": A})[0][0])

# --- 5: a chain is new when its rendered words or its key's people change -----------------------------------------------
(L / "out" / "action-000900_s0_x1a.mp4").write_text("old take")
S.HOLD_CAMERA = True
st1 = S._chain_state(L, ch, key)
check("the camera sentence on: the chain starts again", st1["sig"] != st0["sig"] and
      not (L / "out" / "action-000900_s0_x1a.mp4").exists())
check("... and stays while nothing changes", S._chain_state(L, ch, key)["sig"] == st1["sig"])
S.KEY_PEOPLE = {"action-000900_s0": 3}
st2 = S._chain_state(L, ch, key)
check("the key's people count is part of the chain", st2["sig"] != st1["sig"])
S.KEY_PEOPLE = {"action-000900_s0": 2}
check("... a changed count starts it again", S._chain_state(L, ch, key)["sig"] != st2["sig"])
S.HOLD_CAMERA, S.KEY_PEOPLE = False, {}

# --- 4: everything is checked before anything is written ----------------------------------------------------------------
b = make_book()
before = tree(b)
k = make_keys([TALK["id"], ACT["id"]])
Image.new("RGB", (S.W // 2, S.H // 2)).save(k / f"{ACT['id']}.png")        # the SECOND key is the wrong size
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([TALK, ACT], ES), k, [TALK["id"], ACT["id"]], log=log),
                 "nothing was written", ACT["id"])
check("a wrong-size key stops the run", ok, msg)
check("... before the first key was put in place", tree(b) == before and not ran)
k = make_keys([QUIET["id"]], mode="RGBA")
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([QUIET], ES), k, [QUIET["id"]], log=log), "RGB")
check("a key that is not RGB stops the run", ok, msg)
k = make_keys([TALK["id"]])
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, key_people_=True),
                 "shot_facts.json")
check("--key-people: a shot without its facts file stops the run (no silent mix)", ok, msg)
k = make_keys([TALK["id"]], {TALK["id"]: vis("Nobody Known")})
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, key_people_=True),
                 "not a person of the book")
check("a facts file naming an unknown person stops the run", ok, msg)
(k / f"{TALK['id']}.shot_facts.json").write_text("{not json")
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, key_people_=True),
                 "nothing was written")
check("a facts file that does not parse stops the run", ok, msg)
k = make_keys([TALK["id"]], {TALK["id"]: vis(A, B, boxes=[[0.1, 0.1, 0.4, 0.9], [0.9, 0.1, 0.6, 0.9]])})
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, key_people_=True),
                 "box")
check("a box that is not left < right in 0-1 stops the run", ok, msg)
k = make_keys([QUIET["id"]])
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([QUIET], ES), k, [QUIET["id"]], log=log,
                                         quiet_seeds=("7", "x")), "seeds")
check("a seed that is not a number stops the run", ok, msg)
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([QUIET], ES), k, [QUIET["id"]], log=log,
                                         facts=k / "nope.txt"), "no such file")
check("a missing scene facts file stops the run", ok, msg)
check("none of those wrote into the book", tree(b) == before and not ran)

# --- 3: a book with clips is never overwritten in place; a work root takes every write ---------------------------------
b = make_book(clips=[TALK["id"], ACT["id"]])
before = tree(b)
k = make_keys([TALK["id"]])
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log), "work root")
check("a book with clips from other keys: refused without a work root", ok and tree(b) == before, msg)
root = Path(tempfile.mkdtemp(prefix="root_"))
ran.clear()
out = S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, work_root=root)
wb = root / b.name
sdw = wb / "work" / "video" / STEM / "shots"
check("with a work root the render runs in <root>/<book name>", ran and ran[-1][0] == wb and out[0].is_relative_to(wb))
check("... the supplied key is there, a real file", (sdw / "talk-000100" / TALK["id"] / "key.png").is_file() and
      not (sdw / "talk-000100" / TALK["id"] / "key.png").is_symlink() and
      (sdw / "talk-000100" / TALK["id"] / "key.png").read_bytes() == (k / f"{TALK['id']}.png").read_bytes())
check("... the given shot's old clip is not carried over", not (sdw / "ltx" / "out" / f"{TALK['id']}.mp4").exists())
check("... another shot's clip is there for a cut, linked", (sdw / "ltx" / "out" / f"{ACT['id']}.mp4").is_symlink())
check("... the looks are the book's own", (wb / "work" / "video" / "look.json").is_symlink())
check("... the status line of the book is left alone", S.NO_STATUS is True)
# what a render does there: a stale clip dropped, new files written, the shots folder's own files rewritten
(sdw / "ltx" / "out" / f"{ACT['id']}.mp4").unlink()
(sdw / "ltx" / "out" / f"{TALK['id']}.mp4").write_text("a new clip")
(sdw / "ltx" / "segs.txt").write_text("x")
(sdw / "review").mkdir(exist_ok=True)
check("the book itself is byte for byte as it was", tree(b) == before)
ran.clear()
S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, work_root=root)
check("a second run there resumes: the new clip is kept", (sdw / "ltx" / "out" / f"{TALK['id']}.mp4").read_text() ==
      "a new clip" and any("already in place" in x for x in logs))
ok, msg = raises(lambda: S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log,
                                         work_root=b / "work" / "x"), "inside the book")
check("a work root inside the book is refused", ok, msg)
b2 = make_book(clips=[TALK["id"]])
S.supplied_keys(b2, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, in_place=True)
check("--in-place replaces the book's key where it is (the old behaviour, asked for)",
      (b2 / "work" / "video" / STEM / "shots" / "talk-000100" / TALK["id"] / "key.png").read_bytes() ==
      (k / f"{TALK['id']}.png").read_bytes() and S.NO_STATUS is False)
b3 = make_book()
S.supplied_keys(b3, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log)
check("a book with no clips for the shots needs neither option", ran[-1][0] == b3)

# --- 1: a talk shot's speaker must be in its key ---------------------------------------------------------------------
b = make_book()
ran.clear(); logs.clear()
k = make_keys([TALK["id"], QUIET["id"]], {TALK["id"]: vis(B), QUIET["id"]: vis(B)})
out = S.supplied_keys(b, STEM, make_plan([TALK, QUIET], ES), k, [TALK["id"], QUIET["id"]], log=log, key_people_=True)
check("the speaker is not in the key: the talk shot is not rendered",
      [s["id"] for s in ran[-1][1]] == [QUIET["id"]] and TALK["id"] in S.STOPPED)
why = S.STOPPED.get(TALK["id"], "")
check("... and the message names the speaker, the key's people and the plan's", A in why and B in why and
      "the plan listed" in why and any("NOT RENDERED" in x and TALK["id"] in x for x in logs), why)
check("... no key was put in place for it", not (b / "work" / "video" / STEM / "shots" / "talk-000100").exists())
OTS = dict(TALK, ots=B)
ran.clear(); logs.clear()
k = make_keys([TALK["id"]], {TALK["id"]: vis(A)})
S.supplied_keys(b, STEM, make_plan([OTS], ES), k, [TALK["id"]], log=log, key_people_=True)
check("planned over a shoulder the key does not show: reported, taken as the key shows it",
      ran[-1][1][0].get("ots") is None and any("over the shoulder" in x for x in logs), logs)
ran.clear(); logs.clear()
k = make_keys([TALK["id"]], {TALK["id"]: vis(B, A)})
S.supplied_keys(b, STEM, make_plan([OTS], ES), k, [TALK["id"]], log=log, key_people_=True)
check("... and kept when the key shows both", ran[-1][1][0].get("ots") == B and not any("over the shoulder" in x for x in logs))

# --- 2: side words follow the key ---------------------------------------------------------------------------------------
ran.clear()
k = make_keys([ACT["id"]], {ACT["id"]: vis(A, B)})         # the key has them the other way round
S.supplied_keys(b, STEM, make_plan([ACT], ES), k, [ACT["id"]], log=log, key_people_=True)
act = ran[-1][1][0]["action"]
check("two people, the key's order differs: the director's side words are the key's",
      act == f"{dB} on the right looks up at {dA} on the left.", act)
cx = SimpleNamespace(book=b, stem=STEM)
hold = S._pose_hold(cx, dict(ran[-1][1][0], pose_held={B: "sit"}))
check("... and the pose hold says the same side", f"{dB} on the right stays".lower() in hold.lower(), hold)
ACT3 = {"id": "action-000200_s0", "kind": "action", "who": [A, B, C], "dur": 4.0,
        "action": f"{dA} on the left turns to {dB} in the centre while {dC} on the right watches.",
        "chain": {"lines": [f"{dC} on the right steps forward.", f"{dB} in the centre sits."]}}
ran.clear()
k = make_keys([ACT3["id"]], {ACT3["id"]: vis(C, A, B)})
S.supplied_keys(b, STEM, make_plan([ACT3], ES), k, [ACT3["id"]], log=log, key_people_=True)
got = ran[-1][1][0]
check("three people: every side rewritten from the key",
      got["action"] == f"{dA} in the centre turns to {dB} on the right while {dC} on the left watches.", got["action"])
check("... in the chain's lines too", got["chain"]["lines"] == [f"{dC} on the left steps forward.", f"{dB} on the right sits."],
      got["chain"]["lines"])
ran.clear()
k = make_keys([ACT["id"]], {ACT["id"]: vis(B)})            # the key shows only one of the two
S.supplied_keys(b, STEM, make_plan([ACT], ES), k, [ACT["id"]], log=log, key_people_=True)
act = ran[-1][1][0]["action"]
check("someone the key does not show, and someone alone in it, get no side", act == f"{dB} looks up at {dA}.", act)
ran.clear()
k = make_keys([ACT["id"]], {ACT["id"]: vis(B, A, boxes=[[0.05, 0.1, 0.3, 0.9], [0.62, 0.1, 0.95, 0.9]])})
S.supplied_keys(b, STEM, make_plan([ACT], ES), k, [ACT["id"]], log=log, key_people_=True)
check("with boxes the side is where the box is", S.KEY_SIDES[ACT["id"]] == {B: "on the left", A: "on the right"},
      S.KEY_SIDES)
check("boxes that give two people the same word fall back to their order",
      S.key_sides([A, B], {A: [0.0, 0, 0.2, 1], B: [0.15, 0, 0.35, 1]}) == {A: "on the left", B: "on the right"})
S.KEY_SIDES = {"x_s0": {A: "on the left", C: "on the right"}}
same = {"Person A": LOOKS[A], "Person C": LOOKS[A]}        # two people with one description
bs = Path(tempfile.mkdtemp(prefix="same_")); (bs / "work" / "video").mkdir(parents=True)
(bs / "work" / "video" / "look.json").write_text(json.dumps({"characters": same, "places": {}}))
(bs / "book.cast.json").write_text(json.dumps({"characters": {k_: {} for k_ in same}}))
txt = S.reside(S.RG.Book(bs), SimpleNamespace(book=bs, stem=STEM), {"id": "x_s0"}, f"{dA} on the right nods.")
check("two people with the same description: the side cannot say which, so it goes", txt == f"{dA} nods.", txt)
check("a shot without key people: the director's words untouched",
      S.reside(S.RG.Book(b), cx, {"id": "other_s0"}, ACT["action"]) == ACT["action"])

# --- 6: the key's boxes set the lip-sync zone and the talk words' position ---------------------------------------------
ran.clear()
k = make_keys([TALK["id"]], {TALK["id"]: vis(B, A, boxes=[[0.05, 0.2, 0.35, 0.9], [0.55, 0.1, 0.8, 0.95]])})
S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, key_people_=True)
s_ = ran[-1][1][0]
zone = S._speaker_box(s_, "Person A")
check("the speaker's zone is their own box, a little wider, full height",
      zone == [round(0.55 - S.KEY_BOX_PAD, 3), 0.0, round(0.8 + S.KEY_BOX_PAD, 3), 1.0], zone)
tp = S._talk_prompt(SimpleNamespace(gender_of=lambda look: "man"), cx,
                    {"speaker": "Person A", "words": "Good morning."}, "", s_)
check("the talk words place the speaker where the key does, with no 'nearest the viewer'",
      f"{A} on the right," in tp and "nearest the viewer" not in tp, tp)
ran.clear()
k = make_keys([TALK["id"]], {TALK["id"]: vis(B, A)})       # no boxes: the zone as before, sides from the order
S.supplied_keys(b, STEM, make_plan([TALK], ES), k, [TALK["id"]], log=log, key_people_=True)
s_ = ran[-1][1][0]
check("without boxes the zone is today's (the layout's for the key's order)",
      S._speaker_box(s_, "Person A") == S.talk_layout(s_)[s_["who"].index(A)][0])

# --- 9: the small ones --------------------------------------------------------------------------------------------------
k = make_keys([QUIET["id"]])
S.supplied_keys(b, STEM, make_plan([QUIET], ES), k, [QUIET["id"]], log=log, hold=True,
                facts=(k / "f.txt") if (k / "f.txt").write_text("The fire is lit") else None, quiet_seeds=("7", "1042"))
check("options on ...", S.HOLD_CAMERA and S.SCENE_FACTS == "The fire is lit." and S.QUIET_SEEDS == (7, 1042))
S.supplied_keys(b, STEM, make_plan([QUIET], ES), k, [QUIET["id"]], log=log)
check("... and off again on the next call that leaves them out",
      not S.HOLD_CAMERA and S.SCENE_FACTS == "" and S.QUIET_SEEDS == () and not S.KEY_PEOPLE and not S.KEY_SIDES
      and not S.KEY_BOXES and not S.STOPPED)
check("no seeds given: one take, the retake as before", S.quiet_take_seeds(False) == (S.QUIET_SEED,) and
      S.quiet_take_seeds(True) == (S.QUIET_SEED, S.RETAKE))
S.QUIET_SEEDS = (S.QUIET_SEED, S.RETAKE, 2042)
rt = S.quiet_take_seeds(True)
check("given seeds: the retake is a seed not yet taken", rt[:3] == S.QUIET_SEEDS and len(set(rt)) == 4, rt)
S.QUIET_SEEDS = ()
# the seeds' takes are judged together (judge_takes saw only the takes a retake makes)
jb = make_book()
jo = jb / "work" / "video" / STEM / "shots" / "ltx" / "out"
for n_ in ("", "_t2", "_t3", "_t4", "_t5", "_t6", "_t7"):
    (jo / f"{QUIET['id']}{n_}.mp4").write_text("x")
seen = []
S.MF.fight_clip_ok = lambda mp4, people=None: seen.append(mp4.stem) or []
_run = S.subprocess.run
S.subprocess.run = lambda cmd, *a, **k_: (SimpleNamespace(returncode=0, stdout="") if "systemctl" in map(str, cmd)
                                         else _run(cmd, *a, **k_))
S.QUIET_SEEDS = (7, 1042, 2042, 3042, 4042, 5042, 6042)
S.judge_takes(jb, STEM, [dict(QUIET)], log)
check("seven seeds (more than a retake ever makes): all seven takes are judged", len(seen) == 7, seen)
S.QUIET_SEEDS = ()
S.subprocess.run = _run
pl = make_plan([TALK, ACT, QUIET], ES)
ok, msg = raises(lambda: S.assemble_stretch(b, STEM, pl, QUIET["id"], TALK["id"], b / "o.mp4"), "comes after")
check("a stretch whose first shot comes after its last: a clear error", ok, msg)
ok, msg = raises(lambda: S.assemble_stretch(b, STEM, pl, "nope_s0", TALK["id"], b / "o.mp4"), "not in the plan")
check("a stretch naming a shot the plan does not have: a clear error", ok, msg)
ok, msg = raises(lambda: S.assemble_stretch(b, STEM, pl, TALK["id"], QUIET["id"], b / "o.mp4"), "final mix")
check("a chapter without a final mix: a clear error", ok, msg)

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
