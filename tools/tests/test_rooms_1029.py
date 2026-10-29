"""10-29: the rooms as their pictures are drawn. Layouts from the room pictures go to the director; stacked side words
read as one; `bookcaster look --do` changes a place. Built on test_blocking_1028's scripted director.
10-28 (blocking + group talk) built on the 10-27 audit, end to end on CPU: the real direct() on a synthetic book with a scripted director (no Strata).
Checks the structure the audit asked for: one side map for talk and other shots, talk shots in the scene's postures,
a setting that names people sent back, the cache holding the model's words (replayed through today's code), voice
cuts as long as their shots, one name per person from the voice labels."""
import json, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S
import mannequin_flow as MF
from tts_reader import animate

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

stem = "01_x"
b = Path(tempfile.mkdtemp(prefix="direct_e2e_"))
vd = b / "work" / "video" / stem
vd.mkdir(parents=True)
(b / "work" / "video" / "look.json").write_text(json.dumps({
    "characters": {"Sherlock Holmes": "a man of about 40, wearing a dark frock coat",
                   "Helen Stoner": "a woman of about 30, wearing a black mourning dress",
                   "Julia Stoner": "a young woman, wearing a white nightdress"},
    "places": {"Baker Street Sitting Room (inside)": "a Victorian sitting room with a fireplace and two armchairs"}}))
(b / "book.cast.json").write_text(json.dumps({"characters": {"Holmes": {}, "Stoner": {}, "Julia": {}}}))
(vd / "plan.json").write_text(json.dumps({"look": {"characters": {"Sherlock Holmes": "", "Helen Stoner": "", "Julia Stoner": ""}}}))
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "The lady sat in the chair by the window. Holmes stood by the fire."},
      {"t": 6.0, "kind": "quote", "speaker": "Holmes", "text": "Pray be precise as to details, madam."},
      {"t": 9.0, "kind": "narration", "speaker": "narrator", "text": "She clasped her hands and looked at him."},
      {"t": 14.0, "kind": "quote", "speaker": "Stoner", "text": "It is easy for me to be so."}]
(b / "work" / f"{stem}.timeline.json").write_text(json.dumps(tl))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=20", str(b / "work" / f"{stem}.mp3")],
               check=True)
es = [{"id": "action-000000", "t": 0.0, "dur": 20.0, "route": "shot", "where": "Baker Street Sitting Room (inside)",
       "who": ["Sherlock Holmes", "Helen Stoner", "Julia Stoner"]}]

BREAK = {"scenes": [{"start": 0.0, "place": "Baker Street Sitting Room (inside)", "who": ["Holmes", "Helen Stoner", "Julia Stoner"],
                     "what": "Helen tells Holmes her story.", "cutaways": [], "opens": "Helen sits; Holmes stands.",
                     "blocking": "Helen Stoner on the left, Sherlock Holmes in the centre, Julia Stoner on the right; "
                                 "Holmes and Helen talk; one line from Holmes to Julia, shot with all three."}]}
def shot_q(start, setting):
    return {"start": start, "kind": "quiet", "place": "Baker Street Sitting Room (inside)", "framing": "medium",
            "people": [{"name": "Helen Stoner", "side": "left", "pose": "sitting", "at": "in the chair by the window",
                        "facing": "Holmes", "holds": ""},
                       {"name": "Holmes", "side": "right", "pose": "standing", "at": "by the fire", "facing": "Helen",
                        "holds": ""}],
            "setting": setting, "action": "Helen Stoner clasps her hands.", "pose_end": {}}
BAD = {"state_end": "x", "shots": [shot_q(0.0, "Morning light. Holmes stands by the fire."),
                                   shot_q(3.0, "Morning light."),
                                   {"start": 6.0, "kind": "talk", "speaker": "Holmes", "listener": "Julia", "framing": "group"},
                                   shot_q(9.0, "Morning light."),
                                   {"start": 14.0, "kind": "talk", "speaker": "Stoner", "listener": "Holmes",
                                    "framing": "over-the-shoulder"}]}
GOOD = json.loads(json.dumps(BAD))
GOOD["shots"][0]["setting"] = "Morning light through the window."
calls = []
def converse(url, model, messages, max_tokens=5000, effort=""):
    calls.append(messages[-1]["content"][:60])
    if "SCENES" in messages[-1]["content"]:
        return json.dumps(BREAK), 0
    return json.dumps(BAD if len([c for c in calls if not c.startswith("You are a film")]) == 1 else GOOD), 0
S._converse = converse
animate.igpu = lambda *a, **k: None
REAL_RUN = subprocess.run
S.subprocess.run = (lambda real: (lambda cmd, *a, **k: real(cmd, *a, **k) if cmd and cmd[0] == "ffmpeg"
                                  else type("R", (), {"returncode": 1, "stdout": ""})()))(subprocess.run)
S._cast = lambda book, stem_, e: (e.get("who") or [], [])
MF._dress = lambda *a, **k: {}


asks = []
def conv2(url, model, messages, max_tokens=5000, effort=""):
    asks.append(messages[-1]["content"])
    return converse(url, model, messages, max_tokens, effort)
S._converse = conv2
LY = "The bed against the left wall, the fireplace in the centre of the back wall, the window on the right."
S.direct(b, stem, es, log=lambda m: None, layouts={"Baker Street Sitting Room": LY})
check("the director gets the room as its picture is drawn", any("As every picture of it is drawn: " + LY[:-1] in a
      for a in asks), [a[:200] for a in asks if "fixed" in a][:1])
check("the rule: the room's things stay where drawn; a desk is sat BEHIND; a waking man sits up in his bed",
      "THE ROOM AS DRAWN" in S.SHOT_RULES and "behind the" in S.SHOT_RULES and "sits up in it" in S.SHOT_RULES)

# room_layouts: the pictures drawn first, asked once, cached by their bytes
import characters as CH
hp = b / "work" / "video" / "heroes" / "baker_street_sitting_room.png"
hp.parent.mkdir(parents=True, exist_ok=True)
hp.write_bytes(b"png1")
S.scene_list = lambda *a, **k: [{"place": "Baker Street Sitting Room (inside)"}, {"place": "Nowhere Described"}]
drawn, vl = [], []
MF.heroes_for = lambda book, places, log=print: (drawn.append(dict(places)), {w: hp for w in places})[1]
CH.vl_ask = lambda items: (vl.append(items), {x["id"]: {"layout": "  The bed on the left.\n"} for x in items})[1]
CH.vl_close = lambda: None
got = S.room_layouts(b, stem, log=lambda m: None)
check("only described places get a picture first", drawn and list(drawn[0]) == ["Baker Street Sitting Room (inside)"], drawn)
check("asked for a full answer (not the yes/no 8 tokens)", vl and vl[0][0].get("max_new_tokens") == 120, vl)
check("its layout is the vision model's words, tidied", got == {"Baker Street Sitting Room (inside)": "The bed on the left."}, got)
S.room_layouts(b, stem, log=lambda m: None)
check("asked once: cached beside the picture", len(vl) == 1, len(vl))
hp.write_bytes(b"png2")
S.room_layouts(b, stem, log=lambda m: None)
check("a redrawn picture is asked again", len(vl) == 2, len(vl))

import registry as RG
lp = b / "work" / "video" / "look.json"
L = json.loads(lp.read_text()); L["places"]["Baker Street Study"] = "Outside: Outside: Inside: A desk in the foreground left."
lp.write_text(json.dumps(L))
check("stacked side words read as the last one", RG.Book(b).place_look("Baker Street Study") ==
      "Inside: A desk in the foreground left.", RG.Book(b).place_look("Baker Street Study"))
(b / "book.txt").write_text("x")
r = REAL_RUN([sys.executable, "-m", "tts_reader.bookcaster", "look", str(b), "--do",
                    "Baker Street Study = Inside: a desk in the foreground, square to the viewer, its chair behind it"],
                   cwd=str(HERE.parent.parent), capture_output=True, text=True)
check("bookcaster look --do changes a place's one description", "place Baker Street Study now" in r.stdout and
      RG.Book(b).place_look("Baker Street Study").endswith("its chair behind it"), (r.stdout[-300:], r.stderr[-500:]))
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
