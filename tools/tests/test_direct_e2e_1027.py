"""10-27 audit, end to end on CPU: the real direct() on a synthetic book with a scripted director (no Strata).
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
(vd / "plan.json").write_text(json.dumps({"look": {"characters": {"Sherlock Holmes": "", "Helen Stoner": ""}}}))
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "The lady sat in the chair by the window. Holmes stood by the fire."},
      {"t": 6.0, "kind": "quote", "speaker": "Holmes", "text": "Pray be precise as to details, madam."},
      {"t": 9.0, "kind": "narration", "speaker": "narrator", "text": "She clasped her hands and looked at him."},
      {"t": 14.0, "kind": "quote", "speaker": "Stoner", "text": "It is easy for me to be so."}]
(b / "work" / f"{stem}.timeline.json").write_text(json.dumps(tl))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=20", str(b / "work" / f"{stem}.mp3")],
               check=True)
es = [{"id": "action-000000", "t": 0.0, "dur": 20.0, "route": "shot", "where": "Baker Street Sitting Room (inside)",
       "who": ["Sherlock Holmes", "Helen Stoner"]}]

BREAK = {"scenes": [{"start": 0.0, "place": "Baker Street Sitting Room (inside)", "who": ["Holmes", "Helen Stoner"],
                     "what": "Helen tells Holmes her story.", "cutaways": [], "opens": "Helen sits; Holmes stands."}]}
def shot_q(start, setting):
    return {"start": start, "kind": "quiet", "place": "Baker Street Sitting Room (inside)", "framing": "medium",
            "people": [{"name": "Helen Stoner", "side": "left", "pose": "sitting", "at": "in the chair by the window",
                        "facing": "Holmes", "holds": ""},
                       {"name": "Holmes", "side": "right", "pose": "standing", "at": "by the fire", "facing": "Helen",
                        "holds": ""}],
            "setting": setting, "action": "Helen Stoner clasps her hands.", "pose_end": {}}
BAD = {"state_end": "x", "shots": [shot_q(0.0, "Morning light. Holmes stands by the fire."),
                                   shot_q(3.0, "Morning light."),
                                   {"start": 6.0, "kind": "talk", "speaker": "Holmes", "listener": "Helen", "framing": "two"},
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

new_es, by, _ = S.direct(b, stem, es, log=lambda m: None)
shots = [(e, by[e["id"]][0]) for e in new_es]
talks = [(e, s) for e, s in shots if s["kind"] == "talk"]
quiet = [s for e, s in shots if s["kind"] != "talk"]
check("the scripted director was asked again after a setting that named Holmes",
      len([c for c in calls if not c.startswith("You are a film")]) == 2 and calls[-1].startswith("Your last answer"), calls)
check("voice labels -> cast names: Holmes and Helen Stoner speak", [e["speaker"] for e, _ in talks] ==
      ["Sherlock Holmes", "Helen Stoner"], [e["speaker"] for e, _ in talks])
check("one side map: the talk two-shot has Helen left and Holmes right, as the quiet shots do",
      quiet[0]["who"] == ["Helen Stoner", "Sherlock Holmes"] and talks[0][1]["who"] == ["Helen Stoner", "Sherlock Holmes"],
      (quiet[0]["who"], talks[0][1]["who"]))
check("the talk shot's picture carries the scene's postures (Helen sits, Holmes stands)",
      "Helen Stoner sits in the chair by the window" in talks[0][1]["picture"] and
      "Sherlock Holmes stands by the fire" in talks[0][1]["picture"], talks[0][1]["picture"])
check("talk shots hold their postures for the key check", talks[0][1].get("pose_held") ==
      {"Helen Stoner": "sit", "Sherlock Holmes": "stand"}, talks[0][1].get("pose_held"))
check("the setting never places people (picture = people from fields + a people-free setting)",
      "Holmes stands by the fire." not in quiet[0]["picture"].split(". ", 1)[-1], quiet[0]["picture"])
check("names in the action become the one description", "Helen" not in quiet[0]["action"] and
      "the woman in the black mourning dress" in quiet[0]["action"], quiet[0]["action"])
dur = lambda f: float(REAL_RUN(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
                                     capture_output=True, text=True).stdout)
check("each talk shot's voice cut is as long as the shot", all(abs(dur(e["wav"]) - e["dur"]) < 0.1 for e, _ in talks),
      [(round(e["dur"], 2), round(dur(e["wav"]), 2)) for e, _ in talks])
check("no planner fields leak into a talk moment", all(set(e) <= {"id", "kind", "route", "t", "dur", "speaker", "line",
      "emotion", "creature", "speech_s", "frames", "chunks", "wav"} for e, _ in talks))
cache = [json.loads(f.read_text()) for f in sorted((vd / "shots" / "passages").glob("p*.json"))]
check("the passage cache holds the model's words with their key", cache and all(c.get("k") and c.get("reply") for c in cache))
# a second run: no director at all - the cached words go through today's code again
n0 = len(calls)
S._converse = lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked the director"))
new2, by2, _ = S.direct(b, stem, es, log=lambda m: None)
check("a second run replays the cache (no director call) to the same shots",
      [by2[e["id"]][0]["picture"] for e in new2] == [s["picture"] for _, s in shots])
# 10-28: a cached answer that breaks a rule it didn't break when it was cached is asked again (only that passage)
for f in sorted((vd / "shots" / "passages").glob("p*.json")):
    c = json.loads(f.read_text())
    if "SCENES" not in str(c.get("reply"))[:40] and c.get("reply") and '"shots"' in c["reply"]:
        c["reply"], c["faults"] = json.dumps(BAD), []
        f.write_text(json.dumps(c))
        break
asked3 = []
S._converse = lambda url, model, messages, **k: (asked3.append(1), (json.dumps(GOOD), 0))[1]
S.direct(b, stem, es, log=lambda m: None)
check("a cached answer breaking a newer rule is asked again", len(asked3) == 1, asked3)
asked3.clear()
S.direct(b, stem, es, log=lambda m: None)
check("and once answered clean, it replays again", asked3 == [], asked3)
scr = {}
S.compose({"people": [{"name": "Helen Stoner", "side": "right"}], "setting": ""}, ["Helen Stoner", "Sherlock Holmes"], [], scr)
S.compose({"people": [{"name": "Sherlock Holmes", "side": "right"}], "setting": ""}, ["Helen Stoner", "Sherlock Holmes"], [], scr)
two = S.compose({"people": [{"name": "Helen Stoner", "side": "left"}, {"name": "Sherlock Holmes", "side": "right"}],
                 "setting": ""}, ["Helen Stoner", "Sherlock Holmes"], [], scr)
check("a side tie (both first placed 'right') is settled by the order drawn, and kept: Helen then Holmes",
      two["who"] == ["Helen Stoner", "Sherlock Holmes"] and scr["Helen Stoner"] < scr["Sherlock Holmes"], (two["who"], scr))
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
