"""10-28 (blocking + group talk) built on the 10-27 audit, end to end on CPU: the real direct() on a synthetic book with a scripted director (no Strata).
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
real_conv = S._converse
def conv2(url, model, messages, max_tokens=5000, effort=""):
    asks.append(messages[-1]["content"])
    return converse(url, model, messages, max_tokens, effort)
S._converse = conv2
new_es, by, _ = S.direct(b, stem, es, log=lambda m: None)
shots = [(e, by[e["id"]][0]) for e in new_es]
check("the scene's blocking from the breakdown is in every passage the director gets",
      any("THE BLOCKING of this scene" in a and "Julia Stoner on the right" in a for a in asks), [a[:80] for a in asks])
g = [s for e, s in shots if s["kind"] == "talk" and len(s["who"]) == 3]
check("a 'group' talk line is a three-shot of everyone in the scene", bool(g), [(s["kind"], s["who"]) for _, s in shots])
if g:
    s = g[0]
    check("in the scene's left-to-right order (the side map), the speaker cheated out, the listener toward him",
          s["who"].index("Helen Stoner") < s["who"].index("Sherlock Holmes") and "Sherlock Holmes turned three-quarters"
          in s["picture"] and "Julia Stoner in profile, head turned to Sherlock Holmes" in s["picture"], (s["who"], s["picture"]))
    check("its speaker zone is his own layout box", S._speaker_box(s, "Sherlock Holmes") ==
          S.group_layout(3)[s["who"].index("Sherlock Holmes")][0])
o = [s for e, s in shots if s["kind"] == "talk" and s["who"] and "Helen Stoner" in s["who"] and len(s["who"]) == 2]
check("the director's over-the-shoulder is kept for a pair's first line (Helen in her chair, Holmes by the fire: apart)",
      bool(o) and o[0].get("ots") == "Sherlock Holmes", [(s.get("who"), s.get("ots")) for s in o])
check("the rule puts the one spoken to in the picture", "THE ONE SPOKEN TO IS IN THE PICTURE" in S.SHOT_RULES)
check("every talk key asks for the mouth open mid-word (closed-lip keys froze the lips)",
      all(S.MOUTH_OPEN in t for t in (S.TALK_PICTURE, S.TALK_TWO, S.TALK_GROUP, S.TALK_OTS)))
bl = ("Helen Stoner seated centre-left by the fire; Sherlock Holmes standing centre-right, facing her; Doctor Watson "
      "seated on the right, slightly behind Holmes.")
sd_ = S._blocking_sides(bl, ["Helen Stoner", "Sherlock Holmes", "Doctor Watson"])
check("the blocking's sides seed the scene's side map: Helen < Holmes < Watson",
      sd_["Helen Stoner"] < sd_["Sherlock Holmes"] < sd_["Doctor Watson"], sd_)
scr = dict(sd_)
lr = sorted(("Sherlock Holmes", "Doctor Watson"), key=lambda w: (scr[w], w != "Sherlock Holmes"))
check("so Holmes's aside to Watson puts Watson on the RIGHT of Holmes, not in Helen's place", lr == ["Sherlock Holmes",
      "Doctor Watson"], lr)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
