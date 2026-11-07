"""E22 (11-07): a talk shot with others in it is led by the speaker - as the scene holds them (pose, spot, held thing),
nearest the viewer, the face large and turned three-quarters to the viewer; the others beyond, smaller, each as the
scene holds them, looking at the speaker. Talk shots only: every other shot keeps group_layout and its words. The real
direct() on a synthetic book with a scripted director (no Strata), as test_bed_1029."""
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

# --- the layout: mechanics only ----------------------------------------------------------------------------------
for n in (2, 3, 4, 5):
    who = [f"P{k}" for k in range(n)]
    check(f"{n} people, not a led talk shot: group_layout as before",
          S.talk_layout({"kind": "quiet", "who": who, "speaker": who[0]}) == S.group_layout(n) and
          S.talk_layout({"kind": "talk", "who": who}) == S.group_layout(n) and
          S.talk_layout({"kind": "talk", "who": who, "speaker": who[0], "ots": who[1]}) == S.group_layout(n))
    for i in range(n):
        lay = S.talk_layout({"kind": "talk", "who": who, "speaker": who[i]})
        bx = [b for b, _ in lay]
        sp = bx[i]
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        check(f"{n} people, speaker {i}: the speaker's box the largest, full height, on their own side, words say so",
              all(area(sp) > area(b) for k, b in enumerate(bx) if k != i) and sp[1] == 0.0 and sp[3] == 1.0 and
              "nearest the viewer" in lay[i][1] and all("further back" in lb for k, (_, lb) in enumerate(lay) if k != i)
              and (i != 0 or sp[0] == 0.0) and (i != n - 1 or sp[2] == 1.0), lay)
        check(f"{n} people, speaker {i}: inside the picture, left to right, at most a sliver over the speaker's box",
              all(0.0 <= b[0] < b[2] <= 1.0 for b in bx) and all(bx[k][0] <= bx[k + 1][0] for k in range(n - 1)) and
              all(min(b[2], sp[2]) - max(b[0], sp[0]) <= 0.05 + 1e-9 for k, b in enumerate(bx) if k != i), bx)
s2 = {"kind": "talk", "who": ["A", "B"], "speaker": "B", "listener": "A"}
check("SyncNet's zone is the speaker's own (large) box", S._speaker_box(s2, "B") == S.talk_layout(s2)[1][0])

# --- direct(): a bunk two-shot and a group, end to end on CPU -------------------------------------------------------
stem = "01_x"
b = Path(tempfile.mkdtemp(prefix="talk_lead_"))
vd = b / "work" / "video" / stem
vd.mkdir(parents=True)
Q = "Ship Quarters (inside)"
(b / "work" / "video" / "look.json").write_text(json.dumps({
    "characters": {"Daven Iru": "a man of about 35, wearing a grey flight suit",
                   "Lio": "a girl of about 8, wearing a grey sleep shirt",
                   "Sena": "a woman of about 33, wearing a grey sleep shirt"},
    "places": {Q: "a cramped steel room with a wide bunk"}}))
(b / "book.cast.json").write_text(json.dumps({"characters": {"Daven": {}, "Lio": {}, "Sena": {}}}))
(vd / "plan.json").write_text(json.dumps({"look": {"characters": {"Daven Iru": "", "Lio": "", "Sena": ""}}}))
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "They lay on the bunk in the dark."},
      {"t": 3.0, "kind": "quote", "speaker": "Lio", "text": "Are we going to be all right?"},
      {"t": 6.0, "kind": "quote", "speaker": "Daven", "text": "We are. Go to sleep."},
      {"t": 9.0, "kind": "narration", "speaker": "narrator", "text": "She put the toy down and sat up."},
      {"t": 12.0, "kind": "quote", "speaker": "Lio", "text": "Promise me, all of you."}]
(b / "work" / f"{stem}.timeline.json").write_text(json.dumps(tl))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=16", str(b / "work" / f"{stem}.mp3")],
               check=True)
es = [{"id": "action-000000", "t": 0.0, "dur": 16.0, "route": "shot", "where": Q, "who": ["Daven Iru", "Lio", "Sena"]}]
BREAK = {"scenes": [{"start": 0.0, "place": Q, "who": ["Daven", "Lio", "Sena"], "what": "Night talk on the bunk.",
                     "cutaways": [], "opens": "All three lie on the bunk.",
                     "blocking": "Daven on the left, Lio in the centre, Sena on the right, all lying on the bunk."}]}
PPL = [{"name": "Daven", "side": "left", "pose": "lying", "at": "on the bunk", "facing": "up", "holds": ""},
       {"name": "Lio", "side": "centre", "pose": "lying", "at": "on the bunk", "facing": "Daven", "holds": "a toy fox"},
       {"name": "Sena", "side": "right", "pose": "lying", "at": "on the bunk", "facing": "away", "holds": ""}]
PLAN = {"state_end": "x", "shots": [
    {"start": 0.0, "kind": "quiet", "place": Q, "framing": "wide", "people": PPL,
     "setting": "Dim light.", "action": "They breathe slowly.", "pose_end": {}},
    {"start": 3.0, "kind": "talk", "speaker": "Lio", "listener": "Daven", "framing": "two"},
    {"start": 6.0, "kind": "talk", "speaker": "Daven", "listener": "Lio", "framing": "two"},
    {"start": 9.0, "kind": "quiet", "place": Q, "framing": "wide",
     "people": [dict(p_, holds="") if p_["name"] == "Lio" else p_ for p_ in PPL],
     "setting": "Dim light.", "action": "Lio sets the toy fox down beside her.", "pose_end": {}},
    {"start": 12.0, "kind": "talk", "speaker": "Lio", "listener": "Sena", "framing": "group"}]}
def converse(url, model, messages, max_tokens=5000, effort=""):
    return json.dumps(BREAK if "SCENES" in messages[-1]["content"] else PLAN), 0
S._converse = converse
animate.igpu = lambda *a, **k: None
S.subprocess.run = (lambda real: (lambda cmd, *a, **k: real(cmd, *a, **k) if cmd and cmd[0] == "ffmpeg"
                                  else type("R", (), {"returncode": 1, "stdout": ""})()))(subprocess.run)
S._cast = lambda book, stem_, e: (e.get("who") or [], [])
MF._dress = lambda *a, **k: {}

new_es, by, _ = S.direct(b, stem, es, log=lambda m: None)
shots = [s for e in new_es for s in by[e["id"]]]
talk = [s for s in shots if s["kind"] == "talk"]
pair = [s for s in talk if len(s["who"]) == 2]
lio = [s for s in pair if s.get("speaker") == "Lio"]
dav = [s for s in pair if s.get("speaker") == "Daven Iru"]
check("Lio's line leads with her lying on the bunk, holding the toy, nearest the viewer", lio and lio[0]["picture"].startswith(
    "Lio lies on the bunk, holding a toy fox, ") and "nearest the viewer: seen from the chest up" in lio[0]["picture"],
      [s["picture"] for s in pair])
check("...Daven beyond her, lying on the bunk too, looking at her", lio and "beyond Lio and smaller in the picture: "
      "Daven Iru lies on the bunk, on the left, further back, head turned to look at Lio" in lio[0]["picture"],
      [s["picture"] for s in lio])
check("Daven's reply leads with him lying; Lio beyond him still holding the toy", dav and dav[0]["picture"].startswith(
    "Daven Iru lies on the bunk, on the left, nearest the viewer") and "Lio lies on the bunk, holding a toy fox, on the "
      "right, further back" in dav[0]["picture"], [s["picture"] for s in dav])
check("no side-by-side words, no sitting up", all("arm's length" not in s["picture"] and "sits" not in s["picture"]
                                                  for s in talk), [s["picture"] for s in talk])
check("the poses are judged as held (lying)", all(set(s["pose_held"].values()) == {"lie"} for s in pair),
      [s["pose_held"] for s in pair])
grp = [s for s in talk if len(s["who"]) == 3]
check("the group line: the scene's order, Lio leads, the toy set down is no longer held", grp and
      grp[0]["who"] == ["Daven Iru", "Lio", "Sena"] and grp[0]["picture"].startswith("Lio lies on the bunk, in the "
      "centre, nearest the viewer") and "toy" not in grp[0]["picture"], [(s["who"], s["picture"]) for s in grp])
quiet = [s for s in shots if s["kind"] != "talk"]
check("quiet shots carry no speaker and keep group_layout", quiet and all(
    "speaker" not in s and S.talk_layout(s) == S.group_layout(len(s["who"])) for s in quiet))
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
