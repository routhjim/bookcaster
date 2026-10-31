"""10-29: someone in bed stays in bed in the talk and reaction shots (the two-shot template sat Watson on a chair
beside an empty bed; an over-the-shoulder drew his back standing). The real direct() on a synthetic book with a
scripted director (no Strata), as test_reaction_1029."""
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
b = Path(tempfile.mkdtemp(prefix="bed_e2e_"))
vd = b / "work" / "video" / stem
vd.mkdir(parents=True)
BED = "Baker Street Bedroom (inside)"
(b / "work" / "video" / "look.json").write_text(json.dumps({
    "characters": {"Sherlock Holmes": "a man of about 40, wearing a dark frock coat",
                   "Doctor Watson": "a man of about 35, wearing a white nightshirt"},
    "places": {BED: "a Victorian bedroom with an iron bed and a window"}}))
(b / "book.cast.json").write_text(json.dumps({"characters": {"Holmes": {}, "Watson": {}}}))
(vd / "plan.json").write_text(json.dumps({"look": {"characters": {"Sherlock Holmes": "", "Doctor Watson": ""}}}))
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "I sat up in bed. Holmes stood by me."},
      {"t": 3.0, "kind": "quote", "speaker": "Holmes", "text": "Very sorry to knock you up, Watson."},
      {"t": 6.0, "kind": "quote", "speaker": "Watson", "text": "What is it, then, a fire?"},
      {"t": 9.0, "kind": "quote", "speaker": "Holmes", "text": "No, a client. It seems that a young lady has arrived in a considerable state of excitement, who insists upon seeing me at once."}]
(b / "work" / f"{stem}.timeline.json").write_text(json.dumps(tl))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=20", str(b / "work" / f"{stem}.mp3")],
               check=True)
es = [{"id": "action-000000", "t": 0.0, "dur": 20.0, "route": "shot", "where": BED,
       "who": ["Sherlock Holmes", "Doctor Watson"]}]
BREAK = {"scenes": [{"start": 0.0, "place": BED, "who": ["Holmes", "Watson"], "what": "Holmes wakes Watson.",
                     "cutaways": [], "opens": "Watson in bed; Holmes stands.",
                     "blocking": "Holmes standing on the left, Watson in bed on the right."}]}
PLAN = {"state_end": "x", "shots": [
    {"start": 0.0, "kind": "quiet", "place": BED, "framing": "medium",
     "people": [{"name": "Holmes", "side": "left", "pose": "standing", "at": "by the bed", "facing": "Watson", "holds": ""},
                {"name": "Watson", "side": "right", "pose": "sitting", "at": "in bed", "facing": "Holmes", "holds": ""}],
     "setting": "Grey morning light through the window.", "action": "Watson sits up in bed.", "pose_end": {}},
    {"start": 1.5, "kind": "quiet", "place": BED, "framing": "close",
     "people": [{"name": "Holmes", "side": "left", "pose": "standing", "at": "beside the bed", "facing": "Watson", "holds": ""}],
     "setting": "The dim bedroom wall.", "action": "Holmes waits.", "pose_end": {}},
    {"start": 3.0, "kind": "talk", "speaker": "Holmes", "listener": "Watson", "framing": "two"},
    {"start": 6.0, "kind": "talk", "speaker": "Watson", "listener": "Holmes", "framing": "close"},
    {"start": 9.0, "kind": "talk", "speaker": "Holmes", "listener": "Watson", "framing": "over-the-shoulder"}]}
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
two = [s for s in talk if len(s["who"]) == 2]
check("the two-shot keeps Watson in bed, Holmes at the bedside", two and all(
    "Doctor Watson sitting up in bed under the covers" in s["picture"] and "Sherlock Holmes standing at the bedside"
    in s["picture"] and "arm's length" not in s["picture"] for s in two), [s["picture"] for s in talk])
check("no over-the-shoulder shot past the man in bed", not [s for s in talk if s.get("ots") == "Doctor Watson"],
      [(s["who"], s.get("ots")) for s in talk])
cw = [s for s in talk if s["who"] == ["Doctor Watson"]]
check("Watson's close-up is in bed, against the pillows", cw and all("in bed, against the pillows" in s["picture"]
                                                                    for s in cw), [s["picture"] for s in cw])
rx = [s for s in shots if s["kind"] == "quiet" and s["who"] == ["Doctor Watson"]]
# (10-30 user "remove the cutaways": Holmes's long line no longer cuts to Watson mid-line; a reaction after a line still
# must be the close view in bed)
check("Holmes's long last line stays on Holmes, one talk shot", len([s for s in talk if s["id"].startswith("talk-000090")])
      == 1, [s["id"] for s in talk])
check("Watson's reaction (if any) is a close view in bed", not rx or all(s["picture"].startswith("a close view of Doctor Watson's")
                                                             and "in bed" in s["picture"] for s in rx),
      [s["picture"] for s in rx])
hc = [s for s in shots if s["kind"] == "quiet" and s["who"] == ["Sherlock Holmes"]]
check("Holmes's close shot, Watson in the scene: a true close-up (no room picture, so no empty bed)",
      hc and all(S.closeup(s) and s["picture"].startswith("a close view of Sherlock Holmes's head") for s in hc),
      [(s.get("tight"), s["picture"]) for s in hc])
check("a close shot of someone alone in a scene keeps the room", not S.closeup(
    {"kind": "quiet", "framing": "close", "who": ["Doctor Watson"]}))
check("a close key of a standing one leaves the seat he stands by out", S.SEAT_NEAR.sub("",
      "Sherlock Holmes stands beside the bed, facing right.") == "Sherlock Holmes stands, facing right.")
check("...and keeps what isn't a seat", S.SEAT_NEAR.sub("", "Holmes stands near the desk.") == "Holmes stands near the desk.")
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
