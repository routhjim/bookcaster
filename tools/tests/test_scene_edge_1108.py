"""E26 (11-08), on CPU: a scene's start is the breakdown's guess, and the passage cut there may open on the scene
before's place (Long Dark: scene 3 'Kestrel Cockpit' at 170.0 s, the patrol at 174.8 s; the director kept Sena taking
Daven's hand at Family Quarters, the rule check called it outside the scene, and the cut moved it to the cockpit with
its bunk-side people). The real direct() with a scripted director: the boundary shot stays at its place with its
people, is not sent back, and the people there are not counted as present in the new scene; a shot at that place
between the new scene's own shots is still sent back and moved."""
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


Q, C = "Family Quarters", "Kestrel Cockpit"
# boundary_runs: the opening / closing runs only, talk shots end a run
sh = [{"kind": "quiet", "place": Q}, {"kind": "action", "place": "the family quarters"}, {"kind": "quiet", "place": C},
      {"kind": "quiet", "place": Q}]
check("boundary_runs: the opening run at the scene before's place", S.boundary_runs(sh, Q, "") == {0: Q, 1: Q},
      S.boundary_runs(sh, Q, ""))
check("boundary_runs: the closing run at the next scene's place", S.boundary_runs(sh, "", Q) == {3: Q},
      S.boundary_runs(sh, "", Q))
check("boundary_runs: a talk shot ends the run; no place, no run",
      S.boundary_runs([{"kind": "talk", "place": Q}, sh[0]], Q, "") == {} and S.boundary_runs(sh, "", "") == {})

stem = "01_x"
b = Path(tempfile.mkdtemp(prefix="scene_edge_"))
vd = b / "work" / "video" / stem
vd.mkdir(parents=True)
(b / "work" / "video" / "look.json").write_text(json.dumps({
    "characters": {"Daven Iru": "a man of about 35, wearing a worn grey flight suit",
                   "Sena": "a woman of about 32, wearing a simple grey tunic"},
    "places": {Q: "a cramped room with a narrow bunk and a curtained alcove",
               C: "a tight fighter cockpit, a glowing instrument panel under a wide canopy"}}))
(b / "book.cast.json").write_text(json.dumps({"characters": {"Daven Iru": {}, "Sena": {}}}))
(vd / "plan.json").write_text(json.dumps({"look": {"characters": {"Daven Iru": "", "Sena": ""}}}))
tl = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "Daven sat on the bunk. Sena watched him from the bunk."},
      {"t": 5.0, "kind": "narration", "speaker": "narrator", "text": "Sena did not answer. She took his hand and held it."},
      {"t": 10.0, "kind": "narration", "speaker": "narrator", "text": "The next morning's patrol took him far out. He flew."},
      {"t": 15.0, "kind": "narration", "speaker": "narrator", "text": "He expected nothing. It kept the disappointments small."}]
(b / "work" / f"{stem}.timeline.json").write_text(json.dumps(tl))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=21", str(b / "work" / f"{stem}.mp3")],
               check=True)
es = [{"id": "action-000000", "t": 0.0, "dur": 20.0, "route": "shot", "where": Q, "who": ["Daven Iru", "Sena"]}]
# the breakdown puts the cockpit's start at the quarters' last paragraph (5.0 s; the patrol is at 10.0 s)
BREAK = {"scenes": [{"start": 0.0, "place": Q, "who": ["Daven Iru", "Sena"], "what": "Daven and Sena talk.",
                     "cutaways": [], "opens": "Daven sits on the bunk."},
                    {"start": 5.0, "place": C, "who": ["Daven Iru"], "what": "Daven's patrol.", "cutaways": [],
                     "opens": "Daven in the pilot's seat."}]}


def person(name, side, pose, at, facing=""):
    return {"name": name, "side": side, "pose": pose, "at": at, "facing": facing, "holds": ""}


def shot(start, place, people, action):
    return {"start": start, "kind": "quiet", "place": place, "framing": "medium", "people": people,
            "setting": "Dim light.", "action": action, "pose_end": {}}


HAND = shot(0.0, Q, [person("Daven Iru", "left", "sitting", "on the bunk edge", "Sena"),
                     person("Sena", "right", "sitting", "on the bunk", "Daven Iru")], "Sena takes Daven Iru's hand.")
SEAT = person("Daven Iru", "centre", "sitting", "in the pilot's seat", "the canopy")
FIRST = {"state_end": "x", "shots": [shot(0.0, Q, [person("Daven Iru", "left", "sitting", "on the bunk edge"),
                                                   person("Sena", "right", "sitting", "on the bunk")],
                                          "Daven Iru looks down."),
                                     shot(2.5, Q, [person("Sena", "centre", "sitting", "on the bunk")],
                                          "Sena watches him.")]}
EDGE = {"state_end": "x", "shots": [HAND, shot(5.0, C, [SEAT], "Daven Iru works the controls."),
                                    shot(8.0, C, [SEAT], "Daven Iru looks out."),
                                    shot(11.0, C, [SEAT], "Daven Iru leans back.")]}
MID = {"state_end": "x", "shots": [shot(0.0, C, [SEAT], "Daven Iru works the controls."), dict(HAND, start=4.0),
                                   shot(8.0, C, [SEAT], "Daven Iru looks out."),
                                   shot(11.0, C, [SEAT], "Daven Iru leans back.")]}


def run(script):
    asks = []

    def converse(url, model, messages, max_tokens=5000, effort=""):
        m = messages[-1]["content"]
        if "SCENES" in m:
            return json.dumps(BREAK), 0
        asks.append(m)
        first = next(x["content"] for x in messages if x["role"] == "user" and x["content"].startswith("THE SCENE"))
        return json.dumps(FIRST if "scene 1 of" in first else script), 0
    S._converse = converse
    for f in (vd / "shots" / "passages").glob("p*.json"):
        f.unlink()
    new_es, by, _ = S.direct(b, stem, es, log=lambda m: None)
    return [by[e["id"]][0] for e in new_es], asks


animate.igpu = lambda *a, **k: None
S.subprocess.run = (lambda real: (lambda cmd, *a, **k: real(cmd, *a, **k) if cmd and cmd[0] == "ffmpeg"
                                  else type("R", (), {"returncode": 1, "stdout": ""})()))(subprocess.run)
S._cast = lambda book, stem_, e: (e.get("who") or [], [])
MF._dress = lambda *a, **k: {}
S.CHAINS, S.OPENING, S.TOLD_PEOPLE = False, False, False

shots, asks = run(EDGE)
hand = next(s for s in shots if "hand" in s.get("action", ""))
check("the boundary shot stays at the scene before's place", hand["place"] == Q, hand["place"])
check("... with its own people (Daven and Sena at the bunk)", hand["who"] == ["Daven Iru", "Sena"]
      and "on the bunk" in hand["picture"], (hand["who"], hand["picture"]))
own = [s for s in shots if "Sena" not in s["who"] and "hand" not in s.get("action", "")][2:]
check("the new scene's own shots are at its place", own and all(s["place"] == C for s in own),
      [(s["place"], s.get("action")) for s in shots])
check("the boundary passage is not sent back for its place", not any("this scene is at" in a for a in asks),
      [a[:120] for a in asks if a.startswith("Your last")])
check("the shot after it opens Daven in the pilot's seat, not on the bunk",
      next(s for s in shots if "controls" in s.get("action", ""))["picture"].startswith("Daven Iru sits in the pilot's"),
      next(s for s in shots if "controls" in s.get("action", ""))["picture"])

shots, asks = run(MID)
hand = next(s for s in shots if "hand" in s.get("action", ""))
check("a shot at the scene before's place BETWEEN the new scene's shots is still sent back",
      any("this scene is at" in a for a in asks), [a[:120] for a in asks])
check("... and, kept by the director, is moved into the scene as before (not on the boundary)", hand["place"] == C,
      hand["place"])
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
