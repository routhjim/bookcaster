"""10-27 user: "the whole point of the director is to break up the story into individual shots... it needs more
context". The director now gets whole passages (timed sentences with their speakers) and makes every shot, talk shots
included; its shots become the moments. CPU only.
    ~/tts_reader/.venv/bin/python3 tools/tests/test_director_passages_1027.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as S
tl = json.loads((Path(__file__).parent / "fixtures" / "fire_1026" / "timeline.json").read_text())
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

ss = S.sentences(tl)
check("sentences: every sentence of the chapter, in order, with times", len(ss) > 10 and
      all(a[0] <= b[0] for a, b in zip(ss, ss[1:])) and all(x[1] > x[0] for x in ss), ss[:3])
fire = [x for x in ss if "tongues as of fire" in x[4]]
check("sentences: the fire sentence is timed well after 0:00", fire and fire[0][0] > 8, fire)
end = float(tl[-1]["t"]) + 5
ps = S.passages(tl, 0.0, end)
check("passages cover the stretch without gaps", abs(ps[0][0]) < 1e-6 and abs(ps[-1][1] - end) < 1e-6 and
      all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(ps, ps[1:])), ps)
check("passages are at most PASSAGE_MAX", all(b - a <= S.PASSAGE_MAX + 1e-6 for a, b in ps), ps)
check("a 67 s stretch becomes several passages (10-27: one 67 s ask came back empty)",
      len(S.passages(tl, 0.0, 67.0)) >= 3, S.passages(tl, 0.0, 67.0))
check("who: a timeline speaker as the cast name", S._who_full("Holmes", ["Dr. Watson", "Sherlock Holmes"]) ==
      "Sherlock Holmes" and S._who_full("narrator", ["Dr. Watson"]) == "")

cast = ["Sherlock Holmes", "Dr. Watson"]
q = [(4.0, 7.0, "Sherlock Holmes")]
def sh(start, kind="quiet", **k):
    d = {"start": start, "kind": kind, "who": ["Dr. Watson"], "framing": "medium", "picture": "A man lies in bed.",
         "action": "The man in tweed pushes up on one elbow.", "pose": {"Dr. Watson": "lying"},
         "pose_end": {"Dr. Watson": "sitting"}}
    d.update(k)
    return d
good = {"shots": [sh(0.0), sh(4.0, "talk", speaker="Sherlock Holmes", listener="Dr. Watson", framing="two"), sh(7.0)]}
check("a good passage answer: no faults", S._shot_faults(good, cast, 10.0, q) == [], S._shot_faults(good, cast, 10.0, q))
bad_talk = {"shots": [sh(0.0, "talk", speaker="Sherlock Holmes"), sh(4.0)]}
check("a talk shot when its speaker isn't speaking: faulted",
      any("not speaking" in f for f in S._shot_faults(bad_talk, cast, 8.0, q)))
long = {"shots": [sh(0.0), sh(1.5)]}
check("a 8.5 s quiet shot: faulted", any("lasts" in f for f in S._shot_faults(long, cast, 10.0, q)))
order = {"shots": [sh(0.0), sh(3.0), sh(2.0)]}
check("starts out of order: faulted", any("starts" in f for f in S._shot_faults(order, cast, 8.0, q)))
title = {"shots": [sh(0.0, who=[], picture="A title card reads 'The Speckled Band' in gold lettering.", pose={},
                      pose_end={})]}
fs = S._shot_faults(title, cast, 3.0, [])
check("a title card: faulted for writing, not for having nobody", any("writing" in f for f in fs) and
      not any("who must be" in f for f in fs), fs)
clock = {"shots": [sh(0.0, who=[], picture="A brass clock on the mantelpiece shows a quarter past seven.", pose={},
                      pose_end={}, action="The clock's minute hand ticks forward.")]}
check("a thing alone (the clock): no fault", S._shot_faults(clock, cast, 3.0, []) == [], S._shot_faults(clock, cast, 3.0, []))
check("the prompt says the director makes the shots and gets talk lines",
      "YOU break each passage into the film's SHOTS" in S.SHOT_RULES and "SAYS ALOUD" in S.PASSAGE_ASK and "state_end" in S.SHOT_RULES and "{" not in S.SHOT_RULES.format(maxp=4, closemax=3).split("JSON only")[0])
check("forge/fight/fire rules only when the passage has them",
      "anvil" not in S.SHOT_RULES and any("anvil" in r for _, r in S.RULE_EXTRA))
stand = {"shots": [sh(0.0, picture="A man stands by the desk.", pose={"Dr. Watson": "sitting"},
                      pose_end={"Dr. Watson": "sitting"}, action="The man in tweed leans back.")]}
check("a picture standing someone whose pose is sitting: faulted (10-27 Holmes in his chair)",
      any("pose" in f and "standing" in f for f in S._shot_faults(stand, cast, 3.0, [])), S._shot_faults(stand, cast, 3.0, []))
cam = {"shots": [sh(0.0, picture="A man lies in bed.", action="The camera pushes in on the man in tweed.")]}
check("a camera move in the action: faulted", any("camera" in f for f in S._shot_faults(cam, cast, 3.0, [])))
check("a title card is no place; a room is", bool(S.NO_PLACE.search("Title Card")) and
      not S.NO_PLACE.search("Baker Street Sitting Room (inside)"))
check("Helen's 'made' in a spoken line is no forge; 'forged the blade' is",
      not S.RULE_EXTRA[3][0].search("it made me shudder") and bool(S.RULE_EXTRA[3][0].search("he forged the blade")))
check("every spoken line is a talk shot in the rules", "Every SPOKEN line" in S.SHOT_RULES and
      "Nobody\n  speaks in any other shot" in S.SHOT_RULES)
rep = {"shots": [sh(0.0, picture="Dr. Watson sits at the desk, looking down at the stack of old notes in front of him.")]}
desk = "Dr. Watson sits at the desk, looking down at the stack of old notes in front of him"
check("a picture of an earlier scene brought back (desk, then the bedroom scene, then the desk): faulted",
      any("earlier scene" in f for f in S._repeats(rep, [desk], ["Watson lies in bed.", "Watson sits up in bed."])))
check("cutting back to this scene's master shot: no fault",
      S._repeats(rep, ["Holmes stands by the window."], [desk, "Holmes nods.", "Helen trembles."]) == [])
check("a new picture: no repeat fault", S._repeats(rep, ["Holmes stands by the window."], []) == [])
ps = S.passages(tl, 0.0, 200.0, forced=(67.0, 130.0))
check("passages cut at every scene start (no passage spans two scenes)",
      all(not (a < f < b) for a, b in ps for f in (67.0, 130.0)) and any(abs(a - 67.0) < 1e-6 for a, _ in ps), ps)
check("the breakdown: a good scene list has no faults",
      S._breakdown_faults([{"start": 0, "place": "Baker Street Sitting Room (inside)"},
                           {"start": 67, "place": "Baker Street Bedroom (inside)"}], 600.0) == [])
check("the breakdown: a title card is no place",
      any("no place" in f for f in S._breakdown_faults([{"start": 0, "place": "Title Card"}], 600.0)))
check("the breakdown: starts out of order are faulted",
      any("starts" in f for f in S._breakdown_faults([{"start": 0, "place": "A"}, {"start": 90, "place": "B"},
                                                      {"start": 60, "place": "C"}], 600.0)))
check("the passage turn names its scene; the peek ahead is context only",
      "{scene}" in S.PASSAGE_ASK and "none of it is in this passage's shots" in S.PASSAGE_ASK and "now_at" not in S.SHOT_RULES)
from types import SimpleNamespace as _NS
_R = _NS(gender_of=lambda look: "woman" if "woman" in look else "man")
_cx = _NS(look=lambda w, t: {"Helen Stoner": "a woman of about 30, pale, wearing a black mourning dress",
                             "Sherlock Holmes": "a man of about 40, wearing a dark frock coat"}.get(w, ""))
_tp = S._talk_prompt(_R, _cx, {"speaker": "Stoner", "t": 300.0}, "Baker Street Sitting Room (inside)",
                     {"who": ["Sherlock Holmes", "Helen Stoner"], "listener": "Holmes"})
check("talk prompt: the timeline's 'Stoner' speaks as Helen Stoner, on her side, with her look, Holmes listening "
      "(10-27: his lips moved when she talked)", "Helen Stoner on the right, a woman of about 30" in _tp.split(" speaks")[0]
      and "Sherlock Holmes listens with a closed mouth" in _tp and "Helen Stoner listens" not in _tp and " her voice" in _tp, _tp)
pic = ("Sherlock Holmes stands at the wooden writing desk on the right, facing left; Dr. Watson stands in the center "
       "near the armchair, facing right; Helen Stoner stands on the left near the window, facing right")
check("who follows the picture's left-to-right (10-27: Helen opened Holmes's drawer)",
      S.picture_order(["Sherlock Holmes", "Dr. Watson", "Helen Stoner"], pic) == ["Helen Stoner", "Dr. Watson", "Sherlock Holmes"],
      S.picture_order(["Sherlock Holmes", "Dr. Watson", "Helen Stoner"], pic))
check("a picture that places nobody keeps who as given",
      S.picture_order(["Sherlock Holmes", "Helen Stoner"], "Holmes and Helen by the fire.") == ["Sherlock Holmes", "Helen Stoner"])
check("'Watson sits on the left while Holmes stands on the right' keeps that order",
      S.picture_order(["Sherlock Holmes", "Dr. Watson"], "Dr. Watson sits at the desk on the left, while Sherlock Holmes "
                      "stands on the right") == ["Dr. Watson", "Sherlock Holmes"])
_c = S.compose({"kind": "quiet", "people": [
        {"name": "Holmes", "side": "right", "pose": "standing", "at": "at the writing desk", "facing": "Helen", "holds": ""},
        {"name": "Helen Stoner", "side": "left", "pose": "sitting", "at": "in a chair by the window", "facing": "toward Holmes",
         "holds": "nothing"}], "setting": "Morning light through the window; the desk drawer is closed."},
    ["Sherlock Holmes", "Dr. Watson", "Helen Stoner"], [])
check("people -> who in left-to-right order, by construction", _c["who"] == ["Helen Stoner", "Sherlock Holmes"], _c["who"])
check("people -> one picture that says the same", _c["picture"].startswith(
      "Helen Stoner sits in a chair by the window on the left, facing toward Holmes; Sherlock Holmes stands at the writing "
      "desk on the right, facing Helen. Morning light"), _c["picture"])
check("people -> pose from the same words", _c["pose"] == {"Helen Stoner": "sitting", "Sherlock Holmes": "standing"}, _c["pose"])
check("an old-style answer is left as it is", S.compose({"who": ["A"], "picture": "x"}, ["A"], []) == {"who": ["A"], "picture": "x"})
check("the rules ask for people and setting, not free placement", '"people"' in S.SHOT_RULES and '"setting"' in S.SHOT_RULES
      and '"who": the names in the picture' not in S.SHOT_RULES)
import tempfile as _tf
with _tf.TemporaryDirectory() as _d:
    (Path(_d) / "work").mkdir()
    (Path(_d) / "work" / "x.timeline.json").write_text(json.dumps([
        {"t": 0, "kind": "narration", "speaker": "narrator", "text": "She spoke."},
        {"t": 2, "kind": "quote", "speaker": "Stoner", "text": "It is fear."},
        {"t": 4, "kind": "quote", "speaker": "Watson", "text": "Indeed."}]))
    _tl = S.load_timeline(Path(_d), "x", ["Dr. Watson", "Helen Stoner", "Sherlock Holmes"])
check("the timeline is read with cast names only (one name per person)",
      [x["speaker"] for x in _tl] == ["narrator", "Helen Stoner", "Dr. Watson"], [x["speaker"] for x in _tl])
check("the passage turn shows the scene summary as context only", "show only what" in S.direct.__code__.co_consts.__repr__()
      or "never what comes later" in Path(S.__file__).read_text())
print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
