"""The staging pass's rules (tools/assets/scenes/staging_pass.py) with a scripted stand-in for the language model.

No model is called: the stand-in answers each closed question from the passage the prompt carries, the way a model
reading only that passage would. What is tested is the mechanics around the answers: a change of worn state or a
start action is taken from the shot's OWN span; of the shots that restate a change the first action shot owns it;
an action belongs to every shot whose own plan line states it and to no other; presence changes only on a stated
exit or entrance; a person the passage has already in the place is reported with the set's own thing; an answer the
repeats do not agree on is "not stated".
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import staging_pass as P  # noqa: E402

A, B, C_ = "Alder", "Birch", "Cedar"
CAST = {A: {"key": "a", "id": "fig_a"}, B: {"key": "b", "base": "fig_b"}, C_: {"key": "c", "id": "fig_c"}}


def person(name, pose, at):
    return {"name": name, "pose": pose, "at": at, "facing": ""}


PLAN = {
    "quiet-1": {"kind": "quiet", "people": [person(B, "sitting", "in the settee by the window"),
                                             person(A, "standing", "near the desk")]},
    "action-2": {"kind": "action", "action": "the woman in green walks to the fire and sits.",
                 "people": [person(B, "standing", "by the window")], "pose_end": {B: "sitting"}},
    "action-3": {"kind": "action", "action": "the woman in green raises her veil.",
                 "people": [person(B, "sitting", "in the chair by the fire")]},
    "action-4": {"kind": "action", "action": "the woman in green holds the veil up and waits.",
                 "people": [person(B, "sitting", "in the chair by the fire")]},
    "talk-5": {"kind": "talk", "people": [], "speaker": A, "listener": B},
    "action-6": {"kind": "action", "action": "the woman in green looks down at her sleeve.",
                 "people": [person(B, "sitting", "in the chair by the fire"), person(A, "standing", "near the desk")]},
    "action-7": {"kind": "action", "action": "the woman in green looks down at her sleeve once more.",
                 "people": [person(B, "sitting", "in the chair by the fire"), person(A, "standing", "near the desk")]},
    "talk-8": {"kind": "talk", "people": [], "speaker": A, "listener": B},
    "talk-9": {"kind": "talk", "people": [], "speaker": A, "listener": B},
}
TIMES = {sid: 10.0 * i for i, sid in enumerate(PLAN)}
TIMELINE = [
    {"t": 1.0, "text": "A woman was already seated by the window when the two men came in."},
    {"t": 11.0, "text": "She crossed the room to the fire."},
    {"t": 21.0, "text": "She raises her veil as she speaks."},
    {"t": 31.0, "text": "Her veil raised, she waits for an answer."},
    {"t": 41.0, "speaker": A, "text": "Please sit nearer the fire."},
    {"t": 51.0, "text": "She glances at the sleeve of her jacket."},
    {"t": 61.0, "text": "She looks at the sleeve again."},
    {"t": 72.0, "text": f"{C_} goes out of the room for a moment."},
    {"t": 81.0, "text": f"{C_} comes into the room again."},
]


class Model:
    """Answers a prompt from its own passage; records every prompt. split: prompt substrings whose answer changes
    from one repeat to the next."""

    def __init__(self, split=()):
        self.prompts, self.split = [], split

    def __call__(self, prompt, rep):
        self.prompts.append(prompt)
        psg = prompt.split("<<<\n", 1)[1].split("\n>>>", 1)[0] if "<<<" in prompt else ""
        q = prompt.split("Question:", 1)[1] if "Question:" in prompt else prompt
        who = next((n for n in (A, B, C_) if re.search(rf"\b{n}\b", q)), None)
        if any(s in prompt for s in self.split) and rep % 2:
            return "veiled" if "wearing" in q else "none"
        if "Does this line say" in q:
            line = prompt.split("action line:\n", 1)[1].split("\nQuestion", 1)[0]
            word = "raises her veil" if "lifts a veil" in q else "sleeve" if "sleeve" in q else "points"
            return "yes" if word in line else "no"
        if "In which shot" in q:
            return "shot 2"
        if "wearing at this moment" in q:
            return "veiled" if who == B else "usual"
        if "already there" in q:
            return "yes" if who == B and "already seated" in psg else "no"
        if "Where does the passage put" in q or "Where is" in q:
            want = "settee" if "passage put" in q else "armchair"
            n = next(ln.split(":")[0] for ln in q.splitlines() if ln.startswith("place ") and want in ln)
            return n
        if "go out of the room" in q:
            return ", ".join(n for n in (A, B, C_) if f"{n} goes out" in psg) or "none"
        if "come into the room" in q:
            return ", ".join(n for n in (A, B, C_) if f"{n} comes into" in psg) or "none"
        if "kind of camera" in q:
            return "two_shot"
        if "wearing when this shot ends" in q:
            if "raises her veil" in psg or "veil raised" in psg:
                return "usual"
            m = re.search(rf"{who} wears: ([a-z ]+)", prompt)
            return m.group(1).strip() if m else "not stated"
        if "begin to do" in q:
            act = q.split("The shot's action:", 1)[1].splitlines()[0]
            if who == B and "veil" in act:
                return "veil_lift"                       # claimed by the lift and by the shot that only holds it up
            if who == B and "sleeve" in act:
                return "look_at_sleeve"
            if who == A:
                return "gesture_point"                   # claimed in two shots whose lines do not state it
            return "none"
        return "not stated"


@pytest.fixture
def scene(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [
        {"name": "fireplace", "position": [0.0, 2.3], "size": [1.4, 0.4, 1.2]},
        {"name": "armchair", "position": [1.6, 1.3], "size": [0.8, 0.8, 1.0], "seat": True,
         "description": "Leather armchair, buttoned"},
        {"name": "settee", "position": [-2.2, 0.9], "size": [1.2, 0.8, 1.0], "seat": True,
         "description": "Small settee in patterned cloth"},
        {"name": "desk", "position": [-1.6, -1.6], "size": [1.4, 0.7, 0.8]}],
        lights=[{"type": "point", "position": [0.0, 2.3, 0.4]}])
    for cid in ("fig_a", "fig_b", "fig_c", "fig_b_veiled", "fig_b_seated_armchair"):
        S.add_character(lib, cid)
    monkeypatch.setattr(P.C, "LIBRARY", lib)
    plan = {sid: [sh] for sid, sh in PLAN.items()}
    plan["_es"] = [{"id": sid, "t": t} for sid, t in TIMES.items()]
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "shots.txt").write_text("\n".join(PLAN))
    (tmp_path / "cast.json").write_text(json.dumps(CAST))
    (tmp_path / "timeline.json").write_text(json.dumps(TIMELINE))
    return tmp_path, lib


def run(scene, monkeypatch, model):
    tmp, lib = scene
    monkeypatch.setattr(P.Asker, "one", lambda self, prompt, rep: model(prompt, rep))
    monkeypatch.setattr(sys, "argv", ["staging_pass.py", str(tmp / "plan.json"), str(tmp / "shots.txt"),
                                      str(tmp / "cast.json"), str(tmp / "timeline.json"),
                                      str(lib / "set" / "room_x"), str(tmp / "pass.json")])
    P.main()
    return json.loads((tmp / "pass.json").read_text())


def test_worn_variants_are_the_figures_own(scene):
    _, lib = scene
    assert P.variants(lib, "fig_b") == ["usual", "veiled"]            # a seated stance is not a worn state
    assert P.variants(lib, "fig_a") == ["usual"]


def test_a_worn_change_is_taken_from_the_shots_own_span_and_owned_by_the_first_action_shot(scene, monkeypatch):
    m = Model()
    res = run(scene, monkeypatch, m)
    assert res["worn_start"][B] == "veiled"
    # the walk's wider passage reaches the line where the veil goes up; its own span does not: no early claim
    assert "action-2" not in res["worn_end"]
    # the lift and the shot after it both end with the veil raised: the lift (the first action shot) owns the change
    assert res["worn_end"] == {"action-3": {B: "usual"}}
    walk_q = next(p for p in m.prompts if "wearing when this shot ends" in p and "walks to the fire" in p)
    assert "raises her veil" not in walk_q.split("<<<\n", 1)[1].split("\n>>>", 1)[0]


def test_a_start_action_belongs_to_the_shots_whose_own_line_states_it(scene, monkeypatch):
    m = Model()
    res = run(scene, monkeypatch, m)
    assert res["pose"]["action-3"][B] == "veil_lift"
    assert res["pose"]["action-4"][B] == "none"                        # holds it up: its line does not state the lift
    # two shots in a row whose lines both state the look: both own it
    assert res["pose"]["action-6"][B] == "look_at_sleeve" and res["pose"]["action-7"][B] == "look_at_sleeve"
    # claimed by two shots whose lines do not state it: one owner by the closed question, the other none
    assert res["pose"]["action-6"][A] == "none" and res["pose"]["action-7"][A] == "gesture_point"
    assert res["pose"]["action-2"][B] == "none"
    assert res["_action_owner"][f"{B}: veil_lift"] == ["action-3"]


def test_the_start_action_question_offers_none_first_and_no_gaze(scene, monkeypatch):
    m = Model()
    run(scene, monkeypatch, m)
    q = next(p for p in m.prompts if "begin to do" in p)
    opts = q.rsplit("nothing else: ", 1)[1].split("; ")
    assert opts[0] == "none" and opts[-1] == P.NOT
    assert "Most shots begin with no special action" in q
    assert not any("look" == o or o.startswith("look ") for o in opts) and "look" not in P.POSES
    assert set(P.POSES) <= set(opts)


def test_presence_changes_only_on_a_stated_exit_and_entrance(scene, monkeypatch):
    res = run(scene, monkeypatch, Model())
    assert res["exits"] == {"talk-8": [C_]} and res["enters"] == {"talk-9": [C_]}
    # nobody "left" merely by not being named in a shot's lines
    assert all(sid in ("talk-8",) for sid in res["exits"])


def test_a_person_already_in_the_place_is_reported_with_the_sets_own_thing(scene, monkeypatch):
    m = Model()
    res = run(scene, monkeypatch, m)
    assert res["present_at_start"] == {B: "settee"}
    where = next(p for p in m.prompts if "Where does the passage put" in p)
    places = [ln for ln in where.splitlines() if ln.startswith("place ")]
    assert len(places) == 4 and places[0].startswith("place 1: ")              # offered by number and description
    assert not any(re.match(r"place \d+: (armchair|settee|desk|fireplace) \(", ln) for ln in places)   # not by set names


def test_where_a_move_ends_is_asked_only_when_the_posture_changes(scene, monkeypatch):
    m = Model()
    res = run(scene, monkeypatch, m)
    assert res["move_end"] == {"action-2": {B: "armchair"}}
    assert len({p for p in m.prompts if "when this shot ends? Choose the place" in p}) == 1


def test_talk_shots_get_a_camera_kind_from_the_closed_list(scene, monkeypatch):
    res = run(scene, monkeypatch, Model())
    assert set(res["camera"]) == {"talk-5", "talk-8", "talk-9"} and set(res["camera"].values()) == {"two_shot"}


def test_an_answer_the_repeats_do_not_agree_on_is_not_stated(scene, monkeypatch):
    res = run(scene, monkeypatch, Model(split=("raises her veil.\nThings",)))
    assert "action-3" not in res["pose"] or res["pose"]["action-3"].get(B) in (None, "none")
    assert "action-3" not in res["worn_end"]
    assert res["_split"] >= 1


# ------------------------------------------------------------------ the answer parser
def _asker(answers):
    a = P.Asker("http://unused", Path("."), 3)
    it = iter(answers)
    a.one = lambda prompt, rep: next(it)
    return a


def test_choose_takes_an_option_by_its_name_and_falls_back_to_not_stated():
    opts = ["place 1", "place 2", "place 12"]
    assert _asker(["place 2", "Place 2.", "\"place 2\""]).choose("q", opts) == "place 2"
    assert _asker(["I think place 12 fits", "place 12", "place 12"]).choose("q", opts) == "place 12"
    assert _asker(["somewhere else"] * 3).choose("q", opts) == P.NOT
    desc = ["armchair (leather, by the fire)", "settee (cloth, by the window)"]
    assert _asker(["settee"] * 3).choose("q", desc) == desc[1]
    split = _asker(["place 1", "place 2", "place 1"])
    assert split.choose("q", opts) == P.NOT and split.split == 1


def test_subset_takes_the_names_given_and_none_is_empty():
    names = [A, B, C_]
    assert _asker([f"{C_}"] * 3).subset("q", names) == [C_]
    assert _asker(["none"] * 3).subset("q", names) == []
    assert _asker([P.NOT] * 3).subset("q", names) == P.NOT
    assert _asker([A, f"{A}, {B}", A]).subset("q", names) == P.NOT


def test_a_span_is_the_timelines_lines_between_two_times():
    s = P.span(TIMELINE, 20.0, 30.0)
    assert "raises her veil" in s and "crossed the room" not in s and "veil raised" not in s


def test_a_shot_is_placed_in_the_audio_by_its_own_line_never_by_a_short_line_elsewhere():
    """A shot whose words begin inside a long speech has no line of its own to be found by: it sits by the shots
    around it, not at a short line later in the chapter that its opening letters happen to begin with."""
    tl = [{"t": 10.0, "text": "I shall hear the whole of it from the start."},
          {"t": 14.0, "text": "The house was old and the land was gone. Nothing was left of it but the walls."},
          {"t": 26.0, "text": "No."},
          {"t": 900.0, "text": "I shall hear the whole of it from the start."}]
    es = {"talk-1": {"t": 20.0, "words": "I shall hear the whole of it from the start."},
          "talk-2": {"t": 24.0, "words": "The house was old and the land was gone."},
          "talk-3": {"t": 28.0, "words": "Nothing was left of it but the walls."},
          "talk-4": {"t": 36.0, "words": "No."}}
    at = P.audio_times(tl, es, list(es))
    assert at["talk-1"] == 10.0 and at["talk-2"] == 14.0
    assert at["talk-3"] == pytest.approx(14.0 + (26.0 - 14.0) * 4 / 12)      # between its neighbours' own lines
    assert at["talk-4"] == 26.0                                             # a short line is found whole
