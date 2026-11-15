"""Plan -> scene specs (tools/assets/scenes/from_plan.py) on a made-up plan and a temp library (stdlib).

Places come from the plan's words; a vague place is a set of marks; people stay where they were placed in shots that
list nobody; everyone faces whom the shot is about and a place word never sets facing; an action key shows the
action's start; a move ends where the plan next lists the mover; the staging pass's answers fill only gaps (worn
state, start actions, presence, who is already there) and an answer the staged scene contradicts is dropped.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import from_plan as FP  # noqa: E402

A, B, C_ = "Alder", "Birch", "Cedar"                     # the plan's names for three people (keys a, b, c)
CAST = {A: {"key": "a", "id": "fig_a"}, B: {"key": "b", "base": "fig_b"}, C_: {"key": "c", "id": "fig_c"}}


@pytest.fixture
def lib(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [
        {"name": "fireplace", "position": [0.0, 2.3], "size": [1.4, 0.4, 1.2]},
        {"name": "armchair", "position": [1.6, 1.3], "size": [0.8, 0.8, 1.0], "seat": True},
        {"name": "armchair_2", "position": [-2.2, 0.9], "size": [0.8, 0.8, 1.1], "seat": True},
        {"name": "desk", "position": [-1.6, -1.6], "size": [1.4, 0.7, 0.8]}],
        lights=[{"type": "point", "position": [0.0, 2.3, 0.4]}],
        openings=[{"side": "west", "kind": "window", "from_m": 0.2, "to_m": 1.6, "sill_m": 0.8, "head_m": 2.2}])
    for cid in ("fig_a", "fig_b", "fig_c", "fig_b_veiled"):
        S.add_character(lib, cid)
    monkeypatch.setattr(FP.C, "LIBRARY", lib)
    return lib


def person(name, pose, at, facing=""):
    return {"name": name, "pose": pose, "at": at, "facing": facing}


def shot(kind="quiet", people=(), **kw):
    return dict({"kind": kind, "framing": "medium", "people": list(people), "place": "a sitting room",
                 "room": "a sitting room with a fireplace and a desk", "fixtures": ["fireplace", "desk"]}, **kw)


def run(tmp_path, monkeypatch, shots: dict, pass_=None, cast=None, extra=()):
    (tmp_path / "plan.json").write_text(json.dumps({sid: [sh] for sid, sh in shots.items()}))
    (tmp_path / "shots.txt").write_text("\n".join(shots))
    (tmp_path / "cast.json").write_text(json.dumps(cast or CAST))
    out = tmp_path / "out"
    argv = ["from_plan.py", str(tmp_path / "plan.json"), str(tmp_path / "shots.txt"), str(tmp_path / "cast.json"), str(out)]
    if pass_ is not None:
        (tmp_path / "pass.json").write_text(json.dumps(pass_))
        argv += ["--pass", str(tmp_path / "pass.json")]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    FP.main()
    specs = [json.loads(p.read_text()) for p in sorted(out.glob("pl_g*.json"))]
    return specs, json.loads((out / "plan_facts.json").read_text())


def cam_of(specs, sid):
    return next((sp, c) for sp in specs for c in sp["cameras"] if c["plan_id"] == sid)


OPEN = shot("quiet", [person(B, "sitting", "in the chair by the window", ""),
                      person(A, "standing", "near the desk", f"toward {B}"),
                      person(C_, "standing", f"beside {A}", "")])


# ------------------------------------------------------------------ places, marks, carrying
def test_places_come_from_the_plans_words(lib, tmp_path, monkeypatch):
    specs, facts = run(tmp_path, monkeypatch, {"quiet-1": OPEN})
    ch = specs[0]["characters"]
    assert ch["b"]["seat"] == "armchair_2" and ch["b"]["seated_from"] == "fig_b"     # the seat nearest the window
    assert "marks" not in ch["b"]                                                    # a seat is an exact place
    desk = [-1.6, -1.6]
    assert 0.5 <= math.dist(ch["a"]["position"], desk) <= 1.7
    assert 1.1 <= math.dist(ch["c"]["position"], ch["a"]["position"]) <= 1.3         # "beside" him: near, and a
    # camera can still show one face past the other's head (the standing gap, from_plan.PERSON_GAP_M)
    assert specs[0]["set"] == "room_x"


def test_a_vague_place_is_a_set_of_marks_with_the_chosen_one_first(lib, tmp_path, monkeypatch):
    specs, _ = run(tmp_path, monkeypatch, {"quiet-1": OPEN})
    a = specs[0]["characters"]["a"]
    assert a["marks"][0] == a["position"] and len(a["marks"]) >= 3
    things = FP.setfacts.things(lib / "set" / "room_x")
    for m in a["marks"]:
        assert abs(m[0]) <= 3.0 - 0.35 and abs(m[1]) <= 2.5 - 0.35               # inside the room
        for t in things:
            if t["kind"] == "fixture":                                           # clear of furniture
                assert not (abs(m[0] - t["center"][0]) < t["half"][0] + 0.25 and abs(m[1] - t["center"][1]) < t["half"][1] + 0.25)
    for i, m in enumerate(a["marks"]):
        for n in a["marks"][i + 1:]:
            assert math.dist(m, n) >= 0.2


def test_people_stay_placed_through_shots_that_list_nobody(lib, tmp_path, monkeypatch):
    talk = shot("talk", [], speaker=A, listener=B)
    specs, facts = run(tmp_path, monkeypatch, {"quiet-1": OPEN, "talk-2": talk, "talk-3": dict(talk, speaker=B, listener=A)})
    assert len(specs) == 1                                   # nobody moved: one staging
    sp, c = cam_of(specs, "talk-2")
    assert set(sp["characters"]) == {"a", "b", "c"} and c["speaker"] == "a" and c["listener"] == "b"
    assert facts["talk-2"]["staged"]["c"]["position"] == facts["quiet-1"]["staged"]["c"]["position"]
    assert any("carried from the shot before" in m for m in facts["talk-2"]["missing"])
    assert cam_of(specs, "talk-3")[1]["speaker"] == "b"


def test_a_new_staging_starts_when_someones_place_or_posture_changes(lib, tmp_path, monkeypatch):
    moved = shot("quiet", [person(B, "standing", "by the fire", "")])
    specs, _ = run(tmp_path, monkeypatch, {"quiet-1": OPEN, "quiet-2": OPEN, "quiet-3": moved})
    assert [len(sp["cameras"]) for sp in specs] == [2, 1]
    assert specs[1]["characters"]["b"]["body_pose"]["pose"] == "stand_relaxed"
    # a person on a vague place keeps the shot before's mark first in the next staging
    assert specs[1]["characters"]["a"]["marks"][0] == specs[0]["characters"]["a"]["position"]


# ------------------------------------------------------------------ facing
def test_facing_follows_the_words_or_whom_the_shot_is_about_never_a_place_word(lib, tmp_path, monkeypatch):
    specs, facts = run(tmp_path, monkeypatch, {"quiet-1": OPEN})
    ch, st = specs[0]["characters"], facts["quiet-1"]["staged"]
    assert ch["a"]["facing_to"] == "b"                                 # the plan's words: toward her
    assert ch["c"]["facing_to"] == "b"                                 # no words: the first person the shot lists
    for k in ("a", "c"):
        to_b = S.unit([st["b"]["position"][0] - st[k]["position"][0], st["b"]["position"][1] - st[k]["position"][1]])
        assert abs(ch[k]["facing"][0] - to_b[0]) < 0.02 and abs(ch[k]["facing"][1] - to_b[1]) < 0.02
    # standing "by the window" with no facing words: the window says where, not which way
    by_window = shot("quiet", [person(B, "sitting", "in the chair by the fire", ""),
                               person(A, "standing", "by the window", "")])
    specs, facts = run(tmp_path, monkeypatch, {"quiet-9": by_window})
    a = specs[0]["characters"]["a"]
    assert a["facing_to"] == "b"
    win = [-3.0, 0.9]
    to_win = S.unit([win[0] - a["position"][0], win[1] - a["position"][1]])
    assert a["facing"][0] * to_win[0] + a["facing"][1] * to_win[1] < 0.5


def test_an_action_subject_turning_to_a_named_thing_puts_that_thing_in_the_shot(lib, tmp_path, monkeypatch):
    turn = shot("action", [person(A, "standing", "near the desk", "toward the desk")],
                action=f"{A} turns toward the large desk.")
    specs, _ = run(tmp_path, monkeypatch, {"quiet-1": OPEN, "action-2": turn})
    _, c = cam_of(specs, "action-2")
    assert c["line_subject"] == "a" and c["face_thing"] == "desk"
    quiet = shot("quiet", [person(A, "standing", "near the desk", "toward the desk")], action=f"{A} waits.")
    specs, _ = run(tmp_path, monkeypatch, {"quiet-1": OPEN, "quiet-2": quiet})
    assert cam_of(specs, "quiet-2")[1]["face_thing"] is None           # only an action shot's turn asks for it


def test_a_speaker_faces_whom_he_speaks_to_after_turning_to_a_thing(lib, tmp_path, monkeypatch):
    turn = shot("action", [person(A, "standing", "near the desk", "toward the desk")],
                action=f"{A} turns toward the large desk.")
    talk = shot("talk", [person(A, "standing", "near the desk", "toward the desk")], speaker=A, listener=B)
    specs, facts = run(tmp_path, monkeypatch, {"quiet-1": OPEN, "action-2": turn, "talk-3": talk})
    g, _ = cam_of(specs, "talk-3")
    a, st = g["characters"]["a"], facts["talk-3"]["staged"]
    assert a["facing_to"] == "b"                                       # the listener, not the desk he turned to
    to_b = S.unit([st["b"]["position"][0] - st["a"]["position"][0], st["b"]["position"][1] - st["a"]["position"][1]])
    assert a["facing"][0] * to_b[0] + a["facing"][1] * to_b[1] > 0.99
    # a staging whose people are posed with him facing elsewhere is left when he speaks
    alone = shot("quiet", [person(B, "sitting", "in the chair by the window", ""),
                           person(A, "standing", "near the desk", "toward the desk")], action=f"{A} reads.")
    specs, _ = run(tmp_path, monkeypatch, {"quiet-1": alone, "talk-2": talk})
    g, _ = cam_of(specs, "talk-2")
    assert g is not cam_of(specs, "quiet-1")[0] and g["characters"]["a"]["facing_to"] == "b"


# ------------------------------------------------------------------ cameras
def test_camera_class_and_the_talk_camera_kind(lib, tmp_path, monkeypatch):
    shots = {"quiet-1": dict(OPEN, framing="wide"),
             "quiet-2": shot("quiet", [person(B, "sitting", "in the chair by the window", "")], framing="close"),
             "talk-3": shot("talk", [], speaker=A, listener=B, tight=True),
             "talk-4": shot("talk", [], speaker=A, listener=B),
             "talk-5": shot("talk", [], speaker=A, listener=B)}
    specs, _ = run(tmp_path, monkeypatch, shots, pass_={"camera": {"talk-4": "single", "talk-5": "over_the_shoulder"}})
    assert cam_of(specs, "quiet-1")[1]["kind"] == "full"
    assert cam_of(specs, "quiet-2")[1]["kind"] == "close"              # the plan's close is a close
    assert cam_of(specs, "talk-3")[1]["kind"] == "close"               # "tight"
    c4, c5 = cam_of(specs, "talk-4")[1], cam_of(specs, "talk-5")[1]
    assert c4.get("single") is True and "listener" not in c4
    assert c5.get("ots") == "b"
    assert cam_of(specs, "quiet-1")[1]["show"] == ["a", "b", "c"]      # everyone the shot lists


def test_the_shots_subject_is_the_speaker_else_the_first_person_listed(lib, tmp_path, monkeypatch):
    shots = {"quiet-1": OPEN, "talk-2": shot("talk", [], speaker=A, listener=B),
             "quiet-3": shot("quiet", [person(A, "standing", "near the desk", "")], action=f"{A} scans {B}.")}
    specs, _ = run(tmp_path, monkeypatch, shots)
    assert cam_of(specs, "quiet-1")[1]["subject"] == "b"
    assert cam_of(specs, "talk-2")[1]["subject"] == "a"
    c3 = cam_of(specs, "quiet-3")[1]
    assert c3["subject"] == "a" and c3["line_subject"] == "a"         # named first in the line


# ------------------------------------------------------------------ actions: start posture, where a move ends
def test_an_action_key_shows_the_start_posture(lib, tmp_path, monkeypatch):
    rise = shot("action", [person(B, "standing", "by the window", f"toward {A}")], action=f"{B} stands up.",
                pose={B: "sitting"}, pose_end={B: "standing"})
    specs, facts = run(tmp_path, monkeypatch, {"quiet-1": OPEN, "action-2": rise})
    sp, _ = cam_of(specs, "action-2")
    assert sp["characters"]["b"]["body_pose"]["pose"] == "seated" and sp["characters"]["b"]["seat"] == "armchair_2"
    assert len(specs) == 1                                             # the same staging as the shot before


def test_a_move_ends_where_the_plan_next_lists_the_mover(lib, tmp_path, monkeypatch):
    stand = shot("quiet", [person(B, "standing", "by the window", ""), person(A, "standing", "near the desk", ""),
                           person(C_, "standing", f"beside {A}", "")])
    walk = shot("action", [person(B, "standing", "by the window", "")], action=f"{B} walks across and sits.",
                pose={B: "standing"}, pose_end={B: "sitting"})
    talk = shot("talk", [], speaker=B, listener=A)
    later = shot("quiet", [person(B, "sitting", "in the chair by the fire", "")])
    shots = {"quiet-1": stand, "action-2": walk, "talk-3": talk, "quiet-4": later}
    specs, facts = run(tmp_path, monkeypatch, shots)
    sp2, c2 = cam_of(specs, "action-2")
    assert sp2["characters"]["b"]["body_pose"]["pose"] == "stand_relaxed"          # the key opens before the move
    assert c2["mover"] == "b" and c2["move_to"] == "armchair" and c2["move_landmark"] == "fireplace"
    sp3, _ = cam_of(specs, "talk-3")
    assert sp3["characters"]["b"]["seat"] == "armchair"                             # seated by the fire from the next shot
    assert any("ends where the plan next lists them" in u for u in facts["action-2"]["pass"]["used"])
    # without the look-ahead she is still standing at the window in the talk shot (the fault it fixes)
    specs, _ = run(tmp_path, monkeypatch, shots, extra=["--no-lookahead"])
    assert cam_of(specs, "talk-3")[0]["characters"]["b"]["body_pose"]["pose"] == "stand_relaxed"


def test_the_passes_move_end_is_used_and_marks_the_move(lib, tmp_path, monkeypatch):
    stand = shot("quiet", [person(B, "standing", "by the window", ""), person(A, "standing", "near the desk", "")])
    walk = shot("action", [person(B, "standing", "by the window", "")], action=f"{B} changes her seat.",
                pose={B: "standing"}, pose_end={B: "sitting"})
    shots = {"quiet-1": stand, "action-2": walk, "talk-3": shot("talk", [], speaker=B, listener=A)}
    specs, facts = run(tmp_path, monkeypatch, shots, pass_={"move_end": {"action-2": {B: "armchair"}}})
    assert cam_of(specs, "action-2")[1]["move_to"] == "armchair"
    assert cam_of(specs, "talk-3")[0]["characters"]["b"]["seat"] == "armchair"


# ------------------------------------------------------------------ the pass: worn state, start actions, presence
def test_worn_state_is_carried_from_the_start_and_changes_only_at_a_shots_end(lib, tmp_path, monkeypatch):
    cast = dict(CAST, **{B: {"key": "b", "base": "fig_b", "worn": {"veiled": {"wear_add": "a black veil over the face",
                                                                             "face_covered": True}}}})
    lift = shot("action", [person(B, "sitting", "in the chair by the window", "")], action=f"{B} raises her veil.",
                pose={B: "sitting"})
    shots = {"quiet-1": OPEN, "action-2": lift, "talk-3": shot("talk", [], speaker=B, listener=A)}
    pass_ = {"worn_start": {B: "veiled"}, "worn_end": {"action-2": {B: "usual"}},
             "pose": {"action-2": {B: "veil_lift"}}}
    specs, facts = run(tmp_path, monkeypatch, shots, pass_=pass_, cast=cast)
    b1 = cam_of(specs, "quiet-1")[0]["characters"]["b"]
    assert b1["variant"] == "veiled" and b1["worn"] == "veiled" and b1["face_covered"] is True and b1["wear_add"]
    b2 = cam_of(specs, "action-2")[0]["characters"]["b"]
    assert b2["worn"] == "veiled"                                       # the lift's key opens with the veil down
    assert b2["body_pose"]["params"]["action"] == "veil_lift" and b2["_pose_choice"] == "veil_lift"
    b3 = cam_of(specs, "talk-3")[0]["characters"]["b"]
    assert "worn" not in b3 and "variant" not in b3 and "wear_add" not in b3     # raised from the next shot on
    assert facts["talk-3"]["pass"]["worn"]["b"] == "usual"


def test_a_start_action_the_staged_scene_contradicts_is_dropped_and_reported(lib, tmp_path, monkeypatch):
    act = shot("action", [person(B, "sitting", "in the chair by the window", "")], action=f"{B} starts.",
               pose={B: "sitting"})
    pass_ = {"pose": {"action-2": {B: "stand_hand_on_mantel"}}}
    specs, facts = run(tmp_path, monkeypatch, {"quiet-1": OPEN, "action-2": act}, pass_=pass_)
    b = cam_of(specs, "action-2")[0]["characters"]["b"]
    assert b["body_pose"]["pose"] == "seated" and "action" not in b["body_pose"]["params"]
    assert any("stand_hand_on_mantel" in d and "is sitting" in d for d in facts["action-2"]["pass"]["dropped"])
    # a veil lift with no veil down, a hand on the mantel away from the fire, a reach with nobody seated
    things = FP.setfacts.things(lib / "set" / "room_x")
    st = {"a": {"pose": "standing", "position": [-1.0, -1.0]}, "c": {"pose": "standing", "position": [0.0, -1.0]}}
    assert FP.action_misfit("veil_lift", "b", {"pose": "sitting", "position": [0, 0]}, st, None, things)
    assert FP.action_misfit("veil_lift", "b", {"pose": "sitting", "position": [0, 0]}, st, "veiled", things) is None
    assert FP.action_misfit("stand_hand_on_mantel", "a", st["a"], st, None, things)
    at_fire = {"pose": "standing", "position": [0.0, 1.7]}
    assert FP.action_misfit("stand_hand_on_mantel", "a", at_fire, st, None, things) is None
    assert FP.action_misfit("reach_to_forearm", "a", st["a"], st, None, things)
    st2 = dict(st, b={"pose": "sitting", "position": [1.6, 1.3]})
    assert FP.action_misfit("reach_to_forearm", "a", st["a"], st2, None, things) is None


def test_a_reach_places_the_reacher_beside_the_seated_person(lib, tmp_path, monkeypatch):
    seated = shot("quiet", [person(B, "sitting", "in the chair by the fire", ""), person(A, "standing", "near the desk", "")])
    reach = shot("action", [person(A, "standing", "near the desk", ""), person(B, "sitting", "in the chair by the fire", "")],
                 action=f"{A} bends forward and pats the forearm of {B}.", pose={A: "standing", B: "sitting"})
    specs, facts = run(tmp_path, monkeypatch, {"quiet-1": seated, "action-2": reach},
                       pass_={"pose": {"action-2": {A: "reach_to_forearm"}}})
    sp, c = cam_of(specs, "action-2")
    a, b = sp["characters"]["a"], sp["characters"]["b"]
    assert a["reach"]["to"] == "b" and a["reach"]["her_side"] in ("l", "r") and "marks" not in a
    assert math.dist(a["position"], b["position"]) < 0.8               # beside her, not "near the desk"
    to_b = S.unit([b["position"][0] - a["position"][0], b["position"][1] - a["position"][1]])
    assert a["facing"][0] * to_b[0] + a["facing"][1] * to_b[1] > 0.99
    assert c["show_joints"][0] == ["a", "hand_r"] and c["show_joints"][1][0] == "b"


def test_presence_changes_only_on_a_stated_exit_and_entrance(lib, tmp_path, monkeypatch):
    talk = shot("talk", [], speaker=A, listener=B)
    shots = {"quiet-1": OPEN, "talk-2": talk, "talk-3": talk, "talk-4": talk, "talk-5": talk}
    specs, facts = run(tmp_path, monkeypatch, shots)
    for sid in shots:                                                  # not being mentioned is not leaving
        assert set(facts[sid]["staged"]) == {"a", "b", "c"}
    pass_ = {"exits": {"talk-3": [C_]}, "enters": {"talk-5": [C_]}}
    specs, facts = run(tmp_path, monkeypatch, shots, pass_=pass_)
    assert set(facts["talk-2"]["staged"]) == {"a", "b", "c"}
    assert set(facts["talk-3"]["staged"]) == {"a", "b", "c"}           # he goes out during it: its opening has him
    assert set(facts["talk-4"]["staged"]) == {"a", "b"}
    assert set(facts["talk-5"]["staged"]) == {"a", "b", "c"}
    assert facts["talk-5"]["staged"]["c"]["position"] == facts["talk-2"]["staged"]["c"]["position"]   # back where he left
    # the plan's own list does not bring back someone the passage sent out
    listed = shot("quiet", [person(C_, "standing", "near the desk", "")])
    specs, facts = run(tmp_path, monkeypatch, {"quiet-1": OPEN, "talk-2": talk, "quiet-3": listed},
                       pass_={"exits": {"quiet-1": [C_]}})
    assert "c" not in facts["talk-2"]["staged"] and "c" not in facts["quiet-3"]["staged"]


def test_a_person_the_passage_has_already_there_is_present_from_the_first_shot(lib, tmp_path, monkeypatch):
    first = shot("quiet", [person(A, "standing", "near the desk", ""), person(C_, "standing", f"beside {A}", "")])
    later = shot("quiet", [person(B, "sitting", "in the chair by the window", "")])
    shots = {"quiet-1": first, "quiet-2": later}
    specs, facts = run(tmp_path, monkeypatch, shots)
    assert "b" not in facts["quiet-1"]["staged"]                       # the plan's gap: she appears a shot late
    specs, facts = run(tmp_path, monkeypatch, shots, pass_={"present_at_start": {B: "armchair_2"}})
    st = facts["quiet-1"]["staged"]["b"]
    assert st["pose"] == "sitting" and st["seat"] == "armchair_2"
    assert any("already in the place when the scene opens" in u for u in facts["quiet-1"]["pass"]["used"])
    # "not stated" places nobody
    specs, facts = run(tmp_path, monkeypatch, shots, pass_={"present_at_start": {B: "not stated"}})
    assert "b" not in facts["quiet-1"]["staged"]


# ------------------------------------------------------------------ small mechanics
def test_place_words_resolve_to_set_things_and_people():
    things = [{"name": "fireplace", "kind": "fixture", "center": [0, 2, 0.6]},
              {"name": "desk", "kind": "fixture", "center": [-1, -1, 0.4]},
              {"name": "window_west", "kind": "window", "center": [-3, 0, 1.5]}]
    names = {A: "a", B: "b"}
    assert FP.thing_for("in the chair by the fire", things, names)[0]["name"] == "fireplace"
    assert FP.thing_for("near the desk", things, names)[0]["name"] == "desk"
    assert FP.thing_for("by the window", things, names)[0]["name"] == "window_west"
    assert FP.thing_for(f"beside {A}", things, names) == (None, "a")
    assert FP.thing_for("", things, names) == (None, None)


def test_a_free_mark_is_clear_of_furniture_and_people_and_inside_the_room():
    things = [{"name": "desk", "kind": "fixture", "center": [0.0, 0.0, 0.4], "half": [0.7, 0.35, 0.4]}]
    room = [6.0, 5.0, 3.0]
    others = [[0.0, -1.0]]
    p = FP.free_mark([0.0, 0.0], others, things, room)
    assert not (abs(p[0]) < 0.7 + 0.25 and abs(p[1]) < 0.35 + 0.25)
    assert math.dist(p, others[0]) >= 0.55 and abs(p[0]) <= 3.0 - 0.35 and abs(p[1]) <= 2.5 - 0.35
    marks = FP.free_marks([0.0, 0.0], others, things, room)
    assert marks and all(math.dist(m, others[0]) >= 0.55 for m in marks)
