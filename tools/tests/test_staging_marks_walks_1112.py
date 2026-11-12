"""Marks, facing, walks and staged facts (tools/assets/scenes/speakers.py), on synthetic scenes (stdlib).

A vague place is a set of marks and the person on it may move among them; an exact place and a seated person never
move. Everyone faces whom the shot is about from where they now stand. A walk starts on the mark nearest its end
place with a clear floor path. What the plan asks of a person (worn state, facing, the action's start) is checked on
the measured figure.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import speakers as SP  # noqa: E402

STAND = {"pose": "stand_relaxed", "params": {}}
SEAT = {"pose": "seated", "params": {}}


def _char(pos, facing=(0, -1), pose=None, **kw):
    return dict({"id": "fig", "position": list(pos), "facing": list(facing), "body_pose": dict(pose or STAND)}, **kw)


def _deg(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))))


# ------------------------------------------------------------------ facing
def test_bodies_are_aimed_at_whom_they_face_from_where_everyone_now_stands():
    spec = {"characters": {"a": _char([-2, 0], facing=[0, 1], facing_to="b"),        # turned to the window wall
                           "c": _char([-2.6, 0.4], facing=[0, 1], facing_to="b"),
                           "b": _char([1.5, 1.0], facing=[-1, 0])}}
    aimed = SP.aim_bodies(spec)
    assert sorted(aimed) == ["a", "c"]
    for k in ("a", "c"):
        ch = spec["characters"][k]
        to_b = S.unit([1.5 - ch["position"][0], 1.0 - ch["position"][1]])
        assert _deg(S.unit(ch["facing"]), to_b) < 1.0
    # she is moved (a walk's start, a mark): the same call re-aims them at her new place
    spec["characters"]["b"]["position"] = [0.0, -2.0]
    assert sorted(SP.aim_bodies(spec)) == ["a", "c"]
    a = spec["characters"]["a"]
    assert _deg(S.unit(a["facing"]), S.unit([0.0 - a["position"][0], -2.0 - a["position"][1]])) < 1.0
    assert SP.aim_bodies(spec) == []                              # already aimed: nothing to do


def test_a_seated_person_keeps_the_seats_facing_and_no_target_means_no_turn():
    spec = {"characters": {"b": _char([1, 1], facing=[0, -1], pose=SEAT, facing_to="a"),
                           "a": _char([-1, 0], facing=[0, 1]),                 # faces nobody in particular
                           "c": _char([0, 2], facing=[1, 0], facing_to="c")}}  # "faces" themselves: ignored
    assert SP.aim_bodies(spec) == []
    assert spec["characters"]["b"]["facing"] == [0, -1] and spec["characters"]["a"]["facing"] == [0, 1]


def test_listeners_heads_turn_to_the_speaker_and_the_chest_takes_the_rest():
    spec = {"characters": {"a": _char([0, 0], facing=[0, -1]),
                           "b": _char([2, 0], facing=[0, -1]),                 # speaker is at her right hand side
                           "c": _char([0, -3], facing=[0, -1])}}               # speaker directly behind him
    SP.look_at(spec, "a")
    assert "_look" not in spec["characters"]["a"]
    b = spec["characters"]["b"]["_look"]
    assert b["at"] == "a" and abs(abs(b["yaw"]) - 90.0) < 1.0
    steps_b = spec["characters"]["b"]["body_pose"]["params"]["steps"]
    head_turn = sum(s[3] for s in steps_b if s[1] in ("neck_01", "head") and s[2] == [0, 0, 1])
    chest = sum(s[3] for s in steps_b if s[1] == "spine_03")
    assert abs(abs(head_turn) - SP.LOOK_MAX_DEG) < 0.5 and abs(abs(chest) - (90.0 - SP.LOOK_MAX_DEG)) < 0.5
    c = spec["characters"]["c"]["_look"]
    assert abs(abs(c["yaw"]) - 180.0) < 1.0
    # taken off again without leaving steps behind (heads are re-aimed after every move)
    un = SP.unlook(spec["characters"]["b"])
    assert "_look" not in un and un["body_pose"]["params"]["steps"] == []


def test_look_false_and_a_look_target_of_their_own_are_respected():
    spec = {"characters": {"a": _char([0, 0]), "b": _char([2, 0], look=False),
                           "c": _char([0, -3], look_at="b")}}
    SP.look_at(spec, "a")
    assert "_look" not in spec["characters"]["b"]
    assert spec["characters"]["c"]["_look"]["at"] == "b"


# ------------------------------------------------------------------ marks
def test_a_mark_move_keeps_the_angle_to_the_reference_person_and_re_aims_heads():
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([0, 0], facing=[1, 0]),                  # facing b
                           "b": _char([2, 0], facing=[-1, 0]),
                           "c": _char([1, 1.5], facing=[0, -1])}}
    SP.look_at(spec, "a")
    subj = {k: S.person(v["position"], v["facing"]) for k, v in spec["characters"].items()}
    new = [0.0, 1.0]
    moved = SP.shifted(subj, spec, {"a": new}, "a")
    m = moved["a"]["measured"]
    assert math.dist(m["pelvis_xy"], new) < 1e-6 and math.dist(m["eye_xy"], new) < 1e-6
    to_b = S.unit([2 - new[0], 0 - new[1]])
    assert _deg(S.unit(m["facing"]), to_b) < 1.0                  # still turned to b from the new mark
    assert subj["a"]["measured"]["pelvis_xy"] == [0, 0]            # the measured scene itself is not edited
    # c listens to a: the head follows a to the new mark
    hc = moved["c"]["measured"]["head_fwd_xy"]
    assert _deg(S.unit(hc), S.unit([new[0] - 1, new[1] - 1.5])) < 1.0
    SP.apply_marks(spec, {"a": new}, "a", "a")
    a = spec["characters"]["a"]
    assert a["position"] == [0.0, 1.0] and _deg(S.unit(a["facing"]), to_b) < 1.0
    assert spec["characters"]["c"]["_look"]["at"] == "a"


def test_mark_reference_is_the_look_target_then_the_line_then_the_speaker():
    spec = {"line": ["a", "b"], "characters": {"a": _char([0, 0]), "b": _char([2, 0]),
                                               "c": _char([1, 2], look_at="b"), "d": _char([3, 3])}}
    assert SP.mark_ref(spec, "c", "a") == "b"
    assert SP.mark_ref(spec, "a", "a") == "b"
    assert SP.mark_ref(spec, "d", "a") == "a"
    assert SP.mark_ref(spec, "d", "d") is None


# ------------------------------------------------------------------ walks
@pytest.fixture
def room(tmp_path, monkeypatch):
    """A 6 x 5 m room: a fireplace on the north wall, a leather armchair to its right, a like armchair by the west
    window, a desk; things by their set names."""
    lib = S.make_library(tmp_path, [
        {"name": "fireplace", "position": [0.0, 2.3], "size": [1.4, 0.4, 1.2]},
        {"name": "armchair", "position": [1.6, 1.2], "size": [0.8, 0.8, 1.0], "seat": True},
        {"name": "armchair_2", "position": [-2.2, 0.9], "size": [0.8, 0.8, 1.1], "seat": True},
        {"name": "desk", "position": [-1.8, -1.6], "size": [1.4, 0.7, 0.8]}],
        lights=[{"type": "point", "position": [0.0, 2.3, 0.4]}])
    monkeypatch.setattr(SP.C, "LIBRARY", lib)
    return lib


def _walk_spec(chars, mover="b", to="armchair"):
    return {"set": "room_x", "characters": chars,
            "cameras": [{"name": "s1", "kind": "full", "mover": mover, "move_to": to}]}


def test_a_walk_starts_on_the_allowed_mark_nearest_its_end_place(room):
    chars = {"b": _char([-2.0, -1.2], marks=[[-2.0, -1.2], [-1.4, 0.2], [0.2, 0.6]]),
             "a": _char([-2.4, -0.2])}
    got = SP.clear_walks(_walk_spec(chars), {"armchair", "armchair_2"})
    assert chars["b"]["position"] == [0.2, 0.6] and got["moved"]["b"] == [[-2.0, -1.2], [0.2, 0.6]]
    f = got["shot_facts"]["s1"]
    assert f["moves"]["who"] == "b" and f["moves"]["to"] == "armchair" and f["moves"]["posture_end"] == "sitting"
    assert f["fixed"] == {"a": "standing"} and f["path_blocked_by"] == []


def test_a_stated_start_is_not_moved(room):
    chars = {"b": _char([-2.0, -1.2]), "a": _char([-2.4, 0.8])}      # no marks: her start is an exact place
    got = SP.clear_walks(_walk_spec(chars), {"armchair"})
    assert chars["b"]["position"] == [-2.0, -1.2] and "b" not in got["moved"]


def test_a_start_mark_with_a_like_seat_nearer_than_the_end_place_is_avoided(room):
    # the mark nearest the end chair by distance is nearer still to the other armchair: the next mark wins
    near_like = [-0.4, 1.0]          # nearest the end armchair of all, but the other armchair is nearer still
    clear = [0.0, -0.4]              # a little further from the end, and the end is its nearest armchair
    assert math.dist(near_like, [1.6, 1.2]) < math.dist(clear, [1.6, 1.2])
    assert math.dist(clear, [1.6, 1.2]) < math.dist(clear, [-2.2, 0.9])
    assert math.dist(near_like, [-2.2, 0.9]) < math.dist(near_like, [1.6, 1.2])
    chars = {"b": _char([-2.0, -1.2], marks=[[-2.0, -1.2], near_like, clear]), "a": _char([2.5, -2.0])}
    SP.clear_walks(_walk_spec(chars), {"armchair", "armchair_2"})
    assert chars["b"]["position"] == clear
    assert [x["name"] for x in SP.similar_seats(_walk_spec(chars), SP.end_thing(_walk_spec(chars), {"move_to": "armchair"}))] \
        == ["armchair_2"]


def test_a_vague_place_person_on_the_path_steps_off_it(room):
    start, end = [-1.5, 0.0], [1.6, 1.2]
    on_path = [0.0, 0.6]
    assert SP.seg_dist(on_path, start, end) < SP.PATH_CLEAR_M
    off = [0.2, -1.4]
    assert SP.seg_dist(off, start, end) >= SP.PATH_CLEAR_M
    chars = {"b": _char(start), "a": _char(on_path, marks=[on_path, [0.3, 0.75], off])}
    got = SP.clear_walks(_walk_spec(chars), {"armchair"})
    assert chars["a"]["position"] == off and got["moved"]["a"] == [on_path, off] and got["blocked"] == []
    for k, v in chars.items():
        if k != "b":
            assert SP.seg_dist(v["position"], start, end) >= SP.PATH_CLEAR_M


def test_an_exact_place_or_a_seated_person_on_the_path_is_reported_not_moved(room):
    start = [-1.5, 0.0]
    chars = {"b": _char(start), "a": _char([0.0, 0.6])}                       # no marks: an exact place
    got = SP.clear_walks(_walk_spec(chars), {"armchair"})
    assert chars["a"]["position"] == [0.0, 0.6] and got["blocked"] == ["a"] and got["moved"] == {}
    assert got["shot_facts"]["s1"]["path_blocked_by"] == ["a"]
    chars = {"b": _char(start), "c": _char([0.0, 0.6], pose=SEAT, marks=[[0.0, 0.6], [0.2, -1.4]])}
    got = SP.clear_walks(_walk_spec(chars), {"armchair"})
    assert chars["c"]["position"] == [0.0, 0.6] and got["blocked"] == ["c"]
    assert got["shot_facts"]["s1"]["fixed"] == {"c": "sitting"}


def test_a_move_shots_frame_points_and_its_landmarks_front(room):
    spec = _walk_spec({"b": _char([-1.5, 0.0])})
    c = dict(spec["cameras"][0], move_landmark="fireplace")
    lm = SP.landmark(spec, c)
    assert lm["name"] == "fireplace" and lm["front"] == [0.0, -1.0]           # off the wall it stands nearest
    assert abs(lm["top"] - 1.2) < 1e-6
    pts = SP.move_points(spec, c)
    assert [-1.5, 0.0, 1.65] in pts and [-1.5, 0.0, 0.05] in pts              # the mover's head and feet at the start
    assert any(abs(p[0] - 1.6) < 1e-6 and abs(p[1] - 1.2) < 1e-6 for p in pts)  # the end place
    assert any(abs(p[0] - 0.05) < 1e-6 and abs(p[1] - 0.6) < 1e-6 for p in pts)  # the path's middle
    assert lm["center"] in pts
    assert SP.move_points(spec, {"name": "s2", "kind": "medium"}) == []       # not a move shot


# ------------------------------------------------------------------ staged facts: what the engine still must hold
def test_worn_state_needs_its_figure_and_its_words():
    spec = {"characters": {"b": dict(_char([0, 0]), id="fig_seated_x", worn="veiled")}}
    bad = SP.staged_facts(spec, {"b": S.person([0, 0])})["b"]
    assert any("the staged figure is" in x for x in bad) and any("no words for it reach the redraw" in x for x in bad)
    spec = {"characters": {"b": dict(_char([0, 0]), id="fig_seated_x_veiled", worn="veiled", wear_add="a veil")}}
    assert SP.staged_facts(spec, {"b": S.person([0, 0])})["b"] == []
    spec = {"characters": {"b": dict(_char([0, 0]), id="fig", worn="usual")}}
    assert SP.staged_facts(spec, {"b": S.person([0, 0])})["b"] == []


def test_a_body_more_than_a_quarter_turn_from_whom_it_faces_is_a_miss():
    spec = {"characters": {"a": _char([0, 0], facing_to="b"), "b": _char([2, 0])}}
    ok = {"a": S.person([0, 0], facing=S.unit([1, 0.5])), "b": S.person([2, 0])}
    assert SP.staged_facts(spec, ok)["a"] == []
    away = {"a": S.person([0, 0], facing=[-1, 0.2]), "b": S.person([2, 0])}       # turned to the window instead
    assert any("whom the plan has them face" in x for x in SP.staged_facts(spec, away)["a"])
    edge_in = {"a": S.person([0, 0], facing=[math.cos(math.radians(SP.FACE_TURN_DEG - 3)),
                                              math.sin(math.radians(SP.FACE_TURN_DEG - 3))]), "b": S.person([2, 0])}
    assert SP.staged_facts(spec, edge_in)["a"] == []


def _with_hands(subj, d_l, d_r):
    m = subj["measured"]
    chin = [m["eye_xy"][0], m["eye_xy"][1], m["chin_z"]]
    m["joints"]["hand_l"] = [chin[0] + d_l, chin[1], chin[2]]
    m["joints"]["hand_r"] = [chin[0] - d_r, chin[1], chin[2]]
    return subj


def test_an_actions_start_is_measured_on_the_hands():
    lim_v, lim_s = SP.ACTION_HANDS_M["veil_lift"][0], SP.ACTION_HANDS_M["look_at_sleeve"][0]
    veil = {"characters": {"b": _char([0, 0], pose={"pose": "seated", "params": {"action": "veil_lift"}})}}
    both = {"b": _with_hands(S.person([0, 0], seated=True), lim_v * 0.8, lim_v * 0.8)}
    assert SP.staged_facts(veil, both)["b"] == []
    one = {"b": _with_hands(S.person([0, 0], seated=True), lim_v * 0.8, lim_v * 2.5)}     # one hand in the lap
    assert any("action 'veil_lift': hands" in x for x in SP.staged_facts(veil, one)["b"])
    sleeve = {"characters": {"b": _char([0, 0], pose={"pose": "seated", "params": {"action": "look_at_sleeve"}})}}
    either = {"b": _with_hands(S.person([0, 0], seated=True), lim_s * 0.8, lim_s * 2.5)}
    assert SP.staged_facts(sleeve, either)["b"] == []
    neither = {"b": _with_hands(S.person([0, 0], seated=True), lim_s * 1.5, lim_s * 1.5)}
    assert any("action 'look_at_sleeve'" in x for x in SP.staged_facts(sleeve, neither)["b"])
    unknown = {"characters": {"b": _char([0, 0], pose={"pose": "seated", "params": {"action": "wave"}})}}
    assert any("no start pose for it" in x for x in SP.staged_facts(unknown, both)["b"])


def test_a_reach_is_measured_hand_to_contact():
    spec = {"characters": {"a": dict(_char([0, 0]), reach={"to": "b", "her_side": "l", "side": "r",
                                                         "contact": [0.3, 0.3, 0.8]}), "b": _char([0.6, 0.4], pose=SEAT)}}
    subj = {"a": S.person([0, 0]), "b": S.person([0.6, 0.4], seated=True)}
    subj["a"]["measured"]["joints"]["hand_r"] = [0.3, 0.3, 0.8 + SP.REACH_M]
    assert SP.staged_facts(spec, subj)["a"] == []
    subj["a"]["measured"]["joints"]["hand_r"] = [0.3, 0.3, 0.8 + SP.REACH_M + 0.3]
    assert any("reach: hand" in x for x in SP.staged_facts(spec, subj)["a"])
    unsolved = {"characters": {"a": dict(_char([0, 0]), reach={"to": "b", "her_side": "l"}), "b": _char([0.6, 0.4])}}
    assert any("reach: not solved" in x for x in SP.staged_facts(unsolved, subj)["a"])


# ------------------------------------------------------------------ the camera search
def test_candidate_cameras_are_inside_the_room_in_front_of_the_face_and_not_behind_someone():
    subj = {"a": S.person([0.0, 0.5], facing=[0, -1]), "b": S.person([0.0, -0.6], facing=[0, 1])}
    room = [7.0, 7.0, 3.0]
    c = {"name": "s", "kind": "medium", "focal_mm": 50, "az": 0.0}
    shot = {"camera": S.front_cam(subj["a"], 1.5)}
    got = SP.candidates(c, shot, "a", subj, S.RES, room, None, None, [])
    m = subj["a"]["measured"]
    eye = [m["eye_xy"][0], m["eye_xy"][1], m["eye_z"]]
    for q in got:
        assert SP.in_room(q["position"], room)
        assert SP.plan_angle(m["head_fwd_xy"], m["eye_xy"], q["position"]) <= SP.MAX_HEAD_DEG - 5
        assert not SP.blocked(q["position"], eye, [subj["b"]["measured"]])
    # with nobody in the way there are more of them; one already tried is not offered again
    free = SP.candidates(c, shot, "a", {"a": subj["a"]}, S.RES, room, None, None, [])
    assert len(free) > len(got) and free
    again = SP.candidates(c, shot, "a", {"a": subj["a"]}, S.RES, room, None, None, [free[0]["position"]])
    assert free[0]["position"] not in [q["position"] for q in again]


def test_two_shot_search_returns_only_cameras_that_pass_and_falls_back_to_a_single():
    room = [4.0, 4.0, 3.0]
    subj = {"a": S.person([-0.5, 0.8], facing=[0.3, -1]), "b": S.person([0.6, 0.6], facing=[-0.3, -1])}
    c = {"name": "s", "kind": "medium"}
    shot = {"camera": S.cam([0.0, -1.5, 1.5], [0.0, 0.7, 1.5], 35)}
    got = SP.two_shot_candidates(c, shot, "a", "b", subj, S.RES, room, None, None, [], {"seated": set()})
    assert got
    for q, mode in got[:10]:
        assert SP.in_room(q["position"], room)
        why, _ = SP.framing_why(q, "medium", "a", "b", mode == "ots", None, subj, S.RES,
                                opts={"seated": set()} if mode != "single" else {"seated": set(), "_single": True})
        assert not why, (mode, why)
    # two people face to face a step apart: no lens holds both faces; the search ends on a single of the speaker
    face = {"a": S.person([0.0, 0.5], facing=[0, -1]), "b": S.person([0.0, -0.4], facing=[0, 1])}
    got = SP.two_shot_candidates(c, {"camera": S.cam([1.5, 0.0, 1.5], [0, 0, 1.5], 35)}, "a", "b", face, S.RES, room,
                                 None, None, [], {"seated": set()})
    assert all(mode in ("ots", "single") for _, mode in got)
    # a seated listener: a clean single of the speaker is among the aims, never an over-the-shoulder
    sit = {"a": S.person([-0.5, 0.8], facing=[0.3, -1]), "b": S.person([0.9, -0.6], facing=[-1, 1], seated=True)}
    got = SP.two_shot_candidates(c, shot, "a", "b", sit, S.RES, room, None, None, [], {"seated": {"b"}})
    assert got and all(mode != "ots" for _, mode in got)
