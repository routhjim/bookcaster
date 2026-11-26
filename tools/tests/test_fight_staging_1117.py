"""Staging action between people (scenes/staging_pass.py, from_plan.py, speakers.py) and a work place's working pieces
(scenes/place_spec.py, sets/place_build.py), on synthetic data.

Whom a person faces is the pass's answer by name, over the plan's screen-direction words; a creature and whoever
fights it stand at striking distance by the creature's reach; a creature on an open stage is not pushed off it by
its reach; someone present whom the plan does not list for a shot is a framing fault when in the picture; a two
people facing each other are cheated open, a creature and its opponent are not; a working piece is its own kind
with its size; a piece that gives light is lit and a fire counts little toward a room's own light.
"""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "assets" / "sets"))
import staging_synth as S  # noqa: E402
import from_plan as FP  # noqa: E402
import speakers as SP  # noqa: E402
import staging_pass as PASS  # noqa: E402
import place_spec as PS  # noqa: E402
import place_build as PB  # noqa: E402
from test_creature_cast_1116 import add_creature  # noqa: E402


# ------------------------------------------------------------------ the pass's question
def test_the_face_question_names_the_shots_people_with_their_look_phrases_and_no_story_words():
    sh = {"action": "the man in the grey coat lunges at the beast", "keyshot": ""}
    people = [{"name": "Ash", "pose": "standing", "at": "on the left"}, {"name": "Beast", "pose": "standing", "at": "ahead"}]
    looks = {"Ash": "a man of about 30, wearing a grey coat", "Beast": "a huge beast with scales", "_creatures": ["Beast"]}
    q = PASS.q_face(sh, "Ash", people, looks)
    assert "the man in the grey coat" in q and "the beast" in q
    assert q.rstrip().endswith("Beast; " + PASS.AT_THING + "; " + PASS.NOBODY)   # the others by name, the thing, nobody
    assert "faces the one they act with" in q


# ------------------------------------------------------------------ striking distance
def test_a_fighter_too_near_a_creature_is_set_at_its_reach_on_the_side_he_was():
    q = FP.strike_place([0.0, 0.0], 4.0, [1.0, 0.2], [], [], [30.0, 30.0])
    d = math.dist(q, [0.0, 0.0])
    assert 4.0 + FP.PERSON_GAP_M - 1e-6 <= d <= 4.0 + FP.STRIKE_FAR_M + 1e-6
    assert q[0] > 0 and abs(math.atan2(q[1], q[0]) - math.atan2(0.2, 1.0)) < math.radians(15)


def test_a_fighter_far_off_comes_in_and_one_at_striking_distance_stays():
    far = FP.strike_place([0.0, 0.0], 4.0, [0.0, -12.0], [], [], [30.0, 30.0])
    assert far and math.dist(far, [0, 0]) <= 4.0 + FP.STRIKE_FAR_M + 1e-6 and far[1] < 0
    assert FP.strike_place([0.0, 0.0], 4.0, [0.0, -5.6], [], [], [30.0, 30.0]) is None


def test_striking_distance_keeps_clear_of_fixtures_and_others():
    fx = [{"name": "rock", "kind": "fixture", "center": [5.4, 0.0, 0.5], "half": [0.6, 0.6, 0.5]}]
    q = FP.strike_place([0.0, 0.0], 4.0, [1.0, 0.0], [[5.0, 1.5]], fx, [30.0, 30.0])
    assert q and FP.clear_of_fixtures(q, fx) and not FP.too_near(q, [[5.0, 1.5]])


def test_an_open_stage_does_not_shrink_by_a_creatures_reach():
    assert FP.bounds([13.0, 13.0], 5.2, walls=True)[0] == pytest.approx(0.95)
    assert FP.bounds([13.0, 13.0], 5.2, walls=False)[0] == pytest.approx(6.15)
    p = FP.free_mark([0.0, 0.0], [], [], [13.0, 13.0], (0.0, 1.0), own_r=5.2, walls=False)
    assert math.dist(p, [0, 0]) <= 1.0 + 1e-6                     # a free point near the target, not a fallback


# ------------------------------------------------------------------ from_plan: whom, and how far
@pytest.fixture
def lib(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [{"name": "desk", "position": [-8.0, -8.0], "size": [1.4, 0.7, 0.8]}],
                         size=(24.0, 24.0, 8.0))
    S.add_character(lib, "fig_a")
    S.add_character(lib, "fig_c")
    add_creature(lib, "beast_01")
    monkeypatch.setattr(FP.C, "LIBRARY", lib)
    return lib


def run(tmp_path, monkeypatch, shots, pass_):
    cast = {"Ash": {"key": "a", "id": "fig_a"}, "Beast": {"key": "b", "id": "creature/beast_01", "creature": True},
            "Cora": {"key": "c", "id": "fig_c"}}
    (tmp_path / "plan.json").write_text(json.dumps({sid: [sh] for sid, sh in shots.items()}))
    (tmp_path / "shots.txt").write_text("\n".join(shots))
    (tmp_path / "cast.json").write_text(json.dumps(cast))
    (tmp_path / "pass.json").write_text(json.dumps(pass_))
    monkeypatch.setattr(sys, "argv", ["from_plan.py", str(tmp_path / "plan.json"), str(tmp_path / "shots.txt"),
                                      str(tmp_path / "cast.json"), str(tmp_path / "out"), "--pass",
                                      str(tmp_path / "pass.json")])
    FP.main()
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "out").glob("pl_g*.json"))]


def shot(people, kind="action"):
    return {"kind": kind, "framing": "wide", "place": "a field", "room": "a field", "fixtures": [], "people": people}


def ang(f, frm, to):
    d = [to[0] - frm[0], to[1] - frm[1]]
    return math.degrees(math.acos(max(-1.0, min(1.0, (f[0] * d[0] + f[1] * d[1]) / math.hypot(*d) / math.hypot(*f)))))


def test_the_pass_says_whom_he_faces_and_the_creature_and_he_stand_its_reach_apart(lib, tmp_path, monkeypatch):
    s1 = shot([{"name": "Beast", "pose": "standing", "at": "in the middle", "facing": "left"},
               {"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "right"},
               {"name": "Cora", "pose": "standing", "at": "near the desk", "facing": "left"}])
    s2 = shot([{"name": "Beast", "pose": "standing", "at": "in the middle", "facing": "left"},
               {"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "right"}])
    specs = run(tmp_path, monkeypatch, {"action-1": s1, "action-2": s2},
                {"face": {"action-1": {"Ash": "Beast", "Beast": "Ash"}, "action-2": {"Ash": "Beast", "Beast": "Ash"}}})
    ch = specs[-1]["characters"]
    a, b = ch["a"], ch["b"]
    assert a["facing_to"] == "b" and b["facing_to"] == "a"
    assert ang(a["facing"], a["position"], b["position"]) < 3 and ang(b["facing"], b["position"], a["position"]) < 3
    r = 4.5                                                         # the test creature's reach (half of 9 m)
    assert r + FP.PERSON_GAP_M - 1e-6 <= math.dist(a["position"], b["position"]) <= r + FP.STRIKE_FAR_M + 1e-6
    cam = specs[-1]["cameras"][-1]
    assert cam["listed"] == ["a", "b"] and "c" in ch                 # Cora is there, not listed for this shot


def test_without_a_target_the_screen_words_still_turn_toward_the_others(lib, tmp_path, monkeypatch):
    s1 = shot([{"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "right"},
               {"name": "Cora", "pose": "standing", "at": "near the desk", "facing": "left"}], kind="quiet")
    specs = run(tmp_path, monkeypatch, {"quiet-1": s1}, {})
    a, c = specs[0]["characters"]["a"], specs[0]["characters"]["c"]
    assert a["facing_to"] is None and ang(a["facing"], a["position"], c["position"]) < 3


# ------------------------------------------------------------------ the camera: unlisted people out of the frame
def test_someone_the_plan_does_not_list_is_a_fault_in_the_picture_and_not_out_of_it():
    subj = {"a": S.person([0.0, 0.0]), "c": S.person([0.6, 0.0])}
    cam = S.front_cam(subj["a"], 3.0, focal=35)
    why, _ = SP.framing_why(cam, "medium", None, None, False, ["a"], subj, S.RES, opts={"listed": ["a"]})
    assert any("c (not in the shot's plan)" in w for w in why)
    why2, _ = SP.framing_why(cam, "medium", None, None, False, ["a"], subj, S.RES, opts={"listed": ["a", "c"]})
    assert not any("not in the shot's plan" in w for w in why2)
    far = {"a": subj["a"], "c": S.person([8.0, 0.0])}                # well outside the frame
    why3, _ = SP.framing_why(cam, "medium", None, None, False, ["a"], far, S.RES, opts={"listed": ["a"]})
    assert not any("not in the shot's plan" in w for w in why3)
    why4, _ = SP.framing_why(cam, "medium", None, None, False, ["a"], subj, S.RES, opts={})   # no list: no rule
    assert not any("not in the shot's plan" in w for w in why4)


def test_a_creature_and_its_opponent_square_up_and_two_people_are_cheated_open():
    spec = {"characters": {"a": {"position": [0.0, -6.0], "facing": [1, 0], "facing_to": "b"},
                           "b": {"position": [0.0, 0.0], "facing": [1, 0], "facing_to": "a", "creature": True}}}
    SP.aim_bodies(spec)
    assert spec["characters"]["a"]["facing"] == pytest.approx([0.0, 1.0], abs=1e-3)
    spec["characters"]["b"].pop("creature")
    spec["characters"]["a"]["facing"] = [1, 0]
    SP.aim_bodies(spec)
    assert ang(spec["characters"]["a"]["facing"], [0, -6], [0, 0]) == pytest.approx(SP.CHEAT_DEG, abs=1)


# ------------------------------------------------------------------ the work place
def spec_with(pieces, openings=()):
    return {"room": "A room. Its things.", "fixtures": ["a", "b"], "crowd": "", "schema": "place_spec/1", "place": "p",
            "kind": "shop", "period": "medieval", "size": "medium", "openings": list(openings), "pieces": pieces,
            "relations": [], "named_places": {"at the thing": "p1"}}


def test_a_working_piece_is_its_own_kind_and_needs_its_size():
    assert "work_piece" in PS.KINDS["PIECE_KINDS"]
    ok = spec_with([{"id": "p1", "kind": "work_piece", "look": "a heavy block", "phrases": ["at the thing"],
                     "against": "free", "longest_m": 0.8, "gives_light": True}])
    assert PS.validate(ok, ["at the thing"]) == []
    no_size = spec_with([dict(ok["pieces"][0], longest_m=None)])
    assert any("longest_m" in f for f in PS.validate(no_size, ["at the thing"]))
    bad_light = spec_with([dict(ok["pieces"][0], gives_light="yes")])
    assert any("gives_light" in f for f in PS.validate(bad_light, ["at the thing"]))
    p = PS.prompt("p", ["at the thing"], ["someone works at the thing"], "text")
    assert "work_piece" in p and "gives_light" in p


def test_the_prompt_rule_has_no_story_words():
    p = PS.prompt("p", [], [], "")
    for w in ("anvil", "bellows", "hearth", "hammer", "tongs"):
        assert w not in p.lower()


def test_a_piece_that_gives_light_is_lit_and_a_fire_counts_little_toward_the_room():
    kit = PB.kit_for("medieval")
    ss = {"structure": [], "fixtures": [{"asset": "x", "name": "work_piece", "position": [1.0, 0.5], "facing": [0, -1]}],
          "props": []}
    lay = {"size_m": [5.0, 5.0, 3.0], "openings": [], "pieces": [{"id": "p1", "name": "work_piece",
                                                                   "position": [1.0, 0.5], "size_m": [1.0, 0.8, 0.9]}]}
    out = PB.dress(ss, lay, {"pieces": [{"id": "p1", "gives_light": True}]}, kit)
    fire = [l for l in out["lights"] if l.get("position") and l["position"][:2] == [1.0, 0.5]]
    assert fire and PB.is_fire(fire[0]) and fire[0]["position"][2] > 0.9
    room = {"structure": [{"kind": "ceiling"}], "size_m": [4.0, 4.0, 3.0],
            "lights": [dict(kit["lights"]["fire"], position=[0, 0, 0.3]) for _ in range(3)]}
    assert 3 * kit["lights"]["fire"]["lumens"] >= 300.0 * 16 * 0.9   # as full light it would be near the old level
    assert PB.room_light(room, {})                                   # as a fire's, the room still gets its own light


# ------------------------------------------------------------------ the foreground
def test_a_thing_filling_the_foreground_is_a_fault_and_a_person_there_is_not(tmp_path):
    subj = {"a": S.person([0.0, 0.0])}
    cam = S.front_cam(subj["a"], 3.0, focal=50)
    g = S.depth_guides(tmp_path, "s", cam, subj)
    shot = {"camera": cam, "guides": g}
    why, _ = SP.framing_why(cam, "medium", None, None, False, ["a"], subj, S.RES, shot, {"subject": "a"})
    assert not any("foreground" in w for w in why)
    rows = [[50.0 if x < 48 else S.FAR_CM for x in range(144)] for _ in range(80)]     # a third of it, 0.5 m off
    S.write_exr(Path(g["depth"]), rows)
    why, _ = SP.framing_why(cam, "medium", None, None, False, ["a"], subj, S.RES, shot, {"subject": "a"})
    assert any("a thing in the foreground covers" in w for w in why)
    # the same near pixels drawn as a person's own (a listener's shoulder over which the camera looks): no fault
    S.write_exr(Path(g["depth_a"]), rows)
    assert SP.fg_thing_share(shot, ["a"], 300.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ the listener the plan's picture asks for
def test_a_talk_shot_whose_picture_has_the_listener_keeps_them(lib, tmp_path, monkeypatch):
    s0 = shot([{"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "Cora"},
               {"name": "Cora", "pose": "standing", "at": "near the desk", "facing": "Ash"}], kind="quiet")
    t1 = {"kind": "talk", "framing": "medium", "place": "a field", "speaker": "Ash", "listener": "Cora", "people": [],
          "picture": "Ash speaks, near the viewer; beyond him and smaller, Cora listens."}
    t2 = dict(t1, picture="Ash alone, his face filling the frame.")
    specs = run(tmp_path, monkeypatch, {"quiet-1": s0, "talk-1": t1, "talk-2": t2}, {})
    cams = {c["plan_id"]: c for sp in specs for c in sp["cameras"]}
    assert cams["talk-1"].get("keep_listener") and not cams["talk-2"].get("keep_listener")


# ------------------------------------------------------------------ a person the passage sent out, acting again
def test_someone_sent_out_is_back_when_a_later_shots_action_has_them_act(lib, tmp_path, monkeypatch):
    s1 = shot([{"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "Cora"},
               {"name": "Cora", "pose": "standing", "at": "near the desk", "facing": "Ash"}])
    s2 = dict(shot([{"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "Cora"},
                    {"name": "Cora", "pose": "standing", "at": "near the desk", "facing": "Ash"}]),
              kind="talk", speaker="Cora", action="")
    s3 = dict(s2, kind="action", speaker=None, action="Ash lifts the box while Cora looks on.")
    specs = run(tmp_path, monkeypatch, {"action-1": s1, "talk-2": s2, "action-3": s3},
                {"exits": {"action-1": ["Ash"]}})
    cams = {c["plan_id"]: (sp, c) for sp in specs for c in sp["cameras"]}
    sp2, c2 = cams["talk-2"]
    assert "a" not in sp2["characters"] and c2["subject"] == "c"
    sp3, c3 = cams["action-3"]
    assert "a" in sp3["characters"] and c3["subject"] == "a"


# ------------------------------------------------------------------ the set's things in the camera search
def test_a_camera_is_not_inside_a_thing_nor_behind_one():
    rock = {"name": "rock", "kind": "fixture", "center": [0.0, 3.0, 1.5], "half": [1.5, 1.0, 1.5]}
    SP.set_solids([rock])
    try:
        assert SP.inside_solid([0.0, 3.0, 1.6]) and not SP.inside_solid([0.0, -2.0, 1.6])
        assert not SP.in_room([0.0, 3.0, 1.6], [20.0, 20.0, 6.0]) and SP.in_room([0.0, -2.0, 1.6], [20.0, 20.0, 6.0])
        assert SP.solid_between([0.0, 6.0, 1.6], [0.0, 0.0, 1.6])          # the rock between camera and face
        assert not SP.solid_between([0.0, -4.0, 1.6], [0.0, 0.0, 1.6])
        assert not SP.solid_between([0.0, 6.0, 4.0], [0.0, 0.0, 3.5])      # over its top
    finally:
        SP.set_solids([])


def test_on_open_ground_a_camera_may_stand_past_the_stage_on_the_flat_ground(tmp_path):
    lib = tmp_path / "lib"
    (lib / "prop" / "land_01").mkdir(parents=True)
    (lib / "prop" / "land_01" / "asset.json").write_text(json.dumps({"world": {"stand_m": [1.0, -2.0, 14.0]}}))
    ss = {"size_m": [13.0, 13.0, 6.0], "world": {"flat_r_m": 9.0},
          "fixtures": [{"asset": "land_01", "world": True, "position": [-1.0, 2.0, -14.0]}]}
    og = SP.open_ground(ss, lib)
    assert og[:2] == pytest.approx([0.0, 0.0]) and og[2] == pytest.approx(8.5)
    assert SP.open_ground({"size_m": [4, 4, 3], "fixtures": []}, lib) == []
    room = [13.0, 13.0, 6.0]
    assert not SP.in_room([8.0, 0.0, 1.6], room) and not SP.in_room([1.0, 0.0, 7.0], room)
    SP.OPEN_GROUND[:] = og
    try:
        assert SP.in_room([8.0, 0.0, 1.6], room) and SP.in_room([1.0, 0.0, 7.0], room)
        assert not SP.in_room([8.0, 8.0, 1.6], room)                  # off the flat ground
    finally:
        SP.OPEN_GROUND[:] = []


def test_a_camera_is_not_set_inside_a_creatures_body(tmp_path):
    add_creature(tmp_path, "beast_01")                               # 6 x 9 x 5 m
    spec = {"characters": {"b": {"asset": "creature/beast_01", "creature": True, "position": [0.0, 0.0],
                                 "facing": [0.0, -1.0]}}}
    SP.set_solids([])
    try:
        SP.add_creatures(spec, tmp_path)
        assert SP.inside_solid([0.0, 2.0, 1.6]) and SP.inside_solid([1.0, -2.0, 1.6])
        assert not SP.inside_solid([0.0, -6.0, 1.6]) and not SP.inside_solid([4.0, 0.0, 1.6])
        assert SP.solid_between([0.0, 8.0, 1.6], [0.0, -6.0, 1.6])      # its body between camera and a man
    finally:
        SP.set_solids([])


def test_someone_working_at_a_named_thing_has_it_in_the_frame_of_an_action_shot(lib, tmp_path, monkeypatch):
    sd = next(p for p in (lib / "set").iterdir() if (p / "spec.json").exists())
    (sd / "named_places.json").write_text(json.dumps({"at the desk": "desk"}))
    s1 = shot([{"name": "Ash", "pose": "standing", "at": "at the desk", "facing": "down"}])
    s2 = dict(s1, framing="close")
    specs = run(tmp_path, monkeypatch, {"action-1": s1, "action-2": s2}, {})
    assert not any(c["at_thing"] for sp in specs for c in sp["cameras"])      # not turned to it: no rule
    specs = run(tmp_path, monkeypatch, {"action-1": s1, "action-2": s2},
                {"face": {"action-1": {"Ash": PASS.AT_THING}, "action-2": {"Ash": PASS.AT_THING}}})
    cams = {c["plan_id"]: c for sp in specs for c in sp["cameras"]}
    assert cams["action-1"]["at_thing"] == "desk" and not cams["action-2"]["at_thing"]


def test_someone_who_works_at_a_thing_faces_it_when_the_pass_says_so(lib, tmp_path, monkeypatch):
    sd = next(p for p in (lib / "set").iterdir() if (p / "spec.json").exists())
    (sd / "named_places.json").write_text(json.dumps({"at the desk": "desk"}))
    s1 = shot([{"name": "Ash", "pose": "standing", "at": "at the desk", "facing": "left toward Cora"},
               {"name": "Cora", "pose": "standing", "at": "in the middle", "facing": "Ash"}])
    specs = run(tmp_path, monkeypatch, {"action-1": s1}, {"face": {"action-1": {"Ash": PASS.AT_THING}}})
    a = specs[0]["characters"]["a"]
    assert FP.AT_THING == PASS.AT_THING and a["facing_to"] is None
    assert ang(a["facing"], a["position"], [-8.0, -8.0]) < 3           # toward the desk, not toward Cora
    q = PASS.q_face({"action": "Ash works at the desk."}, "Ash", s1["people"], {})
    assert q.rstrip().endswith("Cora; " + PASS.AT_THING + "; " + PASS.NOBODY)


def test_the_pictures_listener_wins_over_a_single_from_the_pass(lib, tmp_path, monkeypatch):
    s0 = shot([{"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "Cora"},
               {"name": "Cora", "pose": "standing", "at": "near the desk", "facing": "Ash"}], kind="quiet")
    t1 = {"kind": "talk", "framing": "medium", "place": "a field", "speaker": "Ash", "listener": "Cora", "people": [],
          "picture": "Ash speaks, near the viewer; beyond him and smaller, Cora listens."}
    specs = run(tmp_path, monkeypatch, {"quiet-1": s0, "talk-1": t1}, {"camera": {"talk-1": "single"}})
    c = [c for sp in specs for c in sp["cameras"] if c["plan_id"] == "talk-1"][0]
    assert c["keep_listener"] and c.get("listener") == "c" and not c.get("single")


def test_the_thing_they_are_at_is_offered_only_where_asked():
    ppl = [{"name": "Ash", "at": "on the open ground"}, {"name": "Beast", "at": "ahead"}]
    q = PASS.q_face({"action": "Ash fights the beast."}, "Ash", ppl, {}, False)
    assert PASS.AT_THING not in q and q.rstrip().endswith("Beast; " + PASS.NOBODY)


def test_an_action_shots_fighter_may_show_the_camera_his_back():
    assert not SP.face_rule_applies({"speaker": None, "shot_kind": "action"})
    assert SP.face_rule_applies({"speaker": None, "shot_kind": "quiet"})
    assert not SP.face_rule_applies({"speaker": "a", "shot_kind": "talk"})    # (a speaker has the speaker's rules)


def test_a_subject_squared_up_to_someone_in_the_shot_may_be_seen_from_behind():
    spec = {"characters": {"a": {"facing_to": "b"}, "b": {"facing_to": "a"}}}
    assert not SP.face_rule_applies({"shot_kind": "quiet", "line_subject": "a", "listed": ["a", "b"]}, spec)
    assert SP.face_rule_applies({"shot_kind": "quiet", "line_subject": "a", "listed": ["a"]}, spec)


def test_the_last_resort_camera_stands_clear_with_everyone_in_sight():
    rock = {"name": "rock", "kind": "fixture", "center": [0.0, -3.0, 1.5], "half": [1.5, 1.0, 1.5]}
    SP.set_solids([rock])
    try:
        ms = [S.person([0.0, 0.0])["measured"], S.person([1.0, 0.5])["measured"]]
        c = SP.fallback_camera(ms, [20.0, 20.0, 6.0])
        assert c and SP.in_room(c["position"], [20.0, 20.0, 6.0])
        assert not any(SP.solid_between(c["position"], [m["eye_xy"][0], m["eye_xy"][1], m["eye_z"]]) for m in ms)
    finally:
        SP.set_solids([])


def test_work_poses_are_offered_and_measured():
    assert set(PASS.WORK_POSES) <= set(PASS.POSE_WORDS) and set(PASS.WORK_POSES) == set(FP.WORK_ACTIONS)
    assert set(FP.WORK_ACTIONS) <= set(FP.ACTION_NAMES) and \
        set(SP.WORK_HANDS) == set(FP.WORK_ACTIONS) | set(FP.READY_ACTIONS)
    m = S.person([0.0, 0.0])["measured"]                 # hands hanging at the sides: neither raised nor at work
    assert SP.work_hands_why("tool_raised", m) and SP.work_hands_why("striking_down", m)
    m2 = json.loads(json.dumps(m))
    m2["joints"]["hand_r"] = [0.0, -0.5, 1.2]
    m2["joints"]["hand_l"] = [0.1, -0.45, 1.1]
    assert SP.work_hands_why("striking_down", m2) == []
    m2["joints"]["hand_r"] = [0.0, -0.1, 2.0]
    assert SP.work_hands_why("tool_raised", m2) == []
    assert FP.action_misfit("striking_down", "a", {"pose": "sitting"}, {}, None, []) == "a is sitting"


def test_someone_holding_a_tool_in_an_action_shot_has_the_hand_in_frame(lib, tmp_path, monkeypatch):
    s1 = shot([{"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "down", "holds": "a hammer"}])
    specs = run(tmp_path, monkeypatch, {"action-1": s1}, {"pose": {"action-1": {"Ash": "striking_down"}}})
    c = specs[0]["cameras"][0]
    assert c["show_joints"] == [["a", "hand_r"]]
    assert specs[0]["characters"]["a"]["body_pose"]["params"]["action"] == "striking_down"


def test_action_shots_in_a_row_of_the_same_people_take_other_sides(lib, tmp_path, monkeypatch):
    p = [{"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "down"}]
    specs = run(tmp_path, monkeypatch, {f"action-{i}": shot(p) for i in range(3)}, {})
    cams = [c for sp in specs for c in sp["cameras"]]
    angles = [(c.get("az", 0.0), c.get("el", 0.0)) for c in cams]
    assert len(set(angles)) == 3


def test_the_redraw_is_told_where_the_held_thing_is():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "redraw"))
    import redraw as R
    m = {"joints": {"hand_r": [0.0, 0.0, 0.8]}, "head_top_z": 1.8, "pelvis_xy": [0.0, 0.0], "facing": [0.0, -1.0],
         "waist_z": 1.0}
    assert "down at the side" in R.hand_place(m)
    m["joints"]["hand_r"] = [0.0, -0.4, 1.0]
    assert "in front" in R.hand_place(m)
    m["joints"]["hand_r"] = [0.0, -0.1, 1.95]
    assert "above the head" in R.hand_place(m)
    assert R.hand_place({}) == ""


def test_the_thing_worked_on_lies_on_the_piece_its_worker_stands_at(lib, tmp_path, monkeypatch):
    sd = next(p for p in (lib / "set").iterdir() if (p / "spec.json").exists())
    (sd / "named_places.json").write_text(json.dumps({"at the desk": "desk"}))
    s1 = shot([{"name": "Ash", "pose": "standing", "at": "at the desk", "facing": "down"}])
    (tmp_path / "carried.json").write_text(json.dumps({"a bar": "bar_01"}))
    cast = {"Ash": {"key": "a", "id": "fig_a"}}
    (tmp_path / "plan.json").write_text(json.dumps({"action-1": [s1]}))
    (tmp_path / "shots.txt").write_text("action-1")
    (tmp_path / "cast.json").write_text(json.dumps(cast))
    (tmp_path / "pass.json").write_text(json.dumps({"things": {"action-1": {"Ash": [
        {"what": "a bar", "on": "worked", "goes_on": None, "size_cm": 40}]}}}))
    monkeypatch.setattr(sys, "argv", ["from_plan.py", str(tmp_path / "plan.json"), str(tmp_path / "shots.txt"),
                                      str(tmp_path / "cast.json"), str(tmp_path / "out"), "--pass",
                                      str(tmp_path / "pass.json"), "--carried", str(tmp_path / "carried.json")])
    FP.main()
    c = json.loads(next((tmp_path / "out").glob("pl_g*.json")).read_text())["cameras"][0]
    t = c["carried"]["a"][0]
    assert t["asset"] == "bar_01" and t["at"] == "on" and t["xyz"][:2] == [-8.0, -8.0] and t["xyz"][2] > 0.5
    assert "worked" in PS.THING_ON


def test_a_fight_is_someone_with_a_thing_in_hand_squared_up_to_a_creature_in_an_action_shot():
    import from_plan as FP
    crt = {"a": {"radius_m": 5.0, "size_m": [4.0, 10.0, 4.0]}}
    cam = {"shot_kind": "action", "carried": {"d": [{"asset": "x", "at": "hand", "hand": "r"}]}}
    v = {"pose": "standing", "face_target": "a"}
    assert FP.fights("d", v, cam, crt)
    assert not FP.fights("d", v, dict(cam, shot_kind="quiet"), crt)
    assert not FP.fights("d", v, dict(cam, carried={}), crt)
    assert not FP.fights("d", dict(v, face_target="b"), cam, crt)
    assert not FP.fights("d", dict(v, pose="sitting"), cam, crt)
    assert not FP.fights("a", v, cam, crt)                          # (the creature itself)
    assert FP.FIGHT_POSE in FP.ACTION_NAMES


def test_a_fights_camera_stands_at_the_fighters_scale():
    import math
    import from_plan as FP
    rig = {"radius_m": 5.0, "size_m": [4.0, 10.0, 4.0]}
    p, c = [0.0, 0.0], [8.0, 0.0]
    over = FP.fight_camera(p, c, rig, over_shoulder=True)
    assert over["position"][0] < 0 and 2.0 < abs(over["position"][1]) < 3.5   # beside the fighter, a step behind:
    aim = [over["target"][0] - over["position"][0], over["target"][1] - over["position"][1]]     # they are out of it
    to_p = [p[0] - over["position"][0], p[1] - over["position"][1]]
    cosv = (aim[0] * to_p[0] + aim[1] * to_p[1]) / (math.hypot(*aim) * math.hypot(*to_p))
    assert math.degrees(math.acos(cosv)) > 35
    assert math.dist(over["position"][:2], p) < 3.0 and over["target"][0] > 4.0 and over["target"][2] >= 2.0
    side = FP.fight_camera(p, c, rig, over_shoulder=False, take=0)         # (the square one of the side-on three)
    assert abs(side["position"][1]) >= FP.FIGHT_SIDE_M and 0 < side["position"][0] < 8.0
    assert 0 < side["target"][0] < 8.0
    assert FP.fight_camera(p, [0.2, 0.0], rig, True) is None                # (on one spot: no line between them)


def test_once_squared_up_they_hold_their_ground_and_every_shot_sees_them_face_each_other(lib, tmp_path, monkeypatch):
    both = [{"name": "Beast", "pose": "standing", "at": "in the middle", "facing": "left"},
            {"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "right"}]
    s1 = shot(both)
    s2 = shot([dict(both[0])], kind="action")                       # what the creature does: only it is listed
    s3 = shot([dict(both[0]), dict(both[1], at="near the desk")], kind="quiet")   # the plan's words put him far off again
    s4 = shot([dict(both[1])], kind="quiet")
    s4["framing"] = "close"
    face = {"Ash": "Beast", "Beast": "Ash"}
    specs = run(tmp_path, monkeypatch, {"action-1": s1, "action-2": s2, "quiet-3": s3, "quiet-4": s4},
                {"face": {"action-1": face, "action-2": {"Beast": "Ash"}, "quiet-3": {}, "quiet-4": {}}})
    seen = {}
    for sp in specs:
        ch = sp["characters"]
        for c in sp["cameras"]:
            seen[c["plan_id"]] = (c, {k: list(v["position"][:2]) for k, v in ch.items()}, {k: v["facing"] for k, v in ch.items()})
    p0 = seen["action-1"][1]
    r = 4.5
    for sid, (c, pos, fac) in seen.items():
        # the same ground in every shot, each facing the other
        assert pos["a"] == pytest.approx(p0["a"], abs=0.05) and pos["b"] == pytest.approx(p0["b"], abs=0.05), sid
        assert math.dist(pos["a"], pos["b"]) <= r + FP.STRIKE_FAR_M + 1e-6
        assert ang(fac["a"], pos["a"], pos["b"]) < 3 and ang(fac["b"], pos["b"], pos["a"]) < 3, sid
    # both in a shot, or a shot of neither alone: side-on, the camera off the line between them, aimed between them
    for sid in ("action-1", "quiet-3"):
        c, pos, _ = seen[sid]
        assert c.get("fight") == {"person": "a", "creature": "b"}, sid
        u = [pos["b"][0] - pos["a"][0], pos["b"][1] - pos["a"][1]]
        w = [c["position"][0] - pos["a"][0], c["position"][1] - pos["a"][1]]
        assert abs(u[0] * w[1] - u[1] * w[0]) / math.hypot(*u) >= 0.6 * FP.FIGHT_SIDE_M, sid
    # the creature's own shot: over his shoulder at it
    c, pos, _ = seen["action-2"]
    assert c.get("fight") and math.dist(c["position"][:2], pos["a"]) < 3.0
    # a close shot of him alone keeps its own camera
    assert not seen["quiet-4"][0].get("fight")


def test_how_a_shot_of_a_standoff_is_seen():
    assert FP.standoff_view({"show": ["a", "b"]}, "a", "b") is False
    assert FP.standoff_view({"listed": ["b"]}, "a", "b") is True
    assert FP.standoff_view({"listed": ["a"], "kind": "medium"}, "a", "b") is False
    assert FP.standoff_view({"listed": ["a"], "kind": "close"}, "a", "b") is None
    assert FP.standoff_view({"listed": ["c"]}, "a", "b") is None
    assert FP.standoff_view({"listed": ["c", "b"]}, "a", "b") is None
    assert FP.standoff_view({"show": ["a", "b"], "shot_kind": "talk"}, "a", "b") is None


def test_who_stays_where_they_are():
    prev = {"position": [1.0, 2.0], "pose": "standing", "where": ["thing", "desk"]}
    st = {"b": {"position": [1.5, 2.5]}, "c": {"position": [9.0, 9.0]}}
    assert FP.stays(prev, "standing", None, st, {}, False)                      # words that name no place
    assert FP.stays(prev, "crouching", None, st, {}, False)                     # ... nor does a change of stance
    assert FP.stays(prev, "standing", ["thing", "desk"], st, {}, False)         # the same thing by other words
    assert not FP.stays(prev, "standing", ["thing", "door"], st, {}, False)     # another thing of the set
    assert FP.stays(prev, "standing", ["with", "b"], st, {}, False)             # beside someone they are with
    assert not FP.stays(prev, "standing", ["with", "c"], st, {}, False)         # ... and someone they are not
    assert not FP.stays(prev, "sitting", None, st, {}, False)                   # sitting down takes a seat
    assert FP.stays(prev, "lying", None, st, {}, True)                          # knocked flat: where they stood
    assert not FP.stays(dict(prev, pose="lying", bed="bed_1"), "standing", None, st, {}, True)   # up from a bed
    assert not FP.stays({"pose": "standing"}, "standing", None, st, {}, False)  # not placed yet


def test_words_that_name_no_place_move_nobody_and_another_thing_does(lib, tmp_path, monkeypatch):
    def one(at, pose="standing"):
        return shot([{"name": "Ash", "pose": pose, "at": at, "facing": "right"},
                     {"name": "Cora", "pose": "standing", "at": "in the middle", "facing": "left"}], kind="quiet")
    specs = run(tmp_path, monkeypatch, {"quiet-1": one("in the middle"), "quiet-2": one("lower-left foreground"),
                                        "quiet-3": one("in the fading light", "crouching"), "quiet-4": one("at the desk")}, {})
    at = {}
    for sp in specs:
        for c in sp["cameras"]:
            at[c["plan_id"]] = sp["characters"]["a"]["position"][:2]
    assert at["quiet-2"] == pytest.approx(at["quiet-1"], abs=0.02) and at["quiet-3"] == pytest.approx(at["quiet-1"], abs=0.02)
    assert math.dist(at["quiet-4"], [-8.0, -8.0]) < 2.5 and math.dist(at["quiet-4"], at["quiet-1"]) > 3.0


def test_in_the_open_a_fighter_stands_across_the_view_from_the_creature():
    # before it along the view (+y): moved beside it, on the side he was nearer, at striking distance
    q = FP.strike_place([0.0, 0.0], 4.0, [1.0, -5.6], [], [], [30.0, 30.0], walls=False, across=True)
    d = math.dist(q, [0.0, 0.0])
    assert q[0] > 0 and abs(q[1]) <= d * math.sin(math.radians(FP.STRIKE_ACROSS_DEG)) + 1e-6
    assert 4.0 + FP.PERSON_GAP_M - 1e-6 <= d <= 4.0 + FP.STRIKE_FAR_M + 1e-6
    # once there he stays, and a thing in the way on that side sends him to the other, still across
    assert FP.strike_place([0.0, 0.0], 4.0, q, [], [], [30.0, 30.0], walls=False, across=True) is None
    fx = [{"name": "rock", "kind": "fixture", "center": [6.0, 0.0, 2.0], "half": [3.0, 5.0, 2.0]}]
    o = FP.strike_place([0.0, 0.0], 4.0, [1.0, -5.6], [], fx, [30.0, 30.0], walls=False, across=True)
    assert o[0] < 0 and abs(o[1]) <= math.dist(o, [0, 0]) * math.sin(math.radians(FP.STRIKE_ACROSS_DEG)) + 1e-6
    # the side-on camera on the two then looks the way the picture does
    c = FP.fight_camera(q, [0.0, 0.0], {"radius_m": 4.0, "size_m": [8, 4, 4]}, False)
    assert c["position"][1] < min(q[1], 0.0) and c["target"][1] > c["position"][1]


def test_side_on_shots_of_one_fight_are_three_cameras_on_one_side_of_the_line():
    p, c, rig = [7.0, 0.0], [0.0, 0.0], {"radius_m": 4.0, "size_m": [8, 4, 4]}
    cams = [FP.fight_camera(p, c, rig, False, take=i) for i in range(4)]
    assert all("fight_view" in x for x in cams)
    assert len({tuple(x["position"]) for x in cams[:3]}) == 3 and cams[3]["position"] == cams[0]["position"]
    assert all(x["position"][1] < 0 for x in cams)                        # one side of the line, the place's near side
    wide, by_fighter, far = cams[:3]
    assert math.dist(by_fighter["position"][:2], p) < math.dist(far["position"][:2], p)
    for x in cams[:3]:                                    # none from round toward the creature's back: no nearer it
        assert math.dist(x["position"][:2], c) >= math.dist(x["position"][:2], p) - 1e-6      # than the fighter
    assert math.dist(far["position"][:2], wide["target"][:2]) > math.dist(wide["position"][:2], wide["target"][:2])
    for x in cams[:3]:                                    # every one far enough off to hold the two whole
        assert min(math.dist(x["position"][:2], p), math.dist(x["position"][:2], c)) >= 0.6 * FP.FIGHT_SIDE_M
        assert x["target"] == wide["target"]
    assert FP.fight_camera(p, c, rig, True)["position"][1] < 0            # the shoulder camera on that side too


# ------------------------------------------------------------------ a work shot's camera
def test_a_work_shot_is_framed_on_the_work_and_its_shots_are_not_one_camera():
    piece = {"name": "block", "kind": "fixture", "center": [0.0, 0.0, 0.4], "half": [0.3, 0.2, 0.4]}
    w = [0.0, 0.8]                                           # the worker, north of the piece, turned to it
    cams = [FP.work_camera(w, piece, "medium", [], [piece], [8.0, 8.0], take=i) for i in range(3)]
    assert all(cams) and len({tuple(c["position"]) for c in cams}) == 3
    over, close, side = cams
    assert over["position"][1] < 0 and close["position"][1] < 0          # from beyond the piece, looking back at them
    assert math.dist(close["position"][:2], [0, 0]) < math.dist(over["position"][:2], [0, 0])
    assert 0 < close["target"][1] < 0.4 and close["focal_mm"] == over["focal_mm"]   # the close one leans to the work,
    #                                                                       the worker still in it
    assert abs(side["position"][0]) > 2 * abs(side["position"][1])       # the side one: in profile
    assert all(c_["position"][1] <= 0.05 for i in range(6)                # none from behind the worker's shoulder
               for c_ in [FP.work_camera(w, piece, "medium", [], [piece], [8.0, 8.0], take=i)])
    near = [cams[0]["position"][0] * 0.4 + 0.5, cams[0]["position"][1] * 0.4]     # someone in the frame's foreground
    assert FP.work_camera(w, piece, "medium", [near], [piece], [8.0, 8.0], take=0)["position"] != cams[0]["position"]
    # someone standing in the first choice's line sends the camera to the other side; a wall does too
    o = [cams[0]["position"][0] * 0.5, cams[0]["position"][1] * 0.5]
    other = FP.work_camera(w, piece, "medium", [o], [piece], [8.0, 8.0], take=0)
    assert other and (other["position"][0] > 0) != (cams[0]["position"][0] > 0)
    assert FP.work_camera(w, piece, "medium", [], [piece], [1.0, 1.0], take=0) is None     # no room: the shot's own
    # another piece between the camera and the work: from the other side, or nearer
    hearth = {"name": "h", "kind": "fixture", "center": [cams[0]["position"][0] * 0.55, cams[0]["position"][1] * 0.55, 0.4],
              "half": [0.5, 0.5, 0.4]}
    clear = FP.work_camera(w, piece, "medium", [], [piece, hearth], [8.0, 8.0], take=0)
    assert clear and clear["position"] != cams[0]["position"]
    # a tall piece right beside the work hides it as well: no camera looks across it at the work or the worker
    tall = {"name": "t", "kind": "fixture", "center": [0.9, 0.3, 0.9], "half": [0.5, 0.6, 0.9]}
    for i in range(6):
        cam = FP.work_camera(w, piece, "medium", [], [piece, tall], [8.0, 8.0], take=i)
        assert cam and cam["position"][0] < 0.3, (i, cam)
    low = dict(tall, center=[0.9, 0.3, 0.2], half=[0.5, 0.6, 0.2])       # a low one beside it is looked over
    assert FP.work_camera(w, piece, "medium", [], [piece, low], [8.0, 8.0], take=0)["position"] == cams[0]["position"]
    # a full framing stands further off, whatever the take
    assert math.dist(FP.work_camera(w, piece, "full", [], [piece], [12.0, 12.0], take=1)["position"][:2], [0, 0]) > \
        math.dist(over["position"][:2], [0, 0])


def test_someone_looking_on_at_two_who_face_each_other_stands_apart():
    ppl = [{"name": "A", "facing": "B"}, {"name": "B", "facing": "A"}, {"name": "C", "facing": "B"}]
    assert FP.looks_on(ppl[2], ppl) and not FP.looks_on(ppl[0], ppl) and not FP.looks_on(ppl[1], ppl)
    ring = [{"name": "A", "facing": "B"}, {"name": "B", "facing": "C"}, {"name": "C", "facing": "A"}]
    assert not any(FP.looks_on(q, ring) for q in ring)                   # three in one talk: nobody stands apart
    assert not FP.looks_on({"name": "C", "facing": "the window"}, ppl[:2] + [{"name": "C", "facing": "the window"}])
    assert FP.ONLOOK_M[0] > 3 * FP.PERSON_GAP_M


def test_two_on_the_open_floor_who_face_each_other_stand_at_talking_distance():
    assert FP.turned_to("Orla", {"Tam": "Orla"}, ["Tam"]) and FP.turned_to("Tam", {"Tam": "Orla"}, ["Orla"])
    assert not FP.turned_to("Orla", {"Tam": "the thing they are at"}, ["Tam"]) and not FP.turned_to("Orla", {}, ["Tam"])
    assert not FP.turned_to("Orla", {"Orla": "Dr. Marsh"}, ["Tam"])           # faces someone not in the shot
    assert FP.floor_near({"with": True, "turned": True}) == FP.TALK_M and FP.TALK_M[0] >= FP.PERSON_GAP_M
    assert FP.TALK_M[1] < 2.0 <= FP.floor_near({"with": True})[0]             # merely in the same place: a few steps off
    assert FP.floor_near({"with": True, "turned": True, "apart": True}) == FP.ONLOOK_M
    assert FP.floor_near({}) == (0.0, 1.6)


def test_bystander_steps_to_the_side_of_the_work():
    """Someone standing across a piece from the one who works at it stands where the work is seen from: they step
    to the piece's side, and a camera then has its place on the work."""
    import from_plan as FP
    thing = {"name": "block", "kind": "fixture", "center": [0.0, 0.0, 0.35], "half": [0.3, 0.2, 0.35]}
    room = [6.0, 6.0, 3.0]
    spot = [0.0, -0.7]
    by = {"b": {"position": [0.1, 0.6], "pose": "standing"}}
    side = FP.stand_aside(spot, thing, by, [thing], room)
    assert "b" in side
    m = side["b"]
    assert FP.WORK_ASIDE_M[0] - 0.01 <= math.dist(m, [0.0, 0.0]) <= FP.WORK_ASIDE_M[1] + 0.01
    assert abs(m[0]) > 1.2 and m[1] < 0.5, m                  # at its side, not on past it
    for take in range(3):
        assert FP.work_camera(spot, thing, "medium", [m], [thing], room, take=take), take
    # one already at the side, and one far off, stay
    assert FP.stand_aside(spot, thing, {"b": {"position": [1.8, -0.2], "pose": "standing"}}, [thing], room) == {}
    assert FP.stand_aside(spot, thing, {"b": {"position": [0.0, 2.9], "pose": "standing"}}, [thing], room) == {}
