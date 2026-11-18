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
    assert 3 * kit["lights"]["fire"]["lumens"] >= PB.ROOM_LM_M2 * 16 * 0.9   # as full light it would be near enough
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
    assert set(FP.WORK_ACTIONS) <= set(FP.ACTION_NAMES) and set(SP.WORK_HANDS) == set(FP.WORK_ACTIONS)
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
