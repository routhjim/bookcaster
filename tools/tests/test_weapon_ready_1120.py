"""A fighter's start frame is a guard (scenes/ue/pose.py weapon_ready; from_plan FIGHT_POSE), on synthetic data.

The weapon hand ahead of the chest between hip and chest height, the held thing pointed along the line to the target
(never back over the shoulder), both hands on a long thing, the free hand of a short one forward, the stance's limits
(feet planted apart, knees bent, elbows bent the natural way, wrists within WRIST_MAX, centre of mass over the feet);
from_plan gives a fighter the guard (also over a blow the pass names), aimed at the foe and not moved to a work
piece; a sitting person or one with nothing in the hand never gets it; the redraw says "at the ready", not "raised".
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
from test_work_pose_1117 import PO, humanoid  # noqa: E402
from test_fight_staging_1117 import lib, shot  # noqa: E402,F401

SHORT = [0.05, 0.2, 0.85]        # an upright one-handed thing (m), under TWO_HANDED_M
LONG = [0.06, 0.25, 1.3]         # a two-handed one
TARGETS = (None, [0.0, 600.0, 300.0], [150.0, 500.0, 250.0])
CASES = [(sz, tg, hd) for sz in (SHORT, LONG) for tg in TARGETS for hd in ("r", "l")]


def ready(size, target=None, hand="r"):
    P, par = humanoid()
    s = PO.Skel.from_joints(P, par)
    pr = {"action": "weapon_ready", "hand": hand, "held": [{"asset": "w", "hand": hand, "size_m": size}]}
    if target:
        pr["aim_target_ue"] = target
    return PO.pose_skel(s, "stand", pr)


def line_to(s, target):
    palm = s.info["tool"]["palm_cm"]
    if target:
        return PO.unit(PO.sub(target, palm))
    a = math.radians(PO.READY_UP_DEG)
    return [0.0, math.cos(a), math.sin(a)]


@pytest.fixture(scope="module")
def posed():
    return {(tuple(sz), tuple(tg or ()), hd): ready(sz, tg, hd) for sz, tg, hd in CASES}


def get(posed, sz, tg, hd):
    return posed[(tuple(sz), tuple(tg or ()), hd)]


@pytest.mark.parametrize("sz,tg,hd", CASES)
def test_the_weapon_hand_is_ahead_of_the_chest_between_hip_and_chest(posed, sz, tg, hd):
    s = get(posed, sz, tg, hd)
    palm = s.info["tool"]["palm_cm"]
    F = [0.0, 1.0] if not tg else PO.unit([tg[0] - s.pos["pelvis"][0], tg[1] - s.pos["pelvis"][1], 0.0])[:2]
    ahead = lambda p: p[0] * F[0] + p[1] * F[1]                    # noqa: E731
    assert ahead(palm) - ahead(s.pos["spine_05"]) > 0.12 * 175, palm
    assert s.pos["pelvis"][2] < palm[2] < s.pos["spine_05"][2], palm


@pytest.mark.parametrize("sz,tg,hd", CASES)
def test_the_weapon_points_at_the_target_and_never_back(posed, sz, tg, hd):
    s = get(posed, sz, tg, hd)
    d = s.info["tool"]["dir"]
    assert PO.angle_deg(d, line_to(s, tg)) <= 25.0, d
    assert d[1] > 0.5 and d[2] > 0.0                                # forward and up, not over the shoulder
    assert s.info["tool"]["palm_cm"][2] < s.pos["head"][2]            # (and the hand is not up by the head)


@pytest.mark.parametrize("sz,tg,hd", CASES)
def test_both_hands_on_a_long_thing_and_the_free_hand_forward_on_a_short_one(posed, sz, tg, hd):
    s = get(posed, sz, tg, hd)
    t = s.info["tool"]
    ot = "l" if hd == "r" else "r"
    other = PO.palm_point(s, ot)
    if sz is LONG:
        assert t["grip"]["two_hands"]
        a = t["palm_cm"]
        b = PO.add(a, PO.scale(t["dir"], -100.0 * sz[2]))
        assert PO.seg_dist(other, a, b) < 3.0, other               # on the handle, behind the leading hand
    else:
        assert not t["grip"]["two_hands"]
        assert other[1] - s.pos["spine_05"][1] > 0.12 * 175          # forward, not down at the side
        assert s.pos["pelvis"][2] < other[2] < s.pos["neck_01"][2]
        sx = 1.0 if ot == "l" else -1.0
        assert sx * other[0] > 0.0                                   # out on its own side


@pytest.mark.parametrize("sz,tg,hd", CASES)
def test_the_stance_keeps_the_limits(posed, sz, tg, hd):
    s = get(posed, sz, tg, hd)
    w = s.info["work"]
    assert w["action"] == "weapon_ready" and w["standing"]
    assert w["balance_margin_cm"] >= PO.BALANCE_MARGIN_CM - 0.5
    assert max(w["reach"].values()) <= PO.REACH + 1e-6
    for b in ("foot_l", "foot_r", "ball_l", "ball_r"):
        assert abs(s.pos[b][2] - s.ref_pos[b][2]) < 1.0             # planted
    ot = "l" if hd == "r" else "r"
    assert s.pos[f"foot_{ot}"][1] - s.pos[f"foot_{hd}"][1] > 0.12 * 175   # the lead foot forward, feet apart
    assert abs(s.pos["foot_l"][0] - s.pos["foot_r"][0]) > 28.0
    assert s.pos["pelvis"][2] < s.ref_pos["pelvis"][2] - 3.0         # knees soft
    left, _ = PO._body_axes(s)
    for side, sx in (("l", 1.0), ("r", -1.0)):
        sh, el, hd_ = s.pos[f"upperarm_{side}"], s.pos[f"lowerarm_{side}"], s.pos[f"hand_{side}"]
        assert 25.0 <= PO.angle_deg(PO.sub(sh, el), PO.sub(hd_, el)) <= 165.0
        off = PO.perp(PO.sub(el, sh), PO.unit(PO.sub(hd_, sh)))
        assert off[2] < 1.0 and PO.dot(off, PO.scale(left, sx)) > -1.0, (side, off)   # down and out
        assert PO.angle_deg(PO.sub(hd_, el), PO.sub(s.pos[f"middle_01_{side}"], hd_)) <= PO.WRIST_MAX + 0.5
    # the head on the target
    look = tg or PO.add(s.pos["head"], [0.0, 500.0, 0.0])
    assert PO.angle_deg(PO.face_forward(s), PO.sub(look, s.pos["head"])) < 20.0


def test_the_held_report_is_a_grip_with_no_blow_to_measure():
    s = ready(SHORT, [0.0, 600.0, 300.0])
    h = s.info["held"]["w"]
    assert h["how"] == "work" and "target_cm" not in h and not h.get("raised")
    assert PO.blow_contact_why("weapon_ready", True, [], "the thing") == []
    put = PO.prop_in_hand(SHORT, h["palm_cm"], h["dir"], h["face"], grip=h["grip"])
    assert PO.held_contact(put, h["palm_cm"])["why"] == []


def test_a_guard_ignores_a_work_top_and_a_worked_thing():
    a = ready(SHORT)
    P, par = humanoid()
    b = PO.Skel.from_joints(P, par)
    PO.pose_skel(b, "stand", {"action": "weapon_ready", "work_top_cm": 45.0, "work_target_ue": [0.0, 40.0, 50.0],
                              "held": [{"asset": "w", "hand": "r", "size_m": SHORT}]})
    assert b.info["tool"]["palm_cm"] == a.info["tool"]["palm_cm"]


def test_the_measured_guard_has_both_hands_in_front():
    m = S.person([0.0, 0.0])["measured"]                # hands hanging at the sides
    assert SP.work_hands_why("weapon_ready", m)
    m2 = json.loads(json.dumps(m))
    m2["joints"]["hand_r"] = [0.0, -0.4, 1.1]
    m2["joints"]["hand_l"] = [0.1, -0.35, 1.15]
    assert SP.work_hands_why("weapon_ready", m2) == []


# ------------------------------------------------------------------ from_plan
def test_a_sitter_or_an_empty_hand_never_gets_the_guard():
    hand = [{"asset": "x", "at": "hand", "hand": "r"}]
    assert FP.action_misfit("weapon_ready", "a", {"pose": "sitting"}, {}, None, [], hand) == "a is sitting"
    assert FP.action_misfit("weapon_ready", "a", {"pose": "standing"}, {}, None, [], []) == "a has nothing in the hand"
    assert FP.action_misfit("weapon_ready", "a", {"pose": "standing"}, {}, None, [],
                            [{"asset": "y", "at": "head"}]) == "a has nothing in the hand"
    assert FP.action_misfit("weapon_ready", "a", {"pose": "standing"}, {}, None, [], hand) is None
    assert FP.FIGHT_POSE == "weapon_ready" and FP.FIGHT_POSE in FP.ACTION_NAMES
    assert FP.FIGHT_POSE not in FP.WORK_ACTIONS                      # (a fighter is not placed at a work piece)


def test_the_guard_points_at_the_near_part_of_the_foe():
    crt = {"b": {"radius_m": 4.0, "size_m": [4.0, 9.0, 5.0]}}
    st = {"b": {"position": [10.0, 0.0]}}
    aim = FP.foe_aim({"position": [0.0, 0.0], "face_target": "b"}, st, crt)
    assert aim == pytest.approx([8.0, 0.0, 3.0])
    assert FP.foe_aim({"position": [0.0, 0.0], "face_target": "c"}, st, crt) is None


def run_fight(tmp_path, monkeypatch, pose=None):
    cast = {"Ash": {"key": "a", "id": "fig_a"}, "Beast": {"key": "b", "id": "creature/beast_01", "creature": True}}
    s1 = shot([{"name": "Beast", "pose": "standing", "at": "in the middle", "facing": "left"},
               {"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "right", "holds": "a blade"}])
    pass_ = {"face": {"action-1": {"Ash": "Beast", "Beast": "Ash"}},
             "things": {"action-1": {"Ash": [{"what": "a blade", "on": "held", "goes_on": None, "size_cm": 90}]}}}
    if pose:
        pass_["pose"] = {"action-1": {"Ash": pose}}
    (tmp_path / "plan.json").write_text(json.dumps({"action-1": [s1]}))
    (tmp_path / "shots.txt").write_text("action-1")
    (tmp_path / "cast.json").write_text(json.dumps(cast))
    (tmp_path / "pass.json").write_text(json.dumps(pass_))
    (tmp_path / "carried.json").write_text(json.dumps({"a blade": "blade_01"}))
    monkeypatch.setattr(sys, "argv", ["from_plan.py", str(tmp_path / "plan.json"), str(tmp_path / "shots.txt"),
                                      str(tmp_path / "cast.json"), str(tmp_path / "out"), "--pass",
                                      str(tmp_path / "pass.json"), "--carried", str(tmp_path / "carried.json")])
    FP.main()
    return json.loads(next((tmp_path / "out").glob("pl_g*.json")).read_text())


@pytest.mark.parametrize("pose", [None, "tool_raised", "striking_down"])
def test_a_fighter_takes_the_guard_at_the_foe_and_is_not_sent_to_a_work_piece(lib, tmp_path, monkeypatch, pose):
    sp = run_fight(tmp_path, monkeypatch, pose)
    a, b = sp["characters"]["a"], sp["characters"]["b"]
    pr = a["body_pose"]["params"]
    assert pr["action"] == "weapon_ready" and a["_pose_choice"] == "weapon_ready"
    assert "work_at" not in a and "work_top_cm" not in pr and a.get("facing_to") == "b"
    r = 4.5                                                         # the test creature's reach: the fight placed him
    assert r + FP.PERSON_GAP_M - 1e-6 <= math.dist(a["position"], b["position"]) <= r + FP.STRIKE_FAR_M + 1e-6
    aim = pr["aim_at_m"]                                            # at the creature's near part, head to middle
    assert math.dist(aim[:2], b["position"][:2]) < math.dist(a["position"][:2], b["position"][:2]) and aim[2] > 1.5
    import scene
    assert scene.aim_params(pr)["aim_target_ue"] == [round(aim[0] * 100, 2), round(-aim[1] * 100, 2),
                                                     round(aim[2] * 100, 2)]
    assert scene.aim_params(dict(pr, action="tool_raised")).get("aim_target_ue") is None


# ------------------------------------------------------------------ the redraw's words
def test_the_redraw_says_the_guard_holds_it_at_the_ready_not_raised(tmp_path):
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "redraw"))
    import redraw as R
    sd = tmp_path / "sd"
    sd.mkdir()
    spec = {"cameras": [{"name": "close_1", "holds": {"a": "a long blade"}}],
            "characters": {"a": {"facing_to": "b", "body_pose": {"pose": "stand",
                                                                 "params": {"action": "weapon_ready"}}},
                           "b": {"asset": "creature/x"}}}
    (sd / "spec_round1.json").write_text(json.dumps(spec))
    hand = {"hand_r": [0.0, -0.35, 1.1]}
    (sd / "result.json").write_text(json.dumps({"subjects": {"a": {
        "measured": {"joints": hand, "head_top_z": 1.8, "pelvis_xy": [0.0, 0.0], "facing": [0.0, -1.0]},
        "held": {"blade_01": {"grip": {"two_hands": True}}}}}}))
    cast = {"a": {"noun": "man", "pos": "his"}, "b": {"noun": "creature", "creature": True}}
    ppl = [("a", 0.5, 10.0)]
    w = R.holds_words(sd, "close_1", ppl, cast, "close")          # said in a close shot too
    assert "at the ready in front of his body" in w and "in both hands" in w and "at the creature" in w
    assert "raised" not in w
    spec["characters"]["a"]["body_pose"]["params"]["action"] = "tool_raised"
    (sd / "spec_round1.json").write_text(json.dumps(spec))
    assert R.holds_words(sd, "close_1", ppl, cast, "close") == ""  # (any other pose: not in a close shot)
