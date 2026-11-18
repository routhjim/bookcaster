"""Held things are gripped in every pose and a blow lands on the worked thing (scenes/ue/pose.py), on the synthetic
1.75 m humanoid of test_work_pose_1117.

11-18: a hammer hung under an open hand; its face a hand's breadth short of the metal and turned sideways, and a
second held thing shared the first one's palm. Now: (1) a held thing outside the work actions is gripped (palm on
its handle's line, carried clear of the body); (2) two held things take two hands; (3) a blow puts a head's face on
the worked thing's top, flat; (4) a tipped thing's tip; (5) the staged check (pose.held_contact) flags a wrong
placement in words, and speakers.staged_facts passes them on. Plus the head / tip reading from the model.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_work_pose_1117 as W  # noqa: E402

PO = W.PO
HAMMER = W.HAMMER                      # upright one-handed tool: handle along z, a 12 cm head
TONGS = [0.05, 0.03, 0.5]              # a long thing with no head: its far end is a tip
POKER = [0.75, 0.09, 0.02]             # lying along x, a hooked end too narrow for a head


def skel():
    P, par = W.humanoid()
    return PO.Skel.from_joints(P, par)


def held(asset, size, hand="r", name=None, grip=None):
    return {"asset": asset, "hand": hand, "size_m": size, "name": name or asset, "grip": grip}


def placed(s, rep):
    """The prop placed as scene_job places it (prop_in_hand with the pose's palm, direction and face), and the palm."""
    put = PO.prop_in_hand(None, rep["palm_cm"], rep["dir"], rep["face"], grip=rep["grip"])
    return put, PO.palm_point(s, rep["hand"])


def far_end(put):
    g = put["grip"]
    cols = (put["x"], put["y"], put["z"])
    gp = PO.mesh_point(put, g["grip_m"])
    return PO.add(gp, PO.scale(cols[g["axis"]], g.get("sign", 1.0) * 100.0 * (g["len_m"] - g["grip_from_end_m"])))


def in_trunk(s, p, r):
    a, b = s.pos["pelvis"], s.pos["spine_05"]
    return PO.seg_dist(p, a, b) < r


# ------------------------------------------------------------------ (1) gripped outside the work actions
@pytest.mark.parametrize("size,name", [(HAMMER, "hammer"), (TONGS, "tongs")])
def test_a_thing_held_at_the_side_is_gripped_and_carried_clear(size, name):
    s = PO.pose_skel(skel(), "stand", {"held": [held(name, size)]})
    rep = s.info["held"][name]
    assert rep["hand"] == "r" and rep["how"] == "hold"
    put, palm = placed(s, rep)
    c = PO.held_contact(put, palm)
    assert c["grip_cm"] <= 2.0 and c["why"] == [], c
    t = rep["dir"]
    left, fwd = PO._body_axes(s)
    assert PO.dot(t, fwd) > 0.3                                  # forward, not back into the leg
    assert PO.dot(t, left) < 0.1                                 # not across into the body (right hand: left is in)
    tip = far_end(put)
    assert not in_trunk(s, tip, 0.075 * 175)
    for x in ("l", "r"):
        assert PO.seg_dist(tip, s.pos[f"thigh_{x}"], s.pos[f"calf_{x}"]) > 0.05 * 175
    if rep["work"] == "head":
        assert t[2] < -0.3                                       # a tool hangs head-down-forward
    else:
        assert -0.6 < t[2] < 0.0                                 # a long thing points forward and a little down
    # the hand closed on it: the handle across the palm, the fingers curled by its thickness
    hd, m = s.pos["hand_r"], s.pos["middle_01_r"]
    assert abs(PO.dot(PO.unit(PO.sub(m, hd)), t)) < 0.1
    assert PO.CURL_RANGE[0] <= rep["curl_deg"] <= PO.CURL_RANGE[1]
    w = PO.world_info(s)["held"][name]                           # what scene_job reads to place it
    assert all(k in w for k in ("palm_ue", "hand_ue", "dir_ue", "face_ue"))


def test_finger_curl_follows_the_handle_thickness():
    thin = PO.grip_curl({"handle_m": 0.02})
    mid = PO.grip_curl({"handle_m": 0.03})
    thick = PO.grip_curl({"handle_m": 0.05})
    assert thin > mid > thick and mid == pytest.approx(70.0)
    assert PO.grip_curl({"handle_m": 0.2}) == PO.CURL_RANGE[0] and PO.grip_curl({"handle_m": 0.001}) == PO.CURL_RANGE[1]


def test_a_raised_arm_holds_the_handle_across_the_forearm_working_end_up():
    s = PO.pose_skel(skel(), "gesture_point", {"held": [held("hammer", HAMMER)]})
    rep = s.info["held"]["hammer"]
    f = PO.unit(PO.sub(s.pos["hand_r"], s.pos["lowerarm_r"]))
    assert abs(PO.dot(f, rep["dir"])) < 0.2                      # roughly perpendicular to the forearm
    assert rep["dir"][2] > 0.2                                   # working end up
    put, palm = placed(s, rep)
    assert PO.held_contact(put, palm)["grip_cm"] <= 2.0


def test_the_lapel_hand_lets_go_of_the_lapel_to_carry():
    s = PO.pose_skel(skel(), "stand_relaxed", {"held": [held("hammer", HAMMER), held("tongs", TONGS, "l")]})
    assert s.pos["hand_l"][2] < s.pos["pelvis"][2] + 5.0         # at the side, not up at the chest
    assert s.info["held"]["tongs"]["hand"] == "l"


# ------------------------------------------------------------------ (2) two things, two hands
def test_two_held_things_take_two_hands():
    s = PO.pose_skel(skel(), "stand", {"held": [held("hammer", HAMMER, "r"), held("tongs", TONGS, "r")]})
    h = s.info["held"]
    assert {h["hammer"]["hand"], h["tongs"]["hand"]} == {"l", "r"}
    for rep in h.values():
        put, palm = placed(s, rep)
        assert PO.held_contact(put, palm)["grip_cm"] <= 2.0
    # at the blow the hand that strikes takes the thing with a head, the other hand the other thing
    s = PO.pose_skel(skel(), "stand", {"action": "striking_down", "work_top_cm": 85.0, "work_fwd_cm": 58.0,
                                       "work_near_cm": 38.0,
                                       "held": [held("tongs", TONGS, "r"), held("hammer", HAMMER, "l")]})
    h = s.info["held"]
    assert h["hammer"]["hand"] == "r" and h["tongs"]["hand"] == "l"
    assert s.info["tool"]["asset"] == "hammer"
    for rep in h.values():
        put, palm = placed(s, rep)
        assert PO.held_contact(put, palm)["grip_cm"] <= 2.0


# ------------------------------------------------------------------ (3) a blow lands on the worked thing
def strike(size, name):
    # the worked thing lies 0.75 m ahead on a top 0.9 m up and is 5 cm thick (scene.py: its top = where it lies +
    # its thinnest side, in UE cm; off the engine the actor frame is the UE frame)
    top = [0.0, 75.0, 90.0 + 5.0]
    s = PO.pose_skel(skel(), "stand", {"action": "striking_down", "work_top_cm": 90.0, "work_fwd_cm": 75.0,
                                       "work_near_cm": 38.0, "work_target_ue": top,
                                       "held": [held(name, size, name=f"the {name}")]})
    return s, top


def test_a_hammer_face_lands_flat_on_the_worked_things_top():
    s, top = strike(HAMMER, "hammer")
    rep = s.info["held"]["hammer"]
    assert rep["work"] == "head"
    put, palm = placed(s, rep)
    c = PO.held_contact(put, palm, top, [0.0, 0.0, 1.0], "the hammer")
    assert c["grip_cm"] <= 2.0 and c["gap_cm"] <= 2.0 and c["angle_deg"] <= 15.0, c
    assert c["why"] == []
    assert s.info["tool"]["gap_cm"] <= 2.0 and s.info["tool"]["face_angle_deg"] <= 15.0
    # the wrist still within its limit
    hd, el = s.pos["hand_r"], s.pos["lowerarm_r"]
    assert PO.angle_deg(PO.sub(hd, el), PO.sub(s.pos["middle_01_r"], hd)) <= 60.5
    w = PO.world_info(s)["held"]["hammer"]
    assert w["target_ue"] == pytest.approx(top, abs=0.01)


def test_a_hammer_measured_from_its_model_lands_flat_too():
    g = PO.grip_from_points(W._hammer_points(True))
    assert g["work"] == "head"
    top = [0.0, 75.0, 95.0]
    s = PO.pose_skel(skel(), "stand", {"action": "striking_down", "work_top_cm": 90.0, "work_near_cm": 38.0,
                                       "work_target_ue": top,
                                       "held": [held("hammer", g["size_m"], grip=g)]})
    put, palm = placed(s, s.info["held"]["hammer"])
    c = PO.held_contact(put, palm, top, [0.0, 0.0, 1.0])
    assert c["gap_cm"] <= 2.0 and c["angle_deg"] <= 15.0 and c["grip_cm"] <= 2.0, c


# ------------------------------------------------------------------ (4) a long thing's tip
def test_a_long_things_tip_lands_on_the_work():
    s, top = strike(POKER, "poker")
    rep = s.info["held"]["poker"]
    assert rep["work"] == "tip"
    put, palm = placed(s, rep)
    c = PO.held_contact(put, palm, top, [0.0, 0.0, 1.0])
    assert c["gap_cm"] <= 2.0 and c["grip_cm"] <= 2.0 and "angle_deg" not in c, c
    assert rep["dir"][2] < -0.3                                  # coming down onto it


def test_head_or_tip_is_read_from_the_model_not_the_name():
    assert PO.grip_from_points(W._hammer_points())["work"] == "head"
    pts = []
    for n in range(80):                                   # blade: 80 cm, 5 cm wide, 0.6 cm thick
        for w in (-0.025, 0.0, 0.025):
            for t in (-0.003, 0.003):
                pts.append([w, t, 0.2 + 0.8 * n / 79])
    for n in range(12):                                   # grip: 12 cm, 3 cm round, under a broad guard
        for w in (-0.015, 0.015):
            for t in (-0.012, 0.012):
                pts.append([w, t, 0.06 + 0.12 * n / 11])
    for w in range(9):
        pts.append([-0.09 + 0.18 * w / 8, 0.0, 0.19])
        pts.append([-0.09 + 0.18 * w / 8, 0.01, 0.2])
    g = PO.grip_from_points(pts)
    assert g["work"] == "tip" and g["tip_m"][2] == pytest.approx(1.0, abs=0.01)
    assert PO.grip_frame(HAMMER)["work"] == "head" and PO.grip_frame(POKER)["work"] == "tip"


# ------------------------------------------------------------------ (5) the staged check flags a wrong placement
def test_the_check_flags_a_face_short_turned_and_out_of_the_hand():
    s, top = strike(HAMMER, "hammer")
    rep = s.info["held"]["hammer"]
    palm = PO.palm_point(s, "r")
    # turned sideways about the handle and slid along it, away from the palm: as the 11-18 still
    side = PO.unit(PO.cross(rep["dir"], rep["face"]))
    put = PO.prop_in_hand(None, PO.add(palm, PO.scale(rep["dir"], -9.0)), rep["dir"], side, grip=rep["grip"])
    put["origin_cm"] = PO.add(put["origin_cm"], [0.0, 0.0, 4.0])
    c = PO.held_contact(put, palm, top, [0.0, 0.0, 1.0], "the hammer")
    assert c["gap_cm"] > 2.0 and c["angle_deg"] > 15.0 and c["grip_cm"] > 2.0
    why = " | ".join(c["why"])
    assert "the hammer's face is" in why and "cm from the work" in why
    assert "turned" in why and "not in the hand" in why
    assert "the held thing" in PO.contact_why(dict(c), None)[0]
    # the staging's facts carry them (speakers.staged_facts), next to the other reasons
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import speakers as SP
    spec = {"characters": {"a": {"id": "fig", "position": [0, 0], "facing": [0, -1],
                                 "body_pose": {"pose": "stand", "params": {}}}}}
    subj = {"a": {"measured": {}, "held": {"hammer_01": {"contact": c}}}}
    assert SP.staged_facts(spec, subj)["a"] == c["why"]
    subj["a"]["held"]["hammer_01"]["contact"] = PO.held_contact(*placed(s, rep), top, [0.0, 0.0, 1.0])
    assert SP.staged_facts(spec, subj)["a"] == []
