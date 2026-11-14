"""Seats from a mesh, seated placement and its check, framing facts, and the redraw's fact words (stdlib + numpy)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "redraw"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))

import seatbox  # noqa: E402
import seating  # noqa: E402
import redraw  # noqa: E402


def _chair(front_x: bool = False) -> np.ndarray:
    """Points of a box chair: seat top 0.45, arms to 0.65 at |x| > 0.25, back to 1.0 at y > 0.25 (front -y)."""
    pts = []
    for x in np.linspace(-0.35, 0.35, 36):
        for y in np.linspace(-0.35, 0.35, 36):
            top = 1.0 if y > 0.25 else 0.65 if abs(x) > 0.25 else 0.45
            for z in np.linspace(0.0, top, 12):
                pts.append([x, y, z])
    p = np.array(pts)
    if front_x:                               # the same chair turned so its front is +x
        p = np.stack([-p[:, 1], p[:, 0], p[:, 2]], 1)
    return p


def test_seat_box_and_front():
    b = seatbox.seat_box(_chair())
    assert abs(b["seat_z_m"] - 0.45) < 0.03
    assert abs(b["inner_x_m"][0] + 0.25) < 0.04 and abs(b["inner_x_m"][1] - 0.25) < 0.04
    assert abs(b["arm_z_m"] - 0.65) < 0.03
    assert 0.2 < b["back_y_m"] < 0.3
    assert seatbox.front_of(_chair()) == [0.0, -1.0]
    assert seatbox.front_of(_chair(front_x=True)) == [1.0, 0.0]


def test_place_faces_the_real_front_and_check_passes():
    box = seatbox.seat_box(_chair())
    seat = {"name": "chair", "pos": [1.0, 2.0], "box": box, "rot": 90.0 + 90.0, "front": seating._rot([0, -1], 180.0)}
    fitd = {"hip_half_m": 0.2, "hip_back_m": 0.16}
    c = seating.place({"id": "x", "body_pose": {"pose": "sit_armchair_relaxed"}}, seat, fitd)
    assert [round(v, 3) for v in c["facing"]] == [0.0, 1.0]
    assert c["body_pose"]["params"]["seat_z_m"] == box["seat_z_m"]
    hip = c["position"]
    # joints where the placement put the hips, 10 cm above the cushion
    r = seating._rot([0.09, 0.0], 180.0)
    j = {"thigh_l": [hip[0] + r[0], hip[1] + r[1], box["seat_z_m"] + 0.1],
         "thigh_r": [hip[0] - r[0], hip[1] - r[1], box["seat_z_m"] + 0.1]}
    ok = seating.check({"joints": j, "feet_above_floor_m": 0.0}, seat, fitd, 0.0)
    assert ok["ok"], ok["why"]
    # moved half a seat sideways: through the arm
    j2 = {k: [v[0] + 0.2, v[1], v[2]] for k, v in j.items()}
    bad = seating.check({"joints": j2, "feet_above_floor_m": 0.0}, seat, fitd, 0.0)
    assert not bad["ok"] and any("arms" in w for w in bad["why"])


def test_neck_and_limb_words():
    w = {"wear": "a black coat, a white wing collar, a grey silk cravat, grey trousers", "pos": "his"}
    s = redraw.neck_words(w)
    assert "white wing collar" in s and "grey silk cravat" in s and "trousers" not in s
    ppl = [("a", [0.3, 0.1, 0.5, 1.0], 10.0, "standing", 10.0, None)]
    shot = {"people": {"a": {"joints2d": {"lowerarm_l": [0.45, 0.5, True], "hand_l": [0.46, 0.7, True],
                                          "lowerarm_r": [0.35, 0.5, True], "hand_r": [0.34, 0.7, False]}}}}
    s = redraw.limb_words(ppl, shot, {"a": {"noun": "man"}})
    assert "both of his arms" in s and "one hand" in s


def test_frame_facts_cut_at_side():
    import speakers
    cam = {"position": [0.0, -3.0, 1.5], "target": [0.0, 0.0, 1.5], "focal_mm": 35}
    m = {"eye_xy": [0.0, 0.0], "eye_z": 1.6, "head_top_z": 1.75, "chin_z": 1.5, "pelvis_xy": [0.0, 0.0],
         "feet_z": 0.0, "joints": {"upperarm_l": [0.2, 0.0, 1.4], "upperarm_r": [-0.2, 0.0, 1.4]}}
    f = speakers.frame_facts(cam, m, (1152, 640))
    assert f["head_in"] and not f["cut"]
    side = dict(m, eye_xy=[1.45, 0.0], joints={"upperarm_l": [1.7, 0.0, 1.4], "upperarm_r": [1.25, 0.0, 1.4]})
    f = speakers.frame_facts(cam, side, (1152, 640))
    assert f["cut"] and "right" in f["cut_edges"]
    assert math.isfinite(f["in_frac"])


def test_sitters_on_an_open_seat_turn_to_whom_they_face():
    edge = {"front": [0.0, -1.0], "rot": 0.0, "pos": [0.0, 0.0], "box": {"back_y_m": 0.3, "front_y_m": -0.3, "seat_z_m": 0.45, "inner_x_m": [-1.0, 1.0]}}
    chars = {"a": {"position": [-0.5, 0.0], "facing_to": "b"}, "b": {"position": [0.5, 0.0], "facing_to": "a"},
             "c": {"position": [0.0, -3.0], "facing_to": "a"}}
    ta, tb = seating.turn_for(chars["a"], edge, chars), seating.turn_for(chars["b"], edge, chars)
    assert ta == seating.SEAT_TURN_DEG and tb == -seating.SEAT_TURN_DEG        # side by side: each toward the other
    facing = seating.place(chars["a"], edge, {"hip_back_m": 0.1}, turn=ta)["facing"]
    assert facing[0] > 0.4 and facing[1] < -0.8                                # still mostly out of the seat's front
    assert seating.turn_for(dict(chars["a"], facing_to="c"), dict(edge, front=[0.0, -1.0]), chars) < 15   # one ahead
    arms = dict(edge, box=dict(edge["box"], arm_z_m=0.65))
    assert seating.turn_for(chars["a"], arms, chars) == 0.0                    # between arm rests: square
    assert seating.turn_for({"position": [0, 0]}, edge, chars) == 0.0          # nowhere to face: square
    assert seating.turn_for({"position": [0, 0], "facing": [1.0, 0.0]}, edge, chars) == seating.SEAT_TURN_DEG
    assert seating.turn_for({"position": [0, 0], "facing": [1.0, 0.0]}, edge, {"a": {}}) == 0.0   # alone: square
