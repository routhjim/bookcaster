"""A person at work on a piece of the set itself takes hold of a part of it (scenes/ue/pose.py: surface_points,
grip_parts, fixture_hands, fixture_contact), on the synthetic 1.75 m humanoid of test_work_pose_1117.

11-18: the plan stood a person at a standing piece they work with their hands, and the hands hung near it, not on
it. Now the piece's surface round the person is searched for a part a hand closes round (thin, roundish, room for
the fingers, a palm's width long, at a workable height, open above), the free hands are solved onto it, and the
palm-to-part distance is the staged check (measured again in the engine by scene_job and carried to the staging's
facts by speakers.staged_facts). The pieces here are plain geometry: posts, bars, slabs, a shaft, a rope.
"""
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_work_pose_1117 as W  # noqa: E402

PO = W.PO
H = 158.0 / 0.9                         # the humanoid's stature (cm): head joint at 0.9 of it, floor at 0


# ------------------------------------------------------------------ plain geometry (cm, the actor frame)
def box(c, h):
    vs = [[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    t = []
    for a, b, c_, d in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        t += [(a, b, c_), (a, c_, d)]
    return vs, t


def cyl(a, b, r, n=12):
    d = PO.unit(PO.sub(b, a))
    u, v = PO._basis(d)
    vs = [PO.add(e, PO.add(PO.scale(u, r * math.cos(2 * math.pi * k / n)), PO.scale(v, r * math.sin(2 * math.pi * k / n))))
          for e in (a, b) for k in range(n)] + [list(a), list(b)]
    t = []
    for k in range(n):
        k2 = (k + 1) % n
        t += [(k, k2, n + k2), (k, n + k2, n + k), (2 * n, k2, k), (2 * n + 1, n + k, n + k2)]
    return vs, t


def merge(*ms):
    V, T = [], []
    for vs, ts in ms:
        o = len(V)
        V += vs
        T += [tuple(i + o for i in t) for t in ts]
    return V, T


def surface(*ms):
    return PO.surface_points(*merge(*ms))


# a rail: two square posts and a round bar 3.2 cm thick between them, 95 cm up, 55 cm in front
RAIL = (box([-30, 55, 50], [3, 3, 50]), box([30, 55, 50], [3, 3, 50]), cyl([-27, 55, 95], [27, 55, 95], 1.6))
# a table: a 4 cm top on four 5 cm square legs (the legs are thin enough to hold but lie under the top)
TABLE = (box([0, 70, 88], [60, 30, 2]),) + tuple(box([sx * 55, 70 + sy * 25, 43], [2.5, 2.5, 43])
                                                for sx in (-1, 1) for sy in (-1, 1))
# a block with a shaft standing out of it, leaning back
SHAFT = (box([0, 80, 30], [20, 20, 30]), cyl([0, 62, 58], [0, 50, 125], 1.5))
# a short crosswise grip (16 cm, stood off 10 cm on two stems) on a post
HANDLE = (box([0, 70, 55], [4, 4, 55]), cyl([-8, 56, 100], [8, 56, 100], 1.4), cyl([-8, 56, 100], [-8, 66, 100], 1.0),
          cyl([8, 56, 100], [8, 66, 100], 1.0))


def skel():
    P, par = W.humanoid()
    return PO.Skel.from_joints(P, par)


def work(pts, action="hands_at_work", held=None, **extra):
    p = {"action": action, "work_top_cm": 95.0, "work_fwd_cm": 55.0, "work_near_cm": 50.0,
         "fixture": dict({"name": "the piece", "pts_ue": pts}, **extra)}
    if held:
        p["held"] = held
    return PO.pose_skel(skel(), "stand", p)


# ------------------------------------------------------------------ the surface
def test_a_big_face_is_filled_between_its_corners():
    vs, ts = [[0, 0, 0], [60, 0, 0], [0, 60, 0]], [(0, 1, 2)]
    pts = PO.surface_points(vs, ts, step=1.5)
    assert len(pts) > 600                                   # the corners alone say nothing about the middle
    mid = [20.0, 20.0, 0.0]
    assert min(math.dist(p, mid) for p in pts) < 1.5
    near = PO.surface_points(vs, ts, step=1.5, centre=[5.0, 5.0, 0.0], radius=8.0)
    assert near and all(math.dist(p, [5.0, 5.0, 0.0]) <= 8.0 for p in near)
    flat = PO.surface_points(vs, [0, 1, 2], step=1.5)       # a flat index list as the engine gives it
    assert len(flat) == len(pts)


# ------------------------------------------------------------------ finding a part a hand closes round
def test_a_round_bar_between_posts_is_found_along_its_length():
    parts = PO.grip_parts(surface(*RAIL), [0, 30, 127])
    bars = [q for q in parts if abs(q["dir"][0]) > 0.98]
    assert len(bars) == 1, parts
    q = bars[0]
    assert q["thick_cm"] == pytest.approx(3.2, abs=0.6)
    assert q["c"][1] == pytest.approx(55.0, abs=0.5) and q["c"][2] == pytest.approx(95.0, abs=0.5)
    assert q["len_cm"] >= 40.0                              # most of the bar, short of the posts (room for fingers)
    assert max(abs(q["a"][0]), abs(q["b"][0])) <= 27.0 - PO.FIX_PALM_CM / 2 + 1.0


def test_a_shaft_a_rope_and_a_crosswise_handle_are_found():
    shaft = [q for q in PO.grip_parts(surface(*SHAFT), [0, 30, 127]) if q["dir"][2] ** 2 > 0.8]
    assert shaft and shaft[0]["thick_cm"] == pytest.approx(3.0, abs=0.6) and shaft[0]["len_cm"] >= 50.0
    # a rope hanging from above, bending a little
    rope = merge(cyl([0, 50, 200], [0, 52, 140], 1.2, 10), cyl([0, 52, 140], [3, 54, 80], 1.2, 10))
    parts = PO.grip_parts(PO.surface_points(*rope), [0, 30, 127])
    assert parts and max(q["len_cm"] for q in parts) >= 80.0
    handle = [q for q in PO.grip_parts(surface(*HANDLE), [0, 30, 127]) if abs(q["dir"][0]) > 0.9]
    assert handle and handle[0]["c"][2] == pytest.approx(100.0, abs=1.0)


@pytest.mark.parametrize("what", ["slab", "plank", "block"])
def test_flat_or_thick_things_are_not_parts(what):
    m = {"slab": box([0, 60, 90], [40, 25, 2]),            # a table top's edge is not a handle
         "plank": box([0, 60, 95], [30, 2.5, 0.5]),         # 5 x 1 cm: too flat to close a hand round
         "block": box([0, 60, 80], [6, 6, 30])}[what]       # 12 cm square: too thick
    assert PO.grip_parts(PO.surface_points(*m), [0, 30, 127]) == []


def test_a_table_leg_under_its_top_is_not_taken():
    pts = surface(*TABLE)
    assert PO.grip_parts(pts, [0, 30, 127])                 # the legs are thin enough ...
    s = work(pts)
    assert s.info["fixture"]["hands"] == {}                 # ... but no hand can come onto them from above


# ------------------------------------------------------------------ the hands on the part
def test_both_free_hands_take_a_long_bar_a_shoulders_width_apart():
    s = work(surface(*RAIL))
    fx = s.info["fixture"]
    assert set(fx["hands"]) == {"l", "r"}
    for x, h in fx["hands"].items():
        palm = PO.palm_point(s, x)
        c = PO.fixture_contact(palm, h["a_cm"], h["b_cm"], "the piece", x)
        assert c["palm_cm"] <= PO.HELD_LIMITS["grip_cm"] and c["why"] == [], (x, c)
        assert h["solver_palm_cm"] == pytest.approx(c["palm_cm"], abs=0.01)
        # the hand closed round it: the knuckle line along the bar, the fingers curled by its thickness
        k = PO.unit(PO.sub(s.pos[f"index_01_{x}"], s.pos[f"pinky_01_{x}"]))
        assert abs(PO.dot(k, [1.0, 0.0, 0.0])) > 0.7
    left = fx["hands"]["l"]["on_cm"][0]
    right = fx["hands"]["r"]["on_cm"][0]
    assert left > right                                     # the left hand at the left end (x is the figure's left)
    assert left - right == pytest.approx(34.0, abs=1.0)     # the shoulders' width
    assert s.info["work"]["reach"]["l"] <= PO.REACH + 1e-6
    assert s.info["work"]["balance_margin_cm"] >= PO.BALANCE_MARGIN_CM - 0.5
    w = PO.world_info(s)["fixture"]["hands"]["r"]            # what scene_job measures against
    assert all(k in w for k in ("a_ue", "b_ue", "on_ue", "palm_ue", "hand_ue", "dir_ue"))


def test_a_short_handle_takes_the_working_hand_only():
    s = work(surface(*HANDLE))
    hands = s.info["fixture"]["hands"]
    assert list(hands) == ["r"], hands
    c = PO.fixture_contact(PO.palm_point(s, "r"), hands["r"]["a_cm"], hands["r"]["b_cm"])
    assert c["palm_cm"] <= PO.HELD_LIMITS["grip_cm"], c


def test_a_hand_holding_a_thing_does_not_let_go_of_it():
    held = [{"asset": "tool", "hand": "r", "size_m": W.HAMMER, "name": "the tool"}]
    s = work(surface(*RAIL), action="striking_down", held=held)
    assert list(s.info["fixture"]["hands"]) == ["l"]       # only the free hand goes to the part
    assert s.info["held"]["tool"]["hand"] == "r"
    h = s.info["fixture"]["hands"]["l"]
    assert PO.fixture_contact(PO.palm_point(s, "l"), h["a_cm"], h["b_cm"])["palm_cm"] <= 2.0


def test_no_piece_no_change():
    a = PO.pose_skel(skel(), "stand", {"action": "hands_at_work", "work_top_cm": 95.0, "work_fwd_cm": 55.0})
    b = PO.pose_skel(skel(), "stand", {"action": "hands_at_work", "work_top_cm": 95.0, "work_fwd_cm": 55.0,
                                       "fixture": {"name": "the piece"}})       # its surface not read
    assert "fixture" not in a.info and "fixture" not in b.info
    assert a.pos["hand_r"] == pytest.approx(b.pos["hand_r"])


# ------------------------------------------------------------------ the check, the correction, the facts
def test_the_check_says_which_hand_is_off_the_piece():
    c = PO.fixture_contact([0.0, 60.0, 95.0], [-20.0, 55.0, 95.0], [20.0, 55.0, 95.0], "the frame", "l")
    assert c["palm_cm"] == pytest.approx(5.0)
    assert c["why"] == ["the left hand is not on the frame: its palm is 5 cm from the part it holds"]
    assert PO.fixture_contact([0.0, 56.0, 95.0], [-20, 55, 95], [20, 55, 95])["why"] == []
    assert "the piece they work" in PO.fixture_why({"palm_cm": 9.0})[0]


def test_the_correction_moves_the_palm_across_the_part_not_along_it():
    a, b = [-20.0, 55.0, 95.0], [20.0, 55.0, 95.0]
    assert PO.fixture_fix(None, [5.0, 60.0, 99.0], a, b) == pytest.approx([0.0, -5.0, -4.0])
    assert PO.fixture_fix([1.0, 0.0, 0.0], [30.0, 55.0, 95.0], a, b) == pytest.approx([-9.0, 0.0, 0.0])  # past its end


def test_a_correction_moves_the_solved_palm_by_it():
    pts = surface(*RAIL)
    s0 = work(pts)
    s1 = work(pts, fix_ue={"r": [0.0, 0.0, 3.0]})            # (off the engine the actor frame is the UE frame)
    p0, p1 = PO.palm_point(s0, "r"), PO.palm_point(s1, "r")
    assert PO.sub(p1, p0) == pytest.approx([0.0, 0.0, 3.0], abs=0.3)
    assert s1.info["fixture"]["hands"]["r"]["aim_cm"][2] == pytest.approx(s1.info["fixture"]["hands"]["r"]["on_cm"][2] + 3.0)


def test_the_staging_facts_carry_a_hand_off_the_piece():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import speakers as SP
    spec = {"characters": {"a": {"id": "fig", "position": [0, 0], "facing": [0, -1],
                                 "body_pose": {"pose": "stand", "params": {}}}}}
    why = PO.fixture_contact([0.0, 60.0, 95.0], [-20, 55, 95], [20, 55, 95], "the frame", "r")["why"]
    subj = {"a": {"measured": {}, "fixture": {"name": "the frame", "hands": {}, "why": why}}}
    assert SP.staged_facts(spec, subj)["a"] == why
    subj["a"]["fixture"]["why"] = []
    assert SP.staged_facts(spec, subj)["a"] == []


def test_the_piece_is_named_by_the_plans_own_data_not_by_words():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import scene as SC
    pr = {"action": "hands_at_work", "work_top_cm": 95.0}
    assert SC.fixture_params({"work_at": "piece_3"}, pr)["fixture"] == {"name": "piece_3"}
    # a worked thing lying on it is the work (a blow's target), not the piece; no work action, no piece
    assert "fixture" not in SC.fixture_params({"work_at": "piece_3"}, dict(pr, work_target_ue=[0, 0, 90]))
    assert "fixture" not in SC.fixture_params({"work_at": "piece_3"}, {"action": "gesture_point"})
    assert "fixture" not in SC.fixture_params({}, pr)
    assert set(SC.WORK_POSE_ACTIONS) == set(PO.WORK_ACTIONS)
