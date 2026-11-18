"""Work start poses follow body mechanics (scenes/ue/pose.py work actions), on a synthetic 1.75 m humanoid.

For tool_raised, striking_down and hands_at_work at work tops of 45, 75 and 95 cm: both feet planted on the floor,
staggered fore-aft (the foot opposite the tool hand forward); knees bent forward; elbows bent (25-165 deg) toward
down/out; wrists within 60 deg; hands at their targets (a work-top hand on the top, the raised hand over the head);
the centre of mass over the support polygon; the head pitched down at the work; the held tool's grip point in the
palm with its handle along the action's direction. Plus the pure pieces: centre of mass, balance margin, grip frame.
"""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("scene_pose_t", HERE.parents[0] / "assets" / "scenes" / "ue" / "pose.py")
PO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PO)

TOPS = (45.0, 75.0, 95.0)
ACTS = ("tool_raised", "striking_down", "hands_at_work")
HAMMER = [0.04, 0.12, 0.35]      # an upright one-handed tool (m): handle along z, head 12 cm long


def humanoid():
    """Joint positions (actor frame: x = the figure's left, y forward, z up; cm) of a 1.75 m figure in an A-pose,
    the bone names of the MetaHuman body skeleton pose.py uses."""
    P, par = {}, {}

    def j(n, p, parent):
        P[n], par[n] = list(p), parent
    j("root", [0, 0, 0], None)
    j("pelvis", [0, 0, 97], "root")
    for n, p, pa in (("spine_01", [0, -1, 102], "pelvis"), ("spine_02", [0, -1.5, 110], "spine_01"),
                     ("spine_03", [0, -1.5, 119], "spine_02"), ("spine_04", [0, -1, 128], "spine_03"),
                     ("spine_05", [0, -0.5, 137], "spine_04"), ("neck_01", [0, 0, 147], "spine_05"),
                     ("neck_02", [0, 0.5, 152], "neck_01"), ("head", [0, 1, 158], "neck_02")):
        j(n, p, pa)
    for side, sx in (("l", 1.0), ("r", -1.0)):
        a = math.radians(40)
        d = [sx * math.sin(a), 0.0, -math.cos(a)]                 # the arm's line in the A-pose
        nrm = [-sx * math.cos(a), 0.0, -math.sin(a)]               # the palm's side (toward the body)
        j(f"clavicle_{side}", [sx * 3, 2, 144], "spine_05")
        j(f"upperarm_{side}", [sx * 17, -1, 143], f"clavicle_{side}")
        el = [P[f"upperarm_{side}"][i] + 28 * d[i] for i in range(3)]
        j(f"lowerarm_{side}", el, f"upperarm_{side}")
        hd = [el[i] + 26 * d[i] for i in range(3)]
        j(f"hand_{side}", hd, f"lowerarm_{side}")
        for f, along, fy, ln in (("index", 9.0, 2.2, 4.0), ("middle", 9.5, 0.0, 4.5), ("ring", 9.0, -2.0, 4.2),
                                 ("pinky", 8.0, -3.8, 3.4)):
            b = [hd[i] + along * d[i] + (fy if i == 1 else 0.0) for i in range(3)]
            j(f"{f}_01_{side}", b, f"hand_{side}")
            b2 = [b[i] + ln * d[i] for i in range(3)]
            j(f"{f}_02_{side}", b2, f"{f}_01_{side}")
            j(f"{f}_03_{side}", [b2[i] + 2.6 * d[i] for i in range(3)], f"{f}_02_{side}")
        t1 = [hd[i] + 2.0 * d[i] + 1.5 * nrm[i] + (2.5 if i == 1 else 0.0) for i in range(3)]
        j(f"thumb_01_{side}", t1, f"hand_{side}")
        t2 = [t1[i] + 2.0 * d[i] + (2.8 if i == 1 else 0.0) for i in range(3)]
        j(f"thumb_02_{side}", t2, f"thumb_01_{side}")
        j(f"thumb_03_{side}", [t2[i] + 2.0 * d[i] + (1.5 if i == 1 else 0.0) for i in range(3)], f"thumb_02_{side}")
        j(f"thigh_{side}", [sx * 9, 0, 92], "pelvis")
        j(f"calf_{side}", [sx * 9.5, 0.5, 50], f"thigh_{side}")
        j(f"foot_{side}", [sx * 10, -1, 8], f"calf_{side}")
        j(f"ball_{side}", [sx * 10, 12, 3], f"foot_{side}")
    return P, par


def posed(action, top, size=HAMMER, posture="stand", **extra):
    P, par = humanoid()
    s = PO.Skel.from_joints(P, par)
    params = dict(action=action, work_top_cm=top, work_fwd_cm=58.0, work_near_cm=38.0, tool_size_m=size, **extra)
    PO.POSES[posture](s, params)
    PO.ACTIONS[action](s, dict(params, _posture=posture))
    return s


def ang(a, b):
    return PO.angle_deg(a, b)


CASES = [(a, t) for a in ACTS for t in TOPS]


@pytest.fixture(scope="module")
def poses():
    return {(a, t): posed(a, t) for a, t in CASES}


@pytest.mark.parametrize("act,top", CASES)
def test_feet_planted_staggered_and_turned_out(poses, act, top):
    s = poses[(act, top)]
    for b in ("foot_l", "foot_r", "ball_l", "ball_r"):
        assert abs(s.pos[b][2] - s.ref_pos[b][2]) < 1.0, (b, s.pos[b])
    # the foot opposite the tool hand (right) forward; a step apart fore-aft, about shoulder width sideways
    assert s.pos["foot_l"][1] - s.pos["foot_r"][1] > 0.12 * 175
    assert abs(s.pos["foot_l"][0] - s.pos["foot_r"][0]) > 28.0
    for side, sx in (("l", 1.0), ("r", -1.0)):
        toe = PO.sub(s.pos[f"ball_{side}"], s.pos[f"foot_{side}"])
        assert sx * toe[0] > 0.0                                  # toes turned out


@pytest.mark.parametrize("act,top", CASES)
def test_knees_bend_forward_and_hips_lower(poses, act, top):
    s = poses[(act, top)]
    assert s.pos["pelvis"][2] < s.ref_pos["pelvis"][2] - 3.0
    for side in ("l", "r"):
        h, k, a = s.pos[f"thigh_{side}"], s.pos[f"calf_{side}"], s.pos[f"foot_{side}"]
        line = PO.unit(PO.sub(a, h))
        off = PO.perp(PO.sub(k, h), line)
        toe = PO.unit(PO.sub(s.pos[f"ball_{side}"], a))
        assert PO.dot(off, toe) > 2.0, (side, off)                # the knee in front of the hip-ankle line
        assert ang(PO.sub(h, k), PO.sub(a, k)) < 165.0            # not locked


@pytest.mark.parametrize("act,top", CASES)
def test_elbows_bent_the_natural_way_and_wrists_within_limits(poses, act, top):
    s = poses[(act, top)]
    left, fwd = PO._body_axes(s)
    for side, sx in (("l", 1.0), ("r", -1.0)):
        sh, el, hd = s.pos[f"upperarm_{side}"], s.pos[f"lowerarm_{side}"], s.pos[f"hand_{side}"]
        e = ang(PO.sub(sh, el), PO.sub(hd, el))
        assert 25.0 <= e <= 165.0, (side, e)
        off = PO.perp(PO.sub(el, sh), PO.unit(PO.sub(hd, sh)))     # where the elbow points off the shoulder-hand line
        assert PO.dot(off, PO.scale(left, sx)) > -1.0, (side, off)  # out, never into the body
        raised = hd[2] > s.pos["head"][2]
        if not raised:
            assert off[2] < 1.0, (side, off)                         # down, never up through the shoulder
        w = ang(PO.sub(hd, el), PO.sub(s.pos[f"middle_01_{side}"], hd))
        assert w <= 60.5, (side, w)
        # the hand clear of the trunk
        a, b = s.pos["pelvis"], s.pos["spine_05"]
        ab = PO.sub(b, a)
        t = max(0.0, min(1.0, PO.dot(PO.sub(hd, a), ab) / PO.dot(ab, ab)))
        assert PO.norm(PO.sub(hd, PO.add(a, PO.scale(ab, t)))) > 0.07 * 175


@pytest.mark.parametrize("act,top", CASES)
def test_hands_reach_their_targets(poses, act, top):
    s = poses[(act, top)]
    w = s.info["work"]
    for b, t in w["targets"].items():
        assert math.dist(s.pos[b], t) < 3.0, (b, s.pos[b], t)
    assert max(w["reach"].values()) <= PO.REACH + 1e-6, w["reach"]
    off = w["targets"]["hand_l"]                                # the off hand rests on the work's top
    assert top <= off[2] <= top + 8.0
    if act == "tool_raised":
        # beside the head, the elbow bent under it (not over the head on a straight arm)
        assert s.pos["head"][2] - 12.0 < s.pos["hand_r"][2] < s.pos["head"][2] + 0.1 * 175
        assert w["reach"]["r"] < 0.8
    else:
        assert top <= s.pos["hand_r"][2] <= top + 40.0


@pytest.mark.parametrize("act,top", CASES)
def test_centre_of_mass_over_the_feet(poses, act, top):
    s = poses[(act, top)]
    m, com, poly = PO.balance_margin(s.pos)
    assert m >= 2.5, (m, com)
    assert s.info["work"]["balance_margin_cm"] == pytest.approx(m, abs=0.1)


@pytest.mark.parametrize("act,top", CASES)
def test_head_looks_at_the_work(poses, act, top):
    s = poses[(act, top)]
    f = PO.face_forward(s)
    look = s.info["work"]["look_at"]
    assert f[2] < -0.1, f                                      # pitched down
    assert ang(f, PO.sub(look, s.pos["head"])) < 20.0


@pytest.mark.parametrize("act,top", CASES)
def test_trunk_turn_and_lean(poses, act, top):
    s = poses[(act, top)]
    left, fwd = PO._body_axes(s)
    yaw = math.degrees(math.atan2(left[1], left[0]))           # positive: shoulders turned toward the right
    if act == "tool_raised":
        assert yaw > 10.0                                       # wound up toward the tool side
    else:
        assert abs(yaw) < 8.0                                   # squared over the work
    tr = PO.sub(s.pos["neck_01"], s.pos["pelvis"])
    assert tr[1] > 0.0                                          # leaning toward the work
    if act != "tool_raised" and top == 45.0:
        assert ang(tr, [0, 0, 1]) > 25.0


@pytest.mark.parametrize("act,top", CASES)
def test_tool_gripped_in_the_palm_along_the_action(poses, act, top):
    s = poses[(act, top)]
    t = s.info["tool"]
    palm = PO.palm_point(s, "r")
    hd, m = s.pos["hand_r"], s.pos["middle_01_r"]
    assert math.dist(palm, PO.add(hd, PO.scale(PO.sub(m, hd), 0.6))) < 0.35 * math.dist(hd, m)   # in the hand
    put = PO.prop_in_hand(HAMMER, t["palm_cm"], t["dir"], t["face"])
    g = put["grip"]
    cols = [put["x"], put["y"], put["z"]]
    grip_at = list(put["origin_cm"])
    for a in range(3):
        grip_at = PO.add(grip_at, PO.scale(cols[a], g["grip_m"][a] * 100.0))
    assert math.dist(grip_at, palm) < 2.0
    long_ax = cols[g["axis"]]
    assert ang(long_ax, t["want_dir"]) < 20.0, (long_ax, t["want_dir"])
    assert ang(long_ax, t["dir"]) < 1e-3
    det = PO.dot(cols[0], PO.cross(cols[1], cols[2]))
    assert det == pytest.approx(1.0, abs=1e-6)                   # a proper rotation
    up, back = t["want_dir"][2], t["want_dir"][1]
    if act == "tool_raised":
        assert up > 0.3 and back < -0.3                          # up and back
    elif act == "striking_down":
        assert up < -0.2 and back > 0.5                          # forward and down
        head = PO.add(palm, PO.scale(t["dir"], (g["len_m"] - g["grip_from_end_m"]) * 100))
        assert abs(head[2] - (top + 100 * g["head_m"] / 2)) < 6.0   # the head comes down on the work's top
    # the handle runs across the palm: perpendicular to the hand's own line, beyond the thumb side
    h = PO.sub(m, hd)
    assert abs(PO.dot(PO.unit(h), t["dir"])) < 0.1
    assert PO.dot(PO.sub(s.pos["index_01_r"], s.pos["pinky_01_r"]), t["dir"]) > 0.0


def test_two_handed_tool_puts_the_other_hand_on_the_handle():
    size = [0.05, 0.2, 1.0]
    for act in ("tool_raised", "striking_down"):
        s = posed(act, 75.0, size=size)
        t = s.info["tool"]
        g = t["grip"]
        assert g["two_hands"]
        off = PO.palm_point(s, "l")
        line = PO.sub(off, t["palm_cm"])
        along = PO.dot(line, t["dir"])
        assert along == pytest.approx(-100.0 * (g["grip_from_end_m"] - g["off_from_end_m"]), abs=3.0)
        assert PO.norm(PO.perp(line, PO.unit(t["dir"]))) < 3.0     # on the handle's line
        for side in ("l", "r"):
            e = ang(PO.sub(s.pos[f"upperarm_{side}"], s.pos[f"lowerarm_{side}"]),
                    PO.sub(s.pos[f"hand_{side}"], s.pos[f"lowerarm_{side}"]))
            assert 25.0 <= e <= 165.0


def test_left_handed_mirrors():
    s = posed("tool_raised", 75.0, hand="l")
    assert s.pos["foot_r"][1] - s.pos["foot_l"][1] > 0.12 * 175        # the right foot forward now
    assert s.pos["hand_l"][2] > s.pos["head"][2]
    assert PO.balance_margin(s.pos)[0] >= 2.5


def test_seated_worker_keeps_the_seat():
    P, par = humanoid()
    s = PO.Skel.from_joints(P, par)
    params = dict(action="hands_at_work", work_top_cm=75.0, work_fwd_cm=45.0, seat_z_m=0.45)
    PO.POSES["seated"](s, params)
    pel = list(s.pos["pelvis"])
    PO.ACTIONS["hands_at_work"](s, dict(params, _posture="seated"))
    assert math.dist(pel, s.pos["pelvis"]) < 1e-6
    assert s.info["work"]["standing"] is False
    for side in ("l", "r"):
        e = ang(PO.sub(s.pos[f"upperarm_{side}"], s.pos[f"lowerarm_{side}"]),
                PO.sub(s.pos[f"hand_{side}"], s.pos[f"lowerarm_{side}"]))
        assert 25.0 <= e <= 165.0


# ------------------------------------------------------------------ the pure pieces
def test_centre_of_mass_of_the_reference_figure():
    P, _ = humanoid()
    c = PO.centre_of_mass(P)
    assert abs(c[0]) < 0.5 and abs(c[1]) < 3.0
    assert 0.53 * 175 < c[2] < 0.6 * 175                         # ~55-57 % of stature
    assert sum(m for m, *_ in PO.SEGMENTS) == pytest.approx(1.0, abs=1e-6)


def test_balance_margin_signs():
    P, _ = humanoid()
    m, com, poly = PO.balance_margin(P)
    assert m > 3.0
    Q = {k: list(v) for k, v in P.items()}
    for k in Q:
        if not k.startswith(("foot_", "ball_")):
            Q[k][1] += 40.0                                       # the body far ahead of its feet
    assert PO.balance_margin(Q)[0] < 0.0
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert PO.polygon_margin([5, 5], sq) == pytest.approx(5.0)
    assert PO.polygon_margin([12, 5], sq) == pytest.approx(-2.0)


def test_grip_frame_conventions():
    g = PO.grip_frame([0.04, 0.12, 0.35])                         # upright: held at the bottom end
    assert g["axis"] == 2 and g["face_axis"] == 1 and not g["two_hands"]
    assert g["grip_m"][2] == pytest.approx(0.0525)
    g = PO.grip_frame([0.6, 0.05, 0.1])                           # lying along x: the negative end
    assert g["axis"] == 0 and g["grip_m"][0] == pytest.approx(-0.3 + 0.09) and g["grip_m"][2] == pytest.approx(0.05)
    g = PO.grip_frame([0.2, 0.2, 0.2 * 0 + 0.25])
    assert g["grip_from_end_m"] == pytest.approx(0.05)             # never closer than 5 cm to the end
    g = PO.grip_frame([0.05, 0.25, 1.1])
    assert g["two_hands"] and g["off_grip_m"][2] < g["grip_m"][2]


def _hammer_points(flip=False):
    """A hammer lying along x: a 30 cm round handle 3 cm thick and a 6 x 12 x 5 cm head at one end."""
    import math as m
    pts = []
    for n in range(60):
        for a in range(8):
            pts.append([-0.15 + 0.30 * n / 59, 0.015 * m.cos(a * m.pi / 4), 0.06 + 0.015 * m.sin(a * m.pi / 4)])
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                pts.append([0.15 + 0.06 * ix / 5, -0.025 + 0.05 * iy / 5, 0.12 * iz / 5])
    return [[-p[0], p[1], p[2]] for p in pts] if flip else pts


def test_a_tool_is_gripped_on_its_handle_whichever_way_its_model_lies():
    for flip, sign in ((False, 1.0), (True, -1.0)):
        g = PO.grip_from_points(_hammer_points(flip))
        assert g["axis"] == 0 and g["sign"] == sign and g["from"] == "model"
        x = g["grip_m"][0] * (-1 if flip else 1)
        assert -0.15 < x < 0.0, g                        # on the handle, in from its free end
        assert abs(g["grip_m"][2] - 0.06) < 0.01         # on the handle's own line, not the box's middle
        assert g["grip_from_end_m"] < 0.1


def test_a_blade_is_held_by_its_round_grip_not_its_flat_blade():
    pts = []
    for n in range(80):                                   # blade: 80 cm, 5 cm wide, 0.6 cm thick
        for w in (-0.025, 0.0, 0.025):
            for t in (-0.003, 0.003):
                pts.append([w, t, 0.2 + 0.8 * n / 79])
    for n in range(12):                                   # grip: 12 cm, 3 cm round, under a guard
        for w in (-0.015, 0.015):
            for t in (-0.012, 0.012):
                pts.append([w, t, 0.06 + 0.12 * n / 11])
    for w in range(9):
        pts.append([-0.09 + 0.18 * w / 8, 0.0, 0.19])
        pts.append([-0.09 + 0.18 * w / 8, 0.01, 0.2])
    g = PO.grip_from_points(pts)
    assert g["axis"] == 2 and g["sign"] == 1.0 and 0.06 <= g["grip_m"][2] <= 0.18, g


def test_the_model_grip_turns_the_prop_head_first_along_the_tool():
    g = PO.grip_from_points(_hammer_points(True))
    r = PO.prop_in_hand(g["size_m"], [0.0, 0.0, 100.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], grip=g)
    assert r["x"][2] < -0.99                              # the mesh's +x (its handle end) points away from the head
