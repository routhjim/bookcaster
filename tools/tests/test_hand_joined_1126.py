"""How near two who are turned to each other stand, and the stance of it: the staging pass's "near" pairs get the
band's distance, a dance's closed hold or a joined hand (scenes/staging_pass.py q_near, from_plan near_of /
near_position / near_pose, scenes/ue/pose.py pose_dance_hold / pose_hand_joined), on synthetic data."""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "assets" / "sets"))
import staging_synth as S  # noqa: E402,F401
import from_plan as FP  # noqa: E402
from test_work_pose_1117 import PO, humanoid  # noqa: E402


def posed(pose, **params):
    P, par = humanoid()
    s = PO.Skel.from_joints(P, par)
    return PO.pose_skel(s, pose, dict(params, weight_on="l"))


def test_the_joined_hand_is_held_out_ahead_between_waist_and_chest():
    for hand in ("r", "l"):
        s = posed("hand_joined", hand=hand, reach_m=0.5)
        h, pel, other = s.pos[f"hand_{hand}"], s.pos["pelvis"], s.pos["hand_l" if hand == "r" else "hand_r"]
        assert h[1] - pel[1] > 35.0 and abs(h[0] - pel[0]) < 15.0      # ahead of the body, on the line between them
        assert pel[2] < h[2] < s.pos["neck_01"][2]
        assert other[2] < pel[2] + 10.0 and abs(other[1] - pel[1]) < 30.0   # the other arm hangs
    assert posed("hand_joined", reach_m=0.35).pos["hand_r"][1] < posed("hand_joined", reach_m=0.55).pos["hand_r"][1]


def in_frame_of(other_pelvis_xy, q, me_xy=(0.0, 0.0)):
    """A point of a body that stands turned about at other_pelvis_xy, in the frame of the one at me_xy."""
    return [other_pelvis_xy[0] - q[0], other_pelvis_xy[1] - q[1], q[2]]


def hold_pair(gap_cm=38.0, small=1.0, off=(0.0, 0.0)):
    """A leader and a follower (the follower `small` times the size, standing `off` to the side and back of where
    the gap says), each posed from the other's own bones as the scene gives them."""
    P, par = humanoid()
    Pf = {b: [c * small for c in v] for b, v in P.items()}
    at = (off[0], gap_cm + off[1])                                # the follower's origin in the leader's frame
    lead_sees = {b: in_frame_of(at, Pf[b]) for b in PO.PARTNER_BONES}
    fol_sees = {b: in_frame_of(at, P[b]) for b in PO.PARTNER_BONES}     # (turned about: the same offset back)
    lead = PO.pose_skel(PO.Skel.from_joints(P, par), "dance_hold",
                        {"role": "lead", "gap_m": gap_cm / 100, "weight_on": "l", "partner": lead_sees})
    fol = PO.pose_skel(PO.Skel.from_joints(Pf, par), "dance_hold",
                       {"role": "follow", "gap_m": gap_cm / 100, "weight_on": "l", "partner": fol_sees})
    return lead, fol, (lambda q: in_frame_of(at, q)), P, Pf


def test_in_a_dance_hold_the_joined_hands_meet_and_the_others_rest_on_the_partner():
    gap = 0.38
    lead, fol = posed("dance_hold", role="lead", gap_m=gap), posed("dance_hold", role="follow", gap_m=gap)
    lp, fp = lead.ref_pos["pelvis"], fol.ref_pos["pelvis"]

    def seen_by_lead(q):            # a point of the follower (turned about, gap ahead) in the leader's frame
        return [lp[0] - (q[0] - fp[0]), lp[1] + gap * 100 - (q[1] - fp[1]), q[2]]
    assert math.dist(lead.pos["hand_l"], seen_by_lead(fol.pos["hand_r"])) < 12.0           # the joined hands
    assert lead.pos["hand_l"][0] - lp[0] > 20.0 and lead.pos["hand_l"][2] > lp[2] + 20.0   # out to the side, raised
    back = lead.pos["hand_r"]                                    # the leader's hand: past the partner's middle
    assert back[1] - lp[1] > gap * 100 - 3.0 and lp[2] < back[2] < lead.pos["neck_01"][2]
    sh_ = seen_by_lead(fol.pos["hand_l"])                        # the follower's hand: at the leader's right shoulder
    assert math.dist(sh_[:2], lead.ref_pos["upperarm_r"][:2]) < 16.0 and 0.0 < sh_[2] - lead.ref_pos["upperarm_r"][2] < 14.0
    assert lead.info["hold"]["partner"] == "like self"


def test_the_hold_is_aimed_at_the_partners_own_body_whatever_their_size_and_place():
    """A smaller partner who stands a little off the line: the joined wrists are still a palm apart at one height,
    the leader's hand is on the partner's back below their shoulder, the follower's on top of the leader's shoulder
    (not at a height of their own, nor past it in the air)."""
    for small, off in ((1.0, (0.0, 0.0)), (0.88, (0.0, 0.0)), (0.88, (6.0, 6.0)), (1.08, (-5.0, 4.0))):
        lead, fol, to_lead, P, Pf = hold_pair(38.0, small, off)
        jl, jf = lead.pos["hand_l"], to_lead(fol.pos["hand_r"])
        assert 4.0 < math.dist(jl, jf) < 12.0 and abs(jl[2] - jf[2]) < 1.0, (small, off, jl, jf)
        fsh = to_lead(Pf["upperarm_l"])
        assert 0.0 < jl[2] - fsh[2] < 12.0                                       # at the follower's shoulder height
        back, spine = lead.pos["hand_r"], to_lead(Pf["spine_03"])
        assert back[1] > spine[1] - 2.0 and 8.0 < fsh[2] - back[2] < 26.0, (small, off, back, spine)   # round their side
        assert 6.0 < spine[0] - back[0] < 20.0                                   # (their left: the leader's right)
        assert lead.pos["middle_01_r"][1] > back[1] and lead.pos["middle_01_r"][0] > back[0]   # the hand goes on round the back
        top, sh = to_lead(fol.pos["hand_l"]), P["upperarm_r"]
        assert math.dist(top[:2], sh[:2]) < 16.0 and 4.0 < top[2] - sh[2] < 14.0, (small, off, top, sh)
        tip = to_lead(fol.pos["middle_01_l"])
        assert math.dist(tip[:2], sh[:2]) < math.dist(top[:2], sh[:2]) + 1.0      # the hand lies toward the shoulder
        assert lead.info["hold"]["partner"] == "scene"


def test_the_pass_pairs_give_the_distance_and_the_stance():
    sp = {"near": {"s2": [["Orla", "Tam", "dancing together"]], "s3": [["Orla", "Dr. Marsh", "close"]]}}
    assert FP.near_of("Orla", ["s1", "s2"], sp) == ("Tam", "dancing together")
    assert FP.near_of("Tam", ["s2"], sp) == ("Orla", "dancing together")
    assert FP.near_of("Orla", ["s1"], sp) == (None, None) and FP.near_of("Orla", ["s2"], {}) == (None, None)
    assert FP.near_position([2.0, 0.0], [0.0, 0.0], "dancing together") == [0.38, 0.0]
    assert FP.near_position([0.0, 0.3], [0.0, 0.0], "close") == [0.0, 0.8]             # farther too, when placed nearer
    assert FP.near_position([2.0, 0.0], [0.0, 0.0], "conversing") == [2.0, 0.0]        # the placing's own distance
    assert FP.NEAR_GAP_M["dancing together"] < FP.NEAR_GAP_M["touching"] < FP.NEAR_GAP_M["close"] < FP.TALK_M[0]
    a, b = FP.near_pose("dancing together", True), FP.near_pose("dancing together", False)
    assert a["pose"] == b["pose"] == "dance_hold" and (a["params"]["role"], b["params"]["role"]) == ("lead", "follow")
    assert FP.near_pose("touching", True)["pose"] == "hand_joined" and FP.near_pose("close", True) is None


def test_the_question_tells_where_the_shot_has_the_two_and_offers_every_band():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import staging_pass as SPS
    sh = {"action": "Orla looks up.", "people": [{"name": "Orla", "pose": "standing", "at": "in the reel"},
                                                   {"name": "Tam", "pose": "standing", "at": "in the reel"},
                                                   {"name": "Dr. Marsh", "pose": "sitting", "at": "by the wall"}]}
    q = SPS.q_near("They spoke.", sh, "Orla", "Tam")
    assert "- Orla: standing, in the reel" in q and "- Tam: standing, in the reel" in q and "Dr. Marsh" not in q
    assert "still going on" in q and q.rstrip().endswith("; ".join(SPS.NEARS)) and "\\n" not in q
    assert set(FP.NEAR_GAP_M) < set(SPS.NEARS)


def test_two_the_plan_has_in_a_dance_are_partners_without_asking():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import staging_pass as SPS
    sh = {"people": [{"name": "Orla", "pose": "standing", "at": "in the reel"},
                     {"name": "Tam", "pose": "dancing", "at": "on the floor"},
                     {"name": "Dr. Marsh", "pose": "sitting", "at": "by the wall"}]}
    assert SPS.in_a_dance(sh, "Orla", "Tam") and not SPS.in_a_dance(sh, "Orla", "Dr. Marsh")
    assert not SPS.in_a_dance({"people": [{"name": "Orla", "at": "by the abundance of flowers"},
                                           {"name": "Tam", "at": "by the door"}]}, "Orla", "Tam")


def test_two_who_dance_go_on_dancing_through_shots_that_do_not_say():
    sp = {"near": {"s2": [["Orla", "Tam", "dancing together"]], "s5": [["Orla", "Tam", "apart"]]}}
    order = ["s1", "s2", "s3", "s4", "s5", "s6"]
    assert FP.near_of("Orla", ["s3", "s4"], sp, order) == ("Tam", "dancing together")
    assert FP.near_of("Orla", ["s6"], sp, order) == ("Tam", "apart")              # the latest said
    assert FP.near_of("Orla", ["s1"], sp, order) == (None, None)                   # nothing before it
    assert FP.near_of("Orla", ["s3"], sp) == (None, None)
