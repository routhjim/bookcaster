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


def test_in_a_dance_hold_the_joined_hands_meet_and_the_others_rest_on_the_partner():
    gap = 0.45
    lead, fol = posed("dance_hold", role="lead", gap_m=gap), posed("dance_hold", role="follow", gap_m=gap)
    lp, fp = lead.pos["pelvis"], fol.pos["pelvis"]

    def seen_by_lead(q):            # a point of the follower (turned about, gap ahead) in the leader's frame
        return [lp[0] - (q[0] - fp[0]), lp[1] + gap * 100 - (q[1] - fp[1]), q[2]]
    assert math.dist(lead.pos["hand_l"], seen_by_lead(fol.pos["hand_r"])) < 12.0           # the joined hands
    assert lead.pos["hand_l"][0] - lp[0] > 20.0 and lead.pos["hand_l"][2] > lp[2] + 20.0   # out to the side, raised
    back = lead.pos["hand_r"]                                    # the leader's hand: past the partner's middle
    assert back[1] - lp[1] > gap * 100 - 5.0 and lp[2] < back[2] < lead.pos["neck_01"][2]
    sh_ = seen_by_lead(fol.pos["hand_l"])                        # the follower's hand: at the leader's right shoulder
    assert math.dist(sh_[:2], lead.pos["clavicle_r"][:2]) < 25.0 and abs(sh_[2] - lead.pos["clavicle_r"][2]) < 20.0


def test_the_pass_pairs_give_the_distance_and_the_stance():
    sp = {"near": {"s2": [["Orla", "Tam", "dancing together"]], "s3": [["Orla", "Dr. Marsh", "close"]]}}
    assert FP.near_of("Orla", ["s1", "s2"], sp) == ("Tam", "dancing together")
    assert FP.near_of("Tam", ["s2"], sp) == ("Orla", "dancing together")
    assert FP.near_of("Orla", ["s1"], sp) == (None, None) and FP.near_of("Orla", ["s2"], {}) == (None, None)
    assert FP.near_position([2.0, 0.0], [0.0, 0.0], "dancing together") == [0.45, 0.0]
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
