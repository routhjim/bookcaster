"""Two who hold each other by the hand: the staging pass's "joined" pairs get the hand-joined stance, the hands
meeting half way between them (scenes/ue/pose.py pose_hand_joined, from_plan joined_with / joined_pose), on
synthetic data."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "assets" / "sets"))
import staging_synth as S  # noqa: E402,F401
import from_plan as FP  # noqa: E402
from test_work_pose_1117 import PO, humanoid  # noqa: E402


def posed(hand="r", reach=0.5):
    P, par = humanoid()
    s = PO.Skel.from_joints(P, par)
    return PO.pose_skel(s, "hand_joined", {"hand": hand, "reach_m": reach, "weight_on": "l"})


def test_the_joined_hand_is_held_out_ahead_between_waist_and_chest():
    for hand in ("r", "l"):
        s = posed(hand)
        h, pel, other = s.pos[f"hand_{hand}"], s.pos["pelvis"], s.pos["hand_l" if hand == "r" else "hand_r"]
        assert h[1] - pel[1] > 35.0, (hand, h, pel)                    # well ahead of the body
        assert abs(h[0] - pel[0]) < 15.0                               # on the line between the two
        assert pel[2] < h[2] < s.pos["neck_01"][2]                     # above the hips, below the neck
        assert other[2] < pel[2] + 10.0 and abs(other[1] - pel[1]) < 30.0   # the other arm hangs


def test_a_nearer_partner_brings_the_hand_nearer():
    near, far = posed(reach=0.35), posed(reach=0.55)
    assert near.pos["hand_r"][1] < far.pos["hand_r"][1]


def test_the_pass_pairs_give_both_the_stance():
    sp = {"joined": {"s2": [["Orla", "Tam"]]}}
    assert FP.joined_with("Orla", ["s1", "s2"], sp) == "Tam" and FP.joined_with("Tam", ["s2"], sp) == "Orla"
    assert FP.joined_with("Orla", ["s1"], sp) is None and FP.joined_with("Dr. Marsh", ["s2"], sp) is None
    assert FP.joined_with("Orla", ["s2"], {}) is None
    jp = FP.joined_pose([0.0, 0.0], [1.0, 0.0])
    assert jp["pose"] == "hand_joined" and jp["params"]["reach_m"] == 0.5
    assert FP.joined_pose([0.0, 0.0], [3.0, 0.0])["params"]["reach_m"] == FP.JOIN_REACH_M


def test_the_question_tells_where_the_shot_has_the_two():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import staging_pass as SPS
    sh = {"action": "Orla looks up.", "people": [{"name": "Orla", "pose": "standing", "at": "in the reel"},
                                                   {"name": "Tam", "pose": "standing", "at": "in the reel"},
                                                   {"name": "Dr. Marsh", "pose": "sitting", "at": "by the wall"}]}
    q = SPS.q_joined("They spoke.", sh, "Orla", "Tam")
    assert "- Orla: standing, in the reel" in q and "- Tam: standing, in the reel" in q and "Dr. Marsh" not in q
    assert "still going on" in q and q.rstrip().endswith(f"{SPS.JOINED}; {SPS.APART}")
