"""11-27: a hand that only carries something is wanted in an action shot's frame, and given up where no camera holds
both it and the frame (scenes/speakers.py two_shot_candidates, "joints_soft"); a working hand is never given up. A
man with a cup, two people close beside him: every medium that showed his hand at the thigh took the others in, every
one that left them out cut the hand, and the shot had no camera at all. Made-up scenes only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import speakers as SP  # noqa: E402

A = Path(__file__).resolve().parents[1] / "assets"
ROOM = [6.0, 6.0, 3.0]


def scene(monkeypatch):
    """One person facing the open room; a framing rule set that refuses every camera asked to show a joint."""
    subj = {"a": S.person([0.0, 1.0], facing=[0, -1])}
    plain = SP.framing_why

    def why(q, *a, **k):
        w, x = plain(q, *a, **dict(k, opts=dict(k.get("opts") or {}, joints=[])))
        return (w + ["a's hand_r (the action) is not in the frame"] if (k.get("opts") or {}).get("joints") else w), x
    monkeypatch.setattr(SP, "framing_why", why)
    shot = {"camera": S.cam([0.0, -1.5, 1.5], [0.0, 1.0, 1.5], 35)}
    return subj, shot


def search(c, subj, shot):
    return SP.two_shot_candidates(c, shot, None, None, subj, S.RES, ROOM, None, None, [], SP.cam_opts(
        {"characters": {"a": {}}}, c))


def test_a_carrying_hand_no_camera_can_show_is_let_out_of_the_frame(monkeypatch):
    subj, shot = scene(monkeypatch)
    c = {"name": "s", "kind": "medium", "show": ["a"], "subject": "a", "show_joints": [["a", "hand_r"]],
         "joints_soft": True}
    assert search(c, subj, shot)
    assert c["show_joints"] == []            # (the camera's own list: the checks after and the redraw read it)
    assert SP.cam_opts({"characters": {"a": {}}}, c)["joints"] == []


def test_a_working_hand_is_never_given_up(monkeypatch):
    subj, shot = scene(monkeypatch)
    c = {"name": "s", "kind": "medium", "show": ["a"], "subject": "a", "show_joints": [["a", "hand_r"]]}
    assert search(c, subj, shot) == []
    assert c["show_joints"] == [["a", "hand_r"]]


def test_a_hand_a_camera_can_show_stays_asked_for():
    subj = {"a": S.person([0.0, 1.0], facing=[0, -1])}
    c = {"name": "s", "kind": "medium", "show": ["a"], "subject": "a", "show_joints": [["a", "hand_r"]],
         "joints_soft": True}
    shot = {"camera": S.cam([0.0, -1.5, 1.5], [0.0, 1.0, 1.5], 35)}
    if search(dict(c, joints_soft=False), subj, shot):       # (alone in a room, a medium holds his hand)
        assert search(c, subj, shot) and c["show_joints"] == [["a", "hand_r"]]


def test_the_plan_marks_only_carrying_hands_and_the_redraw_says_nothing_of_a_hand_let_out():
    fp = (A / "scenes" / "from_plan.py").read_text()
    assert 'if not any(poses.get(k_) in WORK_ACTIONS for k_, _ in hj_):\n                    cam["joints_soft"] = True' in fp
    rd = (A / "redraw" / "redraw.py").read_text()
    assert 'if c.get("joints_soft") and not c.get("show_joints"):\n        return ""' in rd
