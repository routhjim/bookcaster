"""11-27: someone who lifts what is in their hand (scenes/ue/pose.py act_raise_in_hand), on the synthetic 1.75 m
humanoid. The only raised hand the staging pass could choose was a tool wound up over a piece of work: a person
lifting a mug was stood at the nearest bench and folded over it. Made-up sizes only."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_held_props_1118 as T  # noqa: E402

PO = T.PO
MUG = [0.09, 0.08, 0.1]


def raised(things):
    return PO.pose_skel(T.skel(), "stand_relaxed", {"action": "raise_in_hand", "held": things})


def test_the_hand_with_the_thing_is_up_in_front_at_about_eye_height_and_holds_it():
    for side, other in (("r", "l"), ("l", "r")):
        s = raised([T.held("mug", MUG, side)])
        left, fwd = PO._body_axes(s)
        sh, h = s.pos[f"upperarm_{side}"], s.pos[f"hand_{side}"]
        assert PO.dot(PO.sub(h, sh), fwd) > 20 and -14 <= h[2] - s.pos["head"][2] <= 0
        assert s.pos[f"lowerarm_{side}"][2] < h[2] - 8                       # (the elbow under the hand)
        plain = PO.pose_skel(T.skel(), "stand_relaxed", {"held": [T.held("mug", MUG, side)]})
        assert PO.norm(PO.sub(s.pos[f"hand_{other}"], plain.pos[f"hand_{other}"])) < 1.0   # (the other arm as it was)
        assert h[2] > plain.pos[f"hand_{side}"][2] + 30
        a = s.info["held"]["mug"]
        put, palm = T.placed(s, a)
        assert a["hand"] == side and PO.held_contact(put, palm)["grip_cm"] <= 2.0


def test_with_nothing_held_the_right_hand_lifts_and_the_name_is_known_everywhere():
    s = raised([])
    assert s.pos["hand_r"][2] >= s.pos["upperarm_r"][2] and not (s.info.get("held") or {})
    for sub in ("scenes", "redraw", ""):
        sys.path.insert(0, str(HERE.parent / "assets" / sub))
    import from_plan as FP
    import staging_pass as SPASS
    import speakers as SP
    R = FP.RAISE_ACTIONS[0]
    assert R in PO.ACTIONS and R in FP.ACTION_NAMES and R in SPASS.RAISE_POSES and R in SPASS.POSE_WORDS
    assert R in SP.PLAY_HANDS and R not in FP.WORK_ACTIONS                    # (no piece of work to stand at)
    assert R in (HERE.parent / "assets" / "scenes" / "scene.py").read_text()
    import redraw as RD
    m = {"joints": {"hand_r": [0.0, -0.4, 1.6]}, "head_top_z": 1.78, "chin_z": 1.55, "waist_z": 1.0,
         "pelvis_xy": [0.0, 0.0], "facing": [0.0, -1.0]}
    assert "height of the face" in RD.hand_place(m)
    m["joints"]["hand_r"] = [0.0, -0.4, 1.3]
    assert "chest height" in RD.hand_place(m)
