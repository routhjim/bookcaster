"""11-27: someone who plays an instrument at the shoulder holds it there (scenes/ue/pose.py act_play_at_shoulder), on
the synthetic 1.75 m humanoid. A player's instrument hung at his side in every shot: out of every frame closer than a
full one. Made-up sizes only."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_held_props_1118 as T  # noqa: E402

PO = T.PO
BODY = [0.23, 0.245, 0.6]              # a long thing with a broad body at one end
STICK = [0.75, 0.03, 0.02]


def played(things):
    return PO.pose_skel(T.skel(), "stand_relaxed", {"action": "play_at_shoulder", "held": things})


def test_the_instrument_lies_from_the_left_hand_to_the_left_collar_and_what_plays_it_across_it():
    s = played([T.held("lute", BODY, "r"), T.held("stick", STICK, "l")])     # (whichever hands the scene gave)
    a, b = s.info["held"]["lute"], s.info["held"]["stick"]
    assert (a["hand"], a["how"], b["hand"], b["how"]) == ("l", "play", "r", "play")
    left, fwd = PO._body_axes(s)
    sh_l, sh_r = s.pos["upperarm_l"], s.pos["upperarm_r"]
    hl, hr = s.pos["hand_l"], s.pos["hand_r"]
    # the left hand out in front and to the left, about shoulder height; the elbow under it
    assert PO.dot(PO.sub(hl, sh_l), fwd) > 25 and PO.dot(PO.sub(hl, sh_l), left) > 3 and abs(hl[2] - sh_l[2]) < 15
    assert s.pos["lowerarm_l"][2] < hl[2] - 8
    put, palm = T.placed(s, a)
    assert PO.held_contact(put, palm)["grip_cm"] <= 2.0
    far = T.far_end(put)                                   # its body's end: on the left collar, under the chin
    assert PO.norm(PO.sub(far, sh_l)) < 22 and far[2] < s.pos["head"][2] and far[2] > sh_l[2] - 6
    # the right hand before the chest, right of the instrument; what it holds reaches across the instrument
    assert PO.dot(PO.sub(hr, sh_r), fwd) > 12 and PO.dot(PO.sub(hr, hl), left) < -15 and hr[2] > sh_r[2] - 20
    put_b, palm_b = T.placed(s, b)
    assert PO.held_contact(put_b, palm_b)["grip_cm"] <= 2.0
    assert min(PO.seg_dist(q, palm_b, T.far_end(put_b)) for q in (PO.add(palm, PO.scale(PO.sub(far, palm), f / 10))
                                                               for f in range(2, 9))) < 12


def test_with_one_thing_it_is_the_instrument_and_with_none_the_arms_still_play():
    s = played([T.held("lute", BODY, "r")])
    assert s.info["held"]["lute"]["hand"] == "l" and len(s.info["held"]) == 1
    s0 = played([])
    assert not (s0.info.get("held") or {})
    assert s0.pos["hand_l"][2] > s0.pos["upperarm_l"][2] - 15 and s0.pos["hand_r"][2] > s0.pos["upperarm_r"][2] - 20
    assert "play_at_shoulder" in PO.ACTIONS


def test_someone_who_plays_goes_on_playing_and_keeps_both_things():
    for sub in ("scenes", ""):
        sys.path.insert(0, str(HERE.parent / "assets" / sub))
    import from_plan as FP
    import staging_pass as SPASS
    P = FP.PLAY_ACTIONS[0]
    assert P in FP.ACTION_NAMES and P in PO.ACTIONS and P in SPASS.PLAY_POSES and P in SPASS.POSE_WORDS
    mem = {}
    assert FP.played_on(mem, {"c": P}, {"a", "c"}, {"c"}) == {"c": P}
    assert FP.played_on(mem, {}, {"a", "c"}, {"c"}) == {"c": P}                  # (another's shot: still playing)
    assert FP.played_on(mem, {"c": "none", "a": "gesture_point"}, {"a", "c"}, {"c"}) == {"c": P, "a": "gesture_point"}
    assert FP.played_on(mem, {"c": "gesture_point"}, {"a", "c"}, {"c"}) == {"c": "gesture_point"}   # (another action)
    assert FP.played_on(mem, {}, {"a", "c"}, {"c"}) == {}
    FP.played_on(mem, {"c": P}, {"a", "c"}, {"c"})
    assert FP.played_on(mem, {}, {"a", "c"}, set()) == {} and mem == {}          # (the hands are empty)
    assert FP.played_on(mem, {"c": P}, {"a", "c"}, set()) == {}                  # (nothing to play)
    FP.played_on(mem, {"c": P}, {"a", "c"}, {"c"})
    assert FP.played_on(mem, {}, {"a"}, {"c"}) == {}                             # (gone from the place)
    lute = {"asset": "x1", "at": "hand", "hand": "r", "name": "a lute"}
    stick = {"asset": "x2", "at": "hand", "hand": "l", "name": "a stick"}
    cap = {"asset": "h", "at": "head", "name": "a cap"}
    assert FP.play_hands([cap, lute, stick]) == [cap, dict(lute, hand="l"), dict(stick, hand="r")]
    # the pass names both once, then the one the line is about: the other is in the other hand yet
    held = {}
    FP.held_on(held, {"c": [lute, stick]}, {"c"}, {"c"})
    assert FP.held_on(held, {"c": [lute]}, {"c"}, {"c"}) == {"c": [lute, stick]}
    assert FP.held_on(held, {"c": [cap, lute]}, {"c"}, {"c"}) == {"c": [cap, lute, stick]}
    other = {"asset": "x9", "at": "hand", "hand": "r", "name": "a cup"}
    assert FP.held_on(held, {"c": [other]}, {"c"}, {"c"}) == {"c": [other]}      # (another thing: the hands changed)
    assert FP.action_misfit(P, "c", {"pose": "standing"}, {}, None, [], []) and \
        FP.action_misfit(P, "c", {"pose": "standing"}, {}, None, [], [lute]) is None
