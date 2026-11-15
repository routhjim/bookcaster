"""The staging and framing rules as decided on 11-11 and 11-12, each on a small synthetic scene (stdlib).

Speaker rules are hard; people who are not speaking need only be in the picture; the person a shot is about is
treated like a speaker; a camera stands inside the room; a seat in view is not left reading as empty; a move shot
holds its start, path and end place. Bars come from the code (speakers.MAX_HEAD_DEG and the rest), so a deliberate
change of a bar moves the case with it and a change of a rule fails one named test.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import speakers as SP  # noqa: E402

RES = S.RES
ROOM = [6.0, 5.0, 3.0]


def _why(camera, kind, speaker, subjects, listener=None, ots=False, show=None, shot=None, opts=None):
    return SP.framing_why(camera, kind, speaker, listener, ots, show, subjects, RES, shot, opts)[0]


def _shot(tmp_path, camera, subjects, name="s", hide=()):
    return {"camera": camera, "guides": S.depth_guides(tmp_path / "g", name, camera, subjects, hide=hide)}


def _turned(deg):
    """A head direction turned deg from facing -y (toward +x)."""
    a = math.radians(deg)
    return [math.sin(a), -math.cos(a)]


# ------------------------------------------------------------------ the speaker: hard rules
def test_speaker_facing_the_camera_passes():
    subj = {"a": S.person([0, 0])}
    c = S.front_cam(subj["a"], 3.0)
    chk = SP.check_shot({"camera": c}, "a", subj, RES, "medium", None, None)
    assert chk["ok"], chk["why"]
    assert chk["head_deg"] < 1 and chk["face_h"] >= SP.TALK_FACE_H["medium"]


def test_speaker_head_turned_past_the_bar_fails_and_just_inside_passes():
    cam0 = S.front_cam(S.person([0, 0]), 3.0)
    over = {"a": S.person([0, 0], head=_turned(SP.MAX_HEAD_DEG + 5))}
    chk = SP.check_shot({"camera": cam0}, "a", over, RES, "medium", None, None)
    assert not chk["ok"] and any("deg from the camera" in w for w in chk["why"])
    under = {"a": S.person([0, 0], head=_turned(SP.MAX_HEAD_DEG - 5))}
    chk = SP.check_shot({"camera": cam0}, "a", under, RES, "medium", None, None)
    assert not any("deg from the camera" in w for w in chk["why"])


def test_speaker_face_too_small_for_a_talk_shot_fails():
    subj = {"a": S.person([0, 0])}
    far = S.front_cam(subj["a"], 9.0)
    assert SP.head_h(far, subj["a"]["measured"], RES) < SP.TALK_FACE_H["medium"]
    assert any("speaker's face" in w for w in _why(far, "medium", "a", subj))
    near = S.front_cam(subj["a"], 3.0)
    assert not _why(near, "medium", "a", subj)


def test_speaker_whole_head_and_headroom_inside_the_frame():
    subj = {"a": S.person([0, 0])}
    low = S.cam([0, -1.2, 1.0], [0, 0, 1.0], 50)          # aimed at the chest: the head top is above the frame
    w = _why(low, "medium", "a", subj)
    assert any("head or its headroom is outside the frame" in x for x in w)
    # the head rule is the speaker's alone: the same framing of a person who is not speaking does not fail on it
    two = {"a": S.person([0, 0]), "b": S.person([3.0, 0.0])}
    assert not any("headroom" in x for x in _why(low, "medium", None, two))


def test_speaker_face_covered_by_someone_in_front_fails(tmp_path):
    subj = {"a": S.person([0, 0]), "b": S.person([0, -1.2], facing=[0, -1])}
    c = S.front_cam(subj["a"], 3.0)
    sh = _shot(tmp_path, c, subj)
    chk = SP.check_shot(sh, "a", subj, RES, "medium", None, None)
    assert chk["face_visible"] < SP.MIN_FACE_VISIBLE and any("visible" in w for w in chk["why"])
    # the planner's own test agrees before any render: b's body is between the camera and a's eyes
    m = subj["a"]["measured"]
    assert SP.blocked(c["position"], [0, 0, m["eye_z"]], [subj["b"]["measured"]])
    alone = {"a": subj["a"]}
    chk = SP.check_shot(_shot(tmp_path, c, alone, "s2"), "a", alone, RES, "medium", None, None)
    assert chk["face_visible"] >= SP.MIN_FACE_VISIBLE and chk["ok"], chk["why"]


def test_every_other_head_is_smaller_than_the_speakers():
    subj = {"a": S.person([0, 0]), "b": S.person([0.6, -2.0], facing=[0, 1])}    # b nearer the camera
    c = S.cam([0.3, -3.2, 1.55], [0.2, 0, 1.5], 35)
    w = _why(c, "medium", "a", subj)
    assert any("bigger than the speaker's" in x for x in w)
    behind = {"a": S.person([0, 0]), "b": S.person([0.7, 1.5])}                  # b further away: smaller head
    assert not any("bigger" in x for x in _why(c, "medium", "a", behind))


def test_speaker_cut_by_the_frame_side_fails_in_a_talk_shot(tmp_path):
    subj = {"a": S.person([0, 0])}
    c = S.cam([1.3, -3.0, 1.5], [1.3, 0, 1.5], 50)        # a at the left edge, half out
    sh = _shot(tmp_path, c, subj)
    w = _why(c, "medium", "a", subj, shot=sh)
    assert any("cut by the frame's" in x or "headroom" in x for x in w), w


# ------------------------------------------------------------------ people who are not speaking: only in the picture
def test_non_speaker_in_profile_from_behind_or_partly_hidden_is_accepted(tmp_path):
    for facing in ([1, 0], [0, 1]):                      # in profile; back to the camera
        subj = {"a": S.person([0, 0]), "b": S.person([1.1, 2.0], facing=facing)}
        c = S.cam([0.3, -3.6, 1.55], [0.35, 0, 1.45], 40)
        sh = _shot(tmp_path, c, subj, f"p{facing[0]}{facing[1]}")
        chk = SP.check_shot(sh, "a", subj, RES, "medium", None, None, show=["a", "b"])
        assert chk["ok"], (facing, chk["why"])
    # one man partly behind the other: his head inside the frame, some of his own pixels seen
    subj = {"a": S.person([0, 0]), "b": S.person([0.25, 0.6])}
    c = S.cam([0.0, -3.4, 1.55], [0.1, 0, 1.45], 40)
    sh = _shot(tmp_path, c, subj, "behind")
    w, facts = SP.framing_why(c, "medium", "a", None, False, ["a", "b"], subj, RES, sh, None)
    assert 0 < facts["b"]["mask"]["area"] and SP.in_picture(facts["b"])
    assert not w, w


def test_named_person_missing_from_the_picture_fails():
    subj = {"a": S.person([0, 0]), "b": S.person([4.0, 0.0])}
    c = S.front_cam(subj["a"], 3.0)
    w = _why(c, "medium", None, subj, show=["a", "b"])
    assert any("b (named for the shot) is not in the picture" in x for x in w)
    assert not _why(c, "medium", None, subj, show=["a"])   # not named: out of frame is no fault


def test_face_cut_to_a_sliver_fails_and_a_larger_share_or_a_back_does_not():
    spk = S.person([0, 0])
    c = S.front_cam(spk, 3.0)
    half_w = math.tan(framing_hfov(c) / 2) * 3.0           # the frame's half width at the people's distance
    # b's head is 0.18 m wide (head_share): its centre so that about a fifth of it is inside the right edge
    def at_share(share, facing=(0, -1)):
        x = half_w - 0.09 + 0.18 * (1 - share)
        return {"a": spk, "b": S.person([x, 0.0], facing=facing)}
    low = at_share(SP.SLIVER * 0.5)
    sl = SP.face_sliver(c, low["b"]["measured"], RES)
    assert sl is not None and 0 < sl < SP.SLIVER
    assert any(x.startswith("PARTIAL FACE") for x in _why(c, "medium", "a", low))
    high = at_share(min(0.95, SP.SLIVER + 0.3))
    assert SP.face_sliver(c, high["b"]["measured"], RES) is None
    assert not any(x.startswith("PARTIAL FACE") for x in _why(c, "medium", "a", high))
    back = at_share(SP.SLIVER * 0.5, facing=(0, 1))        # turned away: a back at the edge is not a cut face
    assert not any(x.startswith("PARTIAL FACE") for x in _why(c, "medium", "a", back))


def framing_hfov(c):
    import framing
    return framing.hfov(c["focal_mm"])


def test_foreground_back_over_its_share_fails_only_in_a_talk_shot_and_only_on_the_render(tmp_path):
    subj = {"a": S.person([0, 0]), "b": S.person([0.75, -2.2], facing=[0, 1])}    # b near the camera, back to it
    c = S.cam([0.0, -3.4, 1.5], [0.0, 0, 1.45], 35)
    sh = _shot(tmp_path, c, subj)
    w, facts = SP.framing_why(c, "medium", "a", None, False, None, subj, RES, sh, None)
    assert facts["b"]["mask"]["area"] > SP.FG_AREA
    assert any("turned away, covers" in x for x in w)
    # a candidate camera (no still yet): the body estimate only orders candidates, it never rejects one
    assert not any("turned away, covers" in x for x in _why(c, "medium", "a", subj))
    assert SP.area_excess(c, subj, RES, {"a"}) > 0
    # a shot without a speaker: a back in the foreground is a staging choice
    assert not any("turned away, covers" in x for x in _why(c, "medium", None, subj, shot=sh))
    # a small back further off, under the bar
    small = {"a": S.person([0, 0]), "b": S.person([1.2, 2.0], facing=[0, 1])}
    sh2 = _shot(tmp_path, c, small, "small")
    w2, f2 = SP.framing_why(c, "medium", "a", None, False, None, small, RES, sh2, None)
    assert f2["b"]["mask"]["area"] <= SP.FG_AREA and not any("turned away" in x for x in w2)


def test_the_shots_subject_is_not_counted_as_a_foreground_back(tmp_path):
    subj = {"a": S.person([0, 0]), "b": S.person([0.75, -2.2], facing=[0, 1])}
    c = S.cam([0.0, -3.4, 1.5], [0.0, 0, 1.45], 35)
    sh = _shot(tmp_path, c, subj)
    w = _why(c, "medium", "a", subj, shot=sh, opts={"subject": "b"})
    assert not any("turned away, covers" in x for x in w)


# ------------------------------------------------------------------ the person a shot is about: like a speaker
def test_subject_of_a_quiet_shot_needs_a_face_shown(tmp_path):
    subj = {"b": S.person([0, 0])}
    c = S.front_cam(subj["b"], 3.0)
    assert SP.subject_face_why(_shot(tmp_path, c, subj), "b", subj, RES) == []
    away = {"b": S.person([0, 0], facing=[0, 1])}                     # the back of her head to the camera
    assert any("turned from the camera" in x for x in SP.subject_face_why({"camera": c}, "b", away, RES))
    # facing the camera but covered by someone nearer
    two = {"b": S.person([0, 0]), "a": S.person([0, -1.2])}
    assert any("visible" in x for x in SP.subject_face_why(_shot(tmp_path, c, two, "cov"), "b", two, RES))


def test_a_close_stays_a_close():
    subj = {"b": S.person([0, 0])}
    far = S.front_cam(subj["b"], 3.5)
    assert SP.head_h(far, subj["b"]["measured"], RES) < SP.MIN_FACE_H["close"]
    assert any("not a close" in x for x in _why(far, "close", None, subj, opts={"subject": "b"}))
    near = S.front_cam(subj["b"], 0.9, focal=65)
    assert SP.head_h(near, subj["b"]["measured"], RES) >= SP.MIN_FACE_H["close"]
    assert not any("not a close" in x for x in _why(near, "close", None, subj, opts={"subject": "b"}))


# ------------------------------------------------------------------ listeners, seats, the room
def test_listener_stays_in_view_unless_seated_and_never_over_a_high_chair_back():
    subj = {"a": S.person([0, 0]), "b": S.person([4.0, 0.0])}
    c = S.front_cam(subj["a"], 3.0)
    assert any("the listener is not in view" in x for x in _why(c, "medium", "a", subj, listener="b"))
    seated = {"a": subj["a"], "b": S.person([4.0, 0.0], seated=True)}
    w = _why(c, "medium", "a", seated, listener="b", opts={"seated": {"b"}})
    assert not any("listener" in x for x in w)             # a clean single of the speaker is allowed
    w = _why(c, "medium", "a", seated, listener="b", ots=True, opts={"seated": {"b"}})
    assert any("over the shoulder of a listener seated" in x for x in w)


def test_a_seat_in_view_with_its_sitter_hidden_reads_as_empty(tmp_path):
    subj = {"a": S.person([0, 0]), "b": S.person([0.0, 2.5], seated=True)}      # b directly behind the speaker
    c = S.front_cam(subj["a"], 3.0)
    sh = _shot(tmp_path, c, subj, hide=("b",))
    w = _why(c, "medium", "a", subj, shot=sh, opts={"seated": {"b"}, "seat_xy": {"b": [0.0, 2.5]}})
    assert any("b's seat is in view with b hidden" in x for x in w)
    # her chair out of the frame: no empty chair to read as hers
    off = {"a": subj["a"], "b": S.person([5.0, 2.5], seated=True)}
    sh2 = _shot(tmp_path, c, off, "off")
    assert not any("seat is in view" in x for x in
                   _why(c, "medium", "a", off, shot=sh2, opts={"seated": {"b"}, "seat_xy": {"b": [5.0, 2.5]}}))
    # seen in her chair: fine
    side = {"a": subj["a"], "b": S.person([1.0, 1.0], seated=True)}
    sh3 = _shot(tmp_path, c, side, "side")
    assert not any("seat is in view" in x for x in
                   _why(c, "medium", "a", side, shot=sh3, opts={"seated": {"b"}, "seat_xy": {"b": [1.0, 1.0]}}))


def test_camera_stands_inside_the_room_and_outside_every_body():
    subj = {"a": S.person([0, 2.0])}
    inside = S.cam([0, 0, 1.5], [0, 2.0, 1.5])
    assert SP.placement_why(inside, subj, ROOM) == []
    outside = S.cam([0, ROOM[1] / 2 + 0.4, 1.5], [0, 2.0, 1.5])          # behind the wall the face is turned to
    assert "the camera is outside the room" in SP.placement_why(outside, subj, ROOM)
    assert not SP.in_room(outside["position"], ROOM) and SP.in_room(inside["position"], ROOM)
    in_body = S.cam([0.05, 2.05, 1.4], [0, 0, 1.4])
    assert any(x == "the camera is inside a" for x in SP.placement_why(in_body, subj, ROOM))


# ------------------------------------------------------------------ move shots and a thing turned toward
LM = {"name": "fireplace", "center": [2.0, 2.3, 0.6], "top": 1.2, "front": [0.0, -1.0]}


def test_a_thing_the_subject_turns_toward_is_in_the_frame_seen_from_its_front():
    subj = {"a": S.person([0.5, 0.5], facing=[1, 1])}
    seen = S.cam([0.0, -2.0, 1.5], [1.2, 1.4, 1.1], 28)
    assert not any("fireplace" in x for x in _why(seen, "medium", None, subj, opts={"landmark": LM}))
    beside = S.cam([-2.0, 0.5, 1.5], [0.5, 0.5, 1.3], 50)             # the thing is off the frame's right
    assert any("what the shot turns toward) is out of the frame" in x for x in
               _why(beside, "medium", None, subj, opts={"landmark": LM}))
    lm_mid = dict(LM, center=[0.0, 0.0, 0.6], front=[0.0, -1.0])
    behind = S.cam([0.0, 2.0, 1.5], [0.0, 0.0, 0.8], 28)               # looking at it from behind its front
    assert any("seen from its side or back" in x for x in _why(behind, "medium", None, subj, opts={"landmark": lm_mid}))


def test_a_move_shot_holds_start_path_and_end_and_the_mover_reads():
    subj = {"b": S.person([-1.5, 0.0], facing=[0, -1]), "a": S.person([-2.3, 0.8], facing=[1, -1])}
    pts = [[-1.5, 0.0, 1.65], [-1.5, 0.0, 0.05], [1.5, 0.5, 1.0], [1.5, 0.5, 0.05], [0.0, 0.25, 0.8]]
    wide = S.cam([0.0, -2.1, 1.5], [0.0, 0.3, 1.0], 20)
    w = _why(wide, "full", None, subj, opts={"points": pts, "mover": "b"})
    assert not w, w
    tight = S.cam([-1.5, -2.1, 1.5], [-1.5, 0.0, 1.3], 50)             # framed on the start alone
    w = _why(tight, "full", None, subj, opts={"points": pts, "mover": "b"})
    assert any("the move's place" in x and "out of the frame" in x for x in w)
    turned = {"b": S.person([-1.5, 0.0], facing=[0, 1]), "a": subj["a"]}
    w = _why(wide, "full", None, turned, opts={"points": pts, "mover": "b"})
    assert any("b's face (the one who moves) is not readable" in x for x in w)
    # the others need only be in view: a's back to the camera is no fault of a move shot
    back = {"b": subj["b"], "a": S.person([-2.3, 0.8], facing=[0, 1])}
    assert not _why(wide, "full", None, back, opts={"points": pts, "mover": "b"})


def test_the_end_place_is_not_small_beside_the_people():
    subj = {"b": S.person([-1.0, 0.0]), "a": S.person([-2.0, 0.6], facing=[0, 1])}
    c = S.cam([0.0, -2.2, 1.5], [0.0, 0.5, 1.0], 20)
    far_end = {"center": [0.5, 2.3, 0.5], "top": 0.45}                  # a low thing at the far wall
    w = _why(c, "full", None, subj, opts={"mover": "b", "end_place": far_end})
    assert any("the end place is small in the frame" in x for x in w)
    near_end = {"center": [1.2, 0.0, 0.5], "top": 1.1}
    assert not any("end place is small" in x for x in _why(c, "full", None, subj, opts={"mover": "b", "end_place": near_end}))


def test_action_joints_must_be_in_the_frame():
    subj = {"a": S.person([0, 0]), "b": S.person([0.7, 0.0], seated=True)}
    c = S.cam([0.3, -2.6, 1.3], [0.3, 0.0, 1.1], 35)
    opts = {"joints": [["a", "hand_r"], ["b", "lowerarm_l"]]}
    assert not any("(the action)" in x for x in _why(c, "medium", None, subj, opts=opts))
    high = S.cam([0.3, -1.2, 1.6], [0.3, 0.0, 1.6], 85)                # heads only: the hands are under the frame
    assert any("(the action) is not in the frame" in x for x in _why(high, "medium", None, subj, opts=opts))


# ------------------------------------------------------------------ small geometry the rules stand on
def test_in_picture_needs_the_head_inside_and_some_own_pixels():
    assert SP.in_picture({"head_any": True, "any_in": True})
    assert not SP.in_picture({"head_any": False, "any_in": True})            # a sliver of body, the head off the edge
    assert SP.in_picture({"head_proj": True, "head_any": False, "mask": {"area": 0.004}})     # behind someone
    assert not SP.in_picture({"head_proj": True, "mask": {"area": 0.0}})     # wholly hidden: not in the picture


def test_listener_of_and_side_of_the_line():
    assert SP.listener_of({"single": True, "listener": "b"}, ["a", "b"], "a") is None
    assert SP.listener_of({"ots": "b"}, ["a", "b"], "a") == "b"
    assert SP.listener_of({}, ["a", "b"], "a") == "b" and SP.listener_of({}, ["a", "b"], None) is None
    line = [[0, 0], [2, 0]]
    assert SP.side(line, [1, 1]) == -SP.side(line, [1, -1]) != 0 and SP.side(line, [1, 0]) == 0


def test_across_the_line_fails_for_a_medium_and_not_for_a_wide():
    subj = {"a": S.person([0, 0], facing=[0, 1]), "b": S.person([2, 0], facing=[-1, 0])}
    line = [[0, 0], [2, 0]]
    c = S.front_cam(subj["a"], 3.0)                                  # on the +y side
    est = -SP.side(line, c["position"])
    assert "across the line" in SP.check_shot({"camera": c}, "a", subj, RES, "medium", line, est)["why"]
    assert "across the line" not in SP.check_shot({"camera": c}, "a", subj, RES, "wide", line, est)["why"]


def test_subject_shown_with_others_may_be_in_profile():
    side = {"b": S.person([0, 0], facing=[1, 0])}                     # turned to someone beside her: a profile
    c = S.cam([0.0, -3.0, 1.5], [0.0, 0, 1.45], 35)
    assert any("turned from the camera" in x for x in SP.subject_face_why({"camera": c}, "b", side, RES))
    assert SP.subject_face_why({"camera": c}, "b", side, RES, shared=True) == []
    away = {"b": S.person([0, 0], facing=[0, 1])}                     # the back of her head: never
    assert any("turned from the camera" in x for x in SP.subject_face_why({"camera": c}, "b", away, RES, shared=True))
