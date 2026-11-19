"""A creature's camera stands back by its own head's size, near its head's height (scenes/speakers.py), and a still
that shows nothing is a failed staging, re-framed farther back.

The search's human camera heights and distances once put a camera 0.2 m in front of a 4 m beast's eye and 1.3 m
under it, looking up into its throat: the still was black, the creature's mask the whole frame, and every check
passed (the measured numbers below are that staging's). Now: such a camera is refused before it is rendered
(creature_view_why), the search offers heights at the head, a still nearly black or filled by one body fails
(still_why) and its re-framing stands at least BACK_FACTOR farther back, and the clear wide fallback keeps the
same distance rules. A person's camera rules are unchanged.
"""
import math
import struct
import sys
import zlib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
sys.path.insert(0, str(HERE.parents[0] / "assets"))
import framing  # noqa: E402
import speakers as SP  # noqa: E402

RES = (1152, 640)
# a creature as the engine measured it: eye 2.92 m up, head 0.855 m (top to chin), facing -y; body 11 x 11 x 4.25 m
BEAST = {"eye_xy": [-0.5209, -2.4236], "eye_z": 2.9224, "head_top_z": 3.2644, "chin_z": 2.4094,
         "head_fwd_xy": [-0.0839, -0.9965], "feet_z": 0.0, "waist_z": 1.4612, "chest_xy": [-0.4225, -1.1679],
         "pelvis_xy": [0.12, 1.83], "facing": [-0.0654, -0.9979], "height_m": 4.25, "creature": True, "joints": {}}
PERSON = {"eye_xy": [0.0, 0.0], "eye_z": 1.62, "head_top_z": 1.75, "chin_z": 1.5, "head_fwd_xy": [0.0, -1.0],
          "feet_z": 0.0, "waist_z": 1.0, "chest_xy": [0.0, 0.0], "pelvis_xy": [0.0, 0.0], "facing": [0.0, -1.0],
          "joints": {}}
BLACK_MEDIUM = {"position": [-0.6, -2.6, 1.6], "target": [-0.521, -2.424, 2.802], "focal_mm": 50}   # as staged
BLACK_CLOSE = {"position": [-0.4, -2.6, 1.6], "target": [-0.521, -2.424, 2.802], "focal_mm": 100}


@pytest.fixture
def open_stage():
    """An open stage 8.5 m round, the beast's body box and its head sphere, a rock in front of its head."""
    saved = (list(SP.OPEN_GROUND), list(SP.SOLIDS), list(SP.HEADS))
    SP.OPEN_GROUND[:] = [-0.05, 0.23, 8.5]
    SP.SOLIDS[:] = [[0.2, 3.05, 2.5, 2.5, 5.0, 0.0, 1.0], [-0.7, -4.45, 1.5, 1.5, 3.0, 0.0, 1.0],
                    [0.12, 1.83, 2.08, 2.55, 3.4, -0.0654, -0.9979]]
    SP.set_heads({"a": {"measured": BEAST}})
    yield [17.5, 17.5, 12.0]
    SP.OPEN_GROUND[:], SP.SOLIDS[:], SP.HEADS[:] = saved


def eye(m):
    return [m["eye_xy"][0], m["eye_xy"][1], m["eye_z"]]


def pitch_deg(p, m):
    e = eye(m)
    return math.degrees(math.atan2(abs(e[2] - p[2]), math.hypot(p[0] - e[0], p[1] - e[1])))


def png(path: Path, w: int, h: int, rgb) -> Path:
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


# ------------------------------------------------------------------ the rule before a render
def test_the_staged_black_cameras_under_a_creatures_head_are_refused():
    for cam, kind in ((BLACK_MEDIUM, "medium"), (BLACK_CLOSE, "close")):
        why = SP.creature_view_why(cam, "a", BEAST, kind, RES)
        assert any("from under or over its head" in w for w in why), why
        assert any("of the frame's height" in w for w in why), why
        # and in the framing rules every search and every check runs, for the shot's subject
        fw, _ = SP.framing_why(cam, kind, None, None, False, ["a"], {"a": {"measured": BEAST}}, RES,
                               opts={"line_subject": "a", "subject": "a"})
        assert any(SP.TOO_NEAR in w for w in fw), fw


def test_the_scene_tools_own_creature_shots_pass_and_a_persons_rules_are_unchanged():
    for kind, f in (("close", 85), ("close", 50), ("medium", 50), ("medium", 35)):
        cs = framing.shot(BEAST, kind, f, RES)
        assert SP.creature_view_why(cs, "a", BEAST, kind, RES) == [], (kind, cs)
    near = {"position": [0.0, -0.5, 1.6], "target": [0.0, 0.0, 1.6], "focal_mm": 85}     # a person's tight close
    assert SP.creature_view_why(near, "p", PERSON, "close", RES) == []


def test_the_rule_scales_with_the_head_not_with_one_creatures_numbers():
    small = dict(BEAST, eye_xy=[0.0, 0.0], eye_z=0.5, head_top_z=0.6, chin_z=0.42)        # a dog-sized head
    big = dict(BEAST, eye_xy=[0.0, 0.0], eye_z=9.0, head_top_z=10.2, chin_z=7.8)           # a head 2.4 m high
    for m in (small, big):
        hh = m["head_top_z"] - m["chin_z"]
        for kind, f in (("close", 85), ("medium", 50)):
            d_ok = 1.05 * hh * f / framing.sensor_h(RES) / SP.CREATURE_HEAD_MAX[kind]      # just far enough
            ok = {"position": [0.0, -d_ok, m["eye_z"]], "target": eye(m), "focal_mm": f}
            near = {"position": [0.0, -0.8 * d_ok * SP.CREATURE_HEAD_MAX[kind], m["eye_z"]], "target": eye(m),
                    "focal_mm": f}
            assert SP.creature_view_why(ok, "a", m, kind, RES) == []
            assert SP.creature_view_why(near, "a", m, kind, RES)
        under = {"position": [0.0, -0.4 * m["eye_z"], 0.1 * m["eye_z"]], "target": eye(m), "focal_mm": 24}
        assert any("under or over" in w for w in SP.creature_view_why(under, "a", m, "wide", RES))


# ------------------------------------------------------------------ the search
def test_the_search_stands_back_at_the_heads_height_clear_of_the_head_body_and_rock(open_stage):
    subjects = {"a": {"measured": BEAST}}
    hh = BEAST["head_top_z"] - BEAST["chin_z"]
    for name, kind, cam0 in (("s03", "medium", {"position": [-0.7679, -6.4385, 2.4576], "focal_mm": 50}),
                             ("s04", "close", {"position": [3.9913, -8.5775, 2.6231], "focal_mm": 85})):
        c = {"name": name, "kind": kind, "subject": "a", "line_subject": "a", "show": ["a"], "listed": ["a"]}
        opts = {"subject": "a", "line_subject": "a", "listed": ["a"]}
        tc = SP.two_shot_candidates(c, {"camera": cam0}, "a", None, subjects, RES, open_stage, None, None,
                                    [cam0["position"]], opts)
        assert tc, name
        for q, _ in tc:                      # no candidate under the head, nor inside anything
            p = q["position"]
            assert math.dist(p, BLACK_MEDIUM["position"]) > 0.5
            assert pitch_deg(p, BEAST) <= SP.CREATURE_MAX_PITCH_DEG + 1e-6
            assert SP.in_room(p, open_stage) and not SP.inside_solid(p)
            assert not any(SP.solid_between(p, eye(BEAST)) for _ in [0])
            fh = math.dist(p, eye(BEAST)) * framing.sensor_h(RES) / q["focal_mm"]
            assert hh / fh <= SP.CREATURE_HEAD_MAX[kind] + 1e-6
        best = tc[0][0]["position"]
        assert math.dist(best, eye(BEAST)) > 3.0             # metres off, not 1.3 m
        assert abs(best[2] - BEAST["eye_z"]) <= hh + 0.01    # at the head's height or just over it


def test_a_person_is_still_searched_from_a_persons_eye_heights(open_stage):
    SP.SOLIDS[:], SP.HEADS[:] = [], []
    subjects = {"p": {"measured": dict(PERSON, eye_xy=[0.0, 2.0], pelvis_xy=[0.0, 2.0], chest_xy=[0.0, 2.0])}}
    c = {"name": "s", "kind": "medium", "subject": "p", "line_subject": "p", "show": ["p"], "listed": ["p"]}
    tc = SP.two_shot_candidates(c, {"camera": {"position": [0.0, -1.0, 1.6], "focal_mm": 50}}, "p", None, subjects,
                                RES, open_stage, None, None, [], {"subject": "p", "line_subject": "p"})
    assert tc and {round(q["position"][2], 2) for q, _ in tc} <= {1.45, 1.6}


# ------------------------------------------------------------------ after a render
def test_a_still_nearly_black_or_filled_by_one_body_fails(tmp_path):
    dark = png(tmp_path / "dark.png", 64, 36, (3, 2, 2))
    lit = png(tmp_path / "lit.png", 64, 36, (90, 80, 70))
    assert SP._png_mean(dark) < SP.STILL_DARK <= SP._png_mean(lit)
    assert abs(SP._png_mean(lit) - 80.0) < 0.6
    filled = {"frame_facts": {"a": {"mask": {"area": 0.9979}}}}
    some = {"frame_facts": {"a": {"mask": {"area": 0.42}}}}
    assert any(SP.TOO_NEAR in w for w in SP.still_why({"path": str(lit)}, filled))
    assert any("nearly black" in w for w in SP.still_why({"path": str(dark)}, some))
    assert SP.still_why({"path": str(lit)}, some) == []


def test_a_still_that_showed_nothing_is_reframed_farther_back():
    subjects = {"a": {"measured": BEAST}}
    shot = {"camera": BLACK_MEDIUM, "speaker_check": {"frame_facts": {"a": {"mask": {"area": 0.998}}}}}
    c = {"name": "s03", "kind": "medium", "line_subject": "a", "show": ["a"]}
    bk = SP.back_of(c, shot, subjects)
    d0 = math.dist(BLACK_MEDIUM["position"], eye(BEAST))
    assert bk["key"] == "a" and bk["min_m"] == pytest.approx(SP.BACK_FACTOR * d0, abs=1e-3)
    c["_back"] = bk
    assert SP.cam_opts({"characters": {}}, c)["back"] == bk
    # a camera nearer than that is refused by the framing rules (any subject: a person too)
    p_shot = {"camera": {"position": [0.0, -0.6, 1.6]}, "speaker_check": {"frame_facts": {}}}
    pk = SP.back_of({"line_subject": "p"}, p_shot, {"p": {"measured": PERSON}})
    near = {"position": [0.0, -0.7, 1.6], "target": eye(PERSON), "focal_mm": 50}
    far = {"position": [0.0, -1.0, 1.6], "target": eye(PERSON), "focal_mm": 50}
    why = lambda cam: SP.framing_why(cam, "medium", None, None, False, ["p"], {"p": {"measured": PERSON}}, RES,  # noqa: E731
                                     opts={"line_subject": "p", "back": pk})[0]
    assert any(SP.TOO_NEAR in w for w in why(near))
    assert not any(SP.TOO_NEAR in w for w in why(far))


def test_the_clear_wide_fallback_keeps_a_creatures_distance(open_stage):
    fb = SP.fallback_camera([BEAST], open_stage)
    assert fb and SP.creature_view_why(fb, "a", BEAST, "wide", RES) == []
    assert SP.in_room(fb["position"], open_stage)


def test_a_kept_staging_that_showed_nothing_is_staged_again():
    src = (HERE.parents[0] / "assets" / "story_keys.py").read_text()
    i = src.index("UNREADABLE = (")
    block = src[i:src.index(")", i)]
    assert SP.TOO_NEAR in block and "the still is nearly black" in block
