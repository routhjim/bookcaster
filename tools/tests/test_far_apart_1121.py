"""People too far apart for one wide, and a shot at a thing far taller than a person (scenes/speakers.py)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
import speakers as S  # noqa: E402


def _m(x, y, fwd=(0.0, -1.0), creature=False, z=1.6):
    m = {"eye_xy": [x, y], "eye_z": z, "head_fwd_xy": list(fwd)}
    if creature:
        m["creature"] = True
    return m


def _open():
    S.SOLIDS[:] = []
    S.OPEN_GROUND[:] = [0.0, 0.0, 40.0]


def test_people_a_room_apart_are_no_specks():
    assert not S.speck_wide([_m(0, 0), _m(3, 2)])
    assert not S.speck_wide([_m(0, 0)])
    assert S.speck_wide([_m(-12, 0), _m(12, 12, creature=True, z=3.0), _m(0, 4)])


def test_far_creature_is_seen_past_the_furthest_person():
    _open()
    try:
        beast, near, far = _m(12, 12, creature=True, z=3.0), _m(0, 4), _m(-12, 0)
        q = S.fallback_camera([beast, near, far], [50.0, 50.0, 14.0])
        assert q is not None
        p = q["position"]
        assert q["target"][:2] == [12, 12]                               # aimed at the creature
        assert 3.0 <= math.dist(p[:2], [-12, 0]) <= 10.0                 # at the furthest person's scale ...
        assert math.dist(p[:2], [12, 12]) > math.dist([-12, 0], [12, 12])   # ... from behind them
        assert S.seg_point_dist([-12, 0, 1.6], p, [12, 12, 3.0]) > 0.5      # and not over their head
    finally:
        S.OPEN_GROUND[:] = []


def test_far_person_first_is_seen_with_the_creature_beyond():
    _open()
    try:
        q = S.fallback_camera([_m(0, 4, (-1.0, 0.0)), _m(14, 14, creature=True, z=3.0)], [50.0, 50.0, 14.0])
        assert q is not None
        p = q["position"][:2]
        assert math.dist(p, [0, 4]) < 10.0 and math.dist(p, [14, 14]) > math.dist([0, 4], [14, 14])
    finally:
        S.OPEN_GROUND[:] = []


def test_far_person_first_is_taken_alone_when_the_creature_is_hidden():
    _open()
    S.SOLIDS[:] = [[7.0, 9.0, 2.0, 2.0, 8.0, 0.0, -1.0]]
    try:
        q = S.fallback_camera([_m(0, 4, (-1.0, 0.0)), _m(14, 14, creature=True, z=3.0)], [50.0, 50.0, 14.0])
        assert q is not None and math.dist(q["position"][:2], [0, 4]) < 6.0
    finally:
        S.OPEN_GROUND[:] = []
        S.SOLIDS[:] = []


def test_tall_thing_is_held_by_its_part_beside_the_subject():
    at = {"name": "x", "center": [0.0, 10.0, 4.5], "half": [6.0, 5.0, 4.5], "top": 9.0}
    assert S.at_point(at, [1.0, 4.0]) == [1.0, 5.0, 1.9]
    low = {"name": "y", "center": [0.0, 1.0, 0.4], "half": [0.8, 0.4, 0.4], "top": 0.8}
    assert S.at_point(low, [0.0, 0.0]) == [0.0, 1.0, 0.8]
    assert S.at_point(dict(at, half=None), [1.0, 4.0]) == [0.0, 10.0, 9.0]


def test_a_key_of_nobody_is_staged_again(tmp_path):
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
    import story_keys as SK
    st = SK.Story.__new__(SK.Story)
    st.work, st.keys = tmp_path / "work", tmp_path / "keys"
    st.places = {"P": {"slug": "p", "set": "s"}}
    st.plan = {"A": [{"people": [{"name": "n"}]}], "B": [{"people": []}], "C": [{"people": [{"name": "n"}]}]}
    st.keys.mkdir()
    (st.work / "p").mkdir(parents=True)
    (st.work / "p" / "keys.json").write_text(json.dumps([{"shot": s, "key": True, "why": ""} for s in "ABC"]))
    (st.keys / "A.shot_facts.json").write_text(json.dumps({"visible": [], "count": 0}))
    (st.keys / "B.shot_facts.json").write_text(json.dumps({"visible": [], "count": 0}))
    (st.keys / "C.shot_facts.json").write_text(json.dumps({"visible": [{"name": "n"}], "count": 1}))
    assert st.keyless("P") == ["A"]
