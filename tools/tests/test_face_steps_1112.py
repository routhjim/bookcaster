"""The face steps' pure parts: the skin age table and pick, the blend weights and mix, keep-only-if-it-wins, and
that the one command's options are off unless asked. No engine, no model: the editor-side calls these steps make
(set_face_model_coefficients, commit_skin_settings with a face texture index, the make's read-backs) were written
from the installed plugin's headers and examples and have NOT been run in the editor.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "assets" / "characters"))
sys.path.insert(0, str(ROOT / "tools" / "assets"))

import face_blend as FB  # noqa: E402
import face_skin as FS  # noqa: E402

TABLE = {"n": 8, "attrs": {
    "Face Wrinkles": {"names": ["Low", "Medium", "High"], "values": [0, 0, 1, 1, 2, 2, 0, 2]},
    "Face Stubble": {"names": ["None", "Low", "Medium", "High"], "values": [0, 1, 0, 3, 0, 2, 0, 0]},
    "Face Marks": {"names": ["Low", "Medium", "High"], "values": [0, 0, 2, 0, 1, 0, 0, 0]}}}


def test_allowed_indices():
    assert FS.allowed(TABLE) == [0, 2, 4, 6, 7]                # no stubble shading without a stubble beard
    assert FS.allowed(TABLE, stubble=True) == list(range(8))


def test_pick_nearest_age():
    ages = {0: 25, 2: 41, 4: 58, 6: 27, 7: 60}
    ok = FS.allowed(TABLE)
    assert FS.pick(40, ages, ok, TABLE) == 2
    assert FS.pick(70, ages, ok, TABLE) == 7
    assert FS.pick(26, ages, ok, TABLE) == 0                   # 25 and 27 tie at one year: marks equal, lower index
    # a tie in age goes to fewer marks before the lower index
    assert FS.pick(59, {4: 58, 7: 60, 0: 20}, [0, 4, 7], TABLE) == 7     # index 4 has medium marks, 7 low
    # an index that was not measured is never picked
    assert FS.pick(58, {0: 25, 2: 41, 7: 60}, ok, TABLE) == 7


def test_pick_stops_loudly():
    try:
        FS.pick(40, {}, [0, 2], TABLE)
    except ValueError as e:
        assert "no measured index" in str(e)
    else:
        raise AssertionError("an empty table must stop")
    try:
        FS.pick(40, {0: 30, 2: 33, 4: 34}, [0, 2, 4], TABLE)   # the index does not move the age read
    except ValueError as e:
        assert "HAD NO EFFECT" in str(e)
    else:
        raise AssertionError("ages that do not vary must stop")


def test_table_merge_and_preset_fallback():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "ages.json"
        FS.merge_table(p, "one", {"0": 30, "1": 50})
        FS.merge_table(p, "one", {"2": 70})                     # a later run adds to the preset's entries
        FS.merge_table(p, "two", {"0": 34, "1": 58})
        tab = json.loads(p.read_text())
        assert tab["question"] == FS.AGE_QUESTION
        assert FS.preset_ages(tab, "one") == {0: 30.0, 1: 50.0, 2: 70.0}
        # a preset that was not measured gets the median over the measured ones
        assert FS.preset_ages(tab, "three") == {0: 32.0, 1: 54.0, 2: 70.0}
        assert FS.spread(FS.preset_ages(tab, "one")) == 40.0


def test_same_age_question_as_the_face_pick():
    src = (ROOT / "tools" / "assets" / "characters" / "face_pick.py").read_text()
    assert FS.AGE_QUESTION in src                               # one scale for the drawing, the faces and the skins


def test_shipped_attribute_table_if_installed():
    import config as C
    try:
        t = FS.attributes(C.UE_ENGINE)
    except FileNotFoundError:
        return                                                   # no engine on this machine: nothing to check
    assert t["n"] > 10 and FS._attr(t, "wrinkle") and FS._attr(t, "stubble")
    assert 0 < len(FS.allowed(t)) < t["n"]


def test_sample_indices_spread_over_wrinkle_classes():
    import skin_age_calibrate as SC
    got = SC.sample_indices(TABLE, 3)
    wr = TABLE["attrs"]["Face Wrinkles"]["values"]
    assert len(got) == 3 and {wr[i] for i in got} == {0, 1, 2}  # one of each class among the stubble-free ones
    assert SC.sample_indices(TABLE, 99) == FS.allowed(TABLE)
    assert SC.batches(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    job = SC.make_job("P", [4, 7], "/out", "P")
    assert [m["skin"]["face_texture_index"] for m in job["make"]] == [4, 7]
    assert [s["name"] for s in job["shots"]] == ["idx_4", "idx_7"]
    assert all(m["clear_slots"] == ["Beard", "Mustache"] for m in job["make"])


def test_weights_from_wins():
    w = FB.weights_from_wins({"a": 12, "b": 10, "c": 7, "d": 7, "e": 3, "f": 0})
    assert list(w) == ["a", "b", "c"] and abs(sum(w.values()) - 1) < 1e-9
    assert abs(w["a"] - 12 / 29) < 1e-9 and abs(w["c"] - 7 / 29) < 1e-9   # c before d: the earlier candidate
    assert FB.weights_from_wins({"a": 8, "b": 0, "c": 0}) == {}            # one winner: nothing to blend
    assert FB.weights_from_wins({}) == {}
    # the pick is always in the blend, even when the wins put it outside the top
    w = FB.weights_from_wins({"a": 9, "b": 8, "c": 7, "d": 6}, pick="d", top=2)
    assert set(w) == {"a", "d"} and abs(sum(w.values()) - 1) < 1e-9
    # a power above 1 leans further toward the leader
    assert FB.weights_from_wins({"a": 4, "b": 2}, power=2)["a"] == 0.8


def test_mix_and_deltas():
    co = {"a": [1.0, 0.0, 2.0], "b": [0.0, 1.0, 4.0]}
    assert FB.mix(co, {"a": 0.5, "b": 0.5}) == [0.5, 0.5, 3.0]
    assert FB.mix(co, {"a": 0.25, "b": 0.75}, scale=2.0) == [0.5, 1.5, 7.0]
    assert FB.max_delta([0, 0, 0], [0, 0.5, -1]) == 1
    for bad, why in (({"a": [1.0], "b": [1.0, 2.0]}, "differ in length"),):
        try:
            FB.mix(bad, {"a": 0.5, "b": 0.5})
        except ValueError as e:
            assert why in str(e)
        else:
            raise AssertionError("vectors of different lengths must stop")
    try:
        FB.mix(co, {"a": 0.5, "b": 0.6})
    except ValueError as e:
        assert "sum to" in str(e)
    else:
        raise AssertionError("weights that do not sum to 1 must stop")


def test_keep_only_if_it_wins_both_orders():
    assert FB.keep({"blended": 2, "unblended": 0}, "blended")
    assert not FB.keep({"blended": 1, "unblended": 1}, "blended")          # one each: position bias, not kept
    assert not FB.keep({"blended": 0, "unblended": 2}, "blended")
    assert not FB.keep({}, "blended") and not FB.keep(None, "blended")
    assert FB.choose_face(None, None) == "pick"
    assert FB.choose_face({"keep": True}, {"keep": False}) == "blend"
    assert FB.choose_face({"keep": True}, {"keep": True}) == "conform"     # the conform replaces the coefficients
    assert FB.choose_face({"keep": False}, None) == "pick"


def test_options_are_off_by_default():
    src = (ROOT / "tools" / "assets" / "img3d" / "cast_char.py").read_text()
    for flag in ("--face-blend", "--skin-age", "--conform"):
        m = re.search(r'add_argument\("' + re.escape(flag) + r'",\s*action="store_true"', src)
        assert m, f"{flag} must be a switch that is off unless given"
    # the changed face reaches the built character only through the kept result of its comparison
    assert 'FB.choose_face(blend_, conf_)' in src and 'if a.skin_age:' in src
    # the make's read-backs stop the build when a setting did not land
    mk = (ROOT / "tools" / "assets" / "characters" / "char_make.py").read_text()
    for stop in ("FACE BLEND HAD NO EFFECT", "SKIN INDEX NOT SET"):
        assert stop in mk


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"ok ({n} tests)")
