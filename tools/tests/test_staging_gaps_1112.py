"""Further cases: over-the-shoulder framing, the set's lit list and place words, and three places where the code
does not yet do what was decided (marked as expected failures, each with the decision it falls short of)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import speakers as SP  # noqa: E402
import accept as AC  # noqa: E402
import setfacts  # noqa: E402

RES = S.RES


# ------------------------------------------------------------------ over the shoulder, two-shots
def test_over_the_shoulder_has_the_listener_at_the_side_and_the_speaker_toward_the_middle():
    subj = {"a": S.person([0, 1.0], facing=[0, -1]), "b": S.person([0.0, -1.0], facing=[0, 1])}
    line = [[0, 1.0], [0.0, -1.0]]
    c = SP.ots_camera(subj["a"]["measured"], subj["b"]["measured"], None, line)
    why, facts = SP.framing_why(c, "medium", "a", "b", True, None, subj, RES)
    assert not [w for w in why if "listener" in w or "speaker's face is at the frame's side" in w], why
    le = SP.framing.project(c, [0.0, -1.0, subj["b"]["measured"]["eye_z"]], RES)
    assert le is None or not 0.25 < le[0] < 0.75                    # the listener's head is not in the middle
    # the camera stands behind the listener, on the speaker's far side of him
    assert c["position"][1] < -1.0
    # straight behind the listener's head: his head is in the middle of the frame, over the speaker
    behind = S.cam([0.0, -2.2, 1.66], [0.0, 1.0, 1.5], 40)
    why, _ = SP.framing_why(behind, "medium", "a", "b", True, None, subj, RES)
    assert any("the listener's head is in the middle of the frame" in w for w in why)


def test_a_two_shot_favours_the_speaker_over_the_listener():
    subj = {"a": S.person([0, 0.6]), "b": S.person([0.7, -1.2], facing=[0, 1])}      # the listener nearer the lens
    c = S.cam([0.3, -3.0, 1.55], [0.3, 0.0, 1.5], 35)
    why, _ = SP.framing_why(c, "medium", "a", "b", False, None, subj, RES)
    assert any("the listener's head" in w and "bigger than the speaker's" in w for w in why)


# ------------------------------------------------------------------ the set's own facts
def _set(tmp_path):
    lib = S.make_library(tmp_path, [
        {"name": "fireplace", "position": [0.0, 2.3], "size": [1.4, 0.4, 1.2]},
        {"name": "armchair", "position": [1.6, 1.5], "size": [0.9, 0.9, 1.0], "look": "buttoned oxblood leather"},
        {"name": "armchair_2", "position": [-1.6, 1.5], "size": [0.9, 0.9, 1.1], "look": "dark red patterned damask"},
        {"name": "side_table", "position": [-2.2, -1.0], "size": [0.5, 0.5, 0.7]}],
        props=[{"name": "candle", "position": [-2.2, -1.0, 0.7], "size": [0.08, 0.08, 0.3]}],
        lights=[{"type": "point", "position": [0.0, 2.3, 0.4]}, {"type": "point", "position": [-2.2, -1.0, 0.95]}])
    return {t["name"]: t for t in setfacts.things(lib / "set" / "room_x")}


def test_a_fire_is_lit_and_a_flame_sized_source_is_only_named(tmp_path):
    t = _set(tmp_path)
    assert t["fireplace"]["lit"] and t["fireplace"]["fire"] and "with a fire burning" in t["fireplace"]["words"]
    assert t["candle"]["lit"] is False and t["candle"].get("small_light") and "lit" not in t["candle"]["words"]


def test_each_large_piece_is_named_with_its_look_and_its_place_by_the_rooms_landmark(tmp_path):
    t = _set(tmp_path)
    assert t["armchair"]["place"] == "to the right of the fireplace"
    assert t["armchair_2"]["place"] == "to the left of the fireplace"
    assert "buttoned oxblood leather" in t["armchair"]["words"] and "damask" in t["armchair_2"]["words"]
    assert t["armchair"]["large"] and not t["side_table"].get("large")


# ------------------------------------------------------------------ where the code falls short of what was decided
def _redraw_run(tmp_path, cam_fields):
    sd = tmp_path / "scene1"
    (sd / "guides").mkdir(parents=True)
    people = {"a": {"joints2d": {}}, "b": {"joints2d": {}}}
    (sd / "truth.json").write_text(json.dumps({"shots": {"s1": {"people": people, "camera": {}, "things": []}}}))
    (sd / "result.json").write_text(json.dumps({"subjects": {}, "shots": []}))
    (sd / "spec_round0.json").write_text(json.dumps({"characters": {"a": {}, "b": {}},
                                                     "cameras": [dict({"name": "s1"}, **cam_fields)]}))
    out = tmp_path / "run"
    out.mkdir()
    (out / "meta.json").write_text(json.dumps({"k1": {"scene": "scene1", "cam": "s1", "still": "x.png",
                                                      "people": [{"key": "a"}, {"key": "b"}]}}))
    run = {"out": str(out), "scenes": [str(sd)], "cast": {}}
    AC.cmd_checks(run, tmp_path / "check.json", None, "none")
    items = json.loads((tmp_path / "check.json").read_text())["items"]
    return {p["key"]: p["named"] for p in items[0]["people"]}


def test_the_redraws_face_check_is_asked_for_the_speaker_only(tmp_path):
    assert _redraw_run(tmp_path, {"speaker": "a"}) == {"a": True, "b": False}


@pytest.mark.xfail(strict=True, reason="decided 11-12: the person a shot is about is treated like a speaker; the staging "
                                       "checks her face (subject_face_why) but the redraw's face check still asks for "
                                       "the speaker only, so a redraw that turns her away is not caught")
def test_the_redraws_face_check_also_covers_whom_a_shot_without_a_speaker_is_about(tmp_path):
    assert _redraw_run(tmp_path, {"speaker": None, "line_subject": "b"}) == {"a": False, "b": True}


@pytest.mark.xfail(strict=True, reason="clear_walks hands every move camera the same path_blocked_by list: a later "
                                       "walk's blocker appears in an earlier walk's shot facts")
def test_each_walks_shot_facts_list_only_its_own_blockers(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [{"name": "armchair", "position": [1.6, 1.2], "seat": True},
                                    {"name": "sofa", "position": [-1.6, -1.6], "size": [1.6, 0.8, 0.9], "seat": True}])
    monkeypatch.setattr(SP.C, "LIBRARY", lib)
    ch = lambda pos: {"id": "f", "position": list(pos), "facing": [0, -1], "body_pose": {"pose": "stand_relaxed"}}
    spec = {"set": "room_x", "characters": {"b": ch([-1.5, 0.0]), "a": ch([2.0, -1.0]), "c": ch([0.2, -1.3])},
            "cameras": [{"name": "s1", "kind": "full", "mover": "b", "move_to": "armchair"},      # clear path
                        {"name": "s2", "kind": "full", "mover": "a", "move_to": "sofa"}]}         # c stands on it
    got = SP.clear_walks(spec, {"armchair", "sofa"})
    assert got["shot_facts"]["s2"]["path_blocked_by"] == ["c"]
    assert got["shot_facts"]["s1"]["path_blocked_by"] == []
