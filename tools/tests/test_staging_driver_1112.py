"""The staging loop itself (speakers.py main) run against a stand-in for the engine (stdlib, no engine, no GPU).

The stand-in (staging_synth.FakeEngine) measures every person where the round's spec puts them and writes depth
passes with people as boxes, so the loop's own decisions can be read from what it writes: a failing talk shot is
re-framed; a person on a vague place moves when they are in the way and the move is kept only when it helps; an
exact place and a seated person never move; every shot ends with a key, the least-bad try labelled with its misses.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import speakers as SP  # noqa: E402

ROOM = (4.4, 4.4, 3.0)
STAND = {"pose": "stand_relaxed", "params": {}}


def _char(pos, facing=(0, -1), **kw):
    return dict({"id": "fig", "position": list(pos), "facing": list(facing), "body_pose": json.loads(json.dumps(STAND))}, **kw)


def _cam(name, speaker, position, target, kind="medium", focal=40, **kw):
    return dict({"name": name, "kind": kind, "speaker": speaker, "subject": speaker, "position": list(position),
                 "target": list(target), "focal_mm": focal}, **kw)


def stage(tmp_path, monkeypatch, spec, depth=False, rounds=14):
    """Run the staging loop on a spec with the stand-in engine; -> (report, result, staged, engine)."""
    lib = S.make_library(tmp_path, [{"name": "desk", "position": [1.7, 1.7], "size": [0.6, 0.6, 0.8]}], size=ROOM)
    monkeypatch.setattr(SP.C, "LIBRARY", lib)
    out = tmp_path / "out"
    monkeypatch.setenv("SCENES_OUT", str(out))
    monkeypatch.delenv("SCENES_CHAR_TOOLS", raising=False)
    eng = S.FakeEngine(out, depth=depth)
    monkeypatch.setattr(SP, "run_scene", eng)
    spec = dict({"id": "sc", "set": "room_x"}, **spec)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["speakers.py", str(p), "--rounds", str(rounds), "--no-split"])
    assert SP.main() == 0
    d = out / "sc"
    return (json.loads((d / "speakers.json").read_text()), json.loads((d / "result.json").read_text()),
            json.loads((d / "staged.json").read_text()), eng)


def _shot(result, name):
    return next(s for s in result["shots"] if s["name"] == name)


def test_a_failing_talk_shot_is_re_framed_onto_the_speakers_face(tmp_path, monkeypatch):
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([0, 0.5]), "b": _char([1.1, 0.75], facing=[-1, -0.2])},
            "cameras": [_cam("s1", "a", [-1.9, -0.1, 1.5], [0, 0.5, 1.5]),         # far round to his side
                        _cam("s2", "a", [0.4, -1.7, 1.55], [0.45, 0.6, 1.5], focal=35)]}
    report, result, staged, eng = stage(tmp_path, monkeypatch, spec)
    assert report["shots"]["s2"]["ok"] and len(report["shots"]["s2"]["history"]) == 1
    h = report["shots"]["s1"]["history"]
    assert not h[0]["ok"] and any("deg from the camera" in w for w in h[0]["why"])
    assert report["shots"]["s1"]["ok"] and len(h) >= 2
    sh = _shot(result, "s1")
    assert sh["speaker_check"]["ok"] and "framing_miss" not in sh
    m = result["subjects"]["a"]["measured"]
    assert SP.plan_angle(m["head_fwd_xy"], m["eye_xy"], sh["camera"]["position"]) <= SP.MAX_HEAD_DEG
    assert SP.in_room(sh["camera"]["position"], list(ROOM))
    assert staged["positions"] == {"a": [0, 0.5], "b": [1.1, 0.75]}              # nobody moved for it
    assert eng.calls[1]["cams"] == ["s1"]                                         # only the failed shot renders again


def test_a_passing_shot_is_left_alone(tmp_path, monkeypatch):
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([0, 0.5]), "b": _char([1.1, 0.75], facing=[-1, -0.2])},
            "cameras": [_cam("s1", "a", [0.4, -1.7, 1.55], [0.45, 0.6, 1.5], focal=35)]}
    report, result, _, eng = stage(tmp_path, monkeypatch, spec)
    assert report["shots"]["s1"]["ok"] and len(report["shots"]["s1"]["history"]) == 1 and len(eng.calls) == 1


def test_a_shot_that_cannot_pass_still_ends_with_a_key_labelled_with_its_misses(tmp_path, monkeypatch):
    wall_y = ROOM[1] / 2 - 0.5                               # his face a hand's breadth from the wall he faces
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([0, wall_y], facing=[0, 1]), "b": _char([1.5, 0.0], facing=[-1, 0])},
            "cameras": [_cam("s1", "a", [0, 0.0, 1.5], [0, wall_y, 1.5]),
                        _cam("s2", "b", [0.2, 0.0, 1.5], [1.5, 0.0, 1.5])]}
    spec["cameras"][1]["speaker"] = None                     # one group: the second shot has no speaker
    spec["cameras"][1]["show"] = ["b"]
    report, result, staged, eng = stage(tmp_path, monkeypatch, spec)
    assert {s["name"] for s in result["shots"]} == {"s1", "s2"}                   # no shot is dropped
    sh = _shot(result, "s1")
    assert report["shots"]["s1"]["ok"] is False
    assert sh["framing_miss"] and sh["least_bad"]["misses"] == sh["framing_miss"]
    assert "reject" not in sh
    assert len(report["shots"]["s1"]["history"]) <= SP.MAX_TRIES + 2
    assert staged["positions"]["a"] == [0, wall_y]                                # an exact place never moves
    # the least-bad try is the one with the fewest misses against the speaker's face, then in all
    hist = report["shots"]["s1"]["history"]
    best = min(hist, key=lambda h: (len([w for w in h["why"] if w not in (h.get("framing") or [])]), len(h["why"])))
    assert len(sh["least_bad"]["misses"]) == len(best["why"])


def _blocked_scene(c_extra=None, c_pos=(0.0, 0.1)):
    """a speaks toward -y; c stands a pace in front of him, in the way of every camera on his face."""
    return {"line": ["a", "b"],
            "characters": {"a": _char([0, 0.5]), "b": _char([1.6, 0.4], facing=[-1, 0]),
                           "c": _char(c_pos, facing=[0, -1], **(c_extra or {}))},
            "cameras": [_cam("s1", "a", [0.0, -1.9, 1.55], [0.0, 0.5, 1.5], focal=35)]}


def test_a_vague_place_person_in_the_way_moves_and_the_move_is_kept_when_it_helps(tmp_path, monkeypatch):
    marks = [[0.0, 0.1], [-1.5, 1.2], [1.2, 1.5]]
    report, result, staged, eng = stage(tmp_path, monkeypatch, _blocked_scene({"marks": marks}), depth=True)
    h = report["shots"]["s1"]["history"]
    assert not h[0]["ok"] and any("visible" in w for w in h[0]["why"])            # his face is behind c
    assert report["shots"]["s1"]["ok"], h[-1]["why"]
    tries = report["marks"]["tries"]
    assert tries and tries[0]["k"] == "c" and tries[-1]["kept"] is True
    assert staged["positions"]["c"] in ([-1.5, 1.2], [1.2, 1.5])                  # on one of the allowed marks
    assert staged["positions"]["a"] == [0, 0.5] and staged["positions"]["b"] == [1.6, 0.4]
    # the mark nearest where c stood is tried first
    first = min(marks[1:], key=lambda m: math.dist(m, marks[0]))
    assert tries[0]["to"] == first
    # the move came with a camera found for the new places, rendered together with it
    assert "new_camera" in tries[0] or report["shots"]["s1"]["ok"]
    assert any(c["cams"] is None for c in eng.calls[1:])                          # everyone re-rendered after a move


def test_an_exact_place_in_the_way_never_moves_and_the_shot_keeps_its_least_bad_key(tmp_path, monkeypatch):
    report, result, staged, eng = stage(tmp_path, monkeypatch, _blocked_scene(), depth=True)
    assert staged["positions"]["c"] == [0.0, 0.1]
    assert not report["marks"].get("tries")
    sh = _shot(result, "s1")
    if not report["shots"]["s1"]["ok"]:                      # no camera sees past him: the least-bad try is the key
        assert sh["framing_miss"] and sh["least_bad"]
    assert {s["name"] for s in result["shots"]} == {"s1"}


def test_a_seated_person_in_the_way_never_moves_even_with_marks(tmp_path, monkeypatch):
    spec = _blocked_scene({"marks": [[0.0, 0.1], [-1.5, 1.2]]})
    spec["characters"]["c"]["body_pose"] = {"pose": "seated", "params": {}}
    report, result, staged, eng = stage(tmp_path, monkeypatch, spec, depth=True)
    assert staged["positions"]["c"] == [0.0, 0.1] and not report["marks"].get("tries")


def test_a_mark_that_does_not_help_is_undone(tmp_path, monkeypatch):
    wall_y = ROOM[1] / 2 - 0.5
    # the speaker faces the wall; c stands aside on a vague place: moving c cannot show his face
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([0, wall_y], facing=[0, 1]), "b": _char([1.6, 0.0], facing=[-1, 0]),
                           "c": _char([0.0, wall_y - 0.6], facing=[0, 1], marks=[[0.0, wall_y - 0.6], [-1.6, 0.4]])},
            "cameras": [_cam("s1", "a", [0, -0.8, 1.5], [0, wall_y, 1.5])]}
    report, result, staged, eng = stage(tmp_path, monkeypatch, spec, depth=True)
    assert report["shots"]["s1"]["ok"] is False
    for t in report["marks"].get("tries") or []:
        assert t["kept"] in (False, "camera tried again")
    assert staged["positions"]["c"] == [0.0, wall_y - 0.6]                        # back where the plan put them
    assert _shot(result, "s1")["framing_miss"]


def test_the_tries_are_bounded(tmp_path, monkeypatch):
    wall_y = ROOM[1] / 2 - 0.5
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([0, wall_y], facing=[0, 1]), "b": _char([1.6, 0.0], facing=[-1, 0])},
            "cameras": [_cam("s1", "a", [0, -0.8, 1.5], [0, wall_y, 1.5])]}
    report, _, _, eng = stage(tmp_path, monkeypatch, spec)
    assert len(eng.calls) <= SP.MAX_TRIES + 3


def test_listeners_look_at_the_speaker_in_the_staged_spec(tmp_path, monkeypatch):
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([0, 0.5]), "b": _char([1.2, 0.2], facing=[0, -1])},
            "cameras": [_cam("s1", "a", [0.3, -1.9, 1.55], [0.3, 0.5, 1.5], focal=35)]}
    report, result, _, eng = stage(tmp_path, monkeypatch, spec)
    rnd = json.loads(Path(eng.calls[0]["spec"]).read_text())
    assert rnd["characters"]["b"]["_look"]["at"] == "a" and "_look" not in rnd["characters"]["a"]
    hb = result["subjects"]["b"]["measured"]["head_fwd_xy"]
    to_a = S.unit([0 - 1.2, 0.5 - 0.2])
    assert hb[0] * to_a[0] + hb[1] * to_a[1] > 0.5           # the head turned toward him, as far as a neck goes


def test_a_vague_place_keeps_the_shot_befores_mark_when_it_is_allowed(tmp_path, monkeypatch):
    prev = tmp_path / "prev_staged.json"
    prev.write_text(json.dumps({"positions": {"c": [-1.2, 1.2], "a": [1.0, 1.0]}, "camera": None}))
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([0, 0.5]), "b": _char([1.1, 0.75], facing=[-1, -0.2]),
                           "c": _char([-1.5, 0.2], marks=[[-1.5, 0.2], [-1.2, 1.2], [1.5, 1.6]])},
            "cameras": [_cam("s1", "a", [0.4, -1.7, 1.55], [0.45, 0.6, 1.5], focal=35)]}
    lib = S.make_library(tmp_path, [{"name": "desk", "position": [1.7, 1.7], "size": [0.6, 0.6, 0.8]}], size=ROOM)
    monkeypatch.setattr(SP.C, "LIBRARY", lib)
    out = tmp_path / "out"
    monkeypatch.setenv("SCENES_OUT", str(out))
    monkeypatch.delenv("SCENES_CHAR_TOOLS", raising=False)
    eng = S.FakeEngine(out)
    monkeypatch.setattr(SP, "run_scene", eng)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(dict({"id": "sc", "set": "room_x"}, **spec)))
    monkeypatch.setattr(sys, "argv", ["speakers.py", str(p), "--no-split", "--prev", str(prev)])
    assert SP.main() == 0
    staged = json.loads((out / "sc" / "staged.json").read_text())
    assert staged["positions"]["c"] == [-1.2, 1.2]            # where they stood in the shot before: no unmotivated move
    assert staged["positions"]["a"] == [0, 0.5]               # an exact place is the plan's, whatever the shot before had
    assert eng.calls[0]["positions"]["c"] == [-1.2, 1.2]      # already there in the first render


def test_a_shot_naming_two_people_keeps_both_in_the_picture(tmp_path, monkeypatch):
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([-1.23, 0.61]), "b": _char([1.21, 0.63])},
            "cameras": [dict(_cam("s1", None, [-1.2, -1.2, 1.5], [-1.2, 0.6, 1.5], focal=50), subject="a",
                             show=["a", "b"])]}                 # framed on one of them: the other is out of it
    report, result, staged, eng = stage(tmp_path, monkeypatch, spec)
    h = report["shots"]["s1"]["history"]
    assert any("b (named for the shot) is not in the picture" in w for w in h[0]["why"])
    assert report["shots"]["s1"]["ok"], h[-1]["why"]
    ff = _shot(result, "s1")["speaker_check"]["frame_facts"]
    assert ff["a"]["head_any"] and ff["b"]["head_any"]
    final = json.loads(Path(eng.calls[-1]["spec"]).read_text())["cameras"][0]
    assert final["show"] == ["a", "b"] and not final.get("tight")           # nobody named was dropped to make it pass
    assert staged["positions"] == {"a": [-1.23, 0.61], "b": [1.21, 0.63]}


def test_a_camera_search_point_directly_over_its_aim_does_not_crash(tmp_path, monkeypatch):
    spec = {"line": ["a", "b"],
            "characters": {"a": _char([-1.2, 0.6]), "b": _char([1.2, 0.6])},      # their middle is a grid point
            "cameras": [dict(_cam("s1", None, [-1.2, -1.2, 1.5], [-1.2, 0.6, 1.5], focal=50), subject="a",
                             show=["a", "b"])]}
    report, _, _, _ = stage(tmp_path, monkeypatch, spec)
    assert report["shots"]["s1"]["history"]
