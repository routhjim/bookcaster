"""Redraw words (tools/assets/redraw): things named with their look, big props named, face references, posture."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "redraw"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))

import redraw  # noqa: E402
import setfacts  # noqa: E402


def _lib(tmp: Path) -> Path:
    lib = tmp / "library"
    for cls, aid, desc, extra in (("prop", "rug_x", "Flat knotted rug in an oriental pattern, 3 x 2 m: red field", {}),
                                  ("prop", "chair_x", "Victorian wingback armchair, front -Y: tall back", {}),
                                  ("character", "sitter_x", "A figure", {"tags": ["character", "seated"]})):
        d = lib / cls / aid
        d.mkdir(parents=True)
        (d / "asset.json").write_text(json.dumps(dict({"description": desc, "size_m": [1, 1, 1]}, **extra)))
    s = lib / "set" / "room_x"
    s.mkdir(parents=True)
    (s / "spec.json").write_text(json.dumps({
        "size_m": [5, 5, 3], "structure": [], "lights": [],
        "fixtures": [{"asset": "chair_x", "name": "armchair", "position": [0, 1], "look": "faded gold damask"}],
        "props": [{"asset": "rug_x", "name": "rug", "position": [0, 0], "look": "a deep red carpet"}]}))
    return lib


def test_things_carry_their_look(tmp_path):
    lib = _lib(tmp_path)
    th = {t["name"]: t for t in setfacts.things(lib / "set" / "room_x")}
    assert th["armchair"]["words"] == "the armchair (Victorian wingback armchair; faded gold damask)"
    assert "a deep red carpet" in th["rug"]["words"]


def test_big_free_prop_is_named_small_one_is_not():
    big = {"name": "picture", "kind": "prop", "words": "the picture (a landscape)", "box": [0.1, 0.1, 0.3, 0.3],
           "visible_frac": 0.9}
    small = dict(big, name="vase", words="the vase", box=[0.5, 0.5, 0.52, 0.52])
    w = redraw.set_words({"things": [big, small]})
    assert "the picture (a landscape)" in w and "the vase" not in w


def test_face_refs_only_for_faces_shown():
    cast = {"a": {"face_ref": "/f/a.png"}, "b": {"face_ref": "/f/b.png"}}
    ppl = [("a", [0, 0, 0.4, 1], 30.0, "standing", 30.0, None), ("b", [0.6, 0, 1, 1], 150.0, "standing", 150.0, None)]
    assert redraw.faces(ppl, cast) == ["/f/a.png"]


def test_seated_figure_and_its_seat(tmp_path, monkeypatch):
    lib = _lib(tmp_path)
    import config as C
    monkeypatch.setattr(C, "LIBRARY", lib)
    sd = tmp_path / "scene"
    sd.mkdir()
    (sd / "job.json").write_text(json.dumps({"job": {
        "characters": {"a": {"body_pose": {"pose": "stand_relaxed"}}},
        "figures": {"b": {"id": "character/sitter_x"}}}}))
    poses = redraw.scene_poses(sd)
    assert poses == {"a": "stand_relaxed", "b": "seated"}
    shot = {"people": {"b": {"box": [0.3, 0.1, 0.7, 1.0], "area": 0.3, "head_to_camera_deg": 10.0}}}
    ppl = redraw.people(shot, poses)
    words = redraw.people_words(ppl, cast={"b": {"noun": "woman", "seat": "a leather armchair"}})
    assert "is sitting" in words and "She sits in a leather armchair." in words


def test_two_men_on_one_side_are_counted_from_the_left():
    ppl = [("a", [0.0, 0, 0.2, 1], 30.0, "standing", 30.0, None), ("c", [0.1, 0, 0.3, 1], 30.0, "standing", 30.0, None)]
    nm = redraw.names(ppl, {"a": {"noun": "man"}, "c": {"noun": "man"}})
    assert nm == {"a": "the first man from the left", "c": "the second man from the left"}


def test_seated_person_sits_in_the_chair_under_her():
    ppl = [("b", [0.6, 0.3, 0.8, 0.9], 20.0, "sitting", 20.0, None)]
    shot = {"things": [{"name": "armchair", "words": "the armchair (gold damask)", "box": [0.55, 0.5, 0.85, 0.95]},
                       {"name": "armchair_2", "words": "the armchair (leather)", "box": [0.9, 0.5, 1.0, 0.95]}]}
    assert redraw.seats_of(ppl, shot) == {"b": "the armchair (gold damask)"}


def test_unlit_thing_keeps_its_look():
    t = {"name": "drapes", "kind": "fixture", "words": "the drapes (damask with a fringed pelmet, gold)",
         "box": [0.1, 0.0, 0.4, 0.8], "visible_frac": 0.9, "flame_in_view": None}
    assert "fringed pelmet, gold" in redraw.set_words({"things": [t]})


def test_a_cabins_view_is_told_for_whatever_glass_the_frame_shows(tmp_path):
    import json
    view = {"name": "outside_view_1", "words": "a dark mountain of ice, near", "ahead": [0.0, -1.0], "all": ["outside_view"]}
    cams = [{"name": "fwd", "position": [0, 2, 1.2], "target": [0, 0, 1.2], "view": view, "setting": "the director's words"},
            {"name": "placed", "position": None, "target": None, "view": view, "setting": "the director's words"},
            {"name": "plain", "position": [0, 2, 1.2], "target": [0, 0, 1.2], "setting": "the director's words"}]
    (tmp_path / "spec_round0.json").write_text(json.dumps({"cameras": cams}))
    fwd, placed, plain = (redraw.setting_words(tmp_path, n) for n in ("fwd", "placed", "plain"))
    assert "seen through its glass" in fwd and "a dark mountain of ice, near." in fwd
    assert placed == fwd                                # (the engine hangs the view where each camera looks)
    assert plain == " In this shot: the director's words"                       # a place with no views: as before


def test_what_a_person_holds_is_told_except_in_a_close_shot(tmp_path):
    import json
    (tmp_path / "spec_round0.json").write_text(json.dumps({"cameras": [
        {"name": "c1", "holds": {"a": "helmet (a dark rounded pilot helmet)"}}, {"name": "c2"}]}))
    ppl = [("a", [0.3, 0.1, 0.6, 1.0], 10.0, "standing", 10.0, None)]
    w = redraw.holds_words(tmp_path, "c1", ppl, {}, "medium")
    assert "holds in the hands, not worn: helmet (a dark rounded pilot helmet); it is in the hand" in w
    assert redraw.holds_words(tmp_path, "c1", ppl, {}, "close") == ""        # the hands are out of a close shot
    assert redraw.holds_words(tmp_path, "c2", ppl, {}, "medium") == ""


def test_what_a_person_has_on_is_told_in_every_kind_of_shot(tmp_path):
    import json
    (tmp_path / "spec_round0.json").write_text(json.dumps({"cameras": [
        {"name": "c1", "wears": {"a": "a dark flight helmet with its visor up"}}, {"name": "c2"}]}))
    ppl = [("a", [0.3, 0.1, 0.6, 1.0], 10.0, "sitting", 10.0, None)]
    w = redraw.wears_words(tmp_path, "c1", ppl, {})
    assert "has on, worn and not held: a dark flight helmet with its visor up." in w
    assert redraw.wears_words(tmp_path, "c2", ppl, {}) == ""


def test_a_person_in_something_the_book_has_a_sheet_of_takes_that_sheet():
    w = {"sheet": "/s/usual.png", "worn_sheets": {"flight helmet": "/s/helmet.png", "rain cape": "/s/cape.png"}}
    assert redraw.worn_sheet(w, "helmet") == "/s/helmet.png"
    assert redraw.worn_sheet(w, "a long cape for the rain") == "/s/cape.png"
    assert redraw.worn_sheet(w, "gloves") is None and redraw.worn_sheet(w, None) is None
