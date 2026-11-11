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
