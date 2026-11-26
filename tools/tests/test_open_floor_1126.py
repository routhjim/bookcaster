"""A place phrase that names no thing of the place (a part of the floor, a group of people, a direction) is the open
floor, and no piece stands for it (assets/scenes/place_spec.py open_floor); a part of the room among a rich place's
dressing is the room's own, never a piece (assets/sets/place_build.py with_dressing)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets" / "scenes"))
sys.path.insert(0, str(HERE.parent / "assets" / "sets"))
sys.path.insert(0, str(HERE.parent / "assets"))
import place_spec as PS  # noqa: E402
import place_build as PB  # noqa: E402


def spec():
    return {"pieces": [{"id": "p1", "kind": "work_piece", "look": "threshing floor, beaten earth", "phrases": ["on the threshing floor"],
                        "longest_m": 9.0},
                       {"id": "p2", "kind": "stool", "look": "a three-legged stool", "phrases": ["to the left"]},
                       {"id": "p3", "kind": "work_piece", "look": "winnowing frame, ash wood", "phrases": ["at the frame", "among the reapers"],
                        "longest_m": 1.5},
                       {"id": "p4", "kind": "bench", "look": "an oak bench", "phrases": []}],
            "relations": [{"a": "p4", "rel": "near", "b": "p1"}, {"a": "p4", "rel": "near", "b": "p3"}],
            "named_places": {"on the threshing floor": "p1", "to the left": "p2", "at the frame": "p3",
                             "among the reapers": "p3", "the middle": "floor"}}


def test_phrases_that_name_no_thing_are_the_open_floor(tmp_path, monkeypatch):
    asked = []
    says = {"on the threshing floor": "floor", "to the left": "Direction.", "at the frame": "thing", "among the reapers": "people"}

    def ask(url, text, cache):
        ph = next(k for k in says if f'"{k}"' in text)
        asked.append(ph)
        return says[ph]
    monkeypatch.setattr(PS, "ask", ask)
    sp = spec()
    out = PS.open_floor(sp, list(sp["named_places"]), ["Orla sweeps."], "u", tmp_path)
    assert out == ["on the threshing floor", "to the left", "among the reapers"]
    assert "the middle" not in asked                                 # already the floor: not asked
    assert sp["named_places"] == {"on the threshing floor": "floor", "to the left": "floor", "at the frame": "p3",
                                  "among the reapers": "floor", "the middle": "floor"}
    assert [p["id"] for p in sp["pieces"]] == ["p2", "p3", "p4"]     # the made-up floor piece is gone, furniture stays
    assert sp["pieces"][0]["phrases"] == [] and sp["pieces"][1]["phrases"] == ["at the frame"]
    assert sp["relations"] == [{"a": "p4", "rel": "near", "b": "p3"}]


def test_things_are_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(PS, "ask", lambda url, text, cache: "thing")
    sp = spec()
    assert PS.open_floor(sp, list(sp["named_places"]), [], "u", tmp_path) == [] and sp == spec()


def test_a_part_of_the_room_in_the_dressing_is_no_piece():
    sp = PB.with_dressing({"period": "georgian", "pieces": [], "relations": []},
                          {"dressing": ["oak flooring with high polish", "walls painted pale green", "stone fireplace with carved mantel",
                                        "brass urns with dried flowers"]})
    by = {r["item"]: r for r in sp["_dressing"]}
    for it in ("oak flooring with high polish", "walls painted pale green", "stone fireplace with carved mantel"):
        assert by[it]["piece"] is None, by[it]
    assert by["brass urns with dried flowers"]["piece"] and len(sp["pieces"]) == 1


def test_a_step_whose_rule_changed_runs_again_in_an_unfinished_place(tmp_path):
    import story_keys as SK
    s = SK.Story.__new__(SK.Story)
    s.work, s.places = tmp_path, {"Room": {"slug": "room"}, "Yard": {"slug": "yard"}}
    for pl in ("Room", "Yard"):
        (s.d(pl) / ".done_spec").write_text("2026-01-01 00:00:00")          # marked under the old rule
        (s.d(pl) / ".done_set").write_text("2026-01-01 00:00:00")
    (s.d("Yard") / ".done_keys").write_text("x")                             # a finished place
    assert not s.done("Room", "spec") and s.done("Yard", "spec") and s.done("Room", "set")
    s.mark("Room", "spec")
    assert s.done("Room", "spec")
