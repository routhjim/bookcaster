"""A place's words for the pictures: no sentence of what is absent, a piece by the story's own name, no ground mark
as a thing (sets/place_build.py present_only, redraw/setfacts.py, redraw/redraw.py set_words)."""
import json
import sys
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "assets"
for d in ("sets", "redraw", "scenes", ""):
    sys.path.insert(0, str(A / d))
import place_build as PB  # noqa: E402
import setfacts  # noqa: E402
import redraw as R  # noqa: E402


def test_absent_sentences_are_left_out():
    t = "The place is a bare hill of ash. There are no houses or posts on it. The ground is grey. No trees stand."
    assert PB.present_only(t) == "The place is a bare hill of ash. The ground is grey."


def test_words_of_what_is_there_are_kept():
    t = "A wide yard with a well. Nobody has swept it. North of it a wall."
    assert PB.present_only(t) == t
    assert PB.present_only("") == ""


def _set(tmp_path, fixture, marks=()):
    lib = tmp_path / "library"
    (lib / "prop" / "pc_01").mkdir(parents=True)
    (lib / "prop" / "pc_01" / "asset.json").write_text(json.dumps(
        {"description": "landmark: a hollow in grey rock. Front -Y.", "size_m": [3, 2, 2]}))
    sd = lib / "set" / "s1"
    sd.mkdir(parents=True)
    (sd / "spec.json").write_text(json.dumps({"size_m": [20, 20, 5], "fixtures": [fixture], "marks": list(marks)}))
    return sd


def test_a_piece_is_named_as_the_story_calls_it(tmp_path):
    sd = _set(tmp_path, {"asset": "pc_01", "name": "landmark", "position": [0, 0, 0], "called": "hollow oak",
                         "look": "a wide trunk split open"})
    t = setfacts.things(sd)[0]
    assert t["words"].startswith("the hollow oak (") and "landmark" not in t["words"]


def test_a_piece_without_a_story_name_keeps_its_words(tmp_path):
    sd = _set(tmp_path, {"asset": "pc_01", "name": "bench", "position": [0, 0, 0], "look": "oak, worn"})
    assert setfacts.things(sd)[0]["words"].startswith("the bench (")


def test_a_ground_mark_is_not_named_in_the_frame(tmp_path):
    shot = {"things": [{"name": "land_at5", "kind": "mark", "words": "the land at5", "visible_frac": 1.0,
                        "box": [0.4, 0.4, 0.6, 0.6]},
                       {"name": "bench", "kind": "fixture", "words": "the bench", "visible_frac": 1.0,
                        "box": [0.1, 0.4, 0.3, 0.6]}]}
    w = R.set_words(shot, outdoors=True)
    assert "bench" in w and "land at5" not in w


def test_the_storys_size_decides_whether_the_pictures_thing_is_the_piece():
    assert PB.size_agrees(0.8, [0.78, 0.62, 0.78]) and PB.size_agrees(1.5, [1.97, 1.58, 0.85])
    assert not PB.size_agrees(0.8, [1.42, 1.42, 1.03]) and not PB.size_agrees(3.0, [1.0, 0.4, 0.5])
    assert PB.size_agrees(None, [5.0, 1.0, 1.0])


def test_a_thing_the_picture_shows_is_called_by_the_surer_of_its_two_readings():
    import place_build as PB
    assert PB.seen_called({"name": "bush", "words": "A low fence with split rails, weathered"}) == "low fence"
    assert PB.seen_called({"name": "old post", "words": "Dark, split and leaning"}) == "old post"
    assert PB.seen_called({"name": "old post"}) == "old post"
    assert PB.seen_called({"name": "all of it", "group": True, "words": "Left: a hut"}) == "buildings"


def test_a_rooms_words_call_a_thing_what_the_story_calls_it_and_a_seen_window_plainly():
    import place_build as PB
    out = {"tags": ["shop"], "style": "medieval", "props": [],
           "fixtures": [{"asset": "x1", "name": "landmark", "called": "dark stone"}]}
    lay = {"openings": [{"id": PB.SEEN_OPENING + "3", "kind": "window"}, {"id": "o1", "kind": "door"}]}
    kit = {"words": {"window": "a tall window with heavy drapes", "door": "a panelled door"}}
    recs = {"x1": {"piece_kind": "landmark", "look": "dark stone, low and long, faintly lit"}}
    w, _ = PB.room_words(out, lay, kit, recs)
    assert "landmark" not in w and "dark stone (low and long, faintly lit)" in w
    assert "drapes" not in w and "a window" in w and "a panelled door" in w


def test_what_a_thing_is_likened_to_is_not_in_the_words_it_is_drawn_from():
    import place_build as PB
    assert PB.no_likeness("grey rock, size of a sleeping cat, faintly lit") == "grey rock, faintly lit"
    assert PB.no_likeness("a lump as big as a cart wheel; rough") == "a lump; rough"
    assert PB.no_likeness("oak chest, iron bound") == "oak chest, iron bound"
    assert PB.no_likeness("the size of a barn") == "the size of a barn"       # (nothing else to draw it from)


def test_a_place_at_night_has_no_sun_in_its_room_and_the_moon_over_open_land(tmp_path):
    import place_build as PB
    import inspect
    src = inspect.getsource(PB)
    # the rule's own numbers hold together: at night a window gives a little, the sky less, the moon is not the sun
    assert 0 < PB.NIGHT_WINDOW < 0.2 and 0 < PB.NIGHT_SKY < 0.2 and 0 < PB.NIGHT_SUN < PB.DUSK_SUN < 1
    assert "night" in PB.DARK_TIMES and "dusk" not in PB.DARK_TIMES
    assert 'kit["lights"].get("sun") and not dark' in src


def test_the_time_of_day_is_read_from_one_word_of_the_answer(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
    import place_spec as PS
    plan = {"s1": [{"place": "P", "setting": "A yard.", "action": "They wait."}], "_es": []}
    monkeypatch.setattr(PS, "passage", lambda *a: "text")
    for said, want in (("Night", "night"), ("<think>day?</think>\ndusk.", "dusk"), ("unknown", ""), ("day", "day")):
        monkeypatch.setattr(PS, "ask", lambda url, text, cache, said=said: said)
        assert PS.time_of_day(plan, ["s1"], [], "P", "u", tmp_path) == want


def test_a_period_without_a_kit_takes_nothing_of_another_periods_style():
    import place_build as PB
    own, lent = PB.kit_for("victorian"), PB.kit_for("medieval")
    assert own.get("window_dressing") and not own.get("_plain") and own["window"].get("glass_material")
    assert lent["_plain"] and lent["window_dressing"] == [] and lent["window"]["glass_material"] is None
    assert lent["window"]["panes"] == [1, 1] and lent["door"]["material"] == lent["door"]["frame_material"]
    w = " ".join(lent["words"].values())
    assert "sash" not in w and "drapes" not in w and "panelled" not in w and "grate" not in w
    assert lent["lights"] == own["lights"]                    # (its light is kept)
