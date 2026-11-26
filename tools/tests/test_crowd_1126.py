"""A place's crowd: the story's and the director's unnamed people together, more figures and a fuller picture where
the place is full of people, and the generation looks pieces up as the set builder does (assets/sets/place_build.py,
assets/scenes/place_spec.py, assets/redraw/redraw.py, assets/places/pieces.py)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for d in ("scenes", "sets", "places", "redraw", ""):
    sys.path.insert(0, str(HERE.parent / "assets" / d))
import place_build as PB  # noqa: E402
import place_spec as PS  # noqa: E402
import pieces as P  # noqa: E402


def test_the_crowd_is_the_storys_people_and_the_directors():
    sp = {"crowd": "townsfolk at the stalls", "_director_crowd": "porters with handcarts; Townsfolk at the stalls"}
    assert PB.crowd_line(sp) == "townsfolk at the stalls; porters with handcarts"
    assert PB.crowd_line({"_director_crowd": "two clerks."}) == "two clerks" and PB.crowd_line({}) == ""


def test_a_full_place_stands_more_figures_of_the_same_few_looks():
    base = {"kind": "hall", "size": "large", "period": "georgian", "crowd": "townsfolk; porters", "pieces": [],
            "_room_m": [8.0, 12.0, 4.0]}
    few = [p for p in PB.with_crowd(base)["pieces"] if p["kind"] == "bystander"]
    many = [p for p in PB.with_crowd(dict(base, _crowd_size="many"))["pieces"] if p["kind"] == "bystander"]
    assert len(few) == 4 and len(many) == PB.CROWD_MANY_MAX            # 96 m2: one to 20 m2, one to 8 m2 (capped)
    assert len({p["look"] for p in many}) == 4                         # two groups, two looks of each, stood again
    assert PB.with_crowd(dict(base, crowd=""))["pieces"] == []


def test_how_full_a_place_is_is_asked(tmp_path, monkeypatch):
    for ans, want in (("many", "many"), ("Few.", "few"), ("none", "none"), ("I cannot say", "few")):
        monkeypatch.setattr(PS, "ask", lambda url, text, cache, a=ans: a)
        assert PS.crowd_size("The Hall", {"crowd": "townsfolk"}, "Orla crossed the hall.", "u", tmp_path) == want
    monkeypatch.setattr(PS, "ask", lambda url, text, cache: "?")
    assert PS.crowd_size("The Hall", {}, "Orla crossed the hall.", "u", tmp_path) == "none"


def test_a_full_place_is_drawn_full():
    import redraw as RD
    few, many = RD.crowd_words("townsfolk", "medium"), RD.crowd_words("townsfolk", "medium", "many")
    assert "Far behind them" in few and "crowded" in many and "dozens" in many and "townsfolk" in many
    assert "none stands between the camera" in many
    assert RD.crowd_words("townsfolk", "close", "many") == "" and RD.crowd_words("", "medium", "many") == ""


def test_pieces_of_another_period_are_not_found_for_this_place():
    recs = [{"asset_id": "a", "tags": ["bystander", "medieval"], "description": "x"},
            {"asset_id": "b", "tags": ["bystander"], "description": "a georgian footman"},
            {"asset_id": "c", "tags": ["bystander"], "description": "a plain figure"}]
    assert [r["asset_id"] for r in P.period_records({"period": "georgian"}, recs)] == ["b", "c"]
    assert P.period_records({"period": "not stated"}, recs) == recs
