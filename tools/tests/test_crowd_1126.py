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
    assert len(few) == 4 and len(many) == PB.CROWD_MANY_MAX            # 96 m2: one to 20 m2, one to 4 m2 (capped)
    assert len({p["look"] for p in many}) == 4                         # two groups, two looks of each, stood again
    assert {p["against"] for p in many} == {"wall", "free"} and {p["against"] for p in few} == {"free"}
    who = PB.crowd_line(dict(base, _crowd_who="fair-goers in Sunday clothes"))
    assert who.startswith("fair-goers in Sunday clothes; townsfolk")
    led = [p["look"] for p in PB.with_crowd(dict(base, _crowd_size="many", _crowd_who="fair-goers in Sunday clothes"))["pieces"]]
    mine = [x for x in led if x.startswith("fair-goers")]
    assert len(mine) == 2 * len(led) // 3 and len(set(mine)) == len(PB.CROWD_LOOKS)   # the story's people lead, in four looks
    assert any(x.startswith("townsfolk") for x in led) and any(x.startswith("porters") for x in led)
    assert PB.with_crowd(dict(base, crowd=""))["pieces"] == []


def test_one_of_a_crowd_is_only_the_figure_made_from_its_own_words():
    piece = {"id": "x1", "kind": "bystander", "look": "fair-goers in Sunday clothes, one of them", "against": "free"}
    want = P.generation_request(piece)["asset_id"]
    other = {"asset_id": "bystander_gen_000000_01_tr", "kind": "bystander", "tags": ["bystander"], "class": "prop",
             "description": "bystander: fair-goers in Sunday clothes, one of them, older", "size_m": [0.6, 0.45, 1.75]}
    assert P.resolve([piece], [other])["pieces"][0]["status"] == "generate"
    assert P.resolve([piece], [dict(other, asset_id=want)])["pieces"][0]["status"] == "library"


def test_the_depth_guide_keeps_the_head_and_not_its_features(tmp_path):
    import numpy as np
    from PIL import Image
    import redraw as RD
    d = np.full((90, 160), 40, np.uint8)
    d[20:60, 60:100] = 200
    d[38:42, 78:82] = 90                                              # a feature inside the head
    m = np.zeros((90, 160), np.uint8)
    m[20:60, 60:100] = 255
    Image.fromarray(d).save(tmp_path / "c_depth.png")
    Image.fromarray(m).save(tmp_path / "c_mask_heads.png")
    out = np.asarray(Image.open(RD.soft_heads(tmp_path, "c")))
    assert out[40, 80] > 150 and out[5, 5] == 40                      # the feature is gone, the room untouched
    assert RD.soft_heads(tmp_path, "nothing") is None


def test_how_full_a_place_is_is_asked(tmp_path, monkeypatch):
    for ans, want in (("many: fair-goers in their Sunday clothes.", ("many", "fair-goers in their sunday clothes")),
                      ("Few", ("few", "")), ("none: nobody", ("none", "")), ("I cannot say", ("few", ""))):
        monkeypatch.setattr(PS, "ask", lambda url, text, cache, a=ans: a)
        assert PS.crowd_size("The Hall", {"crowd": "townsfolk"}, "Orla crossed the hall.", "u", tmp_path) == want
    monkeypatch.setattr(PS, "ask", lambda url, text, cache: "?")
    assert PS.crowd_size("The Hall", {}, "Orla crossed the hall.", "u", tmp_path) == ("none", "")


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


def test_the_story_read_is_the_plans_own_chapter(tmp_path):
    import story_keys as SK
    b = tmp_path / "book"
    (b / "work" / "video" / "02_two" / "shots").mkdir(parents=True)
    for n in ("01_one", "02_two"):
        (b / "work" / f"{n}.timeline.json").write_text("[]")
        (b / "work" / f"{n}.raw.timeline.json").write_text("[]")
    plan = b / "work" / "video" / "02_two" / "shots" / "plan_x.json"
    assert SK.chapter_timeline(b, plan).name == "02_two.timeline.json"
    assert SK.chapter_timeline(b, tmp_path / "elsewhere" / "plan.json") is None      # several chapters, none told
    (b / "work" / "02_two.timeline.json").unlink()
    assert SK.chapter_timeline(b, tmp_path / "elsewhere" / "plan.json").name == "01_one.timeline.json"


def test_a_key_of_a_place_with_a_crowd_says_so(tmp_path):
    import json
    import story_keys as SK
    keys, sd = tmp_path / "keys", tmp_path / "set"
    keys.mkdir()
    sd.mkdir()
    for i in ("s1", "s2"):
        (keys / f"{i}.shot_facts.json").write_text(json.dumps({"visible": [{"name": "Orla"}], "count": 1}))
    (sd / "spec.json").write_text(json.dumps({"crowd": "fair-goers"}))
    assert SK.crowd_facts(keys, ["s1", "s3"], sd) == 1
    assert json.loads((keys / "s1.shot_facts.json").read_text())["crowd"] == "fair-goers"
    assert "crowd" not in json.loads((keys / "s2.shot_facts.json").read_text())
    (sd / "spec.json").write_text(json.dumps({"crowd": ""}))
    assert SK.crowd_facts(keys, ["s2"], sd) == 0 and SK.crowd_facts(keys, ["s2"], tmp_path / "none") == 0
