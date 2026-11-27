"""A place in the open whose ground is itself built (11-26): a made floor with an edge round it, under the sky (a
deck, a pier, a rooftop, a terrace, a platform) is read and built as a room with no roof, not as open land. The
concept reader asks one more closed question of a place read "outdoors" (land or floor) and lists a made floor's
things as a room's; the set builder keeps the floor and a low wall round its edge, builds no ceiling, sets the place's
world (the far land and the sky) round it and lights it from there; its cameras stay on the floor and the redraw still
calls the floor a floor. Every other kind of place is as before. Made-up data only."""
import json
import sys
from pathlib import Path

import pytest

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("sets", "places", "scenes", "redraw", ""):
    sys.path.insert(0, str(A / sub))
import concept as CO  # noqa: E402
import pieces as P  # noqa: E402
import place_build as PB  # noqa: E402
import speakers as SPK  # noqa: E402


# ------------------------------------------------------------------ 1. the concept reader
def judge(space: str, ground: str = "", calls: list | None = None):
    """A stand-in for the vision judge: answers by item id."""
    def ask(items):
        if calls is not None:
            calls.append([it["id"] for it in items])
        out = {}
        for it in items:
            out[it["id"]] = {
                "space": {"space": space}, "ground": {"ground": ground},
                "view": {"view": "whole", "cut": "no"},
                "surf": {"walls": "Boards, brown.", "floor": "Planks, pale.", "ceiling": "The sky, blue.",
                         "light": "From the sun, white.", "far": "Low hills."},
                "seen": {"seen": "chair; wood, brown\nrailing; wood, brown\nlamp post; iron, black"},
                "seen_open": {"seen": "shed; timber, grey\nboulder; stone, grey"},
                "seats": {"abreast": "1", "single": "one"},
            }.get(it["id"], {"there": "no", "look": "none"})
        return out
    return ask


SPEC = {"place": "the Terrace", "kind": "outdoors", "pieces": [], "relations": []}


def test_the_question_before_it_is_word_for_word_as_it_was():
    # (answers are kept by the question's own words: a changed word asks every place of every book again)
    assert CO.SPACE_ASK == (
        "What kind of place does this picture show? Answer with exactly one word: room (the inside of a room, "
        "hall, bay or living quarters, on land or aboard a ship: people stand and walk about in it), cockpit (a "
        "cockpit, driving cab or carriage: a space too small to walk about in, where people only sit at its "
        "controls or in its seats), or outdoors.")
    assert "land or floor" in CO.GROUND_ASK


def test_ground_of_reads_one_word():
    assert CO.ground_of("Floor.") == "built" and CO.ground_of(" floor") == "built"
    assert CO.ground_of("land") == "land" and CO.ground_of("") == "land" and CO.ground_of(None) == "land"


def test_a_made_floor_in_the_open_is_read_as_a_room_is():
    calls = []
    res = CO.read(Path("/c/concept.png"), SPEC, judge("outdoors", "floor", calls))
    assert res["space"] == "outdoors" and res["ground"] == "built" and CO.made_floor(res)
    assert [s["name"] for s in res["seen"]] == ["chair", "railing", "lamp post"]      # (the room's list, not the land's)
    # (asked once, after the space; 11-27 with it, what lies round the place: tests/test_outdoor_round_side_1127.py)
    assert calls[1] == ["ground", "around"]


def test_open_land_is_read_as_before():
    res = CO.read(Path("/c/concept.png"), SPEC, judge("outdoors", "land"))
    assert res["ground"] == "land" and not CO.made_floor(res)
    assert [s["name"] for s in res["seen"]] == ["shed", "boulder"]


@pytest.mark.parametrize("space", ["room", "cockpit"])
def test_a_room_and_a_cockpit_are_not_asked(space):
    calls = []
    res = CO.read(Path("/c/concept.png"), SPEC, judge(space, "floor", calls))
    assert "ground" not in res and len(calls) == 1 and "ground" not in calls[0]
    assert [s["name"] for s in res["seen"]] == ["chair", "railing", "lamp post"]


def boxes(calls: list):
    """The judge at the second reading (pieces_from): a box and a kind for each thing."""
    def ask(items):
        calls.append(items)
        out = {}
        for it in items:
            if it["id"].startswith("s:") and "kind" in it["ask"]:
                i = int(it["id"][2:])
                out[it["id"]] = {"box": f"[{100 + 250 * i}, 300, {300 + 250 * i}, 900]",
                                 "kind": ["chair", "bench", "none"][i], "count": "1", "sides": "one",
                                 "own": "separate", "real": "1 1", "made": "made"}
            elif it["id"].startswith("w:"):
                out[it["id"]] = {"box": "[0, 600, 1000, 1000]", "wall": "Wide boards."}
            else:
                out[it["id"]] = {"words": "A black iron post with a lamp on it", "box": "[0, 0, 1000, 1000]",
                                 "real": "8 3"}
        return out
    return ask


def picture(tmp_path) -> Path:
    from PIL import Image
    p = tmp_path / "concept.png"
    Image.new("RGB", (64, 36), (120, 90, 60)).save(p)
    return p


def test_a_made_floors_things_are_each_cut_out_and_its_own_edge_is_no_thing(tmp_path):
    pic = picture(tmp_path)
    res = CO.read(pic, SPEC, judge("outdoors", "floor"))
    calls = []
    CO.pieces_from(pic, SPEC, res, boxes(calls), tmp_path / "crops", "t", ["chair", "bench", "work_piece", "landmark"])
    kinds = {s["name"]: s.get("kind") for s in res["seen"]}
    # (the chair by its kind; the railing is the floor's own edge, whatever kind the judge gave it; the post, of no
    # kind, is a thing of the place as it would be in a room)
    assert kinds == {"chair": "chair", "railing": None, "lamp post": "work_piece"}
    assert not any(s.get("group") for s in res["seen"])                 # (nothing is made as one group piece)
    asked = [it for c in calls for it in c]
    assert all("count" in it["ask"] and "own" not in it["ask"] for it in asked if "kind" in it["ask"])
    assert not any(it["id"] in ("g", "l") for it in asked)
    assert res["seen"][2]["words"].startswith("A black iron post")


def test_open_lands_things_are_read_as_before(tmp_path):
    pic = picture(tmp_path)
    res = CO.read(pic, SPEC, judge("outdoors", "land"))
    calls = []
    CO.pieces_from(pic, SPEC, res, boxes(calls), tmp_path / "crops", "t", ["chair", "bench", "work_piece", "landmark"])
    asked = [it for c in calls for it in c]
    assert all("own" in it["ask"] and "count" not in it["ask"] for it in asked if "kind" in it["ask"])
    assert any(it["id"] == "g" for it in asked)                         # (what is built there, as one group)
    assert any(s.get("group") for s in res["seen"])


# ------------------------------------------------------------------ 2. the place spec against its concept
def spec_of(**kw):
    s = {"room": "A wide timber terrace over the water.", "schema": "place_spec/1", "place": "the Terrace",
         "kind": "outdoors", "period": "medieval", "size": "medium", "openings": [], "outside": "",
         "pieces": [{"id": "p1", "kind": "chair", "look": "a plain oak chair", "phrases": ["by the chair"],
                     "against": "free"},
                    {"id": "p2", "kind": "bench", "look": "a bench", "phrases": [], "against": "free"}],
         "relations": [], "named_places": {"by the chair": "p1", "at the rail": "floor", "by the door": "o1"},
         "land_places": {}, "_director_room": "Outside: a wide timber terrace over the water"}
    s.update(kw)
    return s


def concept_of(**kw):
    c = {"concept": "/c/concept.png", "space": "outdoors", "ground": "built", "seats_abreast": 1,
         "surfaces": {"walls": "boards, brown", "floor": "planks, pale", "ceiling": "the sky, blue"},
         "light": "from the sun, white", "far": "low hills beyond the water",
         "surfaces_words": "The walls: boards, brown; the floor: planks, pale; the ceiling: the sky, blue.",
         "bare": {"walls": {"colour": "#5a4030"}, "floor": {"colour": "#b09060"}},
         "seen": [{"name": "lamp post", "look": "iron, black", "kind": "work_piece", "box": [0.6, 0.1, 0.7, 0.9],
                   "crop": "/c/crops/post.png", "words": "A black iron post with a lamp on it"}],
         "spec": {"p1": {"in_picture": True, "look": "oak, brown", "box": [0.1, 0.4, 0.3, 0.9], "crop": "/c/crops/p1.png"},
                  "p2": {"in_picture": False, "look": ""}},
         "depth": {"room_m": [7.0, 8.0, 2.2], "eye": [0.0, -3.0, 1.5],
                   "things": {"spec:p1": {"xy": [-1.5, 1.0], "size_m": [0.6, 0.6, 1.0], "against": None, "facing": [0.0, -1.0]},
                              "seen:0": {"xy": [2.0, 2.5], "size_m": [0.3, 0.3, 3.0], "against": "north", "facing": [0.0, -1.0]}},
                   "off_floor": {},
                   "faces": [{"face": "floor", "file": "/c/floor.png", "u": [0, -3, 3], "v": [0, -4, 4]},
                             {"face": "north", "file": "/c/north.png", "u": [0, -3, 3], "v": [0, 0, 2]}]}}
    c.update(kw)
    return c


def test_a_made_floor_is_a_room_with_no_roof():
    out = PB.with_concept(spec_of(_roof="rafters", _views={"s1": "hills"}), concept_of())
    assert out["kind"] == "outdoors" and out["_open_top"] is True and out["_edge_m"] == PB.OPEN_EDGE_M
    assert out["openings"] == [] and "_roof" not in out and "_views" not in out
    # the floor as the picture measures it; its height is the open stage's, not a room wall's
    assert out["_room_m"][:2] == [7.0, 8.0] and out["_room_m"][2] == PB.OPEN_TOP_STAGE_M
    assert PB.OPEN_EDGE_M < 1.5 < PB.WALL_MIN_M
    by = {p["id"]: p for p in out["pieces"]}
    # the spec's piece and the picture's own thing, each where the picture has it; the guess the picture lacks is gone
    assert by["p1"]["at_xy"] == [-1.5, 1.0] and by["p1"]["source"] == "/c/crops/p1.png"
    assert by["c1"]["kind"] == "work_piece" and by["c1"]["at_xy"] == [2.0, 2.5] and by["c1"]["against"] == "wall"
    assert "p2" not in by
    assert not any(p.get("group") or p.get("kind") == "landmark" for p in out["pieces"])
    # round it, the place's world: the far land and the sky in words alone, not the floor, and no ground map
    t = [p for p in out["pieces"] if p["kind"] == "terrain"]
    assert len(t) == 1 and t[0] is out["pieces"][-1]
    assert "low hills" in t[0]["look"] and "the sky, blue" in t[0]["look"] and "planks" not in t[0]["look"]
    assert "source" not in t[0] and "map_words" not in t[0] and "view" not in t[0]
    assert "terrace" in t[0]["land_words"] and "Outside" not in t[0]["land_words"]
    assert "the floor: planks, pale" in t[0]["surface_words"] and "ceiling" not in t[0]["surface_words"]
    assert out["named_places"] == {"by the chair": "p1", "at the rail": "floor", "by the door": "floor"}
    # its floor as the picture shows it; no wall's picture (the walls are its low edge)
    assert [f["face"] for f in out["_concept"]["faces"]] == ["floor"]
    assert out["_concept"]["faces"][0]["room_m"] == out["_room_m"]


def test_a_room_kind_read_as_a_made_floor_is_built_so_too():
    out = PB.with_concept(spec_of(kind="hall", openings=[{"id": "o1", "kind": "door", "wall": "north", "words": ""}]),
                          concept_of())
    assert out["kind"] == "outdoors" and out["_open_top"] and out["openings"] == []


def test_open_land_is_built_as_before():
    for con in (concept_of(ground="land"), {k: v for k, v in concept_of().items() if k != "ground"}):
        out = PB.with_concept(spec_of(), con)
        assert out["kind"] == "outdoors" and "_open_top" not in out and "_edge_m" not in out
        assert all(p["against"] == "free" for p in out["pieces"])
        t = next(p for p in out["pieces"] if p["kind"] == "terrain")
        assert "map_words" in t and "faces" not in out["_concept"]
        assert out == PB.with_concept(spec_of(), dict(con, ground="land"))


def test_a_room_is_built_as_before_whatever_ground_says():
    room = concept_of(space="room")
    a = PB.with_concept(spec_of(kind="hall"), room)
    b = PB.with_concept(spec_of(kind="hall"), {k: v for k, v in room.items() if k != "ground"})
    assert a == b and "_open_top" not in a and a["kind"] == "hall" and a["_room_m"][2] == PB.WALL_MIN_M
    assert not any(p["kind"] == "terrain" for p in a["pieces"])


# ------------------------------------------------------------------ 3. the set
WORLD = {"stand_m": [1.5, -2.0, 40.0], "flat_r_m": 9.0,
         "sun": {"dir": [0.3, -0.9, 0.05], "color": "#ffb070", "level": 0.4, "peak": 0.9}}


def build(tmp_path, monkeypatch, spec, world=True, sizes=None):
    lib = tmp_path / "lib"
    (lib / "prop").mkdir(parents=True, exist_ok=True)
    t = next((p for p in spec["pieces"] if p["kind"] == "terrain"), None)
    if world and t:
        d = lib / "prop" / P.generation_request(t)["asset_id"]
        d.mkdir(parents=True)
        (d / "asset.json").write_text(json.dumps({"id": d.name, "size_m": [300, 300, 160], "world": WORLD}))

    def records_for(sp, library):            # (the library's match for each piece: a made-up record)
        rec = {p["id"]: {"asset": f"{p['kind']}_x", "size_m": (sizes or {}).get(p["id"], [0.6, 0.6, 0.95])}
               for p in sp["pieces"] if p["kind"] != "terrain"}
        return rec, P.resolve(sp["pieces"], [])
    monkeypatch.setattr(PB, "records_for", records_for)
    return PB.build(json.loads(json.dumps(spec)), "set_x", lib)


@pytest.fixture
def terrace(tmp_path, monkeypatch):
    sp = PB.with_concept(spec_of(pieces=spec_of()["pieces"] + [
        {"id": "p3", "kind": "lantern", "look": "a lantern", "phrases": ["under the lantern"], "against": "free",
         "gives_light": True, "hangs_m": 2.0}]), concept_of())
    return build(tmp_path, monkeypatch, sp)


def test_the_floor_and_a_low_wall_round_its_edge_and_nothing_overhead(terrace):
    out, named, report = terrace
    kinds = [s["kind"] for s in out["structure"]]
    assert kinds.count("floor") == 1 and kinds.count("wall") == 4
    assert "ceiling" not in kinds and "trusses" not in kinds
    for w in (s for s in out["structure"] if s["kind"] == "wall"):
        assert w["height_m"] == PB.OPEN_EDGE_M and w["openings"] == []
        assert not any(k in w for k in PB.SHELL_DRESSING)               # (no dado, rails or panels of a room's wall)
    floor = next(s for s in out["structure"] if s["kind"] == "floor")
    wall = next(s for s in out["structure"] if s["kind"] == "wall")
    assert floor["extend_m"] == wall["thickness_m"]                     # (the floor ends at its edge)
    assert out["size_m"][:2] == [7.0, 8.0] and out["size_m"][2] >= PB.OPEN_TOP_STAGE_M
    # the picture's own colours on them
    assert out["materials"][floor["material"]]["color"] == "#b09060"
    assert out["materials"][wall["material"]]["color"] == "#5a4030"


def test_its_world_lies_round_it_and_lights_it(terrace):
    out, named, report = terrace
    w = out["fixtures"][0]
    assert w["world"] and w["outside"] and w["name"] == "world"
    assert w["position"] == [-1.5, 2.0, round(-40.0 - PB.OPEN_FLOOR_OVER_M, 3)]      # (its ground under the floor)
    assert report["world"]["world"] == w["asset"] and report["world"]["why"] == "placed"
    assert out["interior"] is False and out["open_top"] is True and out["world"]["flat_r_m"] == 9.0
    types = [l["type"] for l in out["lights"]]
    assert types.count("sun") == 1 and types.count("sky") == 1          # (the world's own: no light of a sealed room)
    assert not any(l.get("type") == "point" and l["position"][2] > 3.0 for l in out["lights"])
    assert "room_light" not in report
    # the lantern of the place still burns, hung as high as the place has it (11-27: on a post rising from the
    # edge, not forced down onto the edge itself: tests/test_open_floor_size_night_1127.py)
    lan = next(f for f in out["fixtures"] + out["props"] if f["name"].startswith("lantern"))
    assert lan["hang"] and lan["position"][2] == 2.0
    assert any(l["type"] == "point" for l in out["lights"])
    assert named["by the chair"] and named["at the rail"] == "floor"


def test_its_words_say_the_open_air_and_keep_the_floor(terrace):
    out, _, _ = terrace
    assert out["words"].startswith("In the open air, medieval: a wide timber terrace over the water; standing there: ")
    assert "the floor: planks, pale" in out["surfaces_words"].lower()
    assert "the low wall round its edge: boards, brown" in out["surfaces_words"]
    assert "ceiling" not in out["surfaces_words"]


def test_the_cameras_stay_on_the_floor(terrace, tmp_path):
    out, _, _ = terrace
    assert SPK.open_ground(out, tmp_path / "lib") == []
    # (a place on open land with the same world: its flat ground, as before)
    land = {k: v for k, v in out.items() if k != "open_top"}
    assert SPK.open_ground(land, tmp_path / "lib")[2] == 8.5
    for c in out["cameras"]:
        assert abs(c["position"][0]) < out["size_m"][0] / 2 and abs(c["position"][1]) < out["size_m"][1] / 2


def test_without_its_world_the_set_says_so(tmp_path, monkeypatch):
    out, _, report = build(tmp_path, monkeypatch, PB.with_concept(spec_of(), concept_of()), world=False)
    assert report["world"]["world"] is None                              # (story_keys step_set stops on it)
    assert [s["kind"] for s in out["structure"]].count("wall") == 4 and out["open_top"] is True


def test_a_tall_thing_on_the_floor_is_under_the_stage(tmp_path, monkeypatch):
    out, _, _ = build(tmp_path, monkeypatch, PB.with_concept(spec_of(), concept_of()), sizes={"c1": [0.4, 0.4, 9.0]})
    assert out["size_m"][2] == 9.0 + PB.OPEN_HEAD_M
    assert all(w["height_m"] == PB.OPEN_EDGE_M for w in out["structure"] if w["kind"] == "wall")


def test_an_edge_the_picture_does_not_show_is_of_the_floors_material(tmp_path, monkeypatch):
    con = concept_of(bare={"floor": {"colour": "#b09060"}})
    out, _, _ = build(tmp_path, monkeypatch, PB.with_concept(spec_of(), con))
    floor = next(s for s in out["structure"] if s["kind"] == "floor")
    assert all(w["material"] == floor["material"] for w in out["structure"] if w["kind"] == "wall")


def test_a_room_keeps_its_ceiling_and_its_walls(tmp_path, monkeypatch):
    sp = PB.with_concept(spec_of(kind="hall"), concept_of(space="room"))
    out, _, report = build(tmp_path, monkeypatch, sp)
    kinds = [s["kind"] for s in out["structure"]]
    assert "ceiling" in kinds and "open_top" not in out and "world" not in out and "world" not in report
    assert all("height_m" not in w for w in out["structure"] if w["kind"] == "wall")
    assert next(s for s in out["structure"] if s["kind"] == "floor") == PB.kit_for("medieval")["floor"]


def test_open_land_keeps_no_structure(tmp_path, monkeypatch):
    sp = PB.with_concept(spec_of(), concept_of(ground="land"))
    out, _, report = build(tmp_path, monkeypatch, sp)
    assert out["structure"] == [] and "open_top" not in out and report["world"]["why"] == "placed"
    assert out["fixtures"][0]["position"] == [-1.5, 2.0, -40.0]
    assert SPK.open_ground(out, tmp_path / "lib")[2] == 8.5


# ------------------------------------------------------------------ 4. the redraw's words
def test_the_redraw_calls_a_made_floor_a_floor(tmp_path, monkeypatch):
    import redraw as RD
    seen = {}

    class Stop(Exception):
        pass

    def stop(run, out):                       # (the first thing main() does after reading the set's words)
        seen.update(run)
        raise Stop
    monkeypatch.setattr(RD, "anchor_crops", stop)
    for name, extra, want in (("made", {"open_top": True}, "The floor: planks. Overhead: the sky."),
                              ("land", {}, "The ground: planks. Overhead: the sky.")):
        sd = tmp_path / name
        sd.mkdir()
        (sd / "spec.json").write_text(json.dumps(dict({"words": "In the open air, a terrace.", "world": {"asset": "w"},
                                                       "surfaces_words": "The floor: planks. Overhead: the sky."}, **extra)))
        rp = tmp_path / f"{name}.json"
        rp.write_text(json.dumps({"set": str(sd), "out": str(tmp_path / f"{name}_out"), "scenes": []}))
        monkeypatch.setattr(sys, "argv", ["redraw.py", str(rp)])
        with pytest.raises(Stop):
            RD.main()
        assert seen["outdoors"] is True and seen["room_surfaces"] == want
