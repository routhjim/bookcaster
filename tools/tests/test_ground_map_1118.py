"""A place in the open's ground map (11-18): the set's real ground was one small bare square of the panorama's ground
repeated over 40 m round the people, so a river bank had no river near them, every key's redraw drew another, and
nobody could be stood at a water's edge. The place's own ground is now drawn from straight above (a step of its
world), read by the vision judge (its water, the plan's land places) and laid over the whole disk; its water is faces
of their own under the ground, nobody is stood in it, and the plan's land places are real points. The land's own
things (a boulder, a bush) are pieces, and a bare place makes no empty group piece. Synthetic data only, CPU."""
import json
import sys
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("sets", "places", "scenes", "img3d", "redraw", ""):
    sys.path.insert(0, str(A / sub))
import concept as CO  # noqa: E402
import from_plan as F  # noqa: E402
import generate as G  # noqa: E402
import ground_map as GM  # noqa: E402
import pano_shell as PS  # noqa: E402
import pieces as P  # noqa: E402
import place_build as PB  # noqa: E402
import setfacts  # noqa: E402

STORY_WORDS = ("knight", "smith", "forge", "anvil", "maren", "holmes", "dragon", "squire", "river bank")


def strip_water(y0=-2.0, y1=2.0):
    """A band of water across the whole map from y0 to y1 metres (set frame)."""
    m = np.zeros((GM.cells(), GM.cells()), bool)
    for i in range(GM.cells()):
        y = GM.cell_mid(i, 0)[1]
        if y0 <= y <= y1:
            m[i] = True
    return {"cell_m": GM.WATER_CELL_M, "half_m": GM.HALF_M, "runs": GM.runs_of(m)}


# ------------------------------------------------------------------ the map's frame
def test_the_maps_thousandths_are_metres_on_the_set():
    """The map's middle is the set's origin, its top +y (away from the concept's camera), its width the disk's."""
    assert GM.HALF_M == PS.FLAT_R_M
    assert GM.to_m(500, 500) == [0.0, 0.0]
    assert GM.to_m(0, 0) == [-40.0, 40.0] and GM.to_m(1000, 1000) == [40.0, -40.0]
    assert GM.to_m(750, 250) == [20.0, 20.0]
    u, v = GM.to_uv(*GM.to_m(123, 877))
    assert abs(u - 123) < 1e-6 and abs(v - 877) < 1e-6
    assert GM.cell_of(-39.9, 39.9) == (0, 0) and GM.cell_of(39.9, -39.9) == (GM.cells() - 1, GM.cells() - 1)
    assert GM.point_of("[250, 750]") == [250.0, 750.0] and GM.point_of("none") is None and GM.point_of("[1200, 3]") is None


def test_the_judges_outline_is_cells_of_water():
    """An outline (or a box) of thousandths fills the cells inside it; a few cells are no water; the whole map
    taken for water is the judge's mistake; runs read back to the same cells."""
    w = GM.water_of("[0, 450], [1000, 450], [1000, 550], [0, 550]")
    assert w and GM.in_water(w, 0.0, 0.0) and GM.in_water(w, -30.0, 3.5) and not GM.in_water(w, 0.0, 4.5)
    assert np.array_equal(GM.mask_of(w), GM.poly_mask(GM.polys_of("[0, 450], [1000, 450], [1000, 550], [0, 550]"), GM.cells()))
    assert GM.water_of("none") is None
    assert GM.water_of("[500, 500], [502, 500], [502, 502]") is None                       # a puddle
    assert GM.water_of("[0, 0], [1000, 0], [1000, 1000], [0, 1000]") is None               # the whole land
    assert GM.polys_of("[100, 100, 300, 200]") == [[[100, 100], [300, 100], [300, 200], [100, 200]]]


def test_the_outline_is_corrected_by_the_maps_colours():
    """The drawn water a little beyond the judge's outline is water too; the bank beside it is not."""
    n = 400
    rgb = np.full((n, n, 3), (120, 100, 60), np.uint8)              # earth
    rgb[180:240] = (40, 80, 160)                                     # the drawn water, rows 180..240 of 400
    judged = "[0, 470], [1000, 470], [1000, 560], [0, 560]"         # (the judge's outline: a little short of it)
    w = GM.water_of(judged, rgb)
    y_mid = lambda r: (0.5 - (r + 0.5) / n) * 2 * GM.HALF_M        # noqa: E731
    assert GM.in_water(w, 0.0, y_mid(185)) and GM.in_water(w, 0.0, y_mid(235))
    assert not GM.in_water(w, 0.0, y_mid(170)) and not GM.in_water(w, 0.0, y_mid(250))


def test_the_judges_reading_puts_each_land_place_on_dry_ground_or_in_the_water_as_it_says():
    """A water's edge the judge put a little out in the water is moved to the bank by it; a ford it says is in the
    water stays in it; a place the map does not show is left to the open ground."""
    places = {"at the water's edge": "the edge of the water", "in the ford": "a shallow crossing",
              "on the far hill": "a hill"}

    def ask(items):
        qs = {it["id"]: it for it in items}
        assert "outline" in qs["water"]["ask"] and len(items) == 4
        return {"water": {"outline": "[0, 475], [1000, 475], [1000, 525], [0, 525]"},
                "l:0": {"at": "[500, 490]", "wet": "No."}, "l:1": {"at": "[300, 500]", "wet": "Yes"},
                "l:2": {"at": "none", "wet": "no"}}
    got = GM.read_map("map.png", places, ask)
    w = got["water"]
    assert "_rows" not in w                                           # (its index is not written)
    edge = got["places"]["at the water's edge"]
    assert not edge["wet"] and not GM.in_water(w, *edge["at"]) and 2.0 <= edge["at"][1] <= 2.6
    ford = got["places"]["in the ford"]
    assert ford["wet"] and GM.in_water(w, *ford["at"]) and ford["at"][0] == -16.0
    assert "on the far hill" not in got["places"]
    json.dumps(got)                                                   # (written as map.json)


# ------------------------------------------------------------------ the staging keeps people out of the water
def test_nobody_is_stood_in_the_water_unless_their_place_is_in_it(tmp_path):
    w = strip_water(-3.0, 3.0)
    things = F.water_things({"water": w})
    assert things and things[0]["kind"] == "water" and F.thing_for("at the water", things, {}) == (None, None)
    room = [30, 30, 6]
    p = F.free_mark([0.0, 0.0], [], things, room, (0.0, 1.6))       # sent to the middle of the water: the bank
    assert not GM.in_water(w, *p) and abs(p[1]) <= 4.5
    assert all(not GM.in_water(w, *m) for m in F.free_marks([0.0, 2.0], [], things, room, (0.6, 3.0)))
    assert not F.dry([0.0, 0.0], things) and F.dry([0.0, 5.0], things) and F.dry([0.0, 0.0], [])
    # a place in the water (the judge's): the water is no bar there
    q = F.free_mark([0.0, 0.0], [], [t for t in things if t["kind"] != "water"], room, (0.0, 1.6))
    assert GM.in_water(w, *q)
    # a set with water gives it; one without gives none
    assert F.water_things({}) == [] and F.water_things({"water": {"runs": []}}) == []
    # a land mark carries whether it is in the water to the staging
    sd = tmp_path / "s1"
    sd.mkdir()
    (sd / "spec.json").write_text(json.dumps({"size_m": [30, 30, 6], "fixtures": [], "props": [], "structure": [],
                                              "marks": [{"name": "land_at1", "position": [-16.0, 0.0], "wet": True}]}))
    assert next(t for t in setfacts.things(sd) if t["name"] == "land_at1")["wet"] is True


# ------------------------------------------------------------------ named land places
def test_a_land_place_the_map_shows_is_a_place_on_the_ground():
    spec = {"named_places": {"at the water's edge": "floor", "in the ford": "floor", "out on the plain": "floor",
                             "by the stone": "p1", "in the middle": "floor"},
            "land_places": {"at the water's edge": "the edge of the water", "in the ford": "a shallow crossing",
                            "out on the plain": "the open plain"}}
    world = {"places_m": {"at the water's edge": {"at": [1.0, 2.4], "wet": False, "words": "the edge of the water"},
                          "in the ford": {"at": [-30.0, 0.0], "wet": True, "words": "a shallow crossing"}}}
    marks, named = PB.land_marks(spec, world, half=10.0)
    assert named == {"at the water's edge": "land_at1", "in the ford": "land_at2"}
    assert marks[0] == {"name": "land_at1", "position": [1.0, 2.4], "toward": None, "look": "the edge of the water",
                        "wet": False}
    assert marks[1]["position"] == [-10.0, 0.0] and marks[1]["wet"] is True            # (kept on the stage)
    assert PB.land_marks(spec, {}) == ([], {})                                          # no map: the open ground
    # the layout's check no longer calls the open floor a missing piece
    import place_layout as L
    probs = L.check_layout({"size_m": [10, 10, 3], "pieces": [], "openings": []},
                           {"named_places": {"in the middle": "floor"}})
    assert not any("floor is not in the room" in p for p in probs)


def test_the_world_is_set_with_its_water(tmp_path):
    t = {"id": "t1", "kind": "terrain", "look": "open land", "map_words": "a plain", "land_places": {}}
    aid = P.generation_request(t)["asset_id"]
    (tmp_path / "prop" / aid).mkdir(parents=True)
    w = strip_water()
    (tmp_path / "prop" / aid / "asset.json").write_text(json.dumps({"world": {
        "stand_m": [1.0, 2.0, 0.0], "water": w, "ground_map": {"half_m": 40.0},
        "water_region": {"name": "water", "uv": PS.WATER_UV}}}))
    out = {"fixtures": [], "words": ""}
    got = PB.world_stage(out, {"pieces": [t]}, tmp_path)
    assert got["world"] == aid and out["water"] == w and out["world"]["ground_map"] is True
    assert out["world"]["water_region"]["name"] == "water"


# ------------------------------------------------------------------ the ground's picture
def test_the_map_is_the_ground_and_fades_into_the_panoramas_at_its_edge():
    n = 200
    m = np.zeros((n, n, 3), np.uint8)
    m[: n // 2] = (0, 0, 200)                                       # the top (+y) half blue, the bottom red
    m[n // 2:] = (200, 0, 0)
    ring = np.tile(np.array([[0.0, 150.0, 0.0]]), (360, 1))           # the panorama's ground: green all round
    b = GM.blend(m, ring, 6.0)
    at = lambda x, y: b[int((GM.HALF_M - y) / (2 * GM.HALF_M) * n), int((x + GM.HALF_M) / (2 * GM.HALF_M) * n)]  # noqa: E731
    assert tuple(at(0, 10)) == (0, 0, 200) and tuple(at(0, -10)) == (200, 0, 0)
    assert tuple(at(39.9, 39.9)) == (0, 150, 0)                       # outside the round edge: the panorama's
    mid = at(0, 37.0)                                                 # half way through the fade
    assert 60 < mid[1] < 100 and 60 < mid[2] < 140
    # the ring is the panorama's ground where the disk's edge is seen in it
    H, W = 256, 512
    dz = np.cos(np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    d = np.where(dz < -1e-3, 1.6 / np.maximum(-dz, 1e-3), 150.0)
    dirs = PS.pano_dirs(H, W)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[dz < 0] = (90, 90, 90)
    rgb[(dz < 0) & (dirs[..., 1] < -0.7)] = (10, 200, 10)             # the ground ahead (-y) green
    r = GM.edge_ring(rgb, d, dz > 0)
    assert tuple(r[270].astype(int)) == (10, 200, 10) and tuple(r[90].astype(int)) == (90, 90, 90)


def test_with_a_map_the_disk_is_one_picture_and_its_water_lies_under_the_ground():
    eye = [0.5, -1.0, 10.0]
    w = strip_water(-2.0, 2.0)
    v, f, uv, n_wet = PS.map_disk(eye, 7, w)
    z0 = 10.0 - PS.EYE_STAND_M
    assert f.min() == 7 and n_wet > 0
    wf, gf = f[len(f) - n_wet:] - 7, f[: len(f) - n_wet] - 7
    assert np.allclose(v[wf.ravel(), 2], z0 - PS.WATER_UNDER_M) and np.allclose(v[gf.ravel(), 2], z0)
    rel = v[wf.ravel(), :2] - np.array(eye[:2])
    assert np.abs(rel[:, 1]).max() <= 2.0 + 1e-6                       # only the band's cells
    u0, u1, v0, v1 = PS.WATER_UV
    assert (uv[wf.ravel(), 0] >= u0).all() and (uv[gf.ravel(), 0] < 0.5).all() and uv[:, 1].max() <= v1
    nrm = np.cross(v[f[:, 1] - 7] - v[f[:, 0] - 7], v[f[:, 2] - 7] - v[f[:, 0] - 7])
    assert (nrm[:, 2] > 0).all()                                      # faces look up
    r = np.hypot(*(v[:, :2] - np.array(eye[:2])).T)
    assert r.max() <= PS.FLAT_R_M + 1.0 and r.max() >= PS.FLAT_R_M - 1.0
    # one picture over the disk: a point's uv is its place on the map (the map's top, +y, at v 0.5)
    k = int(np.argmax(v[:, 1]))
    assert uv[k, 1] > 0.49
    assert len(f) < 120000                                            # (under the asset chain's budget with the shell)
    # no water: every cell is ground
    _, f2, _, n2 = PS.map_disk(eye, 0, None)
    assert n2 == 0 and len(f2) == len(f)


def test_without_a_map_the_ground_is_the_panoramas_as_before():
    gv, gf, guv = PS.ground_disk([0.0, 0.0, 10.0], 0)
    assert len(gf) and guv[:, 1].max() <= 0.5                          # (the tile path is kept)


# ------------------------------------------------------------------ the world's steps
def test_the_world_draws_and_reads_its_ground_map_before_it_is_shaped():
    t = {"id": "t1", "kind": "terrain", "look": "open land, far hills", "map_words": "a sloping bank by water",
         "land_places": {"at the water's edge": "the edge of the water"}, "map_source": "/c/concept.png",
         "map_style": "oil painting"}
    r = P.generation_request(t)
    steps = [s["step"] for s in r["plan"]["steps"]]
    assert steps == ["draw", "map", "map_read", "spec", "shape"]
    by = {s["step"]: s for s in r["plan"]["steps"]}
    mp = " ".join(map(str, by["map"]["argv"]))
    assert "from straight above" in mp and "the edge of the water" in mp and "--sources /c/concept.png" in mp
    assert f"--size {GM.MAP_PX} {GM.MAP_PX}" in mp and by["map_read"]["land_places"] == t["land_places"]
    spec = by["spec"]["spec"]
    assert spec["map_id"] == by["map"]["ref_id"] == by["map_read"]["ref_id"]
    # the panorama keeps the id it had without a map (it is drawn once); the world is another asset per map words
    old = P.generation_request({k: v for k, v in t.items() if not k.startswith("map") and k != "land_places"})
    assert by["draw"]["ref_id"] == spec["panorama_id"] == old["plan"]["draw_id"]
    assert r["asset_id"] != old["asset_id"]
    r2 = P.generation_request(dict(t, map_words="a dry plain"))
    assert r2["asset_id"] != r["asset_id"] and r2["plan"]["steps"][0]["ref_id"] == by["draw"]["ref_id"]
    assert [s["step"] for s in old["plan"]["steps"]] == ["draw", "spec", "shape"]
    # a failed try takes its panorama away with it (the next seed draws a new one); the map is kept
    assert G.draw_refs(r["plan"]) == (spec["panorama_id"],) and G.draw_refs(old["plan"]) == ()


def test_the_maps_words_name_no_story_and_no_standing_thing():
    d = GM.map_desc("open land", {"a": "a path"}, "").lower()
    for w in STORY_WORDS:
        assert w not in d
    assert "no buildings" in d and "no people" in d and "no shadows" in d and "a path" in d
    for q in (GM.WATER_ASK, GM.PLACE_AT_ASK, GM.PLACE_WET_ASK, CO.SEEN_OPEN_ASK, CO.MADE_ASK):
        assert not any(w in q.lower() for w in STORY_WORDS)


def test_the_ground_map_is_drawn_then_read_and_neither_stops_the_world(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "REFS", tmp_path)
    t = {"id": "t1", "kind": "terrain", "look": "land", "map_words": "a plain", "land_places": {"x": "a path"}}
    steps = {s["step"]: s for s in P.generation_request(t)["plan"]["steps"]}
    mid = steps["map"]["ref_id"]
    ran = []

    class R:
        returncode = 0

    def run(argv):
        ran.append(argv)
        if "ref_draw.py" in " ".join(map(str, argv)):
            (tmp_path / mid).mkdir(parents=True, exist_ok=True)
            (tmp_path / mid / "picture.png").write_bytes(b"x")
        else:
            assert "--map" in argv and json.loads((tmp_path / mid / "land_places.json").read_text())["land_places"] == {"x": "a path"}
            Path(argv[-1]).write_text("{}")
        return R()
    assert G.ground_map(steps, 11, run) == {"map": True, "read": True} and len(ran) == 2
    assert "11" in ran[0]
    assert G.ground_map(steps, 11, run) == {"map": True, "read": True} and len(ran) == 2   # (made once)
    monkeypatch.setattr(G, "REFS", tmp_path / "none")
    assert G.ground_map(steps, 11, lambda a: R()) == {"map": False, "read": False}


def test_the_shape_step_is_given_the_map_that_was_made(tmp_path, monkeypatch):
    import shell_one as SO
    monkeypatch.setattr(SO, "REFS", tmp_path)
    assert SO.map_args({}) == [] and SO.map_args({"map_id": "m1"}) == []
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "picture.png").write_bytes(b"x")
    assert SO.map_args({"map_id": "m1"}) == ["--map", str(tmp_path / "m1" / "picture.png")]
    (tmp_path / "m1" / "map.json").write_text("{}")
    assert SO.map_args({"map_id": "m1"})[-2:] == ["--map-facts", str(tmp_path / "m1" / "map.json")]
    sp = SO.world_spec_for({"id": "w1"}, {"size_m": [1, 1, 1], "eye_m": [0, 0, 1.6], "stand_m": [0, 0, 0],
                                          "flat_r_m": 40.0, "water": {"runs": [[0, 0, 1]]}, "places_m": {"x": {}},
                                          "water_region": {"name": "water"}}, Path("/w.glb"))
    assert sp["water"] == {"runs": [[0, 0, 1]]} and sp["places_m"] == {"x": {}} and sp["water_region"]["name"] == "water"


def test_the_terrain_piece_carries_the_maps_words():
    spec = {"kind": "outdoors", "period": "not stated", "room": "A sloping bank by a cold stream.",
            "_style": "Oil painting: soft light", "pieces": [], "named_places": {"at the edge": "floor"},
            "land_places": {"at the edge": "the edge of the water"}, "_phrases": ["at the edge"]}
    t = [p for p in PB.outdoors({"pieces": []}, spec, {"concept": "/c/c.png", "surfaces": {"floor": "mud and grass"}})["pieces"]
         if p["kind"] == "terrain"][0]
    assert t["map_words"].startswith("A sloping bank by a cold stream") and "mud and grass" in t["map_words"]
    assert t["land_places"] == {"at the edge": "the edge of the water"} and t["map_source"] == "/c/c.png"
    assert t["map_style"] == "Oil painting"


# ------------------------------------------------------------------ the land's own things
def test_a_bare_land_place_makes_its_natural_things_pieces_and_no_group(tmp_path):
    from PIL import Image
    pic = tmp_path / "c.png"
    Image.new("RGB", (200, 100), (90, 80, 70)).save(pic)
    # the judge's list: two sentences about the picture are not things
    lines = ("there are no buildings or structures in this picture.\n"
             "1. boulder; grey stone\n- bush; green leaves\nclump of reeds; green\n"
             "there are no other solid things standing on the ground besides the natural elements already described.")
    got = CO.read(pic, {"pieces": []}, lambda items: {"space": {"space": "outdoors"}, "view": {"view": "whole"},
                                                      "seen_open": {"seen": lines}})
    assert [s["name"] for s in got["seen"]] == ["boulder", "bush", "clump of reeds"]
    assert not CO.thing_name("The image contains no man-made objects") and CO.thing_name("fallen log")
    asked = []

    def ask(items):
        asked.extend(i["id"] for i in items)
        if items[0]["id"] == "g":
            return {"g": {"box": "[0, 0, 1000, 1000]"}}                 # (it would box the land itself)
        if "words" in items[0]["ask"]:
            return {i["id"]: {"words": "A rounded grey boulder."} for i in items}
        return {"s:0": {"box": "[100, 500, 300, 900]", "kind": "landmark", "own": "separate", "made": "Natural."},
                "s:1": {"box": "[400, 500, 600, 900]", "kind": "plant", "own": "separate", "made": "natural"},
                "s:2": {"box": "[700, 400, 900, 900]", "kind": "none", "own": "separate", "made": "natural"}}
    res = dict(got, spec={})
    CO.pieces_from(pic, {"pieces": []}, res, ask, tmp_path / "crops", "t", ["landmark", "plant", "chair"])
    assert "g" not in asked and not any(s.get("group") for s in res["seen"])
    assert [s["kind"] for s in res["seen"]] == ["landmark"] * 3 and all(s["natural"] for s in res["seen"])
    res["depth"] = {"eye": [0.0, -5.0], "things": {"seen:0": {"xy": [-2.0, 4.0], "size_m": [1.0, 1.0, 0.8]},
                                                   "seen:1": {"xy": [2.0, 5.0], "size_m": [1.5, 1.2, 1.2]},
                                                   "seen:2": {"xy": [4.0, 3.0], "size_m": [1.0, 0.5, 1.5]}}}
    spec = {"kind": "outdoors", "period": "not stated", "room": "A bank.", "pieces": [], "named_places": {}}
    sp = PB.with_concept(spec, dict(res, concept=str(pic), surfaces={}))
    by = {p["id"]: p for p in sp["pieces"]}
    assert sorted(by) == ["c1", "c2", "c3", "t1"] and all(by[k]["natural"] for k in ("c1", "c2", "c3"))
    assert by["c1"]["at_xy"][0] < by["c2"]["at_xy"][0] < by["c3"]["at_xy"][0]            # where the picture has them


def test_with_a_building_the_lands_own_things_stand_apart_from_the_group(tmp_path):
    from PIL import Image
    pic = tmp_path / "c.png"
    Image.new("RGB", (200, 100), (90, 80, 70)).save(pic)
    res = {"space": "outdoors", "spec": {}, "seen": [{"name": "hut", "look": "wood"}, {"name": "boulder", "look": "grey"}]}

    def ask(items):
        if items[0]["id"] == "g":
            return {"g": {"box": "[50, 100, 500, 800]", "real": "6 4", "words": "A hut of planks."}}
        if "words" in items[0]["ask"]:
            return {i["id"]: {"words": "A thing."} for i in items}
        return {"s:0": {"box": "[50, 100, 500, 800]", "kind": "landmark", "own": "separate", "made": "made"},
                "s:1": {"box": "[700, 600, 900, 900]", "kind": "landmark", "own": "separate", "made": "natural"}}
    CO.pieces_from(pic, {"pieces": []}, res, ask, tmp_path / "crops", "t", ["landmark"])
    assert res["seen"][-1].get("group")
    res["depth"] = {"eye": [0.0, -5.0], "things": {"seen:2": {"xy": [0.0, 8.0], "size_m": [6.0, 4.0, 4.0]},
                                                   "seen:1": {"xy": [5.0, 5.0], "size_m": [1.0, 1.0, 0.8]}}}
    spec = {"kind": "outdoors", "period": "not stated", "room": "A clearing.", "pieces": [], "named_places": {}}
    by = {p["id"]: p for p in PB.with_concept(spec, dict(res, concept=str(pic), surfaces={}))["pieces"]}
    assert "c3" in by and "c2" in by and by["c2"]["natural"]                 # the group and the boulder beside it
