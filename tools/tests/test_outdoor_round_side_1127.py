"""A place in the open, 11-27: what lies round it, and which way its cameras look.

Round it: every place in the open had its world drawn as bare level land to the horizon, so a floor out on the water
stood in sand. Its picture is asked what lies round where the people stand (water, land or buildings); with water or
a town round it the world is drawn as that, and a made floor in water stands on a ground that is water. "land", and a
concept read before the question was asked, is the place as it was built before: the same words, the same world.

Its cameras: such a place is built from ONE picture of it; seen from the side or from behind, what was made from the
picture's depth is shards and stretched paint. A camera there looks into the picture's side: within an arc of the way
the picture looks, turned to the nearest allowed angle, the arc widened (never no camera) when nothing stands in it.
Rooms, and sets that state no picture direction, are untouched. Made-up data only."""
import json
import math
import sys
from pathlib import Path

import pytest

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("sets", "places", "scenes", "img3d", ""):
    sys.path.insert(0, str(A / sub))
import concept as CO  # noqa: E402
import pano_shell as PS  # noqa: E402
import pieces as P  # noqa: E402
import place_build as PB  # noqa: E402
import shell_one as SO  # noqa: E402
import speakers as S  # noqa: E402
import story_keys as SK  # noqa: E402


# ------------------------------------------------------------------ 1. the question and its answer
def judge(space="outdoors", ground="floor", around=None, calls=None):
    """A stand-in for the vision judge: answers by item id (no answer to "around" when it is None)."""
    def ask(items):
        if calls is not None:
            calls.append([it["id"] for it in items])
        out = {}
        for it in items:
            out[it["id"]] = {
                "space": {"space": space}, "ground": {"ground": ground},
                "around": {"around": around} if around is not None else {},
                "view": {"view": "whole", "cut": "no"},
                "surf": {"walls": "Boards, brown.", "floor": "Planks, pale.", "ceiling": "The sky, grey.",
                         "light": "From the sun, white.", "far": "A low headland."},
                "seen": {"seen": "chair; wood, brown\nlamp post; iron, black"},
                "seen_open": {"seen": "shed; timber, grey"},
                "seats": {"abreast": "1", "single": "one"},
            }.get(it["id"], {"there": "yes", "look": "oak, brown"})
        return out
    return ask


SPEC = {"place": "the Pier", "kind": "outdoors", "relations": [],
        "pieces": [{"id": "p1", "kind": "bench", "look": "a bench", "phrases": []}]}


@pytest.mark.parametrize("ans,want", [
    ("water", "water"), ("Water.", "water"), (" WATER\n", "water"), ("open water", "water"),
    ("buildings", "buildings"), ("Building", "buildings"), ("land", "land"), ("Land.", "land"),
    ("", "land"), (None, "land"), ("the sky", "land"), ("water, land or buildings", "land"), ("water and land", "land")])
def test_the_answer_is_one_of_three_words_and_anything_else_is_land(ans, want):
    assert CO.around_of(ans) == want


def test_it_is_asked_of_a_place_in_the_open_beside_what_its_people_stand_on():
    calls = []
    res = CO.read(Path("/c/concept.png"), SPEC, judge(around="Water.", calls=calls))
    assert calls[1] == ["ground", "around"] and len(calls) == 2
    assert res["ground"] == "built" and res["around"] == "water"
    assert "Answer with exactly one word: water, land or buildings." in CO.AROUND_ASK
    # (the questions before it are word for word as they were: tests/test_made_floor_1126.py pins them)
    assert CO.GROUND_ASK.endswith("Answer with exactly one word: land or floor.")


def test_no_answer_is_land_and_a_room_is_not_asked():
    assert CO.read(Path("/c/concept.png"), SPEC, judge())["around"] == "land"
    for space in ("room", "cockpit"):
        calls = []
        res = CO.read(Path("/c/concept.png"), SPEC, judge(space, around="water", calls=calls))
        assert "around" not in res and len(calls) == 1 and "around" not in calls[0]


def picture(tmp_path) -> Path:
    from PIL import Image
    p = tmp_path / "concept.png"
    Image.new("RGB", (64, 36), (120, 90, 60)).save(p)
    return p


def boxes(calls: list):
    def ask(items):
        calls.extend(items)
        ans = {"box": "[100, 300, 400, 900]", "kind": "chair", "count": "1", "sides": "one", "own": "separate",
               "real": "0.6 2.4", "made": "made", "words": "A thing", "wall": "Boards."}
        return {it["id"]: {k: ans[k] for k in it["ask"]} for it in items}     # (only what it is asked)
    return ask


def test_a_made_floors_things_are_asked_how_tall_they_really_are(tmp_path):
    pic = picture(tmp_path)
    res = CO.read(pic, SPEC, judge(ground="floor"))
    calls = []
    CO.pieces_from(pic, SPEC, res, boxes(calls), tmp_path / "crops", "t", ["chair", "bench", "work_piece", "landmark"])
    things = [it for it in calls if it["id"][:2] in ("b:", "s:")]
    assert things and all("real" in it["ask"] for it in things)
    assert not any("made" in it["ask"] or "own" in it["ask"] for it in things)      # (open land's own questions)
    assert res["spec"]["p1"]["real_m"] == [0.6, 2.4] and res["seen"][0]["real_m"] == [0.6, 2.4]


def test_a_rooms_things_are_not_asked_it(tmp_path):
    pic = picture(tmp_path)
    res = CO.read(pic, dict(SPEC, kind="hall"), judge("room"))
    calls = []
    CO.pieces_from(pic, SPEC, res, boxes(calls), tmp_path / "crops", "t", ["chair", "bench", "work_piece", "landmark"])
    assert not any("real" in it["ask"] for it in calls)
    assert "real_m" not in res["spec"]["p1"] and "real_m" not in res["seen"][0]


# ------------------------------------------------------------------ 2. the world's words
def spec_of(**kw):
    s = {"room": "A long timber pier.", "schema": "place_spec/1", "place": "the Pier", "kind": "outdoors",
         "period": "medieval", "size": "medium", "openings": [], "outside": "", "_style": "Ink and wash: quiet greys.",
         "pieces": [{"id": "p1", "kind": "chair", "look": "a plain oak chair", "phrases": ["by the chair"],
                     "against": "free"}],
         "relations": [], "named_places": {"by the chair": "p1"}, "land_places": {},
         "_director_room": "Outside: the Pier, a long timber pier"}
    s.update(kw)
    return s


def concept_of(**kw):
    c = {"concept": "/c/concept.png", "space": "outdoors", "ground": "built", "seats_abreast": 1,
         "surfaces": {"walls": "boards, brown", "floor": "planks, pale", "ceiling": "the sky, grey"},
         "light": "from the sun, white", "far": "A low headland",
         "surfaces_words": "The walls: boards, brown; the floor: planks, pale; the ceiling: the sky, grey.",
         "bare": {"walls": {"colour": "#5a4030"}, "floor": {"colour": "#b09060"}}, "seen": [],
         "spec": {"p1": {"in_picture": True, "look": "oak, brown", "box": [0.1, 0.4, 0.3, 0.9], "crop": "/c/crops/p1.png"}},
         "depth": {"room_m": [4.0, 12.0, 2.2], "eye": [0.0, -5.0, 1.5], "off_floor": {},
                   "things": {"spec:p1": {"xy": [-1.0, 1.0], "size_m": [0.6, 0.6, 1.0], "against": None,
                                          "facing": [0.0, -1.0]}},
                   "faces": [{"face": "floor", "file": "/c/floor.png", "u": [0, -2, 2], "v": [0, -6, 6]}]}}
    c.update(kw)
    return c


def world_of(sp: dict) -> dict:
    return next(p for p in sp["pieces"] if p["kind"] == "terrain")


def drawn(t: dict) -> str:
    """The words the world's panorama is drawn from (the draw step's --desc)."""
    argv = next(s for s in P.generation_request(t)["plan"]["steps"] if s["step"] == "draw")["argv"]
    return argv[argv.index("--desc") + 1]


LAND_GROUND = ("bare", "level ground", "open ground")


def test_a_made_floor_with_water_round_it_has_a_world_of_open_water():
    t = world_of(PB.with_concept(spec_of(), concept_of(around="water")))
    assert t["around"] == "water" and t["ground"] == "water"
    assert t["look"].startswith("medieval: open water on every side") and "far off: a low headland" in t["look"]
    assert "overhead: the sky, grey" in t["look"] and "painted as a Ink and wash" in t["look"]
    words = drawn(t)
    assert "wide stretch of open water" in words and "only that water" in words
    assert "ground" not in words.lower() and "land" not in words.replace("headland", "").lower()
    # the place itself is not in its world's words: not its name, its floor, its edge or its things, and no picture
    for own in ("pier", "timber", "planks", "boards", "chair", "floor"):
        assert own not in words.lower()
    assert "source" not in t and "--sources" not in json.dumps(P.generation_request(t)["plan"])
    assert "no people" in words                                         # (as every world's words end)


def test_with_buildings_round_it_a_far_roofline():
    t = world_of(PB.with_concept(spec_of(), concept_of(around="buildings")))
    assert t["around"] == "buildings" and "ground" not in t
    words = drawn(t)
    assert "the walls and roofs of a town far off on every side" in words
    assert "the buildings stand far off on every side, small" in words and "open water" not in words
    assert "pier" not in words.lower() and "planks" not in words


def test_water_round_a_place_whose_picture_shows_nothing_far_off_is_still_drawn_from_words_alone():
    t = world_of(PB.with_concept(spec_of(), concept_of(around="water", far="")))
    assert t["look"].startswith("medieval: open water on every side") and "source" not in t
    assert "pier" not in drawn(t).lower()


@pytest.mark.parametrize("ground", ["built", "land"])
def test_land_and_a_concept_read_before_the_question_are_the_place_as_it_was(ground):
    old = concept_of(ground=ground)                                     # (no "around": read before 11-27)
    a, b = PB.with_concept(spec_of(), old), PB.with_concept(spec_of(), dict(old, around="land"))
    assert a == b
    t = world_of(a)
    assert "around" not in t and "ground" not in t
    assert all(w in drawn(t) for w in LAND_GROUND) and drawn(t).startswith(P.WORLD_WORDS[:60])
    # its world is the same asset it was: the id is its look and WORLD_WORDS, neither changed
    import hashlib
    assert hashlib.sha1(P.WORLD_WORDS.encode()).hexdigest()[:8] == "f20e8e63"
    assert P.generation_request(t)["asset_id"] == P.generation_request(world_of(b))["asset_id"]


def test_a_room_is_not_touched_whatever_around_says():
    room = concept_of(space="room")
    a = PB.with_concept(spec_of(kind="hall"), room)
    assert a == PB.with_concept(spec_of(kind="hall"), dict(room, around="water"))
    assert "_picture_dir" not in a and not any(p["kind"] == "terrain" for p in a["pieces"])
    assert PB.round_of(dict(room, around="water")) == "land"


def test_open_land_with_water_round_it_keeps_its_own_ground_under_the_people():
    con = concept_of(ground="land", around="water")
    t = world_of(PB.with_concept(spec_of(), con))
    assert t["around"] == "water" and "ground" not in t                 # (the people stand on the land's own ground)
    assert "map_words" in t and "planks" in t["map_words"]              # (its ground map, as before)
    assert "the ground:" not in t["look"] and "ground" not in drawn(t).lower()


def test_a_close_view_of_a_place_in_water_is_not_drawn_anew_with_land_round_it():
    w = SK.OPEN_ROUND_WORDS
    assert "land" not in w["water"] and "ground" not in w["water"] and "open water around it" in w["water"]
    assert "the buildings of the town around it" in w["buildings"] and "the land around it" not in w["buildings"]
    assert "the land around it and the sky above" in SK.OPEN_WORDS      # (no "around", or land: as before)
    assert all(k in v for v in w.values() for k in ("{room}", "{inside}", "{style}"))


# ------------------------------------------------------------------ 3. the ground the engine is given
def test_a_made_floor_in_water_asks_for_a_world_whose_ground_is_water():
    t = world_of(PB.with_concept(spec_of(), concept_of(around="water")))
    req = P.generation_request(t)
    spec = next(s for s in req["plan"]["steps"] if s["step"] == "spec")["spec"]
    assert spec["ground"] == "water" and spec["made_as"] == "world"
    draw = next(s for s in req["plan"]["steps"] if s["step"] == "draw")
    assert spec["panorama_id"] == draw["ref_id"] and spec["id"] != draw["ref_id"]
    assert not any(s["step"] == "map" for s in req["plan"]["steps"])    # (no map: nobody is sent out onto it)
    # ... and the land's world of the same place asks for none
    spec0 = next(s for s in P.generation_request(world_of(PB.with_concept(spec_of(), concept_of())))["plan"]["steps"]
                 if s["step"] == "spec")["spec"]
    assert "ground" not in spec0


def test_the_whole_flat_ground_of_that_world_is_the_waters_faces():
    gv, gf, guv = PS.ground_disk([0.0, 0.0, 10.0], 0, water=True)
    u0, u1, v0, v1 = PS.WATER_UV
    mid = guv[gf].mean(1)                                               # (a face is the water's by its uv's middle)
    assert ((mid[:, 0] >= u0) & (mid[:, 0] <= u1) & (mid[:, 1] >= v0) & (mid[:, 1] <= v1)).all()
    assert abs(gv[:, 2].max() - (10.0 - PS.EYE_STAND_M - PS.WATER_UNDER_M)) < 1e-9
    r = float(max(math.hypot(x, y) for x, y, _ in gv))
    assert PS.FLAT_R_M <= r <= PS.FLAT_R_M + 2 * 2 * PS.TILE_M          # (out to the flat ground's edge, all round)
    assert min(math.hypot(*gv[f].mean(0)[:2]) for f in gf) < 2 * PS.TILE_M      # (and under the place's middle too)
    # the land's ground, as before: its middle shows the ground's own picture, which is no water
    lv, lf, luv = PS.ground_disk([0.0, 0.0, 10.0], 0)
    lm = luv[lf].mean(1)
    assert (lm[:, 0] < u0).any() and lv[:, 2].max() == 10.0 - PS.EYE_STAND_M


def test_the_asset_and_the_set_say_the_ground_is_water(tmp_path, monkeypatch):
    facts = {"size_m": [300, 300, 160], "eye_m": [0, 0, 41.6], "stand_m": [0, 0, 40.0], "flat_r_m": 40.0,
             "ground": "water", "water": None,
             "water_region": {"name": "water", "material": "water", "uv": PS.WATER_UV, "under_m": 0.04, "faces": 280}}
    a = SO.world_spec_for({"id": "w", "class": "prop"}, facts, tmp_path / "w.glb")
    assert a["ground"] == "water" and a["water_region"]["material"] == "water" and a["water"] is None
    assert "ground" not in SO.world_spec_for({"id": "w", "class": "prop"}, dict(facts, ground=None), tmp_path / "w.glb")
    out, _, _ = build(tmp_path, monkeypatch, PB.with_concept(spec_of(), concept_of(around="water")),
                      dict(WORLD, ground="water", water_region=facts["water_region"]))
    assert out["world"]["ground"] == "water" and out["world"]["water_region"]["faces"] == 280
    assert "water" not in out                                           # (no map of water to keep people out of)
    assert out["open_top"] and [s["kind"] for s in out["structure"]].count("floor") == 1    # (the floor stands in it)


# ------------------------------------------------------------------ 4. the set says which way its picture looks
WORLD = {"stand_m": [0.0, 0.0, 40.0], "flat_r_m": 40.0}


def build(tmp_path, monkeypatch, spec, world=None):
    lib = tmp_path / "lib"
    (lib / "prop").mkdir(parents=True, exist_ok=True)
    t = next((p for p in spec["pieces"] if p["kind"] == "terrain"), None)
    if t:
        d = lib / "prop" / P.generation_request(t)["asset_id"]
        d.mkdir(parents=True)
        (d / "asset.json").write_text(json.dumps({"id": d.name, "size_m": [300, 300, 160], "world": world or WORLD}))

    def records_for(sp, library):
        rec = {p["id"]: {"asset": f"{p['kind']}_x", "size_m": [0.6, 0.6, 0.95]}
               for p in sp["pieces"] if p["kind"] != "terrain"}
        return rec, P.resolve(sp["pieces"], [])
    monkeypatch.setattr(PB, "records_for", records_for)
    return PB.build(json.loads(json.dumps(spec)), "set_x", lib)


def test_a_made_floor_and_open_land_measured_from_their_picture_state_its_direction(tmp_path, monkeypatch):
    for ground in ("built", "land"):
        out, _, _ = build(tmp_path / ground, monkeypatch, PB.with_concept(spec_of(), concept_of(ground=ground)))
        assert out["picture_dir"] == [0.0, 1.0] and out["all_round"] is False
        assert S.picture_side(out) == {"dir": [0.0, 1.0]}
        assert S.picture_side(dict(out, all_round=True)) == {}          # (a true panorama all round lifts the rule)


def test_a_place_no_picture_measured_and_a_room_state_none(tmp_path, monkeypatch):
    bare = {k: v for k, v in concept_of(ground="land").items() if k != "depth"}
    out, _, _ = build(tmp_path / "a", monkeypatch, PB.with_concept(spec_of(), bare))
    assert "picture_dir" not in out and "all_round" not in out and S.picture_side(out) == {}
    room, _, _ = build(tmp_path / "b", monkeypatch, PB.with_concept(spec_of(kind="hall"), concept_of(space="room")))
    assert "picture_dir" not in room and S.picture_side(room) == {}
    assert S.picture_side({"size_m": [4, 5, 3]}) == {}                  # (a set built before 11-27)


# ------------------------------------------------------------------ 5. the cameras keep to the picture's side
def person(x, y, fwd=(0.0, -1.0)):
    return {"eye_xy": [x, y], "eye_z": 1.6, "head_fwd_xy": list(fwd), "facing": list(fwd), "pelvis_xy": [x, y],
            "chest_xy": [x, y], "head_top_z": 1.75, "chin_z": 1.5, "waist_z": 1.0, "feet_z": 0.0}


@pytest.fixture
def side():
    """A made floor built from its picture (the picture looks along +y), nothing standing on it."""
    S.SOLIDS[:] = []
    S.OPEN_GROUND[:] = []
    S.PICTURE_SIDE.clear()
    S.PICTURE_SIDE.update(S.picture_side({"picture_dir": [0.0, 1.0], "all_round": False}))
    yield
    S.PICTURE_SIDE.clear()


ROOM = [30.0, 30.0, 6.0]


def cam(px, py, tx=0.0, ty=0.0):
    return {"position": [px, py, 1.6], "target": [tx, ty, 1.4], "focal_mm": 35}


def off(q) -> float:
    """A camera's turn from +y in plan, whatever side is held to."""
    return S.plan_angle([0.0, 1.0], q["position"], q["target"])


def test_a_camera_looking_into_the_picture_is_left_alone(side):
    assert S.SIDE_ARC_DEG == 75.0 and S.side_arcs() == [75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]
    for c in (cam(0, -5), cam(3, -4), cam(-4.8, -1.3)):                 # (straight in, 37 deg, 75 deg)
        assert S.side_off(c) <= 75.0 and S.side_turn(c) is c and S.side_turned(c, ROOM) is c


def test_a_camera_looking_back_is_swung_to_the_nearest_allowed_angle(side):
    back = cam(1.0, 5.0)                                                # (looks back at the picture's own eye)
    assert S.side_off(back) > 165.0
    q = S.side_turned(back, ROOM)
    assert abs(S.side_off(q) - 75.0) < 0.1 and q["position"][0] > 0     # (side-on, on the side it stood nearer)
    assert abs(math.dist(q["position"][:2], [0, 0]) - math.dist(back["position"][:2], [0, 0])) < 0.01
    assert q["position"][2] == 1.6 and q["target"] == back["target"] and q["focal_mm"] == 35
    assert S.side_turned(cam(-1.0, 5.0), ROOM)["position"][0] < 0


def test_where_no_standpoint_is_left_at_the_arc_it_is_widened_step_by_step(side, capsys):
    S.SOLIDS[:] = [[5.0, -1.0, 1.5, 1.5, 8.0, 0.0, -1.0]]              # (a tall thing where the 75 degree camera stands)
    try:
        q = S.side_turned(cam(1.0, 5.0), ROOM)
    finally:
        S.SOLIDS[:] = []
    assert 75.0 < S.side_off(q) <= 105.1 and S.in_room(q["position"], ROOM)
    assert "picture side" in capsys.readouterr().out                    # (and it is said)


def test_the_search_keeps_the_cameras_inside_the_arc_and_widens_only_when_none_is(side, capsys):
    inside, edge, behind = cam(0, -5), cam(5, -0.5), cam(0, 5)
    assert S.within_side([behind, inside, edge]) == [inside]
    assert capsys.readouterr().out == ""
    assert S.within_side([behind, edge]) == [edge]                      # (84 deg: the first wider arc that holds one)
    assert "widened to 90" in capsys.readouterr().out
    assert S.within_side([behind]) == [behind] and S.within_side([]) == []     # (never no camera)
    # (the two-shot search's rows: its camera is the row's second part)
    assert S.within_side([(1.0, behind, "two"), (2.0, inside, "ots")], lambda t: t[1]) == [(2.0, inside, "ots")]


def test_a_failed_speakers_own_cameras_look_into_the_picture(side):
    # Orla faces the picture's eye (-y): every camera before her face looks into the picture
    subj = {"orla": {"measured": person(0.0, 2.0, (0.0, -1.0))}}
    c = {"name": "s1", "kind": "medium", "focal_mm": 50}
    got = S.candidates(c, {"camera": cam(0, -3)}, "orla", subj, [1152, 640], ROOM, None, None, [])
    assert got and all(S.side_off(q) <= 75.0 for q in got)
    # Tam faces into the picture (+y): no camera before his face looks into it; the arc is widened, not the shot lost
    subj = {"tam": {"measured": person(0.0, 0.0, (0.0, 1.0))}}
    got = S.candidates(c, {"camera": cam(0, 3)}, "tam", subj, [1152, 640], ROOM, None, None, [])
    assert got and min(S.side_off(q) for q in got) > 75.0
    best = min(S.side_off(q) for q in got)
    S.PICTURE_SIDE.clear()
    every = S.candidates(c, {"camera": cam(0, 3)}, "tam", subj, [1152, 640], ROOM, None, None, [])
    assert len(got) < len(every) and abs(best - min(off(q) for q in every)) < 15.0


def test_the_last_resort_wide_stands_on_the_pictures_side(side):
    # Tam and Orla face each other across the floor, Tam (the first) looking into the picture
    ms = [person(0.0, 0.0, (0.0, 1.0)), person(0.0, 2.5, (0.0, -1.0))]
    q = S.fallback_camera(ms, ROOM)
    assert q and S.side_off(q) <= 75.0
    S.PICTURE_SIDE.clear()
    free = S.fallback_camera(ms, ROOM)                                  # (in a room: square across the picture's view)
    assert off(free) > 75.0 and q != free


def test_a_wide_with_no_standpoint_inside_the_arc_is_still_found(side, capsys):
    # the floor ends just behind the people on the picture's eye's side: the only standpoints look back
    S.OPEN_GROUND[:] = [0.0, 16.5, 12.0]
    try:
        q = S.fallback_camera([person(0.0, 2.5, (0.0, 1.0))], ROOM)
    finally:
        S.OPEN_GROUND[:] = []
    assert q and S.side_off(q) > 90.0 and "widened" in capsys.readouterr().out


def test_the_first_cameras_are_held_to_the_side_before_anything_is_rendered(side):
    people = {"tam": {"position": [0.0, 0.0, 0.0], "facing": [0.0, 1.0]},
              "orla": {"position": [0.0, 2.5, 0.0], "facing": [0.0, -1.0]}}
    # a placed camera with nobody in it (an insert, a wide) is swung round its aim
    wide = dict(cam(0.0, 6.0), name="s1", kind="wide", subject=None)
    assert "swung" in S.side_first(wide, people, ROOM) and abs(S.side_off(wide) - 75.0) < 0.1
    # a computed one stands the way its subject faces: before Tam's face it would look back; turned round him
    tam = {"name": "s2", "kind": "medium", "subject": "tam"}
    assert "turned round tam" in S.side_first(tam, people, ROOM) and abs(abs(tam["az"]) - 105.0) < 0.1
    # ... before Orla's it looks into the picture already; an over-the-shoulder is left to the search
    orla = {"name": "s3", "kind": "medium", "subject": "orla", "az": 20.0}
    ots = dict(cam(0.0, 6.0), name="s4", kind="medium", subject="tam", ots="orla")
    assert S.side_first(orla, people, ROOM) is None and orla["az"] == 20.0
    assert S.side_first(ots, people, ROOM) is None and ots["position"] == [0.0, 6.0, 1.6]


def test_a_room_and_a_true_panorama_are_untouched():
    S.SOLIDS[:] = []
    S.OPEN_GROUND[:] = []
    for st in ({"size_m": [6, 6, 3]}, {"picture_dir": [0.0, 1.0], "all_round": True}):
        S.PICTURE_SIDE.clear()
        S.PICTURE_SIDE.update(S.picture_side(st))
        back = cam(0.0, 5.0)
        assert S.PICTURE_SIDE == {} and S.side_off(back) == 0.0 and S.side_arcs() == [180.0]
        assert S.side_turned(back, ROOM) is back and S.within_side([back]) == [back]
        wide = dict(back, name="s1", kind="wide")
        assert S.side_first(wide, {}, ROOM) is None and wide["position"] == back["position"]
        ms = [person(0.0, 0.0, (0.0, 1.0)), person(0.0, 2.5, (0.0, -1.0))]
        # (the wide it had before the rule: from the side of the two, looking across the picture's view)
        assert S.fallback_camera(ms, ROOM) == {"position": [-1.917, 1.25, 1.6], "target": [0.0, 1.25, 1.3],
                                                "focal_mm": 24}
