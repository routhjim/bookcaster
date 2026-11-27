"""A made floor in the open keeps its concept picture's own structure (11-27): a picture stage.

A made floor (a deck, a pier, a rooftop, a terrace) was built as a bare floor inside four plain low walls, with a
world drawn from words round it: nothing of what its picture shows standing on and round the floor was in the set.
Measured with its picture's own land (img3d/concept_depth.py stage_land), it is now built as a one-sided picture
stage: the real, level floor of the measured size with its loose things as pieces, as before; round it the picture
itself at its own depth in place of the low walls; beyond that the picture carried on all the way round, with no
flat ground of the world's own beside the floor (water, where it stands in water). The floor's size still holds the
people and the cameras. A made floor measured before this, rooms and open land are built as they were.
Made-up data only."""
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("sets", "places", "scenes", "img3d", ""):
    sys.path.insert(0, str(A / sub))
import concept_depth as CD  # noqa: E402
import pano_shell as PS  # noqa: E402
import pieces as P  # noqa: E402
import place_build as PB  # noqa: E402
import speakers as SPK  # noqa: E402


# ------------------------------------------------------------------ 1. the picture's land round the floor
F, W_PX, H_PX = 0.7, 320, 200            # normalised focal (of the width), picture size
EYE_H = 1.5
INTR = np.array([[F, 0, 0.5], [0, F * W_PX / H_PX, 0.5], [0, 0, 1]])
KINDS = {"crate": [0.8, 0.8, 0.8], "bench": [1.5, 0.5, 0.5]}
CRATE = (-1.0, 4.0, 0.8, 0.8)            # a loose crate on the floor: x, how far before the eye, width, height
HOUSE = (9.0, 2.5)                       # the Pier's own house across its far end: how far, how high


def pier_points(crate=True):
    """The Pier seen from its picture's eye (camera coords: x right, y down, z forward): a floor 7 m wide and 9 m
    long, a house standing across its far end, a crate on the floor; the water lower all round."""
    v, u = np.mgrid[0:H_PX, 0:W_PX]
    r = np.stack([((u + 0.5) / W_PX - 0.5) / F, ((v + 0.5) / H_PX - 0.5) / INTR[1, 1], np.ones(u.shape)], -1)
    t = np.full(u.shape, np.inf)

    def take(tt, ok):
        nonlocal t
        t = np.minimum(t, np.where(ok & (tt > 0), tt, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = EYE_H / r[..., 1]
        take(tf, (np.abs(r[..., 0] * tf) <= 3.5) & (tf <= HOUSE[0]))
        take((EYE_H + 3.0) / r[..., 1], np.ones(u.shape, bool))
        th = np.full(u.shape, HOUSE[0])
        take(th, (np.abs(r[..., 0] * th) <= 3.5) & (EYE_H - r[..., 1] * th >= 0.0) & (EYE_H - r[..., 1] * th <= HOUSE[1]))
        if crate:
            x, z, w, h = CRATE
            tc = np.full(u.shape, z)
            take(tc, (np.abs(r[..., 0] * tc - x) <= w / 2) & (EYE_H - r[..., 1] * tc >= 0.0) & (EYE_H - r[..., 1] * tc <= h))
    mask = np.isfinite(t)
    return np.where(mask[..., None], r * np.where(mask, t, 0.0)[..., None], 0.0), mask


def box_of(x, z, w, h):
    """The picture box of an upright thing standing on the floor at (x, z), w wide and h high."""
    def proj(p):
        return INTR[0, 0] * p[0] / p[2] + 0.5, INTR[1, 1] * p[1] / p[2] + 0.5
    u0, v1 = proj([x - w / 2, EYE_H, z])
    u1, _ = proj([x + w / 2, EYE_H, z])
    _, v0 = proj([x, EYE_H - h, z])
    return [u0, v0, u1, v1]


def measured(boxes):
    pts, mask = pier_points()
    fr = {}
    res = CD.room_from(pts, mask, INTR, boxes, KINDS, fr, open_top=True)
    img = np.full((H_PX, W_PX, 3), 120, np.uint8)
    return res, CD.stage_land(pts, mask, fr, res["things"], {k: v[0] for k, v in boxes.items()}, img)


def risen(vtx, uv, x, y, r=0.5, over=0.3):
    """The points of the land's front that stand over the floor within r of a place on it."""
    f = vtx[uv[:, 0] <= 0.5]
    return f[(np.hypot(f[:, 0] - x, f[:, 1] - y) < r) & (f[:, 2] > over)]


def test_the_pictures_structure_stands_round_the_floor_and_its_loose_things_are_cut_out():
    pytest.importorskip("scipy")
    crate = {"spec:p1": (box_of(*CRATE), "crate")}
    res, got = measured(crate)
    vtx, faces, uv, painted, has = got
    W, D, _ = res["room_m"]
    eye, t = res["eye"], res["things"]["spec:p1"]
    # the house across the far end is there, in the floor's own frame, as high as the picture has it
    house = risen(vtx, uv, eye[0], eye[1] + HOUSE[0], r=1.5, over=1.0)
    assert len(house) and abs(house[:, 2].max() - HOUSE[1]) < 0.4
    assert abs(house[:, 1].mean() - (eye[1] + HOUSE[0])) < 0.5
    # the crate is a piece of the set: not in the land a second time (left in it when the set builds no such piece)
    assert not len(risen(vtx, uv, t["xy"][0], eye[1] + CRATE[1]))
    _, kept = measured({})
    assert len(risen(kept[0], kept[2], t["xy"][0], eye[1] + CRATE[1]))
    assert has == []
    # the picture's own floor lies just under the real one, never over it
    f = vtx[uv[:, 0] <= 0.5]
    on = f[(np.abs(f[:, 0]) < W / 2 - 0.5) & (f[:, 1] > eye[1] + 4.5) & (f[:, 1] < eye[1] + 7.0)]
    assert len(on) and on[:, 2].max() <= 0.0 and on[:, 2].min() >= -CD.LAND_UNDER_M - 1e-6


def test_what_takes_up_much_of_the_picture_is_the_places_structure_and_stays_in_the_land():
    pytest.importorskip("scipy")
    wide = [0.1, 0.05, 0.9, 0.7]             # (rigging over the whole floor, a canopy: no loose thing)
    assert (wide[2] - wide[0]) * (wide[3] - wide[1]) > CD.STAGE_CUT_BOX
    res, got = measured({"spec:p1": (box_of(*CRATE), "crate"), "seen:0": (wide, "bench")})
    vtx, _, uv, _, has = got
    assert has == ["seen:0"]
    eye = res["eye"]
    assert len(risen(vtx, uv, eye[0], eye[1] + HOUSE[0], r=1.5, over=1.0))      # (the house behind it still stands)


def test_open_lands_own_land_is_made_as_before():
    sig = inspect.signature(CD.land).parameters
    assert sig["level"].default is True and sig["cut_most"].default is None


# ------------------------------------------------------------------ 2. the place spec
def spec_of(**kw):
    s = {"room": "A long timber pier out over the water.", "schema": "place_spec/1", "place": "the Pier",
         "kind": "outdoors", "period": "medieval", "size": "medium", "openings": [], "outside": "",
         "pieces": [{"id": "p1", "kind": "chair", "look": "a plain oak chair", "phrases": ["by the chair"],
                     "against": "free"}],
         "relations": [], "named_places": {"by the chair": "p1", "at the rail": "floor"},
         "land_places": {}, "_director_room": "Outside: a long timber pier out over the water"}
    s.update(kw)
    return s


TOPS = {"cell_m": 1.0, "x0": -4.0, "y0": 2.0,          # (the land stands 2.5 m high across the floor's far end)
        "z": [[0.0] * 8, [0.0] * 8, [2.5] * 8, [2.5] * 8]}


def concept_of(tmp_path, land=True, **kw):
    pic, glb = tmp_path / "concept.png", tmp_path / "concept_land.glb"
    pic.write_bytes(b"a picture")
    glb.write_bytes(b"a form")
    c = {"concept": str(pic), "space": "outdoors", "ground": "built", "seats_abreast": 1,
         "surfaces": {"walls": "railings, brown", "floor": "planks, pale", "ceiling": "the sky, blue"},
         "light": "from the sun, white", "far": "low hills beyond the water",
         "surfaces_words": "The walls: railings, brown; the floor: planks, pale; the ceiling: the sky, blue.",
         "bare": {"walls": {"colour": "#5a4030"}, "floor": {"colour": "#b09060"}},
         "seen": [{"name": "crate", "look": "wood, brown", "kind": "crate", "box": [0.6, 0.6, 0.7, 0.9],
                   "crop": "/c/crops/crate.png", "words": "A wooden crate"},
                  {"name": "rigging", "look": "rope, tan", "kind": "work_piece", "box": [0.1, 0.0, 0.9, 0.7],
                   "crop": "/c/crops/rig.png", "words": "Ropes over the whole floor"}],
         "spec": {"p1": {"in_picture": True, "look": "oak, brown", "box": [0.1, 0.4, 0.3, 0.9], "crop": "/c/crops/p1.png"}},
         "depth": {"room_m": [7.0, 8.0, 2.2], "eye": [0.5, -3.0, 1.5], "hfov_deg": 70.0,
                   "things": {"spec:p1": {"xy": [-1.5, 1.0], "size_m": [0.6, 0.6, 1.0], "against": None, "facing": [0.0, -1.0]},
                              "seen:0": {"xy": [2.0, 1.5], "size_m": [0.8, 0.8, 0.8], "against": None, "facing": [0.0, -1.0]},
                              "seen:1": {"xy": [0.0, 2.5], "size_m": [5.0, 3.0, 6.0], "against": None, "facing": [0.0, -1.0]}},
                   "off_floor": {},
                   "faces": [{"face": "floor", "file": "/c/floor.png", "u": [0, -3, 3], "v": [0, -4, 4]}]}}
    if land:
        c["depth"]["land"] = {"glb": str(glb), "xy": [0.3, 9.0], "size_m": [30.0, 40.0, 9.0], "faces": 5000,
                              "z0": -0.03, "under_m": 0.03, "has": ["seen:1"], "tops": TOPS, "stage": True}
    c.update(kw)
    return c


def test_a_made_floor_with_its_pictures_land_is_a_picture_stage(tmp_path):
    con = concept_of(tmp_path)
    out = PB.with_concept(spec_of(), con)
    assert out["_picture_stage"] is True and out["_open_top"] is True and out["kind"] == "outdoors"
    assert out["_room_m"][:2] == [7.0, 8.0] and out["_picture_dir"] == PB.PICTURE_DIR
    by = {p["id"]: p for p in out["pieces"]}
    # its loose things are pieces of the set, each where the picture has it, as before
    assert by["p1"]["at_xy"] == [-1.5, 1.0] and by["c1"]["at_xy"] == [2.0, 1.5]
    # what the picture's land kept is not built a second time
    assert "c2" not in by
    # the world: the picture carried on all the way round from its own eye, nothing of it near, no ground of its own
    t = by["t1"]
    assert t["source"] == con["concept"] and "around" not in t and t["ground"] == "none"
    assert t["view"] == {"eye_at": [0.5, -3.0], "eye_h": 1.5, "front_deg": PB.STAGE_FRONT_DEG} and PB.STAGE_FRONT_DEG == 180.0
    assert "pier" in t["look"] and "planks, pale" in t["look"]
    assert "its edge: railings, brown" in t["surface_words"] and "low wall" not in t["surface_words"]
    # the picture's land: where it was measured, in the floor's own frame, its floor under the real one
    ld = out["pieces"][-1]
    assert ld["id"] == PB.LAND_ID and ld["scenery"] and ld["at_xy"] == [0.3, 9.0] and ld["under_m"] == 0.03
    assert ld["relief"]["glb"] == con["depth"]["land"]["glb"] and ld["_seen_size_m"] == [30.0, 40.0, 9.0]
    assert [p["id"] for p in out["pieces"] if p["kind"] == "terrain"] == ["t1", PB.LAND_ID]
    assert out["named_places"] == {"by the chair": "p1", "at the rail": "floor"}
    assert [f["face"] for f in out["_concept"]["faces"]] == ["floor"]    # (the real floor shows the picture's floor)


def test_a_thing_the_plan_sends_people_to_is_built_though_the_land_has_it(tmp_path):
    con = concept_of(tmp_path)
    con["spec"]["p1"]["box"] = [0.0, 0.0, 0.1, 0.1]
    con["depth"]["land"]["has"] = ["spec:p1", "seen:1"]
    out = PB.with_concept(spec_of(), con)
    assert "p1" in {p["id"] for p in out["pieces"]}


def test_in_water_the_worlds_ground_is_the_water_as_before(tmp_path):
    out = PB.with_concept(spec_of(), concept_of(tmp_path, around="water"))
    t = next(p for p in out["pieces"] if p["id"] == "t1")
    assert t["ground"] == "water" and "view" in t and PB.ROUND_LOOK["water"] in t["look"]


def test_the_world_and_the_land_are_asked_for_from_the_picture(tmp_path):
    out = PB.with_concept(spec_of(), concept_of(tmp_path))
    t, ld = (next(p for p in out["pieces"] if p["id"] == i) for i in ("t1", PB.LAND_ID))
    req = P.generation_request(t)
    steps = {s["step"]: s for s in req["plan"]["steps"]}
    assert any("pano_from_picture.py" in str(a) for a in steps["draw"]["argv"])      # (not drawn from words)
    assert steps["spec"]["spec"]["ground"] == "none" and steps["spec"]["spec"]["view"]["front_deg"] == 180.0
    assert steps["spec"]["spec"]["made_as"] == "world"
    # (another asset than the same panorama with its flat ground, and than the water's)
    ids = {P.generation_request(dict(t, ground=g))["asset_id"] for g in ("none", "water", "")}
    assert len(ids) == 3
    rl = P.generation_request(ld)
    assert next(s for s in rl["plan"]["steps"] if s["step"] == "spec")["spec"]["made_as"] == "relief"
    assert rl["size_m"] == [30.0, 40.0, 9.0]
    # the picture's land gives the picture's own light, as its world does: another asset than the lit land
    sp = next(s for s in rl["plan"]["steps"] if s["step"] == "spec")["spec"]
    assert sp["own_light"] is True
    lit = P.generation_request(dict(ld, relief={k: v for k, v in ld["relief"].items() if k != "own_light"}))
    assert lit["asset_id"] != rl["asset_id"]
    assert "own_light" not in next(s for s in lit["plan"]["steps"] if s["step"] == "spec")["spec"]
    import importlib.util
    m = importlib.util.spec_from_file_location("relief_one", Path(P.__file__).parents[1] / "img3d" / "relief_one.py")
    R1 = importlib.util.module_from_spec(m)
    m.loader.exec_module(R1)
    assert R1.spec_for(dict(sp, id="x"), Path("x.glb"))["emit_texture"] == R1.RELIEF_EMIT
    assert "emit_texture" not in R1.spec_for({"id": "x", "size_m": [1, 1, 1]}, Path("x.glb"))


def test_a_world_with_no_ground_of_its_own_builds_none():
    gv, gf, guv = PS.no_ground()
    assert gv.shape == (0, 3) and gf.shape == (0, 3) and guv.shape == (0, 2)
    assert len(np.concatenate([np.zeros((4, 3)), gv])) == 4
    assert len(PS.ground_disk([0.0, 0.0, 10.0], 0)[1]) > 0                # (every other world: its ground, as before)


# ------------------------------------------------------------------ 3. the set
WORLD = {"stand_m": [1.5, -2.0, 40.0], "flat_r_m": 40.0, "ground": "none",
         "sun": {"dir": [0.3, -0.9, 0.05], "color": "#ffb070", "level": 0.4, "peak": 0.9}}


def build(tmp_path, monkeypatch, spec):
    lib = tmp_path / "lib"
    (lib / "prop").mkdir(parents=True, exist_ok=True)
    for p in spec["pieces"]:
        if p["kind"] == "terrain":
            d = lib / "prop" / P.generation_request(p)["asset_id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "asset.json").write_text(json.dumps({"id": d.name, "size_m": [300, 300, 160],
                                                      **({} if p.get("scenery") else {"world": WORLD})}))

    def records_for(sp, library):            # (the library's match for each piece: a made-up record)
        rec = {p["id"]: {"asset": f"{p['kind']}_x", "size_m": [0.6, 0.6, 0.95]}
               for p in sp["pieces"] if p["kind"] != "terrain"}
        return rec, P.resolve(sp["pieces"], [])
    monkeypatch.setattr(PB, "records_for", records_for)
    return PB.build(json.loads(json.dumps(spec)), "set_x", lib)


LANTERN = {"id": "p3", "kind": "lantern", "look": "a lantern", "phrases": ["under the lantern"], "against": "free",
           "gives_light": True, "hangs_m": 0.6}


def pier_spec(tmp_path, **kw):
    return PB.with_concept(spec_of(pieces=spec_of()["pieces"] + [dict(LANTERN)]), concept_of(tmp_path, **kw))


@pytest.fixture
def stage(tmp_path, monkeypatch):
    return build(tmp_path, monkeypatch, pier_spec(tmp_path))


def test_the_real_floor_is_kept_and_no_wall_of_the_sets_own_stands_round_it(stage):
    out, named, report = stage
    kinds = [s["kind"] for s in out["structure"]]
    assert kinds.count("floor") == 1 and "wall" not in kinds and "ceiling" not in kinds
    assert out["size_m"][:2] == [7.0, 8.0] and out["size_m"][2] >= PB.OPEN_TOP_STAGE_M
    floor = next(s for s in out["structure"] if s["kind"] == "floor")
    assert floor["extend_m"] < 0.3 and out["materials"][floor["material"]]["color"] == "#b09060"
    assert out["open_top"] is True and out["picture_stage"] is True and out["interior"] is False
    assert named["by the chair"] and named["at the rail"] == "floor"


def test_the_pictures_land_and_the_world_stand_round_the_floor(stage):
    out, _, report = stage
    w, ld = out["fixtures"][0], out["fixtures"][1]
    assert w["name"] == "world" and w["world"] and w["position"] == [-1.5, 2.0, round(-40.0 - PB.OPEN_FLOOR_OVER_M, 3)]
    # the land where it was measured, its own floor a little under the real floor's top: one floor is seen
    assert ld["name"] == PB.LAND_ID and ld["world"] and ld["outside"] and ld["position"] == [0.3, 9.0, -0.03]
    assert -0.4 < ld["position"][2] < 0.0                                # (inside the real floor's slab)
    assert report["world"]["why"] == "placed" and report["world"]["land"] == [ld["asset"]]
    assert out["world"]["ground"] == "none"
    # no piece of the floor is the land or the world
    assert not any(f["name"] in ("t1",) for f in out["fixtures"][2:] + out["props"])


def test_the_floors_size_still_holds_the_people_and_the_cameras(stage, tmp_path):
    out, _, _ = stage
    W, D, _ = out["size_m"]
    assert SPK.open_ground(out, tmp_path / "lib") == []                 # (no camera out on the world's ground)
    assert out["picture_dir"] == PB.PICTURE_DIR and out["all_round"] is False
    assert SPK.picture_side(out) == {"dir": [0.0, 1.0]}
    for c in out["cameras"]:
        assert abs(c["position"][0]) < W / 2 and abs(c["position"][1]) < D / 2
        # its own wide pictures look into the picture's side, every one
        assert SPK.plan_angle(PB.PICTURE_DIR, c["position"], c["target"]) < SPK.SIDE_ARC_DEG
    assert len({tuple(c["position"]) for c in out["cameras"]}) == 4
    for f in out["fixtures"][2:] + out["props"]:                        # (every piece stands on the floor)
        assert abs(f["position"][0]) <= W / 2 + 0.3 and abs(f["position"][1]) <= D / 2 + 0.3
    # where the picture's land rises nobody is stood and no camera stands
    hl = out["high_land"]
    assert hl["runs"] and hl["tall_runs"]
    import ground_map as GM
    assert GM.in_water({"cell_m": hl["cell_m"], "half_m": hl["half_m"], "runs": hl["runs"]}, 0.0, 4.5)
    assert not GM.in_water({"cell_m": hl["cell_m"], "half_m": hl["half_m"], "runs": hl["runs"]}, 0.0, 0.0)


def test_what_hangs_there_hangs_on_a_post_and_still_gives_its_light(stage):
    out, _, _ = stage
    lan = next(f for f in out["fixtures"] + out["props"] if f["name"].startswith("lantern"))
    post = next(s for s in out["structure"] if s["kind"] == "box" and s["name"].startswith("post_"))
    floor = next(s for s in out["structure"] if s["kind"] == "floor")
    assert lan["hang"] and post["material"] == floor["material"] and post["size_m"][2] > 0.6
    assert abs(post["base"][0]) <= out["size_m"][0] / 2 + floor["extend_m"]
    assert abs(post["base"][1]) <= out["size_m"][1] / 2 + floor["extend_m"]
    types = [l["type"] for l in out["lights"]]
    assert types.count("sun") == 1 and types.count("sky") == 1 and "point" in types


# ------------------------------------------------------------------ 4. everything else is as it was
def test_a_made_floor_measured_before_is_the_roofless_room_it_was(tmp_path, monkeypatch):
    old = concept_of(tmp_path, land=False)
    sp = PB.with_concept(spec_of(), old)
    assert "_picture_stage" not in sp and not any(p.get("scenery") for p in sp["pieces"])
    t = next(p for p in sp["pieces"] if p["kind"] == "terrain")
    assert "view" not in t and "source" not in t and "ground" not in t and "low hills" in t["look"]
    assert "the low wall round its edge: railings, brown" in t["surface_words"]
    assert "c2" in {p["id"] for p in sp["pieces"]}
    # (a land left by an older measure of the same picture as open land is no stage's)
    stale = concept_of(tmp_path)
    del stale["depth"]["land"]["stage"]
    assert PB.with_concept(spec_of(), stale) == sp and PB.picture_stage(stale) == {}
    out, _, _ = build(tmp_path, monkeypatch, sp)
    walls = [s for s in out["structure"] if s["kind"] == "wall"]
    assert len(walls) == 4 and all(w["height_m"] == PB.OPEN_EDGE_M for w in walls)
    assert next(s for s in out["structure"] if s["kind"] == "floor")["extend_m"] == walls[0]["thickness_m"]
    assert "picture_stage" not in out and "high_land" not in out and len([f for f in out["fixtures"] if f.get("world")]) == 1
    assert [c["position"][1] > 0 for c in out["cameras"]] == [False, False, True, True]


def test_a_room_is_built_as_before(tmp_path, monkeypatch):
    with_land = concept_of(tmp_path, space="room")
    a = PB.with_concept(spec_of(kind="hall"), with_land)
    b = PB.with_concept(spec_of(kind="hall"), concept_of(tmp_path, land=False, space="room"))
    assert a == b and "_picture_stage" not in a and "_open_top" not in a
    assert not any(p["kind"] == "terrain" for p in a["pieces"])
    out, _, _ = build(tmp_path, monkeypatch, a)
    kinds = [s["kind"] for s in out["structure"]]
    assert kinds.count("wall") == 4 and "ceiling" in kinds and "picture_stage" not in out
    assert next(s for s in out["structure"] if s["kind"] == "floor") == PB.kit_for("medieval")["floor"]
    assert [c["position"][1] > 0 for c in out["cameras"]] == [False, False, True, True]


def test_open_land_is_built_as_before(tmp_path, monkeypatch):
    con = concept_of(tmp_path, ground="land")
    a = PB.with_concept(spec_of(), con)
    plain = json.loads(json.dumps(con))
    del plain["depth"]["land"]["stage"]
    assert a == PB.with_concept(spec_of(), plain)                        # (the stage's mark changes nothing there)
    assert "_picture_stage" not in a and "_open_top" not in a
    t, ld = (next(p for p in a["pieces"] if p["id"] == i) for i in ("t1", PB.LAND_ID))
    assert "ground" not in t and t["view"]["front_deg"] < 90.0 and "map_words" in t
    assert ld["at_xy"] != [0.3, 9.0]                                     # (about its own middle, from its eye)
    out, _, report = build(tmp_path, monkeypatch, a)
    assert out["structure"] == [] and "open_top" not in out and "picture_stage" not in out


def test_a_cockpit_is_not_touched(tmp_path):
    a = PB.with_concept(spec_of(kind="cockpit"), concept_of(tmp_path, space="cockpit"))
    assert a["kind"] == "cockpit" and "_picture_stage" not in a and [p["kind"] for p in a["pieces"]] == ["vehicle_cabin"]



def test_a_side_of_the_land_is_one_point_of_the_picture_from_front_to_back():
    """No face of the solid runs from the picture's left half to its right: a side is its edge's point all through."""
    U, V = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 3))
    F = np.stack([U * 4, np.zeros_like(U), (1 - V) * 3], -1).reshape(-1, 3)
    B = F + [0.0, 1.0, 0.0]
    front_f, _ = CD._front(np.ones((2, 3), bool))
    vtx, faces, uv = CD._closed(F, B, front_f, U, V)
    left = uv[faces][..., 0] <= 0.5
    assert not (left.any(1) & ~left.all(1)).any()
    assert len(faces) == 2 * len(front_f) + 2 * 10            # front, back, and two faces to each of its 10 rim edges
    side = faces[2 * len(front_f):]
    assert np.ptp(vtx[side][..., 1], axis=1).max() == pytest.approx(1.0)        # each runs front to back
    # the front's points are still the only ones on the left half (tops_of reads the land's height from them)
    assert (uv[:, 0] <= 0.5).sum() == len(F)


def test_what_hangs_over_seen_ground_shuts_none_of_it():
    """A line of lamps over a floor is overhead; a wall on the floor, and a far house whose foot is hidden, stand."""
    def col(x, y, zs):
        return [[x, y, z] for z in zs]
    pts = (col(0.5, 0.5, [0.0, 3.0, 3.2])                     # ground seen, lamps over it
           + col(1.5, 0.5, [0.0, 0.5, 1.5, 3.0])              # a wall from the ground up
           + col(2.5, 0.5, [3.0, 4.0])                        # a far house's top, its foot behind something
           + col(3.5, 0.5, [0.0, 2.0, 3.0]))                  # a low bough: not clear
    vtx = np.array(pts, float)
    uv = np.full((len(vtx), 2), 0.25)
    z = CD.tops_of(vtx, uv)["z"][0]
    assert z == [0.0, 3.0, 4.0, 3.0]


def test_a_standing_face_seen_end_on_is_no_face_of_the_land():
    """A wall running straight away from the eye is left out; a wall facing it, and ground far off, are kept."""
    eye = np.array([0.0, 0.0, 1.6])
    def grid(o, du, dv):
        return np.array([[o + i * dv + j * du for j in range(3)] for i in range(3)], float)
    facing = grid(np.array([-1.0, 10.0, 2.0]), np.array([1.0, 0, 0]), np.array([0, 0, -1.0]))
    away = grid(np.array([1.5, 10.0, 2.0]), np.array([0.02, 1.2, 0]), np.array([0, 0, -1.0]))
    ground = grid(np.array([-1.0, 40.0, 0.0]), np.array([1.0, 0, 0]), np.array([0, -8.0, 0]))
    assert not CD._end_on(facing, eye).any()
    assert CD._end_on(away, eye).all()
    assert not CD._end_on(ground, eye).any()


def test_a_picture_stage_carries_nothing_past_its_pictures_edges():
    src = inspect.getsource(CD.stage_land)
    assert "carry=False" in src and "back_k=STAGE_BACK_K" in src
    assert 1.0 < CD.STAGE_BACK_K < CD.LAND_BACK_K
    assert "if carry:" in inspect.getsource(CD.land)
    assert inspect.signature(CD.land).parameters["carry"].default is True      # open land: as before


# ------------------------------------------------------------------ people stand across a stage of one picture
def _fp():
    import from_plan as FP
    return FP


def test_a_group_on_a_stage_of_one_picture_stands_across_it(monkeypatch):
    """Every camera of such a place looks one way: nobody is stood before or behind another of their group."""
    FP = _fp()
    room = [12.0, 12.0, 3.0]
    monkeypatch.setattr(FP, "STAGE_DIR", [0.0, 1.0])
    placed = [[0.0, 0.0]]
    for _ in range(3):
        m = FP.free_mark([0.0, 0.0], placed, [], room, (0.5, 0.9), walls=False)
        assert FP.across(m, placed), (m, placed)
        placed.append(m)
    for i, p in enumerate(placed):
        for q in placed[i + 1:]:
            ang = math.degrees(math.atan2(abs(q[0] - p[0]), abs(q[1] - p[1])))
            assert ang >= 90 - FP.STAGE_ACROSS_DEG - 1e-6 or math.dist(p, q) > FP.STAGE_GROUP_M
    marks = FP.free_marks([0.0, 0.0], [[0.0, 0.0]], [], room, (0.6, 1.6))
    assert marks and all(FP.across(m, [[0.0, 0.0]]) for m in marks)


def test_a_room_and_a_true_panorama_place_people_as_before(monkeypatch):
    FP = _fp()
    room = [12.0, 12.0, 3.0]
    assert FP.stage_dir({}) == [] and FP.stage_dir({"picture_dir": [0, 1], "all_round": True}) == []
    assert FP.stage_dir({"picture_dir": [0.0, 2.0]}) == [0.0, 1.0]
    monkeypatch.setattr(FP, "STAGE_DIR", [])
    was = FP.free_mark([0.0, 0.0], [[0.0, 0.0]], [], room, (0.5, 0.9))
    assert FP.across([0.0, -1.2], [[0.0, 0.0]])
    monkeypatch.setattr(FP, "STAGE_DIR", [0.0, 1.0])
    assert not FP.across([0.0, -1.2], [[0.0, 0.0]]) and FP.across([1.2, 0.0], [[0.0, 0.0]])
    assert FP.across([0.0, -1.2], [[0.0, 0.0, 3.0]])           # (a creature's reach is its own rule)
    assert FP.across([0.0, -6.0], [[0.0, 0.0]])                # (not of one group)
    assert was and len(was) == 2


def test_a_stage_with_no_free_mark_across_it_still_places_the_person(monkeypatch):
    """Never no place: where a wall of things leaves only marks before or behind, one of those."""
    FP = _fp()
    monkeypatch.setattr(FP, "STAGE_DIR", [0.0, 1.0])
    wall = [{"kind": "fixture", "name": f"w{s}", "center": [s * 2.6, 0.0, 1.0], "half": [1.6, 6.0, 1.0]} for s in (-1, 1)]
    m = FP.free_mark([0.0, 0.0], [[0.0, 0.0]], wall, [12.0, 12.0, 3.0], (0.5, 0.9))
    assert m and abs(m[0]) < 0.8 and math.dist(m, [0.0, 0.0]) >= FP.PERSON_GAP_M - 1e-6


def test_two_who_face_each_other_on_such_a_stage_open_to_its_cameras(monkeypatch):
    import speakers as S
    monkeypatch.setattr(S, "PICTURE_SIDE", {"dir": [0.0, 1.0]})
    for xa, xb in ((-0.6, 0.6), (0.6, -0.6), (2.0, 3.2), (-3.2, -2.0)):
        spec = {"characters": {"a": {"position": [xa, 1.0], "facing": [1.0, 0.0], "facing_to": "b"},
                               "b": {"position": [xb, 1.0], "facing": [1.0, 0.0], "facing_to": "a"}}}
        S.aim_bodies(spec)
        assert all(c["facing"][1] < -0.3 for c in spec["characters"].values()), spec   # (toward the cameras' side)


def test_someone_turned_to_another_on_such_a_stage_stays_open_to_its_cameras(monkeypatch):
    """Across the stage from each other they would stand in profile to every camera of its side: each is turned
    part of the way toward the cameras. Seated people, creatures, rooms: as they were."""
    FP = _fp()
    import framing as F
    import speakers as S
    assert F.turned_toward([1.0, 0.0], [0.0, -1.0], 30.0) == [round(math.cos(math.radians(-30)), 4), -0.5]
    assert F.turned_toward([0.0, -2.0], [0.0, -1.0], 30.0) == [0.0, -1.0]            # (never past it)
    assert F.turned_toward([0.1736, -0.9848], [0.0, -1.0], 30.0) == [0.0, -1.0]
    monkeypatch.setattr(FP, "STAGE_DIR", [0.0, 1.0])
    for f in ([1.0, 0.0], [-1.0, 0.0], [0.6, 0.8]):
        g = FP.stage_open(f, True)
        assert g[1] < f[1] - 0.3 and abs(math.hypot(*g) - 1) < 1e-3, (f, g)
        assert FP.stage_open(f, False) == f                                           # (seated: the seat's way)
    monkeypatch.setattr(FP, "STAGE_DIR", [])
    assert FP.stage_open([1.0, 0.0], True) == [1.0, 0.0]                                # (a room)
    # the engine's own turn (bodies re-aimed from where they now stand): one turned to another who looks elsewhere
    spec = {"characters": {"a": {"position": [-1.0, 1.0], "facing": [0.0, 1.0], "facing_to": "b"},
                           "b": {"position": [1.0, 1.0], "facing": [0.0, -1.0]}}}
    monkeypatch.setattr(S, "PICTURE_SIDE", {"dir": [0.0, 1.0]})
    S.aim_bodies(spec)
    assert spec["characters"]["a"]["facing"] == [0.866, -0.5] and spec["characters"]["b"]["facing"] == [0.0, -1.0]
    S.aim_bodies(spec)                                                              # (asked again: the same)
    assert spec["characters"]["a"]["facing"] == [0.866, -0.5]
    monkeypatch.setattr(S, "PICTURE_SIDE", {})
    S.aim_bodies(spec)
    assert spec["characters"]["a"]["facing"] == [1.0, 0.0]


def test_a_wide_on_several_named_people_takes_one_of_them_a_little_nearer_than_its_lead():
    """The named of a full or wide shot stand at sizes of one picture; in every other shot the lead's head leads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import staging_synth as SY
    import speakers as S
    S.SOLIDS[:] = []
    subj = {"a": SY.person([0.0, 3.0]), "b": SY.person([1.2, 2.5]), "c": SY.person([-1.2, 3.2])}
    cam = SY.cam([0.0, -1.5, 1.6], [0.0, 3.0, 1.5], 24)

    def why(kind, show, speaker=None):
        return S.framing_why(cam, kind, speaker, None, False, show, subj, [1152, 640],
                             opts={"line_subject": "a", "subject": "a", "listed": show})[0]
    ha, hb = (S.head_h(cam, subj[k]["measured"], [1152, 640]) for k in "ab")
    assert 1.02 * ha < hb < S.FULL_LEAD_K * ha                       # (b a little nearer the camera than a)
    assert not [w for w in why("full", ["a", "b", "c"]) if "is bigger than" in w]
    assert not [w for w in why("wide", ["a", "b", "c"]) if "is bigger than" in w]
    assert [w for w in why("medium", ["a", "b", "c"]) if w.startswith("b's head")]       # (not a wide: as before)
    assert [w for w in why("full", ["a", "c"]) if w.startswith("b's head")]              # (b is not named for it)
    assert [w for w in why("full", ["a", "b", "c"], speaker="a") if w.startswith("b's head")]   # (a line: as before)
    subj["b"] = SY.person([0.6, 0.4])                                # (b far nearer: no picture of the group)
    assert [w for w in why("full", ["a", "b", "c"]) if w.startswith("b's head")]


def test_what_someone_holds_stays_in_their_hand_through_the_shots_that_do_not_name_them():
    FP = _fp()
    mem = {}
    fid = {"asset": "x1", "at": "hand", "hand": "r", "name": "a lute"}
    assert FP.held_on(mem, {"c": [fid], "a": [{"asset": "h", "at": "head", "name": "a cap"}]}, {"a", "c"},
                      {"a", "b", "c"}) == {"c": [fid], "a": [{"asset": "h", "at": "head", "name": "a cap"}]}
    assert FP.held_on(mem, {}, set(), {"a", "b", "c"}) == {"c": [fid]}      # (another's line: still in the hand)
    assert FP.held_on(mem, {"a": []}, {"a"}, {"a", "b", "c"}) == {"a": [], "c": [fid]}
    assert FP.held_on(mem, {}, {"c"}, {"a", "b", "c"}) == {}                # (the pass has them hold nothing now)
    assert FP.held_on(mem, {}, set(), {"a", "b", "c"}) == {}
    FP.held_on(mem, {"c": [fid]}, {"c"}, {"a", "c"})
    assert FP.held_on(mem, {}, set(), {"a"}) == {} and mem == {"a": []} or mem == {}   # (gone from the place)
