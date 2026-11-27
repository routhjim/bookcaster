"""A made floor in the open (a deck, a pier, a rooftop, a terrace: tests/test_made_floor_1126.py), 11-27: how large it
is and how it is lit at night.

Its size: measured as a room is (by everything at standing height across the whole picture) one came out 38 m wide and
15 m deep, and the people stood on a vast empty floor. It is measured by its own floor now: where the points lying on
the floor plane end, no farther than a few metres beyond its things and the eye, each side within its bounds; its
scale corrected by its things' real heights only when several of them speak.

Its light: a place in the open whose panorama's sky shows no light had no sun light at all, so no shot of it had a key,
and its moon at the sky's own night level was under a hundredth of what the exposure needs: the set was black. Every
place in the open has its light from the sky now, never under its share of the exposure's own, and what hangs over a
made floor hangs (and burns) at its own height. Made-up data only."""
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
import pieces as P  # noqa: E402
import place_build as PB  # noqa: E402
import scene  # noqa: E402

# ------------------------------------------------------------------ 1. the floor's size
F, W_PX, H_PX = 0.7, 320, 200            # normalised focal (of the width), picture size
EYE_H = 1.5
INTR = np.array([[F, 0, 0.5], [0, F * W_PX / H_PX, 0.5], [0, 0, 1]])


def deck_points(half=3.5, back=9.0, far_side=18.0, far_back=30.0, drop=3.0):
    """The Pier seen from the eye (camera coords: x right, y down, z forward): a floor `half` wide either side of
    the eye and `back` long; round it the water, `drop` lower; and far beyond its edge other built things at standing
    height (long walls far_side to the left and right, one across far_back ahead). The sky is not measured."""
    v, u = np.mgrid[0:H_PX, 0:W_PX]
    r = np.stack([((u + 0.5) / W_PX - 0.5) / F, ((v + 0.5) / H_PX - 0.5) / INTR[1, 1], np.ones(u.shape)], -1)
    t = np.full(u.shape, np.inf)

    def take(tt, ok):
        nonlocal t
        tt = np.where(ok & (tt > 0), tt, np.inf)
        t = np.minimum(t, tt)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = EYE_H / r[..., 1]                               # the floor
        take(tf, (np.abs(r[..., 0] * tf) <= half) & (tf <= back))
        take((EYE_H + drop) / r[..., 1], np.ones(u.shape, bool))      # the water
        for sx in (-far_side, far_side):                     # walls far off to the sides, 4 m high over the floor
            ts = sx / r[..., 0]
            take(ts, (EYE_H - r[..., 1] * ts >= -drop) & (EYE_H - r[..., 1] * ts <= 4.0))
        take(np.full(u.shape, far_back), (EYE_H - r[..., 1] * far_back >= -drop) & (EYE_H - r[..., 1] * far_back <= 4.0))
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


KINDS = {"crate": [0.8, 0.8, 0.8], "bench": [1.5, 0.5, 0.5]}


def test_a_made_floor_is_as_large_as_its_own_floor_not_as_what_stands_beyond_it():
    pts, mask = deck_points()
    boxes = {"spec:p1": (box_of(-1.0, 4.0, 0.8, 0.8), "crate"), "seen:0": (box_of(1.5, 6.0, 1.5, 0.5), "bench")}
    # (the defect: measured as a room, the far walls are its walls)
    old = CD.room_from(pts, mask, INTR, boxes, KINDS)["room_m"]
    assert old[0] > 25.0 and old[1] > 25.0
    r = CD.room_from(pts, mask, INTR, boxes, KINDS, open_top=True)
    W, D, _ = r["room_m"]
    assert 6.0 <= W <= 8.5                                     # the floor is 7 m wide
    assert 9.0 <= D <= 12.0                                    # ... 9 m long before the eye, and room behind it
    t = r["things"]["spec:p1"]                                 # its things stand where the picture has them, from the eye
    assert abs((t["xy"][0] - r["eye"][0]) - (-1.0)) < 0.2 and abs((t["xy"][1] - r["eye"][1]) - (4.0 + t["size_m"][1] / 2)) < 0.3
    assert abs(r["eye"][0]) < W / 2 and abs(r["eye"][1]) < D / 2          # the eye stands on the floor


def test_a_room_is_measured_as_before():
    pts, mask = deck_points()
    a = CD.room_from(pts, mask, INTR, {}, KINDS)
    b = CD.room_from(pts, mask, INTR, {}, KINDS, open_top=False, real_m={"spec:p1": 9.0})
    assert a == b


def test_a_vast_made_floor_reaches_only_a_little_beyond_its_things():
    pts, mask = deck_points(half=45.0, back=70.0, far_side=60.0, far_back=90.0)      # the Square: paved to every side
    boxes = {"spec:p1": (box_of(-2.0, 5.0, 0.8, 0.8), "crate"), "seen:0": (box_of(2.0, 7.0, 1.5, 0.5), "bench")}
    r = CD.room_from(pts, mask, INTR, boxes, KINDS, open_top=True)
    W, D, _ = r["room_m"]
    assert W <= 4.0 + 2 * CD.OPEN_FLOOR_ROUND_M + 0.5          # its things are 4 m apart
    assert D <= 7.0 + CD.OPEN_FLOOR_ROUND_M + 0.2 * (7.0 + CD.OPEN_FLOOR_ROUND_M) + 0.5
    # with nothing standing on it: each side at most the bound
    W0, D0, _ = CD.room_from(pts, mask, INTR, {}, KINDS, open_top=True)["room_m"]
    assert W0 <= CD.OPEN_FLOOR_M[1] + 0.01 and D0 <= CD.OPEN_FLOOR_M[1] + 0.01


def test_the_bounds_force_no_square_and_keep_the_eye_on_the_floor():
    # the Pier: 3 m wide, 40 m long, one thing 18 m along it
    left, right, front, back = CD.floor_bounds(-1.5, 1.5, 40.0, [[0.0, 18.0]], 1.0)
    assert (left, right) == (-1.5, 1.5)                        # long and narrow stays narrow
    assert back - front == CD.OPEN_FLOOR_M[1] and front < 0.0 < back
    # a strip of floor seen only to one side of the eye: the eye is on it all the same, at its least width
    left, right, _, _ = CD.floor_bounds(0.4, 0.9, 5.0, [], 1.0)
    assert left == -CD.OPEN_FLOOR_M[0] / 2 and right == CD.OPEN_FLOOR_M[0] / 2
    # too wide: the stretch about its things is the one kept
    left, right, _, _ = CD.floor_bounds(-40.0, 40.0, 10.0, [], 1.0)
    assert right - left == CD.OPEN_FLOOR_M[1] and left < 0.0 < right
    # the bounds are real metres: a picture whose metres are short of the real ones is bounded sooner
    left, right, _, _ = CD.floor_bounds(-40.0, 40.0, 10.0, [], 1.5)
    assert abs((right - left) * 1.5 - CD.OPEN_FLOOR_M[1]) < 1e-9


def test_floor_span_reads_the_floor_and_needs_enough_of_it():
    rng = np.random.default_rng(3)
    on = np.column_stack([rng.uniform(-3, 3, 40000), rng.uniform(0.5, 8, 40000), rng.normal(0, 0.02, 40000)])
    off = np.column_stack([rng.uniform(-30, 30, 40000), rng.uniform(0.5, 40, 40000), rng.uniform(0.3, 2.0, 40000)])
    stray = np.column_stack([rng.uniform(15, 21, 3000), rng.uniform(4, 31, 3000), np.zeros(3000)])   # another floor, apart
    left, right, back = CD.floor_span(np.concatenate([on, off, stray]))
    assert abs(left + 3.0) <= 0.5 and abs(right - 3.0) <= 0.5 and abs(back - 8.0) <= 0.5
    assert CD.floor_span(off) is None and CD.floor_span(on[:20]) is None


def test_one_misjudged_thing_does_not_swing_the_scale():
    things = {f"seen:{i}": {"height": 2.0, "kind": "work_piece"} for i in range(3)}
    # one thing judged far too small, alone: no correction (the door's own scale, or none)
    assert CD.things_scale(things, {"seen:0": 0.9}, 1.0) == 1.0
    assert CD.things_scale(things, {"seen:0": 6.0}, 1.0) == 1.0
    assert CD.things_scale(things, {"seen:0": 6.0, "seen:1": 2.2}, 1.1) == 1.1
    # three speak: the middle one, whatever the odd one says
    assert CD.things_scale(things, {"seen:0": 6.0, "seen:1": 2.2, "seen:2": 2.4}, 1.0) == pytest.approx(1.2)
    # a door stands DOOR_M and is one voice among them; a small thing's judged size is no voice
    doors = dict(things, **{"seen:9": {"height": 1.6, "kind": "door"}})
    assert CD.things_scale(doors, {"seen:0": 2.6, "seen:1": 2.6}, 1.25) == pytest.approx(1.3)
    assert CD.things_scale(things, {"seen:0": 1.0, "seen:1": 1.0, "seen:2": 1.0}, 1.0) == 1.0
    # ... and never beyond what the depth model's metres are trusted within
    assert CD.things_scale(things, {k: 9.0 for k in things}, 1.0) == CD.SCALE_CLAMP[1]


# ------------------------------------------------------------------ 2. the light of a place in the open at night
T1 = {"id": "t1", "kind": "terrain", "look": "dark water under a dark sky", "phrases": [], "against": "free"}


def world_lib(tmp_path, sun):
    aid = P.generation_request(T1)["asset_id"]
    d = tmp_path / "prop" / aid
    d.mkdir(parents=True, exist_ok=True)
    (d / "asset.json").write_text(json.dumps({"id": aid, "size_m": [300, 300, 160], "world": {
        "stand_m": [0.0, 0.0, 40.0], "flat_r_m": 9.0, "sun": sun}}))
    return tmp_path


def staged(tmp_path, sun, time, ev=5.0):
    out = {"structure": [], "fixtures": [], "props": [], "lights": [], "size_m": [10, 10, 6], "exposure_ev100": ev}
    PB.world_stage(out, {"pieces": [T1], **({"_time": time} if time else {})}, world_lib(tmp_path, sun))
    return out


def sun_of(out):
    return next(l for l in out["lights"] if l["type"] == "sun")


def test_a_world_whose_sky_shows_no_light_still_has_its_light_and_every_shot_its_key(tmp_path):
    for time in ("night", "dusk", None):
        out = staged(tmp_path, None, time)
        s = sun_of(out)                                        # (the defect: no sun light at all, so no key)
        assert s["toward_sun"] == [PB.OPEN_SUN_TOWARD[0], PB.OPEN_SUN_TOWARD[1], PB.OPEN_SUN_TOWARD[2]]
        assert s["toward_sun"][1] < 0 < s["toward_sun"][2]     # from the picture's eye's side, from above
        assert out["world"]["sun_from"].startswith("no light in its sky")
        p = tmp_path / f"spec_{time}.json"
        p.write_text(json.dumps(out))
        kl = scene.key_light(p, 5.4)
        assert kl and kl["toward_sun"] == s["toward_sun"] and kl["lux"] == round(scene.KEY_LUX_EV * 2 ** 5.4, 1)
    assert staged(tmp_path, {"dir": [0.3, -0.9, 0.5], "level": 0.4}, None)["world"]["sun_from"] == "its sky"


def test_full_lux_is_the_light_a_mid_grey_needs():
    assert PB.full_lux(5.0) == pytest.approx(math.pi * 4.0 / 0.18)        # 69.8 lux
    assert 69.0 < PB.full_lux(5.0) < 70.5 and 91.5 < PB.full_lux(5.4) < 92.5


def test_night_and_dusk_are_dim_but_never_black(tmp_path):
    dark_sky = {"dir": [0.3, -0.9, 0.5], "color": "#8090c0", "level": 0.03, "peak": 0.2}
    full = PB.full_lux(5.0)
    # (the defect: the moon at a dark sky's own level was 0.06 x 120 x 0.08 = 0.6 lux, under a hundredth of `full`)
    assert PB.NIGHT_SUN * PB.WORLD_SUN_LUX * 0.08 < 0.01 * full
    night = sun_of(staged(tmp_path, dark_sky, "night"))
    assert night["lux"] == pytest.approx(PB.OPEN_KEY_LEAST["night"] * full, abs=0.06) and night["kelvin"] == PB.NIGHT_KELVIN
    dusk = sun_of(staged(tmp_path, dark_sky, "dusk"))
    assert dusk["lux"] == pytest.approx(PB.OPEN_KEY_LEAST["dusk"] * full, abs=0.06)
    # night reads as night: under the dusk, the dusk under a full exposure's light
    assert night["lux"] < dusk["lux"] < full
    # the least follows the set's own exposure
    assert sun_of(staged(tmp_path, dark_sky, "night", ev=6.0))["lux"] == pytest.approx(2 * night["lux"], abs=0.11)
    # a bright sky's low sun at dusk is over the least and kept; the day is as before
    bright = {"dir": [0.3, -0.9, 0.5], "color": "#ffffff", "level": 0.5}
    assert sun_of(staged(tmp_path, bright, "dusk"))["lux"] == round(PB.DUSK_SUN * PB.WORLD_SUN_LUX, 1)
    assert sun_of(staged(tmp_path, bright, None))["lux"] == PB.WORLD_SUN_LUX
    assert sun_of(staged(tmp_path, dark_sky, None))["lux"] == round(PB.WORLD_SUN_LUX * 0.08, 1)


# ------------------------------------------------------------------ 3. a made floor at night
def spec_of():
    return {"place": "the Pier", "room": "A long timber pier over the water.", "_director_room": "Outside: a long timber pier",
            "kind": "outdoors", "period": "medieval", "size": "medium", "openings": [], "outside": "", "_time": "night",
            "pieces": [{"id": "p1", "kind": "chair", "look": "a plain oak chair", "phrases": ["by the chair"], "against": "free"},
                       {"id": "p2", "kind": "lantern", "look": "a lantern", "phrases": ["under the lantern"],
                        "against": "free", "gives_light": True, "hangs_m": 2.2},
                       {"id": "p3", "kind": "lantern", "look": "a small lamp", "phrases": ["by the lamp"], "against": "free",
                        "gives_light": True, "hangs_m": 0.3}],
            "relations": [], "named_places": {"by the chair": "p1", "under the lantern": "p2", "by the lamp": "p3"}, "land_places": {}}


def concept_of():
    return {"concept": "/c/concept.png", "space": "outdoors", "ground": "built", "seats_abreast": 1,
            "surfaces": {"walls": "boards, brown", "floor": "planks, pale", "ceiling": "the night sky, black"},
            "light": "from lanterns, warm", "far": "dark water",
            "surfaces_words": "The walls: boards, brown; the floor: planks, pale; the ceiling: the night sky, black.",
            "bare": {"walls": {"colour": "#5a4030"}, "floor": {"colour": "#b09060"}}, "seen": [],
            "spec": {"p1": {"in_picture": True, "look": "oak, brown", "box": [0.1, 0.4, 0.3, 0.9], "crop": "/c/p1.png"},
                     "p2": {"in_picture": False, "look": ""}, "p3": {"in_picture": False, "look": ""}},
            "depth": {"room_m": [7.0, 8.0, 2.2], "eye": [0.0, -3.0, 1.5], "off_floor": {},
                      "things": {"spec:p1": {"xy": [-1.5, 1.0], "size_m": [0.6, 0.6, 1.0], "against": None,
                                             "facing": [0.0, -1.0]}}}}


@pytest.fixture
def pier(tmp_path, monkeypatch):
    sp = PB.with_concept(spec_of(), concept_of())
    lib = tmp_path / "lib"
    t = next(p for p in sp["pieces"] if p["kind"] == "terrain")
    d = lib / "prop" / P.generation_request(t)["asset_id"]
    d.mkdir(parents=True)
    (d / "asset.json").write_text(json.dumps({"id": d.name, "size_m": [300, 300, 160],
                                              "world": {"stand_m": [0.0, 0.0, 40.0], "flat_r_m": 9.0, "sun": None}}))

    def records_for(s, library):
        rec = {p["id"]: {"asset": f"{p['kind']}_x", "size_m": [0.4, 0.4, 0.6]} for p in s["pieces"] if p["kind"] != "terrain"}
        return rec, P.resolve(s["pieces"], [])
    monkeypatch.setattr(PB, "records_for", records_for)
    return PB.build(json.loads(json.dumps(sp)), "set_x", lib)


def test_a_made_floor_at_night_under_a_sky_with_no_light_is_lit(pier):
    out, _, report = pier
    assert out["open_top"] is True and report["world"]["why"] == "placed"
    s = sun_of(out)
    assert s["lux"] >= PB.OPEN_KEY_LEAST["night"] * PB.full_lux(out["exposure_ev100"]) - 0.06
    assert s["lux"] < PB.full_lux(out["exposure_ev100"]) and s["kelvin"] == PB.NIGHT_KELVIN
    assert [l["type"] for l in out["lights"]].count("sky") == 1
    # its own lamps burn on beside the sky's light
    assert sum(1 for l in out["lights"] if l["type"] == "point") == 2


def test_what_hangs_over_a_made_floor_hangs_at_its_own_height_on_a_post(pier):
    out, _, _ = pier
    items = {f["name"]: f for f in out["fixtures"] + out["props"]}
    high, low = items["lantern_p2"], items["lantern_p3"]
    assert high["hang"] and high["position"][2] == 2.2         # (the defect: forced down to the edge, 1.1 m less itself)
    post = next(s for s in out["structure"] if s["kind"] == "box" and s["name"] == "post_p2")
    assert post["size_m"] == [PB.OPEN_POST_M, PB.OPEN_POST_M, 2.8] and post["base"][2] == 0.0
    W, D, _ = out["size_m"]
    # the post stands in the edge wall's thickness, just behind the thing: outside the floor, within the wall
    wall = next(s for s in out["structure"] if s["kind"] == "wall")
    off = max(abs(post["base"][0]) - W / 2, abs(post["base"][1]) - D / 2)
    assert 0.0 < off - PB.OPEN_POST_M / 2 and off + PB.OPEN_POST_M / 2 <= wall["thickness_m"]
    assert math.dist(post["base"][:2], high["position"][:2]) < 0.2 + PB.OPEN_POST_M
    assert post["material"] == wall["material"]
    # what hangs within the edge's height hangs on the edge itself: no post
    assert low["position"][2] == 0.3 and not any(s.get("name") == "post_p3" for s in out["structure"])
    # each burns where it hangs: at its own middle, before its face, not at knee height
    zs = sorted(l["position"][2] for l in out["lights"] if l["type"] == "point")
    assert zs == [pytest.approx(0.3 + 0.3), pytest.approx(2.2 + 0.3)]
    lt = max((l for l in out["lights"] if l["type"] == "point"), key=lambda l: l["position"][2])
    assert abs(lt["position"][0]) < W / 2 and abs(lt["position"][1]) < D / 2          # over the floor


def test_a_room_hangs_and_lights_no_post(tmp_path, monkeypatch):
    lay = {"size_m": [6.0, 5.0, 3.0], "openings": [], "pieces": []}
    PB.hang(lay, [{"id": "p2", "kind": "lantern", "hangs_m": 2.0}], {"p2": {"asset": "lantern_x", "size_m": [0.4, 0.4, 0.6]}})
    q = lay["pieces"][0]
    assert q["z_m"] == 2.0 and "post_m" not in q
    out = {"structure": [{"kind": "wall", "material": "m"}]}
    assert PB.hung_posts(out, lay) == [] and len(out["structure"]) == 1
