"""A place in the open built as a world (11-16): the place spec's outdoors rules, the concept check that turns an
outdoor place into one terrain piece plus free-standing pieces, the terrain's generation request (a panorama drawn
from the concept), the world's stage in the set (no structure, the ground under the eye at the origin, its light),
and the panorama arithmetic (ground scale, flat patch, sky dome, where the light comes from). Synthetic data only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("sets", "places", "scenes", "img3d", ""):
    sys.path.insert(0, str(A / sub))
import pano_shell as PS  # noqa: E402
import pieces as P  # noqa: E402
import place_build as PB  # noqa: E402
import place_spec as SP  # noqa: E402


def spec_out(**kw):
    s = {"room": "Open ground.", "fixtures": ["a", "b"], "crowd": "", "outside": "", "schema": "place_spec/1",
         "place": "X", "kind": "outdoors", "period": "medieval", "size": "medium", "openings": [],
         "pieces": [{"id": "p1", "kind": "landmark", "look": "a grey stone column", "phrases": ["by the column"],
                     "against": "free", "longest_m": 6}],
         "relations": [], "named_places": {"by the column": "p1", "on the slope": "floor"}}
    s.update(kw)
    return s


def test_spec_outdoors_valid():
    assert SP.validate(spec_out(), ["by the column", "on the slope"]) == []


def test_spec_outdoors_faults():
    f = SP.validate(spec_out(openings=[{"id": "o1", "kind": "door", "wall": "north", "words": ""}],
                             pieces=[{"id": "p1", "kind": "landmark", "look": "x", "phrases": [], "against": "wall"}]),
                    [])
    assert any("no openings" in x for x in f)
    assert any("against must be \"free\"" in x for x in f)
    assert any("longest_m" in x for x in f)


def test_prompt_names_outdoors():
    txt = SP.prompt("X", ["here"], ["a"], "text")
    assert "outdoors" in txt and "landmark" in txt


CONCEPT = {"concept": "/c/concept.png", "space": "outdoors", "seats_abreast": 1,
           "surfaces": {"walls": "rock", "floor": "dry grass, yellow", "ceiling": "the sky, grey"},
           "light": "from the low sun, orange", "seen": [], "spec": {"p1": {"in_picture": True, "look": "grey stone"},
                                                                    "p2": {"in_picture": False, "look": ""}}}


def test_with_concept_outdoors():
    sp = spec_out(kind="cabin", outside="hills", _views={"s1": "hills"}, _director_room="Outside: a dry plain",
                  pieces=[{"id": "p1", "kind": "landmark", "look": "x", "phrases": ["by the column"], "against": "wall",
                           "longest_m": 6},
                          {"id": "p2", "kind": "bench", "look": "y", "phrases": ["on the bench"], "against": "wall"}],
                  named_places={"by the column": "p1", "on the bench": "p2", "by the door": "o1"})
    out = PB.with_concept(sp, CONCEPT)
    assert out["kind"] == "outdoors" and out["openings"] == [] and out["outside"] == "" and "_views" not in out
    kinds = [p["kind"] for p in out["pieces"]]
    assert kinds.count("terrain") == 1 and "landmark" in kinds
    assert all(p["against"] == "free" for p in out["pieces"])
    t = next(p for p in out["pieces"] if p["kind"] == "terrain")
    assert t["source"] == "/c/concept.png" and "a dry plain" in t["look"] and "dry grass" in t["look"]
    assert "Outside" not in t["look"]
    assert out["named_places"]["by the door"] == "floor" and out["named_places"]["by the column"] == "p1"
    lm = next(p for p in out["pieces"] if p["kind"] == "landmark")
    assert lm["look"] == "x; grey stone"                      # (its own words first, the picture's material after)


def test_terrain_request_is_a_panorama_from_the_concept():
    r = P.generation_request({"kind": "terrain", "look": "a dry plain", "source": "/c/concept.png"})
    st = {s["step"]: s for s in r["plan"]["steps"]}
    argv = st["draw"]["argv"]
    assert "picture" in argv and "2048" in argv and "/c/concept.png" in argv
    assert any("Equirectangular" in a and "a dry plain" in a for a in argv)
    assert st["spec"]["spec"]["made_as"] == "world"
    assert any(a.endswith("shell_one.py") for a in st["shape"]["argv"])
    # another place's world never serves: only the one made from this piece's own words and picture
    r2 = P.generation_request({"kind": "terrain", "look": "a dry plain", "source": "/d/concept.png"})
    assert r2["asset_id"] != r["asset_id"]


def test_world_stage(tmp_path):
    t = {"id": "t1", "kind": "terrain", "look": "a dry plain", "phrases": [], "against": "free", "source": "/c/c.png"}
    aid = P.generation_request(t)["asset_id"]
    d = tmp_path / "prop" / aid
    d.mkdir(parents=True)
    (d / "asset.json").write_text(json.dumps({"id": aid, "size_m": [300, 300, 160], "world": {
        "stand_m": [1.5, -2.0, 40.0], "flat_r_m": 9.0, "sun": {"dir": [0.3, -0.9, 0.05], "color": "#ffb070",
                                                               "level": 0.4, "peak": 0.9}}}))
    out = {"structure": [{"kind": "floor"}], "fixtures": [], "props": [], "lights": [], "size_m": [10, 10, 6]}
    got = PB.world_stage(out, {"pieces": [t]}, tmp_path)
    assert got["world"] == aid and out["structure"] == [] and out["interior"] is False
    w = out["fixtures"][0]
    assert w["position"] == [-1.5, 2.0, -40.0] and w["world"] and w["outside"]
    sun = next(x for x in out["lights"] if x["type"] == "sun")
    assert sun["toward_sun"][2] >= 0.15 and sun["kelvin"] < 6500          # (a low warm sun: raised, warm)
    assert any(x["type"] == "sky" for x in out["lights"])
    # not made yet: said, nothing placed
    out2 = {"structure": [], "fixtures": [], "props": [], "lights": [], "size_m": [10, 10, 6]}
    assert PB.world_stage(out2, {"pieces": [dict(t, look="other")]}, tmp_path)["world"] is None


def test_kelvin_of():
    assert PB.kelvin_of("#ffffff") == 6500
    assert PB.kelvin_of("#ff9040") < 4000 < 9000 < PB.kelvin_of("#8090ff")


def test_land_scales_flattens_and_domes():
    H, W = 64, 128
    dz = np.cos(np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    # a ground plane 0.4 m under the eye (the depth model's small scale on a painted panorama), sky above
    depth = np.where(dz < -1e-3, 0.4 / np.maximum(-dz, 1e-3), 50.0)
    depth = np.minimum(depth, 30.0)
    sky = dz > 0.05
    d, info = PS.land(depth, sky)
    assert abs(info["scale"] - PS.EYE_STAND_M / 0.4) < 1e-3
    assert np.allclose(d[sky], PS.SKY_R_M)
    near = (dz < -0.3)
    z = d[near] * dz[near]
    assert np.allclose(z, -PS.EYE_STAND_M, atol=1e-6)                  # the ground under the people: level
    assert d[~sky].max() <= PS.LAND_MAX_K * PS.SKY_R_M + 1e-6


def test_land_keeps_what_stands_on_the_ground():
    H, W = 64, 128
    dz = np.cos(np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    depth = np.where(dz < -1e-3, 1.6 / np.maximum(-dz, 1e-3), 50.0)
    depth[40:44, 10:20] = 0.5                                           # a thing standing up near the eye
    d, _ = PS.land(depth, np.zeros((H, W), bool))
    assert np.allclose(d[40:44, 10:20], 0.5)


def test_sun_of_finds_the_bright_sky():
    H, W = 64, 128
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:32] = (20, 20, 40)
    rgb[10:13, 30:33] = (255, 230, 200)
    sky = np.zeros((H, W), bool)
    sky[:30] = True
    s = PS.sun_of(rgb, sky)
    want = PS.pano_dirs(H, W)[11, 31]
    assert np.dot(s["dir"], want) > 0.99 and s["color"].startswith("#ff")
    assert PS.sun_of(rgb, np.zeros((H, W), bool)) is None


def test_sky_mask_joined_to_the_top():
    pytest.importorskip("scipy")
    H, W = 64, 128
    prob = np.zeros((H, W), np.float32)
    prob[:20] = 0.9
    prob[26:30, 50:60] = 0.9                       # a pale patch below the sky, not joined to it
    prob[40:50] = 0.9                              # below the horizon: never sky
    sky = PS.sky_mask(prob, np.ones((H, W), bool))
    assert sky[:18].all() and not sky[26:30, 50:60].any() and not sky[32:].any()


def test_land_takes_off_the_depth_models_ground_step():
    # the depth model's ground rises smoothly over the plane going out (it stood 0.5 to 1.3 m over it a few metres
    # out, a step at the patch's edge before 11-16), with a thing standing on it: the ground is made the plane, the
    # thing stands on the plane as tall as it stood on the model's ground
    H, W = 512, 1024
    dz = np.cos(np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    dn = dz < -1e-3
    t = np.where(dn, 1.6 / np.maximum(-dz, 1e-3), 50.0)
    rp = t * np.sqrt(1 - dz * dz)
    lift = np.clip(0.05 * rp, 0, 0.3)                          # the model's ground: up to 0.3 m over the plane
    depth = np.where(dn, (lift - 1.6) / np.minimum(dz, -1e-3), t)
    depth = np.minimum(depth, 60.0)
    sky = dz > 0.05
    rows = np.where((dz[:, 0] < -0.25) & (dz[:, 0] > -0.32))[0][:4]
    sl = (slice(rows[0], rows[-1] + 1), slice(400, 416))
    over0 = depth * dz + 1.6
    depth[sl] = (over0[sl] + 1.2 - 1.6) / dz[sl]               # a thing 1.2 m over the model's ground there
    d, info = PS.land(depth, sky)
    over = d * dz + 1.6
    ring = (dz < -0.2) & (dz > -0.6)
    ring[rows[0] - 20:rows[-1] + 20, 380:436] = False
    assert np.abs(over[ring]).max() < 0.05
    assert abs(over[sl] - 1.2).max() < 0.1


def test_spot_of_the_nearest_edge():
    H, W = 128, 256
    dz = np.cos(np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    d = np.where(dz < -1e-3, 1.6 / np.maximum(-dz, 1e-3), 150.0)
    sky = dz > 0
    dirs = PS.pano_dirs(H, W)
    p = dirs * d[..., None]
    prob = np.where((p[..., 1] < -4.0) & (np.abs(p[..., 0]) < 3) & (dz < 0), 0.9, 0.05)   # water ahead, from 4 m out
    sp = PS.spot_of(prob, d, sky)
    assert sp is not None and abs(sp[0]) < 0.5 and -4.0 + PS.SPOT_BACK_M - 0.6 < sp[1] < -4.0 + PS.SPOT_BACK_M + 0.3
    assert PS.spot_of(np.zeros((H, W)), d, sky) is None


def test_terrain_spot_in_the_drawing_and_the_id():
    a = P.generation_request({"kind": "terrain", "look": "a plain", "source": "/c/c.png", "spot": "the water's edge"})
    b = P.generation_request({"kind": "terrain", "look": "a plain", "source": "/c/c.png"})
    assert a["asset_id"] != b["asset_id"]
    draw = next(s for s in a["plan"]["steps"] if s["step"] == "draw")["argv"]
    assert any("the water's edge" in x for x in draw)
    assert next(s for s in a["plan"]["steps"] if s["step"] == "spec")["spec"]["spot"] == "the water's edge"


def test_land_places_spot_and_validation():
    sp = spec_out(named_places={"by the column": "p1", "at the water": "floor"}, land_places={"at the water": "the water's edge"},
                  _phrases=["by the column", "at the water"])
    assert SP.validate(sp, ["by the column", "at the water"]) == []
    bad = dict(sp, land_places={"by the column": "x"})
    assert any("land_places" in x for x in SP.validate(bad, ["by the column", "at the water"]))
    out = PB.with_concept(sp, CONCEPT)
    assert next(p for p in out["pieces"] if p["kind"] == "terrain")["spot"] == "the water's edge"


def test_kit_for_a_period_without_a_kit():
    # (the kit it is most like, for its lights and its build; nothing of that period's style: place_build.plain)
    lent, own = PB.kit_for("medieval"), PB.kit_for("victorian")
    assert lent["lights"] == own["lights"] and lent["materials"] == own["materials"] and lent["_plain"]
    assert PB.kit_for("modern") == PB.kit_for("far_future")
    assert PB.kit_for("not stated") == PB.kit_for("victorian")


def test_room_light_in_a_closed_dark_room():
    out = {"size_m": [3.0, 6.0, 3.0], "structure": [{"kind": "floor"}, {"kind": "ceiling"}],
           "lights": [{"type": "sky", "intensity": 1.0}, {"type": "point", "lumens": 500, "position": [0, 0, 0.3]}]}
    assert PB.room_light(out, {})
    add = [l for l in out["lights"] if l["type"] == "point"][1:]
    assert add and sum(l["lumens"] for l in add) + 500 >= PB.ROOM_LM_M2 * 18 - 1
    lit = {"size_m": [3.0, 6.0, 3.0], "structure": [{"kind": "ceiling"}], "lights": [{"type": "sun"}]}
    assert PB.room_light(lit, {})                 # (daylight at its windows too: everyone inside is seen, 11-19)
    assert not PB.room_light(lit, {})             # (lit enough now: nothing more is added)
    assert not PB.room_light({"size_m": [3.0, 6.0, 3.0], "structure": [{"kind": "ceiling"}],
                              "lights": [{"type": "strip_lights"}]}, {})    # (a room with its own strip of lights)
    assert not PB.room_light({"size_m": [9, 9, 6], "structure": [], "lights": []}, {})   # (in the open)


def _cliff_picture():
    """A camera 2 m up looking level: open ground, and from 45% across a rock face about 17 m off that the top
    and the right edge of the frame cut."""
    h, w = 240, 400
    v, u = np.mgrid[0:h, 0:w]
    ray = np.stack([(u + .5) / w - .5, ((v + .5) / h - .5) * h / w, np.full((h, w), 0.6)], -1)
    ground = np.where(ray[..., 1] > 1e-6, 2.0 / np.maximum(ray[..., 1], 1e-6), 1e9)
    cliff = np.where(u > w * 0.45, 10.0 / 0.6 + 2 * np.sin(u / 17.0), 1e9)
    t = np.minimum(ground, cliff)
    return ray * t[..., None], t < 60


def test_relief_is_a_closed_form_carried_on_past_the_frame():
    pytest.importorskip("scipy")
    import concept_depth as CD
    pts, mask = _cliff_picture()
    n, d, _ = CD.fit_floor(pts, mask)
    vtx, faces, uv = CD.relief(pts, mask, CD.frame(n, d), d, [0.5, 0.0, 0.8, 0.5], lambda q: q)
    # closed: every edge has two faces, wound against each other; its volume positive (its faces look outward)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    k = e[:, 0] * len(vtx) + e[:, 1]
    assert len(np.unique(k)) == len(k) and set(k) == set(e[:, 1] * len(vtx) + e[:, 0])
    a, b, c = (vtx[faces[:, i]] for i in range(3))
    assert float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) > 0
    lo, hi = vtx.min(0), vtx.max(0)
    seen_top = float(CD.to_room(pts[0, 300][None], CD.frame(n, d), d)[0][2])
    assert hi[2] > seen_top + 0.9 * CD.RELIEF_ON_M[0]         # carried on up past the frame's top
    assert hi[0] > float(pts[:, -1, 0].max()) + 0.9 * CD.RELIEF_ON_M[1]   # ... and past its right edge
    assert abs(lo[2]) < 1e-6 and 0.0 <= uv.min() and uv.max() <= 1.0
    # nothing rising in the box: no form
    assert CD.relief(pts, mask, CD.frame(n, d), d, [0.05, 0.6, 0.3, 0.95], lambda q: q) is None


def test_land_form_key():
    import concept_depth as CD
    con = {"spec": {"p1": {"box": [0, 0, .2, .2], "natural": True}, "p2": {"box": [0, 0, .5, .5], "natural": True}},
           "_spec_kinds": {"p1": "landmark", "p2": "landmark"}, "seen": []}
    assert CD.land_form_key(con) == "spec:p2"                 # the largest of the land's own landmarks
    con["spec"]["p1"]["land_form"] = True
    assert CD.land_form_key(con) == "spec:p1"
    assert CD.land_form_key({"spec": {"p1": {"box": [0, 0, 1, 1]}}, "_spec_kinds": {"p1": "landmark"}}) is None


def test_relief_piece_is_placed_whole_and_made_without_a_drawing(tmp_path):
    glb = tmp_path / "concept_relief.glb"
    glb.write_bytes(b"glb")
    con = {"depth": {"eye": [0.0, -5.0, 2.0], "relief": {"of": "spec:p1", "glb": str(glb), "size_m": [20.0, 6.0, 12.0]},
                     "things": {"spec:p1": {"xy": [2.0, 9.0], "size_m": [20.0, 6.0, 12.0]},
                                "seen:0": {"xy": [1.0, 0.0], "size_m": [1.9, 0.6, 0.85]}}},
           "spec": {"p1": {"natural": True, "real_m": [5.0, 3.0]}}, "seen": [{"kind": "bench", "real_m": [1.2, 0.4]}]}
    pcs = [{"id": "p1", "kind": "landmark", "look": "a rock face", "phrases": ["x"]}, {"id": "c1", "kind": "bench"}]
    # neither the land's own thing nor a small one sets the picture's scale
    assert PB.open_scale(pcs, con) == 1.0
    out, stage = PB.open_places(pcs, con)
    p1 = next(p for p in out if p["id"] == "p1")
    assert p1["relief"]["glb"] == str(glb) and p1["_seen_size_m"] == [20.0, 6.0, 12.0] and stage[2] == 12.0
    req = P.generation_request(dict(p1, source="crop.png"))
    steps = [s["step"] for s in req["plan"]["steps"]]
    assert steps == ["spec", "shape"] and req["asset_id"].endswith("_tr")
    sp = req["plan"]["steps"][0]["spec"]
    assert sp["made_as"] == "relief" and sp["size_m"] == [20.0, 6.0, 12.0] and "longest_m" not in sp
    assert "relief_one.py" in " ".join(req["plan"]["steps"][1]["argv"])
    # another shape of the same piece is another asset
    glb.write_bytes(b"glb2")
    assert P.generation_request(dict(p1, source="crop.png"))["asset_id"] != req["asset_id"]


def _open_picture():
    """A camera 2 m up looking level: open ground, a hill 30 m off that the left edge of the frame cuts, and an
    upright thing 8 m off on the right."""
    h, w = 240, 400
    v, u = np.mgrid[0:h, 0:w]
    ray = np.stack([(u + .5) / w - .5, ((v + .5) / h - .5) * h / w, np.full((h, w), 0.6)], -1)
    ground = np.where(ray[..., 1] > 1e-6, 2.0 / np.maximum(ray[..., 1], 1e-6), 1e9)
    hill = np.where(u < w * 0.4, 30.0 / 0.6 + 3 * np.sin(u / 23.0), 1e9)
    post = np.where((abs(u - 0.7 * w) < 12) & (v > 0.35 * h), 8.0 / 0.6, 1e9)
    t = np.minimum(np.minimum(ground, hill), post)
    return ray * t[..., None], t < 80


def test_the_pictures_land_is_one_closed_solid_without_the_built_things():
    pytest.importorskip("scipy")
    import concept_depth as CD
    pts, mask = _open_picture()
    n, d, _ = CD.fit_floor(pts, mask)
    img = (np.random.RandomState(0).rand(240, 400, 3) * 255).astype(np.uint8)
    things = {"spec:p1": {"xy": [1.6, 8.3], "size_m": [0.6, 0.6, 3.0]}, "seen:0": {"xy": [-6.0, 30.0], "size_m": [20, 5, 9]}}
    boxes = {"spec:p1": [0.66, 0.3, 0.74, 0.75], "seen:0": [0.0, 0.2, 0.4, 0.55]}
    vtx, faces, uv, painted, has = CD.land(pts, mask, CD.frame(n, d), d, lambda q: q, things, boxes, img)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    k = e[:, 0] * len(vtx) + e[:, 1]
    assert len(np.unique(k)) == len(k) and set(k) == set(e[:, 1] * len(vtx) + e[:, 0])        # closed
    a, b, c = (vtx[faces[:, i]] for i in range(3))
    assert float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) > 0                         # its faces look outward
    assert int((np.linalg.norm(np.cross(b - a, c - a), axis=1) < 2e-10).sum()) == 0          # no face without area
    front = vtx[uv[:, 0] <= 0.5]
    # the far hill is the land's, and stands; the thing that is built as a piece is not in the land, the ground
    # closed over where it stood and the picture painted over there
    assert has == ["seen:0"] and front[front[:, 1] > 25][:, 2].max() > 5.0
    at = front[(abs(front[:, 0] - 1.6) < 0.6) & (abs(front[:, 1] - 8.3) < 1.0)]
    assert len(at) and at[:, 2].max() < 0.05
    assert (painted != img).any() and painted.shape == img.shape
    # carried on past the frame's left edge, its ground just under the set's
    assert front[:, 0].min() < float(pts[:, 0, 0].min()) - 0.5 * CD.RELIEF_ON_M[1]
    assert abs(float(np.percentile(front[:, 2], 20)) + CD.LAND_UNDER_M) < 1e-6


def test_the_pictures_land_is_set_beside_the_world_and_what_it_has_is_not_built(tmp_path):
    glb = tmp_path / "concept_land.glb"
    glb.write_bytes(b"glb")
    con = {"depth": {"eye": [0.0, -5.0, 2.0],
                     "land": {"glb": str(glb), "xy": [3.0, 20.0], "size_m": [80.0, 40.0, 20.0], "has": ["seen:1"], "under_m": 0.3},
                     "things": {"seen:0": {"xy": [1.0, 4.0], "size_m": [2.0, 1.0, 2.0]},
                                "seen:1": {"xy": [-9.0, 15.0], "size_m": [6.0, 4.0, 5.0]}}},
           "seen": [{"kind": "cart"}, {"kind": "tree", "natural": True}]}
    pcs = [{"id": "c1", "kind": "cart"}, {"id": "c2", "kind": "tree"}]
    out, stage = PB.open_places(pcs, con)
    assert [p["id"] for p in out] == ["c1", PB.LAND_ID] and stage
    c1, ld = out
    # the land stands where the picture has it, in the frame the pieces are placed in
    assert ld["scenery"] and ld["kind"] == "terrain" and ld["under_m"] == 0.3
    assert ld["at_xy"][0] - c1["at_xy"][0] == pytest.approx(3.0 - 1.0) and ld["at_xy"][1] - c1["at_xy"][1] == pytest.approx(20.0 - 4.0)
    req = P.generation_request(ld)
    assert [s["step"] for s in req["plan"]["steps"]] == ["spec", "shape"]
    assert req["plan"]["steps"][0]["spec"]["made_as"] == "relief" and req["plan"]["steps"][0]["spec"]["size_m"] == [80.0, 40.0, 20.0]
    # a place with nothing standing on it has its land too
    out, stage = PB.open_places([], dict(con, depth=dict(con["depth"], things={})))
    assert [p["id"] for p in out] == [PB.LAND_ID] and stage is None and out[0]["at_xy"] == [3.0, 20.0 + 5.0 - PB.OPEN_BARE_M]


def test_a_panorama_made_from_the_picture_keeps_its_land_on_every_side_but_ahead():
    h, w = 64, 128
    dirs = PS.pano_dirs(h, w)
    dz = dirs[..., 2]
    eye = 2.0
    ground = np.where(dz < -1e-3, eye / np.maximum(-dz, 1e-3), 1e9)
    hz = np.sqrt(np.maximum(1 - dz * dz, 1e-6))
    d = np.minimum(ground, 12.0 / hz)                       # a wall of land 12 m off all the way round
    sky = np.zeros((h, w), bool)
    out = PS.surround(d, sky, eye, 40.0)
    az = np.abs(np.degrees(np.arctan2(dirs[..., 0], dirs[..., 1])))
    wall = d * hz > 11.9
    # ahead (round +y, where the picture's own land is built) the panorama's is moved out to the far edge;
    # everywhere else it stands where the panorama has it
    assert (out[wall & (az < 39)] >= PS.LAND_MAX_K * PS.SKY_R_M - 1e-6).all()
    assert np.allclose(out[wall & (az > 41)], d[wall & (az > 41)])
    # the panorama's ground near the eye lies just under the real ground, on every side
    g = ~wall & (dz < -0.2)
    assert np.allclose(out[g] * -dz[g], eye + PS.GROUND_UNDER_M)


def test_a_place_with_its_picture_measured_has_its_panorama_made_from_the_picture(tmp_path):
    pic = tmp_path / "concept.png"
    pic.write_bytes(b"png")
    t = {"id": "t1", "kind": "terrain", "look": "open moor under a grey sky", "source": str(pic)}
    old = P.generation_request(dict(t))
    v = {"eye_at": [0.5, -8.0], "eye_h": 2.0, "front_deg": 48.0}
    req = P.generation_request(dict(t, view=v))
    draw = req["plan"]["steps"][0]
    assert draw["step"] == "draw" and "pano_from_picture.py" in " ".join(draw["argv"]) and str(pic) in draw["argv"]
    sp = next(s for s in req["plan"]["steps"] if s["step"] == "spec")["spec"]
    assert sp["view"] == v and sp["panorama_id"] == draw["ref_id"] and req["asset_id"] != old["asset_id"]
    # another eye is another world of the same panorama; another picture another panorama
    r2 = P.generation_request(dict(t, view=dict(v, eye_at=[0.0, -5.0])))
    assert r2["asset_id"] != req["asset_id"] and r2["plan"]["steps"][0]["ref_id"] == draw["ref_id"]
    pic.write_bytes(b"png2")
    assert P.generation_request(dict(t, view=v))["plan"]["steps"][0]["ref_id"] != draw["ref_id"]
    import shell_one as SO
    assert SO.view_args(sp) == ["--eye-h", "2.0", "--eye-at", "0.5", "-8.0", "--front-deg", "48.0"] and SO.view_args({}) == []
    # the picture's eye, from where the pieces are placed
    glb = tmp_path / "concept_land.glb"
    glb.write_bytes(b"glb")
    con = {"depth": {"eye": [0.0, -5.0, 2.2], "hfov_deg": 70.0,
                     "land": {"glb": str(glb), "xy": [3.0, 20.0], "size_m": [80.0, 40.0, 20.0], "has": [], "under_m": 0.3},
                     "things": {"seen:0": {"xy": [1.0, 4.0], "size_m": [2.0, 1.0, 2.0]}}}, "seen": [{"kind": "cart"}]}
    out, _ = PB.open_places([{"id": "c1", "kind": "cart"}], con)
    c1, ld = out
    assert ld["view"]["eye_h"] == 2.2 and ld["view"]["front_deg"] == 47.0
    # (the cart stands 9 m before the eye in the picture: so it does from the eye's place in the set)
    assert c1["at_xy"][0] - ld["view"]["eye_at"][0] == pytest.approx(1.0) and c1["at_xy"][1] - ld["view"]["eye_at"][1] == pytest.approx(9.0)


def test_the_panorama_is_turned_so_the_picture_looks_along_plus_y():
    from PIL import Image
    import pano_from_picture as PF
    a = np.zeros((4, 8, 3), np.uint8)
    a[:, 3:5] = 255                                           # the picture: the middle of what the model paints
    b = np.asarray(PF.turned(Image.fromarray(a)))
    assert (b[:, [0, 7]] == 255).all() and (b[:, 3:5] == 0).all()
    # ... which is where pano_shell's frame has +y
    assert PS.pano_dirs(64, 128)[32, 0][1] > 0.99


def test_people_and_cameras_keep_off_land_that_rises():
    import ground_map as GM
    # the measured picture: a bank 2 m high from x 6 to 10, y 10 to 30 (1 m cells from x0 -20, y0 0); eye at 0, -5
    z = [[2.0 if 26 <= j < 30 and 10 <= i < 30 else 0.0 for j in range(40)] for i in range(40)]
    hi = PB.land_high({"cell_m": 1.0, "x0": -20.0, "y0": 0.0, "z": z}, [0.0, -5.0], 1.0, 0.0, 9.0)
    # the set's middle is 9 m before the eye, at the picture's y 4: the bank lies 6 m and more beyond it
    assert GM.in_water(hi, 8.0, 10.0) and not GM.in_water(hi, 2.0, 10.0) and not GM.in_water(hi, 8.0, 2.0)
    tall = {"cell_m": hi["cell_m"], "half_m": hi["half_m"], "runs": hi["tall_runs"]}
    assert GM.in_water(tall, 8.0, 10.0) and not GM.in_water(tall, 0.0, 0.0)
    # a low rise bars people, not cameras
    low = PB.land_high({"cell_m": 1.0, "x0": -20.0, "y0": 0.0, "z": [[0.6 * (v > 0) for v in r] for r in z]}, [0.0, -5.0], 1.0, 0.0, 9.0)
    assert low["runs"] and not low["tall_runs"]
    # ground that rises nearly everywhere round the set's middle is the ground: it bars nobody
    assert PB.land_high({"cell_m": 1.0, "x0": -20.0, "y0": 0.0, "z": [[0.8] * 40 for _ in range(40)]}, [0.0, -5.0], 1.0, 0.0, 9.0) is None
    import from_plan as FP
    ts = FP.water_things({"high_land": hi})
    assert [t["name"] for t in ts] == ["_high_land"] and not FP.dry([8.0, 10.0], ts) and FP.dry([2.0, 10.0], ts)
    import speakers as SPK
    SPK.TALL_LAND.clear()
    SPK.TALL_LAND.update(tall)
    try:
        assert SPK.in_tall_land([8.0, 10.0, 1.6]) and not SPK.in_tall_land([0.0, 0.0, 1.6])
    finally:
        SPK.TALL_LAND.clear()


def test_a_hillside_in_the_picture_is_levelled_near_the_eye_and_what_stands_on_it_still_stands():
    pytest.importorskip("scipy")
    import concept_depth as CD
    h, w = 240, 400
    v, u = np.mgrid[0:h, 0:w]
    ray = np.stack([(u + .5) / w - .5, ((v + .5) / h - .5) * h / w, np.full((h, w), 0.6)], -1)
    # level ground on the left; on the right it rises a quarter of a metre a metre across; a rock face 20 m off
    den = ray[..., 1] + np.where(ray[..., 0] > 0, 0.25 * ray[..., 0], 0.0)
    ground = np.where(den > 1e-6, 2.0 / np.maximum(den, 1e-6), 1e9)
    t = np.minimum(ground, np.where(u > w * 0.6, 20.0 / 0.6, 1e9))
    pts, mask = ray * t[..., None], t < 80
    n, d, _ = CD.fit_floor(pts, mask & (u < 0.4 * w))
    vtx, faces, uv, _, _ = CD.land(pts, mask, CD.frame(n, d), d, lambda q: q, {}, {}, np.zeros((h, w, 3), np.uint8))
    f = vtx[uv[:, 0] <= 0.5]
    near = f[(f[:, 1] < 12) & (f[:, 0] > 2) & (f[:, 0] < 6)]          # (the slope stood 0.5 to 1.5 m high here)
    assert len(near) > 100 and near[:, 2].max() < 0.1
    assert f[(f[:, 1] > 15) & (f[:, 0] > 4)][:, 2].max() > 8.0
