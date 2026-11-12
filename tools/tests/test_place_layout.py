"""Room layout from a place spec (tools/assets/sets/place_layout.py): synthetic specs only, no engine, no model."""
import json
import math
import sys
from pathlib import Path

import pytest

SETS = Path(__file__).resolve().parents[1] / "assets" / "sets"
sys.path.insert(0, str(SETS))
import place_layout as pl  # noqa: E402

DATA = pl.load_data()


def spec(kind="sitting_room", size="medium", openings=None, pieces=None, relations=None, named=None):
    return {"schema": "place_spec/1", "place": "a room", "kind": kind, "period": "not stated", "size": size,
            "openings": openings or [], "pieces": pieces or [], "relations": relations or [],
            "named_places": named or {}}


def op(i, kind, wall="not stated"):
    return {"id": i, "kind": kind, "wall": wall, "words": ""}


def pc(i, kind, against="not stated", phrases=()):
    return {"id": i, "kind": kind, "look": "plain", "phrases": list(phrases), "against": against}


def by_name(lay):
    return {p["name"]: p for p in lay["pieces"]}


def opening(lay, name):
    return next(o for o in lay["openings"] if o["name"] == name)


def gap(a, b):
    return pl.gap(a["rect"], b["rect"])


# ------------------------------------------------------------------ openings
def test_fireplace_is_centred_on_its_stated_wall():
    lay = pl.solve(spec(openings=[op("o1", "fireplace", "east")]))
    f = opening(lay, "fireplace")
    assert lay["ok"] and f["wall"] == "east" and f["at_m"] == 0.0
    fx = by_name(lay)["fireplace"]
    assert fx["facing"] == [-1.0, 0.0]                      # into the room
    assert fx["rect"][2] <= f["breast"][0] + 1e-6           # in front of the chimney breast


def test_unstated_openings_fall_to_the_room_kinds_defaults():
    lay = pl.solve(spec(openings=[op("o1", "fireplace"), op("o2", "window"), op("o3", "door")]))
    rk = DATA["room_kinds"]["sitting_room"]
    want = {o["kind"]: o["wall"] for o in rk["openings"]}
    assert {o["kind"]: o["wall"] for o in lay["openings"]} == want
    assert opening(lay, "window")["wall"] in rk["outer_walls"]


def test_door_stands_clear_of_window_and_fireplace_on_one_wall():
    lay = pl.solve(spec(openings=[op("o1", "fireplace", "north"), op("o2", "door", "north")]))
    f, d = opening(lay, "fireplace"), opening(lay, "door")
    assert lay["ok"] and d["wall"] == "north"
    assert d["to_m"] < f["breast"][0] or d["from_m"] > f["breast"][2]
    lay = pl.solve(spec(kind="bedroom", openings=[op("o1", "window", "west"), op("o2", "door", "west")]))
    w, d = opening(lay, "window"), opening(lay, "door")
    margin = DATA["opening_margin_m"]
    assert d["to_m"] <= w["from_m"] - margin + 1e-6 or d["from_m"] >= w["to_m"] + margin - 1e-6


def test_an_unstated_door_avoids_the_fireplace_wall():
    lay = pl.solve(spec(kind="hall", openings=[op("o1", "fireplace", "south"), op("o2", "door")]))
    assert opening(lay, "door")["wall"] != "south"


def test_openings_that_cannot_fit_are_named():
    lay = pl.solve(spec(kind="hall", size="small",
                        openings=[op("o1", "fireplace", "north"), op("o2", "door", "north")]))
    assert not lay["ok"] and any("opening o2 (door)" in u and "north" in u for u in lay["unmet"])


# ------------------------------------------------------------------ pieces by kind rule
def test_wall_piece_stands_with_its_back_to_a_wall_facing_in():
    lay = pl.solve(spec(kind="study", openings=[op("o1", "door")], pieces=[pc("p1", "bookcase")]))
    b = by_name(lay)["bookcase"]
    W, D, _ = lay["size_m"]
    n = pl.NORMAL[b["wall"]]
    assert b["facing"] == [n[0], n[1]]
    back = {"north": D / 2 - b["rect"][3], "south": b["rect"][1] + D / 2, "east": W / 2 - b["rect"][2],
            "west": b["rect"][0] + W / 2}[b["wall"]]
    assert 0 <= back <= DATA["wall_gap_m"] + 1e-6


def test_desk_goes_under_the_window_by_its_kind_rule():
    lay = pl.solve(spec(kind="study", openings=[op("o1", "window"), op("o2", "door")], pieces=[pc("p1", "desk")]))
    d, w = by_name(lay)["desk"], opening(lay, "window")
    assert d["wall"] == w["wall"] and abs(d["position"][1] - w["at_m"]) < 0.3


def test_tall_piece_does_not_cover_a_window_and_nothing_blocks_a_door():
    lay = pl.solve(spec(kind="bedroom", size="small", openings=[op("o1", "window", "west"), op("o2", "door", "south")],
                        pieces=[pc("p1", "wardrobe"), pc("p2", "wardrobe"), pc("p3", "bed")]))
    assert lay["ok"], lay["unmet"]
    w, d = opening(lay, "window"), opening(lay, "door")
    for p in lay["pieces"]:
        if p.get("wall") == "west" and p["size_m"][2] > w["sill_m"] + 0.15:
            assert p["rect"][3] <= w["from_m"] + 1e-6 or p["rect"][1] >= w["to_m"] - 1e-6
        assert not pl.overlap(p["rect"], d["swing"])


def test_against_an_opening_puts_the_piece_at_that_opening():
    lay = pl.solve(spec(kind="bedroom", openings=[op("o1", "window", "north"), op("o2", "door")],
                        pieces=[pc("p1", "chest_of_drawers", "opening:o1")]))
    c, w = by_name(lay)["chest_of_drawers"], opening(lay, "window")
    assert lay["ok"] and c["wall"] == "north" and abs(c["position"][0] - w["at_m"]) <= w["width_m"] / 2 + c["size_m"][0] / 2


def test_fireside_armchair_faces_the_fire_from_one_side():
    lay = pl.solve(spec(openings=[op("o1", "fireplace"), op("o2", "door")], pieces=[pc("p1", "armchair")]))
    a, f = by_name(lay)["armchair"], by_name(lay)["fireplace"]
    v = pl.unit((f["position"][0] - a["position"][0], f["position"][1] - a["position"][1]))
    assert a["facing"][0] * v[0] + a["facing"][1] * v[1] > math.cos(math.radians(40))
    assert not pl.overlap(a["rect"], pl.rect_at(f["position"], (f["size_m"][0] / 2, f["size_m"][1] / 2 + 0.7)))


def test_set_item_names_are_the_kind_numbered_on_repeats():
    lay = pl.solve(spec(openings=[op("o1", "door")],
                        pieces=[pc("p1", "wingback_armchair"), pc("p2", "armchair"), pc("p3", "writing_desk")]))
    assert lay["names"] == {"o1": "door", "p1": "armchair", "p2": "armchair_2", "p3": "desk"}


def test_library_record_size_wins_over_the_kinds_default():
    s = spec(kind="study", openings=[op("o1", "door")], pieces=[pc("p1", "desk")])
    lay = pl.solve(s, {"p1": {"asset": "desk_x", "size_m": [2.0, 0.9, 0.8]}})
    d = by_name(lay)["desk"]
    assert d["asset"] == "desk_x" and d["size_m"] == [2.0, 0.9, 0.8] and not d["generate"]
    assert by_name(pl.solve(s))["desk"]["generate"]


# ------------------------------------------------------------------ stated relations
BASE = [op("o1", "fireplace"), op("o2", "window"), op("o3", "door")]


def rel(a, r, b, c=None):
    return {"a": a, "rel": r, "b": b, **({"c": c} if c else {})}


def test_relation_beside():
    lay = pl.solve(spec(openings=BASE, pieces=[pc("p1", "sofa"), pc("p2", "side_table")],
                        relations=[rel("p2", "beside", "p1")]))
    n = by_name(lay)
    assert lay["ok"] and gap(n["side_table"], n["sofa"]) <= DATA["relations"]["beside"]["gap_m"] + 1e-6


def test_relation_by_a_piece_and_by_an_opening():
    lay = pl.solve(spec(openings=BASE, pieces=[pc("p1", "armchair"), pc("p2", "bookcase"), pc("p3", "armchair")],
                        relations=[rel("p1", "by", "o2"), rel("p3", "by", "p2")]))
    n = by_name(lay)
    assert lay["ok"], lay["unmet"]
    w = opening(lay, "window")
    span = pl.span_rect(w["wall"], w["at_m"], w["width_m"] / 2, 0.05, *lay["size_m"][:2])
    assert pl.gap(n["armchair"]["rect"], span) <= DATA["relations"]["by"]["opening_gap_m"] + 1e-6
    assert gap(n["armchair_2"], n["bookcase"]) <= DATA["relations"]["by"]["gap_m"] + 1e-6


def test_relation_facing():
    lay = pl.solve(spec(openings=BASE, pieces=[pc("p1", "sofa"), pc("p2", "chair")],
                        relations=[rel("p2", "facing", "p1")]))
    n = by_name(lay)
    c, s = n["chair"], n["sofa"]
    v = pl.unit((s["position"][0] - c["position"][0], s["position"][1] - c["position"][1]))
    assert lay["ok"] and c["facing"][0] * v[0] + c["facing"][1] * v[1] >= math.cos(math.radians(50)) - 1e-6


def test_relation_in_front_of_and_behind():
    lay = pl.solve(spec(kind="study", openings=[op("o1", "window"), op("o2", "door")],
                        pieces=[pc("p1", "bureau"), pc("p2", "desk_chair"), pc("p3", "table"), pc("p4", "stool")],
                        relations=[rel("p2", "in_front_of", "p1"), rel("p4", "behind", "p3")]))
    n = by_name(lay)
    assert lay["ok"], lay["unmet"]
    b, c = n["bureau"], n["desk_chair"]
    v = (c["position"][0] - b["position"][0], c["position"][1] - b["position"][1])
    assert v[0] * b["facing"][0] + v[1] * b["facing"][1] > 0
    assert abs(v[0] * b["facing"][1] - v[1] * b["facing"][0]) < 0.2          # square before it
    t, s = n["table"], n["stool"]
    v = (s["position"][0] - t["position"][0], s["position"][1] - t["position"][1])
    assert v[0] * t["facing"][0] + v[1] * t["facing"][1] < 0


def test_relation_between():
    lay = pl.solve(spec(openings=BASE, pieces=[pc("p1", "armchair"), pc("p2", "armchair"), pc("p3", "side_table")],
                        relations=[rel("p3", "between", "p1", "p2")]))
    n = by_name(lay)
    assert lay["ok"], lay["unmet"]
    a, b, t = n["armchair"]["position"], n["armchair_2"]["position"], n["side_table"]["position"]
    dx, dy = b[0] - a[0], b[1] - a[1]
    k = ((t[0] - a[0]) * dx + (t[1] - a[1]) * dy) / (dx * dx + dy * dy)
    assert 0.15 <= k <= 0.85
    assert math.hypot(t[0] - (a[0] + k * dx), t[1] - (a[1] + k * dy)) <= DATA["relations"]["between"]["off_m"]


# ------------------------------------------------------------------ clearances and paths
def test_a_seats_front_stays_clear_and_a_desk_chair_may_sit_at_its_desk():
    lay = pl.solve(spec(kind="study", openings=[op("o1", "window"), op("o2", "door")],
                        pieces=[pc("p1", "sofa"), pc("p2", "desk"), pc("p3", "desk_chair"), pc("p4", "table"),
                                pc("p5", "bookcase")]))
    assert lay["ok"], lay["unmet"]
    n = by_name(lay)
    s = n["sofa"]
    f, clear = s["facing"], DATA["piece_kinds"]["sofa"]["front_clear_m"]
    ext = (s["rect"][2] - s["rect"][0]) / 2 if abs(f[0]) > 0.5 else (s["rect"][3] - s["rect"][1]) / 2
    zc = (s["position"][0] + f[0] * (ext + clear / 2), s["position"][1] + f[1] * (ext + clear / 2))
    zh = (clear / 2, (s["rect"][3] - s["rect"][1]) / 2) if abs(f[0]) > 0.5 else ((s["rect"][2] - s["rect"][0]) / 2,
                                                                                clear / 2)
    zone = pl.rect_at(zc, zh)
    assert not any(pl.overlap(p["rect"], zone) for p in lay["pieces"] if p["name"] != "sofa" and not p["flat"])
    assert gap(n["desk_chair"], n["desk"]) < 0.4            # in the desk's own clear floor, by the kind rule


def test_named_places_are_joined_by_clear_floor_and_the_checker_agrees():
    s = spec(kind="bedroom", openings=[op("o1", "window"), op("o2", "door")],
             pieces=[pc("p1", "bed"), pc("p2", "wardrobe"), pc("p3", "bedside_table"), pc("p4", "chair")],
             named={"the bed": "p1", "the window": "o1", "the door": "o2", "the chair": "p4"})
    lay = pl.solve(s)
    assert lay["ok"], lay["unmet"]
    assert pl.check_layout(lay, s) == []
    assert gap(by_name(lay)["bedside_table"], by_name(lay)["bed"]) <= 0.3     # the kind's usual companion


def test_a_bed_keeps_one_long_side_free():
    lay = pl.solve(spec(kind="bedroom", size="small", openings=[op("o1", "window"), op("o2", "door")],
                        pieces=[pc("p1", "bed"), pc("p2", "wardrobe"), pc("p3", "chest_of_drawers"),
                                pc("p4", "washstand")]))
    assert lay["ok"], lay["unmet"]
    b = by_name(lay)["bed"]
    W, D, _ = lay["size_m"]
    r, c = b["rect"], DATA["piece_kinds"]["bed"]["side_clear_m"]
    strips = ([(r[0], r[1] - c, r[2], r[1]), (r[0], r[3], r[2], r[3] + c)] if abs(b["facing"][0]) > 0.5
              else [(r[0] - c, r[1], r[0], r[3]), (r[2], r[1], r[2] + c, r[3])])
    others = [p["rect"] for p in lay["pieces"] if p["name"] != "bed" and not p["flat"]]
    assert any(z[0] >= -W / 2 - 1e-6 and z[2] <= W / 2 + 1e-6 and z[1] >= -D / 2 - 1e-6 and z[3] <= D / 2 + 1e-6
               and not any(pl.overlap(z, o) for o in others) for z in strips)


def test_checker_finds_overlap_a_blocked_door_and_a_cut_off_place():
    s = spec(kind="hall", size="small", openings=[op("o1", "door", "south")], pieces=[pc("p1", "bench")],
             named={"the bench": "p1", "the door": "o1"})
    lay = pl.solve(s)
    assert lay["ok"] and pl.check_layout(lay, s) == []
    W, D, _ = lay["size_m"]
    wall = {"id": "x", "name": "cabinet", "kind": "cabinet", "position": [0.0, 0.0], "facing": [0.0, -1.0],
            "size_m": [W, 0.5, 1.8], "rect": [-W / 2, -0.25, W / 2, 0.25], "flat": False, "seat": False}
    bench = by_name(lay)["bench"]
    far = dict(bench, position=[0.0, D / 2 - 0.3], rect=[-0.6, D / 2 - 0.5, 0.6, D / 2 - 0.1], facing=[0.0, -1.0])
    cut = dict(lay, pieces=[far, wall])
    assert any("no clear path" in p for p in pl.check_layout(cut, s))
    twice = dict(lay, pieces=[bench, dict(bench, name="bench_2", id="p9")])
    assert any("overlaps" in p for p in pl.check_layout(twice, s))
    d = opening(lay, "door")
    c = pl.centre(d["swing"])
    inswing = dict(bench, position=list(c), rect=[c[0] - 0.2, c[1] - 0.2, c[0] + 0.2, c[1] + 0.2])
    assert any("swing" in p for p in pl.check_layout(dict(lay, pieces=[inswing]), s))


# ------------------------------------------------------------------ failure is named, the answer is repeatable
def test_too_many_pieces_for_the_room_names_the_piece_without_a_place():
    lay = pl.solve(spec(kind="hall", size="small", openings=[op("o1", "door", "south"), op("o2", "door", "north")],
                        pieces=[pc(f"p{i}", "wardrobe") for i in range(1, 12)]))
    assert not lay["ok"]
    assert any(u.startswith("piece wardrobe") and "no free position" in u for u in lay["unmet"])


def test_relations_that_contradict_each_other_are_named():
    lay = pl.solve(spec(size="large", openings=[op("o1", "fireplace", "north"), op("o2", "window", "south"),
                                                op("o3", "door", "east")],
                        pieces=[pc("p1", "stool")],
                        relations=[rel("p1", "beside", "o1"), rel("p1", "beside", "o2")]))
    assert not lay["ok"]
    assert any(u.startswith("relation not met: stool beside") for u in lay["unmet"])


def test_named_place_that_is_not_in_the_room_is_named():
    lay = pl.solve(spec(openings=[op("o1", "door")], named={"the piano": "p9"}))
    assert not lay["ok"] and any("the piano" in u for u in lay["unmet"])


def test_same_inputs_and_seed_give_the_same_layout():
    s = spec(openings=BASE, pieces=[pc("p1", "sofa"), pc("p2", "armchair"), pc("p3", "armchair"), pc("p4", "bookcase"),
                                    pc("p5", "side_table"), pc("p6", "rug")], relations=[rel("p5", "beside", "p1")])
    a, b = pl.solve(s, seed=3), pl.solve(json.loads(json.dumps(s)), seed=3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert pl.solve(s, seed=4)["ok"]


def test_unknown_room_kind_is_refused():
    with pytest.raises(ValueError):
        pl.solve(spec(kind="ballroom_of_dreams"))


# ------------------------------------------------------------------ set spec and named places
def sitting_room():
    """A spec with the piece list of a full town-house sitting room (kinds and sizes only)."""
    pieces = [pc("p1", "wingback_armchair", phrases=["the chair by the window"]), pc("p2", "bureau", "wall"),
              pc("p3", "desk_chair"), pc("p4", "bookcase"), pc("p5", "bookcase"), pc("p6", "side_table"),
              pc("p7", "desk", "opening:o2", ["the desk"]), pc("p8", "sofa", phrases=["the sofa"]),
              pc("p9", "armchair", phrases=["the chair by the fire"]), pc("p10", "side_table"),
              pc("p11", "side_table"), pc("p12", "rug"), pc("p13", "lamp")]
    s = spec(openings=BASE, pieces=pieces,
             relations=[rel("p1", "by", "o2"), rel("p9", "by", "o1"), rel("p3", "in_front_of", "p2"),
                        rel("p6", "beside", "p1"), rel("p13", "by", "p6")],
             named={"the chair by the window": "p1", "the chair by the fire": "p9", "the desk": "p7",
                    "the sofa": "p8", "the fire": "o1", "the window": "o2", "the door": "o3"})
    records = {"o1": {"asset": "fireplace_a", "size_m": [1.66, 0.86, 1.25]},
               "p1": {"asset": "chair_a", "size_m": [0.8, 0.85, 1.1]}, "p2": {"asset": "bureau_a", "size_m": [0.9, 0.5, 1.1]},
               "p3": {"asset": "chair_b", "size_m": [0.5, 0.5, 0.95]}, "p4": {"asset": "bookcase_a", "size_m": [1.0, 0.38, 2.0]},
               "p5": {"asset": "bookcase_a", "size_m": [1.0, 0.38, 2.0]}, "p6": {"asset": "table_a", "size_m": [0.5, 0.5, 0.7]},
               "p7": {"asset": "desk_a", "size_m": [1.4, 0.75, 0.9]}, "p8": {"asset": "sofa_a", "size_m": [2.0, 0.9, 0.85]},
               "p9": {"asset": "chair_c", "size_m": [0.85, 0.85, 1.0]}, "p10": {"asset": "table_a", "size_m": [0.5, 0.5, 0.7]},
               "p11": {"asset": "table_a", "size_m": [0.5, 0.5, 0.7]}, "p12": {"asset": "rug_a", "size_m": [4.3, 3.5, 0.01]},
               "p13": {"asset": "lamp_a", "size_m": [0.2, 0.2, 0.5]}}
    return s, records


def test_full_sitting_room_lays_out_with_every_named_place_resolved_and_reachable():
    s, records = sitting_room()
    lay = pl.solve(s, records)
    assert lay["ok"], lay["unmet"]
    assert pl.check_layout(lay, s) == []
    named = pl.named_places(s, lay)
    assert {k: v["item"] for k, v in named.items()} == {
        "the chair by the window": "armchair", "the chair by the fire": "armchair_2", "the desk": "desk",
        "the sofa": "sofa", "the fire": "fireplace", "the window": "window", "the door": "door"}
    assert all(len(v["stand"]) == 2 for v in named.values())
    n = by_name(lay)
    assert n["desk"]["wall"] == opening(lay, "window")["wall"]
    assert n["lamp"]["on"] == "side_table" and n["lamp"]["on_z_m"] == 0.7


def test_set_spec_has_todays_shape_and_passes_the_set_builders_own_plan_check(tmp_path):
    import layout as set_layout                     # the set builder's expander and plan check, same folder
    s, records = sitting_room()
    lay = pl.solve(s, records)
    part = pl.to_set_spec(s, lay)
    assert part["size_m"] == lay["size_m"] and part["to_generate"] == []
    walls = {w["side"]: w for w in part["structure"] if w["kind"] == "wall"}
    assert set(walls) == {"north", "south", "east", "west"}
    win = walls[opening(lay, "window")["wall"]]["openings"][0]
    assert win["kind"] == "window_band" and win["from_m"] < win["to_m"] and win["sill_m"] > 0
    assert any(b["kind"] == "box" and b["name"] == "chimney_breast" for b in part["structure"])
    fire = next(f for f in part["fixtures"] if f["name"] == "fireplace")
    assert fire["asset"] == "fireplace_a" and set(fire) >= {"asset", "name", "position", "facing"}
    assets = {}
    for i, r in records.items():
        w, d, h = r["size_m"]
        assets[r["asset"]] = {"bounds_cm": {"min": [-w * 50, -d * 50, 0], "max": [w * 50, d * 50, h * 100]}}
    placed = []
    for item in part["fixtures"] + part["props"]:
        z = item.get("on_z_m", 0.0)
        placed.append(dict(item, position=[item["position"][0], item["position"][1], z]))
    full = {"size_m": part["size_m"], "structure": part["structure"], "fixtures": part["fixtures"],
            "props": part["props"]}
    assert set_layout.check(full, assets, placed) == []
    out = tmp_path / "named.json"
    pl.write_named_places(out, s, lay)
    assert json.loads(out.read_text())["the desk"]["item"] == "desk"


def test_pieces_without_a_library_record_are_listed_for_generation():
    s = spec(kind="bedroom", openings=[op("o1", "window"), op("o2", "door")], pieces=[pc("p1", "bed"), pc("p2", "washstand")])
    part = pl.to_set_spec(s, pl.solve(s, {"p1": {"asset": "bed_a", "size_m": [1.0, 2.0, 1.1]}}))
    assert [g["name"] for g in part["to_generate"]] == ["washstand"]
    assert by_name(pl.solve(s))["bed"]["support"] == "lie"


def test_bedroom_and_study_default_rooms_lay_out():
    for kind in ("bedroom", "study"):
        dressing = DATA["room_kinds"][kind]["dressing"]
        s = spec(kind=kind, openings=[op(f"o{i}", o["kind"]) for i, o in enumerate(DATA["room_kinds"][kind]["openings"])],
                 pieces=[pc(f"p{i}", k) for i, k in enumerate(dressing)])
        for size in ("small", "medium", "large"):
            lay = pl.solve(dict(s, size=size))
            assert lay["ok"], (kind, size, lay["unmet"])
        assert pl.missing_dressing(s) == []
    assert "bed" in pl.missing_dressing(spec(kind="bedroom"))


def test_data_file_names_no_story_and_every_default_is_usable():
    for kind, rk in DATA["room_kinds"].items():
        assert set(rk["sizes_m"]) == {"small", "medium", "large"}
        for o in rk["openings"]:
            assert o["kind"] in DATA["openings"] and o["wall"] in pl.NORMAL
        for k in rk["dressing"]:
            assert k in DATA["piece_kinds"], (kind, k)
    for k, rule in DATA["piece_kinds"].items():
        assert rule["place"] in ("wall", "free", "floor", "on", "fireplace") and len(rule["size_m"]) == 3, k
