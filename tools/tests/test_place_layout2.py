"""Soft rules, camera pockets, the recovery ladder and the front check of the room layout
(tools/assets/sets/place_cost.py, place_layout.py): synthetic rooms only, no engine, no model."""
import copy
import json
import math
import sys
from pathlib import Path

SETS = Path(__file__).resolve().parents[1] / "assets" / "sets"
sys.path.insert(0, str(SETS))
import place_cost as pc  # noqa: E402
import place_layout as pl  # noqa: E402

DATA = pl.load_data()
R = pc.load_rules()
BASE = [{"id": "o1", "kind": "fireplace", "wall": "north", "words": ""},
        {"id": "o2", "kind": "window", "wall": "west", "words": ""},
        {"id": "o3", "kind": "door", "wall": "east", "words": ""}]


def spec(kind="sitting_room", size="medium", openings=None, pieces=None, relations=None, named=None):
    return {"schema": "place_spec/1", "place": "a room", "kind": kind, "period": "not stated", "size": size,
            "openings": BASE if openings is None else openings, "pieces": pieces or [],
            "relations": relations or [], "named_places": named or {}}


def op(i, kind, wall="not stated"):
    return {"id": i, "kind": kind, "wall": wall, "words": ""}


def pc_(i, kind, phrases=(), against="not stated"):
    return {"id": i, "kind": kind, "look": "plain", "phrases": list(phrases), "against": against}


def rel(a, r, b, c=None, **kw):
    d = {"a": a, "rel": r, "b": b}
    if c:
        d["c"] = c
    d.update(kw)
    return d


def by_name(lay):
    return {p["name"]: p for p in lay["pieces"]}


def thing(i, kind, c, facing, size=(0.8, 0.8, 1.0), seat=False, **kw):
    h = pl.half_extents(list(size), facing)
    d = {"id": i, "name": i, "kind": kind, "c": c, "facing": facing, "rect": pl.rect_at(c, h),
         "size_m": list(size), "flat": False, "seat": seat, "zones": []}
    d.update(kw)
    return d


# ------------------------------------------------------------------ each cost term
def test_plateau_is_one_inside_its_band_and_falls_off_outside():
    assert pc.plateau(1.5, 1.2, 2.4) == 1.0
    assert 0 < pc.plateau(0.6, 1.2, 2.4) < 1 and 0 < pc.plateau(4.8, 1.2, 2.4) < 1
    assert pc.plateau(0.3, 1.2, 2.4) < pc.plateau(0.9, 1.2, 2.4)       # pulled toward the band from anywhere
    assert pc.plateau(0.1, 0.0, 0.3) == 1.0


def test_face_the_focal_point():
    assert pc.c_face((0, 1), (0, 0), (0, 3)) < 1e-9
    assert abs(pc.c_face((0, -1), (0, 0), (0, 3)) - 1.0) < 1e-9
    assert abs(pc.c_face((1, 0), (0, 0), (0, 3)) - 0.5) < 1e-9


def test_conversation_distance_and_angle():
    band = R["conversation"]["distance_m"]
    assert pc.c_conversation_distance(1.8, band) == 0.0
    assert pc.c_conversation_distance(0.4, band) > 0.5 and pc.c_conversation_distance(6.0, band) > 0.5
    assert pc.c_conversation_angle((1, 0), (0, 0), (-1, 0), (2, 0)) < 1e-9        # turned to each other
    assert abs(pc.c_conversation_angle((-1, 0), (0, 0), (1, 0), (2, 0)) - 1.0) < 1e-9   # back to back
    side_by_side = pc.c_conversation_angle((0, 1), (0, 0), (0, 1), (2, 0))
    assert 0.5 < side_by_side < 1.0


def test_reach_and_which_side_a_companion_stands():
    assert pc.c_reach(0.2, [0.0, 0.3]) == 0.0 and pc.c_reach(1.5, [0.0, 0.3]) > 0.9
    assert pc.c_where((1, 0), (0, 0), (0, -1), "side") == 0.0
    assert pc.c_where((0, -1), (0, 0), (0, -1), "side") == 1.0
    assert pc.c_where((0, -1), (0, 0), (0, -1), "front") == 0.0
    assert pc.c_where((0, 1), (0, 0), (0, -1), "front") == 1.0
    assert pc.c_where((0.6, 0.8), (0, 0), (0, -1), "head") == 0.0             # the end the front points away from
    assert pc.c_where((0.6, -0.8), (0, 0), (0, -1), "head") == 1.0


def test_clearance_alignment_wall_prior():
    zone = (0, 0, 1, 1)
    assert pc.c_clearance(zone, []) == 0.0
    assert abs(pc.c_clearance(zone, [(0.5, 0, 2, 1)]) - 0.5) < 1e-9
    assert pc.c_align(0.0) < 1e-9 and pc.c_align(math.pi / 2) < 1e-9
    assert abs(pc.c_align(math.pi / 4) - 1.0) < 1e-9
    assert pc.c_wall_prior(1.0, [0.9, 9.0]) == 0.0 and pc.c_wall_prior(0.1, [0.9, 9.0]) > 0.9


def test_symmetry_balance_empty_middle_bed_and_door():
    axis_pt, axis_dir = (0, 2), (0, -1)
    assert pc.c_symmetry((1, 0), [(-1, 0)], axis_pt, axis_dir, 0.6) < 1e-9
    assert pc.c_symmetry((0, 0), [], axis_pt, axis_dir, 0.6) < 1e-9          # on the axis itself
    assert pc.c_symmetry((1.5, 0), [(1.5, 1)], axis_pt, axis_dir, 0.6) > 0.8
    assert pc.c_balance([(-1, -1, 1, 1)], 5, 5) < 1e-9
    assert pc.c_balance([(1.5, 1.5, 2.5, 2.5)], 5, 5) > 0.9
    em = R["empty_middle"]
    hug = [(-2.5, -1, -1.7, 0), (1.7, -1, 2.5, 0), (-0.5, 1.7, 0.5, 2.5)]
    mid = [(-2.5, -1, -1.7, 0), (-0.4, -0.4, 0.4, 0.4), (0.9, -0.5, 1.7, 0.3)]
    assert pc.c_empty_middle(hug, 5, 5, em) == 1.0
    assert pc.c_empty_middle(mid, 5, 5, em) == 0.0
    assert pc.c_bed_door(0.5, 4, 4) > pc.c_bed_door(4.0, 4, 4)


def test_every_weight_names_a_term_the_code_scores_and_no_rule_names_a_story():
    lay = pl.solve(spec(pieces=[pc_("p1", "sofa"), pc_("p2", "armchair"), pc_("p3", "armchair"), pc_("p4", "bed"),
                                pc_("p5", "bedside_table"), pc_("p6", "bedside_table"), pc_("p7", "side_table")],
                        relations=[rel("p7", "by", "o2", inferred=True)], named={"the sofa": "p1"}))
    seen = set(lay["cost"]["terms"])
    assert seen <= set(R["weights"]), seen - set(R["weights"])
    assert {"face_focal", "conversation_distance", "conversation_angle", "reach", "clearance", "balance",
            "empty_middle", "bed_sides", "bed_door", "inferred", "camera_pocket"} <= seen
    text = json.dumps(R).lower()
    assert "/home" not in text and "~/" not in text


# ------------------------------------------------------------------ the camera pocket
CAM = R["camera"]


def test_a_face_toward_open_floor_has_a_pocket_and_one_at_a_wall_has_none():
    open_room = pc.pocket({"pos": (0.0, 1.5), "facing": (0.0, -1.0), "z": 1.6, "own": None}, [], 5.0, 5.0, CAM)
    assert all(r["valid"] == r["total"] for r in open_room.values())
    at_wall = pc.pocket({"pos": (0.0, 2.0), "facing": (0.0, 1.0), "z": 1.6, "own": None}, [], 5.0, 5.0, CAM)
    assert all(r["valid"] == 0 for r in at_wall.values())


def test_a_tall_piece_blocks_the_sight_line_a_low_one_and_the_persons_own_seat_do_not():
    p = {"pos": (0.0, 1.5), "facing": (0.0, -1.0), "z": 1.6, "own": "seat"}
    wall_of = [((-3.0, 0.8, 3.0, 1.0), 2.0, "screen")]
    assert all(r["valid"] == 0 for r in pc.pocket(p, wall_of, 6.0, 6.0, CAM).values())
    low = [((-3.0, 0.8, 3.0, 1.0), 0.7, "table")]
    assert all(r["valid"] > 0 for r in pc.pocket(p, low, 6.0, 6.0, CAM).values())
    own = [((-0.5, 1.0, 0.5, 2.0), 1.6, "seat")]
    assert all(r["valid"] > 0 for r in pc.pocket(p, own, 6.0, 6.0, CAM).values())


def test_pockets_are_reported_for_every_named_place_and_a_missing_one_is_named():
    s = spec(pieces=[pc_("p1", "armchair", ["the chair"]), pc_("p2", "desk", ["the desk"]), pc_("p3", "desk_chair")],
             named={"the chair": "p1", "the desk": "p2", "the door": "o3"})
    lay = pl.solve(s)
    assert lay["ok"], lay["unmet"]
    got = {k["place"]: k for k in lay["camera_pockets"]}
    assert set(got) == {"the chair", "the desk", "the door"}
    assert all(sh["valid"] > 0 for k in got.values() for sh in k["shots"].values())
    tight = copy.deepcopy(R)
    tight["camera"]["framings"] = {"medium": [40.0]}            # no room is 40 m deep
    bad = pl.solve(s, rules=tight)
    assert not bad["ok"] and any("no camera position for a medium shot" in u for u in bad["unmet"])


def test_a_person_at_a_desk_with_a_chair_sits_in_it_and_may_turn_from_the_desk():
    s = spec(pieces=[pc_("p1", "desk", ["the desk"], "opening:o2"), pc_("p2", "desk_chair")],
             named={"the desk": "p1"})
    lay = pl.solve(s)
    assert lay["ok"], lay["unmet"]
    n, k = by_name(lay), lay["camera_pockets"][0]
    assert pl.gap(n["desk"]["rect"], n["desk_chair"]["rect"]) <= 0.3
    assert math.hypot(k["position"][0] - n["desk_chair"]["position"][0],
                      k["position"][1] - n["desk_chair"]["position"][1]) < 1e-6
    f = n["desk"]["facing"]
    assert k["facing"][0] * f[0] + k["facing"][1] * f[1] > 0     # turned into the room, not at the wall


# ------------------------------------------------------------------ groups: slots, pairs, rugs
def test_a_desk_chair_stands_square_before_its_desk():
    lay = pl.solve(spec(pieces=[pc_("p1", "desk", against="opening:o2"), pc_("p2", "desk_chair")]))
    n = by_name(lay)
    d, c = n["desk"], n["desk_chair"]
    v = (c["position"][0] - d["position"][0], c["position"][1] - d["position"][1])
    along = v[0] * d["facing"][0] + v[1] * d["facing"][1]
    side = abs(v[0] * d["facing"][1] - v[1] * d["facing"][0])
    assert lay["ok"] and along > 0 and side < 0.15
    assert c["facing"][0] * d["facing"][0] + c["facing"][1] * d["facing"][1] < -0.9   # turned to the desk


def test_two_bedside_tables_stand_either_side_of_the_bed_at_its_head():
    lay = pl.solve(spec(kind="bedroom", size="large", openings=[op("o1", "window"), op("o2", "door")],
                        pieces=[pc_("p1", "bed"), pc_("p2", "bedside_table"), pc_("p3", "bedside_table")]))
    assert lay["ok"], lay["unmet"]
    n = by_name(lay)
    b, f = n["bed"], n["bed"]["facing"]
    sides = []
    for t in (n["bedside_table"], n["bedside_table_2"]):
        v = (t["position"][0] - b["position"][0], t["position"][1] - b["position"][1])
        assert v[0] * f[0] + v[1] * f[1] < 0                     # at the head end
        assert pl.gap(t["rect"], b["rect"]) <= 0.35
        sides.append(v[0] * f[1] - v[1] * f[0])
    assert sides[0] * sides[1] < 0
    assert lay["cost"]["terms"]["pair_sides"] == 0.0


def test_a_rug_lies_before_the_fire_between_the_seats_not_just_mid_floor():
    lay = pl.solve(spec(pieces=[pc_("p1", "armchair"), pc_("p2", "armchair"), pc_("p3", "rug")]))
    n = by_name(lay)
    r = n["rug"]["rect"]
    assert lay["ok"], lay["unmet"]
    assert pc.inter_area(r, n["fireplace"]["rect"]) == 0.0      # before the hearth, not under the grate
    assert n["rug"]["position"][1] > 0.2                         # drawn toward the fire on the north wall
    xs = sorted(a["position"][0] for a in (n["armchair"], n["armchair_2"]))
    assert xs[0] - 0.3 <= n["rug"]["position"][0] <= xs[1] + 0.3
    assert all(pl.gap(a["rect"], r) <= 0.8 for a in (n["armchair"], n["armchair_2"]))


# ------------------------------------------------------------------ lived-in, not arranged by rule
def lived_in_room():
    return spec(pieces=[pc_("p1", "sofa"), pc_("p2", "armchair", ["the chair by the fire"]),
                        pc_("p3", "wingback_armchair"), pc_("p4", "side_table"), pc_("p5", "bookcase"),
                        pc_("p6", "rug"), pc_("p7", "desk", against="opening:o2"), pc_("p8", "desk_chair")],
                named={"the chair by the fire": "p2", "the desk": "p7", "the door": "o3"})


def test_a_sitting_room_does_not_end_with_an_empty_middle_and_every_seat_on_a_wall():
    lay = pl.solve(lived_in_room())
    assert lay["ok"], lay["unmet"]
    W, D, _ = lay["size_m"]
    seats = [p for p in lay["pieces"] if p["seat"] and p["kind"] != "desk_chair"]
    off_wall = [p for p in seats if pc.wall_dist(p["rect"], W, D) > 0.3]
    assert len(off_wall) >= 2                                    # the armchairs are pulled into the room
    assert lay["cost"]["terms"]["empty_middle"] < 0.5
    fire = by_name(lay)["fireplace"]["position"]
    for p in off_wall:                                           # and they look at the fire, not at a wall
        d = pl.unit((fire[0] - p["position"][0], fire[1] - p["position"][1]))
        assert p["facing"][0] * d[0] + p["facing"][1] * d[1] > math.cos(math.radians(50))
    ds = [math.hypot(a["position"][0] - b["position"][0], a["position"][1] - b["position"][1])
          for i, a in enumerate(seats) for b in seats[i + 1:]]
    assert min(ds) <= 2.6                                        # close enough to talk


def test_the_search_is_the_beam_and_reports_each_pieces_cost():
    lay = pl.solve(lived_in_room())
    assert lay["search"] == "beam"
    assert all("cost" in p for p in lay["pieces"])
    assert any(p["cost"] for p in lay["pieces"])
    assert lay["cost"]["total"] == lay["cost"]["total"]          # a number


def test_same_inputs_give_the_same_room_with_and_without_the_ladder():
    s = lived_in_room()
    a, b = pl.solve(s, seed=2), pl.solve(json.loads(json.dumps(s)), seed=2)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c, d = pl.solve_recover(s, seed=2), pl.solve_recover(json.loads(json.dumps(s)), seed=2)
    assert json.dumps(c, sort_keys=True) == json.dumps(d, sort_keys=True)


def test_an_inferred_relation_is_a_wish_a_stated_one_is_a_rule():
    far = [pc_("p1", "bookcase", against="opening:o1"), pc_("p2", "stool")]
    wish = pl.solve(spec(size="large", openings=[op("o1", "fireplace", "north"), op("o2", "window", "south"),
                                                 op("o3", "door", "east")],
                         pieces=[pc_("p1", "stool")],
                         relations=[rel("p1", "beside", "o1"), rel("p1", "beside", "o2", inferred=True)]))
    assert wish["ok"] and wish["cost"]["terms"]["inferred"] == 1.0
    rule = pl.solve(spec(size="large", openings=[op("o1", "fireplace", "north"), op("o2", "window", "south"),
                                                 op("o3", "door", "east")],
                         pieces=[pc_("p1", "stool")],
                         relations=[rel("p1", "beside", "o1"), rel("p1", "beside", "o2")]))
    assert not rule["ok"]
    del far


# ------------------------------------------------------------------ the recovery ladder, rung by rung
def test_a_room_that_fits_takes_no_rung():
    lay = pl.solve_recover(lived_in_room())
    assert lay["ok"] and lay["status"] == "ok" and [s["step"] for s in lay["ladder"]] == ["as asked"]
    assert lay["dropped"] == []


def test_rung_one_shrinks_clearances():
    roomy = copy.deepcopy(R)
    roomy["hard"]["around_free_m"] = 6.0                         # every free piece wants 6 m of air: too much
    s = spec(pieces=[pc_("p1", "armchair", ["a chair"]), pc_("p2", "armchair", ["another chair"]),
                     pc_("p3", "stool", ["a stool"])])
    assert not pl.solve(s, rules=roomy)["ok"]
    lay = pl.solve_recover(s, rules=roomy)
    assert lay["ok"] and lay["status"] == "ok after recovery"
    assert [x["step"] for x in lay["ladder"]] == ["as asked", "clearances shrunk to their floor"]
    assert len(lay["pieces"]) == 4                               # three pieces and the fireplace: nothing dropped


def test_rung_two_drops_inferred_relations_and_says_how_many():
    s = spec(size="large", openings=[op("o1", "fireplace", "north"), op("o2", "window", "south"),
                                     op("o3", "door", "east")],
             pieces=[pc_("p1", "stool", ["the stool"])],
             relations=[rel("p1", "beside", "o1"), rel("p1", "beside", "o2"), rel("p1", "by", "o3", inferred=True)])
    lay = pl.solve_recover(s)
    step = next(x for x in lay["ladder"] if x["step"] == "inferred relations dropped")
    assert "1 relation(s) the text does not state" in step["gave_up"]


def test_rung_three_widens_stated_bands():
    tight = copy.deepcopy(DATA)
    tight["relations"]["beside"]["gap_m"] = 0.04                 # tighter than two pieces may stand
    wide = copy.deepcopy(R)
    wide["ladder"]["band_widen"] = 4.0
    s = spec(pieces=[pc_("p1", "bookcase", ["the bookcase"]), pc_("p2", "cabinet", ["the cabinet"])],
             relations=[rel("p2", "beside", "p1")])
    assert not pl.solve(s, data=tight)["ok"]
    lay = pl.solve_recover(s, data=tight, rules=wide)
    assert lay["ok"] and lay["ladder"][-1]["step"] == "stated relations' bands widened"
    n = by_name(lay)
    assert pl.gap(n["bookcase"]["rect"], n["cabinet"]["rect"]) <= 0.16 + 1e-6


def test_rung_four_steps_the_room_up_a_size_class():
    big = {"p1": {"asset": "a_long_table", "size_m": [3.6, 1.0, 0.8]}}
    s = spec(size="small", pieces=[pc_("p1", "table", ["the long table"])], named={"the long table": "p1"})
    assert not pl.solve(s, big)["ok"]                            # with its clear floor it does not go in a small room
    lay = pl.solve_recover(s, big)
    assert lay["ok"] and lay["status"] == "ok after recovery" and lay["size_class"] in ("medium", "large")
    assert any(x["step"] == "room stepped up one size class" and x["ok"] for x in lay["ladder"])
    assert by_name(lay)["table"]["asset"] == "a_long_table"


def test_rung_five_drops_unnamed_pieces_never_a_named_one():
    def hall(n):
        return spec(kind="hall", size="large", openings=[op("o1", "door", "south"), op("o2", "door", "north")],
                    pieces=[pc_("p0", "stool", ["the stool"])] + [pc_(f"p{i}", "wardrobe") for i in range(1, n + 1)],
                    named={"the stool": "p0"})
    keep = copy.deepcopy(R)                                      # so that only dropping can help
    keep["ladder"]["clearance_factor"], keep["ladder"]["clearance_floor_m"] = 1.0, 9.0
    n = next(k for k in range(4, 30) if not pl.solve(hall(k))["ok"])
    lay = pl.solve_recover(hall(n), rules=keep)
    assert lay["ok"] and lay["status"] == "ok after recovery", lay["unmet"]
    assert 1 <= len(lay["dropped"]) <= R["ladder"]["max_drops"]
    assert all(d["kind"] == "wardrobe" for d in lay["dropped"])
    assert "stool" in by_name(lay)
    assert any(x["step"] == "room stepped up one size class" and x.get("skipped") for x in lay["ladder"])


def test_an_over_constrained_room_ends_unresolved_with_the_failure_named_for_a_re_ask():
    s = spec(size="large", openings=[op("o1", "fireplace", "north"), op("o2", "window", "south"),
                                     op("o3", "door", "east")],
             pieces=[pc_("p1", "stool", ["the stool"]), pc_("p2", "ornament")],
             relations=[rel("p1", "beside", "o1"), rel("p1", "beside", "o2")], named={"the stool": "p1"})
    lay = pl.solve_recover(s)
    assert not lay["ok"] and lay["status"] == "unresolved"
    assert any(u.startswith("relation not met: stool beside") for u in lay["reask"])
    steps = [x["step"] for x in lay["ladder"]]
    assert steps[0] == "as asked" and "stated relations' bands widened" in steps
    assert all(d["id"] != "p1" for d in lay["dropped"])          # the named piece was never dropped
    assert all(not x["ok"] for x in lay["ladder"])               # and no rung is passed off as a room


# ------------------------------------------------------------------ the drapes hung at a window after the layout
def test_nothing_stands_in_a_windows_drapes_and_no_tall_piece_before_the_window():
    s = spec(pieces=[pc_("p1", "armchair", ["the chair by the window"]), pc_("p2", "desk", ["the desk"], "opening:o2"),
                     pc_("p3", "bookcase"), pc_("p4", "stool")],
             relations=[rel("p1", "by", "o2")], named={"the chair by the window": "p1", "the desk": "p2"})
    lay = pl.solve(s)
    assert lay["ok"], lay["unmet"]
    dr = R["hard"]["window_dressing"]
    for o in [o for o in lay["openings"] if o["kind"] == "window"]:
        side = (dr["width_m"] - o["width_m"]) / 2
        cols = [pl.span_rect(o["wall"], o["at_m"] + g * (o["width_m"] / 2 + side / 2), side / 2, dr["depth_m"],
                             *lay["size_m"][:2]) for g in (1, -1)]
        front = pl.span_rect(o["wall"], o["at_m"], o["width_m"] / 2, dr["depth_m"], *lay["size_m"][:2])
        for p in lay["pieces"]:
            if p.get("flat") or p.get("on"):
                continue
            assert not any(pl.overlap(p["rect"], z) for z in cols), p["name"]
            if p["size_m"][2] > o["sill_m"] + R["hard"]["window_tall_over_sill_m"]:
                assert not pl.overlap(p["rect"], front), p["name"]


def test_the_drape_rule_can_be_turned_off_and_a_window_keeps_room_for_its_drapes_on_the_wall():
    off = copy.deepcopy(R)
    off["hard"]["window_dressing"]["width_m"] = 0.0
    assert pl.solve(spec(pieces=[pc_("p1", "desk", against="opening:o2")]), rules=off)["ok"]
    lay = pl.solve(spec(size="small", openings=[op("o1", "window", "west")]))
    o = lay["openings"][0]
    L = lay["size_m"][1]
    assert abs(o["at_m"]) + R["hard"]["window_dressing"]["width_m"] / 2 <= L / 2 - 0.1 + 1e-6


# ------------------------------------------------------------------ fronts of library records
def test_a_seat_whose_open_side_is_not_minus_y_is_reported():
    rule = DATA["piece_kinds"]["armchair"]
    good = {"size_m": [0.85, 0.85, 1.0], "seat": {"front_asset": [0.0, -1.0]}}
    turned = {"size_m": [0.85, 0.85, 1.0], "seat": {"front_asset": [1.0, 0.0]}}
    assert pc.front_problems(good, rule, R) == []
    assert any("not -Y" in p for p in pc.front_problems(turned, rule, R))
    assert any("not verified" in p for p in pc.front_problems({"size_m": [0.85, 0.85, 1.0]}, rule, R))
    assert any("size_m" in p for p in pc.front_problems({"seat": {"front_asset": [0.0, -1.0]}}, rule, R))
    assert pc.front_problems({"size_m": [0.5, 0.5, 0.7]}, DATA["piece_kinds"]["side_table"], R) == []


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:                               # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
