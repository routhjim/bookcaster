"""A place's concept picture as a room (img3d/concept_depth.py room_from): the floor fitted to the picture's bottom,
the walls where the points at standing height end, the ceiling overhead, each thing on the floor where its box's
bottom edge meets it, against the wall its back comes near; and the layout keeps a piece where the picture has it."""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "assets" / "img3d"))
import concept_depth as CD  # noqa: E402

F, W_PX, H_PX = 0.7, 160, 100            # normalised focal (of the width), picture size
EYE_H, BACK, SIDE, CEIL = 1.5, 6.0, 2.5, 3.2


def cast(u, v):
    """The point a camera ray meets in a box room (camera coords: x right, y down, z forward; floor at y = EYE_H)."""
    r = np.array([(u - 0.5) / F, (v - 0.5) / (F * W_PX / H_PX), 1.0])
    ts = []
    if r[1] > 1e-6:
        ts.append(EYE_H / r[1])                          # floor
    if r[1] < -1e-6:
        ts.append(-(CEIL - EYE_H) / r[1])                # ceiling
    ts.append(BACK / r[2])                               # back wall
    if abs(r[0]) > 1e-6:
        ts.append((SIDE if r[0] > 0 else -SIDE) / r[0])  # side walls
    return r * min(t for t in ts if t > 0)


def room_points():
    pts = np.zeros((H_PX, W_PX, 3))
    for j in range(H_PX):
        for i in range(W_PX):
            pts[j, i] = cast((i + 0.5) / W_PX, (j + 0.5) / H_PX)
    intr = np.array([[F, 0, 0.5], [0, F * W_PX / H_PX, 0.5], [0, 0, 1]])
    return pts, np.ones((H_PX, W_PX), bool), intr


def box_of_floor_thing(x, z, w, h, intr):
    """The picture box of an upright thing standing on the floor at (x, z), w wide and h high."""
    def proj(p):
        return intr[0, 0] * p[0] / p[2] + 0.5, intr[1, 1] * p[1] / p[2] + 0.5
    u0, v1 = proj([x - w / 2, EYE_H, z])
    u1, _ = proj([x + w / 2, EYE_H, z])
    _, v0 = proj([x, EYE_H - h, z])
    return [u0, v0, u1, v1]


def test_the_room_and_a_thing_in_it_come_out_of_the_pictures_depth():
    pts, mask, intr = room_points()
    boxes = {"spec:p1": (box_of_floor_thing(-0.5, 3.0, 0.8, 0.9, intr), "table"),
             "seen:0": (box_of_floor_thing(0.0, BACK - 0.05, 1.0, 2.0, intr), "door")}
    r = CD.room_from(pts, mask, intr, boxes, {"table": [1.4, 0.9, 0.75], "door": [1.0, 0.1, 2.0]})
    W, D, H = r["room_m"]
    assert abs(r["scale"] - 1.0) < 0.1                         # the door is 2 m: the metres stand
    assert abs(W - 2 * SIDE) < 0.4 and abs(H - CEIL) < 0.3
    assert D >= BACK                                         # the back wall, and room behind the eye
    t = r["things"]["spec:p1"]
    assert abs(t["size_m"][0] - 0.8) < 0.15 and abs(t["size_m"][2] - 0.9) < 0.15
    eye = r["eye"]
    assert abs((t["xy"][0] - eye[0]) - (-0.5)) < 0.15                     # where the picture has it, from the eye
    assert abs((t["xy"][1] - eye[1]) - (3.0 + t["size_m"][1] / 2)) < 0.25
    assert t["against"] is None and abs(eye[2] - EYE_H) < 0.1
    assert r["things"]["seen:0"]["against"] == "north"


sys.path.insert(0, str(HERE.parents[0] / "assets" / "sets"))
import place_build as PB  # noqa: E402
import place_layout as PL  # noqa: E402


def concept_with_depth():
    return {"space": "room", "surfaces": {"ceiling": "Overhead: wooden beams, brown"},
            "spec": {"p1": {"in_picture": True, "look": "dark iron", "box": [0.2, 0.6, 0.4, 0.8]},
                     "p2": {"in_picture": True, "look": "grey stone", "box": [0.4, 0.5, 0.7, 0.9]}},
            "seen": [],
            "depth": {"room_m": [5.0, 6.0, 3.2], "things": {
                "spec:p1": {"xy": [-0.8, 0.9], "facing": [0.0, -1.0], "against": None, "size_m": [0.8, 0.4, 0.8]},
                "spec:p2": {"xy": [0.6, 2.6], "facing": [0.0, -1.0], "against": "north", "size_m": [1.6, 0.8, 1.0]}}}}


def place_spec():
    return {"schema": "place_spec/1", "place": "p", "kind": "shop", "period": "medieval", "size": "medium",
            "openings": [{"id": "o1", "kind": "door", "wall": "west", "words": ""}],
            "pieces": [{"id": "p1", "kind": "work_piece", "look": "block, a thing to work on", "phrases": ["at the block"],
                        "against": "free", "longest_m": 0.8},
                       {"id": "p2", "kind": "work_piece", "look": "hearth, a fire to heat in", "phrases": ["at the fire"],
                        "against": "free", "longest_m": 1.6}],
            "relations": [], "named_places": {"at the block": "p1", "at the fire": "p2"}}


def test_the_concepts_depth_gives_the_room_its_size_roof_and_each_pieces_place():
    sp = PB.with_concept(place_spec(), concept_with_depth())
    assert sp["_room_m"] == [5.0, 6.0, 3.2] and sp["_roof"] == "rafters"
    p1, p2 = (next(p for p in sp["pieces"] if p["id"] == i) for i in ("p1", "p2"))
    assert p1["at_xy"] == [-0.8, 0.9] and p1["against"] == "free"
    assert p2["at_xy"] == [0.6, 2.6] and p2["against"] == "wall"
    lay = PL.solve(sp, {"p1": {"size_m": [0.8, 0.4, 0.8]}, "p2": {"size_m": [1.6, 0.8, 1.0]}})
    assert lay["size_m"][:2] == [5.0, 6.0]
    got = {p["id"]: p for p in lay["pieces"]}
    assert math.dist(got["p1"]["position"][:2], [-0.8, 0.9]) < 1e-6
    assert math.dist(got["p2"]["position"][:2], [0.6, 2.6]) < 1e-6


def test_a_room_with_beams_overhead_gets_rafters_not_plaster():
    kit = PB.kit_for("medieval")
    ss = {"structure": [], "fixtures": [], "props": []}
    lay = {"size_m": [5.0, 6.0, 3.2], "openings": [], "pieces": []}
    out = PB.dress(ss, lay, {"_roof": "rafters"}, kit)
    kinds = [e["kind"] for e in out["structure"]]
    assert "trusses" in kinds and next(e for e in out["structure"] if e["kind"] == "ceiling")["material"] == "panel_wood"
    out2 = PB.dress(ss, lay, {}, kit)
    assert "trusses" not in [e["kind"] for e in out2["structure"]]


def test_floor_fit_of_a_real_pictures_point_count():
    """A real picture's floor is hundreds of thousands of points: the fit must not build a points x points matrix
    (11-17: a full SVD of them took 116 GB and the box's memory killer ended the session)."""
    rng = np.random.default_rng(3)
    h, w = 1000, 1500                                  # the bottom fifth: 300,000 points
    x, z = rng.uniform(-3, 3, (h, w)), rng.uniform(1, 8, (h, w))
    y = np.full((h, w), 1.5) + rng.normal(0, 0.004, (h, w))      # camera coords: y down, the floor 1.5 m under
    pts = np.stack([x, y, z], -1)
    n, d, fit = CD.fit_floor(pts, np.ones((h, w), bool))
    assert abs(abs(n[1]) - 1.0) < 1e-3 and abs(d - 1.5) < 0.01 and fit > 0.9


def test_a_thing_hung_on_the_wall_is_not_stood_on_the_floor():
    pts, mask, intr = room_points()
    fy = intr[1, 1]
    # a rack on the back wall, 1.0 to 1.4 m above the floor (under the eye: its box's foot ray meets the floor far
    # behind the wall its own points lie on)
    box = [0.55, 0.5 + fy * (EYE_H - 1.4) / BACK, 0.75, 0.5 + fy * (EYE_H - 1.0) / BACK]
    r = CD.room_from(pts, mask, intr, {"seen:0": (box, "fire_irons")}, {"fire_irons": [0.5, 0.3, 1.0]})
    assert "seen:0" not in r["things"] and "seen:0" in r["off_floor"]


def test_the_stump_under_the_anvil_is_one_piece_with_it():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import concept as CO
    assert CO.carried_by([0.237, 0.612, 0.404, 0.756], [0.255, 0.712, 0.39, 0.923])       # the forge, 11-17
    assert not CO.carried_by([0.4, 0.3, 0.5, 0.5], [0.2, 0.48, 0.9, 0.8])                 # a lamp on a long table
    assert not CO.carried_by([0.2, 0.6, 0.4, 0.75], [0.2, 0.6, 0.4, 0.75])                # the same thing seen twice


def test_a_worker_stands_at_the_piece_and_turned_to_it():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import from_plan as FP
    anvil = {"name": "anvil", "kind": "fixture", "center": [-0.5, 1.0, 0.45], "half": [0.3, 0.2, 0.45]}
    hearth = {"name": "hearth", "kind": "fixture", "center": [-0.5, 2.6, 0.5], "half": [0.9, 0.5, 0.5]}
    room = [4.4, 6.3, 2.5]
    wp = FP.work_place(anvil, [0.8, 0.2], [], [anvil, hearth], room)
    d = math.dist(wp["position"], anvil["center"][:2])
    assert d < 0.75                                              # a step off its face, not a metre away
    to = [anvil["center"][0] - wp["position"][0], anvil["center"][1] - wp["position"][1]]
    assert to[0] * wp["facing"][0] + to[1] * wp["facing"][1] > 0.99 * d      # turned to it
    assert wp["top_cm"] == 90.0 and 30 <= wp["fwd_cm"] <= 62
    # another person already at that face: the next free one
    wp2 = FP.work_place(anvil, [0.8, 0.2], [wp["position"]], [anvil, hearth], room)
    assert wp2 and math.dist(wp2["position"], wp["position"]) > 0.5
    # a piece with no height known: no work place (the person stays where the staging had them)
    assert FP.work_place({"name": "x", "kind": "fixture", "center": [0, 0], "half": [0.3, 0.3]}, [1, 1], [], [], room) is None


def test_the_piece_a_worker_works_at():
    import from_plan as FP
    anvil = {"name": "anvil", "kind": "fixture", "center": [-0.5, 1.0, 0.45], "half": [0.3, 0.2, 0.45]}
    rack = {"name": "rack", "kind": "fixture", "center": [2.0, 1.0, 1.0], "half": [0.3, 0.2, 1.0]}
    assert FP.work_thing({"position": [1.8, 1.0], "at": "front anvil"}, [rack, anvil]) is anvil
    assert FP.work_thing({"position": [0.2, 0.5], "at": "by the fire"}, [rack, anvil]) is anvil      # the nearest
    assert FP.work_thing({"position": [1.9, -2.5], "at": None}, [rack, anvil]) is None               # nothing in reach


def test_two_pieces_found_in_one_place_are_one_thing():
    import concept as CO
    assert CO.same_place([0.7, 0.656, 0.915, 0.978], [0.703, 0.656, 1.0, 0.997])        # the "sword" on the bench
    assert not CO.same_place([0.237, 0.612, 0.404, 0.923], [0.41, 0.573, 0.738, 0.867])  # the anvil beside the hearth


def test_a_hung_piece_goes_on_its_wall_and_takes_no_floor():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "sets"))
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "places"))
    import place_build as PB
    lay = {"size_m": [4.0, 6.0, 2.7], "pieces": [{"position": [0.5, 2.6], "wall": "north"}],
           "openings": [{"id": "o1", "kind": "door", "wall": "west", "point": [-2.0, 1.0], "normal": [1.0, 0.0],
                         "head_m": 2.0}]}
    hung = [{"id": "p4", "kind": "work_piece", "hangs_m": 1.5, "against": "opening:o1", "look": "x"},
            {"id": "p8", "kind": "work_piece", "hangs_m": 1.2, "against": "wall", "look": "y"}]
    rec = {"p4": {"asset": "a", "size_m": [1.0, 0.1, 0.2]}, "p8": {"asset": "b", "size_m": [0.8, 0.1, 0.6]}}
    PB.hang(lay, hung, rec)
    over, rack = lay["pieces"][-2:]
    assert over["hang"] and over["z_m"] == 2.1 and over["facing"] == [1.0, 0.0]        # over the door's head
    assert abs(over["position"][0] - (-2.0 + 0.06)) < 1e-6 and over["position"][1] == 1.0
    assert rack["hang"] and rack["z_m"] == 1.2
    assert math.dist(rack["position"], [0.5, 2.6]) > 1.0 and math.dist(rack["position"], [-2.0, 1.0]) > 1.0


def test_a_worker_stands_at_the_long_side_of_what_they_work_at():
    sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
    import from_plan as FP
    room = [6.0, 6.0, 3.0]
    long_x = {"name": "w", "kind": "fixture", "center": [0.0, 0.0, 0.43], "half": [0.5, 0.36, 0.43]}
    wp = FP.work_place(long_x, [1.5, 0.1], [], [long_x], room)      # they were nearer its end
    assert wp["facing"][0] == 0.0 and abs(wp["position"][1]) > 0.7 and wp["position"][0] == 0.0
    assert wp["top_cm"] == 86.0


def _thing_off_floor(z, lift, w, h):
    """The room's points with an upright thing w wide and h high whose bottom is `lift` above the floor at z."""
    pts, mask, intr = room_points()

    def proj(p):
        return intr[0, 0] * p[0] / p[2] + 0.5, intr[1, 1] * p[1] / p[2] + 0.5
    u0, v1 = proj([-w / 2, EYE_H - lift, z])
    u1, v0 = proj([w / 2, EYE_H - lift - h, z])
    for j in range(H_PX):
        for i in range(W_PX):
            u, v = (i + 0.5) / W_PX, (j + 0.5) / H_PX
            if u0 <= u <= u1 and v0 <= v <= v1:
                r = np.array([(u - 0.5) / F, (v - 0.5) / (F * W_PX / H_PX), 1.0])
                pts[j, i] = r * z
    return pts, mask, intr, [u0, v0, u1, v1]


def test_a_thing_standing_on_a_raised_part_of_the_floor_is_placed_where_it_stands():
    pts, mask, intr, box = _thing_off_floor(5.0, 0.4, 0.8, 1.4)
    r = CD.room_from(pts, mask, intr, {"spec:p1": (box, "chair")}, {"chair": [0.8, 0.8, 1.4]})
    assert "spec:p1" not in r["off_floor"]
    t = r["things"]["spec:p1"]
    assert abs(t["raised_m"] - 0.4) < 0.1
    assert abs(t["size_m"][0] - 0.8) < 0.2 and abs(t["size_m"][2] - 1.4) < 0.25
    assert abs((t["xy"][1] - r["eye"][1]) - 5.0) < 0.6          # at its own distance, not its foot's on the floor


def test_a_thing_high_on_the_wall_still_hangs():
    pts, mask, intr, box = _thing_off_floor(5.9, 1.2, 0.8, 0.25)
    r = CD.room_from(pts, mask, intr, {"seen:0": (box, "rack")}, {})
    assert "seen:0" in r["off_floor"] and "seen:0" not in r["things"]


def test_an_opening_and_roof_timbers_the_picture_shows_are_the_rooms():
    c = concept_with_depth()
    c["seen"] = list(c.get("seen") or []) + [
        {"name": "arched windows", "look": "stone, grey", "kind": "window", "box": [0.0, 0.0, 0.9, 0.4]},
        {"name": "more windows", "look": "", "kind": "window", "box": [0.1, 0.0, 0.5, 0.3]},
        {"name": "wooden beams", "look": "wood, brown", "kind": "table", "box": [0.2, 0.0, 0.7, 0.2]}]
    spec = dict(place_spec(), openings=[])
    sp = PB.with_concept(spec, c)
    assert [o["kind"] for o in sp["openings"]] == ["window"] and sp["openings"][0]["wall"] == "not stated"
    assert sp["_roof"] == "rafters"
    stated = dict(place_spec(), openings=[{"id": "o1", "kind": "window", "wall": "east", "words": ""}])
    assert len(PB.with_concept(stated, c)["openings"]) == 1 and PB.with_concept(stated, c)["openings"][0]["id"] == "o1"


def test_the_deeper_of_two_things_on_one_stretch_of_wall_stands_before_the_flatter():
    t = {"a": {"xy": [0.0, 4.5], "size_m": [0.9, 0.9, 1.5], "against": "north"},
         "b": {"xy": [0.1, 4.8], "size_m": [2.3, 0.3, 2.8], "against": "north"},
         "c": {"xy": [3.5, 4.6], "size_m": [1.0, 0.7, 1.0], "against": "north"}}
    CD.stand_before(t)
    assert t["b"]["xy"] == [0.1, 4.8] and t["c"]["xy"] == [3.5, 4.6]
    assert abs(t["a"]["xy"][1] - (4.8 - 0.15 - 0.45 - 0.05)) < 1e-6


def test_a_room_takes_every_thing_its_picture_shows():
    c = concept_with_depth()
    c["seen"] = [
        {"name": "barrel", "look": "wood", "kind": "work_piece", "box": [0.75, 0.6, 0.85, 0.8], "crop": "b1.png", "words": "A round cask with iron hoops"},
        {"name": "barrel", "look": "wood", "kind": "work_piece", "box": [0.86, 0.6, 0.96, 0.8], "crop": "b2.png"},
        {"name": "barrel again", "look": "wood", "kind": "work_piece", "box": [0.86, 0.61, 0.96, 0.8], "crop": "b3.png"},
        {"name": "rack of tools", "look": "iron", "kind": "work_piece", "box": [0.1, 0.1, 0.5, 0.3], "crop": "r.png"},
        {"name": "small thing", "look": "", "kind": None, "box": [0.5, 0.5, 0.52, 0.52]}]
    c["depth"]["things"].update({"seen:0": {"xy": [1.8, 0.5], "facing": [0.0, -1.0], "against": None, "size_m": [0.6, 0.6, 0.9]},
                                 "seen:1": {"xy": [2.1, 1.4], "facing": [0.0, -1.0], "against": None, "size_m": [0.6, 0.6, 0.9]}})
    c["depth"]["off_floor"] = {"seen:3": {"kind": "work_piece", "at_m": 2.9, "hangs_m": 1.4, "wall": "north",
                                          "xy": [-1.0, 2.9], "size_m": [1.8, 0.15, 0.6]}}
    sp = PB.with_concept(place_spec(), c)
    by = {p["id"]: p for p in sp["pieces"]}
    assert {"c1", "c2"} <= set(by) and "c3" not in by and "c5" not in by             # each once; nothing unkinded
    assert by["c1"]["at_xy"] == [1.8, 0.5] and by["c1"]["longest_m"] == 0.9 and by["c1"]["called"] == "round cask"
    assert "c4" not in by                         # (a thing on a wall is not built while ROOM_HUNG is off)
    PB.ROOM_HUNG = True
    try:
        by = {p["id"]: p for p in PB.with_concept(place_spec(), c)["pieces"]}
        assert by["c4"]["hangs_m"] == 1.4 and by["c4"]["against"] == "wall" and "at_xy" not in by["c4"]
    finally:
        PB.ROOM_HUNG = False


def test_a_thing_hung_on_a_wall_is_measured_where_it_hangs():
    pts, mask, intr, box = _thing_off_floor(5.9, 1.2, 0.8, 0.25)
    r = CD.room_from(pts, mask, intr, {"seen:0": (box, "rack")}, {})
    h = r["off_floor"]["seen:0"]
    assert abs(h["hangs_m"] - 1.2) < 0.15 and h["wall"] == "north"
    assert abs(h["size_m"][0] - 0.8) < 0.2 and abs(h["size_m"][2] - 0.25) < 0.12


def test_a_room_with_the_sun_at_its_window_is_lit_inside_all_the_same():
    out = {"size_m": [8.0, 10.0, 4.0], "structure": [{"kind": "ceiling"}],
           "lights": [{"type": "sun", "lux": 16000}, {"type": "sky", "intensity": 1.0}, {"type": "rect", "lumens": 1300}]}
    assert PB.room_light(out, {})
    added = [l for l in out["lights"] if l["type"] == "point"]
    assert sum(l["lumens"] for l in added) >= PB.ROOM_LM_M2 * 80 - 1300 - 1
    assert any(l["type"] == "sun" for l in out["lights"])            # the light from outside is kept


def test_a_fire_the_picture_shows_is_not_made_the_kinds_fireplace():
    c = concept_with_depth()
    c["seen"] = [{"name": "hearth", "look": "stone", "kind": "fireplace", "box": [0.3, 0.3, 0.6, 0.7]},
                 {"name": "door", "look": "wood", "kind": "door", "box": [0.8, 0.2, 0.95, 0.8]}]
    sp = PB.with_concept(dict(place_spec(), openings=[]), c)
    assert [o["kind"] for o in sp["openings"]] == ["door"]
