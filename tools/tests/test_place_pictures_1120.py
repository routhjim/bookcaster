"""The pieces a room takes from its concept picture, and fire pieces (tools/assets/sets/place_layout.py prepare,
keeps_off, work_spots; places/pieces.py seat_too_narrow): made-up rooms only, no engine, no model."""
import json
import math
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "assets" / "sets"))
sys.path.insert(0, str(TOOLS / "assets" / "places"))
sys.path.insert(1, str(TOOLS / "assets"))
import place_cost as pc  # noqa: E402
import place_layout as pl  # noqa: E402
import pieces as P  # noqa: E402

R = pc.load_rules()


def room(pieces, relations=(), named=None, size=(8.0, 14.0, 4.0), kind="hall", openings=None):
    return {"schema": "place_spec/1", "place": "a room", "kind": kind, "period": "not stated", "size": "large",
            "openings": openings if openings is not None else [{"id": "o1", "kind": "door", "wall": "south", "words": ""}],
            "pieces": list(pieces), "relations": list(relations), "named_places": dict(named or {}),
            "_room_m": list(size)}


def pc_(i, kind, look="plain", phrases=(), against="not stated", **kw):
    return dict({"id": i, "kind": kind, "look": look, "phrases": list(phrases), "against": against}, **kw)


def by_id(lay):
    return {p["id"]: p for p in lay["pieces"]}


def why(lay, pid):
    return [n for n in lay.get("refused") or [] if n["id"] == pid]


SEAT = pc_("p1", "chair", phrases=["in the tall chair"], at_xy=[0.0, 5.5], facing_xy=[0.0, -1.0])
TALK = pc_("p2", "chair", phrases=["on the far chair"], at_xy=[0.0, 0.5], facing_xy=[0.0, 1.0])
NAMED = {"in the tall chair": "p1", "on the far chair": "p2"}


def test_plural_names_repeat_and_singular_ones_do_not():
    assert pl.plural_count({"kind": "table", "look": "long tables; wood, brown"}, R) == R["picture"]["repeat_default"]
    assert pl.plural_count({"kind": "bench", "look": "benches; wood"}, R) == R["picture"]["repeat_default"]
    assert pl.plural_count({"kind": "table", "look": "three tables; oak"}, R) == 3
    assert pl.plural_count({"kind": "table", "look": "a long table; wood"}, R) == 1
    assert pl.plural_count({"kind": "rug", "look": "rugs; wool"}, R) == 1              # (a floor covering stays one)
    assert pl.plural_count({"kind": "curtain", "look": "curtains; red"}, R) == 1       # (a wall thing stays one)


def test_a_thing_the_picture_shows_several_of_stands_several_times_with_one_mesh():
    s = room([SEAT, TALK, pc_("c3", "table", "long tables; wood, brown", at_xy=[-2.5, -3.0], facing_xy=[0.0, -1.0])],
             named=NAMED)
    rec = {"c3": {"asset": "table_made_once", "size_m": [2.0, 0.9, 0.75]}}
    lay = pl.solve_recover(s, rec)
    tables = [p for p in lay["pieces"] if p["kind"] == "table"]
    assert lay["ok"] and len(tables) >= 3, (lay["unmet"], lay["refused"])
    assert {p["asset"] for p in tables} == {"table_made_once"}
    assert all(p.get("copy_of") == "c3" for p in tables if p["id"] != "c3")
    assert by_id(lay)["c3"]["position"] == [-2.5, -3.0]                               # (the first where it was seen)
    xs = {round(p["position"][0], 2) for p in tables}
    ys = sorted(p["position"][1] for p in tables if round(p["position"][0], 2) == -2.5)
    assert -2.5 in xs and all(b - a >= 0.9 + 0.99 for a, b in zip(ys, ys[1:]))    # (a row, a table and an aisle apart)


def test_a_picture_piece_the_depth_did_not_stand_on_the_floor_is_refused():
    s = room([SEAT, TALK, pc_("c7", "table", "beams; wood")], named=NAMED)            # (no at_xy: overhead)
    lay = pl.solve_recover(s, {})
    assert "c7" not in by_id(lay) and why(lay, "c7")[0]["what"] == "refused"
    raised = pc_("c8", "table", "a table; wood", at_xy=[2.0, -3.0], facing_xy=[0, -1], _raised_m=3.0)
    lay = pl.solve_recover(room([SEAT, TALK, raised], named=NAMED), {})
    assert "c8" not in by_id(lay) and why(lay, "c8")


def test_a_picture_piece_never_stands_before_a_seat_or_between_two_seats_the_plan_names():
    before = pc_("c4", "side_table", "a small table", at_xy=[0.0, 4.3], facing_xy=[0, -1])
    between = pc_("c5", "side_table", "a small table", at_xy=[0.2, 3.0], facing_xy=[0, -1])
    aside = pc_("c6", "side_table", "a small table", at_xy=[3.0, 3.0], facing_xy=[0, -1])
    lay = pl.solve_recover(room([SEAT, TALK, before, between, aside], named=NAMED), {})
    got = by_id(lay)
    assert lay["ok"] and "c4" not in got and "c5" not in got and "c6" in got, lay["refused"]
    # (an unpinned one is put elsewhere, never on that floor)
    loose = pc_("c9", "table", "a table; wood", at_xy=None)
    s = room([SEAT, TALK, dict(loose, at_xy=[0.0, 3.0], facing_xy=[0, -1])], named=NAMED)
    assert "c9" not in by_id(pl.solve_recover(s, {}))


def test_a_picture_piece_that_is_a_spec_piece_seen_again_is_that_piece():
    p = pc_("p3", "table", "plain table", phrases=["at the table"], at_xy=[2.0, -2.0], facing_xy=[0, -1])
    c = pc_("c2", "table", "table; wood", at_xy=[2.2, -2.1], facing_xy=[0, -1])
    lay = pl.solve_recover(room([SEAT, p, c], named={"in the tall chair": "p1", "at the table": "p3"}), {})
    tables = [x for x in lay["pieces"] if x["kind"] == "table"]
    assert [t["id"] for t in tables] == ["p3"] and why(lay, "c2")[0]["what"] == "merged"
    # a spec piece the picture does not place takes the place of its picture twin
    q = pc_("p4", "work_piece", "Loom, a frame for weaving", phrases=["at the loom"], longest_m=1.5)
    c = pc_("c5", "work_piece", "A wooden loom with a heavy frame", called="wooden loom", at_xy=[-2.0, -4.0],
            facing_xy=[0, -1])
    other = pc_("c6", "work_piece", "A barrel with hoops", called="barrel", at_xy=[2.5, -4.0], facing_xy=[0, -1])
    lay = pl.solve_recover(room([SEAT, q, c, other], named={"in the tall chair": "p1", "at the loom": "p4"}), {})
    got = by_id(lay)
    assert "c5" not in got and "c6" in got and math.dist(got["p4"]["position"], [-2.0, -4.0]) < 0.3
    # a picture name that ends with the spec kind's own word is that piece ("workbench" for a bench)
    assert pl.same_thing({"kind": "work_piece", "look": "x", "called": "wooden workbench", "at_xy": [0, 0]},
                         {"kind": "bench", "look": "sturdy"})
    assert not pl.same_thing({"kind": "work_piece", "look": "x", "called": "barrel", "at_xy": [0, 0]},
                             {"kind": "work_piece", "look": "Loom, a frame", "called": "loom"})


def test_a_deep_wall_hanging_behind_a_seat_does_not_hide_the_sitter():
    """(11-20: a made curtain measured 1.46 m deep was clamped off its wall onto the seat in front of it: no camera
    for the sitter, and the curtain was dropped.)"""
    hang = pc_("c8", "curtain", "curtains; red", against="wall", at_xy=[0.0, 6.9], facing_xy=[0, -1])
    seat = dict(SEAT, at_xy=[0.0, 6.0])
    lay = pl.solve_recover(room([seat, TALK, hang], named=NAMED), {"c8": {"asset": "c", "size_m": [2.8, 1.46, 2.8]}})
    got = by_id(lay)
    assert lay["status"] == "ok" and "c8" in got and not lay["dropped"], lay["ladder"]
    assert got["c8"]["size_m"][1] <= 2 * P.kinds()["curtain"]["size_m"][1] + 1e-9
    assert all(v["valid"] > 0 for pk in lay["camera_pockets"] for v in pk["shots"].values())


def fire_room(**kw):
    fire = pc_("p1", "work_piece", "Kiln, a stone oven for firing pots", phrases=["by the kiln"], against="free",
               gives_light=True, longest_m=1.6, at_xy=[0.8, 0.5], facing_xy=[0.0, -1.0])
    pump = pc_("p2", "work_piece", "Pump, a wooden air pump", phrases=["at the pump"], longest_m=0.9)
    work = pc_("p3", "work_piece", "Wheel, a potter's wheel at waist height", phrases=["at the wheel"],
               longest_m=0.8, at_xy=[0.9, 1.6], facing_xy=[0.0, -1.0])
    s = room([fire, pump, work], relations=[{"a": "p2", "rel": "beside", "b": "p1"}],
             named={"by the kiln": "p1", "at the pump": "p2", "at the wheel": "p3"}, size=(5.0, 7.0, 3.0), kind="hall")
    s.update(kw)
    rec = {"p1": {"asset": "k", "size_m": [1.8, 1.3, 1.5]}, "p2": {"asset": "a", "size_m": [0.7, 0.8, 1.0]},
           "p3": {"asset": "w", "size_m": [0.8, 0.6, 0.8]}}
    return s, rec


def test_a_fire_piece_stands_back_to_a_wall_with_what_serves_it_beside_it():
    s, rec = fire_room()
    lay = pl.solve_recover(s, rec)
    got = by_id(lay)
    assert lay["ok"], lay["unmet"]
    f, a = got["p1"], got["p2"]
    assert f.get("wall") and a.get("wall") == f["wall"], (f, a)
    n = pl.NORMAL[f["wall"]]
    assert f["facing"] == [n[0], n[1]]                                        # (its front into the room)
    assert f["wall"] == "north"                                               # (the wall its picture front turns from)
    assert pl.gap(pl.rect_at(f["position"], [x / 2 for x in (f["rect"][2] - f["rect"][0], f["rect"][3] - f["rect"][1])]),
                  a["rect"]) <= 0.5 + 1e-6


def test_the_piece_worked_at_stands_out_before_the_fire_with_floor_round_it():
    s, rec = fire_room()
    lay = pl.solve_recover(s, rec)
    got = by_id(lay)
    f, w = got["p1"], got["p3"]
    assert w["works_before"] == "p1" and not w.get("wall")
    front_f = f["rect"][1]                                                    # (the fire faces -y: its front edge)
    front_w = w["rect"][3]
    assert 1.8 - 1e-6 <= front_f - front_w <= 2.5 + 1e-6, (front_f, front_w)
    assert abs(w["position"][0] - f["position"][0]) <= 0.6 + 1e-6
    assert w["facing"] == [0.0, 1.0] or w["facing"] == [-0.0, 1.0]           # (its front to the fire)
    W, D = s["_room_m"][:2]
    r, cm = w["rect"], 1.5
    strips = [(r[0] - cm, r[1], r[0], r[3]), (r[2], r[1], r[2] + cm, r[3]), (r[0], r[1] - cm, r[2], r[1]),
              (r[0], r[3], r[2], r[3] + cm)]
    others = [p["rect"] for p in lay["pieces"] if p["id"] != "p3"]
    free = [z for z in strips if z[0] >= -W / 2 and z[2] <= W / 2 and z[1] >= -D / 2 and z[3] <= D / 2
            and not any(pl.overlap(z, o) for o in others)]
    assert len(free) >= 3


def test_a_lamp_that_gives_light_is_no_fire_piece():
    s = room([pc_("p1", "floor_lamp", "a standing lamp", phrases=["by the lamp"], gives_light=True)],
             named={"by the lamp": "p1"})
    ctx = pl.Ctx(s, {}, pl.load_data(), 0)
    assert ctx.fire == []


def test_a_room_without_picture_pieces_is_laid_out_as_before():
    s = room([SEAT, TALK], named=NAMED)
    sp, rec, notes = pl.prepare(s, {}, R)
    assert sp is s and notes == []


def rec_(seat):
    return {"asset_id": "chair_x", "tags": ["chair"], "description": "chair: wooden", "size_m": [0.6, 0.6, 1.0],
            "seat": seat, "ok": True, "checks": True, "used_in_sets": []}


def test_an_armed_seat_too_narrow_to_sit_in_is_never_a_seat_piece():
    narrow = rec_({"arm_z_m": 0.39, "width_m": 0.295, "inner_x_m": [-0.148, 0.147]})
    span = rec_({"arm_z_m": 0.39, "inner_x_m": [-0.15, 0.15]})
    wide = rec_({"arm_z_m": 0.39, "width_m": 0.52})
    armless = rec_({"arm_z_m": None, "width_m": 0.3})
    assert P.seat_too_narrow(narrow) and P.seat_too_narrow(span)
    assert not P.seat_too_narrow(wide) and not P.seat_too_narrow(armless)
    piece = {"kind": "chair", "look": "wooden"}
    assert P.lookup(piece, [narrow]) == [] and P.lookup(piece, [wide])
    # (also the one made from the piece's own picture: it is made again)
    src = dict(piece, id="p1", source="/nowhere/p1.png")
    want = P.generation_request(src)["asset_id"]
    res = P.resolve([src], [dict(narrow, asset_id=want)])
    assert res["pieces"][0]["status"] == "generate"


def test_no_story_words_in_the_new_rules():
    banned = ("throne", "dragon", "knight", "forge", "anvil", "king", "bellows", "hearth")
    text = (json.dumps(R.get("picture")) + json.dumps(R.get("fire"))).lower()
    assert not any(w in text for w in banned)
    me = Path(__file__).read_text().lower().split("def test_no_story_words")[0]
    assert not any(w in me for w in banned)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:                               # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
