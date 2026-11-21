"""What a place's picture shows comes through to its set (scenes/concept.py count and sides; sets/place_build.py
with_concept, with_dressing; sets/place_layout.py plural_count, prepare, seats at tables): many of a kind stand as
many, in the arrangement the picture shows; a thing is its named kind; a seat the plan names and the picture's same
seat are one piece; each dressing item is put somewhere and said where. Made-up rooms only, no engine, no model."""
import json
import math
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "assets" / "sets"))
sys.path.insert(0, str(TOOLS / "assets" / "places"))
sys.path.insert(0, str(TOOLS / "assets" / "scenes"))
sys.path.insert(1, str(TOOLS / "assets"))
import place_cost as pc  # noqa: E402
import place_layout as pl  # noqa: E402
import place_build as PB  # noqa: E402
import concept as CO  # noqa: E402

R = pc.load_rules()


def room(pieces, relations=(), named=None, size=(10.0, 20.0, 5.0), kind="dining_room"):
    return {"schema": "place_spec/1", "place": "a room", "kind": kind, "period": "not stated", "size": "large",
            "openings": [{"id": "o1", "kind": "door", "wall": "south", "words": ""}],
            "pieces": list(pieces), "relations": list(relations), "named_places": dict(named or {}),
            "_room_m": list(size)}


def pc_(i, kind, look="plain", phrases=(), against="not stated", **kw):
    return dict({"id": i, "kind": kind, "look": look, "phrases": list(phrases), "against": against}, **kw)


SEAT = pc_("p1", "chair", phrases=["in the tall chair"], at_xy=[0.0, 9.0], facing_xy=[0.0, -1.0])
NAMED = {"in the tall chair": "p1"}


def by_id(lay):
    return {p["id"]: p for p in lay["pieces"]}


# ---------------------------------------------------------------- how many: the count, the name's plural
def test_the_readers_count_wins_and_a_plural_name_repeats_whatever_its_kind():
    assert pl.plural_count({"kind": "table", "look": "tables; oak", "_count": 6}, R) == 6
    assert pl.plural_count({"kind": "table", "look": "a table; oak", "_count": 3}, R) == 3
    assert pl.plural_count({"kind": "table", "look": "x", "_count": 99}, R) == R["picture"]["repeat_most"]
    # (a name the judge filed under another kind: its own last word in the plural)
    assert pl.plural_count({"kind": "bench", "look": "long dining tables; wood"}, R) == R["picture"]["repeat_default"]
    assert pl.plural_count({"kind": "cabinet", "look": "brass; metal"}, R) == 1               # (no plural)
    assert pl.plural_count({"kind": "rug", "look": "rugs", "_count": 3}, R) == 1               # (a floor covering)
    assert pl.plural_count({"kind": "work_piece", "look": "stands", "_count": 3}, R) == 3      # (counted: any kind)
    assert pl.plural_count({"kind": "work_piece", "look": "stands"}, R) == 1                   # (any kind: counted only)
    assert pl.noun_stem("tables") == "table" and pl.noun_stem("benches") == "bench"
    assert pl.noun_stem("tapestries") == "tapestry" and pl.noun_stem("glass") == "glass"


def test_a_readers_count_parses_and_is_asked_in_a_room_only():
    assert CO.count_of("12") == 12 and CO.count_of("about four of them") == 4 and CO.count_of("none") is None
    assert CO.count_of("400") is None
    seen_asks = []

    def ask(items):
        out = {}
        for it in items:
            seen_asks.append(it)
            if it["id"].startswith("s:"):
                out[it["id"]] = {"box": "[100, 500, 400, 900]", "kind": "table", "count": "6", "sides": "Both."}
            else:
                out[it["id"]] = {}
        return out
    with tempfile.TemporaryDirectory() as td:
        from PIL import Image
        pic = Path(td) / "c.png"
        Image.new("RGB", (64, 36), (90, 80, 70)).save(pic)
        res = {"space": "room", "spec": {}, "seen": [{"name": "long tables", "look": "oak"}]}
        CO.pieces_from(pic, {"pieces": []}, res, ask, Path(td) / "crops", "t", ["table", "bench", "work_piece"])
    sn = res["seen"][0]
    assert sn["kind"] == "table" and sn["count"] == 6 and sn["sides"] == "both"
    s_ask = next(it for it in seen_asks if it["id"] == "s:0")["ask"]
    assert "count" in s_ask and "sides" in s_ask


def test_a_base_is_never_looked_for_in_the_picture():
    asked = []

    def ask(items):
        asked.extend(it["id"] for it in items)
        return {}
    spec = {"pieces": [pc_("p1", "chair", phrases=["in the tall chair"]), pc_("p2", "bench", "a low platform")],
            "relations": [{"a": "p1", "rel": "on", "b": "p2"}]}
    res = CO.read(Path("/nowhere.png"), spec, ask)
    assert "p:p1" in asked and "p:p2" not in asked and "p2" not in res["spec"]


# ---------------------------------------------------------------- the reader's name, its kind
def test_a_thing_is_the_kind_its_name_says():
    assert PB.seen_kind({"name": "long dining tables", "kind": "bench"}) == "table"
    assert PB.seen_kind({"name": "bench seats", "kind": "bench"}) == "bench"        # (its own kind's word in it)
    assert PB.seen_kind({"name": "wooden sideboard", "kind": "table"}) == "cabinet"
    assert PB.seen_kind({"name": "tall carved chair", "kind": "chair"}) == "chair"
    assert PB.seen_kind({"name": "stone walls", "kind": None}) is None              # (no kind: none given)


def concept_of(seen, things, spec_chk=None, off=None):
    return {"concept": "/c/concept.png", "space": "room", "seats_abreast": 1, "surfaces": {}, "seen": seen,
            "spec": spec_chk or {},
            "depth": {"room_m": [10.0, 20.0, 5.0], "things": things, "off_floor": off or {}, "faces": []}}


def test_seats_drawn_at_their_tables_are_built_with_them_and_a_base_swallows_nothing():
    spec = {"place": "a room", "kind": "dining_room", "openings": [], "named_places": {"in the tall chair": "p1"},
            "pieces": [pc_("p1", "chair", "carved high-backed chair", phrases=["in the tall chair"]),
                       pc_("p2", "bench", "a low platform", phrases=["before the platform"])],
            "relations": [{"a": "p1", "rel": "on", "b": "p2"}]}
    seen = [{"name": "long tables", "look": "oak", "kind": "bench", "box": [0.1, 0.6, 0.5, 0.9], "crop": "/c/s0.png"},
            {"name": "benches", "look": "pine", "kind": "bench", "box": [0.12, 0.65, 0.48, 0.95], "crop": "/c/s1.png"},
            {"name": "settles", "look": "ash", "kind": "bench", "box": [0.55, 0.6, 0.9, 0.9], "crop": "/c/s2.png"}]
    # (the reader "found" the platform where the right-hand seats are)
    chk = {"p1": {"in_picture": False, "look": ""},
           "p2": {"in_picture": True, "look": "wood", "box": [0.55, 0.6, 0.9, 0.9], "crop": "/c/p2.png"}}
    things = {"seen:0": {"xy": [-2.0, -3.0], "size_m": [2.0, 0.8, 0.75], "facing": [0, -1], "against": None},
              "seen:1": {"xy": [-2.0, -2.0], "size_m": [2.0, 0.4, 0.45], "facing": [0, -1], "against": None},
              "seen:2": {"xy": [2.5, -3.0], "size_m": [2.0, 0.4, 0.45], "facing": [0, -1], "against": None}}
    sp = PB.with_concept(spec, concept_of(seen, things, chk))
    by = {p["id"]: p for p in sp["pieces"]}
    assert by["c1"]["kind"] == "table"                                     # (named tables: a table)
    assert "c2" in by and by["c2"]["kind"] == "bench"                      # (seats at the tables: built too)
    assert "c3" in by and by["c3"]["kind"] == "bench"                      # (not folded into the base)
    assert "p2" in by


# ---------------------------------------------------------------- one seat, not two
def test_the_plans_seat_and_the_pictures_same_seat_are_one_piece():
    spec = {"place": "a room", "kind": "dining_room", "openings": [], "relations": [],
            "named_places": {"in the tall chair": "p1"},
            "pieces": [pc_("p1", "chair", "carved high-backed chair with red cushions", phrases=["in the tall chair"])]}
    seen = [{"name": "high chair", "look": "gilded wood", "kind": "chair", "box": [0.4, 0.4, 0.5, 0.6],
             "crop": "/c/s0.png"}]
    things = {"seen:0": {"xy": [0.5, 8.0], "size_m": [1.1, 1.1, 2.0], "facing": [0, -1], "against": "north",
                         "raised_m": 0.6}}
    sp = PB.with_concept(spec, concept_of(seen, things, {"p1": {"in_picture": False, "look": ""}}))
    assert [p["id"] for p in sp["pieces"]] == ["p1"]
    p = sp["pieces"][0]
    assert p["phrases"] == ["in the tall chair"] and p["source"] == "/c/s0.png"
    assert p["at_xy"] == [0.5, 8.0] and p["_seen_size_m"] == [1.1, 1.1, 2.0] and p["_raised_m"] == 0.6
    assert "red cushions" in p["look"] and "gilded wood" in p["look"]       # (its words and the picture's)
    assert sp["_picture_merged"] == [{"id": "c1", "into": "p1"}]


def test_the_directors_look_is_kept_with_the_pictures():
    spec = {"place": "a room", "kind": "dining_room", "openings": [], "relations": [], "named_places": {"at the desk": "p1"},
            "pieces": [pc_("p1", "desk", "a broad desk with brass handles", phrases=["at the desk"])]}
    con = concept_of([], {}, {"p1": {"in_picture": True, "look": "dark walnut", "box": [0.3, 0.5, 0.6, 0.8],
                                     "crop": "/c/p1.png"}})
    p = PB.with_concept(spec, con)["pieces"][0]
    assert "brass handles" in p["look"] and "dark walnut" in p["look"]


def test_what_is_on_a_wall_in_the_picture_is_not_made_and_a_tall_table_top_kind_stands():
    spec = {"place": "a room", "kind": "dining_room", "openings": [], "relations": [], "named_places": {}, "pieces": []}
    seen = [{"name": "wall hangings", "look": "wool", "kind": "work_piece", "box": [0.2, 0.1, 0.3, 0.3],
             "crop": "/c/s0.png", "words": "A woven hanging"},
            {"name": "candle stands", "look": "iron", "kind": "candlestick", "box": [0.05, 0.4, 0.1, 0.7],
             "crop": "/c/s1.png"}]
    things = {"seen:1": {"xy": [-4.6, 2.0], "size_m": [0.6, 0.6, 2.1], "facing": [1, 0], "against": "west"}}
    sp = PB.with_concept(spec, concept_of(seen, things))
    by = {p["id"]: p for p in sp["pieces"]}
    assert "c1" not in by and sp["_picture_left"][0]["id"] == "c1"
    assert by["c2"]["kind"] == "work_piece" and by["c2"]["longest_m"] == 2.1
    assert by["c2"]["_count"] == R["picture"]["repeat_default"]            # (its plural: several)


# ---------------------------------------------------------------- the arrangement the picture shows
def test_both_sides_of_the_room_two_mirrored_rows_with_an_aisle():
    t = pc_("c3", "table", "long tables; oak", at_xy=[-0.5, -3.0], facing_xy=[0.0, -1.0], _count=4, _sides="both")
    lay = pl.solve_recover(room([SEAT, t], named=NAMED), {"c3": {"asset": "t", "size_m": [2.0, 0.9, 0.75]}})
    tables = [p for p in lay["pieces"] if p["kind"] == "table"]
    assert lay["ok"] and len(tables) == 4, lay["refused"]
    xs = sorted({round(p["position"][0], 2) for p in tables})
    assert len(xs) == 2 and abs(xs[0] + xs[1]) < 1e-6                       # (two rows, mirrored)
    assert xs[1] - xs[0] - 0.9 >= R["picture"]["aisle_m"] - 1e-6            # (an aisle between them)
    assert all(abs(p["facing"][1]) < 1e-6 for p in tables)                  # (long side along the room)
    assert {p["asset"] for p in tables} == {"t"}


def test_seats_the_picture_shows_with_tables_stand_along_their_long_sides():
    t = pc_("c3", "table", "long tables; oak", at_xy=[-2.0, -3.0], facing_xy=[0.0, -1.0], _count=2, _sides="both")
    s = pc_("c4", "bench", "benches; pine", at_xy=[3.0, 2.0], facing_xy=[0.0, -1.0], _count=4)
    rec = {"c3": {"asset": "t", "size_m": [2.0, 0.9, 0.75]}, "c4": {"asset": "b", "size_m": [1.8, 0.4, 0.45]}}
    lay = pl.solve_recover(room([SEAT, t, s], named=NAMED), rec)
    got = by_id(lay)
    tables = [p for p in lay["pieces"] if p["kind"] == "table"]
    seats = [p for p in lay["pieces"] if p["kind"] == "bench"]
    assert len(tables) == 2 and len(seats) == 4, lay["refused"]
    for b in seats:                                     # (each beside a table's long side, its front to it)
        tb = min(tables, key=lambda q: math.dist(q["position"], b["position"]))
        dx, dy = tb["position"][0] - b["position"][0], tb["position"][1] - b["position"][1]
        assert abs(dy) < 1e-6 and 0.45 + 0.2 - 0.01 <= abs(dx) <= 0.45 + 0.2 + 0.1, (b, tb)
        assert b["facing"][0] * dx > 0                                      # (facing the table)
    for tb in tables:                                   # (both sides of every table)
        near = [b for b in seats if abs(b["position"][1] - tb["position"][1]) < 1e-6
                and abs(b["position"][0] - tb["position"][0]) < 1.0]
        assert len(near) == 2 and near[0]["position"][0] != near[1]["position"][0]
    assert got["c4x2"]["copy_of"] == "c4"


def test_things_of_a_wall_are_spread_along_it_and_a_tall_one_half_a_metre_up_still_stands():
    c = pc_("c5", "work_piece", "candle stands", called="candle stands", against="wall", at_xy=[-4.7, 2.0],
            facing_xy=[1.0, 0.0], _count=3, _raised_m=0.55, _seen_size_m=[0.6, 0.6, 2.1], longest_m=2.1)
    lay = pl.solve_recover(room([SEAT, c], named=NAMED), {"c5": {"asset": "w", "size_m": [0.6, 0.6, 2.1]}})
    stands = sorted(p["position"][1] for p in lay["pieces"] if p["kind"] == "work_piece")
    assert len(stands) == 3, lay["refused"]
    assert min(b - a for a, b in zip(stands, stands[1:])) >= 4.0           # (spread, not huddled)
    small = pc_("c6", "table", "a table; wood", at_xy=[2.0, -3.0], facing_xy=[0, -1], _raised_m=1.0,
                _seen_size_m=[0.6, 0.6, 0.7])
    lay = pl.solve_recover(room([SEAT, small], named=NAMED), {})
    assert "c6" not in by_id(lay)                                           # (on something: not on the floor)


# ---------------------------------------------------------------- the dressing: each item somewhere, said where
def test_every_dressing_item_is_placed_or_said_to_be_in_the_rooms_pictures():
    spec = room([SEAT, pc_("c2", "table", "long tables; oak", at_xy=[-2.0, -3.0], facing_xy=[0, -1]),
                 pc_("c3", "cabinet", "low sideboard; walnut", against="wall", at_xy=[4.7, 0.0], facing_xy=[-1, 0])],
                named=NAMED)
    spec["_picture_left"] = [{"id": "c5", "kind": "work_piece", "called": "woven tapestry"}]
    dress = {"station": "rich", "dressing": [
        "Long oak tables with carved legs",
        "Brass lamps standing on the sideboards",
        "Tapestries of red wool",
        "Painted panels on the lower walls",
        "Silk banners hanging from the roof",
        "Fresh rushes strewn across the floor",
        "Folding screens of carved wood",
        "A gilded harp with silver strings",
        "Pewter cups on a cupboard",
    ]}
    sp = PB.with_dressing(spec, dress)
    where = {r["item"].split()[0] + " " + r["item"].split()[1]: r for r in sp["_dressing"]}
    assert len(sp["_dressing"]) == len(dress["dressing"])
    assert where["Long oak"]["piece"] == "c2"
    assert where["Brass lamps"]["piece"] == "c3" and "on the piece" in where["Brass lamps"]["where"]
    assert where["Tapestries of"]["piece"] is None and "wall" in where["Tapestries of"]["where"]
    for k in ("Painted panels", "Silk banners", "Fresh rushes"):
        assert where[k]["piece"] is None, where[k]
    by = {p["id"]: p for p in sp["pieces"]}
    scr = by[where["Folding screens"]["piece"]]
    assert scr["kind"] == "screen" and scr["_count"] == R["picture"]["repeat_default"] and scr["_dressing"]
    harp = by[where["A gilded"]["piece"]]
    assert harp["kind"] == "work_piece" and harp["called"] == "gilded harp" and harp["_count"] == 1
    cups = by[where["Pewter cups"]["piece"]]                                # (on a thing the set lacks: it is made)
    assert cups["kind"] == "cabinet" and cups["look"].startswith("Pewter cups")
    # (and the layout places what was made: not refused for having no spot in the picture)
    lay = pl.solve_recover(sp, {})
    assert sum(1 for p in lay["pieces"] if p["kind"] == "screen") >= 2, lay["refused"]
    rep = PB.dressing_placed(sp["_dressing"], lay)
    assert next(r for r in rep if r["item"].startswith("Folding"))["placed"] >= 2


def test_no_dressing_leaves_the_spec_as_it_was():
    spec = room([SEAT], named=NAMED)
    assert PB.with_dressing(spec, {"dressing": []})["pieces"] == spec["pieces"]


def test_no_story_words_in_the_new_rules():
    banned = ("throne", "dais", "king", "knight", "hall", "forge", "anvil", "dragon", "bellows", "hearth", "royal")
    text = (json.dumps(R.get("picture")) + json.dumps(R.get("dressing")) + CO.SEEN_ASK + CO.COUNT_ASK
            + CO.SIDES_ASK).lower()
    assert not any(w in text for w in banned), [w for w in banned if w in text]
    me = Path(__file__).read_text().lower().split("def test_no_story_words")[0]
    assert not any(w in me for w in banned), [w for w in banned if w in me]


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


def test_dressing_piece_is_never_a_near_match_of_the_library():
    """A thing the place is dressed with is made from its own words (and its place's age is in them): the library's
    nearest piece of the kind, made for another place, does not serve."""
    import pieces as P
    K = P.kinds()
    kind = next(k for k, d in K.items() if d.get("placement") == "floor" and not d.get("needs")
                and d.get("made_as") is None and d.get("sized_by") is None)
    rec = {"asset_id": f"{kind}_gen_aaaaaa_01_tr", "piece_kind": kind, "look": "plain steel and glass",
           "tags": [kind, "generated-on-request"], "size_m": K[kind]["size_m"], "checks_passed": True,
           "description": f"{kind}: plain steel and glass", "class": "prop", "id": f"{kind}_gen_aaaaaa_01_tr"}
    row = P.resolve([{"id": "d1", "kind": kind, "look": "carved dark wood; of an old age", "_dressing": True}], [rec], K,
                    threshold=0.0)["pieces"][0]
    assert row["status"] == "generate", row
