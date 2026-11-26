"""A piece another floor piece stands on is its base (scenes/place_spec.py validate, sets/place_layout.py take_bases /
lay_bases, sets/place_build.py build / base_boxes, scenes/from_plan.py): made-up rooms only, no engine, no model.

The spec's "on" between two floor pieces is kept (not a fault, and not offered in the prompt); the base gets no mesh
of its nominal kind and no place of its own: it is laid under what it carries, low, and built as a plain box in the
floor's material; a plan phrase that names it sends people to the floor before its front edge, facing what it carries.
"""
import json
import math
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TOOLS / "assets" / "sets"))
sys.path.insert(0, str(TOOLS / "assets" / "places"))
sys.path.insert(0, str(TOOLS / "assets" / "scenes"))
sys.path.insert(1, str(TOOLS / "assets"))
import place_cost as pc  # noqa: E402
import place_layout as L  # noqa: E402
import place_build as PB  # noqa: E402
import place_spec as SP  # noqa: E402
import pieces as P  # noqa: E402
import staging_synth as S  # noqa: E402
import from_plan as FP  # noqa: E402

RULES = pc.load_rules()["base"]


def spec_of(pieces, relations=(), named=None, size=(8.0, 12.0, 4.0)):
    return {"schema": "place_spec/1", "place": "a reading room", "kind": "hall", "period": "not stated",
            "size": "large", "room": "A long room. A seat stands raised at its far end.", "fixtures": ["seat", "table"],
            "crowd": "", "openings": [{"id": "o1", "kind": "door", "wall": "south", "words": ""}],
            "pieces": list(pieces), "relations": list(relations), "named_places": dict(named or {}),
            "_room_m": list(size), "land_places": {}}


def piece(i, kind, phrases=(), against="not stated", look="plain wood"):
    return {"id": i, "kind": kind, "look": look, "phrases": list(phrases), "against": against}


RAISED = spec_of([piece("p1", "chair", ["on the seat"], "wall"), piece("p2", "bench", ["before the step"]),
                  piece("p3", "table", []), piece("p4", "rug", [])],
                 [{"a": "p1", "rel": "on", "b": "p2"}],
                 {"on the seat": "p1", "before the step": "p2", "in the middle": "floor"})


# ------------------------------------------------------------------ 1. the validator keeps "on" between floor pieces
def test_on_between_two_floor_pieces_is_kept():
    f = SP.validate(RAISED, ["on the seat", "before the step"])
    assert not [x for x in f if "relation" in x], f


def test_on_that_is_no_base_is_still_a_fault():
    rug = spec_of([piece("p1", "chair"), piece("p2", "rug")], [{"a": "p1", "rel": "on", "b": "p2"}])
    lamp = spec_of([piece("p1", "lamp"), piece("p2", "table")], [{"a": "p1", "rel": "on", "b": "p2"}])
    door = spec_of([piece("p1", "chair")], [{"a": "p1", "rel": "on", "b": "o1"}])
    odd = spec_of([piece("p1", "chair"), piece("p2", "table")], [{"a": "p1", "rel": "atop", "b": "p2"}])
    for sp in (rug, lamp, door, odd):
        assert any("rel must be one of RELATIONS" in x for x in SP.validate(sp, [])), sp["relations"]


def test_the_prompt_does_not_offer_the_base_relation():
    assert "on" not in SP.KINDS["RELATIONS"]
    txt = SP.prompt("a reading room", ["on the seat"], ["shot 1"], "text")
    line = next(ln for ln in txt.splitlines() if ln.startswith("RELATIONS:"))
    assert line == "RELATIONS: " + ", ".join(SP.KINDS["RELATIONS"])


# ------------------------------------------------------------------ 2. a base leaves the pieces and the search
def test_take_bases_moves_the_base_out_and_its_phrase_onto_what_it_carries():
    sp, bases = L.take_bases(RAISED)
    assert set(bases) == {"p2"} and bases["p2"]["of"] == ["p1"] and bases["p2"]["kind"] == "bench"
    assert [p["id"] for p in sp["pieces"]] == ["p1", "p3", "p4"]
    assert sp["relations"] == []
    assert sp["named_places"]["before the step"] == "p1" and sp["named_places"]["in the middle"] == "floor"
    again, bases2 = L.take_bases(sp)                       # (repeated on its own result: the same)
    assert again["pieces"] == sp["pieces"] and set(bases2) == {"p2"} and again["named_places"] == sp["named_places"]
    assert RAISED["pieces"][1]["id"] == "p2"               # (the spec given is not changed)


def test_a_rug_under_a_piece_is_no_base():
    sp = spec_of([piece("p1", "chair"), piece("p2", "rug")], [{"a": "p1", "rel": "on", "b": "p2"}])
    out, bases = L.take_bases(sp)
    assert bases == {} and out is sp


# ------------------------------------------------------------------ 3. laid under what it carries, low
def test_the_base_is_laid_under_its_piece_and_out_in_front_of_it():
    sp, bases = L.take_bases(RAISED)
    lay = L.solve_recover(sp, {})
    host = next(p for p in lay["pieces"] if p["id"] == "p1")
    assert all(p["id"] != "p2" for p in lay["pieces"])      # (no place of its own: nothing to block in the search)
    b, = L.lay_bases(lay, bases)
    assert b["of"] == [host["name"]]
    assert 0.0 < b["size_m"][2] <= RULES["most_height_m"] + 1e-9
    f = host["facing"]
    w, d = host["size_m"][0], host["size_m"][1]
    assert abs(b["size_m"][0] - (w + 2 * RULES["side_m"])) < 0.02
    assert abs(b["size_m"][1] - (d + RULES["front_m"] + RULES["back_m"])) < 0.02
    # its front edge is front_m in front of the piece's own front, on the piece's middle line
    out_ = (b["edge"][0] - host["position"][0]) * f[0] + (b["edge"][1] - host["position"][1]) * f[1]
    side_ = (b["edge"][0] - host["position"][0]) * f[1] - (b["edge"][1] - host["position"][1]) * f[0]
    assert abs(out_ - (d / 2 + RULES["front_m"])) < 0.02 and abs(side_) < 0.02
    r, hr = b["rect"], host["rect"]                          # it is under the piece, inside the room
    assert r[0] <= hr[0] + 1e-3 and r[1] <= hr[1] + 1e-3 and r[2] >= hr[2] - 1e-3 and r[3] >= hr[3] - 1e-3
    W, D, _ = lay["size_m"]
    assert -W / 2 - 1e-3 <= r[0] and r[2] <= W / 2 + 1e-3 and -D / 2 - 1e-3 <= r[1] and r[3] <= D / 2 + 1e-3


def test_a_base_whose_piece_is_not_on_the_floor_is_left_out():
    lay = {"size_m": [6.0, 6.0, 3.0], "pieces": [{"id": "p1", "name": "chair", "hang": True, "facing": [0, -1],
                                                     "position": [0, 2.9], "size_m": [0.5, 0.5, 0.9]}]}
    b, = L.lay_bases(lay, {"p2": {"id": "p2", "of": ["p1"]}})
    assert b["of"] == [] and b["why"]


# ------------------------------------------------------------------ built: a plain low box, no mesh of its kind
@pytest.fixture
def built(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    (lib / "prop").mkdir(parents=True)
    seen = []

    def records_for(spec, library):          # (the library's match for each piece: a made-up record)
        seen.extend(p["id"] for p in spec["pieces"])
        rec = {p["id"]: {"asset": f"{p['kind']}_x", "size_m": [0.6, 0.6, 0.95]} for p in spec["pieces"]}
        return rec, P.resolve(spec["pieces"], [])
    monkeypatch.setattr(PB, "records_for", records_for)
    out = PB.build(json.loads(json.dumps(RAISED)), "set_x", lib)
    assert "p2" not in seen                  # (the base is never looked up nor asked to be made)
    return out


def test_the_base_is_built_as_a_box_in_the_floor_material(built):
    out, named, report = built
    box = [s for s in out["structure"] if s.get("kind") == "box" and s.get("name") == "base_p2"]
    assert len(box) == 1
    floor = next(s for s in out["structure"] if s.get("kind") == "floor")
    assert box[0]["material"] == floor["material"] and box[0]["base"][2] == 0.0
    assert box[0]["size_m"][2] <= RULES["most_height_m"]
    assert report["bases"][0]["built"] and report["bases"][0]["said_kind"] == "bench"
    # no piece of its nominal kind is built or asked to be made
    assert not any("bench" in (f.get("name") or "") or "bench" in (f.get("asset") or "") for f in out["fixtures"])
    assert not any(p["id"] == "p2" for p in report["left_out"])


def test_the_phrase_that_names_the_base_names_what_it_carries(built):
    out, named, _ = built
    host = named["on the seat"]
    assert named["before the step"] == host and named["in the middle"] == "floor"
    item = next(f for f in out["fixtures"] if f["name"] == host)
    assert item["base"]["name"] == "base_p2" and item["base"]["height_m"] <= RULES["most_height_m"]


def test_the_emitted_spec_has_no_base_to_make(tmp_path, monkeypatch):
    src = tmp_path / "spec.json"
    src.write_text(json.dumps(RAISED))
    lib = tmp_path / "lib"
    (lib / "prop").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["place_build.py", str(src), "set_x", "--emit", str(tmp_path / "c.json"),
                                      "--library", str(lib)])
    PB.main()
    em = json.loads((tmp_path / "c.json").read_text())
    assert [p["id"] for p in em["pieces"]] == ["p1", "p3", "p4"] and em["_bases"][0]["id"] == "p2"


# ------------------------------------------------------------------ 4. people sent to it stand before its front edge
A, B = "Alder", "Birch"
CAST = {A: {"key": "a", "id": "fig_a"}, B: {"key": "b", "id": "fig_b"}}
EDGE, FRONT = [0.0, 1.0], [0.0, -1.0]


@pytest.fixture
def raised_lib(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [
        {"name": "chair", "position": [0.0, 2.3], "size": [0.6, 0.6, 1.0], "seat": True, "facing": [0.0, -1.0]},
        {"name": "table", "position": [2.0, -1.5], "size": [1.2, 0.7, 0.8]}], size=(7.0, 6.0, 3.0),
        set_id="room_r", description="a reading room with a raised seat")
    sd = lib / "set" / "room_r"
    sp = json.loads((sd / "spec.json").read_text())
    sp["place"] = "a reading room"
    sp["structure"].insert(0, {"kind": "floor", "material": "floor"})
    sp["structure"].append({"kind": "box", "name": "base_p2", "base": [0.0, 1.8, 0.0], "size_m": [1.6, 1.6, 0.25],
                            "material": "floor"})
    sp["fixtures"][0]["base"] = {"name": "base_p2", "edge": EDGE, "front": FRONT, "rect": [-0.8, 1.0, 0.8, 2.6],
                                 "height_m": 0.25}
    (sd / "spec.json").write_text(json.dumps(sp))
    (sd / "named_places.json").write_text(json.dumps({"on the seat": "chair", "before the step": "chair",
                                                      "by the table": "table"}))
    for cid in ("fig_a", "fig_b"):
        S.add_character(lib, cid)
    monkeypatch.setattr(FP.C, "LIBRARY", lib)
    return lib


def run(tmp_path, monkeypatch, shots):
    (tmp_path / "plan.json").write_text(json.dumps({sid: [sh] for sid, sh in shots.items()}))
    (tmp_path / "shots.txt").write_text("\n".join(shots))
    (tmp_path / "cast.json").write_text(json.dumps(CAST))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["from_plan.py", str(tmp_path / "plan.json"), str(tmp_path / "shots.txt"),
                                      str(tmp_path / "cast.json"), str(out)])
    FP.main()
    return [json.loads(p.read_text()) for p in sorted(out.glob("pl_g*.json"))]


def shot(people):
    return {"kind": "quiet", "framing": "wide", "place": "a reading room", "room": "a reading room",
            "people": [{"name": n, "pose": p, "at": at, "facing": fc} for n, p, at, fc in people]}


def test_someone_before_the_base_stands_on_the_floor_before_its_edge_facing_the_seat(raised_lib, tmp_path, monkeypatch):
    specs = run(tmp_path, monkeypatch, {
        "s1": shot([(A, "sitting", "on the seat", B), (B, "standing", "by the table", A)]),
        "s2": shot([(A, "sitting", "on the seat", B), (B, "standing", "before the step", A)])})
    a, b = specs[-1]["characters"]["a"], specs[-1]["characters"]["b"]
    assert a.get("seat") == "chair"
    d = math.dist(a["position"][:2], b["position"][:2])
    assert 1.5 <= d <= 2.5, (d, b["position"])
    out_ = (b["position"][0] - EDGE[0]) * FRONT[0] + (b["position"][1] - EDGE[1]) * FRONT[1]
    assert out_ >= FP.BASE_CLEAR_M - 1e-6, b["position"]           # off the base, in front of its edge
    to_a = [a["position"][0] - b["position"][0], a["position"][1] - b["position"][1]]
    n = math.hypot(*to_a)
    assert (b["facing"][0] * to_a[0] + b["facing"][1] * to_a[1]) / n > 0.9
    for m in b.get("marks") or []:                                  # every mark the camera may move them to too
        assert (m[0] - EDGE[0]) * FRONT[0] + (m[1] - EDGE[1]) * FRONT[1] >= FP.BASE_CLEAR_M - 1e-6


def test_a_pass_answer_in_front_of_a_thing_on_a_base_is_before_its_edge():
    things = [{"name": "chair", "kind": "fixture", "center": [0.0, 2.3, 0.5], "half": [0.3, 0.3, 0.5],
               "base": {"edge": EDGE, "front": FRONT}}]
    pl = FP.place_from({"thing": "chair", "sitting": False, "side": "front"}, things, {}, [7.0, 6.0, 3.0], [], set())
    out_ = (pl["position"][0] - EDGE[0]) * FRONT[0] + (pl["position"][1] - EDGE[1]) * FRONT[1]
    assert pl["pose"] == "standing" and out_ >= FP.BASE_CLEAR_M - 1e-6


def test_no_story_words():
    banned = ("throne", "dais", "king", "knight", "forge", "anvil", "dragon", "bellows", "hearth", "edric")
    text = (json.dumps(pc.load_rules()["base"]) + Path(__file__).read_text().split("def test_no_story_words")[0]).lower()
    src = (TOOLS / "assets" / "sets" / "place_layout.py").read_text().lower()
    part = src[src.index("a piece's base\n"):src.index("check, named places, set spec")]
    assert not any(w in text or w in part for w in banned)
