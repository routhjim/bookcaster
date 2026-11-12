"""A place spec's pieces -> library assets (tools/assets/places/pieces.py): the word-to-kind mapper, the lookup's
score and threshold, variants and remade records, the dry-run generation request, and that the same input gives the
same answer. Synthetic records in a temporary library; nothing here touches the real one or starts a job."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "places"))
import pieces as P  # noqa: E402


def rec(root, aid, tags, desc, size=(0.85, 0.85, 1.0), source="generated", **extra):
    d = root / "prop" / aid
    d.mkdir(parents=True)
    (d / "asset.json").write_text(json.dumps(dict(id=aid, **{"class": "prop"}, tags=tags, description=desc,
                                                  size_m=list(size), source=source, **extra)))


@pytest.fixture
def lib(tmp_path):
    rec(tmp_path, "wing_gold", ["armchair", "wingback", "damask", "gold"],
        "Wingback armchair, front -Y: upholstered in faded gold damask", (0.85, 0.85, 1.15),
        seat={"seat_z_m": 0.5})
    rec(tmp_path, "wing_red", ["armchair", "wingback", "damask", "burgundy"],
        "Wingback armchair, front -Y: upholstered in deep burgundy damask", (0.85, 0.85, 1.15),
        seat={"seat_z_m": 0.5}, variant_of="wing_gold", variant="burgundy")
    rec(tmp_path, "club_leather", ["interior"], "Button-back library armchair in dark brown leather",
        seat={"seat_z_m": 0.44})
    rec(tmp_path, "bed_iron", ["bed", "single-bed", "iron-bed"], "Black iron single bedstead with bedding",
        (0.99, 2.02, 1.17), source="scripted")
    rec(tmp_path, "desk_old", ["desk"], "Pedestal desk in dark mahogany", (1.5, 0.8, 0.76), source="scripted")
    rec(tmp_path, "desk_old_tr", ["desk", "img3d"], "Pedestal desk in dark mahogany", (1.5, 0.8, 0.76),
        replaces="desk_old")
    rec(tmp_path, "lamp_bad", ["lamp"], "Oil lamp, brass", (0.2, 0.2, 0.5), status="not_passed")
    rec(tmp_path, "lamp_a", ["lamp"], "Oil lamp, brass", (0.2, 0.2, 0.5), checks_passed=False)
    rec(tmp_path, "lamp_b", ["lamp"], "Oil lamp, brass", (0.2, 0.2, 0.5), checks_passed=False)
    rec(tmp_path, "poker_stand", ["fire", "forge"], "Iron bar for the fire")
    (tmp_path / "set" / "room_a").mkdir(parents=True)
    (tmp_path / "set" / "room_a" / "spec.json").write_text(json.dumps({"fixtures": [{"asset": "lamp_b"}]}))
    return P.load_records(tmp_path)


def test_kind_names_and_parents_are_consistent():
    K = P.kinds()
    assert set(P.PIECE_KINDS) == set(K)
    for k, d in K.items():
        assert d["parent"] is None or d["parent"] in K, k
        assert len(d["size_m"]) == 3 and d["needs"] in (None, "seat", "lie")
        assert P.word_to_kind(k.replace("_", " "))["kind"] == k      # a kind's own name maps to itself


@pytest.mark.parametrize("phrase,kind,related", [
    ("the chair by the window", "chair", ["window"]),
    ("his bed", "bed", []),
    ("in the wing-back armchair beside the fire", None, None),       # starts with a place word: see below
    ("the large writing desk in the centre", "writing_desk", []),
    ("a small bedside table", "bedside_table", []),
    ("the hearth rug", "rug", []),
])
def test_word_to_kind(phrase, kind, related):
    r = P.word_to_kind(phrase)
    if kind is None:
        assert r["kind"] == "wingback_armchair" and "fireplace" in r["related"]
    else:
        assert r["kind"] == kind and r["related"] == related


def test_word_to_kind_unresolved_is_reported_not_guessed():
    r = P.word_to_kind("the whatnot by the door")
    assert r["kind"] is None and r["unresolved"] == ["whatnot"] and r["related"] == ["door"]


def test_lookup_orders_by_look_and_colour(lib):
    red = P.lookup({"kind": "wingback_armchair", "look": "dark red damask fabric"}, lib, threshold=0.0)
    assert [h["asset_id"] for h in red][:2] == ["wing_red", "wing_gold"]
    gold = P.lookup({"kind": "wingback_armchair", "look": "faded gold damask"}, lib, threshold=0.0)
    assert gold[0]["asset_id"] == "wing_gold" and gold[0]["score"] > gold[1]["score"]
    assert set(gold[0]["why"]["parts"]) >= {"kind", "look", "colour"}


def test_narrower_record_satisfies_broader_request_not_the_reverse(lib):
    ids = [h["asset_id"] for h in P.lookup({"kind": "armchair", "look": "brown leather"}, lib, threshold=0.0)]
    assert ids[0] == "club_leather" and "wing_gold" in ids            # a wing-back is an armchair
    assert P.lookup({"kind": "wingback_armchair", "look": "brown leather"}, lib, threshold=0.0)
    assert "club_leather" not in [h["asset_id"] for h in
                                  P.lookup({"kind": "wingback_armchair", "look": "leather"}, lib, threshold=0.0)]


def test_variant_counts_for_its_kind_and_names_its_base(lib):
    h = P.lookup({"kind": "armchair", "look": "burgundy damask"}, lib)[0]
    assert h["asset_id"] == "wing_red" and h["why"]["variant_of"] == "wing_gold"


def test_remade_record_hides_the_one_it_replaces_and_failed_records_are_skipped(lib):
    ids = [h["asset_id"] for h in P.lookup({"kind": "desk", "look": "dark mahogany"}, lib, threshold=0.0)]
    assert ids == ["desk_old_tr"]
    lamps = P.lookup({"kind": "lamp", "look": "brass oil lamp"}, lib, threshold=0.0)
    assert [h["asset_id"] for h in lamps] == ["lamp_b", "lamp_a"]      # equal scores: the one a set uses first
    assert lamps[0]["why"]["used_in_sets"] == ["room_a"] and lamps[0]["why"]["checks_passed"] is False


def test_loose_words_do_not_classify_a_record(lib):
    assert P.lookup({"kind": "fireplace", "look": "iron"}, lib, threshold=0.0) == []
    assert P.word_to_kind("by the fire")["related"] == [] and P.word_to_kind("the fire")["kind"] == "fireplace"


def test_scripted_models_score_lower_unless_allowed(lib):
    piece = {"kind": "single_bed", "look": "black iron"}
    a = P.lookup(piece, lib, threshold=0.0)[0]
    b = P.lookup(piece, lib, threshold=0.0, allow_scripted=True)[0]
    assert a["why"]["scripted"] and a["score"] == pytest.approx(b["score"] * P.SCRIPTED_FACTOR, abs=1e-3)
    assert a["why"]["support"] == "lie not measured"


def test_threshold_decides_library_or_generate(lib):
    piece = [{"id": "p1", "kind": "armchair", "look": "bright green velvet"}]
    lo = P.resolve(piece, lib, threshold=0.2)
    hi = P.resolve(piece, lib, threshold=0.95)
    assert lo["pieces"][0]["status"] == "library" and hi["pieces"][0]["status"] == "generate"
    assert hi["summary"] == {"found": 0, "generate": 1, "unresolved": 0}


def test_generation_request_shape_is_a_dry_run():
    r = P.generation_request({"kind": "bed", "look": "dark oak, white bedding"}, python="py", tools="/t", work="/w")
    assert r["dry_run"] and r["kind"] == "bed" and r["needs"] == ["lie"] and r["size_m"] == P.kinds()["bed"]["size_m"]
    steps = {s["step"]: s for s in r["plan"]["steps"]}
    assert list(steps) == ["draw", "spec", "shape", "front", "measure_lie"]
    assert steps["draw"]["argv"][:3] == ["py", "/t/run_asset.py", "--line-cmd"] and "{seed}" in steps["draw"]["argv"]
    assert steps["shape"]["argv"][-2:] == ["/t/img3d/prop_one.py", r["plan"]["draw_id"]]
    assert steps["spec"]["spec"]["size_m"] == r["size_m"] and "Front -Y" in steps["spec"]["spec"]["description"]
    assert r["asset_id"] == r["plan"]["draw_id"] + "_tr" and all("gate" in s for s in r["plan"]["steps"]
                                                                 if s["step"] != "spec")
    assert P.generation_request({"kind": "rug", "look": "red"})["needs"] == []


def test_submit_refuses_without_the_flag_and_only_starts_the_drawing():
    r = P.generation_request({"kind": "wardrobe", "look": "dark mahogany"}, python="py", tools="/t", work="/w")
    with pytest.raises(RuntimeError):
        P.submit(r)
    calls = []
    P.submit(r, really=True, seed=7, run=calls.append)
    assert len(calls) == 1 and any(a.endswith("ref_draw.py") for a in calls[0]) and calls[0][-1] == "7"


def test_resolve_reports_each_piece_and_is_deterministic(lib):
    pieces = [{"id": "p1", "kind": "wingback_armchair", "look": "dark red damask"},
              {"id": "p2", "kind": "wardrobe", "look": "dark mahogany"},
              {"id": "p3", "kind": "whatnot", "look": ""}]
    a, b = P.resolve(pieces, lib), P.resolve(pieces, list(reversed(lib)))
    assert a == b and json.dumps(a, sort_keys=True) == json.dumps(P.resolve(pieces, lib), sort_keys=True)
    st = {r["id"]: r["status"] for r in a["pieces"]}
    assert st == {"p1": "library", "p2": "generate", "p3": "unresolved"}
    assert a["summary"] == {"found": 1, "generate": 1, "unresolved": 1}
    assert a["pieces"][1]["request"]["asset_id"] == P.generation_request(pieces[1])["asset_id"]


def test_a_role_kind_is_met_by_its_parents_records(tmp_path):
    rec(tmp_path, "chair_plain", ["chair"], "Plain wooden chair", (0.5, 0.5, 0.9))
    recs = P.load_records(tmp_path)
    h = P.lookup({"kind": "desk_chair", "look": "plain wooden"}, recs)
    assert [x["asset_id"] for x in h] == ["chair_plain"] and h[0]["why"]["parts"]["kind"] == 0.85
    assert P.lookup({"kind": "armchair", "look": "plain wooden"}, recs, threshold=0.0) == []
    rec(tmp_path, "club", ["armchair"], "Plain wooden armchair")
    both = P.lookup({"kind": "desk_chair", "look": "plain wooden"}, P.load_records(tmp_path), threshold=0.0)
    assert [x["asset_id"] for x in both] == ["chair_plain", "club"] and both[1]["why"]["parts"]["kind"] == 0.6
