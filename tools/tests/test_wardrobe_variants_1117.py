"""A person's change of clothes over the story in the engine flow (scenes/wardrobe.py, staging_pass's wardrobe
default and variants, from_plan's worn plan and variant figures), on a temp library with a scripted model.

The planner's wardrobe rows go on the chapter's clock by their paragraph's start; the row in force at a shot's time
(its id's tenths) names the variant figure base_<suffix>; a variant is offered only when recorded as one (its own
stances are not worn states); the pass starts each shot from the wardrobe's outfit and its own question still
changes it until the wardrobe's next change; the specs stage the variant's own figure with its own words.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import staging_synth as S  # noqa: E402
import staging_pass as P  # noqa: E402
import from_plan as FP  # noqa: E402
import wardrobe as WD  # noqa: E402

A, B = "Alder", "Birch"
ROWS = [{"from": 2, "id": "Plate Mail", "dress": "Dull, unpolished plate mail consisting of a breastplate, worn over a "
                                                 "leather coat"},
        {"from": 4, "id": "helm off", "dress": "Plate mail with the helm removed"}]
STARTS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]


# ------------------------------------------------------------------ pure logic
def test_rows_go_on_the_clock_by_their_paragraph_and_the_last_started_is_in_force():
    tr = WD.timed(ROWS, STARTS)
    assert [(r["t"], r["variant"]) for r in tr] == [(20.0, "plate_mail"), (40.0, "helm_off")]
    assert WD.variant_at(tr, 19.9) == "usual"
    assert WD.variant_at(tr, 19.97) == "plate_mail"            # (as a paragraph read at t: a twentieth's slack)
    assert WD.variant_at(tr, 39.0) == "plate_mail" and WD.variant_at(tr, 400.0) == "helm_off"
    assert WD.in_force(tr, None) is None
    assert WD.timed([{"from": 99, "id": "x", "dress": ""}], STARTS)[0]["t"] == 50.0   # past the end: the last one
    assert WD.timed([{"from": "?", "id": "x"}, {"from": 1, "id": ""}], STARTS) == []


def test_names_and_times():
    assert WD.suffix("Star-Iron  armour!") == "star_iron_armour"
    assert WD.variant_id("man_x_01", "helm off") == "man_x_01_helm_off"
    assert WD.shot_time("quiet-05m01s-003017") == 301.7
    assert WD.shot_time("talk-3", 7.0) == 7.0
    assert WD.is_stance("seated_armchair") and WD.is_stance("plate_lying_bed_01") and not WD.is_stance("plate_mail")


def test_outfit_words_look_line_and_sheet():
    assert WD.outer_garment(ROWS[0]["dress"]) == "Dull unpolished plate mail"
    assert WD.outer_garment("a cloak, worn over a tunic") == "a cloak"
    look = "a man of about 30, short hair, wearing a leather coat and boots, holding a staff."
    assert WD.variant_look(look, "Plate mail, a helm") == "a man of about 30, short hair, wearing plate mail, a helm, holding a staff."
    assert WD.variant_look("a man, wearing a coat.", "mail") == "a man, wearing mail."
    assert WD.variant_look("a tall man.", "mail") == "a tall man, wearing mail."
    sheets = {"plate mail": "/s/a.png", "helm": "/s/b.png", "plate": "/s/c.png"}
    assert WD.sheet_for(sheets, "Plate-Mail") == "/s/a.png"            # named the same
    assert WD.sheet_for(sheets, "plate mail and helm") == "/s/a.png"   # the most words, all of them in the name
    assert WD.sheet_for(sheets, "robe") is None


# ------------------------------------------------------------------ the library's variants
def test_only_recorded_variants_and_older_unrecorded_ones_are_worn_states(tmp_path):
    lib = tmp_path / "lib"
    S.add_character(lib, "fig_b")
    S.add_character(lib, "fig_b_plate_mail", {"variant_of": "fig_b"})
    S.add_character(lib, "fig_b_plate_mail_lying_bed_01", {"stance": {"pose": "lying"}})   # its own stance
    S.add_character(lib, "fig_b_lying_bed_01", {"stance": {"pose": "lying"}})
    S.add_character(lib, "fig_b_seated_armchair_2", {})
    S.add_character(lib, "fig_b_veiled", {})                                                # an older variant
    S.add_character(lib, "fig_b_other", {"variant_of": "fig_c"})
    assert P.variants(lib, "fig_b") == ["plate_mail", "usual", "veiled"]


# ------------------------------------------------------------------ the pass and the specs
def sid(kind, t):
    return f"{kind}-00m00s-{int(round(t * 10)):06d}"


SHOTS = [sid("quiet", 12.0), sid("action", 22.0), sid("action", 32.0), sid("quiet", 36.0), sid("quiet", 42.0)]


def person(name, pose="standing", at="near the desk"):
    return {"name": name, "pose": pose, "at": at, "facing": ""}


class Model:
    """Says "usual" to every worn question (a passage that never names the outfit), or `claim` at one shot."""

    def __init__(self, claim=None):
        self.prompts, self.claim = [], claim or {}

    def __call__(self, prompt, rep):
        self.prompts.append(prompt)
        q = prompt.split("Question:", 1)[1] if "Question:" in prompt else prompt
        if "wearing when this shot ends" in q:
            for k, v in self.claim.items():
                if k in prompt:
                    return v
            return "not stated"
        if "wearing at this moment" in q:
            return "usual"
        if "begin to do" in q:
            return "none"
        if "go out of the room" in q or "come into the room" in q:
            return "none"
        if "already there" in q:
            return "no"
        if "eyes" in q:
            return "open"
        return "not stated"


@pytest.fixture
def scene(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [{"name": "desk", "position": [-1.6, -1.6], "size": [1.4, 0.7, 0.8]}],
                         lights=[{"type": "point", "position": [0.0, 2.3, 0.4]}])
    for cid in ("fig_a", "fig_b"):
        S.add_character(lib, cid)
    S.add_character(lib, "fig_b_plate_mail", {"variant_of": "fig_b"})
    S.add_character(lib, "fig_b_helm_off", {"variant_of": "fig_b"})
    monkeypatch.setattr(P.C, "LIBRARY", lib)
    monkeypatch.setattr(FP.C, "LIBRARY", lib)
    plan = {s: [{"kind": s.split("-")[0], "people": [person(A), person(B)], "action": "they work.",
                 "place": "a forge", "room": "a forge with a desk", "fixtures": ["desk"], "framing": "medium"}]
            for s in SHOTS}
    plan["_es"] = [{"id": s, "t": WD.shot_time(s)} for s in SHOTS]
    cast = {A: {"key": "a", "id": "fig_a"},
            B: {"key": "b", "id": "fig_b", "wardrobe": WD.timed(ROWS, STARTS),
                "worn": {"plate_mail": {"wear_set": "plate mail", "wear_back_set": "the back of his plate mail",
                                        "worn_sheet": "/s/plate.png"},
                         "helm_off": {"wear_set": "plate mail, bare head", "worn_sheet": "/s/helm.png"}}}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "shots.txt").write_text("\n".join(SHOTS))
    (tmp_path / "cast.json").write_text(json.dumps(cast))
    (tmp_path / "timeline.json").write_text(json.dumps([{"t": t, "text": f"Line at {t}."} for t in range(0, 50, 5)]))
    return tmp_path, lib


def run_pass(scene, monkeypatch, model):
    tmp, lib = scene
    import place_spec
    monkeypatch.setattr(place_spec, "things", lambda *a, **k: {})
    monkeypatch.setattr(P.Asker, "one", lambda self, prompt, rep: model(prompt, rep))
    monkeypatch.setattr(sys, "argv", ["staging_pass.py", str(tmp / "plan.json"), str(tmp / "shots.txt"),
                                      str(tmp / "cast.json"), str(tmp / "timeline.json"),
                                      str(lib / "set" / "room_x"), str(tmp / "pass.json")])
    P.main()
    return json.loads((tmp / "pass.json").read_text())


def test_the_wardrobe_is_where_each_shot_starts_and_no_worn_start_question_is_asked(scene, monkeypatch):
    m = Model()
    res = run_pass(scene, monkeypatch, m)
    assert res["worn_start"][B] == "usual"
    assert res["worn_plan"] == {SHOTS[1]: {B: "plate_mail"}, SHOTS[4]: {B: "helm_off"}}
    assert not any("wearing at this moment" in p and B in p.split("Question:")[1] for p in m.prompts)
    # the worn question in the armour's shots is asked with the armour as where things stand
    assert any("wearing when this shot ends" in p and f"{B} wears: plate_mail" in p for p in m.prompts)


def test_a_change_the_passage_shows_holds_until_the_wardrobes_next_change(scene, monkeypatch):
    # the third shot's own passage takes the armour off: from its end until the wardrobe's next row, usual
    tmp, _ = scene
    res = run_pass(scene, monkeypatch, Model(claim={"Line at 35.": "usual"}))
    assert res["worn_end"] == {SHOTS[2]: {B: "usual"}}
    specs, facts = run_fp(tmp, monkeypatch, res)
    worn = {s: facts[s]["pass"]["worn"].get("b") for s in SHOTS}
    assert worn == {SHOTS[0]: "usual", SHOTS[1]: "plate_mail", SHOTS[2]: "plate_mail", SHOTS[3]: "usual",
                    SHOTS[4]: "helm_off"}


def run_fp(tmp, monkeypatch, res):
    (tmp / "pass.json").write_text(json.dumps(res))
    out = tmp / "out"
    monkeypatch.setattr(sys, "argv", ["from_plan.py", str(tmp / "plan.json"), str(tmp / "shots.txt"),
                                      str(tmp / "cast.json"), str(out), "--pass", str(tmp / "pass.json")])
    FP.main()
    return ([json.loads(p.read_text()) for p in sorted(out.glob("pl_g*.json"))],
            json.loads((out / "plan_facts.json").read_text()))


def test_the_specs_stage_the_variants_own_figure_with_its_own_words(scene, monkeypatch):
    tmp, _ = scene
    res = run_pass(scene, monkeypatch, Model())
    specs, _ = run_fp(tmp, monkeypatch, res)
    by = {c["plan_id"]: sp["characters"]["b"] for sp in specs for c in sp["cameras"]}
    assert by[SHOTS[0]]["id"] == "fig_b" and "wear_set" not in by[SHOTS[0]]
    b = by[SHOTS[1]]
    assert b["id"] == "fig_b_plate_mail" and b["worn"] == "plate_mail"
    assert b["wear_set"] == "plate mail" and b["worn_sheet"] == "/s/plate.png"
    assert by[SHOTS[4]]["id"] == "fig_b_helm_off" and by[SHOTS[4]]["wear_set"] == "plate mail, bare head"


def test_own_variant_needs_the_record(tmp_path):
    lib = tmp_path / "lib"
    S.add_character(lib, "fig_b_plate_mail", {"variant_of": "fig_b"})
    S.add_character(lib, "fig_b_veiled", {})
    assert FP.own_variant(lib, "fig_b", "plate_mail") == "fig_b_plate_mail"
    assert FP.own_variant(lib, "fig_b", "veiled") is None
    assert FP.own_variant(lib, "fig_b", "robe") is None
