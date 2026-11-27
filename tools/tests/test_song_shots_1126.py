"""11-26: a song filmed in the 3D flow. The music video's own shot rules are checked (no two talk shots back to back,
at most half a passage sung to camera); one the scene has on stage counts for the stretch's LoRAs though no line is
theirs; and a place shown with nobody in it is a place of the plan, drawn and not built."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "assets"))
import shot_flow as SF  # noqa: E402
import story_keys as SK  # noqa: E402


def sh(start, kind):
    return {"start": start, "kind": kind}


def test_two_talk_shots_never_run_back_to_back():
    f = SF._song_faults([sh(0, "talk"), sh(3, "talk"), sh(6, "quiet"), sh(9, "talk")], 12.0)
    assert any(x.startswith("shot 2 is a talk shot right after another talk shot") for x in f), f
    assert not any(x.startswith("shot 4") or x.startswith("shot 1 ") for x in f), f


def test_talk_shots_take_at_most_half_a_passage():
    f = SF._song_faults([sh(0, "talk"), sh(5, "quiet"), sh(7, "talk")], 12.0)
    assert any("at most half" in x for x in f), f
    assert SF._song_faults([sh(0, "talk"), sh(3, "quiet"), sh(6, "action"), sh(9, "talk")], 12.0) == []
    assert SF._song_faults([sh(0, "talk")], 4.0) == []          # one short line, shown whole
    assert "_song_faults(a[\"shots\"], sec) if song else []" in Path(SF.__file__).read_text()   # (a song only)
    assert "never run back to back" in SF.SONG_RULES and "at most half" in SF.SONG_RULES


def test_one_on_stage_all_through_counts_for_the_stretch(monkeypatch=None):
    with tempfile.TemporaryDirectory() as td:
        book = Path(td)
        sd = SF.shot_dir(book, "01")
        (sd / "scenes").mkdir(parents=True)
        (sd / "scenes" / "breakdown.json").write_text(json.dumps({"scenes": [
            {"start": 0, "end": 60, "place": "the Hall", "who": ["Orla", "Tam"]},
            {"start": 60, "end": 90, "place": "the Yard", "who": ["Wren"]}]}))
        es = [{"id": "a", "t": 0.0, "dur": 20.0, "who": ["Orla"]}, {"id": "b", "t": 20.0, "dur": 20.0, "who": []}]
        r = SF.lora_rank(book, "01", es, ["Orla", "Tam", "Wren"])
        assert r == {"Orla": 2, "Tam": 1, "Wren": 0}, r
        assert SF.stretch_todo({"Orla": "x", "Tam": "x", "Wren": "x"}, r)[1] == ["Wren"]


def test_a_place_with_nobody_in_it_is_drawn_not_built():
    plan = {"_es": [], "m1": [{"place": "the Quay", "people": []}],
            "m2": [{"place": "the Hall", "people": [{"name": "Orla"}]}],
            "m3": [{"place": "the Hall", "people": []}]}
    assert SK.plan_places(plan, ["m1", "m2", "m3"]) == ["the Hall", "the Quay"]

    class S(SK.Story):
        def __init__(self):
            self.plan, self.ids = plan, ["m1", "m2", "m3"]
    s = S()
    assert s.only_drawn("the Quay") and not s.only_drawn("the Hall") and not s.only_drawn("the Loft")


def test_a_failed_figure_is_tried_again_when_its_own_gates_change():
    sys.path.insert(0, str(ROOT / "tools" / "assets" / "img3d"))
    import cast_char as CC
    child, grown = "a girl of about 12, brown hair", "a man of about 40, grey hair"
    assert CC.child_armpit(grown) == 0.70 and CC.child_armpit("a tall woman") == 0.70
    assert CC.child_armpit(child) <= 0.67 < CC.child_armpit("a boy of about 16")     # (measured: 0.67 to 0.70 at 12)
    with tempfile.TemporaryDirectory() as td:
        w = Path(td)
        assert not SK.failed_before(w, child)
        (w / "FAILED").write_text("2026-11-01 10:00:00")            # a mark from before the measures were kept
        assert SK.failed_before(w, grown) and not SK.failed_before(w, child)
        SK.fail_mark(w, child)
        assert SK.failed_before(w, child)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
    print("ok")


def test_a_place_only_drawn_takes_its_manner_from_the_nearest_built_places_key():
    """11-27: drawn from the plan's words and the book's style alone, it was a painting in a film of photographed
    frames. Its reference is an engine key of the built place the story shows nearest it; none while no key is made."""
    plan = {"_es": [], "m1": [{"place": "the Hall", "people": [{"name": "Orla"}]}],
            "m2": [{"place": "the Quay", "people": []}],
            "m3": [{"place": "the Pier", "people": [{"name": "Tam"}]}],
            "m4": [{"place": "the Pier", "people": [{"name": "Tam"}]}]}
    with tempfile.TemporaryDirectory() as td:
        w = Path(td)

        class S(SK.Story):
            def __init__(self):
                self.plan, self.ids = plan, ["m1", "m2", "m3", "m4"]
                self.work, self.keys = w / "work", w / "keys"
                self.keys.mkdir(parents=True)
                self.places = {p: {"slug": SK.slug_of(p)} for p in ("the Hall", "the Quay", "the Pier")}
        s = S()
        assert s._look_picture("the Quay") == [] and s._look_picture("the Loft") == []      # (no key made yet)
        for place, shot in (("the Hall", "m1"), ("the Pier", "m3")):
            d = s.d(place)
            (d / "seq" / "pl_g01").mkdir(parents=True)
            (d / "seq" / "pl_g01" / "spec_round0.json").write_text(json.dumps(
                {"cameras": [{"plan_id": shot, "kind": "full", "show": ["a"]}]}))
            (d / "concept.png").write_bytes(b"x")
        (s.d("the Hall") / "keys.json").write_text(json.dumps([{"shot": "m1", "key": True}]))
        assert s._look_picture("the Quay") == []                 # (a key on the list whose picture is not there)
        (s.keys / "m1_s0.png").write_bytes(b"x")
        assert s._look_picture("the Quay") == [str(s.keys / "m1_s0.png")]
        (s.d("the Pier") / "keys.json").write_text(json.dumps([{"shot": "m3", "key": True}]))
        (s.keys / "m3_s0.png").write_bytes(b"x")
        assert s._look_picture("the Quay") in ([str(s.keys / "m1_s0.png")], [str(s.keys / "m3_s0.png")])
    src = Path(SK.__file__).read_text()
    assert '"refs": sheets + room + manner' in src and "DRAWN_MANNER if manner" in src
    assert "Nothing shown in that picture is taken over" in SK.DRAWN_MANNER
    assert src.index("self.draw_all(places, after, redo)") < src.index("for place in late:")    # (after the keys)
    assert '"drawn" in redo or "spec" in redo' in src                 # (a rebuild draws such a place again)


def test_two_people_whose_names_begin_alike_have_each_their_own_engine_figure():
    """11-27: ids cut to their first ten letters and last two gave two people of one book ONE engine name; the
    second was built over the first. A figure's own name carries the whole id; a figure built before keeps the
    name its record has; two base figures on one name are found, their seated figures left out."""
    sys.path.insert(0, str(ROOT / "tools" / "assets" / "img3d"))
    import cast_char as CC
    a, b = "man_bk_the_harper_auto_01", "man_bk_the_hatter_auto_01"
    assert CC.own_tag(a) != CC.own_tag(b) and len(CC.own_tag(a)) == 12 == len(CC.own_tag(b))
    assert CC.own_tag(a).isalnum() and CC.own_tag(a).endswith("01") and CC.own_tag(a) == CC.own_tag(a)
    assert CC.own_tag("man_bk_the_harper_auto_04") != CC.own_tag(a)
    assert CC.own_tag("orla_01") == "ORLA01"
    with tempfile.TemporaryDirectory() as td:
        lib = Path(td)

        def rec(fid, mh, **kw):
            (lib / "character" / fid).mkdir(parents=True)
            (lib / "character" / fid / "character.json").write_text(json.dumps(dict(kw, id=fid, rebuild_from=f"/Game/Char/{mh}")))
        assert CC.asset_tag(a, lib) == CC.own_tag(a)                       # (never built: its own)
        rec(a, "MHC_AUTO_MANBKTHEHA01"); rec(b, "MHC_AUTO_MANBKTHEHA01")
        rec(a + "_seated_armchair", "MHC_AUTO_MANBKTHEHA01"); rec("girl_bk_wren_auto_01", "MHC_AUTO_GIRLBKWREN01")
        assert CC.asset_tag(a, lib) == "MANBKTHEHA01"                       # (built before: the record's)
        assert CC.asset_tag(a, lib, own=True) == CC.own_tag(a)
        assert CC.shared_assets(lib, [a, b, "girl_bk_wren_auto_01"]) == {a: b, b: a}
        assert CC.shared_assets(lib, ["girl_bk_wren_auto_01", a + "_seated_armchair"]) == {}
    src = Path(SK.__file__).read_text()
    assert '"--own-assets", "--from", "choose"' in src and "CC.shared_assets(" in src


def test_an_arm_webbed_to_the_body_raises_the_arms_at_once_and_a_hidden_face_is_not_chosen():
    """11-27: a stout figure's arms were shaped three times at one angle, each webbed to the body, before the
    angle was raised: seven shapes for one figure. And a whole figure's hidden MetaHuman face was chosen by a
    comparison nobody sees the result of."""
    sys.path.insert(0, str(ROOT / "tools" / "assets" / "img3d"))
    import cast_char as CC
    side = lambda free, gap: {"free": free, "gap_at_elbow": gap, "pass": False}
    assert CC.webbed({"left_of_picture": side(0.66, 0.004), "right_of_picture": side(0.67, 0.06), "pass": False})
    assert CC.webbed({"left_of_picture": side(0.66, 0.0), "right_of_picture": side(0.66, 0.0), "pass": False})
    assert not CC.webbed({"left_of_picture": side(0.68, 0.044), "right_of_picture": side(0.68, 0.04), "pass": False})
    assert not CC.webbed({"left_of_picture": side(0.8, 0.3), "right_of_picture": side(0.7, 0.3), "pass": True})
    assert not CC.webbed({"file": "x.glb", "error": "no mesh"}) and not CC.webbed({})
    assert CC.WEB_GAP < 0.05 and CC.SPREAD_WEB_STEP > 6
    src = Path(CC.__file__).read_text()
    assert "if not a.mh and not a.whole and len(pool) > 1" in src
    assert "SPREAD_WEB_STEP if meshes and webbed(meshes[-1]) else 6" in src
