"""Every staged shot ends with an engine key (story_keys.step_keys, the user's rule 11-17): a shot whose staging failed
every gate keeps its least-bad try, flagged with what it missed; a shot whose redraw passed no check keeps its
drawing, flagged; an insert (a shot with nobody of the cast, from_plan.insert_camera) is keyed like any other."""
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "assets"))
import story_keys as SK  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
import from_plan as FP  # noqa: E402


def still(p: Path, grey=120):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), (grey, grey, grey)).save(p)


def test_a_failed_staging_and_an_insert_both_end_with_a_key(tmp_path, monkeypatch):
    st = SK.Story.__new__(SK.Story)
    st.work, st.keys, st.look_p = tmp_path / "work", tmp_path / "keys", tmp_path / "look.json"
    st.places = {"P": {"slug": "p", "set": "s"}}
    st.keys.mkdir(parents=True)
    monkeypatch.setattr(SK, "sh", lambda *a, **k: 0)
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, place: ["A", "B", "C"])
    d = st.work / "p"
    g1, g2 = d / "seq" / "pl_g01", d / "seq" / "pl_g02"
    for g, cam in ((g1, {"name": "s01", "plan_id": "A"}), (g2, {"name": "s02", "plan_id": "B", "insert": True})):
        g.mkdir(parents=True)
        (g / "spec_round0.json").write_text(json.dumps({"cameras": [cam]}))
        still(g / f"{cam['name']}.png")
    (g1 / "speakers.json").write_text(json.dumps({"shots": {"s01": {"ok": False, "history": [
        {"why": ["a thing in the foreground covers 30% of the frame"]}]}}}))
    still(d / "run" / "pl_g01__s01__s80__r7.png")                     # its drawing, which passed no check
    still(d / "run" / "pl_g02__s02__s80__r7.png")
    (d / "run" / "accepted.json").write_text(json.dumps({"pl_g02__s02__s80__r7": {"path": str(d / "run" / "pl_g02__s02__s80__r7.png")}}))
    assert st.step_keys("P")
    rows = {r["shot"]: r for r in json.loads((d / "keys.json").read_text())}
    assert rows["A"]["key"] and rows["A"]["staging_miss"] and rows["A"]["redraw_miss"]
    assert rows["B"]["key"] and not rows["B"].get("staging_miss")
    assert (st.keys / "A_s0.png").exists() and (st.keys / "B_s0.png").exists()
    assert not rows["C"]["key"] and rows["C"]["why"] == "no scene spec"          # (no spec: from_plan's job)


def test_a_shot_with_nobody_of_the_cast_gets_an_insert_camera_on_the_thing_it_names():
    things = [{"name": "door_south", "kind": "door", "center": [0.0, -4.0, 1.0], "half": [0.5, 0.1, 1.0]},
              {"name": "work_piece", "kind": "fixture", "center": [1.0, 1.5, 0.4], "half": [0.5, 0.3, 0.4]}]
    named = {"at the block": "work_piece", "by the door": "door_south"}
    c = FP.insert_camera({"kind": "quiet", "framing": "close", "action": "A blade hangs over the door."}, things,
                         named, [4.0, 8.0, 3.0], "s09_quiet", "quiet-1")
    assert c["insert"] and c["_on"] == "door_south" and c["target"][:2] == [0.0, -4.0]
    assert abs(c["position"][0]) <= 1.6 and abs(c["position"][1]) <= 3.6 and c["position"][1] > -4.0
    c2 = FP.insert_camera({"kind": "quiet", "framing": "wide", "action": "The place is empty."}, things, named,
                          [4.0, 8.0, 3.0], "s10_quiet", "quiet-2")
    assert c2["_on"] == "the room's middle"


def test_an_insert_camera_does_not_stand_inside_a_standing_thing():
    things = [{"name": "work_piece", "kind": "fixture", "center": [0.0, 2.0, 0.4], "half": [0.5, 0.3, 0.4]},
              {"name": "work_piece_2", "kind": "fixture", "center": [0.0, -0.6, 1.0], "half": [0.8, 0.8, 1.0]}]
    c = FP.insert_camera({"kind": "quiet", "framing": "medium", "action": "at the block"}, things,
                         {"at the block": "work_piece"}, [4.0, 8.0, 3.0], "s1", "q")
    assert c["_on"] == "work_piece" and FP.clear_of_fixtures(c["position"][:2], things[1:], 0.0)


def test_a_place_left_with_a_keyless_shot_is_staged_again_with_the_fallback(tmp_path, monkeypatch):
    st = SK.Story.__new__(SK.Story)
    st.work = tmp_path / "work"
    st.places = {"P": {"slug": "p", "set": "s"}}
    d = st.work / "p"
    (d / "specs").mkdir(parents=True)
    (d / "specs" / "pl_g01.json").write_text(json.dumps({"cameras": [{"plan_id": "A"}]}))
    (d / "specs" / "pl_g02.json").write_text(json.dumps({"cameras": [{"plan_id": "B"}]}))
    (d / "keys.json").write_text(json.dumps([{"shot": "A", "key": True}, {"shot": "B", "key": False}]))
    seen = []

    def step(name):
        def f(self, place, **kw):
            seen.append((name, kw))
            if name == "keys":
                (d / "keys.json").write_text(json.dumps([{"shot": "A", "key": True}, {"shot": "B", "key": True}]))
            return True
        return f
    for n in ("stage", "redraw", "keys"):
        monkeypatch.setattr(SK.Story, f"step_{n}", step(n))
    st.restage_keyless("P")
    # only the keyless shot's scene, and only that shot takes the clear wide
    assert seen == [("stage", {"only": ["pl_g02"], "fallback": ["B"]}), ("redraw", {"only": ["pl_g02"]}), ("keys", {})]
    assert st.keyless("P") == [] and "SPEAKERS_FALLBACK_ALL" not in SK.os.environ
    seen.clear()
    st.restage_keyless("P")                                             # nothing keyless: nothing staged
    assert seen == []


def test_a_last_resort_wide_is_counted_by_what_it_misses_itself():
    import story_keys as SK
    old = ["x (named for the shot) is not in the picture"]
    assert SK.staging_miss({"ok": True, "history": [{"why": old}]}) == []
    assert SK.staging_miss({"ok": False, "history": [{"why": old}]}) == old
    assert SK.staging_miss({"ok": False, "history": [{"why": old}], "least_bad": {"misses": ["y"]}}) == ["y"]
    wide = {"ok": False, "history": [{"why": old}], "least_bad": {"fallback": True, "misses": [], "before": old}}
    assert SK.staging_miss(wide) == []


def test_a_whole_place_shot_of_nobody_in_the_open_is_seen_as_the_picture_sees_it_and_drawn_free(tmp_path):
    import redraw as RD
    sh = {"kind": "action", "framing": "wide", "action": "Many of them cross the open ground."}
    c = FP.insert_camera(sh, [], {}, [20.0, 20.0], "s01_action", "action-1", walls=False)
    assert c["free"] and c["insert"] and c["position"][1] < -9 and c["target"][1] > c["position"][1] + 15
    # indoors, or framed close, or on a thing the words name: the shot's own camera, held to the staging
    assert not FP.insert_camera(sh, [], {}, [20.0, 20.0], "s01_action", "action-1").get("free")
    assert not FP.insert_camera(dict(sh, framing="close"), [], {}, [20.0, 20.0], "s", "a", walls=False).get("free")
    sd = tmp_path / "pl_g01"
    sd.mkdir()
    (sd / "spec_round0.json").write_text(json.dumps({"cameras": [dict(c, name="s01_action"), {"name": "s02_quiet"}]}))
    assert RD.drawn_free(sd, "s01_action") and not RD.drawn_free(sd, "s02_quiet")


def test_a_free_shot_starts_from_the_places_own_picture_when_it_is_the_stills_size(tmp_path):
    import redraw as RD
    from PIL import Image
    Image.new("RGB", (64, 36)).save(tmp_path / "still.png")
    Image.new("RGB", (64, 36)).save(tmp_path / "place.png")
    Image.new("RGB", (40, 40)).save(tmp_path / "other.png")
    st = str(tmp_path / "still.png")
    assert RD.free_start({"extra_refs": [str(tmp_path / "place.png")]}, st) == str(tmp_path / "place.png")
    assert RD.free_start({"extra_refs": [str(tmp_path / "other.png")]}, st) == st
    assert RD.free_start({}, st) == st


def test_a_key_is_told_what_each_person_looks_at(tmp_path):
    import redraw as RD
    (tmp_path / "spec_round0.json").write_text(json.dumps({"characters": {
        "a": {"work_at": "block_2", "facing_to": "b"}, "b": {"facing_to": "a"}, "c": {}}}))
    cast = {"a": {"noun": "man", "pos": "his", "subj": "he"}, "b": {"noun": "woman"}, "c": {"noun": "man"}}
    box = [0.1, 0.1, 0.3, 0.9]
    ppl = [("a", box, 80.0, "standing", 80.0, None), ("b", [0.6, 0.1, 0.8, 0.9], 10.0, "standing", 10.0, None),
           ("c", [0.85, 0.1, 0.95, 0.9], 40.0, "standing", 40.0, None)]
    w = RD.facing_to_words(tmp_path, ppl, cast)
    a, b = w.split("eyes on it.")[0], w.split("eyes on it.")[1]
    assert "looking down at the block" in a and "head bowed to the work" in a      # the work, before whom they face
    assert "not on the viewer" in b.split("woman")[0]                              # a head the scene turned away
    assert "looks at" in b and "not on the viewer" not in b.split("looks at")[1]   # one who faces the camera may
    assert w.count("look") == 2                                                    # nothing known: nothing said


def test_a_free_shots_moment_leads_its_prompt(tmp_path):
    import redraw as RD
    (tmp_path / "spec_round0.json").write_text(json.dumps({"cameras": [
        {"name": "s1", "free": True, "moment": "Five carts roll out of the gate."}, {"name": "s2", "free": True}]}))
    w = RD.moment_words(tmp_path, "s1")
    assert w.startswith("The subject of this picture") and "Five carts roll out of the gate." in w
    assert RD.moment_words(tmp_path, "s2") == "" and RD.FREE_STRENGTH > 0.9
    sh = {"framing": "wide", "kind": "action", "moment": "Five carts roll out of the gate.", "action": "x"}
    cam = FP.insert_camera(sh, [], {}, [20.0, 20.0], "s1", "id1", walls=False)
    assert cam["free"] and cam["moment"] == sh["moment"]


def test_a_close_shot_of_a_person_is_not_told_the_things_of_the_place(tmp_path):
    import redraw as RD
    (tmp_path / "spec_round0.json").write_text(json.dumps({"cameras": [
        {"name": "s1", "setting": "A lamp glows on the shelf."},
        {"name": "s2", "view": {"ahead": True, "words": "hills under cloud"}}]}))
    assert "lamp" in RD.setting_words(tmp_path, "s1") and RD.setting_words(tmp_path, "s1", face_only=True) == ""
    assert "hills" in RD.setting_words(tmp_path, "s2", face_only=True)      # what shows through glass still does
