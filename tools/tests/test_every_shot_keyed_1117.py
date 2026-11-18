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
    d.mkdir(parents=True)
    (d / "keys.json").write_text(json.dumps([{"shot": "A", "key": True}, {"shot": "B", "key": False}]))
    seen = []

    def step(name):
        def f(self, place):
            seen.append((name, SK.os.environ.get("SPEAKERS_FALLBACK_ALL")))
            if name == "keys":
                (d / "keys.json").write_text(json.dumps([{"shot": "A", "key": True}, {"shot": "B", "key": True}]))
            return True
        return f
    for n in ("stage", "redraw", "keys"):
        monkeypatch.setattr(SK.Story, f"step_{n}", step(n))
    st.restage_keyless("P")
    assert seen == [("stage", "1"), ("redraw", "1"), ("keys", "1")] and st.keyless("P") == []
    assert "SPEAKERS_FALLBACK_ALL" not in SK.os.environ
    seen.clear()
    st.restage_keyless("P")                                             # nothing keyless: nothing staged
    assert seen == []
