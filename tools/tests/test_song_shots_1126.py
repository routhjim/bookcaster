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


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
    print("ok")
