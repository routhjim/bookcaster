"""A place of wealth or rule is dressed to its station before it is drawn and built; a plain one is left as it is
(assets/scenes/place_dress.py)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets" / "scenes"))
sys.path.insert(0, str(HERE.parent / "assets"))
import place_dress as PD  # noqa: E402


def test_only_a_place_of_means_gets_dressing():
    rich = PD.read('Here: {"owner": "a merchant", "use": "counting house", "station": "Rich", '
                   '"dressing": ["woven hangings, red.", "a brass lamp", ""]}')
    assert rich["station"] == "rich" and rich["dressing"] == ["woven hangings, red", "a brass lamp"]
    plain = PD.read('{"owner": "a cooper", "use": "workshop", "station": "plain", "dressing": ["gilt mirror"]}')
    assert plain["dressing"] == []                                   # whatever the answer lists
    assert PD.read("no json at all") == {"owner": "", "use": "", "station": "plain", "dressing": []}
    many = PD.read(json.dumps({"station": "royal", "dressing": [f"thing {i}" for i in range(40)]}))
    assert len(many["dressing"]) == PD.MOST


def test_the_question_carries_the_place_and_asks_no_story():
    q = PD.prompt({"place": "The Counting House", "period": "late_medieval", "_director_room": "Inside: a long room."})
    assert "The Counting House" in q and "late medieval" in q and "a long room." in q and "Inside:" not in q
    assert "No people" in q and "never against it" in q


def test_the_flow_draws_a_dressed_picture_from_the_places_own(tmp_path, monkeypatch):
    import story_keys as SK
    from PIL import Image
    s = SK.Story.__new__(SK.Story)
    s.work, s.book, s.prefix = tmp_path / "w", tmp_path / "book", "x"
    s.book.mkdir()
    s.places = {"Room": {"slug": "room", "set": "x_room_01"}}
    d = s.d("Room")
    (d / "place_spec.json").write_text(json.dumps({"place": "Room", "_director_room": "Inside: a long room."}))
    hero = tmp_path / "hero.png"
    Image.new("RGB", (64, 32)).save(hero)
    monkeypatch.setattr(SK.C, "LIBRARY", tmp_path / "lib")
    calls = []

    def fake_llm(jid, logf, cmds):
        (d / "dress.json").write_text(json.dumps(station))
    def fake_line(station_, kind, jid, logf, argv):
        calls.append([str(x) for x in argv])
        pic = tmp_path / "lib" / "_refs" / argv[3] / "picture.png"
        pic.parent.mkdir(parents=True)
        Image.new("RGB", (64, 32)).save(pic)
    monkeypatch.setattr(SK, "llm", fake_llm)
    monkeypatch.setattr(SK, "line", fake_line)
    station = {"station": "plain", "use": "store", "dressing": []}
    assert s.dressed("Room", hero) is None and not calls
    station = {"station": "rich", "use": "counting house", "dressing": ["woven hangings, red", "a brass lamp"]}
    pic = s.dressed("Room", hero)
    assert pic and pic.exists() and len(calls) == 1
    words = calls[0][calls[0].index("--desc") + 1]
    assert "woven hangings, red; a brass lamp" in words and "a long room." in words and "no people" in words.lower()
    assert calls[0][calls[0].index("--sources") + 1] == str(hero)      # drawn from the place's own picture
    assert s.dressed("Room", hero) == pic and len(calls) == 1          # the same words: the same picture, not drawn again
