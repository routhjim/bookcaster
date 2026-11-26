"""The figure drawing says the build the sheet was measured as (img3d/cast_char.py build_words); a photo person's
sheet takes nothing at the neck from the photographs; a figure older than its sheet is made again (story_keys)."""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
sys.path.insert(0, str(HERE.parent / "assets" / "img3d"))
sys.path.insert(0, str(HERE.parent.parent))


def test_the_drawing_says_the_build():
    import cast_char as CC
    slim = CC.build_words({"scores": {"build": "slim"}})
    assert "narrow waist" in slim and "nothing hanging wide" in slim
    assert "heavy" in CC.build_words({"scores": {"build": "very heavily built"}})
    assert CC.build_words({}) == "" and CC.build_words({"scores": {"build": "unknown"}}) == ""


def test_every_build_the_chooser_knows_has_words():
    import ast
    import cast_char as CC
    src = (HERE.parent / "assets" / "characters" / "choose_mh.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and n.targets[0].id == "BUILDS")
    assert {k.value for k in node.value.keys} == set(CC.BUILD_WORDS)


def test_a_photo_sheet_takes_nothing_at_the_neck():
    from tts_reader import video as V
    assert "nothing at the neck but what the description names" in V.sheet_prompt("a tall man in a grey coat", "ink", photos=True)
    assert "neck" not in V.sheet_prompt("a tall man in a grey coat", "ink")


def test_a_figure_older_than_its_sheet_is_made_again(tmp_path, monkeypatch):
    import story_keys as SK
    monkeypatch.setattr(SK.C, "LIBRARY", tmp_path / "lib")
    rec = tmp_path / "lib" / "character" / "tam_01" / "character.json"
    rec.parent.mkdir(parents=True)
    rec.write_text("{}")
    sheet = tmp_path / "tam.png"
    sheet.write_text("x")
    s = SK.Story.__new__(SK.Story)
    s.cast = {"Tam": "tam_01"}
    s.person = lambda n: {"sheet": str(sheet)}
    old = time.time() - 100
    os.utime(sheet, (old, old))
    assert not s._sheet_newer("Tam")
    os.utime(rec, (old - 100, old - 100))
    assert s._sheet_newer("Tam") and not s._sheet_newer("Orla")
