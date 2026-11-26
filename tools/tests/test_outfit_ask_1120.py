"""The outfit question (collar and neckwear against the sheet) is asked only of a costume that has neckwear: the
sheet is asked once, the answer kept beside it."""
import ast
import json
import tempfile
from pathlib import Path

from PIL import Image

SRC = (Path(__file__).resolve().parents[1] / "assets" / "redraw" / "keycheck.py").read_text()


def _load():
    tree = ast.parse(SRC)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "has_neckwear")
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "COLLAR_Q")]
    ns = {"json": json, "Path": Path, "Image": Image}
    exec(compile(ast.Module(keep, []), "keycheck_part", "exec"), ns)
    return ns


class Judge:
    def __init__(self, answer):
        self.answer, self.asked = answer, 0

    def ask_images(self, imgs, q, n, size):
        self.asked += 1
        return self.answer


def test_sheet_is_asked_once_and_the_answer_kept():
    ns = _load()
    with tempfile.TemporaryDirectory() as d:
        sheet = Path(d) / "sheet.png"
        Image.new("RGB", (8, 8)).save(sheet)
        j = Judge("No.")
        assert ns["has_neckwear"](j, str(sheet)) is False
        assert ns["has_neckwear"](j, str(sheet)) is False and j.asked == 1
        assert ns["has_neckwear"](Judge("yes"), str(sheet)) is False           # (kept: not asked again)
        other = Path(d) / "other.png"
        Image.new("RGB", (8, 8)).save(other)
        assert ns["has_neckwear"](Judge("Yes"), str(other)) is True


def test_the_outfit_question_waits_on_it():
    assert 'has_neckwear(judge, p["sheet"])' in SRC


if __name__ == "__main__":
    test_sheet_is_asked_once_and_the_answer_kept()
    test_the_outfit_question_waits_on_it()
    print("all passed")
