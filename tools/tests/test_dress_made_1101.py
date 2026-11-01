"""11-01: the keys' dress follows the made-outfit shift (Edric was drawn forging in his finished armour)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mannequin_flow as MF

BOOK = Path.home() / "audiobooks" / "knight_forged_v3"


def test_forging_in_jerkin():
    if not BOOK.exists():
        return
    assert "Edric" not in MF._dress(BOOK, "01_chapter_1", "action-02m49s-001695", ["Edric"])
    assert "Edric" in MF._dress(BOOK, "01_chapter_1", "action-04m32s-002720", ["Edric"])


if __name__ == "__main__":
    test_forging_in_jerkin(); print("ok")
