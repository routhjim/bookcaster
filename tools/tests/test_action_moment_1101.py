"""11-01: an action shot's key is its MOMENT (the action under way), people by look only, no layout or body-plan words;
other shots keep the layout. And a lone adjective is not a garment ("the man in the dull")."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as F
import registry as RG

BOOK = Path.home() / "audiobooks" / "knight_forged_v3"
SHOT = {"id": "action-05m16s-003162_s0", "kind": "action", "framing": "close", "place": "The Burnt Mountain",
        "who": ["Vorgrath", "Edric"], "pose": {"Vorgrath": "standing", "Edric": "standing"},
        "picture": "Vorgrath stands before the cave on the left, facing right; Edric stands on the right, holding sword.",
        "moment": "Vorgrath's spiked tail sweeps in low from the left as Edric throws himself into a roll beneath it, "
                  "sword in hand.", "action": "Vorgrath's tail sweeps over Edric, who rolls clear."}


def test_moment_only_for_action():
    assert F.moment(SHOT).startswith("Vorgrath's spiked tail")
    assert F.moment(dict(SHOT, kind="quiet")) == ""
    assert F.moment(dict(SHOT, moment="")) == ""


def test_key_prompt_uses_moment():
    if not BOOK.exists():
        return
    p = F._key_prompt(BOOK, "01_chapter_1", "action-05m16s-003162", SHOT)
    assert "tail sweeps in low" in p and "stands before the cave" not in p
    assert "on the left" not in p.split("tail sweeps")[1] and "never upright" not in p
    q = F._key_prompt(BOOK, "01_chapter_1", "action-05m16s-003162", dict(SHOT, moment=""))
    assert "stands before the cave" in q


def test_describe_lone_adjective():
    if not BOOK.exists():
        return
    d = {"Edric": "Dull, unpolished star-iron armour consisting of a breastplate, worn over a jerkin"}
    assert F.describe(RG.Book(BOOK), "Edric", d) == "the man in the dull unpolished star-iron armour"
    assert F.describe(RG.Book(BOOK), "Princess Maren").endswith("stained with ash")


if __name__ == "__main__":
    test_moment_only_for_action(); test_key_prompt_uses_moment(); test_describe_lone_adjective(); print("ok")
