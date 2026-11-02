"""11-01: a close-up's wall/curtains/chair-back background is a room's; outdoors it split the picture (a gothic window
beside the ash mountain behind Edric)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as F

BOOK = Path.home() / "audiobooks" / "knight_forged_v3"


def _shot(room, held="stand"):
    return {"id": "quiet-05m30s-003302_s0", "kind": "quiet", "framing": "close", "tight": True,
            "place": "The Burnt Mountain", "who": ["Edric"], "pose_held": {"Edric": held}, "room": room,
            "picture": "a close view of Edric's head and shoulders: Edric stands, facing camera, holding sword."}


def test_outdoor_closeup_has_no_room():
    if not BOOK.exists():
        return
    p = F._key_prompt(BOOK, "01_chapter_1", "quiet-05m30s-003302", _shot("Outside: black ash and grey rock."))
    assert "curtains" not in p and "open land and sky" in p
    p = F._key_prompt(BOOK, "01_chapter_1", "quiet-05m30s-003302", _shot("Outside: black ash.", "sit"))
    assert "chair back" not in p
    p = F._key_prompt(BOOK, "01_chapter_1", "quiet-05m30s-003302", _shot("Inside: a stone hall.", "stand"))
    assert "curtains and window" in p


if __name__ == "__main__":
    test_outdoor_closeup_has_no_room(); print("ok")
