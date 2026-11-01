"""Close-up room glimpse keeps a sentence's non-seat items; poker glossed (10-31)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as S


def test_seat_item_dropped_rest_kept():
    t = "a Victorian study with a fireplace, a desk, a leather armchair, and a window overlooking London."
    out = S._no_seats(t)
    assert "armchair" not in out and "fireplace" in out and "London" in out


def test_no_seat_unchanged():
    assert S._no_seats("The room is warm.") == "The room is warm."


def test_seat_in_head_drops_sentence():
    assert S._no_seats("A narrow bed against the wall.") == ""


def test_poker_gloss():
    k, v = next((k, v) for k, v in S.GLOSS.items() if "poker" in k)
    assert "iron" in re.sub(k, v, "holding a bent poker")
