"""Director rules (11-01 user): a told action is an "action" shot; a teller's lines play over the telling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as S


def test_action_covers_told_gestures():
    r = S.SHOT_RULES
    assert "reaches for or touches" in r and "with force" not in r


def test_told_lines_are_voice_over():
    t = S.RULE_TOLD.format(others="Helen's Bedroom")
    assert "NO talk shot" in t and "first" in t and "last" in t
    assert "talk shots stay with the teller" not in t


def test_video_words_name_the_side():
    assert S.on_side(["Doctor Grimesby Roylott", "Sherlock Holmes"], "Doctor Grimesby Roylott") == " on the left"
    assert S.on_side(["A", "B", "C"], "B") == " in the centre"
    assert S.on_side(["A"], "A") == "" and S.on_side(["A", "B"], "Z") == ""
    assert "NAME" in S.SHOT_RULES and "never by name" not in S.SHOT_RULES
