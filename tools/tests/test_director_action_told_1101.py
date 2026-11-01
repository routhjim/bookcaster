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
