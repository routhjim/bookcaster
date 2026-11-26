"""What a key's words call a person and say of them (assets/story_keys.py person_noun, look_person)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
import story_keys as SK  # noqa: E402


def test_a_person_is_called_what_the_look_calls_them():
    assert SK.person_noun("a tall, thin young man of 30 with grey eyes, wearing a green coat") == "man"
    assert SK.person_noun("a woman of about 40, wearing a man's coat") == "woman"
    assert SK.person_noun("an old girl of 9") == "girl"
    assert SK.person_noun("a dragon with green scales") == "dragon"


def test_the_person_is_told_without_what_they_wear_or_hold():
    look = "a tall, thin young man of 30 with grey eyes and a short beard, wearing a green coat, holding a lamp."
    assert SK.look_person(look) == "a tall, thin young man of 30 with grey eyes and a short beard"
    assert SK.look_person("a woman of about 40 carrying a basket") == "a woman of about 40"
