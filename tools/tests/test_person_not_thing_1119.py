"""One of the story's people is never a carried thing (scenes/place_spec.is_person)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
import place_spec as PS  # noqa: E402

NAMES = {"Ada Bell", "The Porter", "Tom"}


def test_a_named_person_is_no_thing():
    assert PS.is_person("Ada Bell", NAMES) and PS.is_person("ada bell.", NAMES)
    assert PS.is_person("porter", NAMES) and PS.is_person("the porter", NAMES)


def test_things_stay_things():
    assert not PS.is_person("lamp", NAMES) and not PS.is_person("Tom's coat", NAMES) and not PS.is_person("", NAMES)
