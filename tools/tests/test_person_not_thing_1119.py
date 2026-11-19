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


def test_an_outfit_with_headwear_is_made_wearing_it_and_takes_no_separate_thing_on_the_head():
    import sys
    from pathlib import Path
    A = Path(__file__).resolve().parents[1] / "assets"
    sys.path[:0] = [str(A / "img3d"), str(A / "scenes")]
    import cast_char as CC
    import from_plan as FP
    look = "a man, wearing a mail coat, a steel helm with a nose guard and leather boots, carrying a spear"
    assert "helm" not in CC.spread_look(look) and "spear" not in CC.spread_look(look)
    kept = CC.spread_look(look, headwear=True)
    assert "steel helm" in kept and "mail coat" in kept and "spear" not in kept
    cam = {"carried": {"d": [{"asset": "x", "at": "head", "name": "helm"}, {"asset": "y", "at": "hand", "hand": "r"}],
                       "b": [{"asset": "z", "at": "head", "name": "hood"}]}}
    FP.not_on_head(cam, "d")
    assert cam["carried"] == {"d": [{"asset": "y", "at": "hand", "hand": "r"}], "b": [{"asset": "z", "at": "head", "name": "hood"}]}
    FP.not_on_head(cam, "b")
    assert "b" not in cam["carried"]


def test_headwear_taken_off_is_not_worn():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "img3d"))
    import cast_char as CC
    assert CC.head_worn("a mail coat, a steel cap and boots")
    assert not CC.head_worn("a mail coat, with the cap removed, and boots")
    assert not CC.head_worn("a mail coat and boots")
    assert "cap" not in CC.spread_look("a man, wearing a mail coat, with the cap removed, boots", headwear=True)
