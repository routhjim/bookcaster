"""11-27: a whole figure is built wearing what its usual look puts on its head, as a wardrobe variant's is
(story_keys._cast_cmd, cast_char --headwear). Built bare-headed, a man's hat was the redraw's to invent in every key:
on in the wide shots, gone in the close ones. Made-up looks only."""
import sys
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "assets"
sys.path.insert(0, str(A / "img3d"))
import cast_char as CC  # noqa: E402

LOOK = "a woman of about 30, wearing a grey cloak, a blue sash, and a straw hat tied with a green cord"


def test_the_drawing_keeps_the_hat_and_is_not_told_bare_headed():
    kept = CC.spread_look("fig, " + LOOK, headwear=True)
    assert "straw hat" in kept and "straw hat" not in CC.spread_look("fig, " + LOOK)
    assert CC.BARE_HEAD in CC.POSE_WORDS and "bare-headed" not in CC.POSE_WORDS.replace(CC.BARE_HEAD, "")
    src = (A / "img3d" / "cast_char.py").read_text()
    assert "POSE_WORDS if not hw_ else POSE_WORDS.replace(BARE_HEAD, '')" in src


def test_a_base_figure_is_asked_for_with_its_headwear_and_made_again_once(tmp_path):
    import story_keys as SK
    class K(SK.Story if hasattr(SK, "Story") else object):                  # noqa: E301
        def __init__(self, look):
            self._look, self.whole = look, True
        def person(self, name):                                            # noqa: E301
            return {"look": self._look}
        def _built(self, name):                                            # noqa: E301
            return True
    cls = next(v for v in vars(SK).values() if isinstance(v, type) and hasattr(v, "_head_due"))
    k = type("T", (cls,), {"__init__": K.__init__, "person": K.person, "_built": K._built})(LOOK)
    assert k._head_due("Orla", tmp_path)
    k._head_mark("Orla", tmp_path)
    assert (tmp_path / "headwear_drawn").exists() and not k._head_due("Orla", tmp_path)       # (once)
    bare = type("T", (cls,), {"__init__": K.__init__, "person": K.person, "_built": K._built})(
        "a man of about 50, wearing a grey cloak, his hat in his hands")
    assert not bare._head_due("Tam", tmp_path / "x")
    src = (A / "story_keys.py").read_text()
    assert src.count('["--whole", "--headwear"] if getattr(self, "whole", True)') == 2
