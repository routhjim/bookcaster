"""The place's shot list follows the plan (11-26): a list left by another stretch of the chapter is not done."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
import story_keys as SK  # noqa: E402


def test_a_list_for_other_shots_is_not_this_plans():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "shots.txt"
        assert not SK.shots_listed(f, ["s1", "s2"])
        f.write_text("s1\ns2\n")
        assert SK.shots_listed(f, ["s1", "s2"])
        assert not SK.shots_listed(f, ["s1", "s2", "s3"]) and not SK.shots_listed(f, ["s2", "s1"])
        assert not SK.shots_listed(f, ["s1"])


def test_the_shots_step_is_done_only_with_the_plans_own_list():
    class S(SK.Story):
        def __init__(self, d, ids):
            self._d, self._ids = d, ids

        def d(self, place):
            return self._d

        def shots_of(self, place):
            return self._ids
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        s = S(d, ["s1", "s2"])
        (d / "shots.txt").write_text("s1\ns2\n")
        s.mark("the Hall", "shots")
        s.mark("the Hall", "set")
        assert s.done("the Hall", "shots") and s.done("the Hall", "set")
        s._ids = ["s1", "s2", "s3", "s4"]
        assert not s.done("the Hall", "shots") and s.done("the Hall", "set")
