"""11-28: a wide shot is taken with the longest lens that holds it. Made-up people only."""
import sys
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("scenes", ""):
    sys.path.insert(0, str(A / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import speakers as S  # noqa: E402
import staging_synth as SY  # noqa: E402


def test_a_wide_of_three_people_is_not_the_widest_lens_where_a_longer_one_holds_it(monkeypatch):
    monkeypatch.setattr(S, "SOLIDS", [])
    monkeypatch.setattr(S, "OPEN_GROUND", [])
    monkeypatch.setattr(S, "PICTURE_SIDE", {})
    subj = {"a": SY.person([0.0, 1.0], (0.3, -1.0)), "b": SY.person([1.3, 0.8], (-0.6, -1.0)),
            "c": SY.person([-1.4, 1.4], (0.5, -1.0))}
    c = {"kind": "full", "show": ["a", "b", "c"], "listed": ["a", "b", "c"], "line_subject": "a"}
    shot = {"camera": SY.cam([0.0, -3.0, 1.6], [0.0, 1.0, 1.5], 35)}
    got = S.two_shot_candidates(c, shot, None, None, subj, [1152, 640], [12.0, 12.0, 3.0], None, None, [],
                                {"listed": ["a", "b", "c"], "line_subject": "a"})
    lenses = [q[0]["focal_mm"] for q in got]
    assert lenses and max(lenses) == 35
    assert sum(f >= 28 for f in lenses) > sum(f == 20 for f in lenses)      # (the wide end only where it is needed)
