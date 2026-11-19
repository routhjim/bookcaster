"""The floor a camera is looked for on stays small however wide the ground is (scenes/speakers.py near_people)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
import speakers as S  # noqa: E402


def _subj(*xy):
    return {chr(97 + i): {"measured": {"pelvis_xy": list(p)}} for i, p in enumerate(xy)}


def test_wide_ground_keeps_its_standpoints_bounded():
    xr, yr = S.near_people(_subj((-30, -30), (0, 0), (30, 25)), [90.0, 90.0, 12.0])
    assert len(xr) * len(yr) <= S.FLOOR_POINTS * 1.3


def test_only_the_people_shown_set_the_floor():
    s = _subj((-30, -30), (0, 0), (30, 25))
    xr, yr = S.near_people(s, [90.0, 90.0, 12.0], keys=["b"])
    assert xr[0] >= -91 and xr[-1] <= 91 and yr[0] >= -91 and yr[-1] <= 91


def test_small_room_is_searched_as_before():
    xr, yr = S.near_people(_subj((0, 0), (1, 0)), [5.0, 4.0, 2.6])
    assert xr.step == 1 and yr.step == 1
    xr, yr = S.near_people(_subj((0, 0), (1, 0)), [12.0, 9.0, 3.0])
    assert xr.step == 2


def test_unknown_keys_fall_back_to_everyone():
    xr, _ = S.near_people(_subj((0, 0)), [5.0, 4.0, 2.6], keys=["zz"])
    assert len(xr) > 1
