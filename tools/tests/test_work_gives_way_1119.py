"""A person at work takes the work's best side; one standing by with another mark steps to it (from_plan.give_way)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
import from_plan as FP  # noqa: E402

BENCH = {"name": "bench", "kind": "fixture", "center": [0.0, 0.0, 0.4], "half": [0.5, 0.25, 0.4]}


def test_nobody_in_the_way_moves_nobody():
    assert FP.give_way([0.0, -0.6], BENCH, {"b": {"position": [2.5, 0.0], "marks": [[2.5, 0.0], [0.0, -1.0]]}}) == {}


def test_one_in_the_way_steps_to_a_clear_mark():
    got = FP.give_way([0.0, -0.6], BENCH, {"b": {"position": [0.2, -1.0], "marks": [[0.2, -1.0], [0.1, -0.9], [1.6, -1.2]]}})
    assert got == {"b": [1.6, -1.2]}


def test_never_to_a_mark_between_the_worker_and_the_work():
    got = FP.give_way([0.0, -1.5], BENCH, {"b": {"position": [0.3, -1.4], "marks": [[0.3, -1.4], [1.2, 0.2]]}})
    assert got is None                       # the only other mark is nearer the work than the worker stands


def test_nowhere_to_go_is_said():
    assert FP.give_way([0.0, -0.6], BENCH, {"b": {"position": [0.2, -1.0], "marks": [[0.2, -1.0]]}}) is None


def test_two_who_step_aside_do_not_take_one_mark():
    oth = {"b": {"position": [0.2, -1.0], "marks": [[1.6, -1.2], [-1.6, -1.2]]},
           "c": {"position": [0.4, -1.1], "marks": [[1.6, -1.2], [-1.6, -1.2]]}}
    got = FP.give_way([0.0, -0.6], BENCH, oth)
    assert got and got["b"] != got["c"]
