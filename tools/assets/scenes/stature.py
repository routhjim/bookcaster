"""A child's height: the share of a grown figure's height at an age (stdlib).

The library's figures are built on a grown body. A person the book's look sheet gives an age under sixteen ("a girl
of about 7") is staged at that age's share of a grown height (median growth, rounded): the scene scales the figure,
and lifts it by what the scaling takes from the height of the seat or bed under it.
"""
from __future__ import annotations

import re

SHARE = [(1, 0.44), (2, 0.51), (4, 0.60), (6, 0.68), (8, 0.75), (10, 0.81), (12, 0.88), (14, 0.95), (16, 1.0)]


def age_of(look: str | None) -> int | None:
    m = re.search(r"\bof about (\d+)\b", look or "")
    return int(m.group(1)) if m else None


def scale(look: str | None) -> float:
    """1.0 for a grown person (or no stated age), else the age's share of a grown height."""
    age = age_of(look)
    if age is None or age >= SHARE[-1][0]:
        return 1.0
    if age <= SHARE[0][0]:
        return SHARE[0][1]
    for (a0, s0), (a1, s1) in zip(SHARE, SHARE[1:]):
        if a0 <= age <= a1:
            return round(s0 + (s1 - s0) * (age - a0) / (a1 - a0), 3)
    return 1.0
