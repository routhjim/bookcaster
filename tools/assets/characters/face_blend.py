"""A face between the shipped presets: weights for a blend in the face model's coefficient space.

The comparison (face_pick.py) ranks the candidate presets against the drawn head by wins. Instead of taking the one
winner, the best few are mixed, each by its share of the wins, in the face model's own coefficients
(get_face_model_coefficients / set_face_model_coefficients; char_make.py's "face_blend" does the mixing in the
editor). The blended face then meets the unblended pick in the same comparison, the drawing in the middle, both
orders, and is kept only if it wins both (keep()).

Pure functions only: no engine, no model.
"""
from __future__ import annotations


def weights_from_wins(wins, pick=None, top=3, power=1.0):
    """{preset: weight}, summing to 1: the TOP candidates by wins (ties to the earlier name in WINS' own order),
    each weighted by its wins to the POWER; candidates with no win are left out. The pick (the comparison's
    winner) is always among them. One candidate, or no wins at all, gives {} (nothing to blend)."""
    order = {n: i for i, n in enumerate(wins)}
    ranked = sorted((n for n in wins if wins[n] > 0), key=lambda n: (-wins[n], order[n]))[:max(1, int(top))]
    if pick is not None and pick in wins and wins[pick] > 0 and pick not in ranked:
        ranked[-1] = pick
    if len(ranked) < 2:
        return {}
    raw = {n: float(wins[n]) ** float(power) for n in ranked}
    s = sum(raw.values())
    return {n: raw[n] / s for n in ranked}


def mix(coeffs, weights, scale=1.0):
    """The weighted sum of the presets' coefficient vectors (all of one length), times SCALE."""
    names = list(weights)
    if not names:
        raise ValueError("no weights")
    n = {len(coeffs[k]) for k in names}
    if len(n) != 1:
        raise ValueError(f"FACE BLEND: coefficient vectors differ in length: {sorted(n)}")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"FACE BLEND: weights sum to {sum(weights.values()):.6f}, not 1")
    out = [0.0] * n.pop()
    for k in names:
        w = weights[k]
        for i, x in enumerate(coeffs[k]):
            out[i] += w * x
    return [scale * x for x in out]


def max_delta(a, b):
    """The largest coefficient difference between two vectors of one length."""
    if len(a) != len(b):
        raise ValueError(f"vectors differ in length: {len(a)} and {len(b)}")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def keep(wins, name, need=2):
    """A changed face is kept only if it wins the comparison in both orders against the unchanged one."""
    return int((wins or {}).get(name, 0)) >= int(need)


def choose_face(blend=None, conform=None):
    """Which change the built character wears: a kept conform first (it replaces the coefficients altogether),
    else a kept blend, else the plain pick. Each is {"keep": bool, ...} or None."""
    if conform and conform.get("keep"):
        return "conform"
    if blend and blend.get("keep"):
        return "blend"
    return "pick"
