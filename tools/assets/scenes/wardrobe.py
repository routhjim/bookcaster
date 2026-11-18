"""A person's change of clothes over the story, for the engine flow (stdlib, pure).

The book's planner records a wardrobe per person: rows {"from": paragraph, "id": "<outfit name>", "dress": "<words>"}
in story order (the row in force at a paragraph is the last one that starts at or before it; none = the usual look).
Here those rows are put on the chapter's clock (each paragraph's start time) and named as figure variants: a variant
figure of base figure B for the outfit "<id>" is the library character B_<suffix(id)>, recorded with "variant_of": B.
"""
from __future__ import annotations

import re

SHOT_T = re.compile(r"-(\d{6})$")          # a director's shot id ends in its time, in tenths of a second


def suffix(wid: str) -> str:
    """An outfit name as a figure id's suffix: lower case, every run of other characters one underscore."""
    return re.sub(r"[^a-z0-9]+", "_", str(wid or "").lower()).strip("_")


def variant_id(base: str, wid: str) -> str:
    return f"{base}_{suffix(wid)}"


def shot_time(sid: str, fallback: float | None = None) -> float | None:
    """A shot's time in the chapter's recording, from its id ("quiet-05m01s-003017" = 301.7 s), else `fallback`."""
    m = SHOT_T.search(str(sid or ""))
    return int(m.group(1)) / 10.0 if m else fallback


def timed(rows: list, starts: list) -> list:
    """The rows on the chapter's clock: [{"t": the start of its paragraph, "variant", "id", "dress"}], in order."""
    out = []
    for r in rows or []:
        try:
            p = int(r["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if not starts or not str(r.get("id") or "").strip():
            continue
        p = min(max(p, 0), len(starts) - 1)
        out.append({"t": float(starts[p]), "variant": suffix(r["id"]), "id": r["id"], "dress": r.get("dress") or ""})
    return sorted(out, key=lambda x: x["t"])


def in_force(trows: list, t: float | None) -> dict | None:
    """The timed row in force at t (as a paragraph read at t: started at or before it); None = the usual look."""
    if t is None:
        return None
    rows = [r for r in trows or [] if r["t"] <= t + 0.05]
    return rows[-1] if rows else None


def variant_at(trows: list, t: float | None) -> str:
    r = in_force(trows, t)
    return r["variant"] if r else "usual"


def outer_garment(dress: str) -> str:
    """The outfit's first whole garment (an adjective standing alone joins the part after it; what is worn under it,
    "worn over ...", is left out): "Dull, unpolished star-iron armour consisting of ..." keeps the armour."""
    outer = re.split(r",?\s*\bworn\s+(?:over|under|beneath)\b", str(dress or ""), maxsplit=1)[0]
    lone = ""
    for x in re.split(r"[,;]", outer):
        x = re.sub(r"^(?:and|with)\s+", "", x.strip(), flags=re.I)
        if not x:
            continue
        if " " not in x:
            lone = f"{lone} {x}".strip()
            continue
        x = re.split(r"\s+\b(?:consisting|made|including|covering)\b", x, maxsplit=1)[0]
        return f"{lone} {x}".strip() if lone else x
    return lone


def variant_look(look: str, dress: str) -> str:
    """The person's look line with their clothes said as this outfit: the "wearing ..." clause replaced, up to what
    follows the clothes in a look line (expression, holding, face) or its end."""
    d = str(dress or "").strip().rstrip(".")
    if not d:
        return look
    d = d[0].lower() + d[1:]
    m = re.search(r"wearing (.+?)(?=, (?:expression|holding|face)\b|\.\s*$|$)", look or "")
    if not m:
        return (look or "").rstrip(". ") + f", wearing {d}."
    return look[:m.start(1)] + d + look[m.end(1):]


def sheet_for(worn_sheets: dict, wid: str) -> str | None:
    """The book's own drawn sheet of the person in this outfit ({what is worn: sheet}): the one named the same, else
    the one whose name's words are all in the outfit's name; None when the book has none."""
    s = suffix(wid)
    for n, q in (worn_sheets or {}).items():
        if suffix(n) == s:
            return q
    have = set(s.split("_"))
    hits = sorted((len(set(suffix(n).split("_"))), n, q) for n, q in (worn_sheets or {}).items()
                  if suffix(n) and set(suffix(n).split("_")) <= have)
    return hits[-1][2] if hits else None


def is_stance(suf: str) -> bool:
    """A figure's library suffix that names a stance (seated, lying, walking), not something worn."""
    return bool(re.search(r"(?:^|_)(?:seated|lying|walk)(?:_|$)", suf))
