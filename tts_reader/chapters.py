"""Heuristic chapter detection for plain-text books.

Splits a document into chapters by looking for heading lines such as
``CHAPTER 1. Loomings.``, ``Part II``, ``Prologue`` etc. The detection is
deliberately simple and works well on Project Gutenberg-style texts; use
``--list-chapters`` to preview the result and ``--chapter-regex`` to override
the pattern for unusual books.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Lines that begin a new chapter. Anchored to the start of a line (MULTILINE),
# keyed off common structural words so ordinary prose is not misread.
DEFAULT_HEADING = (
    r"^\s{0,3}("
    r"chapter|chap\.|part|book|section|canto|volume|"
    r"prologue|epilogue|introduction|preface|foreword|afterword"
    r")\b.*$"
)

# Titles we generate ourselves (not real headings) and therefore do not speak.
SYNTHETIC_TITLES = {"Front Matter", "Full Text"}


@dataclass
class Chapter:
    title: str
    text: str


def _clean_title(raw: str) -> str:
    return " ".join(raw.split()).strip()


def detect_chapters(
    text: str,
    pattern: str = DEFAULT_HEADING,
    min_body_chars: int = 24,
) -> list[Chapter]:
    """Split ``text`` into a list of :class:`Chapter`.

    Segments whose body is essentially empty (``< min_body_chars``) — i.e. a
    heading line with no prose after it, typically a table-of-contents entry —
    are merged into the preceding chapter so they do not become spurious
    one-line "chapters". The threshold is intentionally small so genuinely
    short chapters are preserved. If no headings are found the whole document
    is returned as a single chapter.
    """
    regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    matches = list(regex.finditer(text))
    if not matches:
        return [Chapter("Full Text", text.strip())]

    segments: list[Chapter] = []

    front = text[: matches[0].start()].strip()
    if front:
        segments.append(Chapter("Front Matter", front))

    for i, m in enumerate(matches):
        title = _clean_title(m.group(0))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        segments.append(Chapter(title, body))

    # Merge tiny segments (e.g. a contents list) into whatever precedes them.
    merged: list[Chapter] = []
    for ch in segments:
        if merged and len(ch.text) < min_body_chars:
            prev = merged[-1]
            tail = ch.title + (("\n" + ch.text) if ch.text else "")
            merged[-1] = Chapter(prev.title, (prev.text + "\n\n" + tail).strip())
        else:
            merged.append(ch)

    return merged
