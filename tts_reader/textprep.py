"""Text preprocessing to make prose read better aloud.

Applied to chapter text just before synthesis (on by default; disable with
``--no-preprocess``). Everything here is a deterministic, safe-for-speech
rewrite: nothing depends on a network or model, and paragraph breaks and
quotation marks are preserved so chapter detection and speaker segmentation
keep working on the result.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# User-editable pronunciation lexicon: {"word": "respelling"}. Respellings
# are written the way they should sound ("encourageable": "en-courage-able");
# useful mainly for F5, which reads raw characters and can stumble on rare
# words. Applied case-insensitively on word boundaries.
_LEXICON_PATH = Path(
    os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
) / "tts_reader" / "lexicon.json"
_lexicon_cache: dict | None = None


def _lexicon() -> dict[str, str]:
    global _lexicon_cache
    if _lexicon_cache is None:
        try:
            data = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
            _lexicon_cache = {
                str(k): str(v) for k, v in data.items()
            } if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _lexicon_cache = {}
    return _lexicon_cache

# Titles and honorifics that TTS engines otherwise read as "mr dot".
_ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miz",
    "Dr.": "Doctor",
    "St.": "Saint",
    "Prof.": "Professor",
    "Capt.": "Captain",
    "Col.": "Colonel",
    "Gen.": "General",
    "Lieut.": "Lieutenant",
    "Lt.": "Lieutenant",
    "Sgt.": "Sergeant",
    "Rev.": "Reverend",
    "Hon.": "Honorable",
    "Jr.": "Junior",
    "Sr.": "Senior",
}

_LATINISMS = [
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"&c\."), "et cetera"),  # the Gutenberg-era spelling
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\bviz\.", re.IGNORECASE), "namely"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
]

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(numeral: str) -> int:
    total, prev = 0, 0
    for ch in reversed(numeral.upper()):
        val = _ROMAN_VALUES[ch]
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


_HEADING_ROMAN = re.compile(
    r"\b(chapter|section|part|book|volume|act|scene)\s+([IVXLCDM]{1,7})\b(?![a-z])",
    re.IGNORECASE,
)

_ALL_CAPS_WORD = re.compile(r"\b[A-Z][A-Z’']{3,}\b")
_FOOTNOTE = re.compile(r"\[\d{1,3}\]|\[[a-z]\]")
_STAR_RULE = re.compile(r"^[\s*_\-•]{3,}$", re.MULTILINE)
_EMPHASIS = re.compile(r"_([^_\n]{1,120})_")
_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")
_SOFT_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")


def prepare_for_speech(text: str) -> str:
    """Rewrite *text* so a TTS engine reads it naturally."""
    # Rejoin words hyphen-split across line breaks, then unwrap hard-wrapped
    # lines (single newlines) while keeping paragraph breaks intact.
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _STAR_RULE.sub("", text)
    text = _SOFT_NEWLINE.sub(" ", text)

    text = _FOOTNOTE.sub("", text)
    text = _EMPHASIS.sub(r"\1", text)  # _italics_ markers -> bare words

    for abbr, spoken in _ABBREVIATIONS.items():
        text = text.replace(abbr, spoken)
    for pattern, spoken in _LATINISMS:
        text = pattern.sub(spoken, text)
    # "No. 7" -> "Number 7" (only when a digit follows, so plain "No." is safe)
    text = re.sub(r"\bNo\.\s*(?=\d)", "Number ", text)

    text = _HEADING_ROMAN.sub(
        lambda m: f"{m.group(1)} {_roman_to_int(m.group(2))}", text
    )

    # SHOUTED HEADINGS -> Shouted Headings (acronyms without vowels are kept,
    # so "HMS" still spells out but "ETYMOLOGY" reads as a word).
    text = _ALL_CAPS_WORD.sub(
        lambda m: m.group(0).capitalize() if re.search(r"[AEIOUY]", m.group(0)) else m.group(0),
        text,
    )

    for word, spoken in _lexicon().items():
        text = re.sub(rf"\b{re.escape(word)}\b", spoken, text, flags=re.IGNORECASE)

    text = text.replace("…", "...")
    text = re.sub(r"([!?])\1+", r"\1", text)  # "!!" reads fine as "!"
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:?!])", r"\1", text)  # tidy gaps left by removals
    return text.strip()
