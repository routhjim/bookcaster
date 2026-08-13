"""Curated catalog of local Piper voices.

Each entry maps a short, friendly alias to a Piper voice "key" of the form
``<lang_code>-<voice_name>-<quality>``. These keys correspond to the models
hosted in the ``rhasspy/piper-voices`` collection and are downloaded on first
use (see :mod:`tts_reader.engine`).

To use a voice that is not in this catalog, you can always pass its full Piper
key directly on the command line, e.g. ``--voice en_US-kusal-medium``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    alias: str
    key: str          # Piper model key, e.g. "en_US-amy-medium"
    gender: str
    accent: str
    description: str


# A hand-picked selection covering a range of genders, accents and languages.
# All of these exist in the public rhasspy/piper-voices collection.
_VOICES = [
    # --- US English ---
    Voice("amy",        "en_US-amy-medium",        "Female", "US English", "Warm, natural everyday voice"),
    Voice("ryan",       "en_US-ryan-high",         "Male",   "US English", "Clear, confident narrator (high quality)"),
    Voice("lessac",     "en_US-lessac-medium",     "Neutral","US English", "Crisp and articulate, great for docs"),
    Voice("joe",        "en_US-joe-medium",        "Male",   "US English", "Relaxed, conversational tone"),
    Voice("kathleen",   "en_US-kathleen-low",      "Female", "US English", "Soft, lightweight voice (fast)"),
    Voice("hfc_female", "en_US-hfc_female-medium", "Female", "US English", "Balanced, professional female"),
    Voice("hfc_male",   "en_US-hfc_male-medium",   "Male",   "US English", "Balanced, professional male"),
    # --- UK English ---
    Voice("alan",       "en_GB-alan-medium",       "Male",   "UK English", "Calm British male"),
    Voice("jenny",      "en_GB-jenny_dioco-medium","Female", "UK English", "Friendly British female"),
    Voice("alba",       "en_GB-alba-medium",       "Female", "UK English", "Scottish-accented female"),
    # --- A few other languages (to show off multilingual support) ---
    Voice("thorsten",   "de_DE-thorsten-medium",   "Male",   "German",     "German male voice"),
    Voice("siwis",      "fr_FR-siwis-medium",      "Female", "French",      "French female voice"),
    Voice("davefx",     "es_ES-davefx-medium",     "Male",   "Spanish",     "Spanish (Spain) male voice"),
]

# alias -> Voice
VOICES = {v.alias: v for v in _VOICES}

# Alias used when the user does not pick one.
DEFAULT_VOICE = "amy"


def resolve_key(name: str) -> str:
    """Turn a user-supplied voice name into a Piper key.

    Accepts either a friendly alias from the catalog (e.g. ``amy``) or a raw
    Piper key (e.g. ``en_US-amy-medium``). Raw keys are passed through so any
    voice from the wider collection can be used.
    """
    name = name.strip()
    if name in VOICES:
        return VOICES[name].key
    # Looks like a full Piper key: lang_code-voice_name-quality
    if "-" in name and "_" in name.split("-", 1)[0]:
        return name
    known = ", ".join(sorted(VOICES))
    raise ValueError(
        f"Unknown voice '{name}'. Choose a catalog alias ({known}) "
        f"or pass a full Piper voice key like 'en_US-amy-medium'."
    )


# --- Orpheus voices (served by a local Orpheus TTS server) -----------------
# Orpheus ships a fixed set of 8 speakers; emotion tags like <laugh>, <sigh>,
# <chuckle>, <yawn> can be placed inline in the text.
_ORPHEUS_VOICES = [
    ("tara", "Female", "Conversational, clear (default)"),
    ("leah", "Female", "Warm, gentle"),
    ("jess", "Female", "Energetic, youthful"),
    ("leo",  "Male",   "Authoritative, deep"),
    ("dan",  "Male",   "Friendly, casual"),
    ("mia",  "Female", "Professional, articulate"),
    ("zac",  "Male",   "Enthusiastic, dynamic"),
    ("zoe",  "Female", "Calm, soothing"),
]
ORPHEUS_VOICES = {name for name, _, _ in _ORPHEUS_VOICES}
ORPHEUS_DEFAULT_VOICE = "tara"


def resolve_orpheus(name: str) -> str:
    """Validate an Orpheus voice name (unknown names pass through for finetunes)."""
    return name.strip()


def describe_orpheus() -> str:
    rows = [("VOICE", "GENDER", "DESCRIPTION")]
    rows.extend((n, g, d) for n, g, d in _ORPHEUS_VOICES)
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)).rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def describe_catalog() -> str:
    """Return a human-readable, aligned table of the catalog voices."""
    rows = [("ALIAS", "GENDER", "ACCENT", "PIPER KEY", "DESCRIPTION")]
    for v in _VOICES:
        rows.append((v.alias, v.gender, v.accent, v.key, v.description))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, r in enumerate(rows):
        line = "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(r))
        lines.append(line.rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
