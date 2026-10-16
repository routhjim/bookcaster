"""Respell English text so a TTS voice reads it with a regional accent.

An accent lives mostly in its vowels (and rhoticity), not in dialect words,
so this works vowel by vowel: each word's pronunciation comes from espeak-ng
(General American IPA, so names and rare words work too), the vowels are
swapped by the accent's table — keyed by Wells' lexical sets (FACE, PRICE,
GOAT, MOUTH, BATH...) — plus a few accent-defining consonant rules
(TH-fronting, H-dropping, non-rhotic r), and any word that changed is
spelled out phonetically ("say" -> "sigh" for Cockney, "time" -> "toime"
for Irish). Words that don't change keep their spelling.

    respell("I traced her to your house.", "cockney")
"""

from __future__ import annotations

import functools
import re
import subprocess

# IPA (espeak en-us) -> lexical set. Diphthongs first when tokenising.
_VOWELS = {
    "eɪ": "FACE", "aɪ": "PRICE", "oʊ": "GOAT", "aʊ": "MOUTH", "ɔɪ": "CHOICE",
    "iː": "FLEECE", "i": "FLEECE", "uː": "GOOSE", "u": "GOOSE", "ɪ": "KIT",
    "ɛ": "DRESS", "æ": "TRAP", "ɑː": "LOT", "ɑ": "LOT", "ɔː": "THOUGHT",
    "ɔ": "THOUGHT", "ʊ": "FOOT", "ʌ": "STRUT", "ɜː": "NURSE", "ɝ": "NURSE",
    "ɚ": "LETTER", "ə": "COMMA", "ɐ": "COMMA", "ᵻ": "KIT", "ɪɹ": "NEAR",
    "ɛɹ": "SQUARE", "ɑːɹ": "START", "ɔːɹ": "NORTH", "ʊɹ": "CURE", "oːɹ": "NORTH",
}
_CONS = {
    "tʃ": "ch", "dʒ": "j", "θ": "th", "ð": "th", "ʃ": "sh", "ʒ": "zh",
    "ŋ": "ng", "ɹ": "r", "ɡ": "g", "j": "y", "ɾ": "t", "ʔ": "'", "x": "ch",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "f": "f", "v": "v",
    "s": "s", "z": "z", "h": "h", "m": "m", "n": "n", "l": "l", "w": "w",
}
# How each lexical set is spelled when nothing changes it.
_PLAIN = {
    "FACE": "ay", "PRICE": "y", "GOAT": "oh", "MOUTH": "ow", "CHOICE": "oy",
    "FLEECE": "ee", "GOOSE": "oo", "KIT": "i", "DRESS": "e", "TRAP": "a",
    "BATH": "a", "LOT": "ah", "THOUGHT": "aw", "FOOT": "oo", "STRUT": "u",
    "NURSE": "ur", "LETTER": "er", "COMMA": "uh", "NEAR": "eer",
    "SQUARE": "air", "START": "ar", "NORTH": "or", "CURE": "oor",
}
# Accent tables: lexical set -> spelling, and flags for consonant rules.
ACCENTS = {
    "scottish": {   # Scottish Standard English with Scots-leaning MOUTH
        "vowels": {"FACE": "eh", "GOAT": "oh", "MOUTH": "oo", "PRICE": "ey",
                   "FOOT": "u", "BATH": "a", "LOT": "o",
                   "THOUGHT": "o", "NURSE": "urr", "LETTER": "urr",
                   "NEAR": "eerr", "SQUARE": "ehrr", "START": "arr",
                   "NORTH": "orr", "CURE": "oorr"},
        "rhotic": True, "words": {"i": "Ah", "my": "ma"},
    },
    "irish": {      # Dublin / southern Irish
        "vowels": {"PRICE": "oi", "FACE": "eh", "GOAT": "oh", "MOUTH": "eow",
                   "STRUT": "oo", "THOUGHT": "ah", "LOT": "ah", "BATH": "ah",
                   "NURSE": "urr", "START": "ahr"},
        "rhotic": True, "th_stop": True,
    },
    "cockney": {    # London working class
        "vowels": {"FACE": "igh", "PRICE": "oi", "GOAT": "aow", "MOUTH": "ah",
                   "FLEECE": "ey", "GOOSE": "ew", "BATH": "ah", "LOT": "o",
                   "THOUGHT": "aw", "NURSE": "uh", "LETTER": "uh",
                   "NEAR": "ee-uh", "SQUARE": "eh", "START": "ah",
                   "NORTH": "aw", "CURE": "aw", "STRUT": "ah"},
        "rhotic": False, "th_front": True, "h_drop": True,
    },
    "australian": { # broad Australian
        "vowels": {"FACE": "igh", "PRICE": "oi", "GOAT": "aow", "MOUTH": "aeow",
                   "FLEECE": "uhee", "GOOSE": "ew", "BATH": "ah", "LOT": "o",
                   "NURSE": "er", "LETTER": "uh", "NEAR": "ee-uh",
                   "SQUARE": "eh", "START": "ah", "NORTH": "aw", "CURE": "aw"},
        "rhotic": False,
    },
    "english": {    # Received Pronunciation (posh southern English)
        "vowels": {"BATH": "ah", "GOAT": "eu", "LOT": "o", "THOUGHT": "aw",
                   "LETTER": "uh", "NURSE": "uh", "NEAR": "eeuh",
                   "SQUARE": "eh", "START": "ah", "NORTH": "aw", "CURE": "aw"},
        "rhotic": False,
    },
}
_TOKENS = sorted(list(_VOWELS) + list(_CONS), key=len, reverse=True)


@functools.lru_cache(maxsize=None)
def _ipa(word: str) -> str:
    out = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", word],
                         capture_output=True, text=True, timeout=10).stdout
    return re.sub(r"[ˈˌ\s]", "", out)


def _tokens(ipa: str) -> list[str]:
    toks, i = [], 0
    while i < len(ipa):
        for t in _TOKENS:
            if ipa.startswith(t, i):
                toks.append(t)
                i += len(t)
                break
        else:
            i += 1          # length marks, unknown symbols
    return toks


def _bath(toks: list[str], i: int) -> bool:
    """TRAP vowel in the BATH set (path, ask, dance, can't): before f, th,
    s, or n + t/s/ch (not 'and', 'hand')."""
    nxt = toks[i + 1:i + 3]
    return bool(nxt) and (nxt[0] in ("f", "θ", "s")
                          or (nxt[0] == "n" and len(nxt) > 1 and nxt[1] in ("t", "s", "tʃ")))


def respell_word(word: str, accent: str, ipa: str | None = None) -> str:
    rules = ACCENTS[accent]
    if word.lower() in ("a", "an", "the", "of", "to") and ipa is None:
        return word       # citation forms ('a' = eɪ) would be wrong in running speech
    if word.lower() == "a":
        return word       # the article; espeak gives it the letter name's FACE vowel
    fixed = rules.get("words", {}).get(word.lower())
    if fixed:
        return fixed if word[1:2] != "" or not word.isupper() else fixed
    toks = _tokens(ipa if ipa is not None else _ipa(word.lower()))
    if not toks:
        return word
    out, changed = [], False
    for i, t in enumerate(toks):
        if t in _VOWELS:
            lex = _VOWELS[t]
            if lex == "TRAP" and _bath(toks, i):
                lex = "BATH"
            plain = _PLAIN[lex]
            spell = rules["vowels"].get(lex, plain)
            if lex != "COMMA" and spell != plain:
                changed = True
            out.append(spell)
            continue
        if t == "ɹ" and not rules.get("rhotic", True):
            after_vowel = i and toks[i - 1] in _VOWELS
            before_vowel = i + 1 < len(toks) and toks[i + 1] in _VOWELS
            if after_vowel and not before_vowel:
                changed = True
                continue                 # non-rhotic: no r before a consonant/end
        if t in ("θ", "ð") and rules.get("th_front"):
            changed = True
            if t == "ð" and i == 0:
                out.append("th")         # 'that', 'the' keep their th
                continue
            out.append("f" if t == "θ" else "v")
            continue
        if t in ("θ", "ð") and rules.get("th_stop"):
            changed = True
            out.append("t" if t == "θ" else "d")
            continue
        if t == "h" and i == 0 and rules.get("h_drop"):
            changed = True
            out.append("'")
            continue
        out.append(_CONS.get(t, t))
    if not changed:
        return word
    spelled = "".join(out)
    if word[:1].isupper():
        spelled = spelled[:1].upper() + spelled[1:]
    return spelled


def respell(text: str, accent: str) -> str:
    """Respell every word of *text* for *accent* (keys of ACCENTS); keeps
    punctuation, spacing and capitalisation; words in [tags] are left alone."""
    if accent not in ACCENTS:
        raise ValueError(f"unknown accent {accent!r}; known: {', '.join(ACCENTS)}")
    out = []
    # Sentence by sentence, so function words get their in-context (reduced)
    # pronunciations ('to' = tə, 'a' = ə), not citation forms.
    for sent in re.split(r"(?<=[.!?;:])(\s+)", text):
        parts = re.split(r"(\[[^\]]*\]|[A-Za-z][A-Za-z']*)", sent)
        words = [p for p in parts if re.fullmatch(r"[A-Za-z][A-Za-z']*", p)]
        ipas = _sentence_ipa(" ".join(words)) if words else []
        if len(ipas) != len(words):
            ipas = [None] * len(words)        # alignment failed: word by word
        k = 0
        for p in parts:
            if re.fullmatch(r"[A-Za-z][A-Za-z']*", p):
                out.append(respell_word(p, accent, ipas[k]))
                k += 1
            else:
                out.append(p)
    return "".join(out)


@functools.lru_cache(maxsize=4096)
def _sentence_ipa(words: str) -> list[str]:
    """espeak's IPA for a run of words, one entry per word (espeak keeps
    word boundaries as spaces when given no punctuation)."""
    raw = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", words],
                         capture_output=True, text=True, timeout=10).stdout
    return [re.sub(r"[ˈˌ]", "", w) for w in raw.split()]
