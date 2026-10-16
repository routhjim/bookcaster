"""Dialogue detection and speaker attribution.

Splits prose into narration and quoted dialogue, then works out who is
speaking. Two layers:

1. Heuristics (always on): speech-verb patterns adjacent to a quote, e.g.
   ``"...," said Ahab`` / ``Starbuck cried, "..."``. Free and offline.
2. An optional LLM pass (``--llm-url``): remaining unattributed quotes are
   sent, with surrounding context, to any OpenAI-compatible chat endpoint
   (e.g. a local llama.cpp server) which names the speaker.

Nothing here touches audio; :class:`~tts_reader.engine.CastEngine` consumes
the segments and picks a voice per speaker.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .textprep import unspell


@dataclass
class Segment:
    """A run of text spoken by one voice, in reading order."""

    kind: str  # "narration" | "quote"
    text: str
    speaker: Optional[str] = None  # normalized speaker name, quotes only
    emotion: Optional[str] = None  # Orpheus emotion tag for this quote, if any
    # Fish: delivery tags by sentence index within the quote ({0: "angry",
    # 3: "shouting"}); a tag holds until the next one. See fish_sentences().
    sentence_tags: Optional[dict] = None
    # False when the paragraph ended inside this quote: the book's way of
    # saying the same speaker carries on into the next paragraph.
    closed: bool = True


# Verbs that introduce or tag speech. Ordered alternation inside one group.
_SPEECH_VERBS = (
    "said|cried|replied|asked|answered|exclaimed|shouted|muttered|whispered|"
    "continued|added|returned|observed|remarked|repeated|resumed|murmured|"
    "roared|thundered|growled|sighed|yelled|called|interposed|interrupted|"
    "demanded|inquired|urged|insisted|declared|announced|protested|admitted|"
    "agreed|assented|suggested|pleaded|commanded|ordered|bellowed|screamed|"
    "hissed|snapped|gasped|stammered|faltered|persisted|rejoined|retorted|"
    "echoed|laughed|sneered|scoffed|groaned|grumbled|panted|drawled|began"
)

_HONORIFIC = r"(?:Captain|Mr|Mrs|Miss|Dr|Old|Aunt|Uncle|Father|Sir|Lady|Don|Cook)\.?\s+"
# A proper name: optional honorific + one or two capitalized words.
# Hyphens allowed: lexicon respellings ("Star-buk") must be captured whole.
_NAME = rf"((?:{_HONORIFIC})?[A-Z][a-zA-Z'’]+(?:-[a-zA-Z'’]+)*(?:\s+[A-Z][a-zA-Z'’]+)?)"

# "... ," said Ahab   /  "... !" cried old Starbuck
_TAG_VERB_NAME = re.compile(rf"^\W{{0,4}}(?:{_SPEECH_VERBS})\s+{_NAME}")
# "... ," Ahab said   /  "... " Stubb muttered
_TAG_NAME_VERB = re.compile(rf"^\W{{0,4}}{_NAME}\s+(?:{_SPEECH_VERBS})\b")
# Ahab said(,|:)? "..."   — searched at the END of the narration before a quote.
# First person: ', said I.' / 'I asked, "..."' (the name pattern needs 2+ letters).
_TAG_VERB_I = re.compile(rf"^\W{{0,4}}(?:{_SPEECH_VERBS})\s+I\b")
_TAG_I_VERB = re.compile(rf"^\W{{0,4}}I\s+(?:{_SPEECH_VERBS})\b")
_LEAD_I_VERB = re.compile(rf"\bI\s+(?:{_SPEECH_VERBS})[^.!?]{{0,40}}[,:]?\s*$")
_LEAD_NAME_VERB = re.compile(
    rf"{_NAME}\s+(?:{_SPEECH_VERBS})[^.!?]{{0,40}}[,:]?\s*$"
)
_LEAD_VERB_NAME = re.compile(
    rf"(?:{_SPEECH_VERBS})\s+{_NAME}[^.!?]{{0,20}}[,:]?\s*$"
)

_STRIP_HONORIFIC = re.compile(rf"^(?:{_HONORIFIC})", re.IGNORECASE)

# Words that pattern-match a name but are never speakers.
_NOT_NAMES = {
    "the", "he", "she", "i", "they", "we", "you", "it", "one", "all", "aye",
    "oh", "but", "and", "then", "now", "so", "yes", "no", "what", "who", "god",
    "lord", "heaven", "thou", "thee", "there", "here", "this", "that",
}


# Speaker name for the first-person narrator's own lines. The converter's
# voice lookup resolves it to the narrator voice (voice-map key "narrator").
NARRATOR = "Narrator"


def _cap_words(text: str) -> str:
    """Capitalize each word's first letter only: keeps "McGregor" and
    "Scrooge's" intact, where str.title() gives "Mcgregor" / "Scrooge'S"."""
    return " ".join(w[:1].upper() + w[1:] for w in text.split())


def normalize_name(raw: str) -> str:
    """Canonical speaker key: honorific stripped, words capitalized."""
    name = _STRIP_HONORIFIC.sub("", raw.strip()).strip()
    name = re.sub(r"[’']s$", "", name)  # possessive slips
    name = unspell(name)  # "Star-buk" (lexicon respelling) -> "Starbuck"
    return _cap_words(name) if name else _cap_words(raw.strip())


def _tag_speaker(m: re.Match, text: str) -> str:
    """Speaker for a speech-tag match. A possessive names an owner, not the
    speaker: 'repeated Scrooge's nephew' -> "Scrooge's Nephew"."""
    name = m.group(1)
    if name == "I":
        # First-person narrator ("said I"): voiced by the narrator, the way a
        # human reader voices the book's "I".
        return NARRATOR
    if re.search(r"[’']s$", name):
        noun = re.match(r"\s+(?:(?:old|young|little|poor)\s+)?([a-z]{3,})", text[m.end():])
        if noun:
            return f"{normalize_name(name)}'s {noun.group(1).capitalize()}"
    return normalize_name(name)


def _valid_name(raw: str) -> bool:
    return normalize_name(raw).lower() not in _NOT_NAMES


def segment_dialogue(text: str) -> list[Segment]:
    """Split *text* into narration/quote segments with heuristic speakers.

    Quotes never cross paragraph boundaries; an unclosed quote is closed at
    the end of its paragraph (books often re-open quotes each paragraph).
    Handles curly (“ ”) and straight (") quotation marks.
    """
    segments: list[Segment] = []
    for para in re.split(r"\n{2,}", text):
        if not para.strip():
            continue
        segments.extend(_segment_paragraph(para))
        # Paragraph break belongs to the narrator's pacing.
        if segments and segments[-1].kind == "narration":
            segments[-1].text += "\n\n"
        else:
            segments.append(Segment("narration", "\n\n"))
    _attribute_heuristic(segments)
    return segments


def _segment_paragraph(para: str) -> list[Segment]:
    out: list[Segment] = []
    buf: list[str] = []
    in_quote = False
    for ch in para:
        if ch == "“" or (ch == '"' and not in_quote):  # open
            if buf and "".join(buf).strip("\n"):
                out.append(Segment("narration" if not in_quote else "quote", "".join(buf)))
            elif buf:
                out.append(Segment("narration", "".join(buf)))
            buf = []
            in_quote = True
        elif ch == "”" or (ch == '"' and in_quote):  # close
            if buf:
                out.append(Segment("quote", "".join(buf)))
            buf = []
            in_quote = False
        else:
            buf.append(ch)
    if buf:
        out.append(Segment("quote" if in_quote else "narration", "".join(buf),
                           closed=not in_quote))
    return out


def _attribute_heuristic(segments: list[Segment]) -> None:
    """Fill Segment.speaker from speech-verb patterns in adjacent narration."""
    for i, seg in enumerate(segments):
        if seg.kind != "quote" or seg.speaker:
            continue
        # Look at the narration right after the quote: ', said Ahab'
        if i + 1 < len(segments) and segments[i + 1].kind == "narration":
            after = segments[i + 1].text
            m = _TAG_VERB_NAME.search(after) or _TAG_NAME_VERB.search(after)
            if not m and (_TAG_VERB_I.search(after) or _TAG_I_VERB.search(after)):
                seg.speaker = NARRATOR
            if m and _valid_name(m.group(1)):
                seg.speaker = _tag_speaker(m, after)
            if seg.speaker:
                # '"...," said Ahab, "..."' — the tag also covers the next
                # quote, but only within the paragraph: a new paragraph is
                # (nearly always) a new speaker.
                if (
                    len(after.strip()) < 60
                    and "\n\n" not in after
                    and i + 2 < len(segments)
                    and segments[i + 2].kind == "quote"
                    and not segments[i + 2].speaker
                ):
                    segments[i + 2].speaker = seg.speaker
                continue
        # Look at the narration right before: 'Ahab said, "..."'
        if i > 0 and segments[i - 1].kind == "narration":
            before = segments[i - 1].text[-80:]
            m = _LEAD_NAME_VERB.search(before) or _LEAD_VERB_NAME.search(before)
            if m and _valid_name(m.group(1)):
                seg.speaker = _tag_speaker(m, before)
            elif _LEAD_I_VERB.search(before):
                seg.speaker = NARRATOR


def character_counts(segments: list[Segment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seg in segments:
        if seg.kind == "quote" and seg.speaker:
            counts[seg.speaker] = counts.get(seg.speaker, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Attribution cache: LLM answers are expensive, so they persist in a sidecar
# file and are reused across cast sessions, characters runs, and conversions.
# Keys combine the quote text with a tail of the preceding segment, making
# them stable whether the book is parsed whole or chapter-by-chapter.
# ---------------------------------------------------------------------------

def _segment_key(segments: list[Segment], i: int) -> str:
    prev = segments[i - 1].text[-60:] if i else ""
    raw = re.sub(r"\s+", " ", f"{prev}||{segments[i].text}").strip()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class AttributionCache:
    """Speaker attributions persisted next to the book (JSON sidecar)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.speakers: dict[str, str] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version") == 1:
                self.speakers = {
                    str(k): str(v) for k, v in data.get("speakers", {}).items()
                }
        except (OSError, ValueError):
            pass  # missing or corrupt cache: start fresh

    def apply(self, segments: list[Segment]) -> int:
        hits = 0
        for i, seg in enumerate(segments):
            if seg.kind == "quote" and not seg.speaker:
                name = self.speakers.get(_segment_key(segments, i))
                if name:
                    seg.speaker = name
                    hits += 1
        return hits

    def update(self, segments: list[Segment]) -> None:
        for i, seg in enumerate(segments):
            if seg.kind == "quote" and seg.speaker:
                self.speakers[_segment_key(segments, i)] = seg.speaker

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"version": 1, "speakers": self.speakers},
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass  # a failed cache write must never break synthesis


# ---------------------------------------------------------------------------
# Character roster: who the book names with a title ("Miss Stoner", "Dr.
# Grimesby Roylott"). Titles give gender, full forms give aliases, and the
# roster lets the LLM name characters the narration only calls "the old man".
# ---------------------------------------------------------------------------

_TITLE_GENDER = {
    "mister": "M", "mr": "M", "sir": "M", "lord": "M", "colonel": "M",
    "master": "M", "mrs": "F", "missus": "F", "miss": "F", "ms": "F",
    "lady": "F", "madam": "F", "madame": "F",
    "doctor": "", "dr": "", "captain": "", "professor": "",
}
# Words that precede surnames like first names but are ranks/titles.
_RANKS = {"general", "major", "inspector", "sergeant", "reverend", "saint",
          "father", "brother", "sister", "uncle", "aunt", "old", "young",
          "little", "poor", "dear", "late", "enter", "exit", "exeunt"}
_ROSTER_RE = re.compile(
    r"\b(Mister|Mr|Mrs|Missus|Miss|Ms|Doctor|Dr|Sir|Lady|Lord|Captain|"
    r"Colonel|Master|Madame?|Professor)\.?\s+"
    r"((?:[A-Z][a-zA-Z'’-]+\s+){0,2}[A-Z][a-zA-Z'’-]+)"
)


@dataclass
class RosterEntry:
    name: str                      # surname: the short name speakers use
    mentions: int = 0
    gender: str = ""               # "M", "F", or "" when titles disagree/absent
    aliases: set = field(default_factory=set)
    firsts: set = field(default_factory=set)   # first names seen with it
    _votes: dict = field(default_factory=dict)

    def describe(self) -> str:
        g = {"M": "male", "F": "female"}.get(self.gender, "")
        extra = sorted(self.aliases, key=len, reverse=True)[:2]
        bits = [b for b in (g, "also " + " / ".join(extra) if extra else "") if b]
        return f"{self.name} ({'; '.join(bits)})" if bits else self.name


def character_roster(text: str, min_mentions: int = 2) -> dict[str, RosterEntry]:
    """Titled names in *text*, keyed by surname, most-mentioned first."""
    entries: dict[str, RosterEntry] = {}
    for m in _ROSTER_RE.finditer(text):
        title = m.group(1).lower()
        words = [unspell(re.sub(r"[’']s$", "", w)) for w in m.group(2).split()]
        surname = words[-1].title()
        if surname.lower() in _NOT_NAMES or surname.lower() in _TITLE_GENDER:
            continue
        e = entries.setdefault(surname, RosterEntry(name=surname))
        e.mentions += 1
        e.aliases.add(f"{m.group(1)} {' '.join(words)}")
        if len(words) > 1:
            e.firsts.add(words[0].title())
        g = _TITLE_GENDER.get(title, "")
        if g:
            e._votes[g] = e._votes.get(g, 0) + 1
    if entries:
        # Untitled full forms of known surnames: "Sherlock Holmes", "Helen Stoner".
        pat = re.compile(r"\b([A-Z][a-z]+)\s+(" + "|".join(map(re.escape, entries)) + r")\b")
        for m in pat.finditer(text):
            first = m.group(1)
            if (first.lower() in _NOT_NAMES or first.lower() in _TITLE_GENDER
                    or first.lower() in _RANKS):
                continue
            e = entries[m.group(2)]
            e.firsts.add(first)
            e.aliases.add(f"{first} {m.group(2)}")
    for e in entries.values():
        if e._votes:
            top = max(e._votes, key=e._votes.get)
            # Mixed titles (Mr. and Mrs. Smith) leave gender to the LLM, and
            # so do mostly-neutral ones: "Dr. Roylott" x10 + "Mrs. Roylott"
            # (his late wife) x2 is not evidence he is a woman.
            gendered = sum(e._votes.values())
            if e._votes[top] >= 0.8 * gendered and gendered >= 0.5 * e.mentions:
                e.gender = top
    kept = [e for e in entries.values() if e.mentions >= min_mentions]
    return {e.name: e for e in sorted(kept, key=lambda e: -e.mentions)}


def canonical_speaker(name: str, roster: dict[str, RosterEntry]) -> str:
    """Merge alias forms into one speaker: "Sherlock Holmes" -> "Holmes",
    unless the surname is shared ("Helen Stoner" and "Julia Stoner" stay)."""
    words = name.split()
    # A surname carried by both genders ("Mr." and "Mrs. Bennet") is a
    # family, not a person: never fold first names into it.
    single = lambda e: len(e._votes) <= 1
    if len(words) > 1:
        e = roster.get(words[-1])
        if e and single(e) and (e.firsts <= {words[0]}
                                or words[0].lower() in _TITLE_GENDER):
            return e.name   # "Sherlock Holmes" / "Doctor Livesey" -> surname
    elif len(words) == 1:
        owners = [e for e in roster.values() if words[0] in e.firsts]
        if len(owners) == 1 and single(owners[0]) and len(owners[0].firsts) == 1:
            return owners[0].name   # "Sherlock" alone -> "Holmes"
    return name


def attribute_segments(
    segments: list[Segment], attributor: "LlmAttributor | None" = None,
    cache_path: str | Path | None = None, log=print,
) -> dict[str, RosterEntry]:
    """Cache -> LLM -> cache pipeline on top of the heuristic attributions.

    Returns the book's character roster (see :func:`character_roster`).
    """
    roster = character_roster("".join(s.text for s in segments))
    cache = AttributionCache(cache_path) if cache_path else None
    if cache is not None:
        hits = cache.apply(segments)
        if hits and log:
            log(f"  {hits} speaker attribution(s) restored from cache")
    if attributor is not None:
        heard = [n for n in character_counts(segments) if n != NARRATOR]
        known = [roster[n].describe() if n in roster else n for n in heard]
        known += [e.describe() for n, e in roster.items() if n not in heard]
        attributor.refine(segments, known=known, log=log)
    if cache is not None:
        # Cache raw names; alias merging is re-derived on every load, so
        # roster improvements apply to old caches without new LLM calls.
        cache.update(segments)
        cache.save()
    for seg in segments:
        if seg.speaker:
            seg.speaker = canonical_speaker(seg.speaker, roster)
    fixed = enforce_dialogue_conventions(segments)
    if fixed and log:
        log(f"  {fixed} attribution(s) corrected by quotation conventions")
    return roster


# Any speech tag right after a quote: ', said my friend.' / ', she cried,'
_ANY_TAG = re.compile(rf"^\W{{0,4}}(?:[\w'’]+\s+){{0,3}}?(?:{_SPEECH_VERBS})\b", re.IGNORECASE)


# Narration ending on its speech verb, right before a quote: 'Then the bird said '
_LEAD_IN = re.compile(rf"\b(?:{_SPEECH_VERBS})\W{{0,3}}$", re.IGNORECASE)


def _is_tag(narration: Segment) -> bool:
    text = narration.text
    # A paragraph-ending tag ("...said she.\n\n") covers only its own paragraph.
    return "\n\n" not in text and len(text.strip()) < 90 and bool(
        _ANY_TAG.search(text))


def _tag_names(tag: str, name: str) -> bool:
    """Is *name* the subject of the speech verb in *tag*? ("added the March
    Hare", "Mr. Lucas said") — not a listener or a passing mention ("said she
    to Charlotte", "on seeing Bingley"), nor a possessive ("Scrooge's nephew")."""
    n = re.escape(name).replace(r"\ ", r"\s+")
    lead = r"(?:(?:the|a|an|old|young|little|poor|my|our)\s+)*"
    return bool(re.search(
        rf"\b(?:{_SPEECH_VERBS})\s+{lead}{n}\b(?!['’]s)|\b{n}\s+(?:{_SPEECH_VERBS})\b",
        tag, re.IGNORECASE))


def enforce_dialogue_conventions(segments: list[Segment]) -> int:
    """Typographic rules that beat any guess (LLM included); returns fixes.

    1. '"...," said my friend. "..."' — the second quote, same paragraph,
       is the tagged speaker's.
    2. A paragraph whose quote CLOSES, followed directly by an untagged
       quoted paragraph: a new speaker (someone continuing would have left
       the quote open). If both got the same name, the second goes to the
       last different speaker in the conversation.
    """
    fixed = 0
    quotes = [i for i, s in enumerate(segments) if s.kind == "quote"]
    names = sorted({s.speaker for s in segments if s.speaker and s.speaker != NARRATOR},
                   key=len, reverse=True)
    for i in quotes:
        seg, nxt = segments[i], i + 2
        if not (nxt < len(segments) and segments[i + 1].kind == "narration"
                and segments[nxt].kind == "quote" and _is_tag(segments[i + 1])):
            continue
        tag = segments[i + 1].text
        if len(re.findall(rf"\b(?:{_SPEECH_VERBS})\b", tag, re.IGNORECASE)) > 1:
            continue   # "she asked the Gryphon, and the Gryphon answered"
        if nxt + 1 < len(segments) and segments[nxt + 1].kind == "narration" \
                and _ANY_TAG.search(segments[nxt + 1].text):
            continue   # the second quote has its own tag ("said Alice ...")
        if seg.text.rstrip().endswith(".") and _LEAD_IN.search(tag):
            continue   # '"...before." Then the bird said "..."': a new sentence
                       # whose tag leads INTO the second quote

        # The tag's own name, if it gives one ("added the March Hare"), beats
        # any guess for both quotes; "said she to Charlotte" names a listener.
        named = next((n for n in names if _tag_names(tag, n)), None)
        who = named or seg.speaker
        for k in (i, nxt):
            if who and segments[k].speaker != who:
                segments[k].speaker = who
                fixed += 1
    for k, i in enumerate(quotes[1:], 1):
        prev_i = quotes[k - 1]
        seg, prev = segments[i], segments[prev_i]
        between = "".join(s.text for s in segments[prev_i + 1:i])
        if (not seg.speaker or seg.speaker != prev.speaker or not prev.closed
                or between.strip() or "\n\n" not in between):
            continue   # same paragraph, narration between, or an open quote
        if i + 1 < len(segments) and segments[i + 1].kind == "narration" \
                and _is_tag(segments[i + 1]):
            continue   # it carries its own tag; trust it
        other = next((segments[q].speaker for q in reversed(quotes[max(0, k - 12):k - 1])
                      if segments[q].speaker and segments[q].speaker != seg.speaker), None)
        if other:
            seg.speaker = other
            fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# Optional LLM attribution for quotes the heuristics could not tag.
# ---------------------------------------------------------------------------

def post_chat(url: str, body: dict, timeout: float) -> dict:
    """POST an OpenAI-style chat request and return the parsed response.

    Reasoning models (e.g. Qwen) burn the max_tokens budget "thinking"
    before any visible answer, so requests ask Qwen-style chat templates to
    skip thinking via chat_template_kwargs; servers that reject the unknown
    field get one retry without it.
    """
    for attempt in (0, 1):
        req = Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except (ConnectionError, http.client.HTTPException) as exc:
            # A server that dies mid-request (e.g. OOM-killed) drops the
            # connection; report it like any other unreachable server.
            raise URLError(f"connection lost: {exc}") from exc
        except HTTPError as exc:
            if (
                attempt == 0
                and "chat_template_kwargs" in body
                and exc.code in (400, 404, 422)
            ):
                body = {k: v for k, v in body.items()
                        if k != "chat_template_kwargs"}
                continue
            raise


NO_THINKING = {"enable_thinking": False}

# The delivery cues Orpheus understands as inline tags.
ORPHEUS_EMOTION_TAGS = (
    "laugh", "chuckle", "sigh", "gasp", "groan", "yawn", "cough",
)

# Delivery tags Fish S2 acts on inline (from its README; it also takes short
# free-form descriptions). Placed before the sentence where delivery changes.
FISH_TAGS = (
    # emotions, with Fish's documented intensity scale (mild / moderate / intense)
    "irritated", "frustrated", "angry", "furious", "worried", "afraid",
    "terrified", "sad", "grief-stricken", "excited", "surprised", "shocked",
    "amused", "tender", "cold", "sarcastic", "pleading", "menacing", "nervous",
    # tone / volume markers, stackable with an emotion ([angry][shouting])
    "shouting", "screaming", "loud", "whisper", "low voice", "trembling",
    "sobbing", "panting",
    # sounds (need following words)
    "sigh", "gasp", "laughing", "chuckle",
)
# Most cues stacked before one sentence (Fish's guide: up to three).
FISH_MAX_STACK = 2


def fish_sentences(text: str) -> list[str]:
    """A quote's sentences as Fish tagging numbers them (leading speech-tag
    punctuation dropped, split after . ! ? …)."""
    text = re.sub(r"^[\s,;:—\-]+", "", text).strip()
    return [p for p in re.split(r"(?<=[.!?…])[”’\"']?\s+", text) if p.strip()]


def _cue(value) -> tuple[list[str], str]:
    """A sentence's direction -> (stacked tags, stressed word). Accepts the
    old plain-string form ("angry") and {"tags": [...], "stress": "word"}."""
    if isinstance(value, str):
        return ([value] if value else []), ""
    if isinstance(value, dict):
        return list(value.get("tags") or []), str(value.get("stress") or "")
    return [], ""


# Loudness (dB relative to the narration level) for Fish cues: the model's
# tags change HOW a line sounds but not reliably how LOUD (measured: shouted
# Roylott lines landed anywhere from -20 to -15 dB, one calm Holmes line was
# louder), so the tool sets each cued take's level from its intensity.
# Kept modest: shouts at +8 dB (-13 dBFS speech) sounded strained on
# speakers; the top of the scale now lands around -15 dBFS.
FISH_CUE_DB = {
    "whisper": -9.0, "low voice": -4.0, "sobbing": -2.5, "tender": -2.5,
    "sad": -2.5, "grief-stricken": -2.5, "calm": -1.5, "cold": -1.0,
    "worried": -1.0, "nervous": -1.0, "menacing": -1.0, "trembling": -1.0,
    "amused": 0.0, "sarcastic": 0.0, "pleading": 0.0, "afraid": 0.5,
    "irritated": 0.5, "surprised": 1.5, "frustrated": 1.5, "excited": 2.0,
    "angry": 2.0, "shocked": 2.0, "loud": 3.0, "terrified": 3.0,
    "furious": 3.5, "shouting": 4.0, "screaming": 5.5,
}


def cue_db(cue: list) -> float:
    """Loudness offset for a stacked cue: the strongest tag sets it (a
    volume marker like 'low voice' wins over the emotion's own level), and
    a second raising tag adds 40% of its own ([furious][shouting] >
    [angry][shouting] > [shouting])."""
    vals = [FISH_CUE_DB[t] for t in cue if t in FISH_CUE_DB]
    if not vals:
        return 0.0
    quiet = [v for v in vals if v < 0 and t_is_volume(cue)]
    if quiet:
        return min(quiet)
    raising = sorted((v for v in vals if v > 0), reverse=True)
    if len(raising) >= 2:                  # [furious][shouting] > [angry][shouting]
        return min(6.0, raising[0] + 0.4 * raising[1])
    return max(vals, key=abs)


# Register (semitones) and intonation range (x) per cue: anger drifts the
# voice up and widens it, menace/low voice drifts it down and flattens it.
# Applied after rendering with formants kept, so it stays the same person.
FISH_CUE_PITCH = {
    "low voice": (-1.5, 0.85), "menacing": (-1.5, 0.85), "cold": (-0.5, 0.85),
    "calm": (-0.5, 0.95), "sad": (-1.0, 0.85), "grief-stricken": (-1.0, 0.85),
    "tender": (-0.5, 1.0), "whisper": (0.0, 1.0), "irritated": (0.5, 1.1),
    "frustrated": (1.0, 1.15), "angry": (1.5, 1.2), "afraid": (1.0, 1.15),
    "excited": (2.0, 1.25), "surprised": (2.0, 1.25), "shocked": (2.0, 1.25),
    "loud": (1.0, 1.1), "terrified": (3.0, 1.3), "furious": (2.5, 1.25),
    "shouting": (2.5, 1.3), "screaming": (4.0, 1.4),
}


def cue_pitch(cue: list) -> tuple[float, float]:
    """(semitones, range multiplier) for a stacked cue: the strongest move
    sets the direction, a second move the same way adds 40% of its own;
    capped at +4 / -3 semitones so it doesn't sound processed."""
    moves = [FISH_CUE_PITCH[t] for t in cue if t in FISH_CUE_PITCH]
    if not moves:
        return 0.0, 1.0
    up = sorted((m for m in moves if m[0] > 0), key=lambda m: -m[0])
    down = sorted((m for m in moves if m[0] < 0), key=lambda m: m[0])
    side = down if t_is_volume(cue) and down else (up if up else down)
    if not side:
        return 0.0, 1.0
    st = side[0][0] + (0.4 * side[1][0] if len(side) > 1 else 0.0)
    rng = side[0][1] * ((1 + 0.4 * (side[1][1] - 1)) if len(side) > 1 else 1.0)
    return max(-3.0, min(4.0, st)), max(0.75, min(1.5, rng))


def t_is_volume(cue: list) -> bool:
    return any(t in ("whisper", "low voice") for t in cue)


def cue_runs(text: str, tags: dict) -> list[tuple[list, str]]:
    """Split a tagged quote where its cue changes: [(cue, text with Fish
    tags)] — each run becomes its own take, so its loudness can follow the
    intensity (an angry speech that turns to shouting gets louder)."""
    sents, runs, active = fish_sentences(text), [], []
    for i, sent in enumerate(sents):
        cue, stress = _cue(tags.get(i))
        if cue and t_is_volume(cue) and active and cue_db(active) >= 5:
            # No drop from a shout to a whisper/low voice inside one speech
            # (Roylott's "See here." went quiet mid-rant and sounded like the
            # narrator); a new quote can still start quiet.
            cue = active
        if cue and re.search(r"!\W*$", sent) and t_is_volume(cue):
            # An exclamation isn't delivered in a whisper/low voice (the LLM
            # once dropped Roylott's "I am a dangerous man to fall foul of!"
            # to [low voice] right after a shout): keep the cue in force.
            cue = [t for t in cue if t not in ("whisper", "low voice")]
            if not cue or FISH_CUE_DB.get(cue[0], 0) < 0:
                cue = active
        if stress:
            sent = re.sub(rf"\b({re.escape(stress)})\b", r"[emphasis] \1", sent,
                          count=1, flags=re.IGNORECASE)
        if cue and cue != active:
            active = cue
            runs.append((cue, "".join(f"[{t}]" for t in cue) + f" {sent}"))
        elif runs:
            runs[-1] = (runs[-1][0], f"{runs[-1][1]} {sent}")
        else:
            runs.append(([], sent))
    return runs


def apply_sentence_tags(text: str, tags: dict) -> str:
    """Insert Fish cues into *text*: stacked [tag][tag] before a sentence
    where the delivery changes, and [emphasis] before its stressed word."""
    sents, out, active = fish_sentences(text), [], None
    for i, sent in enumerate(sents):
        cue, stress = _cue(tags.get(i))
        if stress:
            sent = re.sub(rf"\b({re.escape(stress)})\b", r"[emphasis] \1", sent,
                          count=1, flags=re.IGNORECASE)
        if cue and cue != active:          # a repeat of the cue in force adds nothing
            out.append("".join(f"[{t}]" for t in cue) + f" {sent}")
            active = cue
        else:
            out.append(sent)
    return " ".join(out)


# Emotional registers for palette-based engines (F5): each maps to a
# reference clip variant of the same narrator. "none" -> neutral base clip.
F5_EMOTION_REGISTERS = (
    "warm", "tense", "angry", "sad", "excited", "cold", "surprised",
)


@dataclass
class EmotionTagger:
    """Decides an (optional) Orpheus emotion tag per quote, LLM-batched.

    Deliberately conservative: most lines get no tag — only clear textual
    cues ("he laughed", "she gasped", a heavy sigh in the narration) earn
    one. Decisions, including "none", persist in a sidecar cache so a book
    is judged once.
    """

    url: str
    model: str = "default"
    timeout: float = 300.0
    batch_size: int = 16
    context_chars: int = 240
    # What the LLM may assign. Orpheus tags are vocal gestures (be sparing);
    # F5 registers are whole-line moods (assign whenever the mood is clear).
    vocabulary: tuple = ORPHEUS_EMOTION_TAGS
    style: str = "gesture"  # "gesture" | "register" | "fish"

    def tag(self, segments: list[Segment], cache_path=None, log=print) -> None:
        cache: dict[str, str] = {}
        if cache_path is not None:
            try:
                data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == 1:
                    cache = {str(k): v if isinstance(v, dict) else str(v)
                             for k, v in data.get("tags", {}).items()}
            except (OSError, ValueError):
                pass

        pending: list[int] = []
        hits = 0
        for i, seg in enumerate(segments):
            if seg.kind != "quote" or len(seg.text.strip()) < 2:
                continue
            key = _segment_key(segments, i)
            if key in cache:
                self._assign(seg, cache[key])
                hits += 1
            else:
                pending.append(i)
        if hits and log:
            log(f"  {hits} emotion decision(s) restored from cache")
        if pending and log:
            log(f"  emotion tagging: {len(pending)} quote(s) -> {self.url}")

        total = (len(pending) + self.batch_size - 1) // self.batch_size
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            if log and total > 1:
                log(f"    emotion batch {start // self.batch_size + 1}/{total}")
            answers = self._ask(segments, batch)
            for idx in batch:
                seg = segments[idx]
                if self.style == "fish":
                    value = self._clean_fish(answers.get(idx), len(fish_sentences(seg.text)))
                else:
                    tag = str(answers.get(idx, "none")).strip().lower()
                    value = tag if tag in self.vocabulary else ""
                self._assign(seg, value)
                cache[_segment_key(segments, idx)] = value

        if cache_path is not None:
            try:
                p = Path(cache_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    json.dumps({"version": 1, "tags": cache}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass

    def _assign(self, seg: Segment, value) -> None:
        if isinstance(value, dict):        # fish: {sentence index: cue}
            tags = {int(k): v for k, v in value.items()}
            seg.sentence_tags = tags or None
            first = _cue(tags[min(tags)])[0] if tags else []
            seg.emotion = " ".join(first) or None
        else:
            seg.emotion = value or None

    @staticmethod
    def _clean_fish(raw, n: int) -> dict:
        """{"1": {"tags": ["furious", "shouting"], "stress": "traced"}} (1-based
        sentences) -> {"0": {...}}; keeps short plain-word cues, at most
        FISH_MAX_STACK stacked, and a stressed word only if it's one word."""
        out = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            try:
                i = int(k) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= i < n:
                continue
            if isinstance(v, str):
                v = {"tags": [t for t in re.split(r"[,\]\[]+", v) if t.strip()]}
            if not isinstance(v, dict):
                continue
            raw_tags = v.get("tags") or []
            if isinstance(raw_tags, str):
                raw_tags = re.split(r"[,\]\[]+", raw_tags)
            tags = []
            for t in raw_tags:
                t = str(t).strip().strip("[]").lower()
                if t and t != "none" and re.fullmatch(r"[a-z][a-z -]{1,30}", t) \
                        and len(t.split()) <= 4 and t not in tags:
                    tags.append(t)
            stress = str(v.get("stress") or "").strip(" .,!?;:'\"").lower()
            cue = {}
            if tags:
                cue["tags"] = tags[:FISH_MAX_STACK]
            if re.fullmatch(r"[a-z']{2,20}", stress):
                cue["stress"] = stress
            if cue:
                out[str(i)] = cue
        return out

    def _context(self, segments: list[Segment], idx: int) -> str:
        before = "".join(s.text for s in segments[max(0, idx - 3):idx])[-self.context_chars:]
        quote = segments[idx].text[:200]
        after = "".join(s.text for s in segments[idx + 1:idx + 3])[: self.context_chars // 2]
        return f"{before}<<QUOTE>>{quote}<</QUOTE>>{after}".replace("\n", " ")

    def _ask(self, segments: list[Segment], batch: list[int]) -> dict[int, str]:
        if self.style == "fish":
            return self._ask_fish(segments, batch)
        numbered = "\n\n".join(
            f"[{i}] ...{self._context(segments, i)}..." for i in batch
        )
        if self.style == "register":
            rules = (
                "decide the emotional register the line between <<QUOTE>> and "
                f"<</QUOTE>> is delivered in: {', '.join(self.vocabulary)}. "
                "Judge from the line itself and its surrounding narration. "
                'Use "none" for ordinary neutral delivery — but do assign a '
                "register whenever the mood is clear (fear, anger, joy, "
                "menace); that is what makes the reading dynamic."
            )
        else:
            rules = (
                "decide whether the spoken line between <<QUOTE>> and "
                "<</QUOTE>> should carry ONE delivery tag from this list: "
                f"{', '.join(self.vocabulary)}. Assign a tag ONLY when the "
                "surrounding text gives a clear cue (e.g. 'he laughed', 'she "
                'gasped\', \'with a heavy sigh\'). Most lines should get "none" '
                "— be sparing; an over-tagged audiobook sounds absurd."
            )
        prompt = (
            f"For each numbered passage from a novel, {rules} "
            "Respond with ONLY a JSON object mapping each number to a value "
            'or "none", e.g. {"3": "tense", "7": "none"}.'
            f"\n\n{numbered}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 20 * self.batch_size,
            "chat_template_kwargs": NO_THINKING,
        }
        try:
            body = post_chat(self.url, payload, self.timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"emotion tagging request to {self.url} failed: {exc}"
            ) from exc
        content = body["choices"][0]["message"]["content"]
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if not match:
            return {}
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        out: dict[int, str] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return out


def _ask_fish_impl(self, segments: list[Segment], batch: list[int]) -> dict:
    """Per-sentence Fish delivery tags for each quote in *batch*."""
    blocks = []
    for i in batch:
        before = "".join(s.text for s in segments[max(0, i - 3):i])[-self.context_chars:]
        after = "".join(s.text for s in segments[i + 1:i + 3])[: self.context_chars // 2]
        sents = fish_sentences(segments[i].text)
        who = segments[i].speaker or "unknown speaker"
        lines = "\n".join(f"  ({n}) {t}" for n, t in enumerate(sents, 1))
        blocks.append(f"[{i}] context: ...{before}".replace("\n", " ")
                      + f"\n  {who} says:\n{lines}\n  then: {after}...".replace("\n\n", " "))
    prompt = (
        "You are directing an audiobook voice actor. For each numbered quote "
        "below (split into numbered sentences), mark the sentences where the "
        "character's delivery STARTS or CHANGES. Give each such sentence:\n"
        f"- \"tags\": 1-{FISH_MAX_STACK} cues from: {', '.join(self.vocabulary)} "
        "(or a 1-3 word description if none fits, e.g. 'through clenched "
        "teeth'). Combine an emotion with a volume/tone cue when the line is "
        "loud or quiet: [\"furious\", \"shouting\"], [\"afraid\", \"whisper\"]. "
        "Use the intensity that fits: irritated < frustrated < angry < "
        "furious; worried < afraid < terrified.\n"
        "- optionally \"stress\": the ONE word an actor would hit hardest in "
        "that sentence (only for genuinely emphatic sentences).\n"
        "A cue holds until the next one. Follow the story: a threat that "
        "builds goes frustrated -> angry -> furious+shouting; a calm or "
        "neutral quote gets no cues at all. Don't cue every sentence. "
        "Respond with ONLY JSON mapping quote number to {sentence number: "
        'cue}, e.g. {"12": {"1": {"tags": ["angry"]}, "4": {"tags": '
        '["furious", "shouting"], "stress": "traced"}}, "13": {}}.\n\n'
        + "\n\n".join(blocks))
    payload = {
        "model": self.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 80 * len(batch) + 300,
        "chat_template_kwargs": NO_THINKING,
    }
    try:
        body = post_chat(self.url, payload, self.timeout)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"emotion tagging request to {self.url} failed: {exc}") from exc
    content = body["choices"][0]["message"]["content"]
    start, end = content.find("{"), content.rfind("}")
    try:
        raw = json.loads(content[start:end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        return {}
    out = {}
    for k, v in raw.items() if isinstance(raw, dict) else []:
        try:
            out[int(k)] = v if isinstance(v, dict) else {}
        except (TypeError, ValueError):
            continue
    return out


EmotionTagger._ask_fish = _ask_fish_impl


@dataclass
class LlmAttributor:
    """Client for any OpenAI-compatible /v1/chat/completions endpoint."""

    url: str
    model: str = "default"
    timeout: float = 300.0
    # Characters of consecutive text per attribution passage. Larger windows
    # show longer conversations (and mean fewer calls) at more tokens/call.
    context_chars: int = 6000
    calls: int = field(default=0, init=False)

    def refine(self, segments: list[Segment], known: list[str], log=print) -> None:
        """Ask the LLM to name speakers for still-unattributed quotes.

        Works passage by passage (~``context_chars`` of consecutive text with
        every quotation numbered inline), so the model follows the turn-taking
        of a conversation instead of judging each line in isolation — the only
        way to separate two unnamed speakers ("the captain" vs Black Dog).
        """
        pending = {
            i for i, s in enumerate(segments)
            if s.kind == "quote" and not s.speaker and len(s.text.strip()) > 1
        }
        if not pending:
            return
        windows, cur, size = [], [], 0
        for i, seg in enumerate(segments):
            cur.append(i)
            size += len(seg.text)
            if size >= self.context_chars and seg.kind == "narration":
                windows.append(cur)
                cur, size = [], 0
        if cur:
            windows.append(cur)
        windows = [w for w in windows if pending.intersection(w)]
        if log:
            log(f"  LLM attribution: {len(pending)} unattributed quote(s) in "
                f"{len(windows)} passage(s) -> {self.url}")
        failed = 0
        for n, window in enumerate(windows, 1):
            if log and len(windows) > 1:
                log(f"    attribution passage {n}/{len(windows)}")
            todo = [i for i in window if i in pending]
            try:
                answers = self._ask_passage(segments, window, todo, known)
            except RuntimeError as exc:
                # One slow/failed passage must not discard the whole book;
                # its quotes stay unattributed (and uncached) for a re-run.
                if n == 1 and isinstance(exc.__cause__, URLError) and not isinstance(
                        getattr(exc.__cause__, "reason", None), TimeoutError):
                    raise
                failed += 1
                if log:
                    log(f"    passage {n} skipped: {exc}")
                continue
            for idx, name in answers.items():
                said = (name or "").strip().lower()
                # "I" / "Narrator": the first-person narrator speaking.
                # "none": not speech (a quoted word, sign or title) — the
                # narrator reads it, not the fallback dialogue voice.
                if said in ("i", "me", "narrator", "the narrator", "none"):
                    segments[idx].speaker = NARRATOR
                elif said and said != "unknown":
                    segments[idx].speaker = normalize_name(name)
        if failed and log:
            log(f"  warning: {failed} passage(s) failed; re-run to fill them in "
                "(finished passages are cached)")

    def _ask_passage(self, segments: list[Segment], window: list[int],
                     todo: list[int], known: list[str]) -> dict[int, str]:
        parts = []
        for i in window:
            seg = segments[i]
            if seg.kind != "quote":
                parts.append(seg.text)
            elif i in todo:
                parts.append(f"[Q{i}]“{seg.text}”")
            else:
                label = f"[{seg.speaker}]" if seg.speaker else ""
                parts.append(f"{label}“{seg.text}”")
        hint = f" Characters in this book: {', '.join(known[:25])}." if known else ""
        prompt = (
            "Below is a passage from a novel. Quotations marked [Q<n>] need a "
            "speaker; quotations marked [Name] are already attributed. Follow "
            f"the conversation's turn-taking.{hint} Use the character's short "
            "name. If a speaker is never named, use a short description "
            '(e.g. "Captain", "Mother", "Portly Gentleman") — never give a line '
            "to a character just because they are mentioned or addressed. If "
            'the book\'s first-person narrator speaks ("said I"), answer "I". '
            "If a quotation is not speech (a quoted phrase, sign, song or "
            'title), answer "none". First list every distinct character who '
            "speaks in this passage with a few words on who they are (two "
            "different people get two different names, even if both are "
            "sailors, both are angry, etc.), then attribute each quotation. "
            'Respond with ONLY JSON shaped like {"speakers": {"Ahab": '
            '"one-legged captain", "Starbuck": "first mate"}, "lines": '
            '{"Q12": "Ahab", "Q13": "Starbuck"}}.\n\n' + "".join(parts)
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            # Short JSON answer; a cap also keeps servers that default
            # max_tokens to "rest of context" from rejecting requests.
            "max_tokens": 30 * len(todo) + 400,
            "chat_template_kwargs": NO_THINKING,
        }
        last = None
        for _attempt in range(2):
            try:
                body = post_chat(self.url, payload, self.timeout)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                last = exc
        else:
            cause = last if isinstance(last, URLError) else URLError(last)
            raise RuntimeError(
                f"LLM attribution request to {self.url} failed: {last}. "
                "Is the server up? Pass --llm-url to point elsewhere."
            ) from cause
        self.calls += 1
        content = body["choices"][0]["message"]["content"] or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        try:
            raw = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            return {}
        lines = raw.get("lines", raw) if isinstance(raw, dict) else {}
        out: dict[int, str] = {}
        for k, v in (lines.items() if isinstance(lines, dict) else []):
            digits = re.sub(r"\D", "", str(k))
            if digits and int(digits) in todo and v:
                out[int(digits)] = str(v).strip()
        return out
