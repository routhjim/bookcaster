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
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class Segment:
    """A run of text spoken by one voice, in reading order."""

    kind: str  # "narration" | "quote"
    text: str
    speaker: Optional[str] = None  # normalized speaker name, quotes only
    emotion: Optional[str] = None  # Orpheus emotion tag for this quote, if any


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
_NAME = rf"((?:{_HONORIFIC})?[A-Z][a-zA-Z'’]+(?:\s+[A-Z][a-zA-Z'’]+)?)"

# "... ," said Ahab   /  "... !" cried old Starbuck
_TAG_VERB_NAME = re.compile(rf"^\W{{0,4}}(?:{_SPEECH_VERBS})\s+{_NAME}")
# "... ," Ahab said   /  "... " Stubb muttered
_TAG_NAME_VERB = re.compile(rf"^\W{{0,4}}{_NAME}\s+(?:{_SPEECH_VERBS})\b")
# Ahab said(,|:)? "..."   — searched at the END of the narration before a quote.
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


def normalize_name(raw: str) -> str:
    """Canonical speaker key: honorific stripped, title-cased last resort."""
    name = _STRIP_HONORIFIC.sub("", raw.strip()).strip()
    name = re.sub(r"[’']s$", "", name)  # possessive slips
    return name.title() if name else raw.strip().title()


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
        out.append(Segment("quote" if in_quote else "narration", "".join(buf)))
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
            if m and _valid_name(m.group(1)):
                seg.speaker = normalize_name(m.group(1))
                # '"...," said Ahab, "..."' — the tag also covers the next quote.
                if (
                    len(after.strip()) < 60
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
                seg.speaker = normalize_name(m.group(1))


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


def attribute_segments(
    segments: list[Segment], attributor: "LlmAttributor | None" = None,
    cache_path: str | Path | None = None, log=print,
) -> None:
    """Cache -> LLM -> cache pipeline on top of the heuristic attributions."""
    cache = AttributionCache(cache_path) if cache_path else None
    if cache is not None:
        hits = cache.apply(segments)
        if hits and log:
            log(f"  {hits} speaker attribution(s) restored from cache")
    if attributor is not None:
        attributor.refine(segments, known=list(character_counts(segments)), log=log)
    if cache is not None:
        cache.update(segments)
        cache.save()


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
    style: str = "gesture"  # "gesture" | "register"

    def tag(self, segments: list[Segment], cache_path=None, log=print) -> None:
        cache: dict[str, str] = {}
        if cache_path is not None:
            try:
                data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == 1:
                    cache = {str(k): str(v) for k, v in data.get("tags", {}).items()}
            except (OSError, ValueError):
                pass

        pending: list[int] = []
        hits = 0
        for i, seg in enumerate(segments):
            if seg.kind != "quote" or len(seg.text.strip()) < 2:
                continue
            key = _segment_key(segments, i)
            if key in cache:
                seg.emotion = cache[key] or None
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
                tag = answers.get(idx, "none").strip().lower()
                seg = segments[idx]
                seg.emotion = tag if tag in self.vocabulary else None
                cache[_segment_key(segments, idx)] = seg.emotion or ""

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

    def _context(self, segments: list[Segment], idx: int) -> str:
        before = "".join(s.text for s in segments[max(0, idx - 3):idx])[-self.context_chars:]
        quote = segments[idx].text[:200]
        after = "".join(s.text for s in segments[idx + 1:idx + 3])[: self.context_chars // 2]
        return f"{before}<<QUOTE>>{quote}<</QUOTE>>{after}".replace("\n", " ")

    def _ask(self, segments: list[Segment], batch: list[int]) -> dict[int, str]:
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


@dataclass
class LlmAttributor:
    """Client for any OpenAI-compatible /v1/chat/completions endpoint."""

    url: str
    model: str = "default"
    timeout: float = 300.0
    batch_size: int = 12
    # How much surrounding text each quote carries. Wider context resolves
    # long unattributed back-and-forth exchanges (the anchor "said Ahab" can
    # be many quotes back) at the cost of larger, slower LLM calls.
    context_chars: int = 350
    calls: int = field(default=0, init=False)

    def refine(self, segments: list[Segment], known: list[str], log=print) -> None:
        """Ask the LLM to name speakers for still-unattributed quotes."""
        pending = [
            i for i, s in enumerate(segments)
            if s.kind == "quote" and not s.speaker and len(s.text.strip()) > 1
        ]
        if not pending:
            return
        if log:
            log(f"  LLM attribution: {len(pending)} unattributed quote(s) -> {self.url}")
        total = (len(pending) + self.batch_size - 1) // self.batch_size
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            if log and total > 1:
                log(f"    attribution batch {start // self.batch_size + 1}/{total}")
            answers = self._ask(segments, batch, known)
            for idx, name in answers.items():
                if name and name.lower() not in ("unknown", "narrator", "none"):
                    segments[idx].speaker = normalize_name(name)

    def _context(self, segments: list[Segment], idx: int) -> str:
        before = "".join(s.text for s in segments[max(0, idx - 12):idx])[-self.context_chars:]
        quote = segments[idx].text[:300]
        after = "".join(s.text for s in segments[idx + 1:idx + 9])[: self.context_chars // 2]
        return f"{before}<<QUOTE>>{quote}<</QUOTE>>{after}".replace("\n", " ")

    def _ask(self, segments: list[Segment], batch: list[int], known: list[str]) -> dict[int, str]:
        numbered = "\n\n".join(
            f"[{i}] ...{self._context(segments, i)}..." for i in batch
        )
        hint = f" Known characters so far: {', '.join(known[:20])}." if known else ""
        prompt = (
            "For each numbered passage from a novel, identify who speaks the text "
            f"between <<QUOTE>> and <</QUOTE>>.{hint} Use the character's short name. "
            'If genuinely unclear, use "unknown". Respond with ONLY a JSON object '
            'mapping each number to a name, e.g. {"3": "Ahab", "7": "unknown"}.'
            f"\n\n{numbered}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            # Answers are a short JSON object; a cap also keeps servers that
            # default max_tokens to "rest of context" from rejecting requests.
            "max_tokens": 40 * self.batch_size,
            "chat_template_kwargs": NO_THINKING,
        }
        try:
            body = post_chat(self.url, payload, self.timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"LLM attribution request to {self.url} failed: {exc}. "
                "Is the server up? Pass --llm-url to point elsewhere."
            ) from exc
        self.calls += 1
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
                out[int(k)] = str(v).strip()
            except (TypeError, ValueError):
                continue
        return out
