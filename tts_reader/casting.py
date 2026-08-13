"""Interactive casting sessions: direct your audiobook like a conversation.

``tts-reader cast book.txt`` parses the book, has a local LLM describe each
speaking character's role, and suggests a voice for each. You then refine the
cast in plain language::

    cast> Starbuck is an old, grizzled, crusty pirate
    cast> Ahab is refined and commanding, almost genteel
    cast> use dan for Stubb
    cast> convert out/moby.mp3

Everything runs against an OpenAI-compatible chat endpoint (any local
llama.cpp server works). Without one, the session still runs: quote counts
replace role summaries and you assign voices directly with ``Name = voice``.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import voices as voice_catalog
from .speakers import (
    LlmAttributor, attribute_segments, character_counts, segment_dialogue,
)
from .textprep import prepare_for_speech

DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"


@dataclass
class CastMember:
    name: str
    quotes: int
    samples: list[str]
    role: str = ""
    voice: str = ""  # may carry a rate variant, e.g. "zac@0.92"
    why: str = ""
    gender: str = ""  # "M", "F", or "" when unknown


@dataclass
class CastSession:
    engine: str  # "orpheus" | "piper"
    llm_url: str | None
    llm_model: str = "default"
    members: list[CastMember] = field(default_factory=list)
    narrator_voice: str = ""
    dialogue_voice: str = ""
    history: list[dict] = field(default_factory=list)
    excerpt: str = ""  # opening of the book, kept for on-demand describes

    # -- voice roster ------------------------------------------------------
    def roster(self) -> list[tuple[str, str, str]]:
        """(voice, gender, traits) options for the active engine."""
        if self.engine == "orpheus":
            return list(voice_catalog._ORPHEUS_VOICES)
        return [
            (v.alias, v.gender, f"{v.accent}; {v.description}")
            for v in voice_catalog._VOICES
            if v.accent.endswith("English")
        ]

    def roster_text(self) -> str:
        return "\n".join(f"- {n} ({g}): {d}" for n, g, d in self.roster())

    def voice_genders(self) -> dict[str, str]:
        return {n: g for n, g, _ in self.roster()}

    def valid_voice(self, voice: str) -> bool:
        from .engine import parse_voice_spec

        try:
            base, _ = parse_voice_spec(voice)
        except ValueError:
            return False
        return base in {n for n, _, _ in self.roster()}

    # -- LLM plumbing ------------------------------------------------------
    def _chat(self, prompt: str, max_tokens: int = 900) -> str:
        from .speakers import NO_THINKING, post_chat

        body = post_chat(self.llm_url, {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "chat_template_kwargs": NO_THINKING,
        }, timeout=600)
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _json_block(text: str) -> dict:
        # The model may wrap JSON in prose or fences; grab the outermost {...}.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    def _cast_state(self) -> str:
        lines = [
            f"narrator: voice={self.narrator_voice or '?'}",
            f"(fallback dialogue voice: {self.dialogue_voice or '?'})",
        ]
        for m in self.members:
            lines.append(
                f"{m.name}: {m.quotes} lines, voice={m.voice or '?'}"
                + (f", role: {m.role}" if m.role else "")
            )
        return "\n".join(lines)

    # -- session stages ----------------------------------------------------
    def describe_and_suggest(self, excerpt: str, roles_only: bool = False,
                             log=print) -> None:
        """One LLM call: summarize each character's role, propose a cast.

        With ``roles_only`` the voice assignments (and narrator/dialogue
        picks) are left untouched — used to refresh descriptions for a
        restored cast without disturbing the user's choices.
        """
        samples = "\n".join(
            f"{m.name} ({m.quotes} lines): " + " / ".join(
                f"“{s}”" for s in m.samples[:3]
            )
            for m in self.members
        )
        prompt = (
            "You are casting an audiobook. Based on this excerpt and each "
            "character's sample lines, write a one-sentence role/personality "
            "description per character and pick the best-fitting voice for "
            "each (plus a narrator voice and a fallback voice for minor "
            "characters) from this roster:\n"
            f"{self.roster_text()}\n\n"
            f"Opening excerpt:\n{excerpt}\n\nCharacters and sample lines "
            "(listed by importance — most lines first; give earlier "
            "characters the best-fitting voices):\n"
            f"{samples}\n\n"
            "Rules: a character's voice MUST match their gender (male "
            "characters get male voices, female get female). Give the "
            "biggest characters distinct voices — never reuse the narrator's "
            "voice for a major character. If same-gender voices run out, "
            "reuse one with a rate variant like 'zac@0.92' or 'zac@1.08' "
            "(slightly slower/faster delivery) instead of switching gender. "
            'Reply with ONLY JSON shaped like: {"narrator": "<voice>", '
            '"dialogue": "<voice>", "characters": {"<Name>": {"role": "...", '
            '"gender": "male|female|unknown", "voice": "<voice>", '
            '"why": "..."}}}'
        )
        data = self._json_block(self._chat(prompt))
        chars = data.get("characters", {})
        by_name = {m.name.lower(): m for m in self.members}
        for name, info in chars.items():
            member = by_name.get(str(name).lower())
            if not member or not isinstance(info, dict):
                continue
            member.role = str(info.get("role", "")).strip()
            member.why = str(info.get("why", "")).strip()
            gender = str(info.get("gender", "")).strip()[:1].upper()
            if gender in ("M", "F"):
                member.gender = gender
            if roles_only:
                continue
            voice = str(info.get("voice", "")).strip().lower()
            if self.valid_voice(voice):
                member.voice = voice
        if roles_only:
            return
        if self.valid_voice(str(data.get("narrator", "")).strip().lower()):
            self.narrator_voice = str(data["narrator"]).strip().lower()
        if self.valid_voice(str(data.get("dialogue", "")).strip().lower()):
            self.dialogue_voice = str(data["dialogue"]).strip().lower()
        self._dedupe_voices()

    # Rate variants handed out when a gender's roster is exhausted: each
    # reuse of a base voice gets the next slightly-different delivery speed.
    _RATE_STEPS = [1.0, 0.92, 1.08, 0.85, 1.15]

    def _dedupe_voices(self) -> None:
        """Enforce the casting rules, in priority order (most lines first):

        * a character's voice matches their gender when it is known;
        * no character reuses the narrator's or fallback voice;
        * no two characters share the exact same voice — once a gender's
          roster is exhausted, a voice is reused at a new rate variant
          (``zac@0.92``) rather than crossing gender.

        Members are iterated as sorted (by line count), so major characters
        keep their preferred voices and later clashes get reassigned.
        """
        genders = self.voice_genders()
        narrator_base = self.narrator_voice.partition("@")[0]
        if self.dialogue_voice.partition("@")[0] == narrator_base:
            others = [n for n in genders if n != narrator_base]
            if others:
                self.dialogue_voice = others[0]
        reserved = {narrator_base, self.dialogue_voice.partition("@")[0]}

        def fits(base: str, member: CastMember) -> bool:
            gender = genders.get(base, "")
            return (
                not member.gender
                or not gender
                or gender == "Neutral"
                or gender[:1].upper() == member.gender
            )

        taken: set[str] = set()          # full specs, e.g. "zac@0.92"
        used_bases: dict[str, int] = {}  # base voice -> times assigned
        for m in self.members:
            base = m.voice.partition("@")[0]
            keeps = (
                base in genders
                and base not in reserved
                and m.voice not in taken
                and fits(base, m)
            )
            if not keeps:
                fresh = [
                    n for n in genders
                    if n not in reserved and n not in used_bases and fits(n, m)
                ]
                if fresh:
                    m.voice = fresh[0]
                else:
                    # Same-gender roster exhausted: reuse the least-used
                    # fitting base at the next rate variant.
                    pool = (
                        [n for n in genders if n not in reserved and fits(n, m)]
                        or [n for n in genders if n not in reserved]
                        or list(genders)
                    )
                    base = min(pool, key=lambda n: used_bases.get(n, 0))
                    step = used_bases.get(base, 0)
                    rate = self._RATE_STEPS[min(step, len(self._RATE_STEPS) - 1)]
                    m.voice = base if rate == 1.0 else f"{base}@{rate}"
            base = m.voice.partition("@")[0]
            used_bases[base] = used_bases.get(base, 0) + 1
            taken.add(m.voice)

    def refine(self, user_message: str) -> str:
        """One chat turn: apply the user's direction to the cast."""
        recent = "\n".join(
            f"{h['who']}: {h['text']}" for h in self.history[-6:]
        )
        prompt = (
            "You are helping cast an audiobook in a chat session. "
            f"Voice roster:\n{self.roster_text()}\n\nCurrent cast:\n"
            f"{self._cast_state()}\n\n"
            + (f"Recent conversation:\n{recent}\n\n" if recent else "")
            + f"The director says: \"{user_message}\"\n\n"
            "If this is casting direction, update the cast to match (change "
            "voices, refine role descriptions), changing only what they asked "
            "about. Keep voices gender-matched to characters; a voice may be "
            "reused with a rate variant like 'zac@0.92' (slightly slower) or "
            "'zac@1.08' (slightly faster) when the roster runs short. If it "
            "is a question or request for information, answer it fully and "
            "make NO updates. Reply with ONLY JSON: "
            '{"reply": "<your complete answer/response to the director — any '
            "explanation, list, or rundown they asked for goes HERE in full; "
            "this field is the only text they will ever see, so never claim "
            'something is shown elsewhere>", "narrator": "<voice or null>", '
            '"dialogue": "<voice or null>", "updates": {"<Name>": {"voice": '
            '"<voice or null>", "role": "<updated description or null>"}}}'
        )
        data = self._json_block(self._chat(prompt, max_tokens=1600))
        if not data:
            return "(couldn't parse the model's reply — try rephrasing)"
        by_name = {m.name.lower(): m for m in self.members}
        for name, info in (data.get("updates") or {}).items():
            member = by_name.get(str(name).lower())
            if not member or not isinstance(info, dict):
                continue
            voice = str(info.get("voice") or "").strip().lower()
            if voice and self.valid_voice(voice):
                member.voice = voice
            role = info.get("role")
            if role:
                member.role = str(role).strip()
        for key in ("narrator", "dialogue"):
            voice = str(data.get(key) or "").strip().lower()
            if voice and self.valid_voice(voice):
                setattr(self, f"{key}_voice", voice)
        reply = str(data.get("reply", "")).strip() or "Updated."
        self.history.append({"who": "director", "text": user_message})
        self.history.append({"who": "assistant", "text": reply})
        return reply

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "engine": self.engine,
            "narrator": self.narrator_voice,
            "dialogue": self.dialogue_voice,
            "characters": {
                m.name: {"voice": m.voice, "role": m.role, "gender": m.gender}
                for m in self.members
            },
        }

    def apply_saved(self, data: dict) -> int:
        """Restore voices/roles from a saved cast; returns members restored."""
        if not isinstance(data, dict) or data.get("version") != 1:
            return 0
        applied = 0
        for key in ("narrator", "dialogue"):
            voice = str(data.get(key) or "").strip().lower()
            if voice and self.valid_voice(voice):
                setattr(self, f"{key}_voice", voice)
        saved = {str(k).lower(): v for k, v in (data.get("characters") or {}).items()}
        for m in self.members:
            info = saved.get(m.name.lower())
            if not isinstance(info, dict):
                continue
            voice = str(info.get("voice") or "").strip().lower()
            if voice and self.valid_voice(voice):
                m.voice = voice
                applied += 1
            if info.get("role"):
                m.role = str(info["role"]).strip()
            gender = str(info.get("gender", "")).strip()[:1].upper()
            if gender in ("M", "F"):
                m.gender = gender
        return applied

    # -- output ------------------------------------------------------------
    def voice_map(self) -> str:
        parts = [f"narrator={self.narrator_voice}"]
        if self.dialogue_voice:
            parts.append(f"dialogue={self.dialogue_voice}")
        parts.extend(f"{m.name}={m.voice}" for m in self.members if m.voice)
        return ",".join(parts)

    def table(self) -> str:
        width = max([len("narrator")] + [len(m.name) for m in self.members])
        lines = [
            f"  {'narrator'.ljust(width)}  {self.narrator_voice or '?':<6}  "
            "(everything outside quotes)"
        ]
        if self.dialogue_voice:
            lines.append(
                f"  {'(other)'.ljust(width)}  {self.dialogue_voice:<6}  "
                "(unattributed/minor speakers)"
            )
        for m in self.members:
            role = m.role or f"{m.quotes} line(s)"
            lines.append(f"  {m.name.ljust(width)}  {m.voice or '?':<6}  {role}")
        return "\n".join(lines)


def load_saved_cast(path) -> dict | None:
    try:
        from pathlib import Path

        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_cast(session: CastSession, path) -> None:
    try:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=1),
                     encoding="utf-8")
    except OSError:
        pass  # never let a failed save break the session


def build_session(text: str, engine: str, llm_url: str | None,
                  llm_model: str, top: int, llm_context: int = 350,
                  cache_path=None, saved: dict | None = None,
                  log=print) -> CastSession:
    """Parse *text*, attribute speakers, and assemble the initial cast."""
    log("Parsing dialogue...")
    # Parse the same speech-prepped text the converter will synthesize, so
    # cached attributions line up between the session and the conversion.
    segments = segment_dialogue(prepare_for_speech(text))
    session = CastSession(engine=engine, llm_url=llm_url, llm_model=llm_model)

    attributor = None
    if llm_url:
        attributor = LlmAttributor(
            llm_url, model=llm_model, context_chars=llm_context
        )
    try:
        attribute_segments(
            segments, attributor=attributor, cache_path=cache_path, log=log
        )
    except RuntimeError as exc:
        log(f"warning: LLM attribution skipped: {exc}")
        session.llm_url = None

    counts = character_counts(segments)
    for name, n in list(counts.items())[:top]:
        samples = [
            s.text.strip().replace("\n", " ")[:90]
            for s in segments if s.kind == "quote" and s.speaker == name
        ][:4]
        session.members.append(CastMember(name=name, quotes=n, samples=samples))

    # Sensible defaults before (or without) the LLM's suggestions.
    roster = [n for n, _, _ in session.roster()]
    session.narrator_voice = "leo" if engine == "orpheus" else "ryan"
    session.dialogue_voice = "tara" if engine == "orpheus" else "amy"
    for member, voice in zip(session.members, roster):
        member.voice = voice if voice not in (
            session.narrator_voice, session.dialogue_voice
        ) else roster[-1]

    session.excerpt = text[:3000]

    if saved:
        restored = session.apply_saved(saved)
        if restored:
            log(f"Restored your saved cast ({restored} character(s)).")
            # Broken/older sessions may have saved empty descriptions;
            # backfill them without touching the restored voice picks.
            if session.llm_url and any(not m.role for m in session.members):
                log("Filling in missing character descriptions...")
                try:
                    session.describe_and_suggest(
                        session.excerpt, roles_only=True, log=log
                    )
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    log(f"warning: descriptions unavailable ({exc})")
            return session

    if session.llm_url:
        log("Asking the LLM to describe each character and suggest voices...")
        try:
            session.describe_and_suggest(session.excerpt, log=log)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            log(f"warning: role descriptions unavailable ({exc}); "
                "using defaults — you can still assign voices directly.")
            session.llm_url = None
    return session


HELP = """Commands:
  <plain language>        e.g. "Starbuck is an old, grizzled, crusty pirate"
                          or "swap Ahab and Starbuck's voices" — the LLM
                          updates the cast and answers you.
  Name = voice            assign a voice directly, no LLM (e.g. Stubb = dan)
  cast                    show the current cast table
  describe                regenerate the character role summaries (LLM)
  voices                  show the voice roster for this engine
  convert [output]        finish casting and synthesize the audiobook
  map                     print the --voice-map string and exit
  help                    this message
  quit                    leave without converting
"""


def run_repl(session: CastSession, on_convert, state_path=None, log=print) -> int:
    """The interactive loop. ``on_convert(voice_map, output)`` runs the job."""

    def checkpoint():
        if state_path is not None:
            save_cast(session, state_path)

    log("\nProposed cast:\n" + session.table())
    log("\nDirect me — plain English or 'help' for commands.\n")
    checkpoint()  # the initial suggestion is worth keeping too
    while True:
        try:
            line = input("cast> ").strip()
        except (EOFError, KeyboardInterrupt):
            log("")
            checkpoint()
            return 0
        if not line:
            continue
        cmd = line.lower()
        if cmd in ("quit", "exit", "q"):
            checkpoint()
            return 0
        if cmd == "help":
            log(HELP)
        elif cmd in ("cast", "list"):
            log(session.table())
        elif cmd == "describe":
            if not session.llm_url:
                log("No LLM connected — descriptions need --llm-url.")
                continue
            try:
                session.describe_and_suggest(
                    session.excerpt, roles_only=True, log=log
                )
                checkpoint()
                log(session.table())
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                log(f"LLM unavailable ({exc})")
        elif cmd == "voices":
            log(session.roster_text())
        elif cmd == "map":
            log(f'--voice-map "{session.voice_map()}"')
            checkpoint()
            return 0
        elif cmd.startswith("convert"):
            output = line.split(None, 1)[1].strip() if " " in line else None
            log(f'\nCast locked in: --voice-map "{session.voice_map()}"')
            checkpoint()
            return on_convert(session.voice_map(), output)
        else:
            direct = re.match(r"^([\w'’ .-]+?)\s*=\s*([\w.@-]+)$", line)
            if direct:
                name, voice = direct.group(1).strip(), direct.group(2).strip().lower()
                if not session.valid_voice(voice):
                    log(f"unknown voice '{voice}' — try 'voices'")
                    continue
                lowered = name.lower()
                if lowered == "narrator":
                    session.narrator_voice = voice
                elif lowered in ("dialogue", "other"):
                    session.dialogue_voice = voice
                else:
                    member = next(
                        (m for m in session.members if m.name.lower() == lowered), None
                    )
                    if member is None:
                        log(f"no character called '{name}' — try 'cast'")
                        continue
                    member.voice = voice
                checkpoint()
                log("done — 'cast' to review")
            elif session.llm_url:
                try:
                    log(session.refine(line))
                    checkpoint()
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    log(f"LLM unavailable ({exc}); use 'Name = voice' instead")
            else:
                log("No LLM connected — use 'Name = voice', or pass --llm-url "
                    "to direct the cast in plain language.")
