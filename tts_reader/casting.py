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
from .speakers import LlmAttributor, character_counts, segment_dialogue

DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"


@dataclass
class CastMember:
    name: str
    quotes: int
    samples: list[str]
    role: str = ""
    voice: str = ""
    why: str = ""


@dataclass
class CastSession:
    engine: str  # "orpheus" | "piper"
    llm_url: str | None
    llm_model: str = "default"
    members: list[CastMember] = field(default_factory=list)
    narrator_voice: str = ""
    dialogue_voice: str = ""
    history: list[dict] = field(default_factory=list)

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

    def valid_voice(self, voice: str) -> bool:
        return voice in {n for n, _, _ in self.roster()}

    # -- LLM plumbing ------------------------------------------------------
    def _chat(self, prompt: str, max_tokens: int = 900) -> str:
        payload = json.dumps({
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = Request(
            self.llm_url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
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
    def describe_and_suggest(self, excerpt: str, log=print) -> None:
        """One LLM call: summarize each character's role, propose a cast."""
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
            f"Opening excerpt:\n{excerpt}\n\nCharacters and sample lines:\n"
            f"{samples}\n\n"
            "Give the biggest characters distinct voices — never reuse the "
            "narrator's voice for a major character, and only double up "
            "voices once the roster runs out. Reply with ONLY JSON shaped "
            'like: {"narrator": "<voice>", "dialogue": "<voice>", '
            '"characters": {"<Name>": {"role": "...", "voice": "<voice>", '
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
            voice = str(info.get("voice", "")).strip().lower()
            if self.valid_voice(voice):
                member.voice = voice
        if self.valid_voice(str(data.get("narrator", "")).strip().lower()):
            self.narrator_voice = str(data["narrator"]).strip().lower()
        if self.valid_voice(str(data.get("dialogue", "")).strip().lower()):
            self.dialogue_voice = str(data["dialogue"]).strip().lower()
        self._dedupe_voices()

    def _dedupe_voices(self) -> None:
        """Keep major characters distinct: no narrator clones or duplicates
        while unused roster voices remain (minor characters may share)."""
        unused = [
            n for n, _, _ in self.roster()
            if n != self.narrator_voice and n != self.dialogue_voice
        ]
        taken: set[str] = set()
        for m in self.members:
            clash = (
                m.voice == self.narrator_voice
                or m.voice == self.dialogue_voice
                or m.voice in taken
            )
            if (not m.voice or clash) and unused:
                m.voice = unused[0]
            if m.voice in unused:
                unused.remove(m.voice)
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
            "about. If it is a question or request for information, answer it "
            "fully and make NO updates. Reply with ONLY JSON: "
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


def build_session(text: str, engine: str, llm_url: str | None,
                  llm_model: str, top: int, llm_context: int = 350,
                  log=print) -> CastSession:
    """Parse *text*, attribute speakers, and assemble the initial cast."""
    log("Parsing dialogue...")
    segments = segment_dialogue(text)
    session = CastSession(engine=engine, llm_url=llm_url, llm_model=llm_model)

    if llm_url:
        try:
            LlmAttributor(llm_url, model=llm_model, context_chars=llm_context).refine(
                segments, known=list(character_counts(segments)), log=log
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

    if session.llm_url:
        log("Asking the LLM to describe each character and suggest voices...")
        try:
            session.describe_and_suggest(text[:3000], log=log)
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
  voices                  show the voice roster for this engine
  convert [output]        finish casting and synthesize the audiobook
  map                     print the --voice-map string and exit
  help                    this message
  quit                    leave without converting
"""


def run_repl(session: CastSession, on_convert, log=print) -> int:
    """The interactive loop. ``on_convert(voice_map, output)`` runs the job."""
    log("\nProposed cast:\n" + session.table())
    log("\nDirect me — plain English or 'help' for commands.\n")
    while True:
        try:
            line = input("cast> ").strip()
        except (EOFError, KeyboardInterrupt):
            log("")
            return 0
        if not line:
            continue
        cmd = line.lower()
        if cmd in ("quit", "exit", "q"):
            return 0
        if cmd == "help":
            log(HELP)
        elif cmd in ("cast", "list"):
            log(session.table())
        elif cmd == "voices":
            log(session.roster_text())
        elif cmd == "map":
            log(f'--voice-map "{session.voice_map()}"')
            return 0
        elif cmd.startswith("convert"):
            output = line.split(None, 1)[1].strip() if " " in line else None
            log(f'\nCast locked in: --voice-map "{session.voice_map()}"')
            return on_convert(session.voice_map(), output)
        else:
            direct = re.match(r"^([\w'’ .-]+?)\s*=\s*([\w-]+)$", line)
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
                log("done — 'cast' to review")
            elif session.llm_url:
                try:
                    log(session.refine(line))
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    log(f"LLM unavailable ({exc}); use 'Name = voice' instead")
            else:
                log("No LLM connected — use 'Name = voice', or pass --llm-url "
                    "to direct the cast in plain language.")
