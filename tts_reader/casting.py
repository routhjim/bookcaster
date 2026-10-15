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

import hashlib
import json
import re
from pathlib import Path
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
    # Designed-voice mode: a description of how the character SOUNDS, the
    # audition line the designer speaks, and the take (seed) in use.
    voice_desc: str = ""
    line: str = ""
    seed: int = 0
    aliases: str = ""  # other names the book uses ("Miss Helen Stoner")
    # Same person under another speaker name ("Ghost" = "Marley"): shares
    # that member's voice instead of getting one of its own.
    same_as: str = ""
    # Sample lines WITH the narration before them ("…said the Ghost") — tells
    # the LLM who a speaker is far better than bare quotes.
    contexts: list = field(default_factory=list)
    # Designed voices only: "scottish", "irish", ... (design.available_accents)
    # — rendered by the Qwen server with a real accented speaker's delivery.
    accent: str = ""


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
    # Roster override for engines with dynamic voices (e.g. an F5 server's
    # cloned-voice registry): list of (name, gender, description).
    custom_roster: list = field(default_factory=list)
    # Designed-voice mode (``cast --design``): characters get original voices
    # built from descriptions by a voice-design server, rendered by F5.
    design_url: str | None = None
    book: str = ""        # slug used to namespace designed voice names
    f5_url: str = ""      # for previews through the real renderer
    qwen_url: str = ""    # renders accented voices (F5 drops the accent)
    # Gap filling (voice_fill): import real VCTK speakers for poor fits, or
    # design one from this server when VCTK has none (old, children).
    fill_voices: bool = False
    design_fallback_url: str = ""
    # Auditions are also saved as MP3s here (open them from a phone);
    # no_play skips speaker playback entirely (remote sessions).
    audition_dir: object = None   # pathlib.Path | None
    no_play: bool = False
    registers: bool = True        # build emotional variants of designed voices

    @property
    def designing(self) -> bool:
        return bool(self.design_url)

    def find_member(self, name) -> CastMember | None:
        lowered = str(name or "").strip().lower()
        return next((m for m in self.members if m.name.lower() == lowered), None)

    def _link(self, member: CastMember, other_name) -> None:
        """Record that *member* and *other_name* are one person; the one
        with more lines keeps its voice, the other borrows it. "narrator"
        means the book's first-person narrator: the narrator voice."""
        if str(other_name or "").strip().lower() == "narrator":
            member.same_as = "narrator"
            return
        other = self.find_member(other_name)
        if other is None or other is member:
            return
        primary, alias = (other, member) if other.quotes >= member.quotes else (member, other)
        if primary.same_as:          # never chain: point at the root
            return
        alias.same_as = primary.name
        for m in self.members:       # re-point anything that aliased the alias
            if m.same_as == alias.name:
                m.same_as = primary.name

    def render_url(self, voice: str) -> str:
        """Server that will really render *voice* in the book."""
        from . import design

        if self.qwen_url and design.renders_on_qwen(voice):
            return self.qwen_url
        return self.f5_url

    def apply_accent(self, member: CastMember, log=print) -> None:
        """Push *member*'s accent to its designed voice and audition it."""
        from . import design

        if not self.is_designed(member):
            return   # recorded; applied when the voice is built
        donor = design.set_voice_accent(member.voice, member.accent, member.gender)
        if member.accent and not donor:
            log(f"  no {member.accent} speaker on file — available: "
                f"{', '.join(design.available_accents()) or 'none'}")
            member.accent = ""
            return
        log(f"  {member.name}: " + (f"{member.accent} accent (renders on Qwen; "
            "emotion is added by pitch/pace shaping)"
            if donor else "no accent"))
        if self.audition_dir is not None and self.render_url(member.voice):
            text = max(member.samples, key=len) if member.samples else member.line
            try:
                stem = f"{design.file_stem(member.name)}_{member.accent or 'plain'}"
                path = design.save_audition(
                    design.f5_preview(self.render_url(member.voice), member.voice, text),
                    self.audition_dir, stem)
                log(f"    audition: {path}")
            except OSError as exc:
                log(f"    (audition render failed: {exc})")

    def voice_of(self, m: CastMember) -> str:
        if m.same_as == "narrator":
            return self.narrator_voice
        root = self.find_member(m.same_as) if m.same_as else None
        return root.voice if root else m.voice

    @staticmethod
    def _who(m: CastMember) -> str:
        """Gender + aliases prefix for LLM listings: "female; also Miss
        Helen Stoner; " — tells the LLM who a bare surname is."""
        g = {"M": "male", "F": "female"}.get(m.gender, "")
        bits = [b for b in (g, f"also {m.aliases}" if m.aliases else "") if b]
        return "; ".join(bits) + "; " if bits else ""

    # -- voice roster ------------------------------------------------------
    def roster(self) -> list[tuple[str, str, str]]:
        """(voice, gender, traits) options for the active engine."""
        if self.custom_roster:
            return list(self.custom_roster)
        if self.engine == "orpheus":
            return list(voice_catalog._ORPHEUS_VOICES)
        return [
            (v.alias, v.gender, f"{v.accent}; {v.description}")
            for v in voice_catalog._VOICES
            if v.accent.endswith("English")
        ]

    def offered(self) -> list[tuple[str, str, str]]:
        """The roster the LLM and the defaults pick from. Outside designed
        mode a designed voice (d_<book>_<character>) belongs to the character
        it was made for, so it is offered only once assigned."""
        if self.designing:
            return self.roster()
        assigned = {m.voice.partition("@")[0] for m in self.members}
        return [r for r in self.roster()
                if not r[0].startswith("d_") or r[0] in assigned]

    def roster_text(self) -> str:
        return "\n".join(f"- {n} ({g}): {d}" for n, g, d in self.offered())

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
                + (f", sounds like: {m.voice_desc}" if m.voice_desc else "")
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
            f"{m.name} ({self._who(m)}{m.quotes} lines):\n" + "\n".join(
                f"   {c}" for c in (m.contexts or [f"“{s}”" for s in m.samples[:3]])
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
            '"why": "...", "same_as": "<another listed Name if this is the '
            'SAME person under a different name or title (e.g. \'Ghost\' and '
            '\'Marley\' for Marley\'s ghost); \'narrator\' if this character '
            'is the book\'s first-person narrator; else null>"}}}'
        )
        data = self._json_block(self._chat(prompt))
        chars = data.get("characters", {})
        for name, info in chars.items():
            member = self._match_member(name)
            if not member or not isinstance(info, dict):
                continue
            member.role = str(info.get("role", "")).strip()
            member.why = str(info.get("why", "")).strip()
            self._link(member, info.get("same_as"))
            gender = str(info.get("gender", "")).strip()[:1].upper()
            if gender in ("M", "F") and not member.gender:
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

    def review_fit(self, accents: list[str], log=print) -> list[dict]:
        """One LLM call: which cast picks fit their role poorly? Returns
        [{"name": <member or "narrator">, "gender", "accent", "age"}]."""
        roster = {n: d for n, _, d in self.roster()}
        lines = [f"narrator: voice {self.narrator_voice} = "
                 f"{roster.get(self.narrator_voice.partition('@')[0], '?')}"]
        for m in self.members:
            if m.same_as:
                continue
            seen = " | ".join(m.contexts[:2])
            lines.append(f"{m.name} ({self._who(m)}{m.quotes} lines; "
                         f"{m.role or 'no description'}; in the text: {seen}): "
                         f"voice {m.voice} = "
                         f"{roster.get(m.voice.partition('@')[0], '?')}")
        prompt = (
            "Review the voice casting of an audiobook. For the narrator and "
            "each character, judge whether the assigned voice (its speaker's "
            "gender, age and accent are in its description) suits the role "
            "and the book's setting. A character the text calls old needs "
            "an older voice; a child needs a child's. Mark \"poor\" only "
            "when a listener would notice the mismatch.\n\n"
            f"Opening excerpt:\n{self.excerpt[:3000]}\n\nCast:\n"
            + "\n".join(lines) + "\n\n"
            "For each, give the IDEAL voice: gender M or F; accent, one of: "
            + ", ".join(accents) + "; age: child, young (16-30), middle "
            "(31-55) or old (56+). Reply with ONLY JSON: {\"narrator\": "
            '{"fit": "good|poor", "gender": "M|F", "accent": "...", '
            '"age": "..."}, "characters": {"<Name>": {same fields}}}'
        )
        data = self._json_block(self._chat(prompt))
        picks = [("narrator", data.get("narrator"))]
        picks += list((data.get("characters") or {}).items())
        wants = []
        for name, info in picks:
            if not isinstance(info, dict):
                continue
            want = {"name": str(name), "gender": str(info.get("gender", ""))[:1].upper(),
                    "accent": str(info.get("accent", "")).strip(),
                    "age": str(info.get("age", "")).strip().lower()}
            if want["gender"] not in ("M", "F") or not want["accent"]:
                continue
            member = None if name == "narrator" else self._match_member(name)
            voice = self.narrator_voice if member is None else member.voice
            # The LLM passes plainly wrong ages ("old man" on a 23-year-old),
            # so check the age band against the speaker's real age ourselves.
            if str(info.get("fit", "")).lower() == "poor" or not _age_fits(
                    roster.get(voice.partition("@")[0], ""), want["age"]):
                wants.append(want)
        return wants

    def _match_member(self, name) -> CastMember | None:
        """The member the LLM means by *name*. It often embellishes the
        attributed name ("Man" -> "The Old Man"), so after an exact match
        fall back to the one member whose name words all appear in it."""
        def words(n):
            return [w for w in re.findall(r"[a-z']+", str(n).lower()) if w != "the"]
        key = words(name)
        exact = [m for m in self.members if words(m.name) == key]
        if exact:
            return exact[0]
        loose = [m for m in self.members
                 if words(m.name) and set(words(m.name)) <= set(key)]
        return loose[0] if len(loose) == 1 else None

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
        offered = [n for n, _, _ in self.offered()]
        # Several library voices can be clones of ONE person ("bob" and
        # "neufeld" are both Bob Neufeld): compare people, not voice names.
        person = {n: _person_of(n, d) for n, _, d in self.roster()}
        who = lambda n: person.get(n, n)
        narrator_base = self.narrator_voice.partition("@")[0]
        if who(self.dialogue_voice.partition("@")[0]) == who(narrator_base):
            others = [n for n in offered if who(n) != who(narrator_base)]
            if others:
                self.dialogue_voice = others[0]
        reserved = {narrator_base, self.dialogue_voice.partition("@")[0]}
        reserved_people = {who(n) for n in reserved}

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
            used_people = {who(b) for b in used_bases if b != base}
            keeps = (
                base in genders
                and who(base) not in reserved_people
                and who(base) not in used_people
                and m.voice not in taken
                and fits(base, m)
            )
            if not keeps:
                used_people = {who(b) for b in used_bases}
                fresh = [
                    n for n in offered
                    if who(n) not in reserved_people and who(n) not in used_people
                    and fits(n, m)
                ]
                if fresh:
                    m.voice = fresh[0]
                else:
                    # Same-gender roster exhausted: reuse the least-used
                    # fitting base at the next rate variant.
                    pool = (
                        [n for n in offered if n not in reserved and fits(n, m)]
                        or [n for n in offered if n not in reserved]
                        or list(offered)
                    )
                    base = min(pool, key=lambda n: used_bases.get(n, 0))
                    step = used_bases.get(base, 0)
                    rate = self._RATE_STEPS[min(step, len(self._RATE_STEPS) - 1)]
                    m.voice = base if rate == 1.0 else f"{base}@{rate}"
            base = m.voice.partition("@")[0]
            used_bases[base] = used_bases.get(base, 0) + 1
            taken.add(m.voice)

    # -- designed voices ---------------------------------------------------
    def write_voice_descriptions(self, members=None, log=print) -> None:
        """One LLM call: describe how each character SOUNDS + an audition line.

        The description is the voice designer's prompt, so it is about
        timbre and delivery, not plot. Characters that already have one are
        skipped unless passed explicitly in *members*.
        """
        targets = members or [m for m in self.members
                              if not m.voice_desc and not m.same_as]
        if not targets:
            return
        listing = "\n".join(
            f"{m.name} ({self._who(m)}{m.quotes} lines): "
            f"{m.role or 'no description'} | sample lines: "
            + " / ".join(f"“{s}”" for s in m.samples[:3])
            for m in targets
        )
        prompt = (
            "You are designing ORIGINAL voices for audiobook characters. For "
            "each character below, write:\n"
            "1. voice: one sentence (max 35 words) describing how the voice "
            "SOUNDS. Start with gender and approximate age (e.g. \"A man in "
            "his fifties\", \"A young woman\"), matching the gender given "
            "in parentheses when known; then pitch, timbre/texture, accent, "
            "pace, and attitude. No plot, no names, and never reference a "
            "real person, actor or celebrity.\n"
            "2. line: an audition line of 20-30 words this character could "
            "plausibly say, in their manner but in an ordinary, calm moment "
            "(it becomes their NEUTRAL reference voice, so no shouting, "
            "panic or urgency). Plain spoken words only: no "
            "names, no stage directions, no quotation marks.\n"
            "Make the voices clearly distinct from one another.\n\n"
            f"Opening excerpt of the book:\n{self.excerpt[:1500]}\n\n"
            f"Characters:\n{listing}\n\n"
            'Reply with ONLY JSON: {"characters": {"<Name>": {"voice": '
            '"...", "line": "..."}}}'
        )
        data = self._json_block(self._chat(prompt, max_tokens=2400))
        chars = data.get("characters") or {}
        for name, info in chars.items():
            member = self._match_member(name)
            if member is None and len(targets) == len(chars) == 1:
                member = targets[0]     # asked about one, got one back
            if member not in targets or not isinstance(info, dict):
                continue
            member.voice_desc = str(info.get("voice") or "").strip()
            member.line = str(info.get("line") or "").strip().strip('"“”')
            if not member.accent:
                from . import design

                member.accent = design.detect_accent(member.voice_desc)
        missing = [m.name for m in targets if not m.voice_desc]
        if missing:
            log(f"warning: no voice description for {', '.join(missing)} "
                f"(LLM named: {', '.join(map(str, chars)) or 'nobody'})")

    def design_member(self, member: CastMember, new_take: bool = False,
                      log=print) -> bool:
        """Build *member*'s voice from its description; returns success."""
        from . import design

        if not member.voice_desc:
            log(f"{member.name}: no voice description yet — describe it first")
            return False
        if new_take:
            member.seed += 1
        line = member.line or (member.samples[0] if member.samples else
                               "It is a pleasure to finally make your acquaintance.")
        wav = design.design_clip(self.design_url,
                                 design.neutral_description(member.voice_desc),
                                 line, seed=member.seed)
        seconds = design.wav_seconds(wav)
        name = design.voice_name(self.book, member.name)
        design.save_reference(name, wav, line)
        gender = {"M": "Male", "F": "Female"}.get(member.gender, "")
        self.custom_roster = [r for r in self.custom_roster if r[0] != name]
        self.custom_roster.append((name, gender, member.voice_desc))
        member.voice = name
        design.set_voice_accent(name, member.accent, member.gender)
        note = "" if seconds <= 15 else " (long — F5 works best under 15 s)"
        log(f"  designed {member.name} ({seconds:.1f}s clip, take "
            f"{member.seed}){note}")
        if self.registers:
            # Emotional variants (tense, angry, sad, ...) of the same voice: F5
            # copies the reference clip's delivery, so one calm clip makes
            # every line calm. Picked per line by the emotion tagger.
            try:
                report = design.design_registers(
                    self.design_url, name, member.voice_desc, member.seed, wav, line)
                weak = [e for e, note in report.items() if "below target" in note]
                log(f"    registers: {len(report)} built"
                    + (f"; best-effort: {', '.join(weak)}" if weak else ""))
            except RuntimeError as exc:
                log(f"    (registers skipped: {exc})")
        if self.audition_dir is not None and self.f5_url:
            # Render a real line through F5 — the voice as the book will use it.
            text = max(member.samples, key=len) if member.samples else line
            try:
                stem = design.file_stem(member.name)
                n = 1 + len(list(self.audition_dir.glob(f"{stem}_v*.mp3"))) \
                    if self.audition_dir.exists() else 1
                # Every rebuild gets a new file (v1, v2, ...): a reworded
                # description keeps the same take, so seed-named files clash.
                path = design.save_audition(
                    design.f5_preview(self.render_url(name), name, text),
                    self.audition_dir, f"{stem}_v{n}")
                log(f"    audition: {path}")
            except OSError as exc:
                log(f"    (audition render failed: {exc})")
        return True

    def is_designed(self, member: CastMember) -> bool:
        from . import design

        return member.voice == design.voice_name(self.book, member.name) and (
            design.f5_voices_dir() / f"{member.voice}.wav").exists()

    def design_missing(self, log=print) -> int:
        """Design every character whose voice isn't built yet."""
        todo = [m for m in self.members
                if m.voice_desc and not m.same_as and not self.is_designed(m)]
        if todo:
            log(f"Designing {len(todo)} voice(s)...")
        done = 0
        for m in todo:
            try:
                done += self.design_member(m, log=log)
            except RuntimeError as exc:
                log(f"  {m.name}: {exc}")
                break
        return done

    def preview(self, member: CastMember | None, raw: bool = False,
                log=print) -> None:
        """Play a character (or the narrator, member=None) through F5.

        F5 renders one of the character's real lines from the book — what
        the audiobook will actually sound like. ``raw`` plays the designer's
        own audition clip instead.
        """
        import tempfile
        from pathlib import Path

        from . import design

        voice = member.voice if member else self.narrator_voice
        clip = design.f5_voices_dir() / f"{voice}.wav"
        who = design.file_stem(member.name if member else "narrator")
        if raw:
            if not clip.exists():
                log(f"no clip for {voice}")
                return
            wav = clip.read_bytes()
            who += "_design"
        else:
            if member and member.samples:
                text = max(member.samples, key=len)
            else:
                text = self.excerpt.strip().split("\n\n")[0][:240]
            log(f"  {voice}: {text}")
            try:
                wav = design.f5_preview(self.render_url(voice), voice, text)
            except OSError as exc:
                log(f"F5 preview failed ({exc}); using the design clip instead")
                if not clip.exists():
                    return
                wav = clip.read_bytes()
        if self.audition_dir is not None:
            log(f"  saved: {design.save_audition(wav, self.audition_dir, who + '_play')}")
        if self.no_play:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
        design.play(Path(f.name))
        Path(f.name).unlink(missing_ok=True)

    def refine(self, user_message: str) -> str:
        """One chat turn: apply the user's direction to the cast."""
        from . import design

        accents = design.available_accents()
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
            '"<voice or null>", "role": "<updated description or null>", '
            '"same_as": "<another Name if the director says they are the same '
            'person (\\"narrator\\" if they say this character is the '
            'narrator), \\"\\" if they say they are different people, else '
            'null>"'
            + (', "sounds_like": "<rewritten voice description (max 35 '
               'words, starting with gender and age, then pitch, timbre, '
               'accent, pace, attitude; never a real person) or null>"'
               + (', "accent": "<' + ' or '.join(accents) + ' if the '
                  'director gives them that accent, \\"none\\" to remove one, '
                  'else null>"' if accents else '')
               if self.designing else '')
            + '}}}'
            + ("\n\nThese characters have ORIGINAL designed voices: when "
               "the director describes how someone should sound, rewrite "
               "that character's sounds_like (keep what they did not ask to "
               "change) and leave voice null." if self.designing else "")
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
            sounds = info.get("sounds_like")
            if sounds and self.designing:
                member.voice_desc = str(sounds).strip()
            accent = str(info.get("accent") or "").strip().lower()
            if accent and self.designing:
                member.accent = "" if accent == "none" else accent
            if "same_as" in info and info["same_as"] is not None:
                if str(info["same_as"]).strip():
                    self._link(member, info["same_as"])
                else:
                    member.same_as = ""   # "they're different people"
        for key in ("narrator", "dialogue"):
            voice = str(data.get(key) or "").strip().lower()
            if voice and self.valid_voice(voice):
                setattr(self, f"{key}_voice", voice)
        reply = str(data.get("reply", "")).strip() or "Updated."
        self.history.append({"who": "director", "text": user_message})
        self.history.append({"who": "assistant", "text": reply})
        return reply

    # -- persistence -------------------------------------------------------
    def screen_voices(self, segments, log=print, lines_per_voice: int = 4) -> set[str]:
        """Render each cast voice on its own hardest lines (one take each,
        verified by the TTS server) and return the voices that fail most of
        them: some cloned references (cut mid-sentence, noisy) babble on
        Qwen, and the per-line retries can't rescue a voice like that."""
        def hardest(texts):
            texts = [" ".join(t.split()) for t in texts]
            texts = [" ".join(t.split()[:40]) for t in texts if len(t.split()) >= 3]
            return sorted(set(texts), key=len, reverse=True)[:lines_per_voice]

        narration = [p for s in segments if s.kind == "narration"
                     for p in re.split(r"(?<=[.!?])\s+", s.text)]
        lines = {self.narrator_voice: hardest(narration + [
            s.text for s in segments if s.kind == "quote" and s.speaker == "Narrator"])}
        for m in self.members:
            v = self.voice_of(m)
            lines.setdefault(v, []).extend(hardest(
                [s.text for s in segments if s.kind == "quote" and s.speaker == m.name]))
        # Verdicts are remembered per (voice, lines): a saved cast reopened
        # isn't re-rendered (minutes), and a coin-flip take can't flip it.
        cache_path = getattr(self, "screen_cache_path", None)
        try:
            remembered = json.loads(Path(cache_path).read_text()) if cache_path else {}
        except (OSError, ValueError):
            remembered = {}
        bad = set()
        for voice, texts in lines.items():
            texts = texts[:lines_per_voice]
            if len(texts) < 2:
                continue
            base, _, _ = voice.partition("@")
            key = base + "|" + hashlib.sha1("\n".join(texts).encode()).hexdigest()[:16]
            if key in remembered:
                if not remembered[key]:
                    bad.add(base)
                continue
            failed = 0
            for t in texts:
                body = json.dumps({"input": t, "voice": base, "verify": True,
                                   "tries": 1}).encode()
                try:
                    req = Request(self.render_url(base), data=body,
                                  headers={"Content-Type": "application/json"})
                    with urlopen(req, timeout=300) as resp:
                        resp.read()
                        verdict = json.loads(resp.headers.get("X-Verify") or "{}")
                except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                    return bad   # no verifying server: nothing to screen with
                if verdict and not verdict.get("ok", True):
                    failed += 1
            log(f"  screen {base}: {len(texts) - failed}/{len(texts)} clean")
            # Strict: a good clone passes its own hardest lines first time;
            # one that fails even one of four babbles all through a book
            # (simon on The Raven: 3/4 here, then re-renders on every page).
            if failed:
                bad.add(base)
            remembered[key] = not failed
            if cache_path:
                Path(cache_path).write_text(json.dumps(remembered, indent=1))
        return bad

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "engine": self.engine,
            "narrator": self.narrator_voice,
            "dialogue": self.dialogue_voice,
            "characters": {
                m.name: {"voice": m.voice, "role": m.role, "gender": m.gender,
                         "voice_desc": m.voice_desc, "line": m.line,
                         "seed": m.seed, "same_as": m.same_as,
                         "accent": m.accent}
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
            m.voice_desc = str(info.get("voice_desc") or "").strip()
            m.same_as = str(info.get("same_as") or "").strip()
            m.line = str(info.get("line") or "").strip()
            m.accent = str(info.get("accent") or "").strip().lower()
            try:
                m.seed = int(info.get("seed") or 0)
            except (TypeError, ValueError):
                m.seed = 0
        return applied

    # -- output ------------------------------------------------------------
    def voice_map(self) -> str:
        parts = [f"narrator={self.narrator_voice}"]
        if self.dialogue_voice:
            parts.append(f"dialogue={self.dialogue_voice}")
        parts.extend(f"{m.name}={self.voice_of(m)}" for m in self.members
                     if self.voice_of(m))
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
            if m.same_as:
                who = ("the first-person narrator — uses the narrator voice"
                       if m.same_as == "narrator"
                       else f"same person — shares {m.same_as}'s voice")
                lines.append(f"  {m.name.ljust(width)}  [= {m.same_as}]  {who}")
                continue
            if self.designing:
                state = "designed" if self.is_designed(m) else "not built"
                lines.append(f"  {m.name.ljust(width)}  [{state}]  {role}")
                lines.append(f"  {''.ljust(width)}  sounds: "
                             f"{m.voice_desc or '(no description yet)'}")
                if m.accent:
                    lines.append(f"  {''.ljust(width)}  accent: {m.accent} "
                                 "(real speaker's delivery, rendered by Qwen)")
            else:
                lines.append(f"  {m.name.ljust(width)}  {m.voice or '?':<6}  {role}")
        return "\n".join(lines)


_MALE_NOUNS = {"man", "men", "gentleman", "boy", "lad", "father", "son", "king",
               "brother", "uncle", "husband", "lord", "sir", "mr", "prince",
               "grandfather", "nephew", "monk", "priest", "duke", "fellow"}
_FEMALE_NOUNS = {"woman", "women", "lady", "girl", "lass", "mother", "daughter",
                 "queen", "sister", "aunt", "wife", "mrs", "miss", "princess",
                 "grandmother", "niece", "nun", "duchess", "madam", "maid"}


_AGE_BANDS = {"child": (0, 15), "young": (16, 30), "middle": (31, 55), "old": (56, 120)}


def _age_fits(description: str, band: str) -> bool:
    """False when *description* gives the speaker's age ("English man, 22
    from Surrey"; "British male, 56-65") and it falls outside *band*;
    True when either is unknown. Allows 5 years' slack at the edges."""
    if band not in _AGE_BANDS:
        return True
    m = re.search(r"\b(?:man|woman|male|female|boy|girl),?\s+(\d{1,2})(?:-(\d{1,2}))?\b",
                  description, re.I)
    if not m:
        return True
    age = (int(m.group(1)) + int(m.group(2) or m.group(1))) / 2
    lo, hi = _AGE_BANDS[band]
    return lo - 5 <= age <= hi + 5


def _noun_gender(name: str) -> str:
    """'M'/'F' from a generic name like "Old Man" or "Lady"; '' otherwise."""
    words = set(re.findall(r"[a-z]+", name.lower()))
    male, female = bool(words & _MALE_NOUNS), bool(words & _FEMALE_NOUNS)
    return "M" if male and not female else "F" if female and not male else ""


def _person_of(voice: str, description: str) -> str:
    """Who a library voice was cloned from: 'Deep, gravelly narrator (Bob
    Neufeld, LibriVox Emerson)' -> 'bob neufeld'; the voice name otherwise."""
    m = re.search(r"\(([A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*)+)\s*,", description or "")
    return m.group(1).lower() if m else voice

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


def fetch_voice_roster(speech_url: str) -> list[tuple[str, str, str]]:
    """Ask a TTS server (e.g. F5) for its /voices registry."""
    from urllib.parse import urlparse

    p = urlparse(speech_url)
    with urlopen(f"{p.scheme}://{p.netloc}/voices", timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    return [
        (str(v["name"]), str(v.get("gender", "")),
         str(v.get("description", "cloned voice")))
        for v in data if isinstance(v, dict) and v.get("name")
    ]


def build_session(text: str, engine: str, llm_url: str | None,
                  llm_model: str, top: int, llm_context: int = 6000,
                  cache_path=None, saved: dict | None = None,
                  custom_roster: list | None = None,
                  design_url: str | None = None, book: str = "",
                  f5_url: str = "", qwen_url: str = "", audition_dir=None,
                  no_play: bool = False,
                  registers: bool = True, fill_voices: bool = False,
                  design_fallback_url: str = "", log=print) -> CastSession:
    """Parse *text*, attribute speakers, and assemble the initial cast."""
    log("Parsing dialogue...")
    # Parse the same speech-prepped text the converter will synthesize, so
    # cached attributions line up between the session and the conversion.
    segments = segment_dialogue(prepare_for_speech(text))
    session = CastSession(engine=engine, llm_url=llm_url, llm_model=llm_model,
                          custom_roster=custom_roster or [],
                          design_url=design_url, book=book, f5_url=f5_url,
                          qwen_url=qwen_url,
                          audition_dir=audition_dir, no_play=no_play,
                          registers=registers, fill_voices=fill_voices,
                          design_fallback_url=design_fallback_url)
    if cache_path:
        session.screen_cache_path = str(cache_path).replace(".speakers.json", "") + ".screened.json"

    attributor = None
    if llm_url:
        attributor = LlmAttributor(
            llm_url, model=llm_model, context_chars=llm_context
        )
    roster = {}
    try:
        roster = attribute_segments(
            segments, attributor=attributor, cache_path=cache_path, log=log
        )
    except RuntimeError as exc:
        log(f"warning: LLM attribution skipped: {exc}")
        session.llm_url = None

    # The first-person narrator's lines already use the narrator voice.
    counts = {k: v for k, v in character_counts(segments).items() if k != "Narrator"}
    for name, n in list(counts.items())[:top]:
        samples = [
            s.text.strip().replace("\n", " ")[:90]
            for s in segments if s.kind == "quote" and s.speaker == name
        ][:4]
        contexts = [
            "…" + " ".join(segments[i - 1].text.split())[-140:] + " “"
            + " ".join(s.text.split())[:110] + "”"
            for i, s in enumerate(segments)
            if s.kind == "quote" and s.speaker == name and i > 0
        ]
        step = max(1, len(contexts) // 4)   # spread across the book
        member = CastMember(name=name, quotes=n, samples=samples,
                            contexts=contexts[::step][:4])
        entry = roster.get(name)
        if entry:
            # Titles are firmer evidence of gender than an LLM guess.
            member.gender = entry.gender
            member.aliases = " / ".join(sorted(entry.aliases, key=len, reverse=True)[:2])
        if not member.gender:
            member.gender = _noun_gender(name)
        session.members.append(member)

    # Sensible defaults before (or without) the LLM's suggestions.
    roster = [n for n, _, _ in session.offered()]
    if not roster:
        raise RuntimeError("no voices available for this engine")
    if session.custom_roster:
        session.narrator_voice = roster[0]
        session.dialogue_voice = roster[1] if len(roster) > 1 else roster[0]
    else:
        session.narrator_voice = "leo" if engine == "orpheus" else "ryan"
        session.dialogue_voice = "tara" if engine == "orpheus" else "amy"
    for member, voice in zip(session.members, roster):
        member.voice = voice if voice not in (
            session.narrator_voice, session.dialogue_voice
        ) else roster[-1]

    # Strata-class servers take long prompts cheaply; more of the book means
    # better role descriptions (who "the Ghost" is, who narrates).
    session.excerpt = text[:24000]

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
            _screen_and_recast(session, segments, log, recast=False)
            _prepare_designs(session, log)
            return session

    if session.llm_url:
        log("Asking the LLM to describe each character and suggest voices...")
        try:
            session.describe_and_suggest(session.excerpt, log=log)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            log(f"warning: role descriptions unavailable ({exc}); "
                "using defaults — you can still assign voices directly.")
            session.llm_url = None
    if session.fill_voices and session.llm_url and session.custom_roster \
            and not session.designing:
        from .voice_fill import fill_gaps

        log("Reviewing how well each voice fits its role...")
        if fill_gaps(session, log=log):
            session._dedupe_voices()
            log("Cast after filling gaps:\n" + session.table())
    _screen_and_recast(session, segments, log)
    _prepare_designs(session, log)
    return session


def _screen_and_recast(session: CastSession, segments, log=print, rounds: int = 2,
                       recast: bool = True) -> None:
    """Screen the cast's voices on the server; drop failures from the roster
    and let the LLM recast (up to *rounds* times). recast=False (a cast the
    user already directed): only report the failures."""
    if session.designing or not session.custom_roster or not session.llm_url:
        return
    log("Screening the cast's voices on their hardest lines...")
    for _ in range(rounds):
        bad = session.screen_voices(segments, log=log)
        if not bad:
            return
        if not recast:
            log(f"  {', '.join(sorted(bad))} stumbled on some of their lines when screened; "
                "kept because this cast was chosen — 'Name = voice' to change it")
            return
        log(f"  {', '.join(sorted(bad))} failed screening on this text — recasting without it")
        session.custom_roster = [r for r in session.custom_roster if r[0] not in bad]
        try:
            session.describe_and_suggest(session.excerpt, log=log)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            log(f"warning: recast failed ({exc})")
            return
        log("Recast:\n" + session.table())

def _prepare_designs(session: CastSession, log=print) -> None:
    """Designed-voice mode: describe how each character sounds, then build
    the voices so they can be auditioned right away."""
    from . import design

    if not session.designing:
        return
    if session.llm_url and any(not m.voice_desc for m in session.members):
        log("Writing a voice description for each character...")
        try:
            session.write_voice_descriptions(log=log)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            log(f"warning: voice descriptions unavailable ({exc})")
    if not design.server_ready(session.design_url):
        log(f"note: voice-design server not ready at {session.design_url} — "
            "start it, then use 'design' (characters keep library voices "
            "until designed).")
        return
    session.design_missing(log=log)


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

DESIGN_HELP = """Designed voices (cast --design):
  <plain language>        e.g. "Holmes should sound older and drier" — the
                          description is rewritten and the voice rebuilt
  play Name               hear Name read one of their lines through F5
                          ('play narrator' too; 'play Name raw' = design clip);
                          also saved as an MP3 in the auditions folder
  redesign Name           another take: same description, new voice
  accent Name scottish    give Name a real accent (also: irish; 'none' removes)
  design [Name]           build voices that aren't built yet (or just Name)
  proceed                 build anything missing, then convert the book
"""


def run_repl(session: CastSession, on_convert, state_path=None, log=print,
             lines=None) -> int:
    """The interactive loop. ``on_convert(voice_map, output)`` runs the job.
    *lines*: commands to run instead of reading the prompt (scripts, agents);
    the session is saved and the loop ends when they run out."""
    feed = iter(lines) if lines is not None else None

    def checkpoint():
        if state_path is not None:
            save_cast(session, state_path)

    def find(name: str):
        """Exact name, else the one character whose name contains it
        ("Roylott" -> "Grimesby Roylott"); None when unknown or ambiguous."""
        from .textprep import unspell

        lowered = unspell(name.strip()).lower()
        exact = [m for m in session.members if m.name.lower() == lowered]
        if exact:
            return exact[0]
        partial = [m for m in session.members if lowered and lowered in m.name.lower()]
        if len(partial) > 1:
            log(f"'{name}' matches {', '.join(m.name for m in partial)} — be more specific")
        return partial[0] if len(partial) == 1 else None

    def rebuild(members, new_take=False):
        if not session.designing:
            return
        from . import design

        if not design.server_ready(session.design_url):
            log("(voice-design server not ready — 'design' once it is up)")
            return
        for m in members:
            try:
                session.design_member(m, new_take=new_take, log=log)
            except RuntimeError as exc:
                log(f"  {m.name}: {exc}")
                return
        checkpoint()

    log("\nProposed cast:\n" + session.table())
    if session.designing:
        log("\n" + DESIGN_HELP.rstrip())
    log("\nDirect me — plain English or 'help' for commands.\n")
    checkpoint()  # the initial suggestion is worth keeping too
    while True:
        try:
            if feed is not None:
                line = next(feed).strip()
                log(f"cast> {line}")
            else:
                line = input("cast> ").strip()
        except (EOFError, KeyboardInterrupt, StopIteration):
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
            log(HELP + ("\n" + DESIGN_HELP if session.designing else ""))
        elif session.designing and cmd.split()[0] == "play":
            rest = line.split(None, 1)[1] if " " in line else ""
            raw = rest.lower().endswith(" raw")
            target = rest[:-4] if raw else rest
            if target.strip().lower() == "narrator":
                session.preview(None, raw=raw, log=log)
            elif (member := find(target)) is not None:
                session.preview(member, raw=raw, log=log)
            else:
                log("usage: play <Name> [raw] — try 'cast' for names")
        elif session.designing and cmd.split()[0] == "accent":
            from . import design

            parts = line.split()
            member = find(" ".join(parts[1:-1])) if len(parts) >= 3 else None
            accent = parts[-1].lower() if len(parts) >= 3 else ""
            if member is None or (accent != "none"
                                  and accent not in design.available_accents()):
                log("usage: accent <Name> <" + "|".join(
                    design.available_accents() + ["none"]) + ">")
            else:
                member.accent = "" if accent == "none" else accent
                session.apply_accent(member, log=log)
                checkpoint()
        elif session.designing and cmd.split()[0] == "redesign":
            member = find(line.split(None, 1)[1]) if " " in line else None
            if member is None:
                log("usage: redesign <Name>")
            else:
                rebuild([member], new_take=True)
        elif session.designing and cmd.split()[0] == "design":
            if " " in line:
                member = find(line.split(None, 1)[1])
                if member is None:
                    log("no such character — try 'cast'")
                else:
                    rebuild([member])
            else:
                from . import design

                if design.server_ready(session.design_url):
                    session.design_missing(log=log)
                    checkpoint()
                else:
                    log(f"voice-design server not ready at {session.design_url}")
        elif session.designing and cmd.startswith("proceed"):
            from . import design

            missing = [m for m in session.members
                       if not m.same_as and not session.is_designed(m)]
            if missing and design.server_ready(session.design_url):
                session.design_missing(log=log)
                missing = [m for m in session.members
                           if not m.same_as and not session.is_designed(m)]
            if missing:
                log("not designed yet (they'll use their library voice): "
                    + ", ".join(m.name for m in missing))
            output = line.split(None, 1)[1].strip() if " " in line else None
            log(f'\nCast locked in: --voice-map "{session.voice_map()}"')
            checkpoint()
            return on_convert(session.voice_map(), output)
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
                    before = {m.name: m.voice_desc for m in session.members}
                    accents = {m.name: m.accent for m in session.members}
                    log(session.refine(line))
                    checkpoint()
                    orphans = [m for m in session.members
                               if session.designing and not m.same_as
                               and not m.voice_desc]
                    if orphans:   # just unlinked: needs a voice of its own
                        session.write_voice_descriptions(members=orphans, log=log)
                    changed = [m for m in session.members
                               if m.voice_desc != before[m.name]]
                    for m in session.members:
                        if m.accent != accents[m.name] and m not in changed:
                            session.apply_accent(m, log=log)
                    if changed:
                        for m in changed:
                            log(f"  {m.name} now sounds: {m.voice_desc}")
                        rebuild(changed)
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    log(f"LLM unavailable ({exc}); use 'Name = voice' instead")
            else:
                log("No LLM connected — use 'Name = voice', or pass --llm-url "
                    "to direct the cast in plain language.")
