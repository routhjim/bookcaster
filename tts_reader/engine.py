"""Text-to-speech engines.

Two backends share one interface so the rest of the app (chapters, MP3
writing, splitting) does not care which is in use:

* :class:`PiperEngine`   — fast, fully local VITS models (default).
* :class:`OrpheusEngine` — a client for a locally-hosted Orpheus TTS server
  (Llama-3B + SNAC) exposing an OpenAI-compatible ``/v1/audio/speech`` endpoint.
  Much more natural / expressive; runs on your GPU via llama.cpp.

Every engine implements ``key``, ``sample_rate``, ``load()`` and
``synthesize_pcm()``; the base class turns PCM into WAV/MP3 for everyone.
"""

from __future__ import annotations

import io
import json
import os
import re as _re
import wave
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from .textprep import VERSE_BREAK
from urllib.request import Request, urlopen

from . import voices as voice_catalog


def default_models_dir() -> Path:
    """Where downloaded Piper voice models are cached.

    Overridable with the ``TTS_READER_MODELS_DIR`` environment variable.
    """
    env = os.environ.get("TTS_READER_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "tts_reader" / "voices"


class BaseEngine:
    """Shared PCM -> WAV/MP3 plumbing. Subclasses provide the synthesis."""

    key: str = "engine"

    @property
    def sample_rate(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def load(self, log=print) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def synthesize_pcm(
        self, text: str, speed: float = 1.0, volume: float = 1.0, log=print
    ) -> bytes:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- shared outputs ---------------------------------------------------
    def synthesize_to_wav(
        self, text: str, wav_path: str | Path, speed: float = 1.0,
        volume: float = 1.0, log=print,
    ) -> None:
        pcm = self.synthesize_pcm(text, speed=speed, volume=volume, log=log)
        wav_path = Path(wav_path)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm)

    def synthesize_to_mp3(
        self, text: str, mp3_path: str | Path, speed: float = 1.0,
        volume: float = 1.0, bitrate: int = 128, log=print,
    ) -> None:
        from .audio import pcm_to_mp3

        pcm = self.synthesize_pcm(text, speed=speed, volume=volume, log=log)
        pcm_to_mp3(pcm, mp3_path, self.sample_rate, bitrate=bitrate)


class PiperEngine(BaseEngine):
    """Loads a single Piper voice and synthesizes speech from text."""

    def __init__(self, voice: str, models_dir: Optional[str | Path] = None):
        self.key = voice_catalog.resolve_key(voice)
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self._voice = None

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.key}.onnx"

    def is_downloaded(self) -> bool:
        return self.model_path.exists() and self.model_path.with_suffix(".onnx.json").exists()

    def ensure_downloaded(self, log=print) -> Path:
        if self.is_downloaded():
            return self.model_path
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if log:
            log(f"Downloading voice '{self.key}' (first use only)...")
        from piper.download_voices import download_voice

        try:
            download_voice(self.key, self.models_dir)
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Could not download voice '{self.key}'. This step needs internet "
                f"access to huggingface.co the first time a voice is used.\n"
                f"Underlying error: {exc}"
            ) from exc
        return self.model_path

    def load(self, log=print) -> None:
        self.ensure_downloaded(log=log)
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(self.model_path)

    @property
    def sample_rate(self) -> int:
        if self._voice is None:
            self.load()
        return int(self._voice.config.sample_rate)

    def synthesize_pcm(
        self, text: str, speed: float = 1.0, volume: float = 1.0, log=print
    ) -> bytes:
        if self._voice is None:
            self.load(log=log)
        if speed <= 0:
            raise ValueError("speed must be greater than 0")
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(length_scale=1.0 / speed, volume=volume)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())


# Orpheus' SNAC decoder always emits 24 kHz mono audio.
ORPHEUS_SAMPLE_RATE = 24000


class OrpheusEngine(BaseEngine):
    """Client for a local Orpheus TTS server (OpenAI-compatible endpoint).

    Point ``base_url`` at the server's ``/v1/audio/speech`` route. The heavy
    Llama-3B + SNAC work happens in that server (on your GPU); this class just
    sends text and reads back audio.
    """

    def __init__(
        self,
        voice: str,
        base_url: str = "http://127.0.0.1:5005/v1/audio/speech",
        model: str = "orpheus",
        timeout: float = 600.0,
        chunk_chars: int = 400,
        emotion_param: bool = False,
    ):
        # True for servers (F5) that select an emotional reference clip via
        # an "emotion" field in the request, rather than inline text tags.
        self.emotion_param = emotion_param
        # Per-request verification reports (X-Verify header from servers
        # that check their own output, e.g. tools/qwen_tts_server.py).
        self.verify_log: list[dict] = []
        self.key = f"orpheus:{voice_catalog.resolve_orpheus(voice)}"
        self.voice = voice_catalog.resolve_orpheus(voice)
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        # Orpheus is an LLM; overly long inputs blow past its token cap. Split
        # long passages on sentence boundaries and stitch the audio back.
        self.chunk_chars = chunk_chars

    @property
    def sample_rate(self) -> int:
        # The server says its rate in /health (Fish: 44.1 kHz); else 24 kHz.
        return getattr(self, "_rate", None) or ORPHEUS_SAMPLE_RATE

    def _origin(self) -> str:
        p = urlparse(self.base_url)
        return f"{p.scheme}://{p.netloc}"

    def load(self, log=print) -> None:
        """Check that the Orpheus server is reachable, with a helpful error."""
        health = self._health()
        self.inline_tags = bool(health.get("inline_tags"))
        self.max_batch = max(1, int(health.get("max_batch") or 1))
        self._rate = int(health.get("sample_rate") or ORPHEUS_SAMPLE_RATE)
        # The server can take earlier takes as context and match a take's
        # pitch/spectrum (Fish adapter): used to steady long narration.
        self.takes_context = bool(health.get("context"))
        self.anchor = None          # {"text", "codes"}: this voice's narration anchor
        try:
            urlopen(self._origin(), timeout=10)
        except HTTPError:
            return  # server answered (any status) => it is up
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach the Orpheus server at {self._origin()} ({exc.reason}). "
                "Start it first (llama.cpp + Orpheus-FastAPI) and point --orpheus-url "
                "at its /v1/audio/speech endpoint. See README (Orpheus setup)."
            ) from exc

    def _health(self) -> dict:
        """The server's /health: "inline_tags" (its model acts on delivery
        tags like [pause]) and "max_batch" (requests it decodes together) —
        tools/fish_tts_server.py reports both."""
        try:
            with urlopen(f"{self._origin()}/health", timeout=10) as resp:
                info = json.loads(resp.read())
            return info if isinstance(info, dict) else {}
        except (HTTPError, URLError, OSError, ValueError):
            return {}

    def _request_pcm(self, text: str, speed: float, emotion: str = "", sink=None,
                     pitch: float = 0.0, pitch_range: float = 1.0, steady: str = "",
                     codes_out: list | None = None) -> bytes:
        body = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "wav",
            "speed": speed,
        }
        if emotion and self.emotion_param:
            body["emotion"] = emotion
        if pitch or pitch_range != 1.0:
            # Register shift for the cue (Fish adapter applies it, formants kept).
            body["pitch"], body["pitch_range"] = round(pitch, 2), round(pitch_range, 3)
        if getattr(self, "temperature", 0):
            body["temperature"] = self.temperature   # sampling (Fish); 0 = server default
        if steady and self.anchor:
            # Every narration chunk is shown the same anchor take, so they
            # share one pitch, pace and tone instead of each drifting.
            body["context"] = [self.anchor]
            if steady == "match":
                body["match"] = self.anchor["codes"]
        payload = json.dumps(body).encode("utf-8")
        req = Request(
            self.base_url, data=payload,
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                audio = resp.read()
                verdict = resp.headers.get("X-Verify")
                if codes_out is not None:
                    codes_out.append(resp.headers.get("X-Codes", ""))
            if verdict:
                try:
                    (self.verify_log if sink is None else sink).append(
                        {"voice": self.voice, "emotion": emotion, "text": text,
                         **json.loads(verdict)})
                except ValueError:
                    pass
        except HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(
                f"Orpheus server returned HTTP {exc.code}: {body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Orpheus request failed ({exc.reason}). Is the server still running?"
            ) from exc
        return _wav_bytes_to_pcm(audio)

    def render(self, text: str, speed: float = 1.0, volume: float = 1.0,
               emotion: str = "", level_db: float = 0.0, pitch: float = 0.0,
               pitch_range: float = 1.0, steady: str = "") -> tuple[bytes, list[dict]]:
        """synthesize_pcm that returns its verification reports instead of
        logging them on the engine — safe to call from several threads."""
        sink: list[dict] = []
        chunks = _chunk_text(text, self.chunk_chars) if self.chunk_chars > 0 else [text]
        # An inline delivery tag holds until the next one: a chunk cut from
        # the middle of a tagged quote starts with the tag still in force.
        tag, carried = "", []
        for c in chunks:
            if tag and not _re.match(r"\s*\[[^\]]+\]", c):
                c = f"{tag} {c}"
            # The last stacked cue group ([furious][shouting]) is what's in
            # force; [emphasis] marks one word, it doesn't carry.
            groups = [g.replace("[emphasis]", "") for g in
                      _re.findall(r"(?:\[[^\]]+\])+", c)]
            groups = [g for g in groups if g]
            tag = groups[-1] if groups else tag
            carried.append(c)
        # Each chunk is a separate take at whatever level the model chose;
        # with level_db (narration) they're levelled one by one so a long
        # paragraph doesn't jump in volume where its chunks meet.
        pcm = b"".join(
            level_speech(a, self.sample_rate, level_db) if level_db else a
            for a in (self._request_pcm(c, speed, emotion=emotion, sink=sink,
                                        pitch=pitch, pitch_range=pitch_range,
                                        steady=steady)
                      for c in carried if c.strip()))
        if volume != 1.0:
            pcm = _apply_volume(pcm, volume)
        return pcm, sink

    def make_anchor(self, text: str, speed: float = 1.0) -> bool:
        """Render *text* once as this voice's narration anchor (a take every
        later narration chunk is shown). False if the server can't."""
        if not getattr(self, "takes_context", False):
            return False
        got: list = []
        self._request_pcm(text, speed, codes_out=got)
        if got and got[0]:
            self.anchor = {"text": text, "codes": got[0]}
        return self.anchor is not None

    def synthesize_pcm(
        self, text: str, speed: float = 1.0, volume: float = 1.0, log=print,
        emotion: str = "",
    ) -> bytes:
        chunks = _chunk_text(text, self.chunk_chars) if self.chunk_chars > 0 else [text]
        pcm = b"".join(
            self._request_pcm(c, speed, emotion=emotion) for c in chunks if c.strip()
        )
        if volume != 1.0:
            pcm = _apply_volume(pcm, volume)
        return pcm


def parse_voice_spec(spec: str) -> tuple[str, float]:
    """Split 'zac@0.9' into (voice, rate). A bare voice means rate 1.0.

    Rate variants let one voice play several characters distinguishably
    (slightly faster/slower delivery) once a roster runs out of voices.
    """
    base, at, rate = str(spec).partition("@")
    base = base.strip()
    if not at:
        return base, 1.0
    try:
        value = float(rate)
    except ValueError:
        raise ValueError(
            f"bad voice spec {spec!r}: expected voice@rate like 'zac@0.9'"
        ) from None
    if not 0.5 <= value <= 2.0:
        raise ValueError(f"bad voice spec {spec!r}: rate must be 0.5-2.0")
    return base, value


class CastEngine(BaseEngine):
    """Multi-voice wrapper: narration and each character get their own voice.

    Wraps one underlying engine per distinct voice and routes each dialogue
    segment (see :mod:`tts_reader.speakers`) to the voice its speaker was
    assigned in ``voice_map``. Values may carry a speed variant
    (``zac@0.9``) so one voice can play multiple characters. Reserved keys:

    * ``narrator`` (required) — everything outside quotation marks
    * ``dialogue`` — fallback for quotes whose speaker is unknown/unmapped
      (defaults to the narrator's voice)
    """

    def __init__(
        self, voice_map: dict[str, str], engine_factory, attributor=None,
        pause_ms: int = 250, emotion_tagger=None,
        paragraph_ms: int = 650, stanza_ms: int = 900, line_ms: int = 300,
        verse_pacing: str = "auto", parallel: int = 0, speech_level_db: float = -20.0,
        steady_narration: str = "anchor",
    ):
        # Pacing: silence at paragraph breaks, between stanzas, and at the
        # end-stopped lines of verse (textprep keeps verse line ends).
        self.paragraph_ms, self.stanza_ms, self.line_ms = paragraph_ms, stanza_ms, line_ms
        # "silence": verse is rendered line by line with silence between;
        # "tags": a stanza goes to the model whole, its line ends marked with
        # delivery tags ([pause] / [short pause]) that models like Fish S2
        # act on, so the model paces the verse itself; "auto": tags for
        # voices whose server says it understands them, else silence.
        self.verse_pacing = verse_pacing
        # Lines in flight at once: 0 = what the voice servers say they batch
        # ("max_batch" in /health; 1 for servers that render one at a time).
        self.parallel = parallel
        # Narration chunks are brought to this speech level (dBFS, pauses
        # ignored) so the narrator doesn't jump between takes; dialogue is
        # left alone. 0 = off.
        self.speech_level_db = speech_level_db
        # Long narration is rendered in chunks, each sampled on its own, so
        # pitch and tone drift where they meet. "anchor": every narration
        # chunk of a voice is shown one shared take of that voice (servers
        # that take context); "match": also nudged to the anchor's pitch and
        # spectrum; "off": independent chunks.
        self.steady_narration = steady_narration
        self.emotion_tagger = emotion_tagger
        # Set per input by the CLI, like cache_path.
        self.emotions_cache_path = None
        self.voice_map = {k.strip().lower(): v for k, v in voice_map.items()}
        self.pause_ms = pause_ms
        if "narrator" not in self.voice_map:
            raise ValueError("voice map needs a 'narrator=<voice>' entry")
        self.assignments = {
            k: parse_voice_spec(v) for k, v in self.voice_map.items()
        }
        self.default_quote = self.assignments.get(
            "dialogue", self.assignments["narrator"]
        )
        self.engines = {
            base: engine_factory(base)
            for base in sorted({b for b, _ in self.assignments.values()})
        }
        self.attributor = attributor
        self.verify_log: list[dict] = []   # per line, with the speaker
        # Where each segment starts in the rendered audio (seconds, per
        # synthesize call) — lets a soundtrack be cued against scenes.
        self.timeline: list[dict] = []
        # Set per input by the CLI so attributions persist in a sidecar file.
        self.cache_path = None
        self.key = "cast(" + ", ".join(
            f"{k}={v}" for k, v in sorted(self.voice_map.items())
        ) + ")"

    @property
    def sample_rate(self) -> int:
        rates = {e.sample_rate for e in self.engines.values()}
        if len(rates) > 1:
            raise RuntimeError(
                "All cast voices must share one sample rate; got "
                + ", ".join(
                    f"{v}={e.sample_rate}" for v, e in self.engines.items()
                )
                + ". Pick Piper voices of the same quality tier (e.g. all "
                "-medium/-high), or use --engine orpheus."
            )
        return rates.pop()

    def load(self, log=print) -> None:
        for eng in self.engines.values():
            eng.load(log=log)
        self.sample_rate  # fail fast on mismatched rates

    def prepare(self, text: str, log=print) -> list:
        """The LLM half of a render: who speaks each quote, and (with an
        emotion tagger) how each line is delivered. Both are cached, so
        running this ahead leaves the render only the voices to do."""
        from .speakers import attribute_segments, segment_dialogue

        segments = segment_dialogue(text)
        attribute_segments(
            segments, attributor=self.attributor,
            cache_path=self.cache_path, log=log,
        )
        if self.emotion_tagger is not None:
            self.emotion_tagger.tag(
                segments, cache_path=self.emotions_cache_path, log=log
            )
        return segments

    def synthesize_pcm(
        self, text: str, speed: float = 1.0, volume: float = 1.0, log=print
    ) -> bytes:
        segments = self.prepare(text, log=log)
        # Plan first (silences, pieces, timeline marks), then render the
        # pieces — several at once when the voice server batches requests
        # (self.parallel) — then assemble in order.
        plan: list = []        # ("gap", ms) | ("mark", entry) | ("piece", job)
        prev_assignment = None
        carry_ms = 0          # a paragraph break owed from the previous segment
        for seg in segments:
            # Narration tags arrive as ', said Ahab.' — drop the leading comma.
            spoken = _re.sub(r"^[\s,;:—\-]+", "", seg.text).strip()
            if not _re.search(r"\w", spoken):
                carry_ms = max(carry_ms, self._break_ms(seg.text))
                continue
            assignment = self._assignment_for(seg)
            lead = seg.text[: len(seg.text) - len(seg.text.lstrip())]
            gap = max(carry_ms, self._break_ms(lead))
            # A short beat when the voice changes keeps switches from feeling
            # abrupt (self.pause_ms).
            if prev_assignment is not None and assignment != prev_assignment:
                gap = max(gap, self.pause_ms)
            if prev_assignment is not None and gap:
                plan.append(("gap", gap))
            tail = seg.text[len(seg.text.rstrip()):]
            carry_ms = self._break_ms(tail, verse=VERSE_BREAK in seg.text)
            base, rate = assignment
            engine = self.engines[base]
            extra = {}
            runs = None
            if seg.kind == "quote" and seg.sentence_tags and getattr(engine, "inline_tags", False):
                # Fish: tags go inline where the delivery changes, so a
                # threat can build from [angry] to [shouting] mid-line; each
                # change starts a new take whose loudness follows the cue.
                from .speakers import cue_db, cue_pitch, cue_runs

                runs = [(text, cue_db(cue) if cue else None, cue_pitch(cue))
                        for cue, text in cue_runs(spoken, seg.sentence_tags)]
            elif seg.kind == "quote" and seg.emotion:
                if getattr(engine, "emotion_param", False):
                    # F5-style: the server swaps in an emotional reference
                    # clip of the same narrator for this line.
                    extra["emotion"] = seg.emotion
                else:
                    # Orpheus-style: inline vocalization tag in the text.
                    spoken = f"<{seg.emotion}> {spoken}"
            plan.append(("mark", {
                "kind": seg.kind,
                "speaker": (seg.speaker or "") if seg.kind == "quote" else "narrator",
                "emotion": seg.emotion or "", "text": spoken}))   # whole: video subtitles read it
            speaker = (seg.speaker or "(unattributed)") if seg.kind == "quote" else "narrator"
            tags = self.verse_pacing == "tags" or (
                self.verse_pacing == "auto" and getattr(engine, "inline_tags", False))
            if runs:
                for text, offset, (st, rng) in runs:
                    level = None if offset is None else self.speech_level_db + offset
                    shaped = dict(extra)
                    if st or rng != 1.0:
                        shaped.update(pitch=st, pitch_range=rng)
                    plan.append(("piece", (engine, text, speed * rate, shaped, speaker, level)))
                prev_assignment = assignment
                continue
            for k, (ms, piece) in enumerate(self._paced_units(spoken, tags=tags)):
                if k and ms:
                    plan.append(("gap", ms))
                # Narration is levelled; plain dialogue keeps the model's level.
                level = self.speech_level_db if seg.kind == "narration" else None
                shaped = extra
                if (seg.kind == "narration" and self.steady_narration != "off"
                        and getattr(engine, "takes_context", False)):
                    shaped = {**extra, "steady": self.steady_narration}
                plan.append(("piece", (engine, piece, speed * rate, shaped, speaker, level)))
            prev_assignment = assignment

        def render(job):
            engine, piece, spd, extra, speaker, level = job
            # level (dBFS) is set for narration (each chunk, so the narrator
            # doesn't jump between takes) and for cued dialogue (from the
            # cue's intensity); None = keep the model's own level.
            if not self.speech_level_db or level is None:
                level = 0.0
            if hasattr(engine, "render"):
                return engine.render(piece, speed=spd, volume=volume, level_db=level, **extra)
            audio = engine.synthesize_pcm(piece, speed=spd, volume=volume, log=None, **extra)
            if level:
                audio = level_speech(audio, self.sample_rate, level)
            return audio, []

        jobs = [item[1] for item in plan if item[0] == "piece"]
        self._make_anchors(jobs, log)
        limits = self.server_limits()
        if sum(limits.values()) > 1:
            # Each server gets as many lines in flight as it batches (Fish
            # decodes them together, whatever their voices); a one-at-a-time
            # server (Qwen) in the same cast doesn't hold the others back.
            import threading

            gates = {srv: threading.BoundedSemaphore(n) for srv, n in limits.items()}
            plain = render

            def render(job):
                with gates[self._server_of(job[0])]:
                    return plain(job)
        workers = max(1, min(sum(limits.values()), len(jobs)))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = iter(list(pool.map(render, jobs)))
        else:
            results = (render(j) for j in jobs)
        silence = lambda ms: b"\x00\x00" * int(self.sample_rate * ms / 1000)
        pcm = bytearray()
        for kind, item in plan:
            if kind == "gap":
                pcm += silence(item)
            elif kind == "mark":
                self.timeline.append({"t": round(len(pcm) / 2 / self.sample_rate, 2), **item})
            else:
                audio, verdicts = next(results)
                pcm += audio
                for entry in verdicts:
                    self.verify_log.append({"speaker": item[4], **entry})
        return bytes(pcm)

    def _make_anchors(self, jobs, log=print) -> None:
        """Before the steady narration renders, give each of its voices an
        anchor take: a full-length narration chunk (first of this text with
        at least ~250 characters, else the longest). Kept for the run, so
        every chapter shares it."""
        firsts: dict = {}
        for engine, piece, spd, extra, _speaker, _level in jobs:
            if not extra.get("steady") or engine.anchor is not None:
                continue
            for c in (_chunk_text(piece, engine.chunk_chars) if engine.chunk_chars > 0 else [piece]):
                best = firsts.get(id(engine))
                if best is None or (len(best[1]) < 250 and len(c) > len(best[1])):
                    firsts[id(engine)] = (engine, c.strip(), spd)
        for engine, text, spd in firsts.values():
            if text and engine.make_anchor(text, spd) and log:
                log(f"  narration anchor for {engine.voice}: {text[:60]!r}...")

    @staticmethod
    def _server_of(engine) -> str:
        origin = getattr(engine, "_origin", None)
        return origin() if callable(origin) else f"local:{id(engine)}"

    def server_limits(self) -> dict[str, int]:
        """Lines in flight per voice server: what it batches ("max_batch"
        in /health), or --parallel when given; in-process engines get 1."""
        limits: dict[str, int] = {}
        for e in self.engines.values():
            srv = self._server_of(e)
            n = self.parallel if self.parallel > 0 else getattr(e, "max_batch", 1)
            if srv.startswith("local:"):
                n = 1
            limits[srv] = max(limits.get(srv, 1), n)
        return limits

    def _break_ms(self, whitespace: str, verse: bool = False) -> int:
        """Silence owed for a paragraph break in *whitespace* (0 if none)."""
        if _re.search(r"\n\s*\n", whitespace):
            return self.stanza_ms if verse else self.paragraph_ms
        return 0

    def _paced_units(self, text: str, tags: bool = False) -> list[tuple[int, str]]:
        """Split *text* into (silence_before_ms, piece) at paragraph breaks
        and at end-stopped verse lines (ending . ! ? ; : or a dash); lines
        that run on, or end on a comma, stay joined so the phrase flows."""
        units: list[tuple[int, str]] = []
        for p, para in enumerate(_re.split(r"\n\s*\n", text)):
            verse = VERSE_BREAK in para
            gap = (self.stanza_ms if verse else self.paragraph_ms) if p else 0
            buf = ""
            lines = para.split(VERSE_BREAK)
            if verse and tags:
                marked = ""
                for i, line in enumerate(lines):
                    line = line.strip()
                    if i < len(lines) - 1 and _re.search(r"[.!?;:—–][”’\"')]*$", line):
                        line += " [pause]"
                    elif i < len(lines) - 1 and line.endswith(","):
                        line += " [short pause]"
                    marked = f"{marked} {line}".strip()
                if _re.search(r"\w", marked):
                    units.append((gap, marked))
                continue
            for i, line in enumerate(lines):
                buf = f"{buf} {line.strip()}".strip()
                end = i == len(lines) - 1
                if not end and not _re.search(r"[.!?;:—–][”’\"')]*$", buf):
                    continue
                if _re.search(r"\w", buf):
                    units.append((gap, buf))
                    gap = self.line_ms
                buf = ""
        return units or [(0, text)]

    def _assignment_for(self, seg) -> tuple[str, float]:
        if seg.kind == "quote":
            return self.assignments.get(
                (seg.speaker or "").lower(), self.default_quote
            )
        return self.assignments["narrator"]


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into <= max_chars chunks, preferring sentence boundaries."""
    import re

    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    # Break into sentence-ish pieces, then greedily pack up to max_chars.
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks, current = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            # A single very long sentence: hard-wrap on whitespace.
            if current:
                chunks.append(current); current = ""
            words, line = piece.split(), ""
            for w in words:
                if len(line) + len(w) + 1 > max_chars:
                    chunks.append(line); line = w
                else:
                    line = f"{line} {w}".strip()
            if line:
                current = line
        elif len(current) + len(piece) + 1 > max_chars:
            chunks.append(current); current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def _wav_bytes_to_pcm(data: bytes) -> bytes:
    if data[:4] != b"RIFF":
        raise RuntimeError(
            "Expected a WAV response from the Orpheus server but got something else "
            f"(first bytes: {data[:8]!r}). Set the server's response_format to wav."
        )
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        return wav_file.readframes(wav_file.getnframes())


def speech_level(pcm: bytes, rate: int) -> float | None:
    """Active-speech level of 16-bit PCM in dBFS: power mean of the 10 ms
    frames within 30 dB of the loudest (pauses and breaths don't count)."""
    import numpy as np

    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    n = max(1, rate // 100)
    if len(a) < n * 10:
        return None
    fr = a[: len(a) // n * n].reshape(-1, n)
    db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    act = db[db > max(db.max() - 30.0, -60.0)]
    if not len(act):
        return None
    return float(10 * np.log10(np.mean(10 ** (act / 10))))


def level_speech(pcm: bytes, rate: int, target_db: float, max_gain_db: float = 12.0,
                 ceiling_db: float = -1.0) -> bytes:
    """Scale *pcm* so its speech sits at *target_db*, never past
    *max_gain_db* either way and never peaking above *ceiling_db*."""
    import numpy as np

    level = speech_level(pcm, rate)
    if level is None:
        return pcm
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    gain_db = max(-max_gain_db, min(max_gain_db, target_db - level))
    y = a * 10 ** (gain_db / 20)
    y = limit_peaks(y, rate, 10 ** (ceiling_db / 20))
    return (np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def limit_peaks(y, rate: int, ceil: float, block_ms: float = 2.5,
                lookahead_blocks: int = 2, release_ms: float = 80.0):
    """Lookahead peak limiter: gain is lowered smoothly *before* a peak and
    recovers over *release_ms*, so loud lines stay under *ceil* without the
    waveshaping distortion a per-sample clipper adds (the first version used
    tanh on the samples and shouts sounded harsh at any volume)."""
    import numpy as np

    n = max(1, int(rate * block_ms / 1000))
    nb = (len(y) + n - 1) // n
    pad = np.zeros(nb * n, dtype=np.float32)
    pad[: len(y)] = np.abs(y)
    peaks = pad.reshape(nb, n).max(1)
    if peaks.max() <= ceil:
        return y
    need = np.minimum(1.0, ceil / np.maximum(peaks, 1e-9))
    # lookahead: a block's gain already accounts for the next blocks' peaks
    ahead = need.copy()
    for k in range(1, lookahead_blocks + 1):
        ahead[:-k] = np.minimum(ahead[:-k], need[k:])
    rel = np.exp(-block_ms / release_ms)
    g = np.empty_like(ahead)
    cur = 1.0
    for i, want in enumerate(ahead):            # instant attack, smooth release
        cur = want if want < cur else want + (cur - want) * rel
        g[i] = cur
    centers = (np.arange(nb) + 0.5) * n
    gains = np.interp(np.arange(len(y)), centers, g)
    out = y * gains
    return np.clip(out, -ceil, ceil)            # catches the few samples between blocks


def _apply_volume(pcm: bytes, volume: float) -> bytes:
    try:
        import audioop  # stdlib (present through Python 3.12)

        return audioop.mul(pcm, 2, volume)
    except Exception:  # pragma: no cover - audioop removed in 3.13
        return pcm
