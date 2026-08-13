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
    ):
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
        return ORPHEUS_SAMPLE_RATE

    def _origin(self) -> str:
        p = urlparse(self.base_url)
        return f"{p.scheme}://{p.netloc}"

    def load(self, log=print) -> None:
        """Check that the Orpheus server is reachable, with a helpful error."""
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

    def _request_pcm(self, text: str, speed: float) -> bytes:
        payload = json.dumps({
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "wav",
            "speed": speed,
        }).encode("utf-8")
        req = Request(
            self.base_url, data=payload,
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                audio = resp.read()
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

    def synthesize_pcm(
        self, text: str, speed: float = 1.0, volume: float = 1.0, log=print
    ) -> bytes:
        chunks = _chunk_text(text, self.chunk_chars) if self.chunk_chars > 0 else [text]
        pcm = b"".join(self._request_pcm(c, speed) for c in chunks if c.strip())
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
    ):
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

    def synthesize_pcm(
        self, text: str, speed: float = 1.0, volume: float = 1.0, log=print
    ) -> bytes:
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
        # A short beat when the voice changes keeps switches from feeling
        # abrupt. 16-bit mono silence at the shared sample rate.
        pause = b"\x00\x00" * int(self.sample_rate * self.pause_ms / 1000)
        pcm = bytearray()
        prev_assignment = None
        for seg in segments:
            # Narration tags arrive as ', said Ahab.' — drop the leading comma.
            spoken = _re.sub(r"^[\s,;:—\-]+", "", seg.text).strip()
            if not _re.search(r"\w", spoken):
                continue
            assignment = self._assignment_for(seg)
            if prev_assignment is not None and assignment != prev_assignment:
                pcm += pause
            if seg.kind == "quote" and seg.emotion:
                spoken = f"<{seg.emotion}> {spoken}"
            base, rate = assignment
            pcm += self.engines[base].synthesize_pcm(
                spoken, speed=speed * rate, volume=volume, log=None
            )
            prev_assignment = assignment
        return bytes(pcm)

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


def _apply_volume(pcm: bytes, volume: float) -> bytes:
    try:
        import audioop  # stdlib (present through Python 3.12)

        return audioop.mul(pcm, 2, volume)
    except Exception:  # pragma: no cover - audioop removed in 3.13
        return pcm
