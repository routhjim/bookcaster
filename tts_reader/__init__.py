"""tts_reader: convert text files into spoken-word MP3s with a local TTS engine."""

__version__ = "0.1.0"

from .engine import PiperEngine  # noqa: F401

__all__ = ["PiperEngine", "__version__"]
