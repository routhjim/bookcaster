"""Designed voices: original character voices built from text descriptions.

A voice-design server (``tools/qwen_design_server.py``, Qwen3-TTS
VoiceDesign) turns a description like "a wiry old sailor, high and nasal,
fast and excitable" into a short clip of that voice speaking an audition
line. The clip + line are saved as an F5 reference pair, so the F5 server
can render the whole book in that voice. No real person's voice is involved.

Designed voices are named ``d_<book>_<character>`` so each book's cast stays
separate in the shared F5 registry.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_DESIGN_URL = "http://127.0.0.1:5012"
DEFAULT_EMBED_URL = "http://127.0.0.1:5013"   # tools/qwen_tts_server.py
PREFIX = "d_"

# Emotional registers for a designed voice, matching the F5 palette names
# (speakers.F5_EMOTION_REGISTERS). Each: how to deliver it (appended to the
# voice description) and an audition line whose words carry the register —
# F5 copies the reference clip's delivery, so the clip must BE the emotion.
REGISTERS = {
    "warm": ("Speaking warmly and gently, with real affection.",
             "Come here and sit beside me. I have missed you terribly, and it does "
             "my heart good to see you safe and smiling again."),
    "tense": ("Speaking in a tense, anxious, hushed voice, barely holding steady.",
              "Did you hear that? Something moved outside the door. Keep very "
              "still, and whatever happens, do not make a sound."),
    "angry": ("Speaking angrily, sharp, forceful and raised.",
              "How dare you come into my house and speak to me like that! Get out, "
              "before I lose what little patience I have left!"),
    "sad": ("Speaking sadly, subdued and sorrowful, close to tears.",
            "She is gone, and nothing will ever bring her back. I keep setting a "
            "place for her at the table, as if she might still come home."),
    "excited": ("Speaking excitedly, eager, bright and animated.",
                "We found it! After all these months, we actually found it! Come "
                "quickly, you simply have to see this for yourself!"),
    "cold": ("Speaking coldly, flat, clipped and detached, without warmth.",
             "Your feelings are of no interest to me. You will do exactly as you "
             "are told, and we will not discuss this again."),
    "surprised": ("Speaking in sudden astonishment, startled and incredulous.",
                  "What? You cannot be serious! I never would have believed it, "
                  "not in a hundred years. Are you absolutely certain?"),
}
# The neutral (base) clip gets a delivery instruction too: without one the
# designer acts out the whole character sketch ("rapid, impatient...") and
# the plain voice comes out more wound-up than its excited register.
NEUTRAL_DELIVERY = ("Speaking in a calm, relaxed, natural conversational tone, "
                    "at an even, unhurried pace.")


def neutral_description(description: str) -> str:
    return f"{description.rstrip()} {NEUTRAL_DELIVERY}"


# F5 over-copies pace: a fast clip makes the line sound rushed, not
# emotional. Max speaking rate vs the voice's neutral clip (words/second).
_PACE_LIMIT = {"angry": 1.30, "excited": 1.30, "surprised": 1.30}
_PACE_LIMIT_CALM = 1.12
# Raw x-vectors all score 0.93-0.99 against each other (different people
# included), so takes are compared after subtracting the registry's mean
# voice; even then same-person and different-person ranges overlap, so the
# score only RANKS takes (keep the one most like the base), never gates.


def f5_voices_dir() -> Path:
    """The F5 server's voice registry (same default as tools/f5_server.py)."""
    env = os.environ.get("F5_VOICES_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".local" / "share" / "tts_reader" / "f5_voices"


def _slug(text: str, maxlen: int) -> str:
    # F5 treats dots in a stem as emotion variants, and voice maps split on
    # ',', '=' and '@' — keep names to [a-z0-9_].
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text[:maxlen].strip("_") or "x"


def book_slug(spec: str) -> str:
    stem = Path(urlparse(spec).path).stem if "://" in spec else Path(spec).stem
    return _slug(stem, 24)


def voice_name(book: str, character: str) -> str:
    return f"{PREFIX}{book}_{_slug(character, 24)}"


def belongs_to(voice: str, book: str) -> bool:
    """True for library voices and this book's designed voices."""
    return not voice.startswith(PREFIX) or voice.startswith(f"{PREFIX}{book}_")


def server_ready(url: str) -> bool:
    try:
        with urlopen(f"{url.rstrip('/')}/health", timeout=5) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except (OSError, ValueError):
        return False


def design_clip(url: str, description: str, text: str, seed: int = 0) -> bytes:
    """Ask the design server for a WAV of *description* speaking *text*."""
    payload = json.dumps(
        {"description": description, "text": text, "seed": seed}
    ).encode("utf-8")
    req = Request(f"{url.rstrip('/')}/design", data=payload,
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=600) as resp:
            return resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"design server HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"cannot reach the voice-design server at {url} ({exc.reason}). "
            "Start tools/qwen_design_server.py first."
        ) from exc


def wav_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def save_reference(name: str, wav_bytes: bytes, line: str,
                   voices_dir: Path | None = None) -> Path:
    """Write <name>.wav + <name>.txt into the F5 registry."""
    d = voices_dir or f5_voices_dir()
    d.mkdir(parents=True, exist_ok=True)
    wav_path = d / f"{name}.wav"
    wav_path.write_bytes(wav_bytes)
    wav_path.with_suffix(".txt").write_text(line.strip() + "\n", encoding="utf-8")
    return wav_path


def f5_preview(f5_url: str, voice: str, text: str) -> bytes:
    """Render *text* through the F5 server — how the book will really sound."""
    payload = json.dumps({"input": text, "voice": voice, "model": "f5",
                          "response_format": "wav"}).encode("utf-8")
    req = Request(f5_url, data=payload,
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=300) as resp:
        return resp.read()


def save_audition(wav_bytes: bytes, folder: Path, stem: str) -> Path:
    """Write *wav_bytes* as <folder>/<stem>.mp3 — a file the user can open
    anywhere (phone, file browser), unlike speaker playback."""
    import tempfile

    from .audio import wav_to_mp3

    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{stem}.mp3"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
    try:
        wav_to_mp3(tmp.name, out)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return out


def file_stem(character: str) -> str:
    """Readable, filesystem-safe name for audition files ("Helen_Stoner")."""
    return re.sub(r"[^A-Za-z0-9]+", "_", character).strip("_") or "voice"


def words_per_second(wav_bytes: bytes, text: str) -> float:
    return len(text.split()) / max(0.1, wav_seconds(wav_bytes))


def speaker_embedding(path: Path, url: str = DEFAULT_EMBED_URL):
    """x-vector of a clip from the Qwen TTS server, or None if it's not up."""
    req = Request(f"{url.rstrip('/')}/embed",
                  data=json.dumps({"path": str(path)}).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["embedding"]
    except (OSError, ValueError, KeyError):
        return None


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def set_register_tempo(name: str, emotion: str, factor) -> None:
    """Record (or clear, factor=None) a per-register tempo for *name*."""
    path = f5_voices_dir() / "voices.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    speeds = meta.setdefault(name, {}).setdefault("emotion_speed", {})
    if factor is None:
        speeds.pop(emotion, None)
    else:
        speeds[emotion] = factor
    if not speeds:
        meta[name].pop("emotion_speed")
    if not meta[name]:
        meta.pop(name)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def voice_meta(name: str) -> dict:
    """voices.json entry for *name* ({} when none)."""
    try:
        meta = json.loads((f5_voices_dir() / "voices.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entry = meta.get(name) if isinstance(meta, dict) else None
    return entry if isinstance(entry, dict) else {}


# -- accents ----------------------------------------------------------------
# VoiceDesign can't do accents (every designed clip classifies as US). What
# works: the Qwen Base server clones the *delivery* of a real accented
# speaker (ref codes + transcript) with the designed voice's *timbre*
# (speaker embedding). The donors are VCTK speakers picked by ear, listed in
# accent_donors/donors.json as {"scottish": {"M": [...], "F": [...]}, ...}.
# F5 loses the accent when cloning such a clip, so accented voices render
# through the Qwen server.
def accent_donors_dir() -> Path:
    env = os.environ.get("TTS_ACCENT_DONORS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".local" / "share" / "tts_reader" / "accent_donors"


def accent_bank() -> dict:
    try:
        data = json.loads((accent_donors_dir() / "donors.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}


def available_accents() -> list[str]:
    return sorted(accent_bank())


def detect_accent(text: str) -> str:
    """An accent we can do, if the description names one ("a soft Irish lilt")."""
    for accent in available_accents():
        if re.search(rf"\b{re.escape(accent)}\b", text or "", re.IGNORECASE):
            return accent
    return ""


def accent_donor(accent: str, gender: str) -> str:
    """Donor clip name for *accent* in the voice's gender ("" if none)."""
    by_gender = accent_bank().get(accent.lower(), {})
    donors = by_gender.get(gender or "M") or by_gender.get("M") or by_gender.get("F") or []
    return donors[0] if donors else ""


def set_voice_accent(name: str, accent: str, gender: str) -> str:
    """Record (accent="" clears) the accent of voice *name*; returns the donor."""
    path = f5_voices_dir() / "voices.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    entry = meta.setdefault(name, {})
    donor = accent_donor(accent, gender) if accent else ""
    if donor:
        entry.update(accent=accent.lower(), accent_donor=donor)
        if gender in ("M", "F"):
            entry.setdefault("gender", gender)   # pitch range for emotion shaping
    else:
        entry.pop("accent", None)
        entry.pop("accent_donor", None)
    if not entry:
        meta.pop(name)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return donor


def is_accented(voice: str) -> bool:
    return bool(voice_meta(voice.split("@", 1)[0]).get("accent_donor"))


def renders_on_qwen(voice: str) -> bool:
    """Voices the Qwen server renders: accented designed voices, and real
    recorded speakers (voices.json "render": "qwen", e.g. VCTK imports),
    whose accents F5 flattens."""
    meta = voice_meta(voice.split("@", 1)[0])
    return bool(meta.get("accent_donor")) or meta.get("render") == "qwen"


def _centering_mean(embed_url: str):
    """Mean x-vector of the library voices (cached beside them)."""
    d = f5_voices_dir()
    cache = d / ".xvec_mean.json"
    try:
        return json.loads(cache.read_text())
    except (OSError, ValueError):
        pass
    vecs = [v for v in (speaker_embedding(p, embed_url) for p in sorted(d.glob("*.wav"))
                        if "." not in p.stem and not p.stem.startswith(PREFIX)) if v]
    if len(vecs) < 4:
        return None
    mean = [sum(col) / len(vecs) for col in zip(*vecs)]
    cache.write_text(json.dumps(mean))
    return mean


def design_registers(url: str, name: str, description: str, seed: int,
                     base_wav: bytes, base_line: str, embed_url: str = DEFAULT_EMBED_URL,
                     tries: int = 3, log=print) -> dict:
    """Build <name>.<emotion>.wav/.txt variants of a designed voice.

    Each register is the same description + seed with a delivery instruction
    and an emotional line. A take is kept only if its pace stays near the
    neutral clip's and (when the Qwen server is up) it still sounds like the
    same person; otherwise another seed is tried. Returns {emotion: note}.
    """
    import tempfile

    d = f5_voices_dir()
    base_wps = words_per_second(base_wav, base_line)
    mean = _centering_mean(embed_url)
    center = (lambda v: [x - m for x, m in zip(v, mean)]) if mean else (lambda v: v)
    base_emb = speaker_embedding(d / f"{name}.wav", embed_url)
    base_emb = center(base_emb) if base_emb else None
    report = {}
    for emotion, (how, line) in REGISTERS.items():
        limit = _PACE_LIMIT.get(emotion, _PACE_LIMIT_CALM)
        best = None
        # Without the Qwen server there is nothing to rank by: one take.
        for k in range(tries if base_emb is not None else 1):
            wav = design_clip(url, f"{description} {how}", line, seed=seed * 100 + k)
            pace = words_per_second(wav, line) / base_wps
            sim = None
            if base_emb is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav)
                emb = speaker_embedding(Path(f.name), embed_url)
                Path(f.name).unlink(missing_ok=True)
                sim = cosine(base_emb, center(emb)) if emb is not None else None
            ok = pace <= limit
            # Pace is a hard rule; among acceptable takes, most like the base.
            score = (ok, sim if sim is not None else 0.0, -pace)
            if best is None or score > best[0]:
                best = (score, wav, pace, sim)
        (ok, _, _), wav, pace, sim = best
        save_reference(f"{name}.{emotion}", wav, line)
        # Too fast in every take: the F5 server slows that register back
        # down (voices.json emotion_speed = pitch-preserving atempo factor).
        set_register_tempo(name, emotion, round(limit / pace, 2) if pace > limit else None)
        report[emotion] = (f"pace x{pace:.2f}" + (f", likeness {sim:.2f}" if sim is not None else "")
                           + ("" if ok else " (too fast in all %d takes, below target)" % tries))
    return report


def play(path: Path) -> None:
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                        str(path)], check=False)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(f"(ffplay not found — the clip is at {path})")
