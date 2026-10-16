"""Fish Audio S2 Pro behind the F5/Qwen servers' HTTP contract.

Fish runs as its own service (tools/fish_engine_server.py, continuous
batching, unit tts-fish on :8891) in its own venv; this adapter gives it
the contract tts_reader speaks — POST /v1/audio/speech, GET /voices,
GET /health, the X-Verify header — so the tool renders through it with
``--engine f5 --f5-url http://127.0.0.1:5014/v1/audio/speech``.

What differs from the Qwen server:
* Voices come from the same registry (<name>.wav + <name>.txt), always the
  neutral clip: Fish takes emotion as an inline tag ([angry], [sad]...),
  so the request's "emotion" becomes a tag instead of a reference clip.
* Fish understands delivery tags such as [pause] and [short pause]; /health
  says so ("inline_tags"), and the tool's --verse-pacing tags writes them.
  Tags are stripped before the Whisper check.
* "context" (a list of {"text", "codes"}) puts earlier takes in Fish's
  prompt so a long passage keeps one delivery; each response's X-Codes
  header names its own take for the next request's context.
* Voice likeness is not checked (no speaker encoder here); Whisper checks
  skipped words, babble loops and runaway length, re-rendering with a new
  seed like the Qwen server.

Run in the Qwen venv (it reuses the Qwen server's registry + checks, and its
Whisper worker); it needs no GPU itself:
    HIP_VISIBLE_DEVICES= ~/tts-ab/.venv-qwen/bin/python -I tools/fish_tts_server.py
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import threading
import wave
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_tts_server as q  # noqa: E402  (registry, Whisper check, fragments)

FISH_URL = os.environ.get("FISH_URL", "http://127.0.0.1:8891")
PORT = int(os.environ.get("FISH_TTS_PORT", "5014"))
# Fish's codec runs at 44.1 kHz; serve that (the old 24 kHz downsample, a
# holdover from Qwen/Orpheus, cut everything above 12 kHz). The Whisper
# check still gets a 24 kHz copy (the Qwen checker's rate).
SAMPLE_RATE = int(os.environ.get("FISH_OUT_RATE", "44100"))
CHECK_RATE = q.SAMPLE_RATE


def _resample(audio: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, to)
    return resample_poly(audio, to // g, sr // g).astype(np.float32)
VERIFY = os.environ.get("FISH_VERIFY", "1") != "0"
VERIFY_RETRIES = int(os.environ.get("FISH_VERIFY_RETRIES", "3"))

# The tool's emotion registers as Fish tags (Fish also takes free-form tags).
EMOTION_TAGS = {
    "angry": "[angry]", "sad": "[sad]", "excited": "[excited]",
    "surprised": "[surprised]", "tense": "[tense]", "warm": "[warm]",
    "cold": "[cold]",
}
_TAG = re.compile(r"\[[^\]]*\]")

app = FastAPI(title="fish-tts")
def _reference(voice: str) -> dict:
    """The voice's clip (by path: the engine encodes it once and caches the
    codes) and transcript."""
    wav, txt = q.VOICES_DIR / f"{voice}.wav", q.VOICES_DIR / f"{voice}.txt"
    if not wav.exists() or not txt.exists():
        raise HTTPException(404, f"unknown voice '{voice}'")
    return {"ref_wav": str(wav), "ref_text": txt.read_text(encoding="utf-8").strip()}


def _fish(text: str, ref: dict, seed: int, temperature: float = 0.0,
          context: list | None = None) -> tuple[np.ndarray, str]:
    # Cap a take like the Qwen server (~1.6 s/word + 4 s at 21.5 frames/s):
    # a babble loop otherwise runs to the frame limit before a retry.
    words = len(_TAG.sub(" ", text).split())
    body = {"text": text, **ref, "max_frames": min(2000, int(21.5 * (1.6 * words + 4)))}
    if temperature > 0:
        body["temperature"] = min(1.0, temperature)
    if context:
        body["context"] = context
    req = Request(f"{FISH_URL}/generate", json.dumps(body).encode(),
                  {"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=900) as resp:
            data = resp.read()
            codes = resp.headers.get("X-Codes", "")
    except (URLError, OSError) as exc:
        raise HTTPException(502, f"Fish server at {FISH_URL}: {exc}") from exc
    with wave.open(io.BytesIO(data)) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        audio = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    audio = _resample(audio, sr, SAMPLE_RATE)
    return audio, codes


def _render(text: str, ref: dict, label: str, tries: int, verify: bool,
            temperature: float = 0.0, context: list | None = None):
    """Best of up to *tries* takes: (audio, report or None, codes id)."""
    plain = re.sub(r"\s+", " ", _TAG.sub(" ", text)).strip()
    best, report = None, None
    for attempt in range(tries):
        try:
            take, codes = _fish(text, ref, seed=7 if attempt == 0 else random.randrange(2**31),
                                temperature=temperature, context=context)
        except HTTPException as exc:
            # An engine error (e.g. out of memory) costs this take, not the
            # book: try again, and fail only if every take errors.
            print(f"engine error [{label}] take {attempt + 1}: {exc.detail}", flush=True)
            if attempt == tries - 1 and best is None:
                raise
            continue
        if not verify:
            return take, None, codes
        try:
            check = q.check_take(_resample(take, SAMPLE_RATE, CHECK_RATE), plain, None)
        except Exception as exc:
            # The checker failing must not fail the line, but it mostly fails
            # on a broken take (empty, no pitch): rank it last and try again.
            print(f"verify [{label}] take {attempt + 1} uncheckable: {exc}", flush=True)
            check = {"ok": False, "problems": ["uncheckable"], "wer": 9.0,
                     "likeness": None, "heard": ""}
        check["takes"] = attempt + 1
        rank = (len(check["problems"]), check["wer"])
        if best is None or rank < best[0]:
            best, report = (rank, take, codes), check
        if check["ok"]:
            break
        print(f"verify [{label}] take {attempt + 1}: {', '.join(check['problems'])}"
              f" | heard: {check['heard'][:120]!r}", flush=True)
    return best[1], report, best[2]


def shift_register(audio: np.ndarray, semitones: float, range_mul: float,
                   gender: str) -> np.ndarray:
    """Move the voice's pitch median by *semitones* and scale its intonation
    range by *range_mul* (Praat "Change gender" with no formant shift, so
    the vocal tract — who it is — stays put). Unchanged if Praat can't."""
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return audio
    lo, hi = (100, 500) if gender.upper().startswith("F") else (60, 300)
    snd = parselmouth.Sound(audio.astype(np.float64), SAMPLE_RATE)
    try:
        median = call(snd.to_pitch(pitch_floor=lo, pitch_ceiling=hi),
                      "Get quantile", 0, 0, 0.5, "Hertz")
        if not median or median != median:
            return audio
        out = call(snd, "Change gender", lo, hi, 1.0,
                   median * 2 ** (semitones / 12), range_mul, 1.0)
    except parselmouth.PraatError:
        return audio
    if int(out.sampling_frequency) != SAMPLE_RATE:
        return audio
    y = out.values[0].astype(np.float32)
    return y[: len(audio)] if len(y) >= len(audio) else np.pad(y, (0, len(audio) - len(y)))


# Voice profiles of recent takes, by X-Codes id: a later request can ask to
# sound like one ("match"), which steadies a narrator across chunks.
_profiles: dict = {}
PROFILES_KEEP = 4096
MATCH_MAX_ST = 2.0      # pitch nudge cap, semitones
MATCH_MAX_DB = 4.0      # EQ cap per band
_BANDS = np.geomspace(80, 16000, 25)    # ~third-octave-ish band edges


def _ltas(audio: np.ndarray) -> np.ndarray:
    """Long-term spectrum of the speech (dB per band), quiet frames left out."""
    from scipy.signal import stft

    f, _, z = stft(audio, SAMPLE_RATE, nperseg=2048)
    p = np.abs(z) ** 2
    e = p.sum(0)
    p = p[:, e > e.max() * 1e-3]
    out = []
    for lo, hi in zip(_BANDS[:-1], _BANDS[1:]):
        m = (f >= lo) & (f < hi)
        out.append(10 * np.log10(p[m].mean() + 1e-12))
    return np.array(out)


def _f0_median(audio: np.ndarray, gender: str) -> float | None:
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return None
    lo, hi = (100, 500) if gender.upper().startswith("F") else (60, 300)
    try:
        snd = parselmouth.Sound(audio.astype(np.float64), SAMPLE_RATE)
        med = call(snd.to_pitch(pitch_floor=lo, pitch_ceiling=hi), "Get quantile", 0, 0, 0.5, "Hertz")
    except Exception:
        return None
    return float(med) if med and med == med else None


def _profile(audio: np.ndarray, gender: str) -> dict:
    return {"f0": _f0_median(audio, gender), "ltas": _ltas(audio)}


def match_profile(audio: np.ndarray, target: dict, gender: str) -> np.ndarray:
    """Nudge *audio* toward *target*'s pitch median and long-term spectrum
    (each capped), so separately sampled chunks of one passage sound like
    one recording. Formants are kept by the pitch step."""
    own = _profile(audio, gender)
    if own["f0"] and target.get("f0"):
        st = float(np.clip(12 * np.log2(target["f0"] / own["f0"]), -MATCH_MAX_ST, MATCH_MAX_ST))
        if abs(st) > 0.15:
            audio = shift_register(audio, st, 1.0, gender)
    diff = np.clip(target["ltas"] - own["ltas"], -MATCH_MAX_DB, MATCH_MAX_DB)
    diff -= np.median(diff)                 # shape only; level is the tool's job
    if np.abs(diff).max() > 0.5:
        from scipy.signal import firwin2, fftconvolve

        centers = np.sqrt(_BANDS[:-1] * _BANDS[1:])
        freq = np.concatenate([[0], centers, [SAMPLE_RATE / 2]]) / (SAMPLE_RATE / 2)
        gain = 10 ** (np.concatenate([[diff[0]], diff, [diff[-1]]]) / 20)
        fir = firwin2(1025, freq, gain)
        audio = fftconvolve(audio, fir, mode="same").astype(np.float32)
    return audio


class SpeechRequest(BaseModel):
    input: str
    voice: str = ""
    model: str = "fish"
    speed: float = 1.0          # accepted for compatibility; not applied
    response_format: str = "wav"
    emotion: str = ""           # becomes an inline Fish tag
    verify: bool = True
    tries: int = 0              # 0 = 1 + FISH_VERIFY_RETRIES; 1 = screening
    temperature: float = 0.0    # sampling temperature (0 = engine default 0.8)
    pitch: float = 0.0          # register shift, semitones (formants kept)
    pitch_range: float = 1.0    # intonation range multiplier
    context: list[dict] = []    # earlier takes [{"text", "codes": X-Codes}]
    match: str = ""             # an X-Codes id whose pitch + spectrum to match


@app.get("/health")
def health():
    try:
        with urlopen(f"{FISH_URL}/health", timeout=5) as resp:
            info = json.loads(resp.read())
    except (URLError, OSError, ValueError):
        info = {}
    # max_batch: how many requests the tool should keep in flight (the
    # engine decodes that many lines at once).
    return {"status": info.get("status", "loading"), "model": "fish-s2-pro",
            "inline_tags": True, "max_batch": int(info.get("rows", 1)), "context": True,
            "sample_rate": SAMPLE_RATE}


@app.get("/voices")
def voices():
    return [{"name": k, **v} for k, v in q.list_voices().items()]


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    text = req.input.strip()
    if not _TAG.sub("", text).strip():
        raise HTTPException(400, "input is empty")
    voice = req.voice or next(iter(q.list_voices()), "")
    # Fragment cleanup applies to the words, not to leading [tags].
    lead = re.match(r"(\s*(?:\[[^\]]*\]\s*)*)", text).group(1)
    text = lead + q._close_fragment(text[len(lead):])
    tag = EMOTION_TAGS.get(req.emotion, f"[{req.emotion}]" if req.emotion else "")
    if tag:
        text = f"{tag} {text}"
    verify = VERIFY and req.verify
    tries = req.tries if req.tries > 0 else 1 + (VERIFY_RETRIES if verify else 0)
    # No lock: concurrent requests decode together in the engine's batch.
    audio, report, codes = _render(text, _reference(voice), voice, tries, verify,
                                   req.temperature, req.context)
    if report and report["takes"] > 1:
        print(f"verify [{voice}] kept a take after {report['takes']} "
              f"({'ok' if report['ok'] else 'best effort: ' + ', '.join(report['problems'])})",
              flush=True)
    gender = str(q.voice_meta(voice).get("gender", ""))
    if codes:
        _profiles[codes] = _profile(audio, gender)
        while len(_profiles) > PROFILES_KEEP:
            _profiles.pop(next(iter(_profiles)))
    if req.match and req.match in _profiles and req.match != codes:
        audio = match_profile(audio, _profiles[req.match], gender)
    if req.pitch or req.pitch_range != 1.0:
        audio = shift_register(audio, req.pitch, req.pitch_range,
                               str(q.voice_meta(voice).get("gender", "")))
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    headers = {"X-Codes": codes} if codes else {}
    if report:
        headers["X-Verify"] = json.dumps({k: report[k] for k in
                                          ("ok", "problems", "wer", "likeness", "takes")})
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn

    print(f"fish adapter -> {FISH_URL}, serving :{PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
