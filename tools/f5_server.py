"""Minimal OpenAI-compatible TTS server for F5-TTS voice cloning.

Serves POST /v1/audio/speech (the same contract tts_reader's Orpheus engine
speaks) plus GET /voices. A "voice" is a pair of files in the registry
directory (default ~/.local/share/tts_reader/f5_voices):

    <name>.wav   ~5-15 s reference clip of the voice
    <name>.txt   exact transcript of that clip

Optional voices.json in the same directory adds metadata:
    {"<name>": {"gender": "Male", "description": "gravelly narrator"}}

Run:  .venv/bin/python server.py         (port 5010)
"""

from __future__ import annotations

import io
import json
import os
import wave
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# torchaudio's decoder (torchcodec) only ships CUDA builds; reference clips
# are plain WAVs, so route loading through soundfile instead.
def _sf_load(path, *args, **kwargs):
    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


torchaudio.load = _sf_load

VOICES_DIR = Path(
    os.environ.get("F5_VOICES_DIR")
    or Path.home() / ".local" / "share" / "tts_reader" / "f5_voices"
)
PORT = int(os.environ.get("F5_PORT", "5010"))

app = FastAPI(title="f5-tts-server")
_tts = None  # loaded lazily so /voices works before the model finishes loading


def get_tts():
    global _tts
    if _tts is None:
        from f5_tts.api import F5TTS

        _tts = F5TTS()
        # Measured 1.65x on gfx1151 with no recompile thrash across shapes;
        # first request pays a ~4-minute compile tax. F5_COMPILE=0 disables.
        if os.environ.get("F5_COMPILE", "1") == "1":
            os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
            print("compiling transformer (first request will be slow)...")
            _tts.ema_model.transformer = torch.compile(
                _tts.ema_model.transformer
            )
    return _tts


def list_voices() -> dict[str, dict]:
    """Registry scan. A voice is <name>.wav+<name>.txt; emotional variants
    of the same narrator are <name>.<emotion>.wav+.txt alongside it."""
    meta = {}
    meta_path = VOICES_DIR / "voices.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    voices = {}
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        if "." in wav.stem:  # emotion variant, collected below
            continue
        if not wav.with_suffix(".txt").exists():
            continue
        info = meta.get(wav.stem, {}) if isinstance(meta, dict) else {}
        emotions = sorted(
            v.stem.split(".", 1)[1] for v in VOICES_DIR.glob(f"{wav.stem}.*.wav")
            if v.with_suffix(".txt").exists()
        )
        voices[wav.stem] = {
            "gender": str(info.get("gender", "")),
            "description": str(info.get("description", "cloned voice")),
            "emotions": emotions,
        }
    return voices


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def voices():
    return [
        {"name": name, **info} for name, info in list_voices().items()
    ]


class SpeechRequest(BaseModel):
    input: str
    voice: str = ""
    model: str = "f5"
    speed: float = 1.0
    response_format: str = "wav"
    # Emotional register: picks <voice>.<emotion>.wav as the reference clip
    # when that variant exists; silently falls back to the neutral base.
    emotion: str = ""


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    registry = list_voices()
    if not registry:
        raise HTTPException(503, f"no voices in {VOICES_DIR} — add "
                                 "<name>.wav + <name>.txt pairs")
    name = req.voice if req.voice in registry else next(iter(registry))
    ref_wav = VOICES_DIR / f"{name}.wav"
    if req.emotion:
        variant = VOICES_DIR / f"{name}.{req.emotion}.wav"
        if variant.exists() and variant.with_suffix(".txt").exists():
            ref_wav = variant
    ref_text = ref_wav.with_suffix(".txt").read_text(encoding="utf-8").strip()

    wav_data, sr, _ = get_tts().infer(
        ref_file=str(ref_wav), ref_text=ref_text, gen_text=req.input,
        speed=max(0.5, min(2.0, req.speed)), remove_silence=False,
    )
    pcm = np.clip(np.asarray(wav_data) * 32767, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    out = _trim_ghosts(buf.getvalue())
    out = _apply_register_tempo(out, name, req.emotion)
    out = _cap_pauses(out, name)
    return Response(content=out, media_type="audio/wav")


def _trim_ghosts(wav_bytes: bytes) -> bytes:
    """Cut barely-audible 'ghost' speech from the clip edges.

    F5 pads very short inputs with quiet mumble before/after the real
    words. Real speech sits near the clip's own level; ghosts sit far
    below it — trim edge audio under 12% of the speech level (plus a
    150 ms guard) back to the first/last strong onset.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    frame = max(sr // 50, 1)  # 20 ms
    n = len(pcm) // frame
    if n < 10:
        return wav_bytes
    rms = np.sqrt(np.mean(
        (pcm[: n * frame].astype(np.float32) / 32768).reshape(n, frame) ** 2,
        axis=1) + 1e-12)
    speech_level = np.percentile(rms, 95)
    strong = rms > speech_level * 0.12
    idx = np.where(strong)[0]
    if len(idx) == 0:
        return wav_bytes
    guard = int(0.15 * sr / frame)  # keep a natural onset/decay
    start = max(idx[0] - guard, 0) * frame
    end = min(idx[-1] + 1 + guard, n) * frame
    out_pcm = pcm[start:end]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_pcm.astype("<i2").tobytes())
    return buf.getvalue()


def _cap_pauses(wav_bytes: bytes, voice: str) -> bytes:
    """Clamp pause lengths into [min_pause_ms, max_pause_ms] (voices.json).

    F5 clones the reference clip's pause structure, so output pauses are as
    erratic as the ref's. Long silences are compacted to the cap; genuine
    sentence gaps (>= 250 ms) shorter than the floor are padded up to it.
    """
    try:
        meta = json.loads((VOICES_DIR / "voices.json").read_text(encoding="utf-8"))
        cap_ms = int(meta.get(voice, {}).get("max_pause_ms", 0))
        floor_ms = int(meta.get(voice, {}).get("min_pause_ms", 0))
    except (OSError, ValueError):
        return wav_bytes
    if cap_ms <= 0 and floor_ms <= 0:
        return wav_bytes
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    frame = max(sr // 50, 1)  # 20 ms
    n = len(pcm) // frame
    if n < 3:
        return wav_bytes
    rms = np.sqrt(np.mean(
        (pcm[: n * frame].astype(np.float32) / 32768).reshape(n, frame) ** 2,
        axis=1) + 1e-12)
    silent = rms < max(np.percentile(rms, 90) * 0.1, 1e-4)
    cap = max(int(cap_ms / 20), 1) if cap_ms > 0 else 10**9
    floor = int(floor_ms / 20)
    gap_min = int(250 / 20)  # runs this long count as sentence gaps

    pieces = []
    i = 0
    frames = pcm[: n * frame].reshape(n, frame)
    while i < n:
        if not silent[i]:
            pieces.append(frames[i])
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        run = j - i
        out_run = min(run, cap)
        if run >= gap_min and out_run < floor:
            out_run = floor
        pieces.extend(frames[i:min(i + min(run, out_run), n)])
        extra = out_run - min(run, out_run)
        if extra > 0:
            pieces.extend([np.zeros(frame, dtype="<i2")] * extra)
        i = j
    out_pcm = np.concatenate(pieces + [pcm[n * frame:]])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_pcm.astype("<i2").tobytes())
    return buf.getvalue()


def _apply_register_tempo(wav_bytes: bytes, voice: str, emotion: str) -> bytes:
    """Per-voice/per-register pitch-preserving time stretch.

    voices.json may carry {"<voice>": {"emotion_speed": {"neutral": 0.85,
    "sad": 0.8, ...}}}; factors < 1 slow the delivery. User-editable.
    """
    import subprocess

    try:
        meta = json.loads((VOICES_DIR / "voices.json").read_text(encoding="utf-8"))
        factor = float(
            meta.get(voice, {}).get("emotion_speed", {}).get(emotion or "neutral", 1.0)
        )
    except (OSError, ValueError):
        return wav_bytes
    if abs(factor - 1.0) < 0.01:
        return wav_bytes
    factor = max(0.5, min(2.0, factor))
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
         "-af", f"atempo={factor}", "-f", "wav", "pipe:1"],
        input=wav_bytes, capture_output=True,
    )
    return proc.stdout if proc.returncode == 0 and proc.stdout else wav_bytes


if __name__ == "__main__":
    import uvicorn

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"voices dir: {VOICES_DIR} ({len(list_voices())} voice(s))")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
