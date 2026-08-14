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
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"voices dir: {VOICES_DIR} ({len(list_voices())} voice(s))")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
