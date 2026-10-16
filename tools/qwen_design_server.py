"""Voice-design server: build a brand-new voice from a text description.

Wraps Qwen3-TTS VoiceDesign (Apache-2.0). POST a description of a voice plus
a line for it to speak; get back a WAV of that (fictional) voice saying the
line. tts_reader's casting session saves the clip + line as an F5 reference
pair, so every character gets an original voice that F5 then renders.

    POST /design  {"description": "...", "text": "...", "seed": 0}  -> audio/wav
    GET  /health

Run (pin it to a GPU; HIP order here: 0=XTX bus 05, 1=iGPU, 2=XTX bus 65):
    HIP_VISIBLE_DEVICES=2 ~/tts-ab/.venv-qwen/bin/python tools/qwen_design_server.py
"""

from __future__ import annotations

import io
import os
import sys
import threading
import wave

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

MODEL = os.environ.get("QWEN_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
PORT = int(os.environ.get("QWEN_DESIGN_PORT", "5012"))
# ROCm speed patches (MIOpen bypass + compiled/graphed decode, ~14x on an
# XTX). Without them, at least skip MIOpen: its conv fallback is ~12x slower.
FAST_DIR = os.path.expanduser(os.environ.get("QWEN_FAST_DIR", "~/tts-ab"))

app = FastAPI(title="qwen-voice-design")
_model = None
_lock = threading.Lock()  # one generation at a time; the GPU is saturated anyway


def load_model():
    global _model
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        MODEL, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    sys.path.insert(0, FAST_DIR)
    try:
        import qwen_fast

        qwen_fast.patch_all(model)
        print("qwen_fast speed patches applied")
    except ImportError:
        torch.backends.cudnn.enabled = False
        print(f"qwen_fast not found in {FAST_DIR}; MIOpen disabled only")
    _model = model


class DesignRequest(BaseModel):
    description: str
    text: str
    language: str = "English"
    seed: int = 0


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "model": MODEL}


@app.post("/design")
def design(req: DesignRequest):
    if _model is None:
        raise HTTPException(503, "model still loading")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    with _lock:
        # Same description + seed -> same voice, so a cast is reproducible
        # and "try another take" is just a new seed.
        torch.manual_seed(req.seed)
        wavs, sr = _model.generate_voice_design(
            text=req.text, instruct=req.description, language=req.language,
        )
    pcm = np.clip(np.asarray(wavs[0], dtype=np.float32) * 32767,
                  -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    print(f"loading {MODEL} on {torch.cuda.get_device_name(0)}...")
    load_model()
    # Pay the compile tax before serving, so the first real design is fast.
    design(DesignRequest(description="A calm adult voice.",
                         text="Warming up the voice designer."))
    print("ready")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
