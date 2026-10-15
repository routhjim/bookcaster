"""Fish Audio S2 Pro engine with continuous batching (runs in .venv-fish).

Several lines decode together (tools/fish_batch.py): the model is dense and
bandwidth-bound, so a step for 8 lines costs ~1.2x a step for one (7900 XTX:
1 line 1.8x real time, 4 lines 6.3x, 8 lines 12.2x in aggregate). A request
joins the running batch as soon as its prompt is prefilled and leaves when it
emits <|im_end|>; reference clips are encoded once per (file, mtime).

POST /generate  {"text", "ref_wav", "ref_text", "max_frames"?, "temperature"?,
                 "top_p"?, "context"?}  -> audio/wav at the codec's rate (44.1 kHz),
                 header X-Codes: an id for this take's codes
"context" is [{"text", "codes": <an X-Codes id>}, ...]: earlier takes put in
the prompt as finished turns (what fish's own generate_long does with the
previous chunks), so a long passage keeps one voice, pitch and pace instead
of each chunk drifting on its own.
GET  /health    {"status", "rows", "active"}

tools/fish_tts_server.py (the tool-facing adapter: voices, emotion tags,
Whisper check) calls this. Run, pinned to one XTX:
    HIP_VISIBLE_DEVICES=0 ~/tts-ab/.venv-fish/bin/python tools/fish_engine_server.py
"""

from __future__ import annotations

import io
import re
import os
import sys
import threading
import time
import wave
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")   # ROCm: error 301
# Compiled kernels persist across restarts (the default lives in /tmp).
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      str(Path.home() / ".cache" / "torchinductor-fish"))

import numpy as np
import torch

torch.backends.cudnn.enabled = False          # MIOpen convs are slow on this box

FISH_DIR = Path(os.environ.get("FISH_SPEECH_DIR", Path.home() / "tts-ab" / "fish-speech"))
sys.path.insert(0, str(FISH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fish_batch as fb  # noqa: E402

PORT = int(os.environ.get("FISH_ENGINE_PORT", "8891"))
ROWS = int(os.environ.get("FISH_ROWS", "8"))
SEQ = int(os.environ.get("FISH_SEQ", "2048"))
# Shapes compiled ahead: batch rows and attended-KV lengths (buckets).
ROW_BUCKETS = [b for b in (1, 4, 8) if b <= ROWS] or [ROWS]
if ROW_BUCKETS[-1] != ROWS:
    ROW_BUCKETS.append(ROWS)
KV_BUCKETS = [b for b in (512, 1024, 2048, 4096) if b < SEQ] + [SEQ]

_model = _codec = _step = None
_lock = threading.Condition()
_pending: list = []           # Job waiting for a row
_refs: dict = {}              # (path, mtime) -> codes (10, T) on GPU
_takes: "OrderedDict[str, torch.Tensor]" = OrderedDict()   # X-Codes id -> codes (10, N), CPU
TAKES_KEEP = int(os.environ.get("FISH_TAKES_KEEP", "4096"))


class Job:
    def __init__(self, text, ref_wav, ref_text, max_frames, temperature, top_p,
                 context=()):
        self.text, self.ref_wav, self.ref_text = text, ref_wav, ref_text
        self.context = list(context)      # [(text, codes (10, N))], oldest first
        self.max_frames, self.temperature, self.top_p = max_frames, temperature, top_p
        self.future: Future = Future()
        self.frames: list = []
        self.t0 = time.time()


def load():
    global _model, _codec, _step
    os.chdir(FISH_DIR)
    from fish_speech.models.dac.inference import load_model as load_codec
    from fish_speech.models.text2semantic.inference import init_model

    _model, _ = init_model("checkpoints/s2-pro", "cuda", torch.bfloat16, compile=False)
    if os.environ.get("FISH_INT8") == "1":
        print(f"int8 weights on {fb.quantize_int8(_model)} Linears", flush=True)
    _codec = load_codec("modded_dac_vq", "checkpoints/s2-pro/codec.pth", device="cuda")
    fb.setup(_model, rows=ROWS, seq_len=SEQ)
    _model._fast_caches = {}
    # One compiled graph per (rows, kv) bucket; dynamo's default limit (8)
    # would silently leave the rest eager (4096 KV made it 12 shapes).
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, len(ROW_BUCKETS) * len(KV_BUCKETS) + 4)
    _step = torch.compile(fb.step, mode=os.environ.get("FISH_COMPILE_MODE", "default"),
                          dynamic=False) if os.environ.get("FISH_COMPILE", "1") == "1" else fb.step


def _fast_caches_for(rows: int):
    """The fast transformer's KV caches sized for *rows* (its stock cache
    writes every row, so the batch must match the cache)."""
    from fish_speech.models.text2semantic.llama import KVCache

    cfg = _model.config
    caches = _model._fast_caches.get(rows)
    if caches is None:
        caches = [KVCache(rows, cfg.num_codebooks, cfg.fast_n_local_heads, cfg.fast_head_dim,
                          dtype=torch.bfloat16).cuda() for _ in _model.fast_layers]
        _model._fast_caches[rows] = caches
    for layer, c in zip(_model.fast_layers, caches):
        layer.attention.kv_cache = c


@torch.inference_mode()
def reference_codes(path: str) -> torch.Tensor:
    p = Path(path)
    key = (str(p), p.stat().st_mtime)
    codes = _refs.get(key)
    if codes is None:
        import soundfile as sf
        import torchaudio.functional as AF

        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(1)).cuda()
        rate = getattr(getattr(_codec, "spec_transform", None), "sample_rate", None) or _codec.sample_rate
        if sr != rate:
            wav = AF.resample(wav, sr, rate)
        audios = wav[None, None]
        codes = _codec.encode(audios, torch.tensor([audios.shape[2]], device="cuda"))[0][0]
        _refs[key] = codes
    return codes


def build_prompt(job: Job):
    """The same conversation fish's generate_long builds: system turn with
    the reference transcript and codes, the context takes as finished
    user/voice turns, the user's text, an open voice turn."""
    import re

    from fish_speech.conversation import Conversation, Message, TextPart, VQPart

    ref_text = job.ref_text
    if not re.search(r"<\|speaker:\d+\|>", ref_text):
        ref_text = f"<|speaker:0|>{ref_text}"
    codes = reference_codes(job.ref_wav).cpu()
    conv = Conversation()
    conv.append(Message(role="system", parts=[
        TextPart(text="convert the provided text to speech reference to the following:\n\nText:\n",
                 cal_loss=False),
        TextPart(text=ref_text, cal_loss=False),
        TextPart(text="\n\nSpeech:\n", cal_loss=False),
        VQPart(codes=codes, cal_loss=False)], cal_loss=False, add_im_start=True, add_im_end=True))
    for text, prior in job.context:
        conv.append(Message(role="user", parts=[TextPart(text=text, cal_loss=False)],
                            cal_loss=False, add_im_start=True, add_im_end=True))
        conv.append(Message(role="assistant", parts=[VQPart(codes=prior.long(), cal_loss=False)],
                            cal_loss=False, modality="voice", add_im_start=True, add_im_end=True))
    conv.append(Message(role="user", parts=[TextPart(text=job.text, cal_loss=False)],
                        cal_loss=False, add_im_start=True, add_im_end=True))
    conv.append(Message(role="assistant", parts=[], cal_loss=False, modality="voice",
                        add_im_start=True, add_im_end=False))
    enc, masks, parts = conv.encode_for_inference(_model.tokenizer,
                                                  num_codebooks=_model.config.num_codebooks)
    return enc.cuda(), masks, parts


def _expected_frames(job: Job) -> int:
    """Frames a take will likely need (generous: 1.7 words/s, where
    narration runs ~2.7), for deciding how much context fits."""
    words = len(re.sub(r"\[[^\]]*\]", " ", job.text).split())
    return min(job.max_frames, int(21.5 * (0.6 * words + 2)))


def _bucket(n: int, buckets: list[int]) -> int:
    return next(b for b in buckets if b >= n)


@torch.inference_mode()
def scheduler():
    """Admit -> step -> retire, forever. Active jobs occupy rows 0..n-1."""
    cfg = _model.config
    rows: list[Job] = []
    tok = torch.zeros(ROWS, 11, dtype=torch.long, device="cuda")
    pos = torch.zeros(ROWS, dtype=torch.long, device="cuda")
    prev = torch.full((ROWS, fb.RAS_WIN), -1, dtype=torch.long, device="cuda")
    temp = torch.ones(ROWS, 1, device="cuda")
    top_p = torch.ones(ROWS, device="cuda")
    host_pos = [0] * ROWS
    while True:
        with _lock:
            while not rows and not _pending:
                _lock.wait()
            admit = []
            while _pending and len(rows) + len(admit) < ROWS:
                admit.append(_pending.pop(0))
        for job in admit:
            try:
                enc, masks, parts = build_prompt(job)
                T = enc.shape[1]
                # Context must leave room for the take itself: drop the
                # oldest turns until ~ the frames it asked for fit.
                while job.context and T + _expected_frames(job) + 1 > SEQ:
                    job.context.pop(0)
                    enc, masks, parts = build_prompt(job)
                    T = enc.shape[1]
                if T + job.max_frames + 1 > SEQ:
                    job.max_frames = max(0, SEQ - T - 1)
                r = len(rows)
                if T > 1:
                    fb.prefill(_model, r, enc[:, :T - 1], masks, parts)
                tok[r] = enc[:, T - 1]
                pos[r] = T - 1
                host_pos[r] = T - 1
                prev[r] = -1
                temp[r] = job.temperature
                top_p[r] = job.top_p
                rows.append(job)
            except Exception as exc:   # bad reference etc.: fail this job only
                job.future.set_exception(exc)
        if not rows:
            continue
        n = len(rows)
        B = _bucket(n, ROW_BUCKETS)
        kv = _bucket(max(host_pos[:n]) + 1, KV_BUCKETS)
        _fast_caches_for(B)
        try:
            frame, main = _step(_model, tok[:B], pos[:B], kv, prev[:B], temp[:B], top_p[:B], 30)
        except torch.OutOfMemoryError as exc:
            # Fail the batch's jobs (the adapter retries them), keep serving:
            # an exception here used to end the scheduler thread for good.
            print(f"step out of memory with {n} rows: failing them", flush=True)
            for job in rows:
                job.future.set_exception(exc)
            rows.clear()
            pos.zero_()
            torch.cuda.empty_cache()
            continue
        frame, main = frame.clone(), main.clone()
        done_mask = (main[:n] == _model._im_end_row).cpu().tolist()
        tok[:B] = frame
        pos[:B] += 1
        pos[n:B] = 0          # padding rows: keep their (ignored) writes in range
        prev[:B] = torch.cat([prev[:B, 1:], main[:, None]], 1)
        finished = []
        for r in range(n):
            host_pos[r] += 1
            job = rows[r]
            if done_mask[r]:
                finished.append(r)
                continue
            job.frames.append(frame[r, 1:])
            if len(job.frames) >= job.max_frames or host_pos[r] + 1 >= SEQ:
                finished.append(r)
        for r in sorted(finished, reverse=True):
            _retire(rows, r, tok, pos, prev, temp, top_p, host_pos)


def _retire(rows, r, tok, pos, prev, temp, top_p, host_pos):
    """Finish row r's job and move the last active row into its slot."""
    job = rows[r]
    last = len(rows) - 1
    if r != last:
        for kc, vc in _model._row_caches:
            n = host_pos[last] + 1
            kc[r, :, :n] = kc[last, :, :n]
            vc[r, :, :n] = vc[last, :, :n]
        for t in (tok, pos, prev, temp, top_p):
            t[r] = t[last]
        host_pos[r] = host_pos[last]
        rows[r] = rows[last]
    rows.pop()
    pos[last] = 0
    try:
        try:
            audio = decode(job.frames)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            audio = decode(job.frames)
        if job.frames:
            job.codes = torch.stack(job.frames, 1).to(torch.int16).cpu()
        job.future.set_result(audio)
    except Exception as exc:
        job.future.set_exception(exc)


DECODE_CHUNK = int(os.environ.get("FISH_DECODE_CHUNK", "384"))   # frames per codec call
# The codec decoder is causal and needs ~200 frames of history to settle:
# 192 frames of left context = 44 dB SNR vs one-shot decoding (32 = 17 dB).
DECODE_CTX = 192
DECODE_LOOKAHEAD = 16
_hop = 0   # audio samples per frame, measured at warm-up


@torch.inference_mode()
def decode(frames: list) -> np.ndarray:
    """Codes -> waveform in overlapping chunks: a long paragraph decoded in
    one call needs GBs of activations beside the batch's KV caches (OOM on
    Speckled Band). Each chunk is decoded with DECODE_CTX frames of history
    (and a little lookahead) and cropped to its own span."""
    if not frames:
        return np.zeros(0, dtype=np.float32)
    codes = torch.stack(frames, 1)                          # (10, N)
    n = codes.shape[1]
    if n <= DECODE_CHUNK + DECODE_CTX or not _hop:
        return _codec.from_indices(codes[None])[0].squeeze().float().cpu().numpy()
    parts = []
    for s in range(0, n, DECODE_CHUNK):
        e = min(n, s + DECODE_CHUNK)
        a, b = max(0, s - DECODE_CTX), min(n, e + DECODE_LOOKAHEAD)
        wav = _codec.from_indices(codes[None, :, a:b])[0].squeeze().float()
        parts.append(wav[(s - a) * _hop:(s - a) * _hop + (e - s) * _hop].cpu())
    return torch.cat(parts).numpy()


def sample_rate() -> int:
    return int(_codec.sample_rate)


# --- HTTP --------------------------------------------------------------------

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="fish-engine")


class GenRequest(BaseModel):
    text: str
    ref_wav: str
    ref_text: str
    max_frames: int = 1024
    temperature: float = 0.8
    top_p: float = 0.8
    context: list[dict] = []      # [{"text", "codes": X-Codes id}]


@app.get("/health")
def health():
    return {"status": "ok" if _step is not None else "loading", "rows": ROWS,
            "pending": len(_pending)}


@app.post("/generate")
def generate(req: GenRequest):
    if _step is None:
        raise HTTPException(503, "model still loading")
    if not Path(req.ref_wav).exists():
        raise HTTPException(404, f"no such reference clip: {req.ref_wav}")
    context = []
    for turn in req.context:
        prior = _takes.get(str(turn.get("codes", "")))
        if prior is None:          # evicted or from before a restart: go without
            continue
        context.append((str(turn.get("text", "")), prior))
    job = Job(req.text, req.ref_wav, req.ref_text, req.max_frames, req.temperature, req.top_p,
              context)
    with _lock:
        _pending.append(job)
        _lock.notify()
    audio = job.future.result()
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate())
        wf.writeframes(pcm.tobytes())
    headers = {}
    if getattr(job, "codes", None) is not None:
        cid = uuid.uuid4().hex
        with _lock:
            _takes[cid] = job.codes
            while len(_takes) > TAKES_KEEP:
                _takes.popitem(last=False)
        headers["X-Codes"] = cid
    print(f"done {len(job.frames)} frames in {time.time() - job.t0:.1f}s"
          f"{f' (+{len(job.context)} context)' if job.context else ''}: {req.text[:60]!r}",
          flush=True)
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


def warm_up(ref_wav: str, ref_text: str):
    """Compile every (rows, kv) shape before serving (cached on disk after
    the first time, so a restart takes seconds, not minutes)."""
    with torch.inference_mode():
        for B in ROW_BUCKETS:
            _fast_caches_for(B)
            for kv in KV_BUCKETS:
                t = time.time()
                tok = torch.zeros(B, 11, dtype=torch.long, device="cuda")
                tok[:, 0] = _model.config.semantic_begin_id
                pos = torch.full((B,), kv - 1, dtype=torch.long, device="cuda")
                prev = torch.full((B, fb.RAS_WIN), -1, dtype=torch.long, device="cuda")
                _step(_model, tok, pos, kv, prev, torch.full((B, 1), 0.8, device="cuda"),
                      torch.full((B,), 0.8, device="cuda"), 30)
                torch.cuda.synchronize()
                print(f"warm rows={B} kv={kv}: {time.time() - t:.1f}s", flush=True)
    if ref_wav:
        reference_codes(ref_wav)
    global _hop
    with torch.inference_mode():
        probe = torch.zeros(_model.config.num_codebooks, 20, dtype=torch.long, device="cuda")
        _hop = _codec.from_indices(probe[None])[0].squeeze().shape[-1] // 20
    print(f"codec: {_hop} samples per frame", flush=True)


if __name__ == "__main__":
    import uvicorn

    print(f"loading Fish S2 Pro on {torch.cuda.get_device_name(0)} "
          f"(rows {ROWS}, seq {SEQ})...", flush=True)
    load()
    warm_up(os.environ.get("FISH_WARM_REF", ""), "")
    threading.Thread(target=scheduler, daemon=True).start()
    print("ready", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
