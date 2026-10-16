"""Qwen3-TTS voice-cloning server with the F5 server's HTTP contract.

Drop-in for tools/f5_server.py: same POST /v1/audio/speech + GET /voices,
same voice registry (<name>.wav + <name>.txt in the F5 voices directory), so
tts_reader renders through it with ``--engine f5 --f5-url <this server>``.
Qwen3-TTS-12Hz Base (Apache-2.0) clones from the reference clip + transcript.

Run (pin a GPU; HIP order here: 0=XTX bus 05, 1=iGPU, 2=XTX bus 65):
    HIP_VISIBLE_DEVICES=2 ~/tts-ab/.venv-qwen/bin/python -I tools/qwen_tts_server.py
"""

from __future__ import annotations

import dataclasses
import io
import json
import random
import re
import unicodedata
import os
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

MODEL = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
PORT = int(os.environ.get("QWEN_TTS_PORT", "5013"))
VOICES_DIR = Path(
    os.environ.get("F5_VOICES_DIR")
    or Path.home() / ".local" / "share" / "tts_reader" / "f5_voices"
).expanduser()
DONORS_DIR = Path(
    os.environ.get("TTS_ACCENT_DONORS_DIR")
    or Path.home() / ".local" / "share" / "tts_reader" / "accent_donors"
).expanduser()
FAST_DIR = os.path.expanduser(os.environ.get("QWEN_FAST_DIR", "~/tts-ab"))
SAMPLE_RATE = 24000
FRAME_RATE = 12.5   # codec frames per second (one generated token each)

# Emotion for voices without emotional reference clips — accented voices
# (an emotional clip would cancel the donor's accent) and real recorded
# speakers (VCTK imports have one neutral clip). Instead the neutral
# render is reshaped (Praat "Change gender" with no formant shift): pitch
# median x, pitch range x, duration x, gain dB. Chosen by ear over
# bootstrapped accented emotion clips (2026-10-12).
EMOTION_DSP = {
    "warm": (0.97, 1.25, 1.06, -1), "tense": (1.05, 0.70, 0.96, -3),
    "angry": (1.08, 1.45, 0.92, 4), "sad": (0.91, 0.60, 1.15, -5),
    "excited": (1.15, 1.60, 0.90, 3), "cold": (0.95, 0.45, 1.00, -1),
    "surprised": (1.20, 1.80, 0.95, 2),
}

# Per-line verification: Whisper transcribes every take and the speaker
# encoder checks it is still the right voice; a failed take (skipped words,
# babble loops, wrong voice) is re-rendered with a new seed and the best of
# up to 1 + VERIFY_RETRIES takes is returned. QWEN_VERIFY=0 disables.
VERIFY = os.environ.get("QWEN_VERIFY", "1") != "0"
VERIFY_RETRIES = int(os.environ.get("QWEN_VERIFY_RETRIES", "3"))
WHISPER_MODEL = os.environ.get("QWEN_VERIFY_WHISPER", "small.en")
# Centered speaker-embedding cosine: same voice renders score ~0.6-0.9, a
# different person ~0-0.3 (registry mean subtracted; raw cosines are useless).
VOICE_MIN = float(os.environ.get("QWEN_VERIFY_VOICE_MIN", "0.35"))

# How far the shaping moves from neutral (1.0 = the presets as auditioned;
# the user found that underwhelming over a whole chapter).
EMOTION_INTENSITY = float(os.environ.get("QWEN_EMOTION_INTENSITY", "1.5"))

app = FastAPI(title="qwen-tts")
_model = None
_lock = threading.Lock()   # one generation at a time
_prompts: dict = {}        # voice -> (wav mtime, clone prompt)


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

        # v3: graphed talker + code predictor, bucketed KV, hipBLASLt GEMVs
        # (~3.0x real time on a 7900 XTX); patch_all is the older fallback.
        (getattr(qwen_fast, "patch_v3", None) or qwen_fast.patch_all)(model)
        print("qwen_fast speed patches applied", flush=True)
    except ImportError:
        torch.backends.cudnn.enabled = False   # MIOpen convs are ~12x slower
        print(f"qwen_fast not found in {FAST_DIR}; MIOpen disabled only")
    _model = model


def _cached_prompt(key: str, wav: Path, txt: Path):
    """Clone prompt for a WAV + transcript, rebuilt when the clip changes
    (a redesigned voice overwrites its .wav)."""
    if not wav.exists() or not txt.exists():
        raise HTTPException(404, f"unknown voice '{key}'")
    mtime = wav.stat().st_mtime
    cached = _prompts.get(key)
    if cached is None or cached[0] != mtime:
        prompt = _model.create_voice_clone_prompt(
            ref_audio=str(wav), ref_text=txt.read_text(encoding="utf-8").strip())
        _prompts[key] = cached = (mtime, prompt)
    return cached[1]


def voice_meta(voice: str) -> dict:
    try:
        meta = json.loads((VOICES_DIR / "voices.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entry = meta.get(voice) if isinstance(meta, dict) else None
    return entry if isinstance(entry, dict) else {}


def clone_prompt(voice: str):
    """Prompt for *voice*. An accented voice (voices.json "accent_donor")
    speaks with the donor's delivery — its reference codes and transcript,
    which carry the accent — and the voice's own timbre (speaker embedding)."""
    own = _cached_prompt(voice, VOICES_DIR / f"{voice}.wav", VOICES_DIR / f"{voice}.txt")
    donor = voice_meta(voice).get("accent_donor")
    if not donor:
        return own
    don = _cached_prompt(f"donor:{donor}", DONORS_DIR / f"{donor}.wav",
                         DONORS_DIR / f"{donor}.txt")
    return [dataclasses.replace(don[0], ref_spk_embedding=own[0].ref_spk_embedding)]


def list_voices() -> dict[str, dict]:
    meta = {}
    try:
        meta = json.loads((VOICES_DIR / "voices.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    voices = {}
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        if "." in wav.stem or not wav.with_suffix(".txt").exists():
            continue   # emotion variant: chosen per request via "emotion"
        info = meta.get(wav.stem, {}) if isinstance(meta, dict) else {}
        voices[wav.stem] = {
            "gender": str(info.get("gender", "")),
            "description": str(info.get("description", "cloned voice")),
            "accent": str(info.get("accent", "")),
            "emotions": sorted(
                v.stem.split(".", 1)[1] for v in VOICES_DIR.glob(f"{wav.stem}.*.wav")
                if v.with_suffix(".txt").exists()),
        }
    return voices


def semitone_spread(snd, lo: int, hi: int) -> float:
    """Std of the voiced pitch contour in semitones: natural speech ~1.7-6,
    under ~1.2 sounds monotone/robotic."""
    f = snd.to_pitch(pitch_floor=lo, pitch_ceiling=hi).selected_array["frequency"]
    f = f[f > 0]
    return float(np.std(12 * np.log2(f / np.median(f)))) if len(f) >= 10 else float("nan")


def shape_emotion(audio: np.ndarray, emotion: str, gender: str) -> np.ndarray:
    """Apply the EMOTION_DSP preset for *emotion* (unchanged if none)."""
    preset = EMOTION_DSP.get(emotion)
    if preset is None or audio.size < SAMPLE_RATE // 4:
        return audio
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return audio
    k = EMOTION_INTENSITY
    # Pitch range never below 0.65x: flatter than that is a monotone and
    # reads as robotic (user caught it on cold/sad/tense lines at k=1.5).
    pitch_mul, range_mul, dur_mul, gain_db = (1 + k * (preset[0] - 1), max(0.65, 1 + k * (preset[1] - 1)),
                                              1 + k * (preset[2] - 1), k * preset[3])
    # Big pitch/tempo moves sound processed: keep them modest.
    pitch_mul = min(1.15, max(0.88, pitch_mul))
    dur_mul = min(1.15, max(0.88, dur_mul))
    lo, hi = (100, 500) if gender.upper().startswith("F") else (60, 300)
    snd = parselmouth.Sound(audio.astype(np.float64), SAMPLE_RATE)
    median = call(snd.to_pitch(pitch_floor=lo, pitch_ceiling=hi),
                  "Get quantile", 0, 0, 0.5, "Hertz")
    if not median or median != median:   # unvoiced / NaN: leave it be
        return audio
    # Never flatten the melody below 1.4 semitones of spread (or below what
    # the take had): that is the robotic sound.
    spread = semitone_spread(snd, lo, hi)
    if spread == spread and spread > 0:
        range_mul = max(range_mul, min(1.0, 1.4 / spread))
    try:
        out = call(snd, "Change gender", lo, hi, 1.0, median * pitch_mul, range_mul, dur_mul)
    except parselmouth.PraatError:   # too few voiced frames (a short gravelly word)
        return audio
    shaped = out.values[0] * 10 ** (gain_db / 20)
    if int(out.sampling_frequency) != SAMPLE_RATE:
        return audio
    return (shaped / max(1.0, float(np.abs(shaped).max()) / 0.98)).astype(np.float32)


_xvec_mean = None
# Whisper child processes (tools/whisper_worker.py), checked out one per
# transcription so concurrent requests (the batched Fish server) can verify
# in parallel; QWEN_VERIFY_WORKERS caps how many.
import queue as _queue

_WHISPER_WORKERS = int(os.environ.get("QWEN_VERIFY_WORKERS", "1"))
_whisper_idle: "_queue.Queue" = _queue.Queue()
_whisper_count = 0
_whisper_lock = threading.Lock()


def _whisper_checkout():
    import subprocess

    global _whisper_count
    with _whisper_lock:
        spawn = _whisper_idle.empty() and _whisper_count < _WHISPER_WORKERS
        if spawn:
            _whisper_count += 1
    if not spawn:
        return _whisper_idle.get()
    # Started outside the lock (a slow start must not hold up the other
    # requests) and offline: the model is cached, and faster-whisper's
    # update check to huggingface.co once hung in an SSL handshake.
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("whisper_worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "", "QWEN_VERIFY_WHISPER": WHISPER_MODEL,
                 "HF_HUB_OFFLINE": "1"})
        json.loads(proc.stdout.readline())   # {"ready": true}
    except (OSError, ValueError):
        with _whisper_lock:
            _whisper_count -= 1
        raise RuntimeError("whisper worker failed to start")
    return proc


def transcribe(audio16k: np.ndarray) -> str:
    """Whisper in a child process (CTranslate2 + ROCm torch can't share one)."""
    import tempfile

    global _whisper_count
    with tempfile.NamedTemporaryFile(suffix=".npy", dir="/dev/shm", delete=False) as f:
        np.save(f, audio16k.astype(np.float32))
    try:
        # A worker that died (crashed on a take) is replaced, once.
        for attempt in range(2):
            proc = _whisper_checkout()
            try:
                proc.stdin.write(json.dumps({"path": f.name}) + "\n")
                proc.stdin.flush()
                reply = json.loads(proc.stdout.readline() or "{}")
            except (BrokenPipeError, OSError, ValueError):
                reply = {}
            if proc.poll() is None:
                _whisper_idle.put(proc)
            else:
                with _whisper_lock:
                    _whisper_count -= 1
            if reply:
                break
    finally:
        os.unlink(f.name)
    if "text" not in reply:
        raise RuntimeError(reply.get("error", "whisper worker died"))
    return reply["text"]


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def _sounds_same(ref: list[str], hyp: list[str]) -> bool:
    """True when a mismatched span is the same words spelled or split
    differently (one-letter-per-five slack, spaces ignored)."""
    import difflib

    a, b = "".join(ref), "".join(hyp)
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def check_take(audio: np.ndarray, text: str, voice_emb) -> dict:
    """Score one take: what Whisper hears vs *text*, and voice likeness."""
    import jiwer

    global _xvec_mean
    idx = np.arange(0, len(audio), SAMPLE_RATE / 16000).astype(int)
    heard = transcribe(audio[idx])
    ref, hyp = _words(text), _words(heard)
    seconds = len(audio) / SAMPLE_RATE
    problems = []
    if ref and hyp:
        out = jiwer.process_words(" ".join(ref), " ".join(hyp))
        # A mismatch that is the same sounds spelled differently is not an
        # error: Whisper writes 'forevermore' for 'for evermore', 'visitor'
        # for Poe's 'visiter', 'on treating' for 'entreating'.
        spans = []   # adjacent non-equal chunks merged: [r0, r1, h0, h1]
        for c in out.alignments[0]:
            if c.type == "equal":
                continue
            if spans and spans[-1][1] == c.ref_start_idx and spans[-1][3] == c.hyp_start_idx:
                spans[-1][1], spans[-1][3] = c.ref_end_idx, c.hyp_end_idx
            else:
                spans.append([c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx])
        sizes = [max(r1 - r0, h1 - h0) for r0, r1, h0, h1 in spans
                 if not _sounds_same(ref[r0:r1], hyp[h0:h1])]
        wer = min(1.0, sum(sizes) / len(ref))
        worst = max(sizes, default=0)
        if len(ref) >= 4 and wer > 0.3:
            problems.append(f"wer {wer:.2f}")
        if worst >= 4:
            problems.append(f"{worst}-word gap")
    else:
        wer = 1.0 if ref else 0.0
        if len(ref) >= 2:
            problems.append("nothing heard")
    loops = re.search(r"\b(\w+)(?: \1\b){3,}", " ".join(hyp))
    if loops and not re.search(re.escape(loops.group(0)), " ".join(ref)):
        problems.append(f"loop '{loops.group(1)}'")
    if len(ref) >= 3 and seconds / len(ref) > 1.2:
        problems.append(f"too long ({seconds:.0f}s)")
    if len(ref) >= 6:
        try:
            import parselmouth

            spread = semitone_spread(parselmouth.Sound(audio.astype(np.float64), SAMPLE_RATE), 60, 500)
            if spread == spread and spread < 1.1:
                problems.append(f"monotone ({spread:.1f} st)")
        except ImportError:
            pass
    likeness = None
    if voice_emb is not None and seconds >= 2.0:   # short clips embed unreliably
        if _xvec_mean is None:
            try:
                _xvec_mean = torch.tensor(json.loads(
                    (VOICES_DIR / ".xvec_mean.json").read_text()))
            except (OSError, ValueError):
                _xvec_mean = torch.zeros(0)
        if _xvec_mean.numel():
            emb = _model.create_voice_clone_prompt(
                ref_audio=(audio, SAMPLE_RATE), x_vector_only_mode=True)[0].ref_spk_embedding
            a = emb.float().cpu() - _xvec_mean
            b = voice_emb.float().cpu() - _xvec_mean
            likeness = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            if likeness < VOICE_MIN:
                problems.append(f"voice drift ({likeness:.2f})")
    return {"ok": not problems, "problems": problems, "wer": round(wer, 3),
            "likeness": likeness, "heard": heard.strip()}


def _close_fragment(text: str) -> str:
    """'...shaven, thou,' -> '...shaven, thou.' and 'thy God...' -> 'Thy God...'
    (see speech()): a fragment that starts lowercase or ends on a comma reads
    to Qwen as the middle of a sentence and it babbles to fill it in."""
    m = re.match(r"(\W*)([a-z])", text)
    if m:
        text = text[:m.end(1)] + m.group(2).upper() + text[m.end():]
    return re.sub(r"[\s,;:\u2014\u2013-]+$", ".", text) if re.search(r"[,;:\u2014\u2013-]\s*$", text) else text


def _clauses(text: str, min_words: int = 8) -> list[str]:
    """Split a long line at clause punctuation (dashes, ; : ! ? .) into
    pieces of at least ~3 words; [] when it is too short to bother."""
    if len(text.split()) < min_words:
        return []
    pieces = [p.strip() for p in re.split(r"(?<=[;:!?.\u2014])\s*|\s+(?=\u2014)", text) if p and p.strip()]
    out = []
    for p in pieces:
        p = p.lstrip("\u2014 ").strip()
        if not p:
            continue
        if out and len(out[-1].split()) < 3:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1].split()) < 3:
        out[-2] = f"{out[-2]} {out.pop()}"
    return [_close_fragment(p) for p in out] if len(out) > 1 else []


def _render_verified(text: str, prompt, voice_emb, label: str, tries: int):
    """Best of up to *tries* takes: (rank, audio, report or None)."""
    best, report = None, None
    for attempt in range(tries):
        if attempt:
            torch.manual_seed(random.randrange(2**31))
        # Cap a take at ~1.6 s/word + 4 s (check_take rejects >1.2 s/word
        # anyway): a babble loop otherwise runs to the 2048-frame default
        # (164 s of audio) before the retry can start.
        cap = int(FRAME_RATE * (1.6 * len(text.split()) + 4))
        wavs, sr = _model.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=prompt,
            max_new_tokens=min(2048, cap))
        take = np.asarray(wavs[0], dtype=np.float32)
        if int(sr) != SAMPLE_RATE:
            raise HTTPException(500, f"model returned {sr} Hz, expected {SAMPLE_RATE}")
        if voice_emb is None:
            return None, take, None
        check = check_take(take, text, voice_emb)
        check["takes"] = attempt + 1
        rank = (len(check["problems"]), check["wer"], -(check["likeness"] or 0))
        if best is None or rank < best[0]:
            best, report = (rank, take), check
        if check["ok"]:
            break
        print(f"verify [{label}] take {attempt + 1}: {', '.join(check['problems'])}"
              f" | heard: {check['heard'][:120]!r}", flush=True)
    return best[0], best[1], report


class EmbedRequest(BaseModel):
    path: str   # a WAV on this machine (e.g. a candidate voice clip)


@app.post("/embed")
def embed(req: EmbedRequest):
    """Speaker embedding (x-vector) of a clip — for "is this the same
    person?" checks between a designed voice and its emotional variants."""
    if _model is None:
        raise HTTPException(503, "model still loading")
    wav = Path(req.path).expanduser()
    if not wav.exists():
        raise HTTPException(404, f"no such file: {wav}")
    with _lock:
        item = _model.create_voice_clone_prompt(ref_audio=str(wav), x_vector_only_mode=True)[0]
    return {"embedding": item.ref_spk_embedding.float().cpu().tolist()}


class SpeechRequest(BaseModel):
    input: str
    voice: str = ""
    model: str = "qwen"
    speed: float = 1.0          # accepted for compatibility; not applied
    response_format: str = "wav"
    emotion: str = ""           # picks <voice>.<emotion>.wav when it exists
    verify: bool = True         # Whisper + voice check, re-render failed takes
    tries: int = 0              # takes to allow (0 = 1 + QWEN_VERIFY_RETRIES); 1 = screening


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "model": MODEL}


@app.get("/voices")
def voices():
    return [{"name": k, **v} for k, v in list_voices().items()]


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if _model is None:
        raise HTTPException(503, "model still loading")
    text = req.input.strip()
    if not text:
        raise HTTPException(400, "input is empty")
    voice = req.voice
    # An accented voice keeps its accent donor's delivery on every line;
    # its emotion is added afterwards (shape_emotion), not by a clip.
    meta0 = voice_meta(voice)
    # Real recorded speakers (render: qwen) also get shaped output rather
    # than a shaped reference clip: re-cloning from a shaped clip pulls the
    # delivery back toward neutral (A/B'd 2026-10-12).
    # ...except speakers whose emotional clips are REAL recordings (EARS
    # imports: "real_emotions"), which clone straight from the matching clip.
    if req.emotion and not meta0.get("accent_donor") \
            and (meta0.get("render") != "qwen" or meta0.get("real_emotions")) and (VOICES_DIR / f"{voice}.{req.emotion}.wav").exists() \
            and (VOICES_DIR / f"{voice}.{req.emotion}.txt").exists():
        voice = f"{voice}.{req.emotion}"   # emotional reference clip, same person
    verify = VERIFY and req.verify
    # A fragment cut at a speech tag ('"...shaven, thou," I said, "art...')
    # ends on a comma or starts lowercase; Qwen then invents the rest of the
    # sentence (babble loops), while the same words as a capitalized sentence
    # ending in a period render first time (3/3 vs 0/3 on simon).
    text = _close_fragment(text)
    with _lock:
        prompt = clone_prompt(voice)
        voice_emb = clone_prompt(req.voice)[0].ref_spk_embedding if verify else None
        tries = req.tries if req.tries > 0 else 1 + (VERIFY_RETRIES if verify else 0)
        rank, audio, report = _render_verified(text, prompt, voice_emb, req.voice, tries)
        clauses = _clauses(text) if report and not report["ok"] and tries > 1 else []
        if len(clauses) > 1:
            # Every take of a long line failed: render it clause by clause
            # (each verified on its own) and keep the joined result if it
            # checks out better than the best whole-line take.
            parts = [_render_verified(c, prompt, voice_emb, req.voice, 1 + min(2, VERIFY_RETRIES))[1]
                     for c in clauses]
            gap = np.zeros(int(0.18 * SAMPLE_RATE), dtype=np.float32)
            joined = np.concatenate([x for part in parts for x in (part, gap)][:-1])
            check = check_take(joined, text, voice_emb)
            check["takes"], check["split"] = report["takes"], len(clauses)
            jrank = (len(check["problems"]), check["wer"], -(check["likeness"] or 0))
            print(f"verify [{req.voice}] split into {len(clauses)} clauses: "
                  f"{', '.join(check['problems']) or 'ok'}", flush=True)
            if jrank < rank:
                audio, report = joined, check
    if report and report["takes"] > 1:
        print(f"verify [{req.voice}] kept a take after {report['takes']} "
              f"({'ok' if report['ok'] else 'best effort: ' + ', '.join(report['problems'])})",
              flush=True)
    meta = voice_meta(req.voice)
    if req.emotion and voice == req.voice:
        # No emotional reference clip was used (accented voice, or a real
        # recorded speaker with only a neutral clip): shape the delivery.
        audio = shape_emotion(audio, req.emotion, str(meta.get("gender", "")))
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    headers = {}
    if report:
        headers["X-Verify"] = json.dumps({k: report[k] for k in
                                          ("ok", "problems", "wer", "likeness", "takes")})
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn

    print(f"loading {MODEL} on {torch.cuda.get_device_name(0)}...")
    load_model()
    # Pay the compile tax before serving.
    first = next(iter(list_voices()), None)
    if first:
        speech(SpeechRequest(input="Warming up the voice.", voice=first))
    print("ready")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
