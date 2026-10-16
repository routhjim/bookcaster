"""Turn a recording into a cast voice: the clearest 8-15 s of it becomes the
reference clip and Whisper writes its transcript.

A voice in the registry is <name>.wav (24 kHz mono) + <name>.txt (exactly
what the clip says) + an entry in voices.json; the voice servers read the
folder on every request, so the voice is castable at once.

    ~/tts-ab/.venv-qwen/bin/python -I tools/add_voice.py me.m4a --name jim \
        --gender Male --description "warm baritone, Midwestern"

Any format ffmpeg reads (phone voice memos: .m4a, .mp3, .wav, .ogg, video).
Best results: a quiet room, the phone ~20 cm away, a minute of natural
reading. Runs on the CPU (faster-whisper, in the adapter's venv).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

VOICES_DIR = Path(
    os.environ.get("F5_VOICES_DIR")
    or Path.home() / ".local" / "share" / "tts_reader" / "f5_voices"
).expanduser()
SR = 24000
MIN_S, MAX_S = 7.0, 15.0


def decode(src: str, rate: int) -> np.ndarray:
    """Mono float32 at *rate*: rumble cut, loudness evened out."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(rate),
         "-af", "highpass=f=70,loudnorm=I=-20:TP=-2:LRA=11", "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<f4").copy()


def segments(audio16k: np.ndarray) -> list[dict]:
    from faster_whisper import WhisperModel
    model = WhisperModel(os.environ.get("ADD_VOICE_WHISPER", "large-v3-turbo"),
                         device="cpu", compute_type="int8", cpu_threads=8)
    segs, _ = model.transcribe(audio16k, beam_size=5, language="en",
                               vad_filter=True, condition_on_previous_text=False)
    return [{"t0": s.start, "t1": s.end, "text": s.text.strip(),
             "lp": s.avg_logprob, "ns": s.no_speech_prob} for s in segs if s.text.strip()]


def best_window(segs: list[dict]) -> tuple[float, float, str] | None:
    """The contiguous run of whole Whisper segments lasting MIN_S-MAX_S that
    Whisper is surest of, preferring runs that end a sentence and have no
    long silences inside (a clip's pauses are copied into every line)."""
    best, best_score = None, -1e9
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            t0, t1 = segs[i]["t0"], segs[j]["t1"]
            if t1 - t0 > MAX_S:
                break
            if t1 - t0 < MIN_S:
                continue
            run = segs[i:j + 1]
            gaps = max([b["t0"] - a["t1"] for a, b in zip(run, run[1:])] or [0])
            text = " ".join(s["text"] for s in run)
            score = (sum(s["lp"] for s in run) / len(run)
                     - sum(s["ns"] for s in run) / len(run)
                     + (0.3 if re.search(r"[.!?]['\"]?$", text) else 0)
                     + (0.15 if text[:1].isupper() else 0)
                     - max(0.0, gaps - 0.8)
                     + 0.02 * (t1 - t0))
            if score > best_score:
                best, best_score = (t0, t1, text), score
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--name", required=True, help="voice name (letters, digits, _)")
    ap.add_argument("--gender", default="", help="Male / Female")
    ap.add_argument("--description", default="", help="how it sounds, for casting")
    ap.add_argument("--accent", default="")
    ap.add_argument("--replace", action="store_true", help="overwrite a voice of that name")
    a = ap.parse_args()

    name = re.sub(r"[^a-z0-9_]", "_", a.name.lower()).strip("_")
    if not name or "." in name:
        sys.exit("give the voice a simple name, e.g. --name grandpa_joe")
    wav_out, txt_out = VOICES_DIR / f"{name}.wav", VOICES_DIR / f"{name}.txt"
    if wav_out.exists() and not a.replace:
        sys.exit(f"there's already a voice called '{name}' (choose another name, or --replace)")

    try:
        audio16 = decode(a.recording, 16000)
        audio24 = decode(a.recording, SR)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"couldn't read {a.recording}: {exc.stderr.decode(errors='replace')[:300]}")
    total = len(audio24) / SR
    segs = segments(audio16)
    speech = sum(s["t1"] - s["t0"] for s in segs)
    if speech < MIN_S:
        sys.exit(f"only {speech:.0f} s of clear speech found in {total:.0f} s of audio; "
                 "record at least 15-20 s of reading, in a quiet room")
    win = best_window(segs)
    if win is None:   # speech, but no run fits: take the longest single segment
        s = max(segs, key=lambda s: s["t1"] - s["t0"])
        win = (s["t0"], min(s["t1"], s["t0"] + MAX_S), s["text"])
    t0, t1, text = win
    a0, a1 = max(0, int((t0 - 0.12) * SR)), min(len(audio24), int((t1 + 0.25) * SR))
    clip = audio24[a0:a1].copy()
    fade = int(0.03 * SR)
    clip[:fade] *= np.linspace(0, 1, fade)
    clip[-fade:] *= np.linspace(1, 0, fade)

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(suffix=".wav", dir=VOICES_DIR)[1])
    sf.write(tmp, clip, SR, subtype="PCM_16")
    txt_out.write_text(text + "\n", encoding="utf-8")
    tmp.replace(wav_out)   # the wav last: a voice is listed once both files exist

    meta_path = VOICES_DIR / "voices.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta[name] = {k: v for k, v in {
        "gender": a.gender.capitalize(), "description": a.description or "user recording",
        "accent": a.accent.lower(), "source": f"user recording ({Path(a.recording).name}), "
        f"added {time.strftime('%Y-%m-%d')}"}.items() if v}
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"name": name, "clip_seconds": round(len(clip) / SR, 1),
                      "from": [round(t0, 1), round(t1, 1)], "recording_seconds": round(total, 1),
                      "speech_seconds": round(speech, 1), "text": text}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
