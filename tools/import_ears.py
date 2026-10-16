"""Import EARS speakers with their REAL emotional recordings.

EARS (Expressive Anechoic Recordings of Speech; Meta, CC BY-NC 4.0) has 107
speakers who each read transcribed sentences in 22 emotions plus speaking
styles (loud, whisper, slow, fast, high/low pitch). Every one becomes a
reference clip of the same person, so a line tagged "anger" is cloned from
that speaker genuinely angry — no synthetic shaping. Registered as
``ears_pXXX`` (+ ``ears_pXXX.<emotion>.wav``) in the voice registry with
"render": "qwen" and "real_emotions": true.

    ~/tts-ab/.venv-qwen/bin/python -I tools/import_ears.py --list
    ~/tts-ab/.venv-qwen/bin/python -I tools/import_ears.py p008 p012
    ~/tts-ab/.venv-qwen/bin/python -I tools/import_ears.py --gender female --age 26-35 --count 3

American English for 97 of 107 speakers; p008 is the one British speaker.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main"
RELEASE = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset"
CACHE = Path.home() / ".cache" / "tts_reader" / "ears"
SAMPLE_RATE = 24000
VOICES_DIR = Path(
    os.environ.get("F5_VOICES_DIR")
    or Path.home() / ".local" / "share" / "tts_reader" / "f5_voices"
).expanduser()

# The 22 EARS emotions (their spelling) -> registry emotion names.
EMOTIONS = {
    "adoration": "adoration", "amazement": "amazement", "amusement": "amusement",
    "anger": "anger", "confusion": "confusion", "contentment": "contentment",
    "cuteness": "cuteness", "desire": "desire", "disappointment": "disappointment",
    "disgust": "disgust", "distress": "distress", "embarassment": "embarrassment",
    "extasy": "ecstasy", "fear": "fear", "guilt": "guilt", "interest": "interest",
    "pain": "pain", "pride": "pride", "realization": "realization",
    "relief": "relief", "sadness": "sadness", "serenity": "serenity",
}
# Speaking styles: (file stem, registry name). Different sentences per style.
STYLES = [("sentences_05_loud", "loud"), ("sentences_03_whisper", "whisper"),
          ("sentences_07_slow", "slow"), ("sentences_09_fast", "fast"),
          ("sentences_11_highpitch", "highpitch"), ("sentences_13_lowpitch", "lowpitch")]
NEUTRAL = "emo_neutral_sentences"


def fetch_json(name: str) -> dict:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(f"{REPO}/{name}", timeout=60).read())
    return json.loads(path.read_text())


def speaker_zip(sid: str) -> zipfile.ZipFile:
    path = CACHE / f"{sid}.zip"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {sid} (~700 MB)...", file=sys.stderr)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(f"{RELEASE}/{sid}.zip", timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return zipfile.ZipFile(path)


def read_clip(zf: zipfile.ZipFile, sid: str, stem: str) -> np.ndarray | None:
    name = f"{sid}/{stem}.wav"
    if name not in zf.namelist():
        return None
    a, sr = sf.read(io.BytesIO(zf.read(name)), dtype="float32", always_2d=True)
    a = a.mean(1)
    if sr != SAMPLE_RATE:
        t = np.arange(int(len(a) * SAMPLE_RATE / sr)) * (sr / SAMPLE_RATE)
        a = np.interp(t, np.arange(len(a)), a).astype(np.float32)
    idx = np.flatnonzero(np.abs(a) > 0.02 * np.abs(a).max())
    a = a[max(0, idx[0] - SAMPLE_RATE // 20): idx[-1] + SAMPLE_RATE // 10]
    return (a / max(1e-6, np.abs(a).max()) * 0.9).astype(np.float32)


def register(name: str, info: dict, emotions: list[str]) -> None:
    path = VOICES_DIR / "voices.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    lang = info.get("native language", "")
    accent = {"american english": "american", "british english": "british"}.get(lang, lang)
    meta[name] = {
        "gender": info.get("gender", "").capitalize(),
        "description": f"{accent.title()} {info.get('gender', '')}, {info.get('age', '?')} "
                       f"(real speaker with recorded emotions, EARS {name[5:]})",
        "accent": accent,
        "age": info.get("age", ""),
        "render": "qwen",
        "real_emotions": True,
        "emotions": emotions,
        "source": f"EARS {name[5:]} (CC BY-NC 4.0)",
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def import_speaker(sid: str, speakers: dict, transcripts: dict) -> None:
    zf = speaker_zip(sid)
    name = f"ears_{sid}"
    base = read_clip(zf, sid, NEUTRAL)
    if base is None:
        print(f"{sid}: no neutral clip", file=sys.stderr)
        return
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(VOICES_DIR / f"{name}.wav", base, SAMPLE_RATE)
    (VOICES_DIR / f"{name}.txt").write_text(transcripts[NEUTRAL] + "\n", encoding="utf-8")
    done = []
    for stem, reg in [(f"emo_{e}_sentences", r) for e, r in EMOTIONS.items()] + STYLES:
        clip = read_clip(zf, sid, stem)
        if clip is None or stem not in transcripts:
            continue
        sf.write(VOICES_DIR / f"{name}.{reg}.wav", clip, SAMPLE_RATE)
        (VOICES_DIR / f"{name}.{reg}.txt").write_text(transcripts[stem] + "\n", encoding="utf-8")
        done.append(reg)
    register(name, speakers.get(sid, {}), done)
    info = speakers.get(sid, {})
    print(f"{name}: {info.get('gender', '?')} {info.get('age', '?')} "
          f"{info.get('native language', '?')} — {len(done)} emotion/style clips")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("speakers", nargs="*", help="EARS ids, e.g. p008 p012")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gender", choices=["male", "female"])
    ap.add_argument("--age", help="e.g. 26-35, 56-65")
    ap.add_argument("--count", type=int, default=1)
    args = ap.parse_args()
    speakers = fetch_json("speaker_statistics.json")
    transcripts = fetch_json("transcripts.json")
    if args.list:
        for sid, v in sorted(speakers.items()):
            print(f"{sid}  {v.get('gender', ''):7s} {v.get('age', ''):7s} {v.get('native language', '')}")
        return 0
    ids = args.speakers or [
        sid for sid, v in sorted(speakers.items())
        if (not args.gender or v.get("gender") == args.gender)
        and (not args.age or v.get("age") == args.age)
        and not (VOICES_DIR / f"ears_{sid}.wav").exists()][: args.count]
    for sid in ids:
        import_speaker(sid, speakers, transcripts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
