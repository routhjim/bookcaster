"""Import real recorded voices and give each a synthetic emotional range.

Flow: (1) fetch real speakers from the VCTK corpus (CC-BY 4.0; 109 speakers
with accent/gender/age/region metadata), build a ~10 s reference clip from
their own sentences, and register it in the voice registry as
``vctk_<id>``; (2) derive the emotional range from that clip — one
``vctk_<id>.<emotion>.wav`` per register, reshaped in pitch, pitch range,
pace and loudness (Praat); (3) use them: they appear in ``/voices`` for
casting, and the Qwen server renders them ("render": "qwen" — F5 flattens
their accents), picking the emotional clip per line.

    # what's available
    ~/tts-ab/.venv-qwen/bin/python -I tools/import_voices.py --list
    # two Scottish women and one Irish man
    ~/tts-ab/.venv-qwen/bin/python -I tools/import_voices.py --accent scottish --gender F --count 2
    ~/tts-ab/.venv-qwen/bin/python -I tools/import_voices.py --speakers p298
    # only rebuild the emotional clips (e.g. after tuning PRESETS)
    ~/tts-ab/.venv-qwen/bin/python -I tools/import_voices.py --speakers p252 --emotions-only

Needs: pyarrow, fsspec, aiohttp, requests, soundfile, numpy, praat-parselmouth.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

DATASET = "sanchit-gandhi/vctk"   # parquet mirror of VCTK 0.92 with metadata
HF = "https://huggingface.co"
SAMPLE_RATE = 24000
VOICES_DIR = Path(
    os.environ.get("F5_VOICES_DIR")
    or Path.home() / ".local" / "share" / "tts_reader" / "f5_voices"
).expanduser()
CACHE = Path.home() / ".cache" / "tts_reader" / "vctk_speakers.json"

# Emotional range from a neutral clip: pitch median x, pitch range x,
# duration x, gain dB. Same presets the Qwen server uses to shape output
# (tools/qwen_tts_server.py EMOTION_DSP); chosen by ear 2026-10-12.
PRESETS = {
    "warm": (0.97, 1.25, 1.06, -1), "tense": (1.05, 0.70, 0.96, -3),
    "angry": (1.08, 1.45, 0.92, 4), "sad": (0.91, 0.60, 1.15, -5),
    "excited": (1.15, 1.60, 0.90, 3), "cold": (0.95, 0.45, 1.00, -1),
    "surprised": (1.20, 1.80, 0.95, 2),
}
ACCENT_NAMES = {"NorthernIrish": "Northern Irish", "SouthAfrican": "South African",
                "NewZealand": "New Zealand", "Australian English": "Australian"}


def shard_urls() -> list[str]:
    import requests

    tree = requests.get(f"{HF}/api/datasets/{DATASET}/tree/main/data", timeout=60).json()
    return sorted(f"{HF}/datasets/{DATASET}/resolve/main/{f['path']}"
                  for f in tree if f["path"].endswith(".parquet"))


def speakers() -> dict:
    """Speaker metadata (accent, gender, age, region, shard), cached."""
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        pass
    import fsspec
    import pyarrow.parquet as pq

    fs, spk = fsspec.filesystem("https"), {}
    for url in shard_urls():
        with fs.open(url, block_size=2**20) as fh:
            t = pq.ParquetFile(fh).read(columns=["speaker_id", "accent", "gender", "age", "region"])
        for row in t.to_pylist():
            entry = spk.setdefault(row["speaker_id"], {**row, "shard": url.split("/main/", 1)[1], "n": 0})
            entry["n"] += 1
        print(f"  scanned {url.rsplit('/', 1)[1]} ({len(spk)} speakers)", file=sys.stderr)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(spk, indent=1))
    return spk


def accent_of(info: dict) -> str:
    return ACCENT_NAMES.get(info["accent"], info["accent"])


def fetch_rows(shard: str, ids: list[str], per_speaker: int = 40) -> dict:
    """Read only the row groups holding *ids* (audio is the bulk of a shard)."""
    import fsspec
    import pyarrow.parquet as pq

    got = collections.defaultdict(list)
    with fsspec.filesystem("https").open(f"{HF}/datasets/{DATASET}/resolve/main/{shard}",
                                         block_size=2**22) as fh:
        pf = pq.ParquetFile(fh)
        for rg in range(pf.num_row_groups):
            present = set(pf.read_row_group(rg, columns=["speaker_id"]).column(0).to_pylist())
            if not present & set(ids):
                continue
            for row in pf.read_row_group(rg, columns=["speaker_id", "audio", "text"]).to_pylist():
                if row["speaker_id"] in ids and len(got[row["speaker_id"]]) < per_speaker:
                    got[row["speaker_id"]].append(row)
            if all(len(got[i]) >= per_speaker for i in ids):
                break
    return got


def to_24k(a: np.ndarray, sr: int) -> np.ndarray:
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SAMPLE_RATE:
        t = np.arange(int(len(a) * SAMPLE_RATE / sr)) * (sr / SAMPLE_RATE)
        a = np.interp(t, np.arange(len(a)), a)
    return a


def reference_clip(rows: list[dict], target_s: float = 10.0):
    """Longest sentences, silence-trimmed, joined to ~target_s seconds."""
    clips = []
    for r in rows:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        a = to_24k(a, sr)
        idx = np.flatnonzero(np.abs(a) > 0.02 * np.abs(a).max())
        if idx.size == 0:
            continue
        a = a[max(0, idx[0] - SAMPLE_RATE // 20): idx[-1] + SAMPLE_RATE // 10]
        clips.append((a, r["text"].strip()))
    clips.sort(key=lambda c: -len(c[0]))
    pick, total = [], 0.0
    for a, text in clips:
        if total >= target_s:
            break
        pick.append((a, text))
        total += len(a) / SAMPLE_RATE
    gap = np.zeros(int(0.35 * SAMPLE_RATE))
    audio = np.concatenate([x for a, _ in pick for x in (a, gap)])
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.9
    return audio.astype(np.float32), " ".join(t for _, t in pick)


def emotional_range(name: str, female: bool) -> list[str]:
    """Write <name>.<emotion>.wav/.txt for every preset; returns emotions."""
    import parselmouth
    from parselmouth.praat import call

    base = VOICES_DIR / f"{name}.wav"
    text = (VOICES_DIR / f"{name}.txt").read_text(encoding="utf-8")
    snd = parselmouth.Sound(str(base))
    lo, hi = (100, 500) if female else (60, 300)
    median = call(snd.to_pitch(pitch_floor=lo, pitch_ceiling=hi), "Get quantile", 0, 0, 0.5, "Hertz")
    done = []
    for emotion, (pitch_mul, range_mul, dur_mul, gain_db) in PRESETS.items():
        out = call(snd, "Change gender", lo, hi, 1.0, median * pitch_mul, range_mul, dur_mul)
        a = out.values[0] * 10 ** (gain_db / 20)
        a = a / max(1.0, float(np.abs(a).max()) / 0.98)
        sf.write(VOICES_DIR / f"{name}.{emotion}.wav", a, int(out.sampling_frequency))
        (VOICES_DIR / f"{name}.{emotion}.txt").write_text(text, encoding="utf-8")
        done.append(emotion)
    return done


def register(name: str, info: dict) -> None:
    path = VOICES_DIR / "voices.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    female = info["gender"] == "F"
    where = f" from {info['region']}" if info.get("region") else ""
    meta[name] = {
        "gender": "Female" if female else "Male",
        "description": f"{accent_of(info)} {'woman' if female else 'man'}, "
                       f"{info.get('age', '?')}{where} (real speaker, VCTK)",
        "accent": accent_of(info).lower(),
        "render": "qwen",
        "source": f"VCTK {info['speaker_id']} (CC-BY 4.0)",
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="list speakers by accent and exit")
    ap.add_argument("--accent", help="e.g. scottish, irish, english, american, indian")
    ap.add_argument("--gender", choices=["M", "F"])
    ap.add_argument("--count", type=int, default=1, help="speakers to import per call")
    ap.add_argument("--speakers", help="explicit VCTK ids, comma-separated (p252,p264)")
    ap.add_argument("--emotions-only", action="store_true",
                    help="rebuild the emotional clips of already-imported speakers")
    args = ap.parse_args()

    spk = speakers()
    if args.list:
        by = collections.defaultdict(list)
        for sid, info in sorted(spk.items()):
            by[accent_of(info)].append(f"{sid}{info['gender']}{info.get('age', '')}")
        for accent, ids in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f"{accent:16s} {len(ids):3d}  {' '.join(ids)}")
        return 0

    if args.speakers:
        ids = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:
        pool = [sid for sid, info in sorted(spk.items())
                if (not args.accent or accent_of(info).lower().replace(" ", "")
                    == args.accent.lower().replace(" ", ""))
                and (not args.gender or info["gender"] == args.gender)
                and not (VOICES_DIR / f"vctk_{sid}.wav").exists()]
        ids = pool[: args.count]
    unknown = [i for i in ids if i not in spk]
    if unknown or not ids:
        print(f"no matching speakers{': ' + ', '.join(unknown) if unknown else ''} "
              "(try --list)", file=sys.stderr)
        return 1

    if not args.emotions_only:
        by_shard = collections.defaultdict(list)
        for sid in ids:
            by_shard[spk[sid]["shard"]].append(sid)
        for shard, group in by_shard.items():
            rows = fetch_rows(shard, group)
            for sid in group:
                if not rows.get(sid):
                    print(f"{sid}: no audio found", file=sys.stderr)
                    continue
                audio, text = reference_clip(rows[sid])
                VOICES_DIR.mkdir(parents=True, exist_ok=True)
                sf.write(VOICES_DIR / f"vctk_{sid}.wav", audio, SAMPLE_RATE)
                (VOICES_DIR / f"vctk_{sid}.txt").write_text(text + "\n", encoding="utf-8")
                register(f"vctk_{sid}", spk[sid])
    for sid in ids:
        name = f"vctk_{sid}"
        if not (VOICES_DIR / f"{name}.wav").exists():
            continue
        emotions = emotional_range(name, spk[sid]["gender"] == "F")
        info = spk[sid]
        print(f"{name}: {accent_of(info)} {info['gender']} {info.get('age', '')} "
              f"{info.get('region', '')} — {len(emotions)} emotional clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
