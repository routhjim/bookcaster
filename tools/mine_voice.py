"""Mine emotional reference clips for the F5 voice library.

For each grid cell (narrator x emotion -> a LibriVox chapter + a hint about
where the text gets emotional), this:

1. downloads the chapter MP3 from archive.org (cached),
2. transcribes it with faster-whisper (segment timestamps = our alignment),
3. asks a local LLM to pick spans where the narrator's own delivery should
   match the target emotion (7-14 s, sentence-aligned),
4. extracts the clips, quality-gates them (clipping, SNR), and
5. writes surviving candidates to a staging dir for human audition:
       staging/<narrator>.<emotion>.cand<N>.wav + .txt

Usage: .venv/bin/python mine_voice.py grids/smith.json [--cells tense,angry]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

CACHE = Path.home() / "models" / "librivox_mining"
STAGING = Path.home() / ".local/share/tts_reader/f5_voices/staging"
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
SR = 24000


def chat(prompt: str, max_tokens: int = 700) -> str:
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        LLM_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def fetch(item: str, fname: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    mp3 = CACHE / fname
    if not mp3.exists():
        print(f"    downloading {item}/{fname}")
        urllib.request.urlretrieve(
            f"https://archive.org/download/{item}/{fname}", mp3
        )
    wav = mp3.with_suffix(".wav")
    if not wav.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
             "-ac", "1", "-ar", str(SR), str(wav)], check=True)
    return wav


_whisper = None


def transcribe(wav: Path) -> list[dict]:
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    cache = wav.with_suffix(".segments.json")
    if cache.exists():
        return json.loads(cache.read_text())
    print("    transcribing (cached afterwards)...")
    segments, _ = _whisper.transcribe(str(wav), vad_filter=True)
    out = [
        {"i": i, "start": s.start, "end": s.end, "text": s.text.strip()}
        for i, s in enumerate(segments)
    ]
    cache.write_text(json.dumps(out))
    return out


def pick_spans(segs: list[dict], emotion: str, hint: str) -> list[dict]:
    numbered = "\n".join(
        f"[{s['i']}] ({s['start']:.0f}s) {s['text']}" for s in segs
    )[:60000]
    prompt = (
        "Below is a numbered transcript of one audiobook chapter read by a "
        "single narrator. I need short reference clips where the narrator's "
        f"own delivery is likely *{emotion}* — passages whose content forces "
        f"that register. Hint about this chapter: {hint}\n\n"
        "Pick the THREE best spans of consecutive segments, each spanning "
        "roughly 7-14 seconds (use the timestamps), preferring vivid "
        "mid-scene lines over scene-setting. Reply with ONLY JSON: "
        '[{"start_seg": <i>, "end_seg": <i>, "why": "..."}]\n\n' + numbered
    )
    reply = chat(prompt)
    match = re.search(r"\[.*\]", reply, re.DOTALL)
    if not match:
        return []
    try:
        picks = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [p for p in picks if isinstance(p, dict)]


def gate(clip: np.ndarray) -> tuple[bool, str]:
    if len(clip) < 6 * SR:
        return False, "too short"
    if np.abs(clip).max() > 0.985:
        return False, "clipping"
    frame = SR // 20  # 50 ms
    n = len(clip) // frame
    rms = np.sqrt(np.mean(
        clip[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    floor = np.percentile(rms, 10)
    speech = np.percentile(rms, 90)
    snr = 20 * np.log10(speech / max(floor, 1e-9))
    if snr < 16:
        return False, f"noisy (snr {snr:.0f} dB)"
    return True, f"ok (snr {snr:.0f} dB)"


def mine_cell(narrator: str, emotion: str, cell: dict) -> list[str]:
    print(f"  [{narrator}.{emotion}] {cell['item']}/{cell['file']}")
    wav = fetch(cell["item"], cell["file"])
    segs = transcribe(wav)
    picks = pick_spans(segs, emotion, cell.get("hint", ""))
    audio, _ = sf.read(wav, dtype="float32")
    kept = []
    STAGING.mkdir(parents=True, exist_ok=True)
    for n, pick in enumerate(picks, 1):
        try:
            i, j = int(pick["start_seg"]), int(pick["end_seg"])
            span = [s for s in segs if i <= s["i"] <= j]
            start, end = span[0]["start"], span[-1]["end"]
        except (KeyError, ValueError, IndexError):
            continue
        if not 6 <= end - start <= 16:
            print(f"    cand{n}: bad duration {end-start:.1f}s, skipped")
            continue
        clip = audio[int(start * SR):int(end * SR)]
        ok, verdict = gate(clip)
        print(f"    cand{n}: {end-start:.1f}s {verdict} | "
              f"{' '.join(s['text'] for s in span)[:60]}")
        if not ok:
            continue
        stem = STAGING / f"{narrator}.{emotion}.cand{n}"
        sf.write(stem.with_suffix(".wav"), clip, SR)
        stem.with_suffix(".txt").write_text(
            " ".join(s["text"] for s in span).strip() + "\n", encoding="utf-8")
        kept.append(stem.name)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", help="JSON grid file for one narrator")
    ap.add_argument("--cells", default=None,
                    help="comma-separated emotions to mine (default: all)")
    args = ap.parse_args()

    grid = json.loads(Path(args.grid).read_text())
    narrator = grid["narrator"]
    only = set(args.cells.split(",")) if args.cells else None
    results = {}
    for emotion, cell in grid["cells"].items():
        if only and emotion not in only:
            continue
        try:
            results[emotion] = mine_cell(narrator, emotion, cell)
        except Exception as exc:  # keep mining other cells
            print(f"  [{narrator}.{emotion}] FAILED: {exc}")
            results[emotion] = []
    print("\nSummary:")
    for emotion, kept in results.items():
        print(f"  {emotion}: {len(kept)} candidate(s) staged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
