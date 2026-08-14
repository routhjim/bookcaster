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
        "CRITICAL: pick spans of pure NARRATION in the narrator's own voice. "
        "Absolutely no quoted character dialogue — narrators perform "
        "characters (funny voices, falsetto for women), which would "
        "contaminate the clip with a voice that is not theirs. Avoid any "
        "segment containing quotation marks or speech attribution.\n\n"
        "Pick the THREE best spans of consecutive segments, each spanning "
        "roughly 7-14 seconds (use the timestamps). Reply with ONLY JSON: "
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


# F5 clones pace above all; a "warm" clip that is merely fast reading makes
# a fast voice, not a warm one. Calm registers must sit near the narrator's
# baseline pace; energetic ones may lift, but pace alone is not emotion.
_CALM = {"neutral", "warm", "sad", "cold"}
_MAX_PACE = {"calm": 1.12, "energetic": 1.30}


def _wps(span: list[dict]) -> float:
    words = sum(len(s["text"].split()) for s in span)
    dur = span[-1]["end"] - span[0]["start"]
    return words / max(dur, 0.1)


def mine_cell(narrator: str, emotion: str, cell: dict) -> list[str]:
    print(f"  [{narrator}.{emotion}] {cell['item']}/{cell['file']}")
    wav = fetch(cell["item"], cell["file"])
    segs = transcribe(wav)
    baseline = _wps(segs) if segs else 3.0
    limit = baseline * _MAX_PACE["calm" if emotion in _CALM else "energetic"]
    picks = pick_spans(segs, emotion, cell.get("hint", ""))
    audio, _ = sf.read(wav, dtype="float32")
    kept = []
    STAGING.mkdir(parents=True, exist_ok=True)
    for n, pick in enumerate(picks, 1):
        try:
            i, j = int(pick["start_seg"]), int(pick["end_seg"])
            span = [s for s in segs if i <= s["i"] <= j]
            span[0]  # noqa: expression — raises IndexError when empty
        except (KeyError, ValueError, IndexError):
            continue
        # The LLM finds the right location but often overshoots the length;
        # keep whole segments from the start of the pick until 8-14 s, then
        # extend to a sentence-final segment — a ref clip cut mid-sentence
        # makes F5 hallucinate a completion of it into the output.
        trimmed, fallback = [], None
        for s in span:
            trimmed.append(s)
            dur = trimmed[-1]["end"] - trimmed[0]["start"]
            if dur >= 8 and fallback is None:
                fallback = list(trimmed)  # mid-sentence cut as a last resort
            if dur >= 8 and trimmed[-1]["text"].rstrip().endswith((".", "!", "?", "”", "'")):
                break
            if dur >= 15:
                trimmed = fallback or trimmed  # no clean ending in range
                break
        else:
            trimmed = trimmed if trimmed else span[:1]
        start, end = trimmed[0]["start"], trimmed[-1]["end"]
        if not 6 <= end - start <= 16:
            print(f"    cand{n}: bad duration {end-start:.1f}s, skipped")
            continue
        text_all = " ".join(s["text"] for s in trimmed)
        if re.search(r"[\"“”‘’']\s|\b(said|asked|cried|replied|shouted|whispered|exclaimed)\b",
                     text_all):
            print(f"    cand{n}: contains dialogue (character-voice risk), skipped")
            continue
        pace = _wps(trimmed)
        if pace > limit:
            print(f"    cand{n}: too fast ({pace:.1f} w/s vs narrator "
                  f"baseline {baseline:.1f}), skipped")
            continue
        clip = audio[int(start * SR):int(end * SR)]
        ok, verdict = gate(clip)
        print(f"    cand{n}: {end-start:.1f}s {verdict} {pace:.1f}w/s | "
              f"{' '.join(s['text'] for s in trimmed)[:60]}")
        if not ok:
            continue
        wav_path = STAGING / f"{narrator}.{emotion}.cand{n}.wav"
        sf.write(wav_path, clip, SR)
        wav_path.with_name(f"{narrator}.{emotion}.cand{n}.txt").write_text(
            " ".join(s["text"] for s in trimmed).strip() + "\n",
            encoding="utf-8")
        kept.append(wav_path.name)
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
