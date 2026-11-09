"""Mine emotional reference clips for the F5 voice library.

For each grid cell (narrator x emotion -> a LibriVox chapter + a hint about
where the text gets emotional), this:

1. downloads the chapter MP3 from archive.org (cached),
2. transcribes it with faster-whisper (segment timestamps = our alignment),
3. asks a local LLM to pick spans where the narrator's own delivery should
   match the target emotion (7-14 s, sentence-aligned),
4. extracts the clips, quality-gates them (clipping, SNR, liveliness), and
5. writes surviving candidates to a staging dir for human audition, ranked by
   liveliness (cand1 = the liveliest):
       staging/<narrator>.<emotion>.cand<N>.wav + .txt

Usage (CPU only; the venv needs numpy, soundfile, praat-parselmouth and
faster-whisper, which ~/tts-ab/.venv-qwen has):
    ~/tts-ab/.venv-qwen/bin/python tools/mine_voice.py tools/grids/smith.json [--cells tense,angry]
    ~/tts-ab/.venv-qwen/bin/python tools/mine_voice.py --check clip.wav [more.wav ...]
--check prints the liveliness measures and the verdict for existing clips
(the transcript is read from the clip's sibling .txt for the ending check).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts_reader.bookcaster import LLM_MODEL, LLM_URL  # noqa: E402  (Strata, one definition)

CACHE = Path.home() / "models" / "librivox_mining"
STAGING = Path.home() / ".local/share/tts_reader/f5_voices/staging"
SR = 24000


def chat(prompt: str, max_tokens: int = 700) -> str:
    body = {
        "model": LLM_MODEL,
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
        "roughly 10-15 seconds and ending at the end of a sentence (use the "
        "timestamps). Reply with ONLY JSON: "
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
    if len(clip) < CLIP_MIN_S * SR:
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


# --- Liveliness gates (11-09) -------------------------------------------------
# The narrator smith sounded "dead"; kilmer, weiss and melin sounded alive. The
# 11-09 analysis (loop_state/scratch/voice/why_alive.md, its section 5) found
# smith's reference clip is flat on EVERY channel at once (phrasing, pitch
# movement, loudness movement, pace variation), while each liked clip has at
# least one live channel, and not always the same one. These cheap CPU measures
# port the minimal part of that battery (Praat intensity + autocorrelation
# pitch at 16 kHz). One take per voice and seven voices in all, so every
# threshold is a little looser than the analysis numbers: smith.wav must fail,
# and the liked staging clips (kilmer cand2, weiss cand1, melin cand1) pass.
# Not gated, on purpose: SNR beyond gate(), bandwidth, pitch level and the
# clip's own pace (none tracked the ear; Fish resets pace anyway).
LIVE_SR = 16000
PAUSE_MIN_S = 0.15          # a pause = >= 150 ms below (loudest - 25 dB)
SILENCE_BELOW_DB = 25.0
# 1. phrasing: pauses per 10 s of clip OR mean phrase length (the cleanest split:
#    liked 3.4-5.0 pauses/10 s and 1.4-1.9 s phrases; smith 2.2 and 2.6 s).
PHRASING_MIN_PAUSES_10S = 2.8      # analysis: 3
PHRASING_MAX_PHRASE_S = 2.2        # analysis: ~2
# 2. flat on every channel: low pitch movement (SD AND 5-95 % range, semitones)
#    AND low loudness movement (syllable-loudness SD AND in-phrase swell).
#    smith's clip: pitch 2.8 st / 9.2 st, loudness 2.7 dB / 1.7 dB/s (both low).
#    kilmer: pitch 3.3/10.8 (low-ish) but loudness 3.3/3.9; weiss: loudness
#    2.5/1.4 (low) but pitch 4.4/14.0. A clip fails only when BOTH are low.
FLAT_PITCH_SD_ST = 3.1             # analysis: 3.2
FLAT_PITCH_RANGE_ST = 10.2         # analysis: 10.5
FLAT_LOUD_SYLL_SD_DB = 2.9         # analysis: 3
FLAT_LOUD_SWELL_DB_S = 2.4         # analysis: 2.5
# 3. pace variation across the clip's phrases (CV of syllable rate):
#    smith 0.05, liked 0.10-0.24.
PACE_CV_MIN = 0.08                 # analysis: 0.1
# 4. shape: 10-15 s, ending on a sentence end (0.5 s slack each side).
CLIP_MIN_S = 9.5
CLIP_MAX_S = 15.5
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*\s*$")


def ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))


def _st(hz):
    return 12 * np.log2(np.asarray(hz, float) / 100.0)


def liveliness(clip: np.ndarray, sr: int) -> dict:
    """The 11-09 liveliness measures of one mono clip (CPU, ~1 s)."""
    import parselmouth

    snd = parselmouth.Sound(np.asarray(clip, dtype="float64"), sr)
    if sr != LIVE_SR:
        snd = snd.resample(LIVE_SR)
    dur = snd.get_total_duration()
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=50, pitch_ceiling=300)
    f0 = pitch.selected_array["frequency"]
    ft = pitch.xs()
    sv = _st(f0[f0 > 0])
    inten = snd.to_intensity(minimum_pitch=60, time_step=0.01)
    level, it = inten.values[0], inten.xs()
    top = np.percentile(level, 99)
    sil = level < top - SILENCE_BELOW_DB
    pauses, start = [], None
    for i, s in enumerate(np.append(sil, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * 0.01 >= PAUSE_MIN_S:
                pauses.append((it[start], it[min(i, len(it) - 1)]))
            start = None
    if pauses and pauses[0][0] <= it[0] + 0.02:      # leading / trailing silence
        pauses = pauses[1:]
    if pauses and pauses[-1][1] >= it[-1] - 0.02:
        pauses = pauses[:-1]
    voiced_t = it[~sil]
    edges = [voiced_t[0]] + [x for p in pauses for x in p] + [voiced_t[-1]]
    phrases = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    # syllable nuclei: voiced intensity peaks with >= 2 dB dips, near the top level
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(level, prominence=2.0, distance=5)

    def f0_at(t):
        return f0[min(len(f0) - 1, max(0, int(round((t - ft[0]) / 0.01))))]

    nuc = [i for i in peaks if level[i] > top - 22 and f0_at(it[i]) > 0]
    nt, ni = it[nuc], level[nuc]
    rates = []
    for a, b in phrases:
        m = (nt >= a) & (nt <= b)
        if m.sum() >= 3:
            rates.append(m.sum() / max(b - a, 1e-3))
    # in-phrase swell, as the analysis's loudshape.py measured it: label voiced
    # frames by phrase (a phrase ends after >= 150 ms of silence), then the mean
    # |slope| of a line through each phrase's syllable-peak levels, in dB/s
    lab, ph, run, inph = np.zeros(len(sil), int), 0, 0, False
    for i, s in enumerate(sil):
        run = run + 1 if s else 0
        if not s and not inph:
            ph, inph = ph + 1, True
        if run >= round(PAUSE_MIN_S / 0.01):
            inph = False
        lab[i] = 0 if s else ph
    swell_nuc = [i for i in peaks
                 if level[i] > top - 22 and f0[min(len(f0) - 1, i)] > 0 and lab[i] > 0]
    swells = []
    for k in range(1, ph + 1):
        idx = [i for i in swell_nuc if lab[i] == k]
        if len(idx) >= 3:
            x = it[idx]
            swells.append(abs(np.polyfit(x - x[0], level[idx], 1)[0]))
    nan = float("nan")
    return {
        "duration_s": float(dur),
        "pauses": len(pauses),
        "pauses_per_10s": 10.0 * len(pauses) / max(dur, 1e-3),
        "phrase_mean_s": float(np.mean([b - a for a, b in phrases])),
        "f0_sd_st": float(sv.std()) if len(sv) else 0.0,
        "f0_range_st": float(np.percentile(sv, 95) - np.percentile(sv, 5)) if len(sv) else 0.0,
        "syll_loud_sd_db": float(ni.std()) if len(ni) else 0.0,
        "swell_db_s": float(np.mean(swells)) if swells else 0.0,
        "pace_cv": float(np.std(rates) / np.mean(rates)) if len(rates) > 1 else nan,
    }


def live_verdict(m: dict, text: str | None = None) -> tuple[list[str], float]:
    """(failed gate reasons, liveliness score) for liveliness() measures.

    The score ranks survivors: each channel scores its measure over its
    threshold (the better of its two measures, capped at 2), and the mean over
    the four channels is the score; 1.0 = right at the line on average."""
    fails = []
    phrasing_ok = (m["pauses_per_10s"] >= PHRASING_MIN_PAUSES_10S
                   or m["phrase_mean_s"] <= PHRASING_MAX_PHRASE_S)
    if not phrasing_ok:
        fails.append(f"long phrases ({m['pauses_per_10s']:.1f} pauses/10 s, "
                     f"{m['phrase_mean_s']:.2f} s phrases)")
    pitch_flat = m["f0_sd_st"] < FLAT_PITCH_SD_ST and m["f0_range_st"] < FLAT_PITCH_RANGE_ST
    loud_flat = (m["syll_loud_sd_db"] < FLAT_LOUD_SYLL_SD_DB
                 and m["swell_db_s"] < FLAT_LOUD_SWELL_DB_S)
    if pitch_flat and loud_flat:
        fails.append("flat on every channel (pitch and loudness)")
    pace_cv = m["pace_cv"]
    if pace_cv != pace_cv:   # NaN: fewer than two phrases of 3+ syllables, no variation shown
        fails.append("pace variation unmeasurable (fewer than two phrases of 3+ syllables)")
    elif pace_cv < PACE_CV_MIN:
        fails.append(f"even pace (CV {pace_cv:.2f})")
    if not CLIP_MIN_S <= m["duration_s"] <= CLIP_MAX_S:
        fails.append(f"length {m['duration_s']:.1f} s")
    if text is not None and not ends_sentence(text):
        fails.append("ends mid-sentence")
    channels = [
        max(m["pauses_per_10s"] / PHRASING_MIN_PAUSES_10S,
            PHRASING_MAX_PHRASE_S / max(m["phrase_mean_s"], 1e-3)),
        max(m["f0_sd_st"] / FLAT_PITCH_SD_ST, m["f0_range_st"] / FLAT_PITCH_RANGE_ST),
        max(m["syll_loud_sd_db"] / FLAT_LOUD_SYLL_SD_DB, m["swell_db_s"] / FLAT_LOUD_SWELL_DB_S),
        (pace_cv / PACE_CV_MIN) if pace_cv == pace_cv else 0.0,
    ]
    score = float(np.mean([min(c, 2.0) for c in channels]))
    return fails, score


def format_check(name: str, m: dict, fails: list[str], score: float) -> str:
    return (f"{name}: {'FAIL' if fails else 'pass'}  score {score:.2f}  "
            f"{m['duration_s']:.1f} s | phrasing {m['pauses']} pauses "
            f"({m['pauses_per_10s']:.1f}/10 s), phrases {m['phrase_mean_s']:.2f} s | "
            f"pitch SD {m['f0_sd_st']:.2f} st, range {m['f0_range_st']:.1f} st | "
            f"loudness syll SD {m['syll_loud_sd_db']:.2f} dB, swell {m['swell_db_s']:.2f} dB/s | "
            f"pace CV {m['pace_cv']:.2f}"
            + (f"\n    fails: {'; '.join(fails)}" if fails else ""))


def check_clip(wav: Path) -> tuple[dict, list[str], float]:
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32", always_2d=True)
    txt = wav.with_suffix(".txt")
    text = txt.read_text(encoding="utf-8") if txt.exists() else None
    m = liveliness(audio.mean(axis=1), sr)
    fails, score = live_verdict(m, text)
    return m, fails, score


# F5 clones pace above all; a "warm" clip that is merely fast reading makes
# a fast voice, not a warm one. Calm registers must sit near the narrator's
# baseline pace; energetic ones may lift, but pace alone is not emotion.
_CALM = {"neutral", "warm", "sad", "cold"}
_MAX_PACE = {"calm": 1.12, "energetic": 1.30}


def _wps(span: list[dict]) -> float:
    words = sum(len(s["text"].split()) for s in span)
    dur = span[-1]["end"] - span[0]["start"]
    return words / max(dur, 0.1)


def trim_span(span: list[dict]) -> list[dict] | None:
    """Whole segments from the start of the pick until CLIP_MIN_S, extended to
    a sentence-final segment within CLIP_MAX_S. The LLM finds the right place
    but often misjudges the length; a clip cut mid-sentence makes F5
    hallucinate a completion of it, and the 11-09 gates want 10-15 s.
    None when no sentence end falls in range."""
    trimmed = []
    for s in span:
        trimmed.append(s)
        dur = trimmed[-1]["end"] - trimmed[0]["start"]
        if dur > CLIP_MAX_S:
            return None
        if dur >= CLIP_MIN_S and ends_sentence(s["text"]):
            return trimmed
    return None


def rank_survivors(survivors: list[dict]) -> list[dict]:
    """Liveliest first: cand1 = the best."""
    return sorted(survivors, key=lambda c: -c["score"])


def mine_cell(narrator: str, emotion: str, cell: dict) -> list[str]:
    import soundfile as sf

    print(f"  [{narrator}.{emotion}] {cell['item']}/{cell['file']}")
    wav = fetch(cell["item"], cell["file"])
    segs = transcribe(wav)
    baseline = _wps(segs) if segs else 3.0
    limit = baseline * _MAX_PACE["calm" if emotion in _CALM else "energetic"]
    picks = pick_spans(segs, emotion, cell.get("hint", ""))
    audio, _ = sf.read(wav, dtype="float32")
    survivors = []
    for n, pick in enumerate(picks, 1):
        try:
            i = int(pick["start_seg"])
            # from the pick's start; it may run past end_seg to reach a sentence end
            span = [s for s in segs if s["i"] >= i]
            span[0]  # noqa: expression — raises IndexError when empty
        except (KeyError, ValueError, IndexError):
            continue
        trimmed = trim_span(span)
        if trimmed is None:
            print(f"    pick{n}: no sentence end within {CLIP_MIN_S}-{CLIP_MAX_S} s, skipped")
            continue
        start, end = trimmed[0]["start"], trimmed[-1]["end"]
        text_all = " ".join(s["text"] for s in trimmed).strip()
        if re.search(r"[\"“”‘’']\s|\b(said|asked|cried|replied|shouted|whispered|exclaimed)\b",
                     text_all):
            print(f"    pick{n}: contains dialogue (character-voice risk), skipped")
            continue
        pace = _wps(trimmed)
        if pace > limit:
            print(f"    pick{n}: too fast ({pace:.1f} w/s vs narrator "
                  f"baseline {baseline:.1f}), skipped")
            continue
        clip = audio[int(start * SR):int(end * SR)]
        ok, verdict = gate(clip)
        print(f"    pick{n}: {end-start:.1f}s {verdict} {pace:.1f}w/s | {text_all[:60]}")
        if not ok:
            continue
        m = liveliness(clip, SR)
        fails, score = live_verdict(m, text_all)
        print("      " + format_check("liveliness", m, fails, score))
        if fails:
            continue
        survivors.append({"clip": clip, "text": text_all, "score": score})
    STAGING.mkdir(parents=True, exist_ok=True)
    kept = []
    for rank, c in enumerate(rank_survivors(survivors), 1):
        wav_path = STAGING / f"{narrator}.{emotion}.cand{rank}.wav"
        sf.write(wav_path, c["clip"], SR)
        wav_path.with_name(f"{narrator}.{emotion}.cand{rank}.txt").write_text(
            c["text"] + "\n", encoding="utf-8")
        kept.append(f"{wav_path.name} (score {c['score']:.2f})")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", nargs="?", help="JSON grid file for one narrator")
    ap.add_argument("--cells", default=None,
                    help="comma-separated emotions to mine (default: all)")
    ap.add_argument("--check", nargs="+", metavar="WAV", type=Path,
                    help="print the liveliness measures and verdict of existing clips; "
                         "exit 1 if any fails")
    args = ap.parse_args()
    if args.check:
        failed = 0
        for wav in args.check:
            try:
                m, fails, score = check_clip(wav)
            except Exception as exc:  # unreadable clip: report it, keep screening
                print(f"{wav}: ERROR {exc}")
                failed += 1
                continue
            print(format_check(str(wav), m, fails, score))
            failed += bool(fails)
        return 1 if failed else 0
    if not args.grid:
        ap.error("a grid file (or --check WAV) is required")

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
