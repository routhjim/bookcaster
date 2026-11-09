# Adding a narrator: mining an emotional voice palette from LibriVox

Every voice in the library is a **palette**: a neutral reference clip plus up
to seven emotional variants of the *same narrator* (`smith.wav`,
`smith.tense.wav`, ...). This guide is the full workflow for adding a new
narrator, as automated by `tools/mine_voice.py`.

## 1. Pick a narrator and verify solo recordings

Browse LibriVox for a **straight reader** — someone who reads in their own
natural voice with genuine tonal range, *not* a theatrical character-actor
(you would clone the performance, not the person). Then, for every recording
you plan to mine, **verify it's their solo work**:

```bash
curl -s "https://archive.org/metadata/<item_id>" | grep -o "Read by [^.<]*"
```

Group recordings (a different volunteer per chapter) are the #1 way to end
up with someone else's voice in your palette. If the metadata doesn't name a
single reader, skip the item.

## 2. Write the sourcing grid

A grid maps each of the 8 emotions to a chapter where the *text* forces that
register (see `tools/grids/*.json` for examples):

```json
{
  "narrator": "smith",
  "meta": {"gender": "Male", "description": "Steady American baritone"},
  "cells": {
    "neutral": {"item": "whitefang2_1010_librivox",
                "file": "whitefang2_01_london_64kb.mp3",
                "hint": "Opening wilderness chapter; calm scene-setting."},
    "tense":   {"item": "treasure_island_1307_librivox",
                "file": "treasureisland_04_stevenson_64kb.mp3",
                "hint": "Blind Pew's approach; dread, fear of discovery."}
  }
}
```

Check filenames against the item's actual file list (archive.org metadata) —
naming conventions vary per item.

## 3. Run the miner

The miner is CPU only. Its venv needs numpy, scipy, soundfile,
praat-parselmouth and faster-whisper; `~/tts-ab/.venv-qwen` has all of them
(the repo's `.venv` does not). The LLM is Strata, at the URL and model the
pipeline uses (`tts_reader/bookcaster.py`: `LLM_URL`, `LLM_MODEL`, which
`BOOKCASTER_LLM_URL` / `BOOKCASTER_LLM_MODEL` override).

```bash
PY=~/tts-ab/.venv-qwen/bin/python
$PY tools/mine_voice.py tools/grids/smith.json            # all cells
$PY tools/mine_voice.py tools/grids/smith.json --cells tense,angry
$PY tools/mine_voice.py --check ~/.local/share/tts_reader/f5_voices/smith.wav   # screen any clip
```

Per cell it: downloads the chapter (cached), transcribes with faster-whisper
(cached), asks a local LLM to locate spans where the narrator's delivery
should match the emotion, then applies the gates learned the hard way:

- **narration-only** — no quoted dialogue (narrators do character voices;
  a "warm" cell mined from fairy-tale dialogue gave us falsetto princesses)
- **pace vs. the narrator's own baseline** — calm registers ≤1.12×,
  energetic ≤1.30× (F5 clones pace above all; fast clips make fast voices,
  not emotional ones)
- **sentence-final endings** — clips cut mid-sentence make F5 hallucinate
  quiet "ghost" completions into generated audio
- **SNR / clipping**
- **liveliness** (11-09: the narrator smith sounded "dead"; kilmer, weiss
  and melin sounded alive). Smith's clip was flat on *every* channel at
  once; each liked clip has at least one channel that moves, not always the
  same one. Measured on the clip (Praat intensity and pitch at 16 kHz); the
  thresholds are constants in `mine_voice.py`, a little looser than the
  analysis numbers because they come from one take of seven voices:
  1. **phrasing** — at least 2.8 pauses of ≥150 ms per 10 s, *or* a mean
     phrase of ≤2.2 s (smith 2.2 pauses / 2.57 s; liked 3.4-5.0 / 1.4-1.9 s);
  2. **not flat on every channel** — reject when pitch movement is low (SD
     < 3.1 st *and* 5-95 % range < 10.2 st) *and* loudness movement is low
     (syllable-peak SD < 2.9 dB *and* in-phrase swell < 2.4 dB/s);
  3. **pace variation** — CV of syllable rate across the clip's phrases
     ≥ 0.08 (smith 0.05); fewer than two phrases of 3+ syllables fails;
  4. **shape** — 9.5-15.5 s (10-15 s target) and ending on a sentence end.

  Not gated, on purpose: SNR beyond the check above, bandwidth, pitch level
  and the clip's own pace (none tracked the ear, and Fish resets pace).

Survivors are **ranked by a liveliness score** (each channel's measure over
its threshold, capped at 2, averaged over the four channels; 1.0 = on the
line) and land in the registry's `staging/` liveliest first:
`smith.tense.cand1.wav` (the best) + matching transcript.

`--check WAV...` prints the same measures, the score and the verdict for any
existing clip (the transcript for the ending check is the clip's sibling
`.txt`), so library voices can be screened. On the 11-09 clips: smith.wav
fails every gate (score 0.83); kilmer cand2 1.44, weiss cand1 1.51 and
melin cand1 1.33 pass; smithb cand1 (0.80) fails only on pace variation,
because its phrasing is still smith's.

## 4. Audition — the ear is the final gate

Synthesize the same test line from each candidate and listen. The gates
catch the measurable failures; they cannot hear that a clip is "just fast
reading" or subtly wrong. Promote winners by copying candidate → palette
slot (`smith.tense.cand2.wav` → `smith.tense.wav`, plus the `.txt`), and
register the voice in `voices.json`:

```json
"smith": {"gender": "Male", "description": "Steady American baritone",
          "locked": ["warm"]}
```

`locked` lists cells a human approved — automation must never overwrite them.

## 5. Optional per-register shaping

If a register renders too fast/slow or with odd pauses, fix it in
`voices.json` instead of re-mining — the server applies these at synthesis:

```json
"smith": {"emotion_speed": {"excited": 0.92},
          "min_pause_ms": 550, "max_pause_ms": 850}
```

`emotion_speed` is a pitch-preserving time-stretch per register;
the pause band clamps silence lengths (F5 clones the reference's pause
structure, so a ref with a long gap teaches dead air).

## Measured expectations

With a straight reader, register differences are **tonal, not theatrical** —
intensity, weight, attack. In our approved palettes, angry consistently runs
10–25% *faster* than the narrator's neutral; warm/cold/sad run slower.
Expect to audition 2–3 candidates per cell and re-source a few cells per
narrator. A full 8-cell palette takes roughly an hour of machine time and
fifteen minutes of listening.
