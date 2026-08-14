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

```bash
python tools/mine_voice.py grids/smith.json            # all cells
python tools/mine_voice.py grids/smith.json --cells tense,angry
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
- **SNR / clipping / duration** (8–15 s target)

Survivors land in the registry's `staging/` as `smith.tense.cand1.wav` +
matching transcript.

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
