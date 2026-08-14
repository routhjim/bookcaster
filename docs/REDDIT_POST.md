# [Reddit draft — r/LocalLLaMA. Title goes in the title field, body below the line. Paste in Markdown editor mode.]

**Title:** Turned my Strix Halo into a local audiobook studio: an LLM casts the characters, public-domain LibriVox narrators voice them, emotional delivery switches line-by-line. ~$0.35 of electricity per book.

---

Demo first. Same scene (the climax of Sherlock Holmes' *The Speckled Band*), rendered two ways, fully local:

- [single narrator](https://github.com/routhjim/bookcaster/raw/main/docs/demo/demo_single_narrator.mp3) (one cloned voice reads everything)
- [full cast](https://github.com/routhjim/bookcaster/raw/main/docs/demo/demo_full_cast.mp3) (LLM attributes every line, four different narrators play the characters, and each line is synthesized from a reference clip matching its emotion. The terrified client sounds terrified, the menacing stepfather sounds menacing)

None of these narrators ever recorded this scene. They never met. All voices are LibriVox public domain, so no likeness issues.

Repo: https://github.com/routhjim/bookcaster (MIT)

## Numbers, since that's why we're all here

Hardware is a Ryzen AI MAX+ 395 (Strix Halo, 128GB unified, 8060S iGPU). Everything runs on this one box, including the LLM doing the casting.

| thing | number |
|---|---|
| F5-TTS synthesis | 0.53x real-time (was 0.33x, torch.compile bought 1.65x) |
| 16-step NFE mode | measured 2.05x faster, but I could hear the difference, so everything here is 32-step |
| full multi-voice novel | overnight render, ~$0.35 of power |
| same book via cloud TTS APIs | $120-360, per render (I re-rendered constantly while tuning) |
| speaker attribution, full novel | ~10 min on Qwen3.6-35B-A3B, cached forever after |
| new 8-emotion voice from LibriVox | ~1 hr machine time + 15 min of listening |

Strix Halo notes you might actually want: stock pytorch ROCm wheels **segfault on bare matmul** on gfx1151. AMD's own wheel index (rocm.nightlies.amd.com/v2/gfx1151/) works fine. llama.cpp runs Vulkan on the same box concurrently. Also tested: batching/concurrent F5 requests gain ~nothing, one request already saturates the APU. The NPU sits idle, there's no Linux VitisAI EP (checked against Ryzen AI 1.8.0, all wheels are win_amd64).

## The pipeline

```
book.txt / Gutenberg URL
 1. preprocess    abbreviations, roman numerals, LLM-built pronunciation
                  lexicon ("clanging" was coming out "clan-jing")
 2. chapterize    heading detection, everything resumable per chapter
 3. segment       narration vs quoted dialogue
 4. attribute     LLM names each line's speaker (regex catches the free
                  "said Ahab" cases first) -> cached sidecar json
 5. cast          LLM describes each character, proposes voices from the
                  library; you override in plain English at a REPL
                  ("Starbuck is an old, grizzled, crusty pirate")
 6. tag emotion   LLM assigns each line a register from textual cues
 7. synthesize    speaker -> voice, register -> that narrator's matching
                  reference clip, F5-TTS clones the delivery
 8. assemble      tagged per-chapter MP3s + playlist, or .m4b with
                  chapter marks, Author/Title layout for Audiobookshelf
```

LLM steps hit any OpenAI-compatible endpoint (mine is llama.cpp + Qwen3.6-35B-A3B on the same APU). Every decision caches next to the book, so the expensive passes happen once and recasting is basically instant.

The voice library is its own offline pipeline: LibriVox chapter → faster-whisper alignment → LLM finds passages where the text forces an emotion (Blind Pew approaching = tense) → quality gates → I audition 2-3 candidates per cell. Docs: [VOICE_MINING.md](https://github.com/routhjim/bookcaster/blob/main/docs/VOICE_MINING.md). Each voice ends up as 8 reference clips of the same narrator: neutral/warm/tense/angry/sad/excited/cold/surprised.

## Things that went wrong (the useful part)

- **F5 clones pace above everything.** My first "excited" clips were just the narrator reading fast, so I got a fast voice, not an excited one. Had to gate mined clips against each narrator's own words-per-second baseline. Fun finding from calibrating against clips my ear had approved: angry consistently runs 10-25% *faster* than neutral. Menace accelerates.
- **Never use reference clips containing dialogue.** Narrators perform character voices. I cloned a "warm" clip from a fairy tale and got the narrator's falsetto princess. Related: LibriVox group recordings have a different volunteer per chapter. Verify solo readers per file or you will clone a stranger.
- **Reference clips cut mid-sentence make F5 hallucinate.** It quietly "completes" the reference's unfinished sentence into your audiobook. One of my voices kept whispering "I am lost" before lines. Fixed with sentence-final clip cuts.
- **Short lines are cursed.** Feed F5 "Always." against a 12-second reference and it pads the output with barely-audible ghost mumbling. Fixed with an edge energy gate on the output.
- **The tags got read aloud.** For one glorious render the cast dramatically announced "tensuh" before tense lines because the emotion tags leaked into the text path. My favorite bug of the project.
- Whisper drops quotation marks in transcripts, so text-based dialogue filters miss things. The human audition pass is not optional and my ear caught multiple things the gates passed.

## Honest limitations

Attribution is ~95%, not 100% (misses fall back to a neutral dialogue voice, sounds fine). Emotional range from straight readers is tonal, not theatrical — these are audiobook narrators, not voice actors. F5 has run-to-run variance. And pronunciation has a long tail even with the lexicon.

Stack: F5-TTS, Orpheus 3B and Piper as alternate engines, llama.cpp, faster-whisper, lameenc/ffmpeg. Happy to answer anything, share the sourcing grids, or take suggestions for narrator #7.
