# [Reddit draft — r/LocalLLaMA]

**Title:** I built a fully local audiobook engine that auto-casts characters and switches emotional delivery line-by-line — voices cloned from public-domain LibriVox narrators, running on a Strix Halo APU

---

**The demo**: the climax of *The Adventure of the Speckled Band* rendered two ways from the same text —
([single narrator](https://github.com/routhjim/bookcaster/raw/main/docs/demo/demo_single_narrator.mp3) · [full cast](https://github.com/routhjim/bookcaster/raw/main/docs/demo/demo_full_cast.mp3))

1. **Single narrator** — one cloned voice reading everything (F5-TTS's own contextual prosody, which is already surprisingly good)
2. **Full cast** — an LLM attributes every line of dialogue, four different narrators voice the characters (gender-matched, cast by an LLM from voice descriptors, user-overridable), and each line is synthesized from an *emotional reference clip* matching its register — the terrified client actually sounds terrified, the menacing stepfather actually sounds menacing

A scene that never existed as a single recording, assembled from four narrators who never met, all public domain, all local.

**The pipeline** — what actually happens when a book goes in:

```
book.txt / Gutenberg URL
 │
 ├─ 1. PREPROCESS      abbreviations→spoken, roman numerals, footnotes out,
 │                     pronunciation lexicon applied (LLM-built per book)
 ├─ 2. CHAPTERIZE      heading detection → per-chapter units (resumable)
 ├─ 3. SEGMENT         narration vs. quoted dialogue, line by line
 ├─ 4. ATTRIBUTE       LLM names each line's speaker (heuristics catch the
 │                     easy "said Ahab" cases free) → book.speakers.json
 ├─ 5. CAST            LLM builds character roster + suggests voices from
 │                     the library's descriptors; you override at a REPL
 │                     ("Starbuck is a grizzled old pirate") → voice map
 ├─ 6. TAG EMOTION     LLM assigns each line a register (tense/angry/...)
 │                     from textual cues → book.emotions.json
 └─ 7. SYNTHESIZE      per segment: {speaker → voice, register → that
                       narrator's matching reference clip} → F5-TTS clones
                       delivery; output shaped (per-register tempo, pause
                       band, ghost-speech gate) → chapter audio
 → 8. ASSEMBLE         tagged per-chapter MP3s + playlist, or one .m4b
                       with chapter marks, in Author/Title library layout
```

Steps 4–6 are LLM calls against any OpenAI-compatible endpoint (mine:
llama.cpp serving Qwen3.6-35B-A3B on the same APU) and every decision is
cached in sidecar files next to the book — the expensive passes happen once
per book ever; recasting or re-rendering afterwards is nearly instant.
Serving topology: llama.cpp on :8080 (all LLM passes), the F5-TTS palette
server on :5010 (synthesis + voice registry), optional Orpheus on :5005.

The **voice library** feeding step 7 is built by a separate offline
pipeline (docs/VOICE_MINING.md): LibriVox chapter → faster-whisper
alignment → LLM locates emotional passages → quality gates → human
audition → an 8-register palette per narrator.

**Beyond the pipeline** (github.com/routhjim/bookcaster):

- **Adding narrators is repeatable, not artisanal**: write a small JSON grid mapping emotions to chapters where the text forces that register ("Blind Pew's approach" → tense), run `tools/mine_voice.py`, audition the staged candidates, promote your picks — full workflow in [docs/VOICE_MINING.md](https://github.com/routhjim/bookcaster/blob/main/docs/VOICE_MINING.md). A new 8-register narrator: ~1 hour of machine time, ~15 minutes of listening
- Three interchangeable engines behind one interface: Piper (CPU, 13.7× real-time, robotic), Orpheus 3B (GPU, ~0.7×, natural + inline `<sigh>`-style tags), F5-TTS (GPU, cloning + palettes — the headliner)
- Long renders are resumable at every chapter; an LLM `abridge` mode condenses books while preserving all dialogue verbatim

**Hardware + numbers** (Ryzen AI MAX+ 395, 8060S iGPU, 128GB unified, Vulkan for llama.cpp / TheRock torch for F5 — the stock pytorch ROCm wheels segfault on gfx1151, AMD's own wheel index works):

- F5 renders at 0.53× real-time (torch.compile bought 1.65×; bf16 was already on; concurrency measured useless — one request saturates the APU; 16-step NFE is a measured 2.05× if you accept slightly-different renders)
- A ~10-hour multi-voice audiobook ≈ overnight-plus render, ~$0.35 of electricity. The same book through a cloud TTS API at per-character pricing: $120–360, per render, and I re-rendered *constantly* while tuning
- LLM passes on the same box: full-novel speaker attribution ~10 min, emotion tagging similar, all cached

**Hard-won lessons** (the failure museum):

- F5 clones *pace* above all: an "excited" reference that's merely fast reading produces a fast voice, not an excited one — pace-gate your clips against the narrator's own baseline
- Never mine reference clips containing quoted dialogue: narrators perform character voices (falsetto princesses included), and you'll clone the *performance*, not the person. LibriVox group recordings will burn you the same way — verify solo readers per file
- Ref clips cut mid-sentence make F5 hallucinate quiet "ghost" completions into your audio; ditto very short generation texts ("Always.") — fixed with sentence-final clip cuts + an edge energy gate
- Whisper transcription drops quotation marks, so text-based dialogue filters can't catch everything; the human audition pass is not optional
- Character-level TTS spells its way into pronunciation errors ("clanging" → "clan-jing"); a per-book LLM-built respelling lexicon fixes the long tail

**Honest limitations:** attribution is ~95% not 100% (unattributed lines fall back to a neutral dialogue voice — sounds fine); the emotional shifts from straight readers are tonal, not theatrical; F5 output has run-to-run variance; the NPU sits idle (no Linux VitisAI EP exists — verified against Ryzen AI 1.8.0, wheels are win_amd64 only).

Everything's MIT, built on F5-TTS/Orpheus/Piper/llama.cpp/faster-whisper. Voices are LibriVox public domain end-to-end — no likeness games.
