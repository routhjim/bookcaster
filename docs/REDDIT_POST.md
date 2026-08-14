# [Reddit draft — r/LocalLLaMA]

**Title:** I built a fully local audiobook engine that auto-casts characters and switches emotional delivery line-by-line — voices cloned from public-domain LibriVox narrators, running on a Strix Halo APU

---

**The demo** (attach both clips): the climax of *The Adventure of the Speckled Band* rendered two ways from the same text —

1. **Single narrator** — one cloned voice reading everything (F5-TTS's own contextual prosody, which is already surprisingly good)
2. **Full cast** — an LLM attributes every line of dialogue, four different narrators voice the characters (gender-matched, cast by an LLM from voice descriptors, user-overridable), and each line is synthesized from an *emotional reference clip* matching its register — the terrified client actually sounds terrified, the menacing stepfather actually sounds menacing

A scene that never existed as a single recording, assembled from four narrators who never met, all public domain, all local.

**What it does** (github.com/routhjim/bookcaster):

- Text/URL/Gutenberg in → chaptered MP3s or proper `.m4b` audiobooks (chapter marks, resume, `Author/Title/` library layout for Audiobookshelf) out
- **Interactive casting**: `bookcaster cast book.txt` — an LLM reads the book, describes each character's role, proposes voices; you direct recasts in plain English at a REPL ("Starbuck is an old, grizzled, crusty pirate") and it recasts
- **Emotional palettes**: each library voice is 8 reference clips of the *same* narrator (neutral/warm/tense/angry/sad/excited/cold/surprised), semi-automatically mined from their LibriVox catalog — whisper alignment finds candidate passages, an LLM picks emotional spans, quality gates (SNR, clipping, pace-vs-baseline, no-dialogue, sentence-final endings) filter them, human ear makes the final call
- Speaker attribution, emotion tagging, and pronunciation-lexicon building are LLM passes against any OpenAI-compatible endpoint (I use llama.cpp + Qwen3.6-35B-A3B locally), all cached per book in sidecar files — a full novel's attribution costs one pass, ever
- Three engines: Piper (CPU, 13.7× real-time, robotic), Orpheus 3B (GPU, ~0.7×, natural + inline emotion tags), F5-TTS (GPU, voice cloning + the palette system)

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
