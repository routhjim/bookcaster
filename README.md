# TTS Reader

Turn plain **text files into spoken-word MP3s**, entirely on your own machine.

It uses [Piper](https://github.com/OHF-Voice/piper1-gpl) — a fast, fully local
neural text-to-speech engine — so there is **no cloud service** and **no API
key**. Pick from several voices, adjust the speaking rate, and get an `.mp3`
out the other side. MP3 encoding is done in-process with `lameenc`, so you
**don't need ffmpeg installed**.

---

## Features

- 🎙️ **Several voices** out of the box — US/UK English plus a few other
  languages — and you can use any voice from the Piper collection.
- 🔒 **100% local synthesis.** After a voice model is downloaded once, it runs
  offline on CPU.
- 🎧 **Direct MP3 output** (no ffmpeg required) with adjustable bitrate.
- ⏩ **Speed & volume control.**
- 📚 **Batch mode** — convert many text files in one command.

## Install

Requires Python 3.9+.

```bash
# from the project root
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> The **first time** you use a given voice, its model (a few tens of MB) is
> downloaded from Hugging Face and cached locally. Every run after that is
> fully offline.

## Usage

List the available voices:

```bash
python -m tts_reader voices
```

Convert a file with the default voice (`amy`):

```bash
python -m tts_reader convert notes.txt
# -> notes.mp3
```

Try the included sample:

```bash
python -m tts_reader convert examples/sample.txt -o hello.mp3 --voice ryan
```

Read straight from a URL (Project Gutenberg license boilerplate is trimmed
automatically; pass `--no-strip-gutenberg` to keep it):

```bash
python -m tts_reader convert https://www.gutenberg.org/cache/epub/2701/pg2701.txt \
    -o moby_dick.mp3 --voice ryan
```

> Note: full-length books are **hours** of audio and take a long time to
> synthesize. Try a single chapter first.

### Chapters (browseable audiobooks)

TTS Reader can detect chapters (`CHAPTER 1`, `Part II`, `Prologue`, …) and turn
a book into something you can navigate.

Preview the detected chapters without synthesizing anything:

```bash
python -m tts_reader convert moby.txt --list-chapters
```

By default (`--chapters auto`), when multiple chapters are found the output MP3
gets **embedded chapter markers** (ID3 `CHAP`/`CTOC`) so you can jump between
chapters in a chapter-aware player (VLC, most podcast apps, Foobar2000, …):

```bash
python -m tts_reader convert moby.txt -o moby.mp3 --voice ryan
```

Prefer one file per chapter plus a playlist? Use split mode — this works in
*any* player as a track list:

```bash
python -m tts_reader convert moby.txt -o moby_audiobook/ --chapters split
# moby_audiobook/001_....mp3, 002_....mp3, ... + playlist.m3u
```

Chapter options:

| Option              | Description                                                            |
|---------------------|-----------------------------------------------------------------------|
| `--chapters auto`   | Embed markers when chapters are detected, else a plain MP3 (default). |
| `--chapters embed`  | Force a single MP3 with jump-to-chapter markers.                       |
| `--chapters split`  | One MP3 per chapter + `playlist.m3u` in a folder.                      |
| `--chapters off`    | Ignore chapters; one continuous MP3.                                   |
| `--list-chapters`   | Print detected chapters and exit (no audio).                          |
| `--chapter-regex`   | Custom per-line regex marking where chapters begin.                   |

> Not every player supports embedded MP3 chapters. If yours doesn't show them,
> use `--chapters split` for universally-browseable per-chapter files. (The
> classic audiobook `.m4b` format has the broadest chapter support but requires
> AAC/ffmpeg, which this tool intentionally avoids.)

Pick a voice, speed, and bitrate:

```bash
python -m tts_reader convert chapter1.txt -o audiobook/ch1.mp3 \
    --voice lessac --speed 1.1 --bitrate 192
```

Batch-convert a folder of text files (one MP3 each) into `out/`:

```bash
python -m tts_reader convert *.txt -o out/
```

### Options (`convert`)

| Option          | Default | Description                                                        |
|-----------------|---------|--------------------------------------------------------------------|
| `inputs`        | —       | One or more `.txt` files to read.                                  |
| `-o, --output`  | `<name>.mp3` | Output MP3 file (single input) or output directory (many inputs). |
| `-v, --voice`   | `amy`   | Voice alias or full Piper key.                                     |
| `--speed`       | `1.0`   | Rate multiplier: `1.2` faster, `0.85` slower.                      |
| `--volume`      | `1.0`   | Volume multiplier.                                                 |
| `--bitrate`     | `128`   | MP3 bitrate in kbps (try `192` / `256`).                           |
| `--models-dir`  | data dir | Where to cache downloaded voice models.                           |
| `-V, --verbose` | off     | Show Piper's low-level logs (e.g. harmless missing-phoneme notes). |

## Higher quality: the Orpheus engine (GPU)

Piper is fast but can sound flat. **Orpheus** (Canopy Labs, Apache-2.0) is a
Llama-3B-based TTS model that sounds dramatically more natural and supports
inline emotion tags. It runs locally on your GPU; TTS Reader talks to it over
an OpenAI-compatible endpoint, so all the chapter/MP3 features work unchanged.

```bash
python -m tts_reader convert book.txt -o book.mp3 --engine orpheus --voice leo
```

Orpheus voices: `tara` (default), `leah`, `jess`, `leo`, `dan`, `mia`, `zac`,
`zoe`. Emotion tags can be embedded inline in the text:

```text
Well, that went about as well as expected. <sigh> Let's try again. <laugh>
```

### Setting up the Orpheus server

Orpheus has two parts: the **LLM** (served by llama.cpp) and the
**Orpheus-FastAPI** front end that decodes tokens to audio (SNAC) and exposes
`/v1/audio/speech`.

1. **Build llama.cpp with GPU support.** On AMD (e.g. Strix Halo / Radeon
   8060S) the **Vulkan** backend is usually the smoothest path on Linux and
   avoids a ROCm install:

   ```bash
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
   ```

2. **Get an Orpheus GGUF** (Q4_K_M is a good balance; Q8_0 is highest quality)
   and serve it — note the port, which Orpheus-FastAPI expects on 5006:

   ```bash
   ./build/bin/llama-server -m orpheus-3b-0.1-ft-q4_k_m.gguf \
       --port 5006 --ctx-size 8192 --n-predict 8192 \
       --rope-scaling linear -ngl 99
   ```

   `-ngl 99` offloads all layers to the GPU.

3. **Run Orpheus-FastAPI** pointing at that llama.cpp server:

   ```bash
   git clone https://github.com/Lex-au/Orpheus-FastAPI && cd Orpheus-FastAPI
   pip install -r requirements.txt
   export ORPHEUS_API_URL=http://127.0.0.1:5006/v1/completions
   python app.py            # serves on port 5005
   ```

4. **Point TTS Reader at it** (this is the default URL, so you can omit it):

   ```bash
   python -m tts_reader convert book.txt -o book.mp3 \
       --engine orpheus --voice leo \
       --orpheus-url http://127.0.0.1:5005/v1/audio/speech
   ```

Orpheus options:

| Option                  | Default                                    | Description                             |
|-------------------------|--------------------------------------------|-----------------------------------------|
| `--engine orpheus`      | `piper`                                    | Use the Orpheus backend.                |
| `--orpheus-url`         | `http://127.0.0.1:5005/v1/audio/speech`    | Server endpoint.                        |
| `--orpheus-model`       | `orpheus`                                  | Model name sent in the request.         |
| `--orpheus-chunk-chars` | `400`                                      | Split long text into chunks this size.  |

> **Why chunking?** Orpheus generates audio *as tokens*, so a whole chapter in
> one request can exceed its token limit. TTS Reader splits long text on
> sentence boundaries and stitches the audio back together automatically.

> **Speed expectations:** Orpheus is far heavier than Piper — expect closer to
> real-time generation rather than Piper's ~13×. It's the right choice for
> quality; for a 20-hour book, Piper is still the pragmatic option.

## Voice cloning: the F5-TTS engine (GPU)

**F5-TTS** clones any voice from a ~10-second reference clip — there are no
baked-in voices; a "voice" is just `<name>.wav` (the clip) plus `<name>.txt`
(its exact transcript) dropped into the server's voices directory
(`~/.local/share/tts_reader/f5_voices/`). LibriVox is a goldmine of
public-domain narrator clips. An optional `voices.json` there adds
`{"name": {"gender": "Male", "description": "..."}}` metadata that the
`cast` command uses for gender-matched casting.

```bash
tts-reader convert book.txt -o book.mp3 --engine f5 --voice my_narrator
tts-reader cast book.txt --engine f5          # roster comes from the server
```

The engine talks to a small local server (same OpenAI-style
`/v1/audio/speech` API as Orpheus, port 5010) that wraps F5-TTS on your GPU.
On AMD (gfx1151/Strix Halo), install PyTorch from TheRock's wheel index —
the generic pytorch.org ROCm wheels crash on this hardware.

## Multi-voice narration (character casting)

TTS Reader can give the narrator and each speaking character **their own
voice**. It splits prose into narration and quoted dialogue, works out who is
speaking, and routes each line to the voice you assigned.

First, see who talks in your book:

```bash
python -m tts_reader characters moby.txt
```

This prints the speakers found (with quote counts and an example line) plus a
ready-to-edit `--voice-map` suggestion. Attribution uses built-in heuristics
(`"...," said Ahab` and friends) — free and offline. To resolve the quotes the
heuristics can't attribute (unattributed back-and-forth exchanges, "said he",
…), point it at any **OpenAI-compatible LLM endpoint**, such as a local
llama.cpp server:

```bash
python -m tts_reader characters moby.txt \
    --llm-url http://127.0.0.1:8080/v1/chat/completions
```

Then convert with a cast:

```bash
python -m tts_reader convert moby.txt -o moby.mp3 --engine orpheus \
    --voice-map "narrator=leo,dialogue=tara,Ahab=zac,Starbuck=dan,Stubb=mia" \
    --llm-url http://127.0.0.1:8080/v1/chat/completions
```

Voice-map keys: `narrator` (required) reads everything outside quotes;
`dialogue` (optional) is the fallback for quotes whose speaker is unknown or
unmapped (defaults to the narrator's voice); every other key is a character
name as shown by `characters` (case-insensitive).

Casting works with both engines. With Piper, pick voices from the same quality
tier (all `-medium`/`-high`) — mixed sample rates are rejected. With Orpheus,
any mix of its eight voices works.

**Voice variants**: any voice can carry a delivery-rate variant —
`Ahab=zac@0.92` speaks zac's voice slightly slower, `Stubb=zac@1.08` slightly
faster — so one voice can play several characters distinguishably. The `cast`
command uses this automatically: characters are cast in priority order (most
lines first), voices are matched to each character's gender, and when a
gender's roster runs out it hands out rate variants rather than casting
across gender.

A short beat of silence (250 ms) is inserted whenever the speaking voice
changes, so switches don't feel abrupt; tune it with `--cast-pause-ms`
(`0` disables).

**Emotion tags (`--emote`, Orpheus only)**: an LLM reads each line's
surrounding narration ("he laughed", "she sighed") and injects a sparse
inline tag (`<laugh>`, `<sigh>`, `<gasp>`, ...) where the text clearly calls
for one — most lines get none, and decisions are cached per book
(`<book>.emotions.json`). Works with `--voice-map` or single-voice; needs
`--llm-url`. Orpheus's tag vocabulary covers laughter, sighs, gasps and the
like — it can't do "coldly" or "furiously", so tags add texture at emotional
beats rather than full performance direction.

> Attribution is good but not perfect — expect an occasional line in the wrong
> voice, especially in older prose. The LLM pass helps a lot; without it,
> unattributed quotes simply use the `dialogue` voice, which always sounds
> reasonable.

**Attribution is cached.** LLM speaker attributions are saved to a sidecar
file next to the book (`moby.txt.speakers.json`) and reused by `characters`,
`cast`, and `convert` alike — the expensive first pass over a book happens
once, and recasting or re-converting afterwards is nearly instant. Delete the
sidecar to start fresh.

### Interactive casting (`cast`)

Rather than hand-writing a voice map, let the tool interview the book and you
direct the cast in plain language:

```text
$ tts-reader cast moby.txt -o moby_audiobook/ --chapters split
Parsing dialogue...
Asking the LLM to describe each character and suggest voices...

Proposed cast:
  narrator  dan   (everything outside quotes)
  (other)   zoe   (unattributed/minor speakers)
  Ahab      leo   A tormented, obsessive captain ... deep authority
  Starbuck  tara  The prudent, loyal first mate ...
  Stubb     zac   A sardonic, observant second mate ...

cast> Starbuck is an old, grizzled, crusty pirate
cast> Ahab is refined and commanding, almost genteel
cast> Stubb = dan
cast> convert
```

The session needs an OpenAI-compatible LLM endpoint (default
`http://127.0.0.1:8080/v1/chat/completions`, llama.cpp's standard port;
change with `--llm-url`) for role descriptions, voice suggestions, and
plain-language directing. Without one it still works — you just assign
voices directly with `Name = voice`. `map` prints the resulting
`--voice-map` string instead of converting, so you can reuse it later.

### Abridged audiobooks (`abridge`)

For long books, an LLM can condense the text before synthesis — keeping every
dialogue exchange verbatim (so casting still works), trimming only very long
monologues, and compressing narration to key events, imagery, and themes in
the book's own style:

```bash
tts-reader abridge moby.txt --level medium       # -> moby_abridged.txt
tts-reader cast moby_abridged.txt -o moby_short/ --chapters split
```

Levels: `light` (~60% of narration kept), `medium` (~40%), `heavy` (~25%) —
dialogue always survives, so dialogue-heavy chapters stay closer to full
length. The output is a plain text file: read it, edit it, then cast/convert
it like any book. Progress is checkpointed, so an interrupted run resumes
where it stopped.

## Speech-friendly text preprocessing

Before synthesis, text is cleaned up so it *reads* well (on by default, skip
with `--no-preprocess`):

- Abbreviations are spoken: `Mr.` → "Mister", `Dr.` → "Doctor", `&c.`/`etc.` →
  "et cetera", `No. 3` → "Number 3", `e.g.` → "for example", …
- Roman-numeral headings become numbers: `CHAPTER XLI` → "Chapter 41".
- SHOUTED HEADINGS are softened so they aren't spelled out letter-by-letter
  (true acronyms without vowels are left alone).
- Footnote markers (`[3]`), `_emphasis_` underscores, and `* * *` section
  rules are removed; hyphen-split line-wrapped words are rejoined.

## Voices

Run `python -m tts_reader voices` for the full table. The catalog includes:

| Alias        | Voice                     | Notes                         |
|--------------|---------------------------|-------------------------------|
| `amy`        | US English, female        | Default; warm and natural     |
| `ryan`       | US English, male          | Confident narrator (high-q)   |
| `lessac`     | US English, neutral       | Crisp, great for documents    |
| `joe`        | US English, male          | Conversational                |
| `kathleen`   | US English, female        | Light and fast                |
| `hfc_female` | US English, female        | Balanced/professional         |
| `hfc_male`   | US English, male          | Balanced/professional         |
| `alan`       | UK English, male          | Calm British                  |
| `jenny`      | UK English, female        | Friendly British              |
| `alba`       | UK English, female        | Scottish accent               |
| `thorsten`   | German, male              |                               |
| `siwis`      | French, female            |                               |
| `davefx`     | Spanish (Spain), male     |                               |

You can also pass any voice key from the
[Piper voices collection](https://huggingface.co/rhasspy/piper-voices)
directly, e.g. `--voice en_US-kusal-medium`.

## Use as a library

```python
from tts_reader import PiperEngine

engine = PiperEngine("ryan")          # alias or full Piper key
engine.synthesize_to_mp3(
    "Hello from a local text-to-speech engine.",
    "hello.mp3",
    speed=1.0,
    bitrate=192,
)
```

## How it works

1. Your text file is read (UTF-8).
2. Piper synthesizes 16-bit PCM audio locally using the selected neural voice
   model (downloaded and cached on first use).
3. The audio is encoded to MP3 in-process with `lameenc`.

## Configuration

- `TTS_READER_MODELS_DIR` — override where voice models are cached
  (default: `~/.local/share/tts_reader/voices`).

## Notes & limitations

- The first download of each voice needs internet access to
  `huggingface.co`; synthesis afterwards is offline.
- Piper focuses on natural single-speaker narration; it is not designed for
  fine-grained SSML/emotion control.

## License

MIT
