# Bookcaster

Turn a book into a **full-cast audiobook with music and sound effects**, and from there into a
**video**, from still illustrations up to fully animated shots. Everything runs on your own
hardware: there is no cloud service and no API key.

The repository holds two command-line tools:

- **`bookcaster`** is the studio. It keeps a library of books and runs the whole job in plain,
  non-interactive steps: find a book, pick chapters, cast it, record it, score it, film it.
- **`tts-reader`** is the voice engine layer underneath. It converts one text file to an MP3 or
  `.m4b` with one voice or a cast. Its simplest path (Piper) needs no GPU. Its full reference is in
  [docs/TTS_READER.md](docs/TTS_READER.md).

Sample audio: [single narrator](docs/demo/demo_single_narrator.mp3),
[full cast](docs/demo/demo_full_cast.mp3).

## How far a book can go

`bookcaster make <book> --level <level>` chooses how much is produced. Each level includes the
ones above it.

| Level | What you get |
|-------|--------------|
| `audio` | Each chapter read by a cast: a narrator plus a voice per character, with delivery cues. |
| `music` (default) | The same with a composed score, one cue per scene. |
| `pictures` | A video of illustrations that change with the story, with a slow pan and zoom and subtitles, over the audiobook. |
| `moments` | The picture video with a few key moments animated. |
| `animated` | A fully animated video: the narration is split into 2-5 second shots, and each is drawn and then animated. |

Sound effects (ambience beds and spot effects) are added by default at the video levels. Add
them to an audiobook with `--sfx` or `bookcaster sfx <book>`.

**Status.** The audio levels are the settled part. The `animated` level is work in progress. It
produces watchable chapters, but the pictures don't always agree from shot to shot about where
people stand and which way they face, and some shots need a redraw or a retake.

## What it needs

`tts-reader` with Piper runs on a CPU. Everything else needs GPUs and local model servers.

| Part | Model | Runs as |
|------|-------|---------|
| Voices | Fish Audio S2 Pro | `tools/fish_engine_server.py` (:8891) and the adapter `tools/fish_tts_server.py` (:5014) |
| Planning, casting, directing | Any OpenAI-compatible chat model | `BOOKCASTER_LLM_URL` (default `http://127.0.0.1:18100/v1/chat/completions`) |
| Music | ACE-Step 1.5 | its API server (:8001) |
| Sound effects | MOSS-SoundEffect v2.0 | `tools/sfx_gen.py`, started per job |
| Pictures and keyframes | Qwen-Image 2.1 with a 4-step Lightning LoRA | `tools/qwen_image.py`, started per job |
| Character identity | One LoRA per character, trained from the book's own reference sheets | `tools/charlora.py`, started per job |
| Animation | LTX-2.5 | started per job, on both cards |
| Checks | A vision model for pictures and takes, SyncNet for lip sync, Whisper for timings | started per job |

The reference machine has two Radeon RX 7900 XTX cards (24 GB each, ROCm) and an AMD Strix Halo
whose integrated GPU runs the language model in shared memory (see
[The reference machine](#the-reference-machine)). ffmpeg is needed for video and `.m4b`.

**Portability.** The audio path is configured through environment variables (see
[Configuration](#configuration)). The video path is not packaged for other machines yet: several
interpreter and model paths under `tools/` are the reference machine's own, and you would have to
edit them. Each model has its own licence. Check them before you publish what you make.

## Install

Requires Python 3.9 or later.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # installs the `bookcaster` and `tts-reader` commands
```

That is enough for `tts-reader` with Piper. For bookcaster, set up the model servers above, then
start them:

```bash
tools/fish_stack.sh up --music     # voices + music; the first start compiles for a few minutes
tools/fish_stack.sh up all         # the same, then the language model and the agent proxy
tools/fish_stack.sh status         # also: bookcaster services
tools/fish_stack.sh down all
```

The servers run as memory-capped systemd user units. `up all` starts the voice and music servers
before the language model, which pins about 90 GiB of shared memory on the reference machine, and
refuses a start that would not fit. The venv paths and GPU pins come from environment variables
listed in the script's header.

## Quick start

```bash
bookcaster search "sherlock holmes"          # find books on Project Gutenberg
bookcaster fetch 1661                        # download one into the library
bookcaster chapters adventures               # list its chapters (a unique part of the name is enough)
bookcaster pick adventures "speckled"        # choose what to record: numbers, ranges, title words, or all
bookcaster cast adventures                   # see the proposed cast
bookcaster cast adventures --do "Roylott should sound old and gravelly"
bookcaster audition adventures Roylott       # hear one of his lines
bookcaster storyboard adventures             # a cheap review pack before the long job
bookcaster estimate adventures               # rough machine time for each level
bookcaster make adventures --level music     # record in the background
bookcaster status                            # progress
```

To use your own text, make a folder in the library with the text saved as `book.txt`.

Books live in `$BOOKCASTER_HOME/<book>/` (default `~/audiobooks`):

```
book.txt, meta.json     the text and its title and author
chapters/               the chapters you picked
work/                   caches, the job log (work/job.log), and work/video/<chapter>/ for video
out/                    finished files: <book>_<chapter>_<date-time>.mp3 and .mp4, and out/storyboard/
```

`make`, `video`, `sfx` and `storyboard` run as background jobs, one per book. `bookcaster status`
reports them, and `bookcaster stop <book>` cancels one. Finished steps are cached, so a job that is
stopped or fails picks up where it left off.

## Commands

| Command | What it does |
|---------|--------------|
| `search`, `fetch`, `library` | Find a book on Project Gutenberg, download it, list the books on this machine. |
| `chapters`, `pick` | List a book's chapters and choose which to record. |
| `cast [--do "..."] [--fresh]` | Show the proposed cast, or direct it in plain English or with `Name = voice`. |
| `voices [--filter text] [--mine]` | The voice library. |
| `voice-add <recording> --name NAME`, `voice-remove` | Make a cast voice from your own recording, or delete one. |
| `audition` | Render one of a character's lines in their cast voice. |
| `storyboard` | A review pack before recording: the main voices, every character's look, a music sample and the first scenes in the art style. |
| `estimate` | Rough machine time for each level. |
| `make [--level ...] [--music-only] [--sfx] [--no-sfx]` | Record the picked chapters. |
| `music [style] [--sample] [--clear]` | Set or show the music style. `make --music-only` then redoes only the music. |
| `pace [value] [--who NAME]` | Listening speed, pitch kept (default 1.25), for the book or one voice. |
| `sfx` | Add sound effects to chapters already recorded. |
| `video [--level ...] [--flow ...]` | Make or remake the video of recorded chapters (see below). |
| `song <dir> --audio FILE --lyrics FILE` | Turn a song into a one-chapter book that `video` films as a music video (see below). |
| `art [style] [--sample] [--auto] [--clear]` | Set or show the picture style. |
| `period [text] [--clear]` | Set the story's time period. Every picture is drawn in it. |
| `look [--do "Name = description"] [--swap A B]` | Show or change how characters and places look. |
| `redraw <book> 12:30` | Redraw the picture on screen at that time, then rebuild the video. |
| `keyframes`, `sheets` | Preview the keyframe stage on a few frames; list unused character sheets. |
| `plan`, `assets`, `stage`, `review`, `approve` | The stages of the staged-3D video flows. |
| `status`, `stop` | Job progress; cancel a job. |
| `jellyfin` | Copy finished videos to a Jellyfin server. |
| `services` | Status of the voice, music and language model servers. |

Every command has `--help`.

## Video

```bash
bookcaster make  <book> --level animated                # record, then film, in one job
bookcaster video <book> --level animated                # film chapters that are already recorded
bookcaster video <book> --flow shots --keys-only        # draw the keyframes only and preview them
bookcaster video <book> --flow shots --from 1:00 --to 4:00 --render 1:06-2:06
```

`--level animated` uses the **shots flow** (`tools/shot_flow.py`):

1. **Plan.** The language model reads the chapter and fixes how each character and place looks,
   with reference sheets for each.
2. **Character LoRAs.** One is trained per character on both cards while the next step runs.
3. **Direct.** The language model, as director, splits the narration into shots of 2-5 seconds
   with one physical action each, working through the chapter in passages of up to 30 seconds.
4. **Keyframes.** Qwen-Image draws one opening picture per shot from the place's reference image,
   the characters' LoRAs and their looks. A vision model checks each picture.
5. **Animate.** LTX-2.5 animates each shot from its keyframe. Action shots use the two-stage
   model, quiet shots the distilled one, and talking shots are driven by the recorded voice.
   A vision model judges the takes and faulty ones are taken again. Talking shots are scored
   for lip sync with SyncNet and retaken, up to three takes, when they score low.
6. **Assemble.** The clips are laid under the narration, music and sound effects. The video
   model's own sound is not used.

Useful options for the shots flow:

- `--keys-only` draws and checks the keyframes and makes a preview video without animating.
- `--from` and `--to` limit the job to a stretch of the chapter. `--render a-b` directs that
  whole stretch but draws and animates only the window, which makes a short test cycle.
- `--review user` (or `agent`) holds each action line's takes on a review card so that a person
  (or an agent) picks one. After 10 minutes the vision judge's pick is used.

The other flows, chosen with `--flow`:

| Flow | What it is |
|------|------------|
| `2d` | The picture video, with animated moments at `--level moments`. The default below `animated`. |
| `3d` | Every moment rendered from a staged 3D scene. Needs `plan`, `assets`, `stage` and `review` first. |
| `mannequin` | The 3D staging used only as pose and depth maps for the keyframes, which are then animated. |

`3d` and `mannequin` came first and are kept, but new work goes into `shots`.

### Music videos

`bookcaster song` turns a recording and its lyrics into a one-chapter book, which the shots flow then films:

```
bookcaster song ~/audiobooks/my_song --audio song.flac --lyrics lyrics.txt \
    --direction "an 80s hair band playing in a chip factory"
bookcaster video ~/audiobooks/my_song --level animated --flow shots
```

- The lyrics file marks its sections with lines like `[Verse 1]` and `[Chorus]`.
- The song is used as it is. The finished video carries the original recording, and nothing is re-sung or
  remixed. The vocals are separated only to time the words and to drive the singers' lips.
- `--direction` is your own description of the video. Looks, outfits, places and shots follow it. Changing it
  plans the video again; with `--keep-looks` only the scenes and shots are planned again.
- `--singer` names who sings lead on screen. Without it the language model picks.
- `--tags` gives the recording's style notes, and `--suno clip.json` reads the title and tags from a Suno clip file.

## Changing the result

Most fixes are a plain-language change and a rebuild of only what it affects:

- **A voice:** `cast --do "..."`, check it with `audition`, then `make` again.
- **The music:** `music <book> solo piano, romantic era --sample`, then `make --music-only`.
- **The picture style or period:** `art` and `period`.
- **How someone looks:** `look <book> --do "Roylott = a man of about 60, clean-shaven, ..."`.
- **One picture:** `redraw <book> 12:30`.

## Your own voices

```bash
bookcaster voice-add me.m4a --name jim --gender Male --description "warm baritone"
```

Give it 20 seconds to a few minutes of clear reading in any format ffmpeg reads. The clearest
8-15 seconds become the voice's reference clip, and the voice can be cast at once.

`tools/mine_voice.py` builds voices from public-domain LibriVox recordings and rejects clips that
are flat or robotic. See [docs/VOICE_MINING.md](docs/VOICE_MINING.md).

## Jellyfin

Bookcaster can copy finished videos to a Jellyfin library over an SMB share and ask Jellyfin to
rescan. Set it up once in `~/.config/bookcaster/jellyfin` (mode 600):

```
JELLYFIN_URL=http://<server>:8096
JELLYFIN_KEY=<Dashboard -> API Keys>
SMB_SHARE=//<server>/bookcaster
SMB_USER=bookcaster
SMB_PASS=...
```

With that file in place a finished whole-chapter video is sent automatically.
`bookcaster jellyfin [book]` sends on request; `--tag v2` sends a new version beside the old one,
and `--part` sends a partial cut. Without the file nothing is sent.

## Driving it with an agent

Every command is non-interactive and long jobs run in the background, so a coding agent can run
the studio for you. Start the servers with `tools/fish_stack.sh up all`, open the agent in the
library folder with a short instruction file that lists the commands above, and ask in plain
language:

```
> I'd like to listen to The Speckled Band. Roylott should sound old and gravelly.
```

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `BOOKCASTER_HOME` | `~/audiobooks` | The library folder. |
| `BOOKCASTER_LLM_URL`, `BOOKCASTER_LLM_MODEL` | `http://127.0.0.1:18100/v1/chat/completions`, `flashnext-c2t8` | The chat model for planning, casting and directing. |
| `BOOKCASTER_FISH_URL` | `http://127.0.0.1:5014/v1/audio/speech` | The voice adapter. |
| `BOOKCASTER_ACE_URL` | `http://127.0.0.1:8001` | The music server. |
| `BOOKCASTER_PACE` | `1.25` | Default listening speed. |
| `BOOKCASTER_EXCLUDE_VOICES` | `ears_` | Voice-name prefixes left out of casting, comma-separated. |
| `BOOKCASTER_VIDEO_CODEC`, `BOOKCASTER_VIDEO_BITRATE` | `libopenh264`, `2500k` | Video encoding. |
| `BOOKCASTER_REVIEW` | `off` | Review cards for action takes: `off`, `agent` or `user`. |
| `BOOKCASTER_SFX_PYTHON`, `BOOKCASTER_WHISPER_PYTHON` | reference-machine paths | Interpreters for the sound-effect model and Whisper. |

`tools/fish_stack.sh` has its own variables for venvs and GPU pins, listed in its header.
`tts-reader` uses `TTS_READER_MODELS_DIR` for its Piper voice cache.

## The reference machine

Everything runs on one desktop. No cloud GPU is used.

| Part | What it is | What runs on it |
|------|------------|-----------------|
| CPU and memory | AMD Ryzen AI Max+ 395 (Strix Halo), 16 cores, about 123 GiB of usable RAM | The job itself, ffmpeg, assembly |
| Integrated GPU | Radeon 8060S. It uses system RAM as its memory; the kernel's limit for that is raised to about 124 GiB | The language model (about 90 GiB, pinned), the vision judge, keyframes for quiet shots |
| Two cards | Radeon RX 7900 XTX, 24 GB each, each on a PCIe 4.0 x4 link | Voices on one and music on the other while recording. LoRA training, keyframes and LTX while filming |
| Software | Fedora 44, PyTorch 2.11 on ROCm 7 nightly builds | |

### How the work is spread

Filming is the slow part, so the aim is that no device waits for another.

- The language model plans on the integrated GPU while both cards train the character LoRAs.
- LTX's transformer is split evenly over the two cards (24 blocks each), with three clips in
  flight. Each card does one piece of work at a time, and finished clips are decoded in pairs,
  one on each card.
- For prompt encoding, the text encoder runs on one card. The connector on the other card embeds
  each batch while the encoder runs the next.
- The cards draw keyframes and the integrated GPU judges each one as it lands. While the cards
  render the main shots, the integrated GPU draws the keyframes for the quiet shots.
- The director is one running conversation per chapter, so the language model reads only the new
  turn each time.

### What was tuned

Each figure was measured on this machine against the version before it.

| Change | Effect |
|--------|--------|
| LTX weights cast to fp8 ahead of time and read from the internal NVMe | Model ready in 18.7 s instead of 101.9 s |
| All guidance passes of a step in one batched call | Action clips: 16-20% less card time |
| Guidance on the first 60% of steps only, for talking shots | 13-17% less card time; the lip sync score stayed within 0.15 |
| The character LoRA folded into the weight as it is upcast | Quiet clips: 7.6% less card time |
| Paired decode, the even split, one job at a time per card | 7 talking clips: 463 s to 387 s |
| A Triton kernel for the video decoder's 3x3x3 convolution | A 177-frame decode: 21.0 s to 9.5 s, with half the peak memory |
| Three clips in flight under a full lock on the first card | 13% less wall time than two clips; 0 bad clips in 9 |
| Hand-written fused kernels in place of `torch.compile` | 446 s to 372 s on the same clips, and no 50-70 s of tracing at each start |
| Prompt encoding: one model load, real tokens only, sorted batches, the connector on the other card | A chapter's 221 prompts: about 331 s to 45 s |
| LoRA training under bf16 autocast with fused AdamW | 1.52 s to 1.33 s a step |
| The integrated GPU judges a LoRA's pictures while a card draws them | Judging ends 34 s after drawing instead of 4.5 min |
| Quiet-shot keyframes drawn on the integrated GPU during the render | 6.6% less time from keyframes to the end |
| The integrated GPU's host threads kept on the first group of CPU cores | The chip runs at its heat limit, so where those threads run sets its clock: 19.5 against 11.9 TFLOPS |

### Tried and not kept

- **`torch.compile` for LTX.** It beat plain eager code, but it traced for 50-70 s at every
  start. The fused kernels are faster still.
- **MIOpen on the cards.** Each new tensor shape cost about 10 minutes of tuning. It is off, and
  convolutions go through the project's own kernels or run in tiles.
- **fp8 weights in Qwen-Image.** They gave NaNs, so its stored weights are int8.
- **Parallel requests to the language model.** They gave no gain with the model in use.
- **hipBLASLt for prompt encoding.** It saved 6-8 s and changed the embeddings by about 0.5%.

## Tests

```bash
tools/tests/run_suite.sh          # the fast suite, CPU only
tools/tests/run_suite.sh full     # every test except the manual harnesses
```

The suite expects the project venv at `~/tts_reader/.venv`.

## License

MIT for the code in this repository. The models it runs have their own licences.
