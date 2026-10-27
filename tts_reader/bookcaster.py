"""bookcaster: plain, non-interactive steps for turning a public-domain book
into a cast, scored audiobook. Built for an agent (or a person) to drive:

    bookcaster search "sherlock holmes"        find books on Project Gutenberg
    bookcaster fetch 1661                      download one into the library
    bookcaster chapters <book>                 list its chapters
    bookcaster pick <book> 8 | "speckled"      choose what to record
    bookcaster cast <book> [--do "..."]        propose / direct the cast
    bookcaster voices [--filter old] [--mine]  the voice library
    bookcaster voice-add <recording> --name N  make a voice from your own recording
    bookcaster voice-remove <name>             delete one of your recorded voices
    bookcaster audition <book> Holmes          hear a character's voice
    bookcaster music <book> ["style"]          set / show the music style
    bookcaster pace <book> [1.25]              set / show the listening speed (pitch kept)
    bookcaster art <book> ["style"]            set / show the picture style (--sample)
    bookcaster make <book> [--level L]         record it (background job); L, cheapest first:
                                               audio | music (default) | pictures | moments | animated
    bookcaster video <book> [--level L]        picture/animated video of what's recorded
    bookcaster estimate <book>                 rough box time for each level
    bookcaster sfx <book>                      add sound effects to recorded chapters (job)
    bookcaster storyboard <book>               review pack before recording: voices, looks,
                                               music, art (out/storyboard/; background job)
    bookcaster look <book> [--do "N = desc"]   show / change how a character looks
    bookcaster redraw <book> 12:30 [p021]      redraw pictures you don't like
                    [--music-only]             redo just the music, keep the voices
    bookcaster jellyfin [<book>]               copy finished videos to the Jellyfin server
    bookcaster status [<book>]                 what's running / done
    bookcaster stop <book>                     cancel a job
    bookcaster library                         books on this machine
    bookcaster services                        the voice / music / LLM servers

A book lives in $BOOKCASTER_HOME/<slug>/ (default ~/audiobooks): book.txt
(the download), meta.json, chapters/ (what was picked), book.* caches (cast,
speaker attribution, delivery cues), work/ (renders, music) and out/ (the
finished MP3s: the ones to download).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

HOME = Path(os.environ.get("BOOKCASTER_HOME", Path.home() / "audiobooks")).expanduser()
LLM_URL = os.environ.get("BOOKCASTER_LLM_URL", "http://127.0.0.1:18100/v1/chat/completions")
LLM_MODEL = os.environ.get("BOOKCASTER_LLM_MODEL", "flashnext-c2t8")
FISH_URL = os.environ.get("BOOKCASTER_FISH_URL", "http://127.0.0.1:5014/v1/audio/speech")
ACE_URL = os.environ.get("BOOKCASTER_ACE_URL", "http://127.0.0.1:8001")
STACK = Path(__file__).resolve().parents[1] / "tools" / "fish_stack.sh"
TTS = [sys.executable, "-m", "tts_reader"]
# add_voice.py needs faster-whisper, which lives in the Fish adapter's venv.
WHISPER_PY = Path(os.environ.get("BOOKCASTER_WHISPER_PYTHON",
                                 Path.home() / "tts-ab" / ".venv-qwen" / "bin" / "python"))
VOICES_DIR = Path(os.environ.get("F5_VOICES_DIR")
                  or Path.home() / ".local" / "share" / "tts_reader" / "f5_voices").expanduser()
USER_SOURCE = "user recording"
CHARS_PER_SEC = 16.0          # Fish narration pace, for time estimates
# What a recording includes, cheapest first; each level adds to the one before:
# voices; + a score; + the still-picture video; + animated key moments;
# fully animated (lip-synced dialogue, moving narration scenes).
LEVELS = ["audio", "music", "pictures", "moments", "animated"]


def at_least(level: str, floor: str) -> bool:
    return LEVELS.index(level) >= LEVELS.index(floor)
UA = {"User-Agent": "bookcaster/0.1 (personal audiobook tool)"}


# --- small helpers -------------------------------------------------------------

def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "_", s).strip("_")[:60] or "book"


def book_dir(name: str) -> Path:
    """A book by slug, by path, or by a unique part of its slug."""
    p = Path(name).expanduser()
    if p.is_dir() and (p / "book.txt").exists():
        return p.resolve()
    if (HOME / name / "book.txt").exists():
        return HOME / name
    hits = [d for d in HOME.glob("*") if (d / "book.txt").exists() and slugify(name) in d.name]
    if len(hits) == 1:
        return hits[0]
    if hits:
        sys.exit(f"'{name}' matches several books: {', '.join(d.name for d in hits)}")
    sys.exit(f"no book '{name}' in {HOME} — `bookcaster library` lists them")


def minutes(chars: int) -> float:
    return chars / CHARS_PER_SEC / 60


def http_ok(url: str) -> bool:
    try:
        with urlopen(url, timeout=5):
            return True
    except Exception:
        return False


def tts(*args, capture=False, log=None) -> subprocess.CompletedProcess:
    cmd = [*TTS, *map(str, args)]
    if log is not None:
        with open(log, "a") as fh:
            fh.write(f"\n$ {' '.join(cmd[2:])}\n")
            fh.flush()
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    return subprocess.run(cmd, capture_output=capture, text=True)


def stem(book: Path) -> Path:
    """Where the book's shared caches live: book.speakers.json, book.cast.json..."""
    return book / "book"


def chapter_files(book: Path) -> list[Path]:
    return sorted((book / "chapters").glob("[0-9][0-9]_*.txt"))


# --- finding and fetching books -----------------------------------------------

def search(query: str, limit: int = 12) -> list[dict]:
    """Project Gutenberg's own catalog search (most-downloaded first)."""
    url = f"https://www.gutenberg.org/ebooks/search.opds/?query={quote_plus(query)}"
    with urlopen(Request(url, headers=UA), timeout=30) as resp:
        root = ET.fromstring(resp.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        m = re.search(r"/ebooks/(\d+)", e.findtext("a:id", "", ns))
        if not m:
            continue
        out.append({"id": int(m.group(1)), "title": e.findtext("a:title", "", ns).strip(),
                    "author": (e.findtext("a:content", "", ns) or "").strip()})
    return out[:limit]


def fetch(book_id: int, name: str | None = None) -> Path:
    """Download a Gutenberg book's plain text into the library."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    with urlopen(Request(url, headers=UA), timeout=120) as resp:
        raw = resp.read().decode("utf-8", "replace")
    title = re.search(r"^Title:\s*(.+)$", raw, re.M)
    author = re.search(r"^Author:\s*(.+)$", raw, re.M)
    title = title.group(1).strip() if title else f"Gutenberg {book_id}"
    author = author.group(1).strip() if author else ""
    d = HOME / (name or slugify(title))
    d.mkdir(parents=True, exist_ok=True)
    (d / "book.txt").write_text(raw, encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({
        "title": title, "author": author, "gutenberg_id": book_id, "source": url,
        "license": "Project Gutenberg (public domain in the USA); see the license text in book.txt",
        "fetched": time.strftime("%Y-%m-%d")}, indent=1))
    return d


def load_book_text(book: Path) -> str:
    from .cli import _load_text

    return _load_text(str(book / "book.txt"), strip_gutenberg=True)


def chapters_of(book: Path):
    from .chapters import DEFAULT_HEADING, detect_chapters

    return detect_chapters(load_book_text(book), pattern=DEFAULT_HEADING)


def pick(book: Path, selectors: list[str]) -> list[Path]:
    """Write the chosen chapters to chapters/NN_<title>.txt (and the whole
    selection to selection.txt, which the cast is built from)."""
    chs = chapters_of(book)
    chosen: list[int] = []
    for sel in selectors:
        sel = sel.strip()
        if sel.lower() in ("all", "*"):
            chosen += range(len(chs))
        elif re.fullmatch(r"\d+(-\d+)?", sel):
            a, _, b = sel.partition("-")
            chosen += range(int(a) - 1, int(b or a))
        else:
            hits = [i for i, c in enumerate(chs) if sel.lower() in c.title.lower()]
            if not hits:
                sys.exit(f"no chapter title contains '{sel}' — see `bookcaster chapters`")
            chosen += hits
    chosen = sorted({i for i in chosen if 0 <= i < len(chs)})
    if not chosen:
        sys.exit("nothing picked")
    out = book / "chapters"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    files = []
    for i in chosen:
        c = chs[i]
        f = out / f"{i + 1:02d}_{slugify(c.title)[:40]}.txt"
        f.write_text(c.title + "\n\n" + c.text.strip() + "\n", encoding="utf-8")
        files.append(f)
    (book / "selection.txt").write_text(
        "\n\n".join(f.read_text(encoding="utf-8") for f in files), encoding="utf-8")
    return files


# --- cast -----------------------------------------------------------------------

# Voice sources kept out of casting (comma-separated, matched in name or
# description). EARS: declined by the user (its emotion clips need Qwen).
EXCLUDE = [x for x in os.environ.get("BOOKCASTER_EXCLUDE_VOICES", "ears_").split(",") if x]


def cast_args(book: Path) -> list:
    args = ["cast", book / "selection.txt", "--cache-stem", stem(book),
            "--llm-url", LLM_URL, "--llm-model", LLM_MODEL, "--no-play"]
    for x in EXCLUDE:
        args += ["--exclude-voices", x]
    return args


def voice_map(book: Path) -> str:
    r = tts(*cast_args(book), "--do", "map", capture=True)
    m = re.search(r'--voice-map "([^"]+)"', r.stdout or "")
    if not m:
        sys.exit("couldn't read the cast — run `bookcaster cast <book>` first\n"
                 + (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:])
    return m.group(1)


def speaker_lines(book: Path) -> dict[str, list[str]]:
    """Each character's quotes, from the cached attribution."""
    from .speakers import attribute_segments, segment_dialogue
    from .textprep import prepare_for_speech

    text = prepare_for_speech((book / "selection.txt").read_text(encoding="utf-8"))
    segs = segment_dialogue(text)
    attribute_segments(segs, attributor=None, cache_path=Path(str(stem(book)) + ".speakers.json"),
                       log=lambda *a, **k: None)
    lines: dict[str, list[str]] = {}
    for s in segs:
        if s.kind == "quote" and s.speaker:
            lines.setdefault(s.speaker, []).append(s.text.strip())
    return lines


def audition(book: Path, who: str, text: str | None = None) -> Path:
    vm = dict(kv.split("=", 1) for kv in voice_map(book).split(","))
    key = next((k for k in vm if k.lower() == who.lower()), None) or next(
        (k for k in vm if who.lower() in k.lower()), None)
    if key is None:
        sys.exit(f"'{who}' isn't in the cast: {', '.join(vm)}")
    voice = vm[key]
    if not text:
        quotes = speaker_lines(book).get(key) or []
        # A whole sentence of theirs, not a fragment cut by a speech tag.
        good = sorted((q for q in quotes if 40 <= len(q) <= 260 and q[:1].isupper()),
                      key=len, reverse=True)
        text = (good or quotes or ["I have nothing to say."])[0]
    body = json.dumps({"input": text, "voice": voice, "model": "fish"}).encode()
    with urlopen(Request(FISH_URL, body, {"Content-Type": "application/json"}), timeout=300) as r:
        wav = r.read()
    out = book / "out" / "auditions"
    out.mkdir(parents=True, exist_ok=True)
    mp3 = out / f"{slugify(key)}.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "-", "-b:a", "128k", str(mp3)],
                   input=wav, check=True)
    print(f"{key} ({voice}): \"{text}\"")
    return mp3


# --- your own voices -------------------------------------------------------------

TEST_LINE = ("It was late in the evening when the letter came. I read it twice, "
             "and then I laughed out loud! Who on earth, I wondered, would send such a thing?")


def say(text: str, voice: str, mp3: Path) -> Path:
    body = json.dumps({"input": text, "voice": voice, "model": "fish"}).encode()
    with urlopen(Request(FISH_URL, body, {"Content-Type": "application/json"}), timeout=300) as r:
        wav = r.read()
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "-", "-b:a", "128k", str(mp3)],
                   input=wav, check=True)
    return mp3


def voice_add(recording: Path, name: str, gender: str, description: str,
              accent: str, replace: bool) -> int:
    if not recording.exists():
        sys.exit(f"no such file: {recording}")
    r = subprocess.run([str(WHISPER_PY), "-I", str(STACK.parent / "add_voice.py"), str(recording),
                        "--name", name, "--gender", gender, "--description", description,
                        "--accent", accent] + (["--replace"] if replace else []),
                       capture_output=True, text=True, env={**os.environ, "HIP_VISIBLE_DEVICES": ""})
    if r.returncode:
        sys.exit((r.stderr.strip().splitlines() or ["adding the voice failed"])[-1])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"added voice '{info['name']}': {info['clip_seconds']} s taken from "
          f"{info['from'][0]}-{info['from'][1]} s of your {info['recording_seconds']} s recording")
    print(f"  it says: \"{info['text']}\"")
    if http_ok(FISH_URL.rsplit("/v1/", 1)[0] + "/health"):
        print(f"  test: {say(TEST_LINE, info['name'], HOME / 'voices' / (info['name'] + '_test.mp3'))}")
    print(f"  cast it with: bookcaster cast <book> --do \"<Character> = {info['name']}\"")
    return 0


def voice_meta() -> dict:
    p = VOICES_DIR / "voices.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def voice_remove(name: str) -> None:
    meta = voice_meta()
    if not str(meta.get(name, {}).get("source", "")).startswith(USER_SOURCE):
        sys.exit(f"'{name}' isn't one of your recorded voices (only those can be removed here)")
    for f in [VOICES_DIR / f"{name}.wav", VOICES_DIR / f"{name}.txt", HOME / "voices" / f"{name}_test.mp3"]:
        f.unlink(missing_ok=True)
    del meta[name]
    (VOICES_DIR / "voices.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"removed voice '{name}'")


# --- pace -------------------------------------------------------------------------
# The Fish server can't speak faster ("speed" is accepted but not applied), so
# a book's pace is a pitch-preserving time-stretch of each chapter's voice
# track right after recording, with the timeline scaled to match: music,
# effects, captions, pictures and animation all follow it. ffmpeg's atempo:
# the user's pick by ear (10-16) over Sonic, Rubber Band R2/R3, SoundTouch
# and PSOLA on the same passage at 1.25x (R2 "sounded awful").

# the user's standing choice (10-17): every book at 1.25x unless its pace.txt says otherwise
DEFAULT_PACE = float(os.environ.get("BOOKCASTER_PACE", "1.25"))


def book_pace(book: Path) -> float:
    f = book / "pace.txt"
    return float(f.read_text().strip()) if f.exists() else DEFAULT_PACE


def set_pace(book: Path, pace: float) -> None:
    if not 0.7 <= pace <= 1.6:
        sys.exit("pace must be between 0.7 and 1.6 (1.25 = a quarter faster)")
    (book / "pace.txt").write_text(f"{pace:g}\n")      # always written: no file = DEFAULT_PACE


def restore_idle(log=print, llm: bool = True) -> None:
    """After a stopped job: end its video renderers (they run in scopes of their own, outside the
    job's unit, and kept rendering) and give the box back its usual servers - a job pauses the music
    server while drawing and the voice server and language model while rendering, and a stop in the
    middle left them down (10-18: the next run recorded without music)."""
    from . import animate
    found = subprocess.run(["pgrep", "-f", r"run_(i2v|a2v|split|v2v|infinitetalk)\.py"],
                           capture_output=True, text=True).stdout.split()
    for pid in found:
        subprocess.run(["kill", pid], capture_output=True)
    if found:
        log(f"  ended {len(found)} video renderer(s)")
    if llm:      # not after a 3D job (10-21: a stop started the 77 GB model mid-stage; RAM fell under the floor)
        animate.igpu("strata", log)              # the language model back, the VAE server off
    # the picture checker holds the voice server's card: left running by a stop, the voice server
    # couldn't start and the next make refused to run (10-18)
    subprocess.run(["systemctl", "--user", "stop", "bookcaster-checker"], capture_output=True)
    subprocess.run(["bash", str(STACK), "up", "--music"], capture_output=True)
    log("  voice and music servers up")


def voices_key(f: Path, args) -> str:
    """What a chapter's voice track was recorded from: its text and the recording options (cast)."""
    import hashlib
    h = hashlib.sha256(f.read_bytes())
    h.update("\0".join(map(str, args)).encode())
    return h.hexdigest()


def apply_pace(book: Path, f: Path, log, fresh: bool = False) -> bool:
    """Bring a chapter's voice track and timeline to the book's pace, from the
    unstretched copies (kept as .raw.*). *fresh*: the voices were just
    recorded (at 1.0). True if anything changed."""
    work = book / "work"
    r, tl = work / f"{f.stem}.mp3", work / f"{f.stem}.timeline.json"
    raw, raw_tl, mark = work / f"{f.stem}.raw.mp3", work / f"{f.stem}.raw.timeline.json", work / f"{f.stem}.pace"
    if fresh:
        for x in (raw, raw_tl, mark):
            x.unlink(missing_ok=True)
    if not r.exists() or not tl.exists():
        return False
    target, current = book_pace(book), float(mark.read_text()) if mark.exists() else 1.0
    if abs(target - current) < 1e-6:
        return False
    if not raw.exists():
        shutil.copyfile(r, raw)
        shutil.copyfile(tl, raw_tl)
    timeline = json.loads(raw_tl.read_text())
    if abs(target - 1.0) < 1e-6:
        shutil.copyfile(raw, r)
        mark.unlink(missing_ok=True)
    else:
        tmp = r.with_name("." + r.name)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af",
                        f"atempo={target:g}",
                        "-b:a", "192k", "-f", "mp3", str(tmp)], check=True)
        tmp.replace(r)
        for seg in timeline:
            seg["t"] = round(float(seg["t"]) / target, 3)
        mark.write_text(f"{target:g}")
    tl.write_text(json.dumps(timeline, ensure_ascii=False))
    # Everything timed in seconds against the old track is stale now - except, for a fresh recording,
    # the music composed ahead from estimated lengths: the score run re-times those cues to the real
    # voices. Deleting them raced the composer still writing into the folder ("composing ahead failed:
    # no such file cue08.wav") and threw the composed-ahead music away on every recording (10-18).
    if not fresh:
        shutil.rmtree(work / "score" / f.stem, ignore_errors=True)
    shutil.rmtree(work / "sfx" / f.stem, ignore_errors=True)
    shutil.rmtree(_video_work(book, f) / "anim", ignore_errors=True)
    log(f"  {f.stem}: voices at {target:g}x (music, effects and animated shots will be redone)")
    return True


# --- music style ------------------------------------------------------------------

def music_style(book: Path) -> str:
    f = book / "music_style.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


def set_music_style(book: Path, style: str) -> None:
    f = book / "music_style.txt"
    if style.strip():
        f.write_text(style.strip() + "\n", encoding="utf-8")
    elif f.exists():
        f.unlink()


def music_sample(book: Path, seconds: float = 30.0) -> Path:
    """A short cue in the book's music style, to hear before recording."""
    from .score import compose
    style = music_style(book) or "sparse cinematic underscore, strings and piano"
    out = book / "out" / "auditions"
    out.mkdir(parents=True, exist_ok=True)
    wav, mp3 = out / "music_sample.wav", out / "music_sample.mp3"
    compose(ACE_URL, {"music": f"{style}, instrumental underscore, no vocals",
                      "t0": 0.0, "t1": seconds}, wav)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-t", str(seconds),
                    "-af", f"afade=t=out:st={seconds - 3}:d=3", "-b:a", "160k", str(mp3)],
                   check=True)
    wav.unlink()
    return mp3


# --- storyboard and looks -----------------------------------------------------------

STORYBOARD_SCENES = 12
STORYBOARD_VOICES = 6

_SHEET = """
import sys, json
from PIL import Image, ImageDraw, ImageFont
items, out, cols, w = json.loads(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
font = None
for f in ("/usr/share/fonts/google-noto/NotoSans-Regular.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"):
    try:
        font = ImageFont.truetype(f, 22); break
    except OSError:
        pass
font = font or ImageFont.load_default()
ims = [Image.open(p).convert("RGB") for p, _ in items]
h = max(int(w * im.height / im.width) for im in ims)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * (h + 34)), "white")
d = ImageDraw.Draw(sheet)
for n, (im, (_, label)) in enumerate(zip(ims, items)):
    x, y = (n % cols) * w, (n // cols) * (h + 34)
    sheet.paste(im.resize((w, int(w * im.height / im.width))), (x, y))
    d.text((x + 6, y + h + 4), label, fill="black", font=font)
sheet.save(out, quality=88)
"""


def _contact_sheet(items: list[tuple[str, str]], out: Path, cols: int, width: int) -> Path:
    """A grid of pictures with a caption under each (PIL lives in the drawing venv)."""
    from . import video
    subprocess.run([str(video.ART_PYTHON), "-I", "-c", _SHEET, json.dumps(items), str(out),
                    str(cols), str(width)], check=True)
    return out


def run_storyboard_job(book: Path) -> int:
    """A cheap review pack before the expensive recording: the main voices,
    every character's look, a music sample and the first scenes in the art
    style, in out/storyboard/ with a summary of what to change and how."""
    from . import video
    files = chapter_files(book)
    log = _logger(book)
    if not files:
        _set_status(book, state="failed", step="nothing picked (bookcaster pick)")
        return 1
    _set_status(book, state="running", step="storyboard", chapters=len(files), done=0, outputs=[],
                started=time.strftime("%Y-%m-%d %H:%M"))
    stamp = time.strftime("%m%d-%H%M")
    sb = book / "out" / "storyboard"
    sb.mkdir(parents=True, exist_ok=True)
    made, notes = [], []
    style = art_style(book) or video.DEFAULT_STYLE
    try:
        _set_status(book, step="storyboard: planning the pictures (characters, places, scenes)")
        plan = plan_pictures_for(book, files[0], log)
        work = _video_work(book, files[0])
        first = plan["pictures"][:STORYBOARD_SCENES]
        _set_status(book, step=f"storyboard: drawing the character sheets and {len(first)} scenes")
        video.draw(first, work / "img", log=log, plan=plan, style=style)
        chars, places = video.looks(plan)
        prompts = video.ref_prompts(plan, style)
        refs = work.parent / "refs"
        seen, sheets = set(), []
        for name in chars:
            if name in prompts and video.canon(name) not in seen:
                seen.add(video.canon(name))
                sheets.append((str(video.ref_path(refs, name, prompts[name][0])), name))
        for name, rows in (plan.get("wardrobe") or {}).items():     # each outfit change, after the base
            for r in rows:
                sheets.append((str(video.outfit_ref(refs, plan, style, name, r)), f"{name} ({r['id']})"))
        sheets = [x for x in sheets if Path(x[0]).exists()]
        if sheets:
            made.append(_contact_sheet(sheets, sb / f"characters-{stamp}.jpg", 3, 640))
            notes.append(f"characters-{stamp}.jpg: how each character looks, and each outfit change the "
                         f"story makes (Name (outfit)). Change one with "
                         f"`bookcaster look {book.name} --do \"Name = new description\"` or "
                         f"`--do \"Name (outfit) = what they wear\"` "
                         f"(see `bookcaster look {book.name}` for the current ones).")
        scenes = [(str(work / "img" / f"{p['id']}.png"), f"{p['id']}: {p['prompt'][:46]}") for p in first
                  if (work / "img" / f"{p['id']}.png").exists()]
        if scenes:
            made.append(_contact_sheet(scenes, sb / f"scenes-{stamp}.jpg", 4, 480))
            notes.append(f"scenes-{stamp}.jpg: the first {len(scenes)} scenes of {files[0].stem} in the art "
                         f"style ({style[:60]}...). Change the style with `bookcaster art {book.name} "
                         f"\"...\"`, one picture with `bookcaster redraw`.")
    except Exception as exc:
        log(f"storyboard pictures failed: {exc}")
        notes.append(f"(pictures failed: {exc})")
    try:
        _set_status(book, step="storyboard: voices")
        lines = speaker_lines(book)
        cast = sorted(lines, key=lambda k: -len(lines[k]))[:STORYBOARD_VOICES]
        clips, order = [], []
        for who in cast:
            try:
                clips.append(audition(book, who))
                order.append(who)
            except (SystemExit, Exception) as exc:
                log(f"audition {who} failed: {exc}")
        if clips:
            gap = sb / ".gap.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            "anullsrc=r=44100:cl=mono", "-t", "0.9", str(gap)], check=True)
            lst = sb / ".voices.txt"
            lst.write_text("".join(f"file '{c}'\nfile '{gap}'\n" for c in clips))
            out = sb / f"voices-{stamp}.mp3"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-ar", "44100", "-ac", "1", "-b:a", "128k", str(out)], check=True)
            lst.unlink(); gap.unlink()
            made.append(out)
            vm = dict(kv.split("=", 1) for kv in voice_map(book).split(","))
            notes.append(f"voices-{stamp}.mp3: in order " + ", ".join(f"{w} ({vm.get(w, '?')})" for w in order)
                         + f". Change one with `bookcaster cast {book.name} --do \"...\"`.")
    except Exception as exc:
        log(f"storyboard voices failed: {exc}")
        notes.append(f"(voices failed: {exc})")
    try:
        _set_status(book, step="storyboard: music sample")
        if http_ok(f"{ACE_URL}/health"):
            m = music_sample(book)
            out = sb / f"music-{stamp}.mp3"
            shutil.copyfile(m, out)
            made.append(out)
            notes.append(f"music-{stamp}.mp3: 30 s in the music style "
                         f"({music_style(book) or 'default: the planner picks per scene'}). "
                         f"Change it with `bookcaster music {book.name} \"...\"`.")
        else:
            notes.append("(no music sample: the music server is down)")
    except Exception as exc:
        log(f"storyboard music failed: {exc}")
        notes.append(f"(music failed: {exc})")
    summary = sb / f"storyboard-{stamp}.txt"
    summary.write_text(f"Storyboard for {book.name}\n\n" + "\n\n".join(notes)
                       + f"\n\nWhen it's right: bookcaster make {book.name} --level <audio|music|pictures|"
                         f"moments|animated> (see bookcaster estimate {book.name}).\n")
    made.append(summary)
    _set_status(book, state="done", step="storyboard ready", outputs=[str(m) for m in made],
                finished=time.strftime("%Y-%m-%d %H:%M"))
    return 0


def keyframes_sample(book: Path, n: int = 8, chapter: int = 1, replan: bool = False, clips: bool = False) -> str:
    """The keyframe stage on n key frames spread through a recorded chapter, in a separate folder
    (<chapter>-sample), with a contact sheet: test a change here before a real run (user 10-18: hours went
    to fixing things live on the real run). Plans the keyframes first if the chapter has none."""
    from . import animate, video
    files = chapter_files(book)
    f = files[min(max(chapter, 1), len(files)) - 1]
    work = _video_work(book, f)
    log = print
    timeline = book / "work" / f"{f.stem}.timeline.json"
    if not timeline.exists():
        sys.exit("record the chapter first (bookcaster make)")
    animate.igpu("strata", log)
    draw_pictures_for(book, f, log)
    prep = animate_for(book, f, "animated", log, phase="plan-only")
    kf, plan = prep["kf"], prep["plan"]
    keys = [x for x in kf["frames"] if x.get("mode") == "key"]
    pick = [keys[round(i * (len(keys) - 1) / max(1, n - 1))] for i in range(min(n, len(keys)))]
    style = art_style(book) or video.DEFAULT_STYLE
    if replan:
        # the sampled scenes planned afresh under the current rules (poses, end frames, ...): the chapter's
        # keyframes.json stays as it is
        want = {x["pic"] for x in pick}
        shots = [s for s in prep["shots"] if s["pic"] in want]
        fresh = animate.plan_keyframes(shots, prep["pics"], prep["timeline"], prep["total"], plan, style,
                                       LLM_URL, LLM_MODEL, log, prep["starts"])
        animate._drop_people_props(fresh, plan)
        animate._add_creatures(fresh["frames"], video.looks(plan)[0])
        animate.apply_wardrobe(fresh, plan, prep["starts"])
        plan["look"]["props"] = {**fresh.get("props", {}), **plan["look"].get("props", {})}
        keys = [x for x in fresh["frames"] if x.get("mode") == "key"]
        pick = [keys[round(i * (len(keys) - 1) / max(1, n - 1))] for i in range(min(n, len(keys)))]
        kf = fresh
    sample = dict(kf, frames=[dict(x, carry=False) for x in pick])
    sw = work.with_name(work.name + "-sample")
    shutil.rmtree(sw, ignore_errors=True)
    (sw / "anim").mkdir(parents=True)
    (sw / "anim" / "keyframes.json").write_text(json.dumps(sample, indent=1, ensure_ascii=False))
    t = time.time()
    with video.drawing_session(sw / "anim" / "draw.log", log):
        animate.draw_keyframes(sample, plan, style, sw, log, llm_url=LLM_URL, model=LLM_MODEL)
    st = json.loads((sw / "anim" / "keyframe_checks.json").read_text())
    items = []
    for x in pick:
        png = sw / "anim" / "keyframes" / f"{x['shot']}.png"
        c = st.get("chosen", {}).get(x["shot"], {})
        kept = [e for e in st.get("log", []) if e.get("id") == x["shot"] and e.get("kept")]
        cap = f"{x['shot']}: {c.get('cand', '?')} of {c.get('of', '?')}" + (" KEPT WITH FAULTS" if kept else "")
        if png.exists():
            items.append((str(png), cap))
            end = animate.end_path(png)
            if end.exists():
                items.append((str(end), f"{x['shot']}: END"))
    t_draw = time.time() - t
    out = book / "out" / "storyboard" / f"keyframes-sample-{time.strftime('%m%d-%H%M')}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    _contact_sheet(items, out, 4, 480)
    msg = (f"{len(pick)} key frame(s) in {t_draw / 60:.1f} min -> {out}\n"
           f"(folder {sw}; the chapter's own keyframes are untouched)")
    if clips:
        msg += "\n" + _sample_clips(sw, pick, kf, plan, style, log)
    return msg


def _sample_clips(sw: Path, pick: list[dict], kf: dict, plan: dict, style: str, log) -> str:
    """The sample's key frames animated with the action engine (Wan2.2 pipelined: start -> end frame)."""
    from . import animate
    kdir = sw / "anim" / "keyframes"
    shots = [{"id": x["shot"], "kind": "action", "t": animate._shot_t(x["shot"]), "dur": animate.ACTION_S,
              "pic": x.get("pic"), "seed": 9665, "out": str(sw / "anim" / f"{x['shot']}.mp4"),
              "prompt": f"{str(x.get('motion', '')).rstrip('. ')}. {animate.MOTION} {style}.",
              "refs": [str(kdir / f"{x['shot']}.png")], "mode": "key"}
             for x in pick if (kdir / f"{x['shot']}.png").exists()]
    resume = animate._servers("down", log)
    animate.igpu("vae", log)
    t = time.time()
    try:
        if animate.ACTION_ENGINE == "ltx":
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
            import ltx_flow
            ltx_flow.render_ltx(shots, [], plan, style, sw, sw / "none.wav", log)
        elif animate.ACTION_ENGINE == "i2v22":
            for p in animate._action_pipeline(shots, {}, plan, style, sw):
                p.wait()
        else:
            p = animate._action_batch("05", shots, {}, plan, style, sw)
            if p:
                p.wait()
    finally:
        animate._vae_server("down", log)
        if resume:
            animate._servers("up", log)
        animate.igpu("strata", log)
    ok = [s for s in shots if Path(s["out"]).exists()]
    return f"{len(ok)}/{len(shots)} clip(s) in {(time.time() - t) / 60:.1f} min, in {sw / 'anim'}"


def show_looks(book: Path) -> str:
    from . import video
    reg = book / "work" / "video" / "look.json"
    if not reg.exists():
        return "no looks yet: they're written with the first picture plan (bookcaster storyboard)"
    chars = json.loads(reg.read_text()).get("characters", {})
    seen, out = set(), []
    for k, v in chars.items():
        if video.canon(k) not in seen:
            seen.add(video.canon(k))
            out.append(f"{k}: {v}")
    for plan_f in sorted((book / "work" / "video").glob("*/plan.json")):
        for name, rows in (json.loads(plan_f.read_text()).get("wardrobe") or {}).items():
            for r in rows:
                out.append(f"{name} ({r['id']}), {plan_f.parent.name} from paragraph {r['from']}: {r['dress']}")
    return "\n".join(out)


def set_outfit(book: Path, name: str, oid: str, dress: str) -> str | None:
    """Change one wardrobe outfit ("Edric (star-iron armor) = ...") in every chapter that has it: its
    sheet and the pictures and keyframes with it are redrawn at the next job. None if there's none."""
    from . import video
    hit = None
    for plan_f in sorted((book / "work" / "video").glob("*/plan.json")):
        plan = json.loads(plan_f.read_text())
        for n, rows in (plan.get("wardrobe") or {}).items():
            if video.canon(n) != video.canon(name) and name.lower() not in n.lower():
                continue
            for r in rows:
                if r["id"].lower() == oid.lower() or oid.lower() in r["id"].lower():
                    r["dress"] = video.literal_safe(dress).rstrip(".")
                    hit = f"{n} ({r['id']})"
        if hit:
            plan_f.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    return hit


def set_look(book: Path, name: str, desc: str, pinned: bool = False) -> str:
    """Give a character a new look, under every spelling of the name. Their
    reference sheet is redrawn and scenes with them are redrawn at the next
    storyboard / video job (the scene prompts include the look)."""
    from . import video
    reg_path = book / "work" / "video" / "look.json"
    if not reg_path.exists():
        sys.exit("no looks yet: run bookcaster storyboard first")
    reg = json.loads(reg_path.read_text())
    chars = reg.get("characters", {})
    hits = [k for k in chars if video.canon(k) == video.canon(name)] or \
           [k for k in chars if name.lower() in k.lower()]
    if len({video.canon(k) for k in hits}) > 1:
        sys.exit(f"which character? " + ", ".join(sorted({k for k in chars})))
    if not hits:
        # someone the storyboard's few pictures didn't need yet (John, a crowd, ...): the full picture
        # plan keeps looks already in the registry, so setting one now fixes how they'll be drawn
        hits = [name.strip()]
    desc = video.literal_safe(desc)
    for k in hits:
        chars[k] = desc
        if pinned:                               # set by hand (`bookcaster look`): the plan's look audit never
            reg.setdefault("_pinned", {})[k] = desc   # overrides it (10-22: it took the Knight's armor off)
    reg_path.write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    # their outfits were written from the old look: planned again from the new one at the next video job
    # (10-25: the Disciple's new brown + ochre look kept the old white + blue in Acts 2's wardrobe, every key)
    for pp in sorted(reg_path.parent.glob("[0-9][0-9]_*/plan.json")):
        plan = json.loads(pp.read_text())
        pc = (plan.get("look") or {}).get("characters") or {}
        mine = [n for n in pc if any(video.canon(n) == video.canon(k) for k in hits)]
        w = plan.get("wardrobe") or {}
        worn = [n for n in w if any(video.canon(n) == video.canon(k) for k in hits)]
        if not mine and not worn:
            continue
        for n in mine:                  # the chapter's own copy of the looks draws its keys (10-25: still white + blue)
            pc[n] = desc
        if worn:
            plan["wardrobe"] = {n: r for n, r in w.items() if n not in worn}
            plan.pop("wardrobe_k", None)
        pp.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    return hits[0]


def swap_looks(book: Path, a: str, b: str, log=print) -> None:
    """Two characters trade appearances (10-25 user: "John looks like what most people have seen as Jesus and vice
    versa ... just swap, no need to redo the LoRA"): the looks (pinned), the trained LoRAs and their base sheets
    change owner, and each LoRA is stamped current for its new owner's inputs so nothing retrains. Outfits stay with
    the person (they come from the story); their sheets redraw from the new look. Keys + clips with them redraw at
    the next video job."""
    import shutil
    from . import video
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    vd = book / "work" / "video"
    reg = json.loads((vd / "look.json").read_text())
    ch = reg.get("characters", {})
    ka = next((k for k in ch if video.canon(k) == video.canon(a)), None)
    kb = next((k for k in ch if video.canon(k) == video.canon(b)), None)
    if not ka or not kb:
        sys.exit(f"no look for {a if not ka else b} (bookcaster look shows them)")
    la, lb = ch[ka], ch[kb]
    set_look(book, ka, lb, pinned=True)
    set_look(book, kb, la, pinned=True)
    for pj in vd.glob("[0-9][0-9]_*/plan.json"):  # the chapter plans carry the looks too
        P = json.loads(pj.read_text())
        pc = (P.get("look") or {}).get("characters") or {}
        if ka in pc and kb in pc:
            pc[ka], pc[kb] = pc[kb], pc[ka]
            pj.write_text(json.dumps(P, indent=1, ensure_ascii=False))
    slug = lambda n: re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_")
    cl = vd / "charlora"
    da, db, tmp = cl / slug(ka), cl / slug(kb), cl / f".swap_{slug(ka)}"
    if da.exists() and db.exists():
        da.rename(tmp)
        db.rename(da)
        tmp.rename(db)
        for d, old in ((da, db), (db, da)):       # the LoRA's paths name its old folder
            bj = d / "build.json"
            if bj.exists():
                bj.write_text(bj.read_text().replace(f"/charlora/{old.name}/", f"/charlora/{d.name}/"))
    # the base reference sheets: each drawing goes to the name its look now belongs to (named by prompt hash)
    # (10-25: the first version read chapter 2's old plan, which lists neither, and deleted both sheets unrenamed)
    refs = vd / "refs"
    olds = {n: sorted(refs.glob(f"{video.slug(n)}-*.png")) for n in (ka, kb)}
    want = {ka: set(), kb: set()}                      # each chapter plan names a sheet by its own prompt
    for pj in vd.glob("[0-9][0-9]_*/plan.json"):
        st = pj.parent / "style.txt"
        pr = video.ref_prompts(json.loads(pj.read_text()), st.read_text().strip() if st.exists() else video.DEFAULT_STYLE)
        if ka in pr and kb in pr:                      # (a plan naming one of them wasn't swapped)
            for n in (ka, kb):
                want[n].add(video.ref_path(refs, n, pr[n][0]))
    keep = {ka: olds[kb][:1], kb: olds[ka][:1]}         # the drawing of the look each one now has
    for n in (ka, kb):
        for dst in want[n]:
            if keep[n] and dst not in olds[n] + olds[ka if n == kb else kb]:
                shutil.copy2(keep[n][0], dst)
    for n in (ka, kb):
        for f in olds[n]:
            if f not in want[ka] | want[kb]:
                f.unlink()                             # stale names (outfit sheets redraw from the new look)
        for dst in want[n]:
            if dst in olds[ka] + olds[kb] and keep[n]:
                shutil.copy2(keep[n][0], dst.with_suffix(".swap"))
    for n in (ka, kb):
        for dst in want[n]:
            if dst.with_suffix(".swap").exists():
                dst.with_suffix(".swap").rename(dst)
    log(f"sheets: {sorted(p_.name for p_ in want[ka] | want[kb] if p_.exists())}")
    import charlora as L
    for n in (ka, kb):
        bj = cl / slug(n) / "build.json"
        if bj.exists():
            d = json.loads(bj.read_text())
            c = L.Char(str(book), n)
            d.update(hash=L.input_hash(c, L.STEPS), shape_hash=L.shape_hash(c, L.STEPS), swapped_from=d.get("trigger"))
            bj.write_text(json.dumps(d, indent=1))
    log(f"{ka} now looks like {kb} did and {kb} like {ka} did (looks, LoRAs, sheets); keys and clips with them "
        f"redraw at the next video job")


# --- finished files ------------------------------------------------------------------
# The Happier phone app saves a download under the file's own name and fails
# ("FileSystemFile.create ... already exists") if it has saved that name
# before; every book had an 01_full_text.mp3. So the tool keeps each finished
# file in work/final/<stem>.<ext> and publishes it to out/ under a name used
# once: <book>_<stem>_<MMDD-HHMM>.<ext> (older copies of it are removed).

def final_path(book: Path, f: Path, ext: str = "mp3") -> Path:
    p = book / "work" / "final" / f"{f.stem}.{ext}"
    legacy = book / "out" / f"{f.stem}.{ext}"
    if not p.exists() and legacy.exists():          # made before this layout
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(legacy, p)
        publish(book, p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def published(book: Path, stem: str, ext: str) -> list[Path]:
    pat = re.compile(rf"^{re.escape(book.name)}_{re.escape(stem)}_\d{{4}}-\d{{4}}\.{ext}$")
    out = book / "out"
    return sorted(p for p in out.glob(f"*.{ext}") if pat.match(p.name)) if out.exists() else []


def publish(book: Path, final: Path) -> Path:
    """Hard-link a finished file into out/ under a fresh name."""
    out = book / "out"
    out.mkdir(exist_ok=True)
    stem, ext = final.stem, final.suffix[1:]
    stamp = time.strftime("%m%d-%H%M", time.localtime(final.stat().st_mtime))
    dest = out / f"{book.name}_{stem}_{stamp}.{ext}"
    for old in published(book, stem, ext):
        if old != dest:
            old.unlink()
    if not dest.exists():
        os.link(final, dest)
    return dest


def migrate_outputs(book: Path) -> None:
    for f in chapter_files(book):
        for ext in ("mp3", "mp4"):
            if (book / "out" / f"{f.stem}.{ext}").exists():
                final_path(book, f, ext)


# --- pictures / video ----------------------------------------------------------------

def art_style(book: Path) -> str:
    f = book / "art_style.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


def set_art_style(book: Path, style: str) -> None:
    f = book / "art_style.txt"
    if style.strip():
        f.write_text(style.strip() + "\n", encoding="utf-8")
    elif f.exists():
        f.unlink()


def _chapter_text(f: Path) -> str:
    from .cli import _load_text
    return _load_text(str(f), strip_gutenberg=True)


def _video_work(book: Path, f: Path) -> Path:
    return book / "work" / "video" / f.stem


def plan_pictures_for(book: Path, f: Path, log) -> dict:
    from . import video
    text = _chapter_text(f)
    return video.ensure_plan(text, _video_work(book, f), LLM_URL, LLM_MODEL,
                             art_style(book) or video.DEFAULT_STYLE,
                             len(text) / CHARS_PER_SEC, log=log)


def draw_pictures_for(book: Path, f: Path, log) -> None:
    from . import video
    plan = plan_pictures_for(book, f, log)
    video.draw_checked(plan["pictures"], _video_work(book, f) / "img", plan,
                       art_style(book) or video.DEFAULT_STYLE, LLM_URL, LLM_MODEL, log=log)


def redraw_pictures(book: Path, f: Path, which: list[str]) -> list[str]:
    """Mark pictures of a finished video to be drawn again (by time or id)."""
    from . import video
    work = _video_work(book, f)
    plan = json.loads((work / "plan.json").read_text())
    timeline = json.loads((book / "work" / f"{f.stem}.timeline.json").read_text())
    pics = video.timed(plan, _chapter_text(f), timeline, video.audio_seconds(final_path(book, f)))
    try:
        return video.reject(pics, work / "img", which)
    except ValueError as exc:
        sys.exit(str(exc))


def narrator_character(book: Path) -> str | None:
    """The cast member who is the book's first-person narrator (Watson)."""
    try:
        chars = json.loads((book / "book.cast.json").read_text())["characters"]
    except (OSError, ValueError, KeyError):
        return None
    return next((n for n, c in chars.items() if str(c.get("same_as", "")).lower() == "narrator"), None)


def animate_all(book: Path, files: list[Path], level: str, log, status=lambda **k: None) -> None:
    """Animated shots for every chapter: all planning first (keyframes planned, drawn, checked: the
    language model on the iGPU), then all rendering (the VAE server on the iGPU). One switch for the
    whole job; Strata takes minutes to load (10-17)."""
    from . import animate
    animate.igpu("strata", log)                # the picture plans and checks need it too
    for n, f in enumerate(files, 1):
        status(step=f"planning shots {n}/{len(files)}: {f.stem} ({level})")
        draw_pictures_for(book, f, log)
        animate_for(book, f, level, log, phase="plan")
    for rnd in range(animate.CLIP_ROUNDS + 1):
        for n, f in enumerate(files, 1):
            status(step=f"animating {n}/{len(files)}: {f.stem} ({level})"
                   + (f", again: clips that failed their check" if rnd else ""))
            animate_for(book, f, level, log, phase="render")
        # every clip checked against its plan (the language model's turn on the iGPU)
        animate.igpu("strata", log)
        again = []
        for n, f in enumerate(files, 1):
            status(step=f"checking clips {n}/{len(files)}: {f.stem}")
            again += animate_for(book, f, level, log,
                                 phase="check-final" if rnd == animate.CLIP_ROUNDS else "check")
        if not again:
            break


def animate_for(book: Path, f: Path, level: str, log, phase: str = "all") -> list:
    """Render the chapter's animated shots ("moments" or "animated"); the
    pictures must be drawn. Talking shots are lip-synced to the voices-only
    render (no music under the words)."""
    from . import animate, video
    voices = book / "work" / f"{f.stem}.mp3"
    timeline = book / "work" / f"{f.stem}.timeline.json"
    if not voices.exists() or not timeline.exists():
        raise RuntimeError(f"{f.stem} isn't recorded yet")
    return animate.animate_chapter(level, _chapter_text(f), voices, timeline, _video_work(book, f),
                                   art_style(book) or video.DEFAULT_STYLE, log=log,
                                   narrator=narrator_character(book), llm_url=LLM_URL, llm_model=LLM_MODEL,
                                   phase=phase)


def estimate_text(book: Path, only: str | None = None) -> str:
    """Rough box time per level for the picked chapters (recorded ones use
    their real length and dialogue share)."""
    from . import animate, video
    total = dialogue = 0.0
    pics = 0
    for f in chapter_files(book):
        tl = book / "work" / f"{f.stem}.timeline.json"
        rec = book / "work" / f"{f.stem}.mp3"
        if tl.exists() and rec.exists():
            secs = video.audio_seconds(rec)
            dialogue += sum(r["t1"] - r["t0"] for r in animate.dialogue_runs(json.loads(tl.read_text()), secs))
        else:
            secs = len(_chapter_text(f)) / CHARS_PER_SEC
            dialogue += 0.6 * secs
        total += secs
        pics += max(3, round(secs / 60))
    hours = total / 3600
    base = {"audio": 15 * hours + 5, "music": 18 * hours + 5}
    base["pictures"] = base["music"] + pics * 12 / 60 + 4 * hours + 3
    for lv in ("moments", "animated"):
        base[lv] = base["pictures"] + animate.estimate_minutes(lv, total, dialogue, pics)
    desc = {"audio": "voices only", "music": "voices + score", "pictures": "+ still-picture video",
            "moments": f"+ {animate.MOMENTS_TALK} talking and {animate.MOMENTS_ACTION} action shots",
            "animated": f"fully animated ({dialogue / 60:.0f} min lip-synced dialogue)"}

    def fmt(m):
        return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"

    rows = [lv for lv in LEVELS if only is None or lv == only]
    head = f"{book.name}: {total / 60:.0f} min of audio" + ("" if only else ", rough box time per level:")
    return "\n".join([head] + [f"  {lv:9s} {fmt(base[lv]):>8s}  {desc[lv]}" for lv in rows])


def video_for(book: Path, f: Path, log, level: str = "pictures") -> Path:
    from . import video
    audio = final_path(book, f)
    timeline = book / "work" / f"{f.stem}.timeline.json"
    if not audio.exists() or not timeline.exists():
        raise RuntimeError(f"{f.stem} isn't recorded yet")
    mp4 = video.make_video(_chapter_text(f), audio, timeline, final_path(book, f, "mp4"),
                           _video_work(book, f), LLM_URL, LLM_MODEL,
                           style=art_style(book), log=log, animated=at_least(level, "moments"))
    out = publish(book, mp4)
    if to_jellyfin(book, f, log):
        from . import jellyfin
        jellyfin.rescan(log)
    return out


def to_jellyfin(book: Path, f: Path, log=print, tag: str = "", mp4: Path | None = None) -> bool:
    """Copy a chapter's finished video to the Jellyfin server, if one is set up
    (~/.config/bookcaster/jellyfin). A failed copy never fails the job. *tag* names a new version beside the
    one already there ("v2": "02 Chapter 2 v2.mp4"; 10-25 user: don't clobber the old one). *mp4*: another video
    of the chapter instead of its finished one (a --from/--to cut, sent as-is)."""
    from . import jellyfin
    mp4 = mp4 or book / "work" / "final" / f"{f.stem}.mp4"
    if not mp4.exists() or jellyfin.config() is None:
        return False
    meta = json.loads((book / "meta.json").read_text()) if (book / "meta.json").exists() else {}
    title = f.read_text(encoding="utf-8").split("\n", 1)[0]
    num = re.match(r"\d+", f.stem)
    name = f"{num.group(0) + ' ' if num else ''}{jellyfin.nice_title(title)}{' ' + tag if tag else ''}.mp4"
    try:
        return jellyfin.send(mp4, meta.get("title") or book.name, name, log=log)
    except Exception as exc:
        log(f"jellyfin: couldn't copy {f.stem}: {exc}")
        return False


def art_sample(book: Path) -> Path:
    """One picture of the opening in the book's art style, to judge it."""
    from . import video
    files = chapter_files(book)
    if not files:
        sys.exit("pick chapters first: bookcaster pick <book> ...")
    style = art_style(book) or video.DEFAULT_STYLE
    paras = video.paragraphs(_chapter_text(files[0]))[:40]
    plan = video.plan_pictures(paras, LLM_URL, LLM_MODEL, style, 1)
    p = plan["pictures"][0]
    work = book / "work" / "art_sample"
    shutil.rmtree(work, ignore_errors=True)
    video.draw([{"id": "sample", "seed": 1000,
                 "full_prompt": video.full_prompt(p["prompt"], plan.get("look", {}), style)}],
               work, log=lambda m: None)
    out = book / "out" / "auditions" / "art_sample.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(work / "sample.png", out)
    print(f"  {p['prompt'][:200]}")
    return out


def _logger(book: Path):
    log = book / "work" / "job.log"

    def write(msg: str) -> None:
        with open(log, "a") as fh:
            fh.write(msg + "\n")
    return write


def run_video_job(book: Path, level: str = "pictures") -> int:
    files = [f for f in chapter_files(book) if final_path(book, f).exists()]
    if not files:
        _set_status(book, state="failed", step="nothing recorded yet (bookcaster make)")
        return 1
    log = _logger(book)
    _set_status(book, state="running", step="video", chapters=len(files), done=0, outputs=[],
                started=time.strftime("%Y-%m-%d %H:%M"))
    outputs = []
    if at_least(level, "moments"):
        try:
            animate_all(book, files, level, log, lambda **k: _set_status(book, **k))
        except Exception as exc:
            log(f"animation failed: {exc}")
            _set_status(book, state="failed", step=f"animation: {exc}"[:300])
            return 1
    for n, f in enumerate(files, 1):
        _set_status(book, step=f"video {n}/{len(files)}: {f.stem} (planning, drawing, building)")
        try:
            outputs.append(str(video_for(book, f, log, level)))
        except Exception as exc:      # report it; the audio is unaffected
            log(f"video failed for {f.stem}: {exc}")
            _set_status(book, state="failed", step=f"video {f.stem}: {exc}"[:300])
            return 1
        _set_status(book, done=n, outputs=outputs)
    _set_status(book, state="done", step="finished", outputs=outputs,
                finished=time.strftime("%Y-%m-%d %H:%M"))
    return 0


# --- the recording job ------------------------------------------------------------

def _status_path(book: Path) -> Path:
    return book / "work" / "job.json"


def _set_status(book: Path, **kw) -> None:
    p = _status_path(book)
    st = json.loads(p.read_text()) if p.exists() else {}
    st.update(kw, updated=time.strftime("%H:%M:%S"))
    p.write_text(json.dumps(st, indent=1))


def sfx_for(book: Path, f: Path, final: Path, log) -> None:
    """Mix sound effects into a finished chapter file (in place)."""
    from . import sfx
    sfx.add_sfx(final, book / "work" / f"{f.stem}.mp3", book / "work" / f"{f.stem}.timeline.json",
                book / "work" / "sfx" / f.stem, LLM_URL, LLM_MODEL, log=log)


def run_sfx_job(book: Path) -> int:
    files = [f for f in chapter_files(book) if final_path(book, f).exists()]
    if not files:
        _set_status(book, state="failed", step="nothing recorded yet (bookcaster make)")
        return 1
    log = _logger(book)
    _set_status(book, state="running", step="sound effects", chapters=len(files), done=0, outputs=[],
                started=time.strftime("%Y-%m-%d %H:%M"))
    outputs = []
    for n, f in enumerate(files, 1):
        _set_status(book, step=f"sound effects {n}/{len(files)}: {f.stem}")
        try:
            final = final_path(book, f)
            sfx_for(book, f, final, log)
            outputs.append(str(publish(book, final)))
        except Exception as exc:
            log(f"sound effects failed for {f.stem}: {exc}")
            _set_status(book, state="failed", step=f"sound effects {f.stem}: {exc}"[:300])
            return 1
        _set_status(book, done=n, outputs=outputs)
    _set_status(book, state="done", step="finished", outputs=outputs, finished=time.strftime("%Y-%m-%d %H:%M"))
    return 0


def run_job(book: Path, level: str = "music", music_only: bool = False, sfx: bool = False) -> int:
    """Everything after casting, in order: delivery cues and the scene plan
    (LLM), then the voices (Fish) while the music composes (ACE), then the
    mix. Finished files go to out/. *music_only* keeps the existing voice
    recordings and redoes just the music (after a style change)."""
    music, video = at_least(level, "music"), at_least(level, "pictures")
    work, out = book / "work", book / "out"
    work.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    log = work / "job.log"
    files = chapter_files(book)
    if not files:
        _set_status(book, state="failed", step="nothing picked (bookcaster pick)")
        return 1
    if music and not http_ok(f"{ACE_URL}/health"):   # 10-27: the music job brings up what it needs
        subprocess.run(["bash", str(STACK), "up", "--music"], capture_output=True)
        for _ in range(60):
            if http_ok(f"{ACE_URL}/health"):
                break
            time.sleep(5)
    if music and not http_ok(f"{ACE_URL}/health"):
        if music_only:                  # asked for the music alone: no music is a failure, not voices "as the mix"
            _set_status(book, state="failed", step="the music server didn't come up (tools/fish_stack.sh up --music)")
            return 1
        music = False
        with open(log, "a") as fh:
            fh.write("music server not up: recording without music\n")
    if music:                           # the score is planned by the language model (10-27: down, so this morning's
        from . import animate           # --music-only "mixed" the voices alone and reported done)
        animate.igpu("strata", _logger(book))
    _set_status(book, state="running", step="reading the cast", chapters=len(files),
                done=0, outputs=[], started=time.strftime("%Y-%m-%d %H:%M"), music=music)
    vm = voice_map(book)
    common = ["--engine", "fish", "--voice-map", vm, "--emote", "--chapters", "off",
              "--llm-url", LLM_URL, "--llm-model", LLM_MODEL, "--bitrate", "192",
              "--cache-stem", stem(book)]
    renders = [work / (f.stem + ".mp3") for f in files]
    scores = [work / "score" / f.stem for f in files]
    style = ["--style", music_style(book)]

    if music_only:
        missing = [r.name for r in renders
                   if not (r.exists() and r.with_name(r.stem + ".timeline.json").exists())]
        if missing or not music:
            _set_status(book, state="failed", step="nothing to redo the music on: "
                        + (f"not recorded yet ({', '.join(missing)})" if missing else "music server down"))
            return 1
    for n, f in enumerate(files, 1):
        if music_only:
            _set_status(book, step=f"scene plan {n}/{len(files)}: {f.stem}")
            tts("score", f, "--audio", renders[n - 1], "--work-dir", scores[n - 1],
                "--llm-url", LLM_URL, "--llm-model", LLM_MODEL, "--plan-only", *style, log=log)
            continue
        _set_status(book, step=f"delivery cues {n}/{len(files)}: {f.stem}")
        if tts("convert", f, *common, "--prepare-only", log=log).returncode:
            _set_status(book, state="failed", step=f"tagging {f.stem} (see work/job.log)")
            return 1
        if music:
            _set_status(book, step=f"scene plan {n}/{len(files)}: {f.stem}")
            tts("score", f, "--audio", renders[n - 1], "--work-dir", scores[n - 1],
                "--llm-url", LLM_URL, "--llm-model", LLM_MODEL, "--plan-only", *style, log=log)
        if video:
            _set_status(book, step=f"picture plan {n}/{len(files)}: {f.stem}")
            try:
                plan_pictures_for(book, f, _logger(book))
            except Exception as exc:
                _logger(book)(f"picture plan failed for {f.stem}: {exc}")

    composer = None
    if music or video:
        # Music, then pictures, on the music GPU while Fish renders on its own.
        def compose_all():
            if music:
                for f, r, sc in zip(files, renders, scores):
                    tts("score", f, "--audio", r, "--work-dir", sc, "--llm-url", LLM_URL,
                        "--llm-model", LLM_MODEL, "--ace-url", ACE_URL, "--ahead", *style, log=log)
            if video:
                for f in files:
                    try:
                        draw_pictures_for(book, f, _logger(book))
                    except Exception as exc:
                        _logger(book)(f"drawing failed for {f.stem}: {exc}")
        composer = threading.Thread(target=compose_all, daemon=True)
        composer.start()

    outputs = []
    for n, (f, r) in enumerate(zip(files, renders), 1):
        if music_only:
            break
        _set_status(book, step=f"recording {n}/{len(files)}: {f.stem}"
                    + (" (music composing alongside)" if music else ""))
        key, key_f = voices_key(f, common), r.with_suffix(".voices")
        if r.exists() and key_f.exists() and key_f.read_text() == key:
            # recorded already with this text and cast (a restarted job): Fish samples afresh every run,
            # so recording again moved every line's timing, and the music and animation plan built on
            # the old timings were thrown away (10-18: the knight's fixed keyframes were lost that way)
            _logger(book)(f"  {f.stem}: voices already recorded with this cast, kept")
        else:
            key_f.unlink(missing_ok=True)
            if tts("convert", f, *common, "-o", r, log=log).returncode or not r.exists():
                _set_status(book, state="failed", step=f"recording {f.stem} (see work/job.log)")
                return 1
            apply_pace(book, f, _logger(book), fresh=True)
            key_f.write_text(key)
        if not music:
            final = final_path(book, f)
            shutil.copyfile(r, final)
            if sfx:
                _set_status(book, step=f"sound effects {n}/{len(files)}: {f.stem}")
                try:
                    sfx_for(book, f, final, _logger(book))
                except Exception as exc:
                    _logger(book)(f"sound effects failed for {f.stem}: {exc}")
            outputs.append(str(publish(book, final)))
            _set_status(book, done=n, outputs=outputs)
    if composer is not None:
        _set_status(book, step="waiting for the music" + (" and pictures" if video else "")
                    + " to finish")
        composer.join()
    if music:
        for n, (f, r, sc) in enumerate(zip(files, renders, scores), 1):
            _set_status(book, step=f"mixing music {n}/{len(files)}: {f.stem}")
            final = final_path(book, f)
            if tts("score", f, "--audio", r, "--work-dir", sc, "-o", final,
                   "--llm-url", LLM_URL, "--llm-model", LLM_MODEL, "--ace-url", ACE_URL,
                   *style, log=log).returncode or not final.exists():
                if music_only:                   # the old mix stays; the job fails loudly
                    _set_status(book, state="failed", step=f"music mix failed for {f.stem} (see work/job.log)")
                    return 1
                shutil.copyfile(r, final)        # the voices without music beat nothing
                with open(log, "a") as fh:
                    fh.write(f"music mix failed for {f.stem}; saved it without music\n")
            if sfx and not video:            # a video's effects come after its animation (below)
                _set_status(book, step=f"sound effects {n}/{len(files)}: {f.stem}")
                try:
                    sfx_for(book, f, final, _logger(book))
                except Exception as exc:
                    _logger(book)(f"sound effects failed for {f.stem}: {exc}")
            outputs.append(str(publish(book, final)))
            _set_status(book, done=n, outputs=outputs)
    missing: list[str] = []
    if video:
        if level == "moments":
            try:
                animate_all(book, files, level, _logger(book), lambda **k: _set_status(book, **k))
            except Exception as exc:
                _logger(book)(f"animation failed: {exc}")
        # sound effects by default for videos, made after the animation (the user's order: effects after
        # the video, not alongside it) and before the build, so the video carries them (10-18: the knight
        # video came out without any because --sfx was opt-in)
        if sfx:
            for n, f in enumerate(files, 1):
                _set_status(book, step=f"sound effects {n}/{len(files)}: {f.stem}")
                try:
                    final = final_path(book, f)
                    sfx_for(book, f, final, _logger(book))
                    outputs.append(str(publish(book, final)))
                except Exception as exc:
                    _logger(book)(f"sound effects failed for {f.stem}: {exc}")
                    missing.append(f.stem)
        for n, f in enumerate(files, 1):
            _set_status(book, step=f"building video {n}/{len(files)}: {f.stem}")
            if level == "animated":              # the shot flow (10-24 user: everything optimised is the default)
                if run_mannequin_job(book, f.stem, None, flow="shots"):
                    _logger(book)(f"video failed for {f.stem}: see the shots flow lines above")
                else:
                    outputs.append(str(book / "work" / "final" / f"{f.stem}.mp4"))
                _set_status(book, outputs=outputs)
                continue
            try:
                outputs.append(str(video_for(book, f, _logger(book), level)))
            except Exception as exc:
                _logger(book)(f"video failed for {f.stem}: {exc}")
            _set_status(book, outputs=outputs)
    # said in the status, not only the log: the knight video went out without its effects unnoticed (10-19)
    _set_status(book, state="done", step="finished" + (f" WITHOUT sound effects for {', '.join(missing)} "
                                                       f"(see work/job.log; bookcaster sfx retries)" if missing else ""),
                outputs=outputs, finished=time.strftime("%Y-%m-%d %H:%M"))
    return 0


def unit_name(book: Path) -> str:
    return "bookcaster-" + re.sub(r"[^A-Za-z0-9_-]", "-", book.name)[:40]


def run_mannequin_job(book: Path, chapter: str | None, span: str | None, keys_only: bool = False,
                      guide: str | None = None, flow: str = "mannequin") -> int:
    """The mannequin flow (tools/mannequin_flow.py) over one chapter's stretch, as the book's job: needs the 3D plan
    + stage (the moments' shots and motion); draws the hero images and keys, runs LTX, assembles out/*_mannequin.mp4."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import flow3d
    import mannequin_flow as MF
    if flow == "shots":                                       # the shot-list flow (tools/shot_flow.py, 10-23)
        import shot_flow as MF                                # noqa: F811 - same run() contract
    from . import animate
    log = _logger(book)
    stem = flow3d.chapter_stem(book, chapter or "1")
    t0, t1 = (float(x) for x in (span or "0:1e9").split(":"))
    if not mix_matches_voices(book, stem):                  # the music again, on the paced voices, while the servers are up
        log(f"{stem}: the mix doesn't match the voices (a pace change since?): mixing the music again first")
        run_job(book, "music", music_only=True)
        if not mix_matches_voices(book, stem):
            _set_status(book, state="failed", step=f"{flow} flow: the mix of {stem} doesn't match its voices "
                        "(`bookcaster make --music-only` failed)")
            return 1
    _set_status(book, state="running", step=f"{flow} flow {stem} {t0:.0f}-{min(t1, 9999):.0f}s",
                started=time.strftime("%Y-%m-%d %H:%M"), outputs=[])
    resume = animate._servers("down", log)                   # keys + LTX use both cards
    try:
        mix = book / "work" / "final" / f"{stem}.mp3"
        if flow == "shots":
            shots_ready(book, stem, log, lambda **k: _set_status(book, **k))
        out = MF.run(book, stem, t0, t1, None, mix, log, until="keys" if keys_only else None, guide=guide)
        if not keys_only and (span is None or covers_chapter(book, stem, out)):  # a whole chapter is the chapter's video
            chapter_video(book, stem, out, log, send=span is None)   # a CLI span run is never sent on its own
        _set_status(book, state="done", step=f"{flow} flow finished: {out}", outputs=[str(out)],
                    finished=time.strftime("%Y-%m-%d %H:%M"))
        return 0
    except (Exception, SystemExit) as exc:                    # noqa: BLE001 - said in the status, not lost
        # (10-25: the LTX memory floor's sys.exit left the status "stopped", the reason only in the journal)
        log(f"{flow} flow failed: {exc}")
        _set_status(book, state="failed", step=f"{flow} flow: {exc}"[:300])
        return 1
    finally:
        if resume:
            animate._servers("up", log)


def mix_matches_voices(book: Path, stem: str) -> bool:
    """True when the chapter's mix (work/final) is as long as its voices (work/<stem>.mp3). 10-27 Speckled Band: the
    1.25x pace re-timed the voices and the timeline (57.4 -> 45.9 min) but the mix stayed the 1.0x one, so the shot
    video, timed from the timeline, played over audio that ran 25% slower ("completely out of sync")."""
    from .score import audio_seconds
    mix, voices = book / "work" / "final" / f"{stem}.mp3", book / "work" / f"{stem}.mp3"
    if not (mix.exists() and voices.exists()):
        return False
    v = audio_seconds(voices)
    return abs(audio_seconds(mix) - v) <= max(2.0, 0.01 * v)


def covers_chapter(book: Path, stem: str, mp4: Path) -> bool:
    """True when a video runs the whole chapter's mix (a --from 0 --to <past the end> span is the whole chapter)."""
    from .score import audio_seconds
    mix = book / "work" / "final" / f"{stem}.mp3"
    # (10-25: the Acts cut ends 1.3 s before its mix, the narration's closing silence; 1 s missed it)
    return mix.exists() and audio_seconds(mp4) >= audio_seconds(mix) - max(3.0, 0.02 * audio_seconds(mix))


def shots_ready(book: Path, stem: str, log=print, status=lambda **k: None) -> None:
    """What the shot flow reads, made first when a chapter has none yet (10-24 user: one bookcaster run from a
    recorded chapter to its video): the chapter plan (pictures, looks, wardrobe, moments, sheets: flow3d plan),
    the moment list (flow3d/plan.json, CPU) and a LoRA for every cast member (both XTXs). No LoRA = no run."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import flow3d
    vd = book / "work" / "video" / stem
    if not (vd / "plan.json").exists() or not (vd / "anim" / "manifest.json").exists():
        status(step=f"shots: planning {stem} (looks, moments, sheets)")
        flow3d.plan(book, stem, log=log)
    if not (vd / "flow3d" / "plan.json").exists():
        R, cx = flow3d._ctx(book, stem)
        try:
            R.plan(cx)
        finally:                                 # only the moment routing reads it; the shot flow runs without
            os.environ.pop("REMAKE_FULL3D", None)
    import shot_flow                         # the outfits first: each LoRA learns its person's outfits (10-25: a
    shot_flow.wardrobe(book, stem, log)      # wardrobe planned after the LoRAs made them stale at the keys)
    from . import video
    plan = json.loads((vd / "plan.json").read_text())   # then any sheet a changed look or outfit needs: each LoRA
    st = vd / "style.txt"                               # learns from its sheet (10-25: the Disciple's trained on
    video.draw([], vd / "img", log=log, plan=plan,     # his old white-and-blue one)
               style=st.read_text().strip() if st.exists() else video.DEFAULT_STYLE)
    cast = set(video.looks(plan)[0])                    # this chapter's cast only
    if flow3d.lora_todo(book, cast):
        status(step=f"shots: character LoRAs {sorted(flow3d.lora_todo(book, cast))}")
        # (10-25: planning left the iGPU's language model up, ~90 GB; the LoRA memory floor refused, 15 < 20 GB)
        from . import animate
        if subprocess.run(["systemctl", "--user", "is-active", "-q", animate.STRATA_UNIT]).returncode == 0:
            log("stopping the language model (LoRA training needs its memory)")
            subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
            time.sleep(5)
            animate._flush()
        left = flow3d.ensure_loras(book, log, only=cast)
        if left:
            raise RuntimeError(f"no LoRA for {left} (see work/video/charlora_build_*.log)")


def chapter_video(book: Path, stem: str, mp4: Path, log=print, send: bool = True) -> None:
    """Make a finished whole-chapter video the chapter's video (work/final/<stem>.mp4, what `bookcaster jellyfin`
    sends) and send it to Jellyfin when a server is set up (send=False: only the first)."""
    final = book / "work" / "final" / f"{stem}.mp4"
    final.unlink(missing_ok=True)
    os.link(mp4, final)
    log(f"{stem}: chapter video is now {mp4.name}")
    if send and to_jellyfin(book, book / "chapters" / f"{stem}.txt", log):
        from . import jellyfin
        jellyfin.rescan(log)


def run_flow3d_job(book: Path, stage: str, chapter: str | None = None, auto: bool = False,
                   jellyfin: bool = False, keys_only: bool = False) -> int:
    """The assets and stage steps draw on both XTX cards too: the voice and music servers are paused for them as
    for video (10-21: `bookcaster stop` brought the servers back onto card 05 and the stage step's prop drawing
    ran out of memory there), and come back after."""
    resume = False
    if stage in ("assets", "stage"):
        from . import animate
        resume = animate._servers("down", _logger(book))
    try:
        return _run_flow3d_job(book, stage, chapter, auto, jellyfin, keys_only)
    finally:
        if resume:
            animate._servers("up", _logger(book))


def _run_flow3d_job(book: Path, stage: str, chapter: str | None = None, auto: bool = False,
                    jellyfin: bool = False, keys_only: bool = False) -> int:
    """A full-3D stage as the book's job (tools/flow3d.py): plan | assets | stage | video, per chapter."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import flow3d
    log = _logger(book)
    chapters = [chapter] if chapter else [f.stem for f in chapter_files(book) if final_path(book, f).exists()]
    _set_status(book, state="running", step=f"3d {stage}", chapters=len(chapters), done=0, outputs=[],
                started=time.strftime("%Y-%m-%d %H:%M"))
    outputs = []
    try:
        if stage == "plan":                                   # stage 0: from the text (no animation)
            for n, c in enumerate(chapters, 1):
                r = flow3d.plan(book, c, log=log)
                outputs.append(r["manifest"])
                _set_status(book, done=n, outputs=outputs)
            _set_status(book, state="done", step=f"3d plan finished: next `bookcaster assets {book.name}`",
                        outputs=outputs, finished=time.strftime("%Y-%m-%d %H:%M"))
            return 0
        if stage == "assets":
            r = flow3d.assets(book, chapter, log=log)
            _set_status(book, state="done", step=f"3d assets: {len(r['pending'])} item(s) wait for review "
                        f"(`bookcaster review {book.name}`)", finished=time.strftime("%Y-%m-%d %H:%M"))
            return 0
        for n, c in enumerate(chapters, 1):
            if stage == "stage":
                flow3d.stage(book, c, log=log)
            else:
                resume = None
                from . import animate
                resume = animate._servers("down", log)        # LTX needs both cards (voice + music paused)
                try:
                    mp4 = flow3d.video(book, c, log=log, status=lambda **k: _set_status(book, **k), auto=auto,
                                       jellyfin=jellyfin, keys_only=keys_only)
                finally:
                    if resume:
                        animate._servers("up", log)
                if mp4 is None:                               # waiting for review / nothing ready: say so
                    _set_status(book, state="waiting", step=f"3d {stage}: waiting (see the log: review, holds or blocked)",
                                finished=time.strftime("%Y-%m-%d %H:%M"))
                    return 2
                outputs.append(str(mp4))
            _set_status(book, done=n, outputs=outputs)
    except SystemExit as exc:                                 # the preflight / a stage refused: say why
        log(f"3d {stage} stopped: {exc}")
        _set_status(book, state="failed", step=f"3d {stage}: {exc}"[:300])
        return 1
    except Exception as exc:
        log(f"3d {stage} failed: {exc}")
        _set_status(book, state="failed", step=f"3d {stage}: {exc}"[:300])
        return 1
    _set_status(book, state="done", step=f"3d {stage} finished", outputs=outputs, finished=time.strftime("%Y-%m-%d %H:%M"))
    return 0


def start_job(book: Path, level: str = "music", music_only: bool = False,
              video_only: bool = False, storyboard: bool = False, sfx: bool = False,
              sfx_only: bool = False, flow3d: str | None = None, chapter: str | None = None,
              auto_approve: bool = False, jellyfin: bool = False, keys_only: bool = False,
              span: str | None = None, guide: str | None = None) -> None:
    unit = unit_name(book)
    if subprocess.run(["systemctl", "--user", "is-active", "-q", unit]).returncode == 0:
        sys.exit(f"already recording {book.name} — `bookcaster status {book.name}`")
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    (book / "work").mkdir(exist_ok=True)
    _status_path(book).write_text("{}")
    # the 3D stages run Blender, Qwen-Image and LTX children (host RAM ~20-60 GB): the box's floor rules apply inside
    cmd = ["systemd-run", "--user", f"--unit={unit}", "-p", f"MemoryMax={'100G' if flow3d else '8G'}",
           "-p", "MemorySwapMax=0", f"--working-directory={book}",
           sys.executable, "-m", "tts_reader.bookcaster", "_job", str(book)]
    cmd += ["--level", level]
    if storyboard:
        cmd.append("--storyboard")
    if sfx:
        cmd.append("--sfx")
    if sfx_only:
        cmd.append("--sfx-only")
    if music_only:
        cmd.append("--music-only")
    if video_only:
        cmd.append("--video-only")
    if flow3d:
        cmd += ["--flow3d", flow3d] + (["--chapter", str(chapter)] if chapter else []) + \
            (["--auto-approve"] if auto_approve else []) + (["--jellyfin"] if jellyfin else []) + \
            (["--keys-only"] if keys_only else []) + (["--span", span] if span else []) + \
            (["--guide", guide] if guide else [])
    subprocess.run(cmd, check=True, capture_output=True)


def describe_status(book: Path) -> str:
    migrate_outputs(book)
    p = _status_path(book)
    if not p.exists():
        return f"{book.name}: not recorded yet"
    st = json.loads(p.read_text() or "{}")
    active = subprocess.run(["systemctl", "--user", "is-active", "-q", unit_name(book)]).returncode == 0
    state = st.get("state", "?")
    if state == "running" and not active:
        state = "stopped (the job is no longer running)"
    lines = [f"{book.name}: {state} — {st.get('step', '')} "
             f"[{st.get('done', 0)}/{st.get('chapters', '?')} chapters, updated {st.get('updated', '?')}]"]
    for f in chapter_files(book):        # what's in out/ now (names change per recording)
        for ext in ("mp3", "mp4"):
            for o in published(book, f.stem, ext):
                lines.append(f"  ready: {o}")
    log = book / "work" / "job.log"
    if state not in ("done",) and log.exists():
        tail = [l for l in log.read_text(errors="replace").splitlines() if l.strip()][-4:]
        lines += ["  log: " + l[:160] for l in tail]
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bookcaster", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="search Project Gutenberg")
    s.add_argument("query", nargs="+")
    s = sub.add_parser("fetch", help="download a Gutenberg book by id")
    s.add_argument("id", type=int)
    s.add_argument("--name", help="library folder name (default: from the title)")
    s = sub.add_parser("chapters", help="list a book's chapters")
    s.add_argument("book")
    s = sub.add_parser("pick", help="choose chapters: numbers, ranges (3-5), title words, or all")
    s.add_argument("book")
    s.add_argument("selectors", nargs="+")
    s = sub.add_parser("cast", help="propose or direct the cast (no --do = show it)")
    s.add_argument("book")
    s.add_argument("--do", action="append", default=[], metavar="COMMAND",
                   help="direction in plain English, 'Name = voice', 'voices'... (repeatable)")
    s.add_argument("--fresh", action="store_true", help="start the cast over")
    s = sub.add_parser("voices", help="the voice library")
    s.add_argument("--filter", default="", help="only voices whose description has this text")
    s.add_argument("--mine", action="store_true", help="only voices made from your own recordings")
    s = sub.add_parser("voice-add", help="make a cast voice from your own recording")
    s.add_argument("recording", help="any audio/video file; 20 s to a few minutes of clear reading")
    s.add_argument("--name", required=True)
    s.add_argument("--gender", default="", help="Male / Female")
    s.add_argument("--description", default="", help="how it sounds (used when casting)")
    s.add_argument("--accent", default="")
    s.add_argument("--replace", action="store_true", help="overwrite an existing voice of that name")
    s = sub.add_parser("voice-remove", help="delete one of your recorded voices")
    s.add_argument("name")
    s = sub.add_parser("audition", help="render a character's line in their cast voice")
    s.add_argument("book")
    s.add_argument("who")
    s.add_argument("--text", help="the line to say (default: one of theirs from the book)")
    s = sub.add_parser("music", help="set or show the book's music style (no style = show it)")
    s.add_argument("book")
    s.add_argument("style", nargs="*", help="plain words, e.g. 'solo piano, romantic era'")
    s.add_argument("--clear", action="store_true", help="back to the default (the planner picks per scene)")
    s.add_argument("--sample", action="store_true", help="compose a 30 s sample in this style")
    s = sub.add_parser("period", help="set or show the story's time period (every picture is drawn in it)")
    s.add_argument("book")
    s.add_argument("period", nargs="*", help="plain words, e.g. 'early medieval Northern Europe, around 1000 AD'")
    s.add_argument("--clear", action="store_true")
    s = sub.add_parser("art", help="set or show the picture style for videos (no style = show it)")
    s.add_argument("book")
    s.add_argument("style", nargs="*", help="plain words, e.g. 'moody oil painting'")
    s.add_argument("--clear", action="store_true", help="back to the default style")
    s.add_argument("--sample", action="store_true", help="draw one picture of the opening")
    s = sub.add_parser("video", help="make the picture video of the recorded chapters (background job)")
    s.add_argument("book")
    s.add_argument("--level", choices=LEVELS[2:], default="pictures",
                   help="pictures (stills, default) | moments (+ animated key moments) | animated")
    s.add_argument("--flow", choices=["2d", "3d", "mannequin", "shots"], default=None,
                   help="3d: every moment from the staged 3D scene (bookcaster assets + stage + review first); "
                        "mannequin: the 3D staging as control maps only (pose + set depth) -> keyframes drawn with "
                        "ControlNet + the character LoRAs + the place's hero image -> LTX (IC-LoRA for physical action); "
                        "shots: the director splits the narration into 2-5 s shots of one action each -> one opening "
                        "picture per shot (LoRAs + the place, no 3D) -> LTX two-stage for action (two takes), pipelined")
    s.add_argument("--from", dest="t_from", help="mannequin: the stretch's start (seconds or m:ss; default 0)")
    s.add_argument("--to", dest="t_to", help="mannequin: the stretch's end (seconds or m:ss; default the chapter's end)")
    s.add_argument("--chapter", help="3d: the chapter number (default: every recorded chapter)")
    s.add_argument("--auto-approve", action="store_true", help="3d: the automated QA gates instead of the review")
    s.add_argument("--jellyfin", action="store_true", help="3d: send the finished video to Jellyfin")
    s.add_argument("--keys-only", action="store_true",
                   help="3d: draw + check every keyframe and make the keyframe preview video (out/*_keys.mp4); no LTX")
    s.add_argument("--guide", choices=["depth", "full", "pins"],
                   help="mannequin: how a landing action's staging steers its clip - depth (the staged depth video, "
                        "light), full (the depth video at full strength), pins (start/hit/end keys drawn on the staging)")
    # the full-3D stages (tools/flow3d.py): each resumable, cached under work/video/
    s = sub.add_parser("plan", help="3d stage 0: plan a chapter from the text (pictures, looks, moments, ref sheets)")
    s.add_argument("book")
    s.add_argument("--chapter", required=True)
    s = sub.add_parser("assets", help="3d: model every character, creature, prop and set of the book (background job)")
    s.add_argument("book")
    s.add_argument("--chapter")
    s.add_argument("--plan-only", action="store_true", help="plan and list, no GPU")
    s = sub.add_parser("stage", help="3d: stage every scene of a chapter (scenes, marks, cameras, motion)")
    s.add_argument("book")
    s.add_argument("--chapter", required=True)
    s.add_argument("--plan-only", action="store_true")
    s = sub.add_parser("review", help="3d: the QA sheets to look at and what still waits for approval")
    s.add_argument("book")
    s.add_argument("--chapter")
    s = sub.add_parser("approve", help="3d: approve assets/scenes (names from `bookcaster review`)")
    s.add_argument("book")
    s.add_argument("items", nargs="*")
    s.add_argument("--auto", action="store_true", help="approve whatever passed its automated checks")
    s.add_argument("--reject", action="store_true")
    s = sub.add_parser("estimate", help="rough box time for each level")
    s.add_argument("book")
    s = sub.add_parser("pace", help="set or show the listening speed (1.25 = a quarter faster, pitch kept)")
    s.add_argument("book")
    s.add_argument("value", nargs="?", type=float)
    s = sub.add_parser("sfx", help="add sound effects to the recorded chapters (background job)")
    s.add_argument("book")
    s = sub.add_parser("storyboard", help="review pack before recording: voices, looks, music, art (background job)")
    s.add_argument("book")
    s = sub.add_parser("keyframes", help="preview the keyframe stage on a few frames (minutes)")
    s.add_argument("book")
    s.add_argument("--sample", type=int, default=8, help="how many key frames, spread through the chapter")
    s.add_argument("--replan", action="store_true", help="plan the sampled scenes afresh under the current rules")
    s.add_argument("--clips", action="store_true", help="also animate the sample (action engine)")
    s.add_argument("--chapter", type=int, default=1)
    s = sub.add_parser("look", help="show or change how characters look in the pictures")
    s.add_argument("book")
    s.add_argument("--do", action="append", default=[], metavar="NAME = DESCRIPTION",
                   help='full new description, e.g. "Roylott = a man of about 60, clean-shaven, black hair, ..."')
    s.add_argument("--swap", nargs=2, metavar=("A", "B"), help="two characters trade looks (LoRAs + sheets too)")
    s = sub.add_parser("redraw", help="redraw video pictures you don't like, then rebuild the video")
    s.add_argument("book")
    s.add_argument("which", nargs="+", help="when it's on screen (12:30) or its number (p021)")
    s.add_argument("--chapter", help="which chapter's video (a number or title words), if several")
    s = sub.add_parser("make", help="record the picked chapters (background job)")
    s.add_argument("book")
    s.add_argument("--level", choices=LEVELS, default=None,
                   help="audio | music (default) | pictures (+ still-picture video) | "
                        "moments (+ animated key moments) | animated (fully animated)")
    s.add_argument("--no-music", action="store_true", help="same as --level audio")
    s.add_argument("--music-only", action="store_true",
                   help="keep the recorded voices, redo only the music (after `music` changes the style)")
    s.add_argument("--video", action="store_true", help="same as --level pictures")
    s.add_argument("--sfx", action="store_true",
                   help="add sound effects (ambience + spot effects); on by default for the video levels")
    s.add_argument("--no-sfx", action="store_true", help="no sound effects, even for a video")
    s = sub.add_parser("status", help="recording progress (all books, or one)")
    s.add_argument("book", nargs="?")
    s = sub.add_parser("stop", help="cancel a recording job")
    s.add_argument("book")
    s = sub.add_parser("jellyfin", help="copy finished videos to the Jellyfin server (all books, or one)")
    s.add_argument("book", nargs="?")
    s.add_argument("--video", default=None, help="first make this finished whole-chapter video the chapter's video")
    s.add_argument("--chapter", default=None, help="only this chapter (default: every picked chapter)")
    s.add_argument("--tag", default="", help="send as a new version beside the old one: --tag v2")
    s.add_argument("--part", default=None, help="send this cut of a chapter (a --from/--to video in out/) as-is, "
                   "beside the chapter: \"09 The Speckled Band (0-10 min).mp4\"")
    sub.add_parser("library", help="books on this machine")
    sub.add_parser("services", help="voice / music / LLM server status")
    s = sub.add_parser("_job")
    s.add_argument("book")
    s.add_argument("--span", default=None)
    s.add_argument("--level", choices=LEVELS, default="music")
    s.add_argument("--music-only", action="store_true")
    s.add_argument("--video-only", action="store_true")
    s.add_argument("--storyboard", action="store_true")
    s.add_argument("--sfx", action="store_true")
    s.add_argument("--sfx-only", action="store_true")
    s.add_argument("--flow3d", choices=["plan", "assets", "stage", "video", "mannequin", "shots"], help="a full-3D stage (tools/flow3d.py)")
    s.add_argument("--chapter")
    s.add_argument("--plan-only", action="store_true")
    s.add_argument("--auto-approve", action="store_true")
    s.add_argument("--jellyfin", action="store_true")
    s.add_argument("--keys-only", action="store_true")
    s.add_argument("--guide")
    a = ap.parse_args(argv)
    if a.cmd == "video" and a.flow is None:          # animated = the shot flow (10-24); stills keep the 2D path
        a.flow = "shots" if a.level == "animated" else "2d"

    if a.cmd == "search":
        res = search(" ".join(a.query))
        if not res:
            print("no matches")
        for r in res:
            print(f"{r['id']:>6}  {r['title'][:70]}" + (f"  — {r['author']}" if r["author"] else ""))
    elif a.cmd == "fetch":
        d = fetch(a.id, a.name)
        meta = json.loads((d / "meta.json").read_text())
        chs = chapters_of(d)
        total = sum(len(c.text) for c in chs)
        print(f"saved {meta['title']} by {meta['author'] or 'unknown'} as '{d.name}' "
              f"({len(chs)} chapters, ~{minutes(total) / 60:.1f} h of narration)")
    elif a.cmd == "chapters":
        d = book_dir(a.book)
        for i, c in enumerate(chapters_of(d), 1):
            print(f"{i:>3}. {c.title[:70]:<70} ~{minutes(len(c.text)):.0f} min")
    elif a.cmd == "pick":
        d = book_dir(a.book)
        files = pick(d, a.selectors)
        total = sum(len(f.read_text(encoding="utf-8")) for f in files)
        for f in files:
            print(f"  {f.stem}")
        print(f"picked {len(files)} chapter(s), ~{minutes(total):.0f} min of narration")
    elif a.cmd == "cast":
        d = book_dir(a.book)
        if not (d / "selection.txt").exists():
            sys.exit("pick chapters first: bookcaster pick <book> ...")
        args = cast_args(d) + (["--fresh-cast"] if a.fresh else [])
        for c in (a.do or ["cast"]):
            args += ["--do", c]
        return tts(*args).returncode
    elif a.cmd == "voices":
        with urlopen(FISH_URL.rsplit("/v1/", 1)[0] + "/voices", timeout=10) as r:
            vs = json.loads(r.read())
        mine = {k for k, v in voice_meta().items()
                if str(v.get("source", "")).startswith(USER_SOURCE)}
        for v in vs:
            if a.mine and v["name"] not in mine:
                continue
            desc = " ".join(str(v.get(k, "")) for k in ("gender", "age", "accent", "description") if v.get(k))
            if a.filter.lower() in (v["name"] + " " + desc).lower():
                print(f"{v['name']:<28} {desc[:110]}")
    elif a.cmd == "voice-add":
        return voice_add(Path(a.recording).expanduser(), a.name, a.gender, a.description,
                         a.accent, a.replace)
    elif a.cmd == "voice-remove":
        voice_remove(a.name)
    elif a.cmd == "audition":
        d = book_dir(a.book)
        print(audition(d, a.who, a.text))
    elif a.cmd == "music":
        d = book_dir(a.book)
        if a.clear or a.style:
            set_music_style(d, "" if a.clear else " ".join(a.style))
            if (d / "out").exists() and any((d / "out").glob("*.mp3")):
                print("(finished chapters keep their old music; `bookcaster make <book> "
                      "--music-only` redoes just the music in a few minutes, keeping the voices)")
        print(f"music style for {d.name}: {music_style(d) or 'default (chosen per scene)'}")
        if a.sample:
            if not http_ok(f"{ACE_URL}/health"):
                sys.exit("the music server isn't up: tools/fish_stack.sh up all")
            print(music_sample(d))
    elif a.cmd == "period":
        d = book_dir(a.book)
        f = d / "period.txt"
        if a.clear and f.exists():
            f.unlink()
        elif a.period:
            f.write_text(" ".join(a.period).strip() + "\n", encoding="utf-8")
        print(f"time period for {d.name}: {f.read_text().strip() if f.exists() else '(not set: the 3D plan reads it from the text)'}")
    elif a.cmd == "art":
        d = book_dir(a.book)
        if a.clear or a.style:
            set_art_style(d, "" if a.clear else " ".join(a.style))
        from .video import DEFAULT_STYLE
        print(f"picture style for {d.name}: {art_style(d) or DEFAULT_STYLE + ' (default)'}")
        if a.sample:
            print(art_sample(d))
    elif a.cmd == "video" and a.flow in ("mannequin", "shots"):
        d = book_dir(a.book)

        def secs(v, dflt):
            if not v:
                return dflt
            m_, _, s_ = v.rpartition(":")
            return float(m_ or 0) * 60 + float(s_)
        span = f"{secs(a.t_from, 0.0)}:{secs(a.t_to, 1e9)}"
        start_job(d, flow3d=a.flow, chapter=a.chapter or "1", span=span, keys_only=a.keys_only, guide=a.guide)
        print(f"making the {a.flow}-flow {'keyframes (stops before the video; review sheet in out/)' if a.keys_only else 'video'} "
              f"of {d.name} ({a.t_from or '0'}-{a.t_to or 'end'}) in the background — `bookcaster status {d.name}`")
    elif a.cmd == "video" and a.flow == "3d":
        d = book_dir(a.book)
        start_job(d, flow3d="video", chapter=a.chapter, auto_approve=a.auto_approve, jellyfin=a.jellyfin,
                  keys_only=a.keys_only)
        print(f"making the 3D video of {d.name} in the background — `bookcaster status {d.name}`"
              + ("" if a.auto_approve else f"; it stops for review when assets wait (`bookcaster review {d.name}`)"))
    elif a.cmd == "plan":
        d = book_dir(a.book)
        start_job(d, flow3d="plan", chapter=a.chapter)
        print(f"3d plan for {d.name} in the background — `bookcaster status {d.name}`, then `bookcaster assets {d.name}`")
    elif a.cmd in ("assets", "stage"):
        d = book_dir(a.book)
        if a.plan_only:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
            import flow3d
            r = (flow3d.assets(d, a.chapter, plan_only=True) if a.cmd == "assets"
                 else flow3d.stage(d, a.chapter, plan_only=True))
            print("\n".join(r.get("blocked", [])) or json.dumps(r.get("pending", r.get("rc", {})), indent=1))
        else:
            start_job(d, flow3d=a.cmd, chapter=a.chapter)
            print(f"3d {a.cmd} for {d.name} in the background — `bookcaster status {d.name}`, then `bookcaster review {d.name}`")
    elif a.cmd in ("review", "approve"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import flow3d
        d = book_dir(a.book)
        if a.cmd == "review":
            flow3d.review(d, a.chapter)
        else:
            done = flow3d.auto_approve(d) if a.auto else flow3d.approve(d, a.items, "rejected" if a.reject else "approved")
            print(f"{'rejected' if a.reject else 'approved'}: {', '.join(done)}" if done else "nothing matched")
    elif a.cmd == "video":
        d = book_dir(a.book)
        if not any(final_path(d, f).exists() for f in chapter_files(d)):
            sys.exit("nothing recorded yet: bookcaster make <book> (or make --video)")
        start_job(d, level=a.level, video_only=True)
        print(f"making the video ({a.level}) for {d.name} in the background — `bookcaster status {d.name}`; "
              f"the .mp4 appears in {d / 'out'}")
    elif a.cmd == "redraw":
        d = book_dir(a.book)
        files = [f for f in chapter_files(d) if final_path(d, f, "mp4").exists()]
        if a.chapter:
            files = [f for f in files if f.stem.startswith(a.chapter.zfill(2) + "_")
                     or a.chapter.lower() in f.stem.lower()]
        if len(files) != 1:
            sys.exit("which chapter? --chapter <number>: " + ", ".join(f.stem for f in files)
                     if files else "no finished video to redraw")
        f = files[0]
        print("rejected " + ", ".join(redraw_pictures(d, f, a.which)))
        start_job(d, level="pictures", video_only=True)
        print(f"redrawing and rebuilding in the background — `bookcaster status {d.name}`")
    elif a.cmd == "make":
        d = book_dir(a.book)
        if not chapter_files(d):
            sys.exit("pick chapters first: bookcaster pick <book> ...")
        if not http_ok(FISH_URL.rsplit("/v1/", 1)[0] + "/health"):
            sys.exit("the voice server isn't up: tools/fish_stack.sh up all")
        level = a.level or ("pictures" if a.video else "audio" if a.no_music else "music")
        if a.no_music and at_least(level, "music"):
            sys.exit("--no-music conflicts with --level " + level)
        # videos get sound effects unless told not to (user 10-18: "why didn't the sound effects go in
        # by default?"); audio-only levels keep them opt-in
        sfx = (a.sfx or at_least(level, "pictures")) and not a.no_sfx
        start_job(d, level=level, music_only=a.music_only, sfx=sfx)
        if at_least(level, "moments"):
            print(estimate_text(d, only=level))
        print(f"recording {d.name} in the background — `bookcaster status {d.name}`; "
              f"finished files appear in {d / 'out'}")
    elif a.cmd == "status":
        books = [book_dir(a.book)] if a.book else [p.parent.parent for p in sorted(HOME.glob("*/work/job.json"))]
        print("\n".join(describe_status(b) for b in books) or "nothing recorded yet")
    elif a.cmd == "stop":
        d = book_dir(a.book)
        was3d = str((json.loads(_status_path(d).read_text()) if _status_path(d).exists() else {}).get("step", "")
                    ).startswith("3d")
        subprocess.run(["systemctl", "--user", "stop", unit_name(d)])
        _set_status(d, state="stopped", step="cancelled")
        print(f"stopped {d.name}")
        restore_idle(print, llm=not was3d)
    elif a.cmd == "jellyfin":
        from . import jellyfin
        if jellyfin.config() is None:
            sys.exit(f"no Jellyfin server set up ({jellyfin.CONFIG})")
        books = [book_dir(a.book)] if a.book else [p.parent for p in sorted(HOME.glob("*/book.txt"))]
        if a.part:
            if not a.book:
                sys.exit("--part needs the book")
            v = Path(a.part).expanduser().resolve()
            if not v.exists():
                sys.exit(f"{v} is missing")
            files = chapter_files(books[0])
            if a.chapter:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
                import flow3d
                files = [f for f in files if f.stem == flow3d.chapter_stem(books[0], a.chapter)]
            if len(files) != 1:
                sys.exit("--part: say which chapter it is cut from (--chapter N)")
            if not a.tag:                       # "_0-600_shots.mp4" -> "(0-10 min)"
                m = re.search(r"_(\d+)-(\d+)_shots\.mp4$", v.name)
                a.tag = (f"({int(m.group(1)) // 60}-{int(m.group(2)) // 60} min)" if m else "(part)")
            if to_jellyfin(books[0], files[0], tag=a.tag, mp4=v):
                jellyfin.rescan()
            return 0
        if a.video:
            if not a.book:
                sys.exit("--video needs the book")
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
            import flow3d
            stem, v = flow3d.chapter_stem(books[0], a.chapter or "1"), Path(a.video).expanduser().resolve()
            if not v.exists() or not covers_chapter(books[0], stem, v):
                sys.exit(f"{v} is missing or doesn't run the whole of {stem}")
            chapter_video(books[0], stem, v, send=False)
        if a.chapter and not a.book:
            sys.exit("--chapter needs the book")
        if a.chapter:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
            import flow3d
        sent = [to_jellyfin(b, f, tag=a.tag) for b in books for f in chapter_files(b)
                if not a.chapter or f.stem == flow3d.chapter_stem(b, a.chapter)]
        if any(sent):
            jellyfin.rescan()
        elif not any((b / "work" / "final" / f"{f.stem}.mp4").exists()
                     for b in books for f in chapter_files(b)):
            print("no finished videos to send")
    elif a.cmd == "library":
        for d in sorted(HOME.glob("*/book.txt")):
            meta = json.loads((d.parent / "meta.json").read_text()) if (d.parent / "meta.json").exists() else {}
            picked = len(chapter_files(d.parent))
            done = len(list((d.parent / "out").glob("*.mp3"))) if (d.parent / "out").exists() else 0
            print(f"{d.parent.name:<34} {meta.get('title', '')[:40]:<40} picked {picked}, finished {done}")
    elif a.cmd == "services":
        return subprocess.run(["bash", str(STACK), "status"]).returncode
    elif a.cmd == "_job":
        if a.flow3d in ("mannequin", "shots"):
            return run_mannequin_job(Path(a.book), a.chapter, a.span, keys_only=a.keys_only, guide=a.guide,
                                     flow=a.flow3d)
        if a.flow3d:
            return run_flow3d_job(Path(a.book), a.flow3d, a.chapter, auto=a.auto_approve, jellyfin=a.jellyfin,
                                  keys_only=a.keys_only)
        if a.storyboard:
            return run_storyboard_job(Path(a.book))
        if a.sfx_only:
            return run_sfx_job(Path(a.book))
        if a.video_only:
            return run_video_job(Path(a.book), a.level)
        return run_job(Path(a.book), a.level, music_only=a.music_only, sfx=a.sfx)
    elif a.cmd == "estimate":
        print(estimate_text(book_dir(a.book)))
    elif a.cmd == "pace":
        d = book_dir(a.book)
        if a.value is None:
            print(f"{d.name}: pace {book_pace(d):g}x")
        else:
            set_pace(d, a.value)
            changed = [f.stem for f in chapter_files(d) if apply_pace(d, f, print)]
            print(f"{d.name}: pace {book_pace(d):g}x")
            if changed:
                st = json.loads(_status_path(d).read_text() or "{}") if _status_path(d).exists() else {}
                if st.get("music", True) and http_ok(f"{ACE_URL}/health"):
                    start_job(d, level="music", music_only=True)
                    print("re-mixing the music to the new pace in the background — `bookcaster status "
                          f"{d.name}`; then remake the video / effects if they had them")
                else:
                    print(f"recorded chapters re-paced; remix with `bookcaster make {d.name} --music-only` "
                          "(music server needed), then remake the video / effects if they had them")
    elif a.cmd == "sfx":
        d = book_dir(a.book)
        if not any(final_path(d, f).exists() for f in chapter_files(d)):
            sys.exit("nothing recorded yet: bookcaster make <book>")
        start_job(d, sfx_only=True)
        print(f"adding sound effects to {d.name} in the background — `bookcaster status {d.name}`")
    elif a.cmd == "storyboard":
        d = book_dir(a.book)
        if not chapter_files(d):
            sys.exit("pick chapters first: bookcaster pick <book> ...")
        start_job(d, storyboard=True)
        print(f"making the storyboard for {d.name} in the background (a few minutes) — "
              f"`bookcaster status {d.name}`; files appear in {d / 'out' / 'storyboard'}")
    elif a.cmd == "keyframes":
        print(keyframes_sample(book_dir(a.book), a.sample, a.chapter, a.replan, a.clips))
    elif a.cmd == "look":
        d = book_dir(a.book)
        if a.swap:
            swap_looks(d, *a.swap)
            return 0
        for cmd in a.do:
            if "=" not in cmd:
                sys.exit('use "Name = full new description"')
            name, desc = (x.strip() for x in cmd.split("=", 1))
            m = re.fullmatch(r"(.+?)\s*\((.+)\)", name)
            if m:                      # "Edric (star-iron armor) = ..." : one outfit from the wardrobe
                who = set_outfit(d, m.group(1).strip(), m.group(2).strip(), desc)
                if not who:
                    sys.exit(f"no outfit '{m.group(2)}' for {m.group(1)} (bookcaster look shows them)")
                print(f"{who} now: {desc}")
                continue
            print(f"{set_look(d, name, desc, pinned=True)} now: {desc} (pinned: the look audit keeps it)")
        if a.do:
            print("their character sheet and scenes are redrawn at the next storyboard or video job")
        else:
            print(show_looks(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
