"""bookcaster: plain, non-interactive steps for turning a public-domain book
into a cast, scored audiobook. Built for an agent (or a person) to drive:

    bookcaster search "sherlock holmes"        find books on Project Gutenberg
    bookcaster fetch 1661                      download one into the library
    bookcaster chapters <book>                 list its chapters
    bookcaster pick <book> 8 | "speckled"      choose what to record
    bookcaster cast <book> [--do "..."]        propose / direct the cast
    bookcaster voices [--filter old]           the voice library
    bookcaster audition <book> Holmes          hear a character's voice
    bookcaster make <book> [--no-music]        record it (background job)
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
CHARS_PER_SEC = 16.0          # Fish narration pace, for time estimates
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


# --- the recording job ------------------------------------------------------------

def _status_path(book: Path) -> Path:
    return book / "work" / "job.json"


def _set_status(book: Path, **kw) -> None:
    p = _status_path(book)
    st = json.loads(p.read_text()) if p.exists() else {}
    st.update(kw, updated=time.strftime("%H:%M:%S"))
    p.write_text(json.dumps(st, indent=1))


def run_job(book: Path, music: bool = True) -> int:
    """Everything after casting, in order: delivery cues and the scene plan
    (LLM), then the voices (Fish) while the music composes (ACE), then the
    mix. Finished files go to out/."""
    work, out = book / "work", book / "out"
    work.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    log = work / "job.log"
    files = chapter_files(book)
    if not files:
        _set_status(book, state="failed", step="nothing picked (bookcaster pick)")
        return 1
    if music and not http_ok(f"{ACE_URL}/health"):
        music = False
        with open(log, "a") as fh:
            fh.write("music server not up: recording without music\n")
    _set_status(book, state="running", step="reading the cast", chapters=len(files),
                done=0, outputs=[], started=time.strftime("%Y-%m-%d %H:%M"), music=music)
    vm = voice_map(book)
    common = ["--engine", "fish", "--voice-map", vm, "--emote", "--chapters", "off",
              "--llm-url", LLM_URL, "--llm-model", LLM_MODEL, "--bitrate", "192",
              "--cache-stem", stem(book)]
    renders = [work / (f.stem + ".mp3") for f in files]
    scores = [work / "score" / f.stem for f in files]

    for n, f in enumerate(files, 1):
        _set_status(book, step=f"delivery cues {n}/{len(files)}: {f.stem}")
        if tts("convert", f, *common, "--prepare-only", log=log).returncode:
            _set_status(book, state="failed", step=f"tagging {f.stem} (see work/job.log)")
            return 1
        if music:
            _set_status(book, step=f"scene plan {n}/{len(files)}: {f.stem}")
            tts("score", f, "--audio", renders[n - 1], "--work-dir", scores[n - 1],
                "--llm-url", LLM_URL, "--llm-model", LLM_MODEL, "--plan-only", log=log)

    composer = None
    if music:
        def compose_all():
            for f, r, sc in zip(files, renders, scores):
                tts("score", f, "--audio", r, "--work-dir", sc, "--llm-url", LLM_URL,
                    "--llm-model", LLM_MODEL, "--ace-url", ACE_URL, "--ahead", log=log)
        composer = threading.Thread(target=compose_all, daemon=True)
        composer.start()

    outputs = []
    for n, (f, r) in enumerate(zip(files, renders), 1):
        _set_status(book, step=f"recording {n}/{len(files)}: {f.stem}"
                    + (" (music composing alongside)" if music else ""))
        if tts("convert", f, *common, "-o", r, log=log).returncode or not r.exists():
            _set_status(book, state="failed", step=f"recording {f.stem} (see work/job.log)")
            return 1
        if not music:
            final = out / r.name
            shutil.copyfile(r, final)
            outputs.append(str(final))
            _set_status(book, done=n, outputs=outputs)
    if music:
        _set_status(book, step="waiting for the music to finish composing")
        composer.join()
        for n, (f, r, sc) in enumerate(zip(files, renders, scores), 1):
            _set_status(book, step=f"mixing music {n}/{len(files)}: {f.stem}")
            final = out / r.name
            if tts("score", f, "--audio", r, "--work-dir", sc, "-o", final,
                   "--llm-url", LLM_URL, "--llm-model", LLM_MODEL, "--ace-url", ACE_URL,
                   log=log).returncode or not final.exists():
                shutil.copyfile(r, final)        # the voices without music beat nothing
                with open(log, "a") as fh:
                    fh.write(f"music mix failed for {f.stem}; saved it without music\n")
            outputs.append(str(final))
            _set_status(book, done=n, outputs=outputs)
    _set_status(book, state="done", step="finished", outputs=outputs,
                finished=time.strftime("%Y-%m-%d %H:%M"))
    return 0


def unit_name(book: Path) -> str:
    return "bookcaster-" + re.sub(r"[^A-Za-z0-9_-]", "-", book.name)[:40]


def start_job(book: Path, music: bool) -> None:
    unit = unit_name(book)
    if subprocess.run(["systemctl", "--user", "is-active", "-q", unit]).returncode == 0:
        sys.exit(f"already recording {book.name} — `bookcaster status {book.name}`")
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    (book / "work").mkdir(exist_ok=True)
    _status_path(book).write_text("{}")
    cmd = ["systemd-run", "--user", f"--unit={unit}", "-p", "MemoryMax=6G",
           "-p", "MemorySwapMax=0", f"--working-directory={book}",
           sys.executable, "-m", "tts_reader.bookcaster", "_job", str(book)]
    if not music:
        cmd.append("--no-music")
    subprocess.run(cmd, check=True, capture_output=True)


def describe_status(book: Path) -> str:
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
    for o in st.get("outputs", []):
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
    s = sub.add_parser("audition", help="render a character's line in their cast voice")
    s.add_argument("book")
    s.add_argument("who")
    s.add_argument("--text", help="the line to say (default: one of theirs from the book)")
    s = sub.add_parser("make", help="record the picked chapters (background job)")
    s.add_argument("book")
    s.add_argument("--no-music", action="store_true")
    s = sub.add_parser("status", help="recording progress (all books, or one)")
    s.add_argument("book", nargs="?")
    s = sub.add_parser("stop", help="cancel a recording job")
    s.add_argument("book")
    sub.add_parser("library", help="books on this machine")
    sub.add_parser("services", help="voice / music / LLM server status")
    s = sub.add_parser("_job")
    s.add_argument("book")
    s.add_argument("--no-music", action="store_true")
    a = ap.parse_args(argv)

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
        for v in vs:
            desc = " ".join(str(v.get(k, "")) for k in ("gender", "age", "accent", "description") if v.get(k))
            if a.filter.lower() in (v["name"] + " " + desc).lower():
                print(f"{v['name']:<28} {desc[:110]}")
    elif a.cmd == "audition":
        d = book_dir(a.book)
        print(audition(d, a.who, a.text))
    elif a.cmd == "make":
        d = book_dir(a.book)
        if not chapter_files(d):
            sys.exit("pick chapters first: bookcaster pick <book> ...")
        if not http_ok(FISH_URL.rsplit("/v1/", 1)[0] + "/health"):
            sys.exit("the voice server isn't up: tools/fish_stack.sh up all")
        start_job(d, music=not a.no_music)
        print(f"recording {d.name} in the background — `bookcaster status {d.name}`; "
              f"finished files appear in {d / 'out'}")
    elif a.cmd == "status":
        books = [book_dir(a.book)] if a.book else [p.parent.parent for p in sorted(HOME.glob("*/work/job.json"))]
        print("\n".join(describe_status(b) for b in books) or "nothing recorded yet")
    elif a.cmd == "stop":
        d = book_dir(a.book)
        subprocess.run(["systemctl", "--user", "stop", unit_name(d)])
        _set_status(d, state="stopped", step="cancelled")
        print(f"stopped {d.name}")
    elif a.cmd == "library":
        for d in sorted(HOME.glob("*/book.txt")):
            meta = json.loads((d.parent / "meta.json").read_text()) if (d.parent / "meta.json").exists() else {}
            picked = len(chapter_files(d.parent))
            done = len(list((d.parent / "out").glob("*.mp3"))) if (d.parent / "out").exists() else 0
            print(f"{d.parent.name:<34} {meta.get('title', '')[:40]:<40} picked {picked}, finished {done}")
    elif a.cmd == "services":
        return subprocess.run(["bash", str(STACK), "status"]).returncode
    elif a.cmd == "_job":
        return run_job(Path(a.book), music=not a.no_music)
    return 0


if __name__ == "__main__":
    sys.exit(main())
