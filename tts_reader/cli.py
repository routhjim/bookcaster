"""Command-line interface for tts_reader.

Examples
--------
List the available voices::

    python -m tts_reader voices

Convert a text file to an MP3 with the default voice::

    python -m tts_reader convert notes.txt

Read straight from a URL (e.g. a plain-text book)::

    python -m tts_reader convert https://example.com/book.txt -o book.mp3

Pick a voice, speed and bitrate, and set the output path::

    python -m tts_reader convert chapter1.txt -o audiobook/ch1.mp3 \
        --voice ryan --speed 1.1 --bitrate 192

Convert several files at once (one MP3 each) into a folder::

    python -m tts_reader convert *.txt -o out/
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import voices as voice_catalog
from .audio import ChapteredMp3Writer, pcm_to_mp3, write_m3u_playlist
from .chapters import DEFAULT_HEADING, SYNTHETIC_TITLES, Chapter, detect_chapters
from .engine import CastEngine, OrpheusEngine, PiperEngine, default_models_dir
from .speakers import LlmAttributor, character_counts, segment_dialogue
from .textprep import prepare_for_speech


def _parse_voice_map(spec: str) -> dict[str, str]:
    """Parse 'narrator=leo,Ahab=zac,dialogue=tara' into a dict."""
    mapping: dict[str, str] = {}
    for part in spec.split(","):
        name, sep, voice = part.partition("=")
        if not sep or not name.strip() or not voice.strip():
            raise ValueError(
                f"bad --voice-map entry {part!r}; expected name=voice pairs "
                "like 'narrator=leo,Ahab=zac,dialogue=tara'"
            )
        mapping[name.strip().lower()] = voice.strip()
    if "narrator" not in mapping:
        raise ValueError("--voice-map needs a 'narrator=<voice>' entry")
    return mapping


def _make_attributor(args) -> LlmAttributor | None:
    url = getattr(args, "llm_url", None)
    if not url:
        return None
    return LlmAttributor(
        url, model=args.llm_model,
        context_chars=getattr(args, "llm_context", 350),
    )


def _build_engine(args):
    """Construct the requested TTS engine from parsed args."""
    if args.engine == "orpheus":
        def factory(voice):
            return OrpheusEngine(
                voice, base_url=args.orpheus_url, model=args.orpheus_model,
                chunk_chars=args.orpheus_chunk_chars,
            )
        default_voice = voice_catalog.ORPHEUS_DEFAULT_VOICE
    else:
        def factory(voice):
            return PiperEngine(voice, models_dir=args.models_dir)
        default_voice = voice_catalog.DEFAULT_VOICE

    if getattr(args, "voice_map", None):
        return CastEngine(
            _parse_voice_map(args.voice_map), factory,
            attributor=_make_attributor(args),
            pause_ms=args.cast_pause_ms,
        )
    return factory(args.voice or default_voice)


def _is_url(spec: str) -> bool:
    return spec.startswith(("http://", "https://"))


def _source_stem(spec: str) -> str:
    """A sensible base filename for an input (used for default output names)."""
    if _is_url(spec):
        return Path(urlparse(spec).path).stem or "download"
    return Path(spec).stem


# Project Gutenberg wraps its plain-text books in license boilerplate marked by
# these lines; we trim to just the actual work when they are present.
_GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)
_GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL
)


def _strip_gutenberg(text: str) -> str:
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def _fetch_url(url: str) -> str:
    req = Request(url, headers={"User-Agent": "tts_reader/0.1 (+https://pypi.org/project/tts-reader)"})
    with urlopen(req, timeout=30) as resp:  # nosec - user-supplied URL by design
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def _load_text(spec: str, strip_gutenberg: bool = True) -> str:
    """Load text from a local file path or an http(s) URL."""
    if _is_url(spec):
        text = _fetch_url(spec)
    else:
        text = Path(spec).read_text(encoding="utf-8", errors="replace")

    if strip_gutenberg and ("PROJECT GUTENBERG" in text.upper()):
        text = _strip_gutenberg(text)

    text = text.strip()
    if not text:
        raise ValueError(f"Input has no readable text: {spec}")
    return text


def _resolve_outputs(specs: list[str], output: str | None) -> list[Path]:
    """Work out the MP3 destination for each input."""
    if len(specs) == 1 and output and not output.endswith(("/", "\\")):
        out = Path(output)
        if out.is_dir():
            return [out / (_source_stem(specs[0]) + ".mp3")]
        if out.suffix.lower() != ".mp3":
            out = out.with_suffix(".mp3")
        return [out]

    # Multiple inputs, or output points at a directory: one MP3 per input.
    out_dir = Path(output) if output else None
    results = []
    for spec in specs:
        stem = _source_stem(spec)
        if out_dir:
            results.append(out_dir / (stem + ".mp3"))
        elif _is_url(spec):
            results.append(Path(stem + ".mp3"))  # URLs default to CWD
        else:
            results.append(Path(spec).with_suffix(".mp3"))
    return results


def _slugify(text: str, maxlen: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s_]+", "_", text)
    return text[:maxlen] or "chapter"


def _speak_text(chapter: Chapter, preprocess: bool = True) -> str:
    """Text to actually synthesize for a chapter (spoken title + body)."""
    if chapter.title in SYNTHETIC_TITLES or not chapter.title:
        text = chapter.text
    else:
        text = f"{chapter.title}.\n\n{chapter.text}".strip()
    return prepare_for_speech(text) if preprocess else text


def _convert_chapters_embedded(engine, chapters, outp, args) -> None:
    """One MP3 with embedded ID3 chapter markers (jump-to-chapter)."""
    n = len(chapters)
    with ChapteredMp3Writer(
        outp, engine.sample_rate, bitrate=args.bitrate, album=outp.stem
    ) as writer:
        for i, ch in enumerate(chapters, 1):
            print(f"  [{i}/{n}] {ch.title[:60]}  ({len(ch.text):,} chars)")
            pcm = engine.synthesize_pcm(
                _speak_text(ch, preprocess=not args.no_preprocess),
                speed=args.speed, volume=args.volume, log=None,
            )
            writer.add_chapter(ch.title, pcm)


def _convert_chapters_split(engine, chapters, outp, args) -> Path:
    """One MP3 per chapter plus an M3U playlist, in a folder."""
    out_dir = Path(args.output) if args.output else outp.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(chapters)
    entries = []
    for i, ch in enumerate(chapters, 1):
        fname = f"{i:03d}_{_slugify(ch.title)}.mp3"
        print(f"  [{i}/{n}] {ch.title[:60]}  ({len(ch.text):,} chars)")
        pcm = engine.synthesize_pcm(
            _speak_text(ch, preprocess=not args.no_preprocess),
            speed=args.speed, volume=args.volume, log=None,
        )
        pcm_to_mp3(pcm, out_dir / fname, engine.sample_rate, bitrate=args.bitrate)
        seconds = len(pcm) / 2 / engine.sample_rate
        entries.append((fname, ch.title, seconds))
    write_m3u_playlist(out_dir / "playlist.m3u", entries)
    return out_dir


def cmd_voices(args: argparse.Namespace) -> int:
    print("PIPER voices (default engine; use the ALIAS with --voice):\n")
    print(voice_catalog.describe_catalog())
    print(
        "\nYou can also pass any full Piper voice key directly, e.g. "
        "--voice en_US-kusal-medium"
    )
    print(f"\nModels are cached in: {default_models_dir()}")

    print("\n\nORPHEUS voices (use with --engine orpheus --voice <name>):\n")
    print(voice_catalog.describe_orpheus())
    print(
        "\nRequires a running Orpheus server (see README). Emotion tags such as "
        "<laugh>, <sigh>, <chuckle>, <yawn> can be embedded inline in the text."
    )
    return 0


# Orpheus voices handed out when suggesting a cast (narrator/dialogue excluded).
_SUGGEST_VOICES = ["zac", "dan", "mia", "jess", "zoe", "leah"]


def cmd_characters(args: argparse.Namespace) -> int:
    """Detect quoted dialogue, attribute speakers, and print the cast."""
    attributor = _make_attributor(args)
    status = 0
    for spec in args.inputs:
        try:
            text = _load_text(spec, strip_gutenberg=not args.no_strip_gutenberg)
        except (ValueError, OSError) as exc:
            print(f"skip: could not read {spec}: {exc}", file=sys.stderr)
            status = 1
            continue

        segments = segment_dialogue(text)
        quotes = [s for s in segments if s.kind == "quote" and s.text.strip()]
        if attributor:
            try:
                attributor.refine(segments, known=list(character_counts(segments)))
            except RuntimeError as exc:
                print(f"warning: {exc}", file=sys.stderr)
        counts = character_counts(segments)
        unattributed = sum(1 for s in quotes if not s.speaker)

        print(f"\n{spec}: {len(quotes)} quoted passage(s), "
              f"{len(counts)} named speaker(s), {unattributed} unattributed")
        for name, n in list(counts.items())[: args.top]:
            example = next(
                s.text.strip().replace("\n", " ")[:56]
                for s in segments if s.kind == "quote" and s.speaker == name
            )
            print(f"  {name:<18} {n:>4} quote(s)   e.g. “{example}…”")

        main_cast = [n for n, _ in list(counts.items())[: len(_SUGGEST_VOICES)]]
        pairs = ",".join(
            f"{name}={voice}" for name, voice in zip(main_cast, _SUGGEST_VOICES)
        )
        print("\nSuggested Orpheus cast (edit to taste):")
        print(f"  --engine orpheus --voice-map \"narrator=leo,dialogue=tara"
              + (f",{pairs}" if pairs else "") + "\"")
        if not attributor:
            print("Tip: add --llm-url http://127.0.0.1:8080/v1/chat/completions "
                  "to resolve unattributed quotes with a local LLM.")
    return status


def cmd_cast(args: argparse.Namespace) -> int:
    """Interactive casting session: describe roles, direct voices, convert."""
    from .casting import build_session, run_repl

    spec = args.input
    try:
        text = _load_text(spec, strip_gutenberg=not args.no_strip_gutenberg)
    except (ValueError, OSError) as exc:
        print(f"error: could not read {spec}: {exc}", file=sys.stderr)
        return 2

    session = build_session(
        text, engine=args.engine, llm_url=args.llm_url,
        llm_model=args.llm_model, top=args.top,
        llm_context=args.llm_context,
    )

    def on_convert(voice_map: str, output: str | None) -> int:
        argv = [
            "convert", spec, "--engine", args.engine,
            "--voice-map", voice_map,
            "--chapters", args.chapters, "--speed", str(args.speed),
            "--bitrate", str(args.bitrate),
            "--cast-pause-ms", str(args.cast_pause_ms),
        ]
        out = output or args.output
        if out:
            argv += ["-o", out]
        if session.llm_url:
            argv += ["--llm-url", session.llm_url, "--llm-model", args.llm_model]
        if args.no_strip_gutenberg:
            argv.append("--no-strip-gutenberg")
        return main(argv)

    return run_repl(session, on_convert)


def cmd_convert(args: argparse.Namespace) -> int:
    specs: list[str] = args.inputs
    missing = [s for s in specs if not _is_url(s) and not Path(s).is_file()]
    if missing:
        for s in missing:
            print(f"error: no such file: {s}", file=sys.stderr)
        return 2

    heading = args.chapter_regex or DEFAULT_HEADING

    # --list-chapters: just detect and print, no synthesis (and no model needed).
    if args.list_chapters:
        for spec in specs:
            try:
                text = _load_text(spec, strip_gutenberg=not args.no_strip_gutenberg)
            except (ValueError, OSError) as exc:
                print(f"skip: could not read {spec}: {exc}", file=sys.stderr)
                continue
            chapters = detect_chapters(text, pattern=heading)
            print(f"\n{spec}: {len(chapters)} chapter(s) detected")
            for i, ch in enumerate(chapters, 1):
                print(f"  {i:>3}. {ch.title[:70]:<70} ({len(ch.text):,} chars)")
        return 0

    try:
        engine = _build_engine(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    outputs = _resolve_outputs(specs, args.output)

    print(f"Voice: {engine.key}")
    try:
        engine.load()  # download (if needed) + load once, reused for every input
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    failures = 0
    for spec, outp in zip(specs, outputs):
        try:
            text = _load_text(spec, strip_gutenberg=not args.no_strip_gutenberg)
        except (ValueError, OSError) as exc:
            print(f"skip: could not read {spec}: {exc}", file=sys.stderr)
            failures += 1
            continue

        chapters = None
        if args.chapters != "off":
            chapters = detect_chapters(text, pattern=heading)

        use_chapters = chapters is not None and len(chapters) > 1

        try:
            if args.chapters == "split" and chapters:
                print(f"Converting {spec}  ->  {len(chapters)} chapter file(s)")
                out_dir = _convert_chapters_split(engine, chapters, outp, args)
                print(f"  wrote {out_dir}/ (+ playlist.m3u)")
            elif use_chapters and args.chapters in ("auto", "embed"):
                print(
                    f"Converting {spec}  ->  {outp}  "
                    f"({len(text):,} chars, {len(chapters)} chapters)"
                )
                _convert_chapters_embedded(engine, chapters, outp, args)
                size_mb = outp.stat().st_size / 1_048_576
                print(f"  wrote {outp} ({size_mb:,.1f} MB, {len(chapters)} chapters)")
            else:
                print(f"Converting {spec}  ->  {outp}  ({len(text):,} chars)")
                speak = text if args.no_preprocess else prepare_for_speech(text)
                engine.synthesize_to_mp3(
                    speak, outp, speed=args.speed, volume=args.volume,
                    bitrate=args.bitrate, log=None,
                )
                size_kb = outp.stat().st_size / 1024
                print(f"  wrote {outp} ({size_kb:,.1f} KB)")
        except Exception as exc:  # keep going for the remaining inputs
            print(f"error: failed to convert {spec}: {exc}", file=sys.stderr)
            failures += 1
            continue

    done = len(specs) - failures
    print(f"\nDone: {done}/{len(specs)} input(s) converted.")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts_reader",
        description="Turn text files (or URLs) into spoken-word MP3s using a local "
        "neural TTS engine. Default engine is Piper (fast, CPU); pass "
        "--engine orpheus to use a local Orpheus GPU server for higher-quality, "
        "more natural speech.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_voices = sub.add_parser("voices", help="List the available voices")
    p_voices.set_defaults(func=cmd_voices)

    p_conv = sub.add_parser("convert", help="Convert text file(s)/URL(s) to MP3")
    p_conv.add_argument(
        "inputs", nargs="+", help="One or more .txt files or http(s) URLs to read"
    )
    p_conv.add_argument(
        "-o", "--output",
        help="Output MP3 file (single input) or output directory (one MP3 per input). "
        "Defaults to the input name with a .mp3 extension.",
    )
    p_conv.add_argument(
        "-e", "--engine", choices=["piper", "orpheus"], default="piper",
        help="TTS backend: 'piper' (fast, local, default) or 'orpheus' (higher "
        "quality via a local Orpheus GPU server). See README for Orpheus setup.",
    )
    p_conv.add_argument(
        "-v", "--voice", default=None,
        help="Voice to use. Piper: an alias or Piper key (default: "
        f"{voice_catalog.DEFAULT_VOICE}). Orpheus: one of tara/leah/jess/leo/dan/"
        f"mia/zac/zoe (default: {voice_catalog.ORPHEUS_DEFAULT_VOICE}). "
        "Run 'tts_reader voices' to see all options.",
    )
    p_conv.add_argument(
        "--orpheus-url", default="http://127.0.0.1:5005/v1/audio/speech",
        help="Orpheus server endpoint (OpenAI-compatible /v1/audio/speech).",
    )
    p_conv.add_argument(
        "--orpheus-model", default="orpheus",
        help="Model name to request from the Orpheus server (default: orpheus).",
    )
    p_conv.add_argument(
        "--orpheus-chunk-chars", type=int, default=400,
        help="Split text into chunks of at most this many characters before "
        "sending to Orpheus (0 disables). Keeps requests within the model's token "
        "limit for long passages. Default: 400.",
    )
    p_conv.add_argument(
        "--voice-map", default=None,
        help="Multi-voice narration: comma-separated name=voice pairs, e.g. "
        "\"narrator=leo,Ahab=zac,Ishmael=dan,dialogue=tara\". 'narrator' is "
        "required; 'dialogue' is the fallback voice for unattributed quotes. "
        "Run the 'characters' subcommand first to see who was detected.",
    )
    p_conv.add_argument(
        "--cast-pause-ms", type=int, default=250,
        help="Silence inserted when the speaking voice changes in --voice-map "
        "mode, in milliseconds (default: 250; 0 disables).",
    )
    p_conv.add_argument(
        "--llm-url", default=None,
        help="OpenAI-compatible /v1/chat/completions endpoint used to attribute "
        "quotes the built-in heuristics can't (e.g. a local llama.cpp server). "
        "Only used together with --voice-map.",
    )
    p_conv.add_argument(
        "--llm-model", default="default",
        help="Model name sent to --llm-url (default: 'default').",
    )
    p_conv.add_argument(
        "--llm-context", type=int, default=350,
        help="Characters of surrounding text sent with each quote during LLM "
        "attribution. Wider resolves long exchanges better but costs slower "
        "calls (default: 350).",
    )
    p_conv.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip the speech-friendly text cleanup (abbreviation expansion, "
        "roman-numeral chapter numbers, ALL-CAPS softening, footnote removal).",
    )
    p_conv.add_argument(
        "--speed", type=float, default=1.0,
        help="Speaking-rate multiplier: 1.0 = normal, 1.2 = faster, 0.85 = slower.",
    )
    p_conv.add_argument(
        "--volume", type=float, default=1.0,
        help="Volume multiplier applied to the audio (default: 1.0).",
    )
    p_conv.add_argument(
        "--bitrate", type=int, default=128,
        help="MP3 bitrate in kbps (default: 128). Try 192 or 256 for higher quality.",
    )
    p_conv.add_argument(
        "--models-dir", default=None,
        help="Directory to cache voice models in (default: platform data dir).",
    )
    p_conv.add_argument(
        "--no-strip-gutenberg", action="store_true",
        help="Do not auto-trim Project Gutenberg license header/footer text.",
    )
    p_conv.add_argument(
        "--chapters", choices=["auto", "embed", "split", "off"], default="auto",
        help="Chapter handling: 'auto' embeds chapter markers when chapters are "
        "detected (else a plain MP3); 'embed' forces one MP3 with jump-to-chapter "
        "markers; 'split' writes one MP3 per chapter plus a playlist; 'off' ignores "
        "chapters. Default: auto.",
    )
    p_conv.add_argument(
        "--list-chapters", action="store_true",
        help="Detect and print the chapters for each input, then exit (no audio).",
    )
    p_conv.add_argument(
        "--chapter-regex", default=None,
        help="Custom regex (matched per line) marking where chapters begin.",
    )
    p_conv.add_argument(
        "-V", "--verbose", action="store_true",
        help="Show Piper's low-level log output (e.g. missing-phoneme warnings).",
    )
    p_conv.set_defaults(func=cmd_convert)

    p_chars = sub.add_parser(
        "characters",
        help="Detect quoted dialogue and list the speakers found in a text",
    )
    p_chars.add_argument("inputs", nargs="+", help="Text file(s) or URL(s) to scan")
    p_chars.add_argument(
        "--llm-url", default=None,
        help="OpenAI-compatible /v1/chat/completions endpoint to attribute "
        "quotes the heuristics can't (e.g. http://127.0.0.1:8080/v1/chat/completions).",
    )
    p_chars.add_argument(
        "--llm-model", default="default",
        help="Model name sent to --llm-url (default: 'default').",
    )
    p_chars.add_argument(
        "--llm-context", type=int, default=350,
        help="Characters of context per quote in LLM attribution (default: 350).",
    )
    p_chars.add_argument(
        "--top", type=int, default=15,
        help="How many speakers to list (default: 15).",
    )
    p_chars.add_argument(
        "--no-strip-gutenberg", action="store_true",
        help="Do not auto-trim Project Gutenberg license header/footer text.",
    )
    p_chars.set_defaults(func=cmd_characters)

    p_cast = sub.add_parser(
        "cast",
        help="Interactively cast voices for a book's characters, then convert",
    )
    p_cast.add_argument("input", help="Text file or URL to cast and convert")
    p_cast.add_argument(
        "-o", "--output", default=None,
        help="Output MP3/directory used when you say 'convert' in the session.",
    )
    p_cast.add_argument(
        "-e", "--engine", choices=["piper", "orpheus"], default="orpheus",
        help="TTS backend for the cast (default: orpheus).",
    )
    p_cast.add_argument(
        "--llm-url", default="http://127.0.0.1:8080/v1/chat/completions",
        help="OpenAI-compatible /v1/chat/completions endpoint that powers "
        "role descriptions, voice suggestions and plain-language directing "
        "(default: %(default)s — a local llama.cpp server's default port).",
    )
    p_cast.add_argument(
        "--llm-model", default="default",
        help="Model name sent to --llm-url (default: 'default').",
    )
    p_cast.add_argument(
        "--llm-context", type=int, default=350,
        help="Characters of context per quote in LLM attribution (default: 350).",
    )
    p_cast.add_argument(
        "--top", type=int, default=10,
        help="How many characters to cast; the rest use the fallback "
        "dialogue voice (default: 10).",
    )
    p_cast.add_argument("--chapters", choices=["auto", "embed", "split", "off"],
                        default="auto", help="Chapter handling for the final "
                        "conversion (default: auto).")
    p_cast.add_argument("--speed", type=float, default=1.0,
                        help="Speaking-rate multiplier for the conversion.")
    p_cast.add_argument("--bitrate", type=int, default=128,
                        help="MP3 bitrate in kbps for the conversion.")
    p_cast.add_argument("--cast-pause-ms", type=int, default=250,
                        help="Silence at voice changes, ms (default: 250).")
    p_cast.add_argument(
        "--no-strip-gutenberg", action="store_true",
        help="Do not auto-trim Project Gutenberg license header/footer text.",
    )
    p_cast.set_defaults(func=cmd_cast)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Piper logs a benign "Missing phoneme from id map" warning for rare IPA
    # diacritics it drops; hide those (and other sub-error noise) unless -v.
    verbose = getattr(args, "verbose", False)
    logging.getLogger("piper").setLevel(logging.INFO if verbose else logging.ERROR)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
