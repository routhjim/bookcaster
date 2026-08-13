"""LLM-powered abridgement: condense a book while keeping its dialogue.

``tts-reader abridge book.txt`` writes ``book_abridged.txt`` — a shortened
version built chapter by chapter with these rules:

* every dialogue exchange survives verbatim, quotation marks and attribution
  tags included, so speaker detection and casting still work on the result;
* very long monologues may be trimmed to their key sentences;
* narration/exposition is condensed to key events, essential imagery and
  thematic elements, written in the book's own style.

The output is a plain text file you can read, edit, and then feed to
``characters``/``cast``/``convert`` like any other book. Progress is
checkpointed next to the output, so an interrupted run resumes where it
stopped instead of re-paying completed chapters.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .chapters import SYNTHETIC_TITLES, detect_chapters
from .speakers import NO_THINKING, post_chat

# --level -> rough share of the original length to keep.
LEVELS = {"light": 0.6, "medium": 0.4, "heavy": 0.25}

# Chapters longer than this are split at paragraph boundaries per LLM call.
_CHUNK_CHARS = 7000


def _prompt(chunk: str, keep_ratio: float, continuation: bool) -> str:
    where = (
        "This passage continues a chapter you are already abridging."
        if continuation else "This passage is a chapter (or the whole text)."
    )
    return (
        "You are abridging a novel for audiobook listening. "
        f"{where} Rewrite it condensed, following these rules strictly:\n"
        "1. KEEP every dialogue exchange: quoted speech stays verbatim, with "
        "its quotation marks (same style as the source) and attribution tags "
        "('said X', 'cried Y') intact, so each line's speaker remains "
        "identifiable. Never merge two speakers' lines.\n"
        "2. A very long single-speaker monologue (over ~150 words) may be "
        "trimmed to its key sentences — keep its opening words and any "
        "famous or plot-critical lines, still inside quotation marks.\n"
        "3. CONDENSE narration, description, and exposition AGGRESSIVELY: "
        f"rewrite them at about {int(keep_ratio * 100)}% of their original "
        "length. Never copy a narration paragraph verbatim — compress "
        "several paragraphs into one, keeping key events, essential imagery, "
        "and thematic elements, in the book's own voice and period style. "
        "Short bridges like 'said Ahab, pacing the deck' survive as-is.\n"
        "4. Do not add commentary or invent anything. Do not write headings.\n"
        "Dialogue stays; narration shrinks. The result should read as a "
        "faster version of the same scene, not a summary about it.\n\n"
        f"Text to abridge:\n{chunk}\n\n"
        "Reply with ONLY the abridged text."
    )


def _split_paragraphs(body: str, max_chars: int) -> list[str]:
    chunks, current = [], ""
    for para in re.split(r"\n{2,}", body):
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


class Abridger:
    def __init__(self, llm_url: str, llm_model: str = "default",
                 level: str = "medium", timeout: float = 600.0):
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.keep_ratio = LEVELS[level]
        self.timeout = timeout

    def _call(self, chunk: str, continuation: bool) -> str:
        # Budget: enough tokens for the kept share of the chunk plus slack.
        max_tokens = max(600, int(len(chunk) * self.keep_ratio / 2.5) + 300)
        body = post_chat(self.llm_url, {
            "model": self.llm_model,
            "messages": [{
                "role": "user",
                "content": _prompt(chunk, self.keep_ratio, continuation),
            }],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "chat_template_kwargs": NO_THINKING,
        }, timeout=self.timeout)
        text = body["choices"][0]["message"]["content"].strip()
        # Strip a code fence if the model added one.
        text = re.sub(r"^```[a-z]*\n|\n```$", "", text).strip()
        return text

    def abridge_body(self, body: str) -> str:
        chunks = _split_paragraphs(body, _CHUNK_CHARS)
        return "\n\n".join(
            self._call(c, continuation=(i > 0)) for i, c in enumerate(chunks)
        )


def abridge_file(text: str, out_path: Path, llm_url: str, llm_model: str,
                 level: str, chapter_pattern: str, log=print) -> Path:
    """Abridge *text* chapter by chapter into *out_path*, with resume."""
    chapters = detect_chapters(text, pattern=chapter_pattern)
    abridger = Abridger(llm_url, llm_model, level)

    progress_path = out_path.with_name(out_path.name + ".progress.json")
    source_sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
    pieces: dict[str, str] = {}
    try:
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("source_sha") == source_sha and saved.get("level") == level:
            pieces = dict(saved.get("pieces", {}))
            if pieces:
                log(f"Resuming: {len(pieces)}/{len(chapters)} chapter(s) "
                    "already abridged.")
    except (OSError, ValueError):
        pass

    def checkpoint() -> None:
        progress_path.write_text(
            json.dumps({"source_sha": source_sha, "level": level,
                        "pieces": pieces}, ensure_ascii=False),
            encoding="utf-8",
        )

    total = len(chapters)
    for i, ch in enumerate(chapters):
        if str(i) in pieces:
            continue
        if len(ch.text.strip()) < 200:  # heading stubs pass through as-is
            pieces[str(i)] = ch.text.strip()
            continue
        log(f"  [{i + 1}/{total}] abridging: {ch.title[:60]} "
            f"({len(ch.text):,} chars)")
        pieces[str(i)] = abridger.abridge_body(ch.text)
        checkpoint()

    parts = []
    for i, ch in enumerate(chapters):
        body = pieces.get(str(i), "").strip()
        if not body:
            continue
        if ch.title and ch.title not in SYNTHETIC_TITLES:
            parts.append(f"{ch.title}\n\n{body}")
        else:
            parts.append(body)
    out_path.write_text("\n\n\n".join(parts) + "\n", encoding="utf-8")
    try:
        progress_path.unlink()
    except OSError:
        pass
    return out_path
