"""Abridge a book to a target length.

``abridge.py`` keeps every spoken line and shrinks only the narration, so a
dialogue-heavy book barely gets shorter. This module works to a length instead:

1. map    each chapter is divided into scenes (paragraph ranges + one sentence)
2. rate   the whole scene list is rated 0-5 for how much the story needs it
3. budget the target length is shared out: top scenes are dramatised at a share
          of their own length, "2" scenes become a sentence of bridging
          narration, the rest are dropped
4. write  each kept scene is rewritten to its budget from the book's own lines
5. fit    scenes that ran over are cut back until the total meets the target

Every step is checkpointed in one state file, and the map and ratings do not
depend on the target, so another length reuses them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .chapters import DEFAULT_HEADING, detect_chapters

CHARS_PER_WORD = 5.7          # English prose incl. spaces and punctuation
MAP_CHARS = 14000             # a longer chapter is mapped in pieces
WEIGHT = {5: 1.0, 4: 0.6, 3: 0.3}   # a full scene's share of its own length, before scaling
MAX_SHARE = 0.85              # never ask for a "rewrite" longer than this share of the source
MIN_SHARE = 0.12              # below this a scene cannot be shown, only told: its level becomes bridges
MIN_FULL = 350                # chars: the smallest scene worth writing as a scene
BRIEF_CHARS = 170             # a bridged scene: a sentence or two
OVER = 1.15                   # a written scene this far over its budget is cut back
PICTURE = re.compile(r"\[Illustration(?:[^\[\]]|\[[^\[\]]*\])*\]")


def words(chars: float) -> int:
    return max(1, round(chars / CHARS_PER_WORD))


def paragraphs(body: str) -> list[str]:
    """A chapter's paragraphs, one line each, without picture captions."""
    body = PICTURE.sub("", body)
    out = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body)]
    return [p for p in out if p]


def _json_list(reply: str) -> list:
    m = re.search(r"\[.*\]", reply, re.S)
    try:
        v = json.loads(m.group(0)) if m else []
    except ValueError:
        v = []
    return v if isinstance(v, list) else []


# --- 1. map -------------------------------------------------------------------

MAP_PROMPT = (
    "Below is one chapter of a book (or a part of one), its paragraphs numbered.\n"
    "Divide it into scenes. A scene is one continuous stretch in one place and time: a "
    "conversation, an event, a letter, or a passage of narration covering a stretch of time. "
    "Most chapters have 1 to 4 scenes. Every paragraph belongs to exactly one scene, in order.\n"
    "Reply with ONLY a JSON list, one object per scene:\n"
    '{{"from": first paragraph number, "to": last paragraph number, "what": "one sentence: what '
    'happens and what it changes, naming the people", "story": true}}\n'
    'Use "story": false for text that is not the story itself (a preface, an editor\'s note, a '
    "contents list, a dedication).\n\n{text}"
)


def fix_ranges(raw: list, first: int, last: int) -> list[dict]:
    """Scene ranges that cover first..last exactly once, whatever the model returned."""
    sc = []
    for r in raw:
        try:
            a = int(r["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if first <= a <= last and all(a != s["from"] for s in sc):
            sc.append({"from": a, "what": str(r.get("what") or "").strip(), "story": r.get("story") is not False})
    sc.sort(key=lambda s: s["from"])
    if not sc:
        sc = [{"from": first, "what": "", "story": True}]
    sc[0]["from"] = first
    for s, nxt in zip(sc, sc[1:] + [None]):
        s["to"] = nxt["from"] - 1 if nxt else last
    return sc


def map_chapter(paras: list[str], chat) -> list[dict]:
    out, i = [], 0
    while i < len(paras):
        j, size = i, 0
        while j < len(paras) and (j == i or size + len(paras[j]) <= MAP_CHARS):
            size += len(paras[j])
            j += 1
        text = "\n\n".join(f"[{k + 1}] {paras[k]}" for k in range(i, j))
        reply = chat(MAP_PROMPT.format(text=text), 300 + 120 * max(4, (j - i) // 4))
        out += fix_ranges(_json_list(reply), i + 1, j)
        i = j
    return out


# --- 2. rate ------------------------------------------------------------------

RATE_PROMPT = (
    "A book is being abridged to about {pct}% of its length, for a listener who has not read it. "
    "Its scenes are listed in order with their length in words.\n\n{listing}\n\n"
    "Rate scenes S{a} to S{b} for the abridged version, judging each against the whole book:\n"
    "5 = a turning point: the story cannot be told without showing this scene at length\n"
    "4 = an important scene that should be shown, shortened\n"
    "3 = worth showing briefly\n"
    "2 = only its outcome matters: a sentence or two of narration will do\n"
    "1 = can be left out without the listener missing anything they need later\n"
    "0 = not part of the story\n"
    "Be strict: about one scene in ten is a 5. A scene whose outcome a later scene depends on "
    "is at least a 2.\n"
    "Reply with ONLY one line per scene: S<number>: <rating>"
)
RATE_WINDOW = 300


def listing(scenes: list[dict]) -> str:
    return "\n".join(f"S{i + 1} (chapter {s['ch'] + 1}, {words(s['chars'])} words): {s['what']}"
                     for i, s in enumerate(scenes))


def rate(scenes: list[dict], pct: int, chat) -> None:
    text = listing(scenes)
    for a in range(0, len(scenes), RATE_WINDOW):
        b = min(len(scenes), a + RATE_WINDOW)
        if all("rating" in s for s in scenes[a:b]):
            continue
        reply = chat(RATE_PROMPT.format(pct=pct, listing=text, a=a + 1, b=b), 200 + 8 * (b - a))
        got = {int(n): int(r) for n, r in re.findall(r"S(\d+)\s*[:=]\s*([0-5])", reply)}
        for i in range(a, b):
            scenes[i]["rating"] = got.get(i + 1, 2) if scenes[i]["story"] else 0


# --- 3. budget ----------------------------------------------------------------

def budget(scenes: list[dict], target: float) -> dict:
    """Set each scene's mode (full / brief / cut) and budget in chars; returns the totals."""
    full_levels = [5, 4, 3]
    while True:
        for s in scenes:
            r = s["rating"]
            s["mode"] = "full" if r in full_levels else "brief" if r >= 2 else "cut"
        briefs = sum(BRIEF_CHARS for s in scenes if s["mode"] == "brief")
        full = [s for s in scenes if s["mode"] == "full"]
        room = max(0.0, target - briefs)

        def total(k: float) -> float:
            return sum(min(MAX_SHARE, k * WEIGHT[s["rating"]]) * s["chars"] for s in full)

        lo, hi = 0.0, MAX_SHARE / min(WEIGHT.values())
        if total(hi) <= room:
            k = hi                                   # the book is shorter than the target allows
        else:
            for _ in range(50):
                k = (lo + hi) / 2
                lo, hi = (k, hi) if total(k) < room else (lo, k)
        lowest = full_levels[-1]
        if len(full_levels) > 1 and k * WEIGHT[lowest] < MIN_SHARE:
            full_levels.pop()                        # too thin to show: tell those scenes instead
            continue
        break
    for s in scenes:
        if s["mode"] == "full":
            s["budget"] = round(min(MAX_SHARE, k * WEIGHT[s["rating"]]) * s["chars"])
            if s["budget"] < MIN_FULL:
                s["mode"] = "brief"
        if s["mode"] != "full":
            s["budget"] = BRIEF_CHARS if s["mode"] == "brief" else 0
    return {"k": round(k, 3), "full_levels": full_levels,
            "planned_chars": sum(s["budget"] for s in scenes)}


# --- 4. write -----------------------------------------------------------------

WRITE_PROMPT = (
    "You are abridging a book for an audiobook. Rewrite the scene below in about {n} words "
    "(it is {m} words now).\n"
    "1. Speech: use the book's own spoken lines word for word, inside the same kind of quotation "
    "marks, each with a tag naming its speaker ('said X') so a listener can tell who speaks. "
    "Choose the lines that carry the scene (the ones that turn it, show character, or are "
    "memorable) and leave the others out entirely. Never invent a spoken line, never reword one, "
    "never merge two speakers' lines. A long speech may be cut down to its key sentences.\n"
    "2. Narration: tell the rest in few words, in the book's own voice, tense and period style, "
    "keeping its best phrases.\n"
    "3. {bridge}\n"
    "4. No headings, no commentary, no notes, no picture captions, no summing-up of what the "
    "scene means.\n"
    "Keep to the length: {n} words.\n\nThe scene:\n{text}\n\n"
    "Reply with ONLY the abridged scene."
)
BRIDGE_FIRST = "This is where the abridged version begins."
BRIDGE_AFTER = "The listener has heard only the abridged version, which so far ends: «{tail}»."
BRIDGE_TOLD = (" Since then the following happened and has NOT been told:\n{told}\n"
               "Begin with one or two sentences telling it as the book's own narrator would (the book's tense "
               "and vocabulary; the notes above are a modern present-tense summary, do not copy their wording), "
               "then go into the scene.")
BRIDGE_PLAIN = " Begin so the scene follows on from that; if the place or time has changed, say so in a few words."
TAIL_PROMPT = (
    "You are abridging a book for an audiobook. The abridged version so far ends: «{tail}».\n"
    "The following happens next and closes this part of the story:\n{told}\n"
    "Tell it in about {n} words of narration, in the book's own voice, tense and period style. "
    "Reply with ONLY the narration."
)
TRIM_PROMPT = (
    "This passage from an abridged book must be cut to about {n} words (it is {m}). Cut whole "
    "sentences and whole exchanges; do not reword what you keep. Every spoken line you keep stays "
    "word for word, in its quotation marks, with its speaker tag.\n\n{text}\n\n"
    "Reply with ONLY the cut passage."
)


def _clean(reply: str) -> str:
    t = re.sub(r"^```[a-z]*\n|\n```$", "", reply.strip()).strip()
    return re.sub(r"\n{3,}", "\n\n", t)


def write(scenes: list[dict], src, chat, save, log) -> None:
    """src(scene) -> the scene's source text. Fills scene["text"] for every full scene."""
    tail, told = "", []
    n_full = sum(s["mode"] == "full" for s in scenes)
    done = 0
    for s in scenes:
        if s["mode"] == "brief":
            told.append(s["what"])
        if s["mode"] != "full":
            s.pop("text", None)
            continue
        done += 1
        fresh = s.get("text") and abs(s.get("wrote_for", 0) - s["budget"]) <= 0.15 * s["budget"] \
            and s.get("wrote_told") == told
        if not fresh:
            bridge = BRIDGE_FIRST if not tail else BRIDGE_AFTER.format(tail=tail)
            if told:
                bridge += BRIDGE_TOLD.format(told="\n".join("- " + t for t in told))
            elif tail:
                bridge += BRIDGE_PLAIN
            n = words(s["budget"])
            log(f"  [{done}/{n_full}] scene of {words(s['chars'])} words -> {n}")
            s["text"] = _clean(chat(WRITE_PROMPT.format(n=n, m=words(s["chars"]), bridge=bridge,
                                                         text=src(s)), int(n * 1.8) + 300))
            s["wrote_for"], s["wrote_told"] = s["budget"], list(told)
            save()
        tail, told = s["text"][-500:], []
    full = [s for s in scenes if s["mode"] == "full"]
    for s in full[:-1] if told else full:             # only the last shown scene can carry a closing bridge
        s.pop("tail", None)
        s.pop("tail_told", None)
    if told and full:                                 # bridged scenes after the last shown one
        last = full[-1]
        if last.get("tail_told") != told:
            n = words(BRIEF_CHARS * len(told))
            last["tail"] = _clean(chat(TAIL_PROMPT.format(tail=tail, n=n, told="\n".join(
                "- " + t for t in told)), int(n * 1.8) + 200))
            last["tail_told"] = list(told)
            save()


# --- 4b. voice ----------------------------------------------------------------

VOICE_PROMPT = (
    "Below is a scene from an abridged book. Some of its narration (above all the sentences that bridge from one "
    "scene to the next) may read like a modern summary: present tense, or words the book would not use. This is "
    "how the book itself narrates:\n«{sample}»\n\n"
    "Return the scene with ONLY such narration rewritten in the book's own tense and voice, at the same length. "
    "Everything inside quotation marks stays exactly as it is, word for word, and so does narration that already "
    "sounds like the book. If nothing needs changing, return the scene unchanged.\n\n{text}\n\n"
    "Reply with ONLY the scene."
)


def spoken(text: str) -> list[str]:
    """The quoted passages of a text, in order (curly or straight double quotes)."""
    return re.findall(r"“[^”]*”", text) or re.findall(r'"[^"]*"', text)


def voice(scenes: list[dict], src, chat, save, log) -> None:
    """One pass per written scene: narration that reads like a summary (11-23: bridges came out in the present
    tense, in the scene notes' wording) is put into the book's own voice. A reply that touches any spoken line or
    changes the length by more than a fifth is thrown away."""
    todo = [s for s in scenes if s["mode"] == "full" and s.get("voiced") != s.get("text")]
    for n, s in enumerate(todo, 1):
        sample = next((p for p in src(s).split("\n\n") if len(p) > 200 and not spoken(p)), src(s)[:600])[:700]
        out = _clean(chat(VOICE_PROMPT.format(sample=sample, text=s["text"]), int(words(len(s["text"])) * 1.8) + 300))
        ok = spoken(out) == spoken(s["text"]) and 0.8 * len(s["text"]) <= len(out) <= 1.2 * len(s["text"])
        if ok and out != s["text"]:
            s["text"] = out
        s["voiced"] = s["text"]
        if n % 10 == 0 or n == len(todo):
            log(f"  [{n}/{len(todo)}] narration checked against the book's voice")
        save()


# --- 5. fit -------------------------------------------------------------------

def fit(scenes: list[dict], target: float, chat, save, log, rounds: int = 3) -> None:
    """Cut back the scenes that ran over until the whole meets the target."""
    full = [s for s in scenes if s["mode"] == "full"]
    for _ in range(rounds):
        total = sum(len(s["text"]) + len(s.get("tail", "")) for s in full)
        if total <= 1.04 * target:
            return
        over = [s for s in full if len(s["text"]) > OVER * s["budget"]]
        if not over:                                  # every scene near its budget, the whole too long
            for s in full:
                s["budget"] = round(len(s["text"]) * target / total * 0.97)
            over = full
        log(f"  {total / target:.0%} of the target: cutting {len(over)} scene(s) back")
        for s in over:
            n = words(s["budget"])
            cut = _clean(chat(TRIM_PROMPT.format(n=n, m=words(len(s["text"])), text=s["text"]),
                              int(n * 1.8) + 300))
            if 0.3 * len(s["text"]) < len(cut) < len(s["text"]):
                s["text"] = cut
                save()


# --- checks and output --------------------------------------------------------

def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def quote_check(out: str, source: str) -> dict:
    """How many spoken passages of 5+ words are found word for word in the source."""
    src = " " + _norm(source) + " "
    spans = re.findall(r"“([^”]{20,})”", out) or re.findall(r'"([^"]{20,})"', out)
    n = bad = 0
    examples = []
    for sp in spans:
        for piece in re.split(r"(?<=[.!?;])\s+", sp):   # a trimmed speech keeps whole sentences
            q = _norm(piece)
            if len(q.split()) < 5:
                continue
            n += 1
            if f" {q} " not in src:
                bad += 1
                if len(examples) < 12:
                    examples.append(piece[:120])
    return {"spoken_sentences": n, "not_in_source": bad, "examples": examples}


def assemble(scenes: list[dict], chapter_chars: float) -> str:
    """Join the scenes into new chapters of about chapter_chars, breaking only where the book does."""
    chapters, cur, size, last_ch = [], [], 0, None
    for s in scenes:
        if s["mode"] != "full":
            continue
        if cur and s["ch"] != last_ch and size >= chapter_chars:
            chapters.append(cur)
            cur, size = [], 0
        piece = s["text"] + ("\n\n" + s["tail"] if s.get("tail") else "")
        cur.append(piece)
        size += len(piece)
        last_ch = s["ch"]
    if cur:
        if chapters and size < 0.3 * chapter_chars:
            chapters[-1] += cur
        else:
            chapters.append(cur)
    return "\n\n\n".join(f"Chapter {i + 1}\n\n" + "\n\n".join(c) for i, c in enumerate(chapters)) + "\n"


def outline(scenes: list[dict]) -> str:
    rows = []
    for i, s in enumerate(scenes):
        got = f" (came out {words(len(s['text']))})" if s.get("text") else ""
        plan = {"full": f"{words(s['budget'])} words{got}", "brief": "one sentence", "cut": "left out"}[s["mode"]]
        rows.append(f"S{i + 1}  chapter {s['ch'] + 1}  rated {s['rating']}  {words(s['chars'])} words -> {plan}\n"
                    f"    {s['what']}")
    return "\n".join(rows) + "\n"


def condense(text: str, target_chars: float, chat, state_path: Path, log=print,
             pattern: str = DEFAULT_HEADING, chapter_chars: float = 9600.0,
             ratings: dict | None = None, outline_only: bool = False,
             trim: bool = True) -> tuple[str | None, dict]:
    """Abridge *text* to about *target_chars*. chat(prompt, max_tokens) -> reply.

    ratings: {scene number (1-based): 0-5} set by hand; they persist in the state file.
    Returns (abridged text or None when outline_only, report).
    """
    chs = [paragraphs(c.text) for c in detect_chapters(text, pattern=pattern)]
    sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
    try:
        st = json.loads(state_path.read_text(encoding="utf-8"))
        if st.get("sha") != sha:
            st = {}
    except (OSError, ValueError):
        st = {}
    st.setdefault("sha", sha)
    st.setdefault("mapped", {})
    st.setdefault("hand", {})

    def save() -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")

    for ci, paras in enumerate(chs):
        if str(ci) in st["mapped"] or not paras:
            continue
        log(f"  [{ci + 1}/{len(chs)}] finding the scenes ({words(sum(map(len, paras)))} words)")
        st["mapped"][str(ci)] = map_chapter(paras, chat)
        st.pop("scenes", None)
        save()
    if "scenes" not in st:
        st["scenes"] = []
        for ci, paras in enumerate(chs):
            for r in st["mapped"].get(str(ci), []):
                st["scenes"].append({**r, "ch": ci, "chars": sum(len(p) for p in paras[r["from"] - 1:r["to"]])})
    scenes = st["scenes"]
    source_chars = sum(s["chars"] for s in scenes if s["story"])
    if any("rating" not in s for s in scenes):
        log(f"  rating {len(scenes)} scenes")
        rate(scenes, max(1, round(100 * target_chars / max(1, source_chars))), chat)
        save()
    st["hand"].update({str(k): int(v) for k, v in (ratings or {}).items()})
    for k, v in st["hand"].items():
        if 1 <= int(k) <= len(scenes):
            scenes[int(k) - 1]["rating"] = v
    plan = budget(scenes, target_chars)
    save()
    report = {"source_chars": source_chars, "target_chars": round(target_chars), **plan,
              "scenes": len(scenes), **{m: sum(s["mode"] == m for s in scenes) for m in ("full", "brief", "cut")}}
    if outline_only:
        return None, report

    def src(s: dict) -> str:
        return "\n\n".join(chs[s["ch"]][s["from"] - 1:s["to"]])

    write(scenes, src, chat, save, log)
    if trim:                                          # trim=False: keep the scenes as written, however long
        fit(scenes, target_chars, chat, save, log)
    voice(scenes, src, chat, save, log)
    out = assemble(scenes, chapter_chars)
    report["chars"] = len(out)
    report["quotes"] = quote_check(out, text)
    return out, report
