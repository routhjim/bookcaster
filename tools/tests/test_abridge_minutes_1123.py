"""11-23: `bookcaster abridge <book> --minutes N` works to a length: scenes are mapped, rated, given a share of the
target, rewritten to it and cut back when they run over; the map and ratings are reused for another length; fetch
falls back to Project Gutenberg's mirror."""
import json, re, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from tts_reader import condense as C

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# a made-up book: 6 chapters, each two scenes of 8 paragraphs
LINE = "“We shall cross the ford before the light goes,” said Orla, and nobody answered her at all. "
def chapter(i):
    return f"Chapter {i + 1}\n\n" + "\n\n".join(f"Paragraph {i}-{j}. " + LINE * 3 for j in range(16))
BOOK = "\n\n".join(chapter(i) for i in range(6))
RATINGS = [5, 1, 4, 2, 3, 2, 5, 1, 4, 2, 3, 2]
calls = {"map": 0, "rate": 0, "write": 0, "trim": 0, "tail": 0}
LONG = False

def chat(prompt, max_tokens):
    if prompt.startswith("Below is one chapter"):
        calls["map"] += 1
        return 'Here: [{"from": 1, "to": 8, "what": "first half", "story": true}, {"from": 9, "what": "second half"}]'
    if prompt.startswith("A book is being abridged"):
        calls["rate"] += 1
        return "\n".join(f"S{i + 1}: {r}" for i, r in enumerate(RATINGS))
    if prompt.startswith("This passage from an abridged book"):
        calls["trim"] += 1
        n = int(re.search(r"cut to about (\d+) words", prompt).group(1))
        return " ".join(["cut"] * n)
    if "closes this part of the story" in prompt:
        calls["tail"] += 1
        return "And so it ended."
    calls["write"] += 1
    n = int(re.search(r"in about (\d+) words", prompt).group(1))
    return " ".join(["words"] * (n * 2 if LONG else n))

check("ranges are repaired to cover the chapter once", C.fix_ranges(
    [{"from": 5, "what": "b"}, {"from": 3, "what": "a"}, {"from": 99}, {"x": 1}], 1, 10)
    == [{"from": 1, "what": "a", "story": True, "to": 4}, {"from": 5, "what": "b", "story": True, "to": 10}])
check("picture captions are dropped from the paragraphs",
      C.paragraphs("One.\n\n[Illustration: a cap [_note_]]\n\nTwo\nlines.") == ["One.", "Two lines."])

with tempfile.TemporaryDirectory() as td:
    st = Path(td) / "state.json"
    src = sum(len(p) for c in C.detect_chapters(BOOK) for p in C.paragraphs(c.text))
    target = 0.25 * src
    out, rep = C.condense(BOOK, target, chat, st, log=lambda m: None, chapter_chars=target / 3)
    S = json.loads(st.read_text())["scenes"]
    check("every chapter mapped once, the list rated once", calls["map"] == 6 and calls["rate"] == 1, calls)
    check("12 scenes with their ratings", [s["rating"] for s in S] == RATINGS, [s["rating"] for s in S])
    check("rating 1 is left out, 2 is told in a sentence", all(
        s["mode"] == {1: "cut", 2: "brief"}.get(s["rating"], "full") for s in S), [s["mode"] for s in S])
    check("the plan adds up to the target", abs(rep["planned_chars"] - target) < 0.03 * target, rep)
    b = {s["rating"]: s["budget"] for s in S}
    check("a higher rating gets more room", b[5] > b[4] > b[3] > b[2], b)
    check("only shown scenes are written; the closing bridge is told", calls["write"] == 6 and calls["tail"] == 1, calls)
    check("new chapters carry headings the chapter finder reads",
          len(C.detect_chapters(out)) >= 2 and out.startswith("Chapter 1\n"), out[:40])
    check("the result is near the target", 0.8 * target < len(out) < 1.1 * target, (len(out), target))
    n = dict(calls)
    out2, _ = C.condense(BOOK, target, chat, st, log=lambda m: None, chapter_chars=target / 3)
    check("a second run reuses everything", calls == n and out2 == out, calls)
    C.condense(BOOK, target / 2, chat, st, log=lambda m: None)
    check("another length reuses the map and ratings, rewrites the scenes",
          calls["map"] == 6 and calls["rate"] == 1 and calls["write"] > n["write"], calls)
    _, rep3 = C.condense(BOOK, target, chat, st, log=lambda m: None, ratings={2: 5}, outline_only=True)
    S = json.loads(st.read_text())["scenes"]
    check("a rating set by hand sticks and the outline stops before writing",
          S[1]["rating"] == 5 and S[1]["mode"] == "full" and "S2 " in C.outline(S), S[1])

with tempfile.TemporaryDirectory() as td:
    LONG, calls["trim"] = True, 0
    out, rep = C.condense(BOOK, target, chat, Path(td) / "s.json", log=lambda m: None)
    check("scenes that ran over are cut back to the target", calls["trim"] > 0 and len(out) < 1.1 * target,
          (calls["trim"], len(out), target))

    calls["trim"] = 0
    out, _ = C.condense(BOOK, target, chat, Path(td) / "t.json", log=lambda m: None, trim=False)
    check("trim=False keeps the scenes as written", calls["trim"] == 0 and len(out) > 1.5 * target, (calls["trim"], len(out)))

src_txt = "He stood. “I will not go back to the mill, whatever you say to me,” said Tam. “Then stay and be hungry.”"
q = C.quote_check("“I will not go back to the mill, whatever you say to me,” said Tam. "
                  "“You are a wicked and ungrateful boy indeed,” she said.", src_txt)
check("spoken lines not in the source are counted", q["spoken_sentences"] == 2 and q["not_in_source"] == 1, q)

sc = [{"rating": 3, "chars": 10000, "story": True} for _ in range(40)] + [{"rating": 5, "chars": 10000, "story": True}]
p = C.budget(sc, 20000)
check("scenes too thin to show at this length are told instead", 3 not in p["full_levels"]
      and sum(s["mode"] == "brief" for s in sc) == 40, p)

from tts_reader import bookcaster as B
u = B._gutenberg_urls(1342)
check("fetch tries the main site, then the mirror", u[0].endswith("/cache/epub/1342/pg1342.txt")
      and u[1].endswith("/1/3/4/1342/1342-0.txt"), u)
check("a one-digit book id has a mirror path", B._gutenberg_urls(7)[1].endswith("/0/7/7-0.txt"), B._gutenberg_urls(7))

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
