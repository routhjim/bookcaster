"""11-12: a work root is safe by construction. A root built for shot list A and reused for shot list B gives no write
target that resolves into the source book (the audit run wrote a key and records into the book through a shot
folder linked when that shot was "another shot", and through linked logs). Synthetic book, no media.
   python3 tools/tests/test_work_root_1112.py"""
import os
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got!r}"))
    ok or fails.append(name)

STEM = "01_chapter"
A, B = ["action-00m10s-000100_s0"], ["quiet-00m20s-000200_s0"]


def make_book() -> Path:
    b = Path(tempfile.mkdtemp(prefix="wroot_book_"))
    sd = b / "work" / "video" / STEM / "shots"
    for sid in A + B:
        d = sd / sid.rsplit("_s", 1)[0] / sid
        d.mkdir(parents=True)
        (d / "key.png").write_bytes(b"png of the book")
        (d / "key.q").write_text("q")
        (sd / "ltx" / "out").mkdir(parents=True, exist_ok=True)
        (sd / "ltx" / "out" / f"{sid}.mp4").write_bytes(b"the book's clip")
        (sd / "ltx" / "out" / f"{sid}.json").write_text("{}")
        (sd / "ltx" / "out" / f"{sid}_chk1.png").write_bytes(b"chk")
    (sd / "ltx" / "chains").mkdir(parents=True)
    (sd / "audio").mkdir()
    (sd / "audio" / "line_speech.wav").write_bytes(b"wav")
    (sd / "log.jsonl").write_text("{}\n")
    (sd / "ltx" / "out" / "timeline.jsonl").write_text("{}\n")
    (sd / "ltx" / "out" / "pp_timing.jsonl").write_text("{}\n")
    (b / "work" / "job.json").write_text("{}")
    (b / "work" / "job.log").write_text("")
    (b / "work" / "sfx" / STEM).mkdir(parents=True)
    (b / "work" / "sfx" / STEM / "plan.json").write_text("{}")
    (b / "work" / "final").mkdir()
    (b / "work" / "final" / f"{STEM}.mp3").write_bytes(b"mp3")
    (b / "chapters").mkdir()
    (b / "chapters" / f"{STEM}.txt").write_text("text")
    return b


def into_book(root: Path, book: Path) -> list:
    """Every path under the root that resolves into the book and is not a linked large media file read by name."""
    bad = []
    for x in root.rglob("*"):
        r = x.resolve()
        if r == book or book in r.parents:
            if x.is_symlink() and r.is_file() and r.suffix in S.WB_MEDIA:
                continue
            bad.append(str(x))
    return bad


book = make_book().resolve()
root = Path(tempfile.mkdtemp(prefix="wroot_root_"))
wb = S.work_book(book, STEM, root, A, log=lambda *a: None)
wb2 = S.work_book(book, STEM, root, B, log=lambda *a: None)         # the same root, another shot list
sd = wb2 / "work" / "video" / STEM / "shots"
check("same root reused", wb == wb2)
check("no folder in the root is a link", not [x for x in wb2.rglob("*") if x.is_symlink() and x.resolve().is_dir()])
check("nothing but linked media files resolves into the book", not into_book(root, book), into_book(root, book))
for sid in A + B:
    mid = sid.rsplit("_s", 1)[0]
    check(f"{sid}: its shot folder is real", (sd / mid).is_dir() and not (sd / mid).is_symlink())
for sid in B:
    mid = sid.rsplit("_s", 1)[0]
    check(f"{sid} (rendered now, 'another shot' before): its own folder is real, nothing in it linked",
          (sd / mid / sid).is_dir() and not (sd / mid / sid).is_symlink() and
          not [x for x in (sd / mid / sid).iterdir() if x.is_symlink()])
    check(f"{sid}: no link to the book's clip or records left",
          not [x for x in (sd / "ltx" / "out").glob(f"{sid}*") if x.is_symlink()])
for f in ("log.jsonl",):
    check(f"{f}: not linked", not (sd / f).is_symlink())
for f in ("timeline.jsonl", "pp_timing.jsonl"):
    check(f"ltx/out/{f}: not linked", not (sd / "ltx" / "out" / f).is_symlink())
check("job.json / job.log: not linked", not (wb2 / "work" / "job.json").is_symlink() and not (wb2 / "work" / "job.log").is_symlink())
check("the sfx folder (written by a cut): real", not (wb2 / "work" / "sfx" / STEM).is_symlink())
check("another shot's clip is there for a cut (linked by its own name)", (sd / "ltx" / "out" / f"{A[0]}.mp4").exists())

# a write through the root never reaches the book
(sd / B[0].rsplit("_s", 1)[0] / B[0] / "key.png").write_bytes(b"the supplied key")
(sd / "log.jsonl").open("a").write("{}\n")
(sd / "ltx" / "out" / "timeline.jsonl").open("a").write("{}\n")
(wb2 / "work" / "job.json").write_text('{"state": "running"}')
bk = book / "work" / "video" / STEM / "shots"
check("the book's key untouched", (bk / B[0].rsplit("_s", 1)[0] / B[0] / "key.png").read_bytes() == b"png of the book")
check("the book's logs untouched", (bk / "log.jsonl").read_text() == "{}\n" and
      (bk / "ltx" / "out" / "timeline.jsonl").read_text() == "{}\n" and (book / "work" / "job.json").read_text() == "{}")

# the check stops a root that a hand (or an old version) left with a link through which the render would write
bad_root = Path(tempfile.mkdtemp(prefix="wroot_bad_"))
wb3 = S.work_book(book, STEM, bad_root, A, log=lambda *a: None)
sd3 = wb3 / "work" / "video" / STEM / "shots"
mid = B[0].rsplit("_s", 1)[0]
import shutil
shutil.rmtree(sd3 / mid)
(sd3 / mid).symlink_to(bk / mid)
try:
    S.work_root_check(wb3, bad_root, book, STEM, B)
    check("a linked shot folder stops the run", False)
except RuntimeError as e:
    check("a linked shot folder stops the run, naming it", mid in str(e), str(e))
wb4 = S.work_book(book, STEM, bad_root, B, log=lambda *a: None)    # rebuilt for B: the link replaced, then fine
check("rebuilding for B replaces the link with a real folder", not (sd3 / mid).is_symlink())

print({"fails": fails})
sys.exit(1 if fails else 0)
