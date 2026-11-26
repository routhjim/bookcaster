"""What a book keeps and what is waste (bookcaster tidy / archive).

A book's folder holds its essentials (the text, the recordings, the cast, looks and plans, the character sheets, each
character's LoRA in use, the figures, sets and keys, the finished audio and video) beside what making them left
behind. Only what a rule below names is waste; everything else stays. The rules name leftovers the flow itself can
tell are no longer read:

  lora        a trained character LoRA's leftovers, once its pick stands (lora/USE names a file that exists): the
              other training steps, the training state kept to resume a training, the training cache, and the copy
              of the LoRA before it (lora.prev)
  old flow    the work folder of the retired staging flow (work/video/<chapter>/mannequin)
  superseded  folders set aside under a dated name when their content was made again (work/video/old-MMDD,
              charlora_old_MMDD, charlora_prev_MMDD), and a place's snapshots before a rebuild, all but the newest

tidy() lists them (and removes them when told to); archive() copies a book without them to another disk.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

AUTO = ("lora",)                 # what a finished job clears by itself: only what the pick of a LoRA proves unused
DATED = re.compile(r"^(old-\d{4}|charlora_(old|prev)_\d{4})$")


def lora_waste(who: Path) -> list:
    """One character's LoRA folder (charlora/<who>): its leftovers, none while no pick stands."""
    ld = who / "lora"
    use = ld / "USE"
    if not use.is_file():
        return []
    name = use.read_text().strip()
    if not name or not (ld / name).is_file():
        return []
    out = [(p, "lora", "a training step not in use") for p in sorted(ld.glob("step_*.safetensors")) if p.name != name]
    out += [(p, "lora", "training state, kept only to resume a training") for p in sorted(ld.glob("*.pt"))]
    if (ld / "cache").is_dir():
        out.append((ld / "cache", "lora", "the training cache"))
    if (who / "lora.prev").is_dir():
        out.append((who / "lora.prev", "lora", "the LoRA before the one in use"))
    return out


def waste(book: Path) -> list:
    """[(path, kind, why)] of a book's waste, by the rules above."""
    vd = Path(book) / "work" / "video"
    out = []
    if not vd.is_dir():
        return out
    for who in sorted((vd / "charlora").glob("*")) if (vd / "charlora").is_dir() else []:
        if who.is_dir():
            out += lora_waste(who)
    for d in sorted(vd.glob("*/mannequin")):
        if d.is_dir():
            out.append((d, "old flow", "the retired staging flow's work"))
    for d in sorted(vd.iterdir()):
        if d.is_dir() and DATED.match(d.name):
            out.append((d, "superseded", "set aside when it was made again"))
    for h in sorted(vd.glob("engine/*/history")):
        snaps = sorted((s for s in h.iterdir() if s.is_dir()), key=lambda s: s.stat().st_mtime)
        out += [(s, "superseded", "a place's snapshot before a rebuild, not the newest") for s in snaps[:-1]]
    return out


def size(p: Path) -> int:
    """Bytes on disk of a file or a folder."""
    r = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 0


def tidy(book: Path, do: bool = False, kinds: tuple | None = None, log=print) -> int:
    """A book's waste listed by kind with its size; removed when do. kinds: only these kinds. -> bytes found
    (freed, when do)."""
    rows = [(p, k, w) for p, k, w in waste(book) if kinds is None or k in kinds]
    total, by = 0, {}
    for p, k, w in rows:
        n = size(p)
        total += n
        by.setdefault((k, w), [0, 0])
        by[(k, w)][0] += n
        by[(k, w)][1] += 1
        if do:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)
    for (k, w), (n, c) in sorted(by.items(), key=lambda x: -x[1][0]):
        log(f"  {n / 2**30:7.1f} GB  {k}: {w} ({c})")
    log(f"{Path(book).name}: {total / 2**30:.1f} GB of waste " + ("removed" if do else "found (add --do to remove it)")
        if rows else f"{Path(book).name}: nothing to tidy")
    return total


def archive(book: Path, dest: Path, log=print) -> int:
    """The book without its waste copied to dest/<book> (only what changed since the last copy; a file gone from
    the book stays in the copy). -> rsync's return code."""
    book = Path(book)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ex = [f"--exclude=/{p.relative_to(book)}" for p, _, _ in waste(book)]
    r = subprocess.run(["rsync", "-a", *ex, f"{book}/", f"{dest / book.name}/"])
    log(f"{book.name}: " + (f"essentials copied to {dest / book.name}" if r.returncode == 0
                            else f"the copy FAILED (rsync {r.returncode})"))
    return r.returncode
