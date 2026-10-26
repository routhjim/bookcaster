"""The keyframe preview of a 3D chapter: every moment's key (or the chapter still between moments) on the final
audio mix, labelled, BEFORE any LTX time (user 10-21: review the keys first). CPU only, ~1 min.

    python3 tools/keys_preview.py <book> <chapter stem> [--audio score_sfx.mp3] [--out file.mp4]

A moment without a key shows its still picture with "NO KEY" in red, so a gap is seen, never hidden.

--staging (user 10-22): the same cut from the 3D staging renders instead of the keys, before any key is drawn: each
moment's start frame for its first half, its end frame for the second (render/start_color.png, end_color.png).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 960, 540


def _frame(src: Path | None, label: str, warn: bool, out: Path) -> None:
    im = Image.new("RGB", (W, H), (0, 0, 0))
    if src and src.exists():
        s = Image.open(src).convert("RGB")
        s.thumbnail((W, H))
        im.paste(s, ((W - s.width) // 2, (H - s.height) // 2))
    d = ImageDraw.Draw(im)
    d.rectangle((0, H - 24, W, H), fill=(0, 0, 0))
    d.text((8, H - 19), label, fill=(255, 80, 80) if warn else (255, 255, 0))
    im.save(out)


def build(book: Path, stem: str, audio: Path | None = None, out: Path | None = None, staging: bool = False) -> Path:
    vd = book / "work" / "video"
    cd = vd / stem
    fo = cd / "flow3d"
    man = json.loads((cd / "anim" / "manifest.json").read_text())
    dur_all = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(audio)], capture_output=True, text=True).stdout.strip() or 0) if audio else 0
    man = sorted(man, key=lambda e: e["t"])
    fo.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="keys_preview_", dir=fo))
    lines, n_missing = [], 0

    def add(png, d):
        lines.append(f"file '{png}'\nduration {max(d, 0.04):.3f}\n")
    for i, e in enumerate(man):
        t0 = e["t"]
        t1 = man[i + 1]["t"] if i + 1 < len(man) else (dur_all or t0 + e.get("dur", 5))
        key = fo / "stage" / e["id"] / "key.png"
        still = cd / "img" / f"{e.get('pic')}.png" if e.get("pic") else None
        if i == 0 and t0 > 0.05:                      # the opening before the first moment: its still
            p = tmp / "lead.png"
            _frame(still, f"{stem}  0:00  still", False, p)
            add(p, t0)
        mm, ss = divmod(int(t0), 60)
        p = tmp / f"{i:03d}.png"
        if staging:
            rd = fo / "stage" / e["id"] / "render"
            a, b = rd / "start_color.png", rd / "end_color.png"
            ok = a.exists()
            n_missing += not ok
            lab = f"{e['id']}  {mm}:{ss:02d}  {e['kind']}  staging"
            _frame(a if ok else still, lab + ("" if ok else "  NOT STAGED"), not ok, p)
            if ok and b.exists():
                _frame(b, lab + " (end)", False, tmp / f"{i:03d}b.png")
                add(p, (t1 - t0) / 2)
                add(tmp / f"{i:03d}b.png", (t1 - t0) / 2)
            else:
                add(p, t1 - t0)
            continue
        ok = key.exists()
        n_missing += not ok
        _frame(key if ok else still, f"{e['id']}  {mm}:{ss:02d}  {e['kind']}" + ("" if ok else "  NO KEY"), not ok, p)
        add(p, t1 - t0)
    lst = tmp / "list.txt"
    lst.write_text("".join(lines) + lines[-1].split("\n")[0] + "\n")
    out = out or book / "out" / f"{book.name}_{stem}_{time.strftime('%m%d-%H%M')}_{'staging' if staging else 'keys'}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio:
        cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-vf", "fps=12,format=yuv420p", "-c:v", "libopenh264", "-b:v", "900k", str(out)]
    subprocess.run(cmd, check=True)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    (out.with_suffix(".json")).write_text(json.dumps({"moments": len(man), "no_key": n_missing}, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("stem")
    ap.add_argument("--audio")
    ap.add_argument("--out")
    ap.add_argument("--staging", action="store_true", help="the staging renders instead of the keys")
    a = ap.parse_args()
    b = Path(a.book).expanduser()
    if not a.audio:                                  # the chapter's mix with its effects, as the video uses
        import flow3d
        a.audio = str(flow3d._mix_with_sfx(b, a.stem, Path(tempfile.mkdtemp(prefix="preview_audio_"))))
    print(build(b, a.stem, Path(a.audio), Path(a.out) if a.out else None, a.staging))
