"""Frames from orbit clips (ffmpeg; stdlib only).

    python frames.py strip clip.mp4 out.png [n=12]          n evenly spaced frames, numbered, 2 rows
    python frames.py pick clip.mp4 outdir 0 24 48 ...       those frames as PNG (outdir/<clip>_f<NNN>.png)
    python frames.py tile out.png a.png b.png ... [--cols 4] [--h 320]   pictures side by side (labels = names)
"""

import os
import subprocess
import sys


def nframes(clip):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", clip], capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def strip(clip, out, n=12):
    f = nframes(clip)
    idx = [round(i * (f - 1) / (n - 1)) for i in range(n)]
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    vf = (f"select='{sel}',scale=384:-1,drawtext=text='%{{n}}':x=8:y=8:fontsize=22:fontcolor=white:box=1:"
          f"boxcolor=black@0.5,tile={n // 2}x2")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf", vf, "-frames:v", "1", "-fps_mode", "vfr", out],
                   check=True)


def pick(clip, outdir, idx):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(clip))[0]
    for i in idx:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf", f"select='eq(n\\,{i})'", "-frames:v", "1",
                        "-fps_mode", "vfr", os.path.join(outdir, f"{base}_f{int(i):03d}.png")], check=True)


def tile(out, pics, cols=4, h=320):
    args, fl = [], []
    for i, p in enumerate(pics):
        args += ["-i", p]
        lab = os.path.splitext(os.path.basename(p))[0]
        fl.append(f"[{i}]scale=-2:{h},drawtext=text='{lab}':x=6:y=6:fontsize=18:fontcolor=white:box=1:"
                  f"boxcolor=black@0.5,pad=iw+4:ih+4:2:2:white[v{i}]")
    n = len(pics)
    rows = (n + cols - 1) // cols
    lay = "|".join(f"{'+'.join(['w0'] * (i % cols)) or '0'}_{'+'.join(['h0'] * (i // cols)) or '0'}" for i in range(n))
    if n == 1:
        g = "[v0]copy[o]"
    else:
        g = "".join(f"[v{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={lay}:fill=white[o]"
    del rows
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", ";".join(fl) + ";" + g, "-map", "[o]",
                    "-frames:v", "1", out], check=True)


def main():
    a = sys.argv[1:]
    if a[0] == "strip":
        strip(a[1], a[2], int(a[3]) if len(a) > 3 else 12)
    elif a[0] == "pick":
        pick(a[1], a[2], [int(x) for x in a[3:]])
    elif a[0] == "tile":
        cols, h, pics = 4, 320, []
        it = iter(a[2:])
        for x in it:
            if x == "--cols":
                cols = int(next(it))
            elif x == "--h":
                h = int(next(it))
            else:
                pics.append(x)
        tile(a[1], pics, cols, h)


if __name__ == "__main__":
    main()
