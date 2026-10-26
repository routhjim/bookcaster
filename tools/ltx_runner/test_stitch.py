"""Regression test (CPU): chunked talk lines are joined ONCE, over the overlap, with the line's own audio.
    ~/tts-ab/.venv-ltx/bin/python tools/ltx_runner/test_stitch.py
1. stitch_talk.py crossfades the 12-frame overlap: frames = start(c1) + n(c1), not n0 + n1 (no repeated 0.5 s);
2. the stitched mp4's audio is the padded line track from 0 (never the chunks' audio concatenated);
3. the chapter assembly (video.build -> _anim_clip) drops every clip's audio (-an): the chapter mix is the only sound."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
td = Path(tempfile.mkdtemp(prefix="stitch_test_"))
n0, n1, ov = 49, 41, 12
st1 = n0 - ov
for name, n, col in (("r_x_c0", n0, "red"), ("r_x_c1", n1, "blue")):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color={col}:s=64x64:r=24", "-frames:v", str(n),
                    "-pix_fmt", "yuv420p", str(td / f"{name}.mp4")], check=True)
dur = (st1 + n1) / 24 + 0.2
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur:.3f}", "-ac", "2",
                str(td / "pad.wav")], check=True)
jobs = [{"name": "r_x_c0", "line": "x", "fps": 24, "audio": str(td / "pad.wav"), "chunk": [0, 2, 0, n0]},
        {"name": "r_x_c1", "line": "x", "fps": 24, "audio": str(td / "pad.wav"), "chunk": [1, 2, st1, n1]}]
(td / "jobs.json").write_text(json.dumps(jobs))
r = subprocess.run([sys.executable, str(HERE / "stitch_talk.py"), str(td / "jobs.json"), str(td)], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-800:]
out = td / "r_x.mp4"
nf = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                         "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip())
assert nf == st1 + n1, f"stitched {nf} frames, want {st1 + n1} (n0 + n1 = {n0 + n1} would repeat the overlap)"
ad = float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration",
                           "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip())
assert abs(ad - nf / 24) < 0.1, f"audio {ad:.2f}s vs video {nf / 24:.2f}s: not the single line track"
src = open("/home/jrouth/tts_reader/tts_reader/video.py").read()
body = src[src.index("def _anim_clip"):src.index("def _segments")]
assert '"-an"' in body, "the assembly must drop clip audio (-an)"
print(f"ok: {n0}+{n1} frames with a {ov}-frame overlap -> {nf}; audio {ad:.2f}s = the line track; assembly uses -an")
