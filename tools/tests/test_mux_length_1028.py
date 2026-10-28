"""10-28: a talk piece whose audio ends with its picture lost 4 frames to ffmpeg -shortest; the runners and the stitcher
mux with the audio padded and cut at the picture's own length. Checked on a real mux."""
import subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = HERE.parent / "ltx_runner"
fails = []
src = {f: (R / f).read_text() for f in ("ltx_split.py", "ltx_split_next.py", "stitch_talk.py")}
bad = [f for f, t in src.items() if '"-shortest"' in t]
print(("PASS" if not bad else "FAIL") + " no runner muxes with -shortest", bad or "")
bad and fails.append("shortest")
d = Path(tempfile.mkdtemp(prefix="mux_"))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=320x180:r=24", "-frames:v", "57", "-c:v",
                "libopenh264", str(d / "v.mp4")], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=6.0417", str(d / "a.wav")], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "v.mp4"), "-ss", "3.6667", "-i", str(d / "a.wav"), "-map", "0:v",
                "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-t", f"{57 / 24:.6f}", str(d / "o.mp4")], check=True)
n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(d / "o.mp4")], capture_output=True, text=True).stdout)
print(("PASS" if n == 57 else "FAIL") + f" the padded mux keeps all 57 frames ({n})")
n == 57 or fails.append("frames")
sys.exit(1 if fails else 0)
