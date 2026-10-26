"""10-26 user: "Peter's sermon just gets further and further out ... the last frame from one 3s clip should be the
starting frame for the next". Chained talk pieces now start ON the previous piece's last frame (lossless PNG from the
render) and the stitch hands over there with no crossfade. CPU only; the stitch part runs in the LTX venv.
    ~/tts_reader/.venv/bin/python3 tools/tests/test_talk_chain_1026.py"""
import json, subprocess, sys, tempfile, textwrap
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import remake_chapter as R
import shot_flow as S
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

total = 734                                          # a 30.5 s line at 24 fps
env = np.abs(np.sin(np.arange(total + 200) / 7.0))
ch = R.talk_chunks(total, env, S.TALK_PIECE, S.TALK_CHAIN_OVERLAP)
check("chunks: every piece is 8m+1 frames, at most a piece long", all((n - 1) % 8 == 0 and n <= S.TALK_PIECE for _, n in ch), ch)
check("chunks: each starts ON the previous one's last frame", all(ch[k][0] == ch[k - 1][0] + ch[k - 1][1] - 1
                                                                 for k in range(1, len(ch))), ch)
check("chunks: the line is covered", ch[-1][0] + ch[-1][1] >= total, ch[-1])
check("the old 12-frame overlap stays for other flows", R.talk_chunks(total, env)[1][0] == R.talk_chunks(total, env)[0][1] - 12)
# what shot_flow writes into a chained job (chain_frame = st - pst, overlap = pst + pn - st)
(pst, pn), (st, _) = ch[0], ch[1]
check("job: chain_frame = the previous piece's last frame, overlap 1", (st - pst, pst + pn - st) == (pn - 1, 1))

stitch = textwrap.dedent(r'''
import json, sys, subprocess, numpy as np, torch, av
sys.path.insert(0, sys.argv[1] + "/ltx_runner")
d = sys.argv[2]
from ltx_split import save_mp4
# three pieces of a line: frame value = its time index (a gray ramp), piece k starts on piece k-1's last frame
pieces = [(0, 33), (32, 33), (64, 25)]
jobs = []
for k, (st, n) in enumerate(pieces):
    fr = torch.stack([torch.full((64, 96, 3), (st + i) * 2, dtype=torch.uint8) for i in range(n)])
    save_mp4(fr, f"{d}/p_c{k}.mp4", 24, crf=0)
    jobs.append({"name": f"p_c{k}", "line": "p", "chunk": [k, 3, st, n], "fps": 24, "audio": f"{d}/a.wav"})
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "4", f"{d}/a.wav"], check=True)
json.dump(jobs, open(f"{d}/jobs.json", "w"))
subprocess.run([sys.executable, "-I", sys.argv[1] + "/ltx_runner/stitch_talk.py", f"{d}/jobs.json", d], check=True, capture_output=True)
v = np.stack([x.to_ndarray(format="rgb24") for x in av.open(f"{d}/p.mp4").decode(video=0)]).astype(float)[:, 32, 48, 0]
print(json.dumps({"n": len(v), "vals": [round(x) for x in v]}))
''')
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "stitch_test.py").write_text(stitch)
    r = subprocess.run([str(R.LTX_PY), "-I", f"{d}/stitch_test.py", str(HERE), d], capture_output=True, text=True)
    out = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"err": r.stderr[-400:]}
    check("stitch: one frame per time step (89 = 64 + 25), none doubled or dropped", out.get("n") == 89, out)
    vals = out.get("vals", [])
    check("stitch: frame t shows time t everywhere, no blended seam frames",
          bool(vals) and max(abs(v - 2 * t) for t, v in enumerate(vals)) <= 3, vals[28:40])
    # the lossless last frame
    r = subprocess.run([str(R.LTX_PY), "-I", "-c", textwrap.dedent(f'''
        import sys, torch, numpy as np; from PIL import Image
        src = open("{HERE}/ltx_runner/ltx_split_next.py").read()
        code = "def save_last_png" + src.split("def save_last_png")[1].split("def save_mp4")[0]
        g = {{"torch": torch}}; exec(code, g)
        fr = torch.rand(5, 3, 40, 60) * 2 - 1
        g["save_last_png"](fr, "{d}/last.png")
        a = np.asarray(Image.open("{d}/last.png"), np.int16)
        ref = ((fr[-1].permute(1, 2, 0) + 1) / 2 * 255).round().clamp(0, 255).to(torch.int16).numpy()
        print(int(np.abs(a - ref).max()))''')], capture_output=True, text=True)
    check("last frame saved lossless (max error 0)", r.stdout.strip() == "0", (r.stdout, r.stderr[-300:]))
print(json.dumps({"fails": fails}))
