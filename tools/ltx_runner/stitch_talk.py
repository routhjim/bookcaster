"""stitch_talk.py JOBS.json OUTDIR : join chunked A2Vid lines (crossfade over the overlap), mux the padded audio,
and print a seam score (mean abs diff across the seam vs the clip's typical frame-to-frame diff)."""
import json, subprocess, sys
from collections import defaultdict
import av, numpy as np, torch
sys.argv, (jobs_f, out) = sys.argv[:1], sys.argv[1:3]
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from ltx_split import save_mp4
lines = defaultdict(list)
for j in json.load(open(jobs_f)):
    if len(j["chunk"]) == 4 and j["chunk"][1] > 1:
        lines[j["line"]].append(j)
for line, cs in lines.items():
    cs.sort(key=lambda j: j["chunk"][0])
    frames = None
    seams = []
    for j in cs:
        k, n_ch, st, n = j["chunk"]
        f = np.stack([x.to_ndarray(format="rgb24") for x in av.open(f"{out}/{j['name']}.mp4").decode(video=0)]).astype(np.float32)
        if frames is None:
            frames = f
            continue
        ov = frames.shape[0] - st
        if ov <= 1:     # 10-26: the piece starts ON the previous one's last frame: hand over there, no blend
            seams.append(float(np.abs(frames[st:] - f[:ov]).mean()) if ov else 0.0)
            frames = np.concatenate([frames[:st], f], 0)
            continue
        w = np.linspace(0, 1, ov + 2)[1:-1][:, None, None, None]
        blend = (1 - w) * frames[st:] + w * f[:ov]
        seam_raw = float(np.abs(frames[st:] - f[:ov]).mean())
        frames = np.concatenate([frames[:st], blend, f[ov:]], 0)
        seams.append(seam_raw)
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))
    name = cs[0]["name"].rsplit("_c", 1)[0]
    tmp = f"{out}/{name}_noaudio.mp4"
    save_mp4(torch.from_numpy(frames.clip(0, 255).astype(np.uint8)), tmp, cs[0]["fps"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", cs[0]["audio"], "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-shortest", f"{out}/{name}.mp4"], check=True)
    subprocess.run(["rm", "-f", tmp])
    print(json.dumps({"line": line, "out": f"{out}/{name}.mp4", "frames": int(frames.shape[0]),
                      "seam_overlap_absdiff": [round(s, 1) for s in seams],
                      "frame_diff_median": round(float(np.median(d)), 2), "frame_diff_max": round(float(d.max()), 2),
                      "frame_diff_at_seams": [round(float(d[c['chunk'][2] - 1:c['chunk'][2] + 9].max()), 2) for c in cs[1:]]}))
