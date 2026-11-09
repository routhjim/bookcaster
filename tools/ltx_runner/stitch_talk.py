"""stitch_talk.py JOBS.json OUTDIR : join chunked A2Vid lines (crossfade over the overlap), mux the padded audio,
and print a seam score (mean abs diff across the seam vs the clip's typical frame-to-frame diff).

Each piece's own lip-sync shift (its job's "sync_shift": frames to show its picture earlier, + = its lips lag its
voice; None = no confident reading, 0) is applied inside the join (E29, 11-08: one shift for the whole joined line
left a part of 13 of 37 lines 3+ frames off): the joined picture is read through a time map P(t) = t + shift(t),
the shift each piece's own over its stretch and ramped from one to the next across each seam (where the voice
pauses: the cuts are put there) over RAMP frames or more, so the picture never jumps or runs backwards and the
handover frame the next piece started from is passed through in order. Frame count and audio are unchanged (the
ends hold the first / last frame, as the assembly does for a whole clip). The line's shifts go to <name>.join.json."""
import json, os, subprocess, sys
from collections import defaultdict
import numpy as np
RAMP = 8                # the shortest move from one piece's shift to the next (4 frames a frame of change, or more)


def shift_map(n: int, starts: list, shifts: list) -> np.ndarray:
    """Source frame for each of the n joined frames: t + the shift of the piece t is in, ramped linearly across each
    seam (centred on the piece's start) over max(RAMP, 4 x the change) frames, at most half of either piece; never
    decreasing, clamped to the clip."""
    s = np.zeros(n)
    for k, st in enumerate(starts):
        s[st:] = shifts[k]
    for k in range(1, len(starts)):
        dl = shifts[k] - shifts[k - 1]
        if not dl:
            continue
        nxt = starts[k + 1] if k + 1 < len(starts) else n
        half = min(max(RAMP, 4 * abs(dl)) // 2, (starts[k] - starts[k - 1]) // 2, (nxt - starts[k]) // 2)
        a, b = starts[k] - half, starts[k] + half
        if b > a:
            s[a:b] = shifts[k - 1] + dl * (np.arange(a, b) - a + 0.5) / (b - a)
    return np.maximum.accumulate(np.clip(np.arange(n) + np.round(s).astype(int), 0, n - 1))


def main(jobs_f: str, out: str, only: set | None) -> None:
    import av, torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ltx_split import save_mp4
    lines = defaultdict(list)
    for j in json.load(open(jobs_f)):
        if len(j["chunk"]) == 4 and j["chunk"][1] > 1 and (only is None or j["line"] in only):
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
        sh = [j.get("sync_shift") for j in cs]
        frames = frames.clip(0, 255).astype(np.uint8)   # (the map indexes the 8-bit copy: a 9-piece line is ~15 GB
        frames = frames[shift_map(frames.shape[0], [j["chunk"][2] for j in cs], [x or 0 for x in sh])]  # in float)
        d = np.array([np.abs(frames[i + 1].astype(np.int16) - frames[i]).mean() for i in range(frames.shape[0] - 1)])
        name = cs[0]["name"].rsplit("_c", 1)[0]
        tmp = f"{out}/{name}_noaudio.mp4"
        save_mp4(torch.from_numpy(frames), tmp, cs[0]["fps"])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", cs[0].get("audio_mux", cs[0]["audio"]), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-af", "apad", "-t", f"{frames.shape[0] / cs[0]['fps']:.6f}",   # (10-28: not -shortest:
                        f"{out}/{name}.mp4"], check=True)                                      # it cut whole packets)
        subprocess.run(["rm", "-f", tmp])
        json.dump({"shifts": sh, "frames": int(frames.shape[0])}, open(f"{out}/{name}.join.json", "w"))
        print(json.dumps({"line": line, "out": f"{out}/{name}.mp4", "frames": int(frames.shape[0]), "shifts": sh,
                          "seam_overlap_absdiff": [round(s, 1) for s in seams],
                          "frame_diff_median": round(float(np.median(d)), 2), "frame_diff_max": round(float(d.max()), 2),
                          "frame_diff_at_seams": [round(float(d[c['chunk'][2] - 1:c['chunk'][2] + 9].max()), 2) for c in cs[1:]]}))


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.argv = sys.argv[:1]
    # (10-28: the third argument - the lines to join, one line as its pieces land)
    main(args[0], args[1], set(args[2].split(",")) if len(args) > 2 and args[2] else None)
