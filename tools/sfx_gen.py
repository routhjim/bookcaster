"""Generate sound effects with MOSS-SoundEffect v2.0 (Apache 2.0, 1.3B DiT +
Qwen3-1.7B text encoder, 48 kHz mono), one model load for a whole list.

    ~/src/MOSS-TTS/moss_soundeffect_v2/.venv/bin/python -I tools/sfx_gen.py jobs.json

jobs.json: [{"id": "s012", "prompt": "a heavy wooden door bursts open",
             "seconds": 2.5, "path": "/abs/s012.wav"}, ...]
The model always denoises a fixed canvas and crops it, so jobs are grouped by
canvas (10 s for spot effects, 30 s for beds) and batched. Skips jobs whose
file exists. One JSON line per finished batch on stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MOSS = Path(os.environ.get("MOSS_DIR", Path.home() / "src" / "MOSS-TTS"))
MODEL = Path(os.environ.get("MOSS_MODEL", Path.home() / "models" / "moss-soundeffect-v2"))
sys.path.insert(0, str(MOSS))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")     # its compile + CUDA graphs are untested on ROCm

import soundfile as sf                                 # noqa: E402
import torch                                           # noqa: E402

BATCH = int(os.environ.get("MOSS_BATCH", "8"))
# 30 s beds: 8 at once ran the audio decoder out of the card's 24 GB, every bed lost (10-19)
BED_BATCH = int(os.environ.get("MOSS_BED_BATCH", "2"))
STEPS = int(os.environ.get("MOSS_STEPS", "100"))


def main() -> int:
    jobs = [j for j in json.loads(Path(sys.argv[1]).read_text()) if not Path(j["path"]).exists()]
    if not jobs:
        return 0
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    torch.backends.cudnn.enabled = False               # MIOpen's conv search: slow (as elsewhere here)
    t = time.time()
    pipe = MossSoundEffectPipeline.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, device="cuda")
    print(json.dumps({"load_s": round(time.time() - t, 1)}), flush=True)
    groups: dict = {}
    for j in jobs:
        groups.setdefault(10 if float(j["seconds"]) <= 10 else 30, []).append(j)
    for canvas, group in sorted(groups.items()):
        group.sort(key=lambda j: float(j["seconds"]))
        size, k = BATCH if canvas <= 10 else BED_BATCH, 0
        while k < len(group):
            batch = group[k:k + size]
            # one length per call: the batch's longest, each cropped to its own below
            secs = round(max(float(j["seconds"]) for j in batch), 1)
            t = time.time()
            try:
                audio = pipe(prompt=[j["prompt"] for j in batch], seconds=secs, num_inference_steps=STEPS,
                             cfg_scale=4.0, sigma_shift=5.0, seed=int(batch[0].get("seed", 7)),
                             max_inference_seconds=canvas, progress_bar_cmd=lambda x, **kw: x)
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size = max(1, size // 2)       # the same jobs again, fewer at a time
                print(json.dumps({"oom": True, "batch_now": size}), flush=True)
                continue
            k += len(batch)
            audio = audio.float().cpu().numpy()
            for j, a in zip(batch, audio):
                a = a[:, :int(pipe.sample_rate * float(j["seconds"]))]
                dest = Path(j["path"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name("." + dest.name)
                sf.write(tmp, a[0], pipe.sample_rate, format="WAV")
                tmp.replace(dest)
            print(json.dumps({"batch": len(batch), "seconds": secs, "canvas": canvas,
                              "took_s": round(time.time() - t, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
