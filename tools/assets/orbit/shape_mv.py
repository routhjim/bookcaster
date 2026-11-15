"""Multi-view image -> 3D shape (Hunyuan3D-2mv through tools/blockout_shape.py's loader), several view sets or
seeds in one model load, on one card.

    HIP_VISIBLE_DEVICES=<card> <hy3d venv python> -I shape_mv.py RUNS.json
    RUNS.json: {"out_dir": ..., "steps": 30, "octree": 384, "faces": 200000,
                "runs": [{"name": "a", "seed": 0, "views": {"front": "f.png", "left": ..., "back": ..., "right": ...}}]}

View tags follow Tencent's order: front; left = the object turned 90 degrees clockwise seen from above (its left
side to the camera, a forward-facing head points to the picture's left); back; right. Images are RGBA cut-outs.
Writes <out_dir>/<name>.glb (Y-up, as the model emits) and prints one JSON line per run.
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # tools/
import blockout_shape as BS  # noqa: E402

cfg = json.load(open(sys.argv[1]))
out = Path(cfg["out_dir"])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BS.SRC / "hy3dshape"))
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer  # noqa: E402

t0 = time.time()
pipe, keys = BS.load_mv()
print(json.dumps({"load_s": round(time.time() - t0, 1)}), flush=True)
for r in cfg["runs"]:
    t1 = time.time()
    img = {t: Image.open(p).convert("RGBA") for t, p in r["views"].items()}
    g = torch.Generator("cuda").manual_seed(int(r.get("seed", 0)))
    with torch.inference_mode():
        mesh = pipe(image=img, num_inference_steps=int(cfg.get("steps", 30)),
                    octree_resolution=int(cfg.get("octree", 384)), guidance_scale=float(cfg.get("guidance", 5.0)),
                    generator=g, num_chunks=20000, output_type="trimesh")[0]
    t2 = time.time()
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=int(cfg.get("faces", 200000)))
    mesh.export(out / f"{r['name']}.glb")
    print(json.dumps({"name": r["name"], "views": sorted(img), "gen_s": round(t2 - t1, 1),
                      "post_s": round(time.time() - t2, 1), "faces": int(len(mesh.faces)),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
