"""Image-to-3D mesh -> rigged character for the blockout (prep -> DWPose -> rig), CPU only.

    ~/tts-ab/.venv-pose/bin/python -I tools/blockout_model.py MESH.glb OUT_DIR [--height 1.8] [--yaw 90]
                                   [--texture cut.png] [--faces 60000] [--size 1024]

MESH.glb is tools/blockout_shape.py's output (Hunyuan3D-2.1, glTF Y-up; Blender imports it Z-up facing -Y,
so --yaw 90 turns it to face +X, the rest pose tools/blockout_scene.py's import_model() expects).
Steps (each one a Blender run with -t 16, or rtmlib on the CPU):
    1. prep   tools/blockout_model_bl.py: normalise (Z-up, facing +X, feet at 0, --height), decimate,
              paint vertex colours from --texture (front projection), save OUT_DIR/model.blend, render an
              orthographic front view OUT_DIR/front.png (+ front.json: the ortho camera)
    2. DWPose rtmlib Body (onnxruntime CPU) on front.png -> 18 COCO points (OUT_DIR/front_kps.json,
              drawn over the render in OUT_DIR/front_kps.png)
    3. rig    tools/blockout_model_bl.py: lift the joints to 3D (middle of the mesh along each view ray),
              build the named armature, skin with bone heat (envelope fallback) -> OUT_DIR/rigged.blend,
              OUT_DIR/rig.json
Prints one JSON line {"out", "rigged", "skin", "unweighted", "seconds": {...}}.
Use the result in a shot: figure "model": "OUT_DIR/rigged.blend" (tools/blockout.py).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BLENDER = os.environ.get("BLOCKOUT_BLENDER", os.path.expanduser("~/opt/blender/current/blender"))


def blender(job: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(job, f, indent=1)
    r = subprocess.run([BLENDER, "--background", "--factory-startup", "-t", os.environ.get("BLOCKOUT_THREADS", "16"),
                        "--python", os.path.join(HERE, "blockout_model_bl.py"), "--", path],
                       capture_output=True, text=True, timeout=float(os.environ.get("BLOCKOUT_TIMEOUT", 900)))
    if r.returncode or "Traceback" in r.stderr:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise RuntimeError(f"blender {job['op']} failed")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--height", type=float, default=1.8)
    ap.add_argument("--yaw", type=float, default=90.0)
    ap.add_argument("--texture")
    ap.add_argument("--texture-back", help="a back-view picture (same character, A-pose) for the back")
    ap.add_argument("--texture-left", help="the left-side profile (the person's left side, facing picture-left)")
    ap.add_argument("--texture-right", help="the right-side profile (facing picture-right)")
    ap.add_argument("--faces", type=int, default=60000)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--thr", type=float, default=0.3)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    T = {}
    t0 = time.time()
    job = {"op": "prep", "mesh": os.path.abspath(a.mesh), "height": a.height, "yaw": a.yaw, "faces": a.faces,
           "out_blend": f"{out}/model.blend", "front_png": f"{out}/front.png", "size": a.size}
    if a.texture:
        job["texture"] = {"image": os.path.abspath(a.texture)}
        for k_ in ("back", "left", "right"):
            if getattr(a, f"texture_{k_}"):
                job["texture"][k_] = os.path.abspath(getattr(a, f"texture_{k_}"))
    blender(job, f"{out}/prep.json")
    T["prep"] = round(time.time() - t0, 1)

    t1 = time.time()
    from poses import extract_pose, draw_people
    kps, scores = extract_pose(f"{out}/front.png", thr=a.thr)
    with open(f"{out}/front_kps.json", "w") as f:
        json.dump({"kps": kps, "scores": scores}, f)
    import numpy as np
    from PIL import Image
    base = np.asarray(Image.open(f"{out}/front.png").convert("RGB")).astype(np.float32)
    sk = draw_people([kps], a.size, a.size).astype(np.float32)
    m = sk.max(-1, keepdims=True) > 0
    Image.fromarray(np.where(m, sk, base * 0.7).astype(np.uint8)).save(f"{out}/front_kps.png")
    T["dwpose"] = round(time.time() - t1, 1)
    need = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    missing = [i for i in need if kps[i] is None]
    if missing:
        raise RuntimeError(f"DWPose missed body joints {missing} on {out}/front.png (see front_kps.png)")

    t2 = time.time()
    so = blender({"op": "rig", "blend": f"{out}/model.blend", "front_json": f"{out}/front.json", "kps": kps,
                  "out_blend": f"{out}/rigged.blend", "out_json": f"{out}/rig.json"}, f"{out}/rig_job.json")
    T["rig"] = round(time.time() - t2, 1)
    rj = json.load(open(f"{out}/rig.json"))
    print(json.dumps({"out": out, "rigged": f"{out}/rigged.blend", "skin": rj["skin"],
                      "unweighted": rj["unweighted"], "verts": rj["verts"], "seconds": T,
                      "rig_log": [ln for ln in so.splitlines() if ln.startswith("RIG")]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
