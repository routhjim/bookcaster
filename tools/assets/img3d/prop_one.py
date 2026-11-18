"""One library prop remade by picture-to-3D: its reference drawing -> TRELLIS.2 -> a new library asset <id>_tr
beside the scripted <id> (kept, so a set can switch back). Run at the engine station, one prop per line entry:

    python tools/assets/img3d/prop_one.py ID [--view three_quarter] [--res 1024]

1. the drawing: <refs>/<id>/<view>.png (the reference sets are drawn on a plain light background; the port keys it
   out with its own matting model);
2. img3d.py (this station's device, ROCm build) -> <work>/<id>.glb, skipped when it exists;
3. the spec: the scripted asset's spec.json (class, description, tags, tier, size_m, placement, cameras) with
   id <id>_tr, the .glb as source, keep_uvs, its height from size_m, and per-prop "regions" from
   regions.json beside this file (glass, flames: see model_img3d.py); front: the generator puts the object's front
   toward glTF +Z (-Y in the asset frame) whatever the drawing's angle, as the dragon, the person and the props
   showed; "yaw_deg" in regions.json overrides it for a piece that comes out turned;
4. the chain: run_asset.py SPEC model_img3d.py --detach (bake on the Blender card, engine import and stills); when
   the line is full the entry goes to <work>/pending_chain.txt for a later `run_asset.py --feed` pass.

Environment: ASSETS_TRELLIS_OMP (ROCm), IMG3D_WORK (default $ASSETS_UE_HOME/lab/img3d_props).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C  # noqa: E402

REFS = Path(os.environ.get("REFS_DIR", str(C.LIBRARY / "_refs")))
WORK = Path(os.environ.get("IMG3D_WORK", str(C.UE_HOME / "lab" / "img3d_props")))


def log(m):
    print(f"[prop_one] {time.strftime('%H:%M:%S')} {m}", flush=True)


def find_asset(aid):
    for cls in ("prop", "vehicle", "creature", "set"):
        d = C.LIBRARY / cls / aid
        if (d / "source" / "spec.json").exists():
            return d
    raise SystemExit(f"no scripted asset {aid} with source/spec.json in the library")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--view", default="three_quarter")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--no-chain", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    regs = json.loads((HERE / "regions.json").read_text()) if (HERE / "regions.json").exists() else {}
    extra = regs.get(a.id, {})
    view = extra.get("view", a.view)
    pic = REFS / a.id / f"{view}.png"
    if not pic.exists():
        raise SystemExit(f"no reference drawing {pic}")
    glb = WORK / f"{a.id}.glb"
    spec0 = json.loads((find_asset(a.id) / "source" / "spec.json").read_text())
    t0 = time.time()
    if not glb.exists():
        # resolution rule: the shape and texture flows cost about the square of the occupied cells at 1024 (the
        # port never backs off below 1024, so its token budget does not help there): a fireplace filled 22k
        # tokens against the dragon's 7k and took 3.3x as long. Large pieces (largest side >= LARGE_M, mostly
        # furniture and architecture that read at medium and wide) run at 512; smaller ones keep 1024.
        res = extra.get("res") or spec0.get("shape_res") or (512 if max(spec0["size_m"]) >= float(os.environ.get("IMG3D_LARGE_M", "1.2"))
                                   else a.res)
        log(f"{a.id}: res {res} (largest side {max(spec0['size_m'])} m)")
        r = subprocess.run([sys.executable, str(HERE / "img3d.py"), str(pic), str(glb), "--res", str(res),
                            "--build", os.environ.get("IMG3D_BUILD", "rocm")])
        if r.returncode:
            raise SystemExit(r.returncode)
    log(f"{a.id}: shape {time.time() - t0:.0f}s")
    src = find_asset(a.id)
    spec = json.loads((src / "source" / "spec.json").read_text())
    keep = {k: spec[k] for k in ("class", "description", "tags", "tier", "size_m", "placement", "cameras",
                                 "camera_preset", "texture_px", "sight_cut_m", "longest_m") if k in spec}
    new = dict(keep, id=f"{a.id}_tr", source="generated", keep_uvs=True, source_glb=str(glb),
               licence="internal: reference drawn by the local image model; shape and texture by TRELLIS.2 (MIT) "
                       "through trellis.cpp (MIT)",
               size_tolerance=1.0 if spec.get("longest_m") else 0.35, height_m=spec["size_m"][2], yaw_deg=extra.get("yaw_deg", spec.get("yaw_deg", "auto")),
               tags=list(keep.get("tags", [])) + ["img3d", "trellis2"], replaces=a.id)
    if extra.get("regions") or spec.get("regions"):        # (the table's by hand, else the piece's own spec's)
        new["regions"] = extra.get("regions") or spec["regions"]
    sp = WORK / f"{a.id}_tr.json"
    sp.write_text(json.dumps(new, indent=1))
    if a.no_chain:
        return 0
    r = subprocess.run([sys.executable, str(HERE.parent / "run_asset.py"), str(sp), str(HERE / "model_img3d.py"),
                        "--detach"])
    pend = WORK / "pending_chain.txt"
    if r.returncode == 75:                    # the line is full: never wait here (this holds the engine card)
        with open(pend, "a") as f:
            f.write(f"{sp} {HERE / 'model_img3d.py'}\n")
        log(f"{a.id}: line full; chain left in {pend} for a --feed pass")
        return 0
    log(f"{a.id}: chain submitted rc={r.returncode}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
