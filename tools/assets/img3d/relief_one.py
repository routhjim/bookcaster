"""The land's own form of a place in the open, shaped already from the depth of the place's picture
(concept_depth.py relief), through the asset chain. Run at the image station, one per line entry:

    python tools/assets/img3d/relief_one.py ID

The piece's source spec (<library>/prop/<id>/source/spec.json) names the shape ("relief_glb") and its size; the
asset spec written here has id <id>_tr and the shape as its source, its own UVs kept (the front on the left half of
its picture, the back on the right), then run_asset.py SPEC model_img3d.py --detach (bake, engine import, stills).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C  # noqa: E402

WORK = Path(os.environ.get("IMG3D_WORK", str(C.UE_HOME / "lab" / "img3d_props")))


RELIEF_EMIT = 30.0           # a picture's land that gives the picture's own light gives it as the world does
#                             (shell_one.py WORLD_EMIT): the two are one picture and stand side by side


def spec_for(spec0: dict, glb: Path) -> dict:
    keep = {k: spec0[k] for k in ("class", "description", "tags", "placement", "cameras", "camera_preset", "piece_kind",
                                  "look") if k in spec0}
    return dict(keep, id=f"{spec0['id']}_tr", source="generated", keep_uvs=True, source_glb=str(glb), tier="hero",
                size_m=spec0["size_m"], height_m=spec0["size_m"][2], size_tolerance=0.05, yaw_deg=0, bump_m=0,
                **({"emit_texture": RELIEF_EMIT} if spec0.get("own_light") else {}),
                tags=list(keep.get("tags", [])) + ["img3d", "picture_relief"], replaces=spec0["id"],
                licence="internal: the place's picture drawn by the local image model; depth by MoGe-2 (MIT)")


def main():
    aid = sys.argv[1]
    WORK.mkdir(parents=True, exist_ok=True)
    spec0 = json.loads((C.LIBRARY / "prop" / aid / "source" / "spec.json").read_text())
    glb = Path(spec0["relief_glb"])
    if not glb.exists():
        raise SystemExit(f"no shape {glb}")
    sp = WORK / f"{aid}_tr.json"
    sp.write_text(json.dumps(spec_for(spec0, glb), indent=1))
    r = subprocess.run([sys.executable, str(HERE.parent / "run_asset.py"), str(sp), str(HERE / "model_img3d.py"), "--detach"])
    print(f"[relief_one] {time.strftime('%H:%M:%S')} {aid}: chain submitted rc={r.returncode}", flush=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
