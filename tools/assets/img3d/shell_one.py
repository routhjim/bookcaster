"""One cabin people sit inside, made as a shell from its panorama picture (pano_shell.py), then the asset chain. Run
at the image station, one cabin per line entry:

    python tools/assets/img3d/shell_one.py ID

1. the picture: <refs>/<id>/picture.png, an equirectangular panorama of the cabin's inside from its seat;
2. pano_shell.py (the interpreter with the depth model: ASSETS_SHELL_PY, default this one) -> <work>/<id>.glb and
   <work>/<id>.json (size, eye, seat);
3. the spec: the piece's source spec with id <id>_tr, the .glb as source, its size and seat as measured, faces
   inward ("seen_from_inside");
4. the chain: run_asset.py SPEC model_img3d.py --detach (bake, engine import, stills).
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

REFS = Path(os.environ.get("REFS_DIR", str(C.LIBRARY / "_refs")))
WORK = Path(os.environ.get("IMG3D_WORK", str(C.UE_HOME / "lab" / "img3d_props")))


def log(m):
    print(f"[shell_one] {time.strftime('%H:%M:%S')} {m}", flush=True)


def spec_for(spec0: dict, facts: dict, glb: Path) -> dict:
    """The asset spec of a made shell: the piece's own spec with the shell's measured size and seat."""
    keep = {k: spec0[k] for k in ("class", "description", "tags", "placement", "cameras", "camera_preset") if k in spec0}
    return dict(keep, id=f"{spec0['id']}_tr", source="generated", keep_uvs=True, source_glb=str(glb), tier="hero",
                size_m=facts["size_m"], height_m=facts["size_m"][2], size_tolerance=0.05, yaw_deg=0, bump_m=0,
                seen_from_inside=True, seat=facts["seat"], eye_m=facts["eye_m"], regions=facts.get("regions") or [],
                tags=list(keep.get("tags", [])) + ["img3d", "panorama_shell"], replaces=spec0["id"],
                licence="internal: panorama drawn by the local image model; depth by MoGe-2 (MIT), panes by CLIPSeg (MIT)")


def main():
    aid = sys.argv[1]
    WORK.mkdir(parents=True, exist_ok=True)
    pic = REFS / aid / "picture.png"
    if not pic.exists():
        raise SystemExit(f"no panorama {pic}")
    glb = WORK / f"{aid}.glb"
    spec0 = json.loads((C.LIBRARY / "prop" / aid / "source" / "spec.json").read_text())
    if not glb.exists() or not glb.with_suffix(".json").exists():
        r = subprocess.run([os.environ.get("ASSETS_SHELL_PY") or sys.executable, str(HERE / "pano_shell.py"), str(pic), str(glb)])
        if r.returncode:
            glb.unlink(missing_ok=True)
            raise SystemExit(r.returncode)
    sp = WORK / f"{aid}_tr.json"
    sp.write_text(json.dumps(spec_for(spec0, json.loads(glb.with_suffix(".json").read_text()), glb), indent=1))
    r = subprocess.run([sys.executable, str(HERE.parent / "run_asset.py"), str(sp), str(HERE / "model_img3d.py"), "--detach"])
    log(f"{aid}: chain submitted rc={r.returncode}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
