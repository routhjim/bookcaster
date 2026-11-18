"""One cabin people sit inside, or one place in the open (a piece made_as "world": pano_shell.py --outdoor), made as
a shell from its panorama picture (pano_shell.py), then the asset chain. Run at the image station, one per line
entry:

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
                shell_build=facts.get("shell_build", 1),
                tags=list(keep.get("tags", [])) + ["img3d", "panorama_shell"], replaces=spec0["id"],
                licence="internal: panorama drawn by the local image model; depth by MoGe-2 (MIT), panes by CLIPSeg (MIT)")


WORLD_EMIT = 30.0            # an outdoor place's land and sky give the picture's own light (as a view's backdrop,
#                             sets/place_build.py VIEW_EMIT): lit only by the scene's lights, the sky dome came out
#                             shaded like a wall


def world_spec_for(spec0: dict, facts: dict, glb: Path) -> dict:
    """The asset spec of a made outdoor place (pano_shell.py --outdoor): its measured size, where people stand,
    the sky's light; seen from inside, giving its own light."""
    keep = {k: spec0[k] for k in ("class", "description", "tags", "placement", "cameras", "camera_preset") if k in spec0}
    return dict(keep, id=f"{spec0['id']}_tr", source="generated", keep_uvs=True, source_glb=str(glb), tier="hero",
                size_m=facts["size_m"], height_m=facts["size_m"][2], size_tolerance=0.05, yaw_deg=0, bump_m=0,
                seen_from_inside=True, eye_m=facts["eye_m"], stand_m=facts["stand_m"], flat_r_m=facts["flat_r_m"],
                spot_m=facts.get("spot_m"),
                sun=facts.get("sun"), world_build=facts.get("world_build", 1), emit_texture=WORLD_EMIT,
                tags=list(keep.get("tags", [])) + ["img3d", "panorama_world"], replaces=spec0["id"],
                licence="internal: panorama drawn by the local image model; depth by MoGe-2 (MIT), sky by CLIPSeg (MIT)")


def main():
    aid = sys.argv[1]
    WORK.mkdir(parents=True, exist_ok=True)
    pic = REFS / aid / "picture.png"
    if not pic.exists():
        raise SystemExit(f"no panorama {pic}")
    glb = WORK / f"{aid}.glb"
    spec0 = json.loads((C.LIBRARY / "prop" / aid / "source" / "spec.json").read_text())
    import pano_shell
    world = spec0.get("made_as") == "world"             # a place in the open (pano_shell.py --outdoor)
    key, now = ("world_build", pano_shell.WORLD_BUILD) if world else ("shell_build", pano_shell.SHELL_BUILD)
    old = json.loads(glb.with_suffix(".json").read_text()).get(key, 1) if glb.with_suffix(".json").exists() else 0
    if not glb.exists() or old < now:                     # (a shell of an older build is made again)
        r = subprocess.run([os.environ.get("ASSETS_SHELL_PY") or sys.executable, str(HERE / "pano_shell.py"), str(pic), str(glb)]
                           + (["--outdoor"] if world else []) + (["--spot", spec0["spot"]] if world and spec0.get("spot") else [])
                           + [x for k, v in ((spec0.get("marks") or {}).items() if world else []) for x in ("--mark", f"{k}={v}")])
        if r.returncode:
            glb.unlink(missing_ok=True)
            raise SystemExit(r.returncode)
    sp = WORK / f"{aid}_tr.json"
    facts = json.loads(glb.with_suffix(".json").read_text())
    sp.write_text(json.dumps((world_spec_for if world else spec_for)(spec0, facts, glb), indent=1))
    r = subprocess.run([sys.executable, str(HERE.parent / "run_asset.py"), str(sp), str(HERE / "model_img3d.py"), "--detach"])
    log(f"{aid}: chain submitted rc={r.returncode}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
