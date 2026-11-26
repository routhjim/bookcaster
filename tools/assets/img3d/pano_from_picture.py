"""A place in the open's panorama made from the place's own picture: the picture-to-panorama model (tools/world.py
stage_pano: the edit model with its panorama LoRA, 8 steps) keeps the picture as the view straight ahead and paints
the rest of the way round in its look. Run at the image station, one per line entry:

    python tools/assets/img3d/pano_from_picture.py PICTURE REF_ID --desc "the land's words" [--seed N]

Writes <library>/_refs/<REF_ID>/picture.png, 2048 x 1024, turned so the place's picture is at its left and right
edges: pano_shell.py's frame has the middle of a panorama at -y, and the picture looks along +y, as the place's
things are placed (sets/place_build.py open_places).

Why: the panorama drawn from words alone, told of bare level ground with everything far off (so nothing built on the
set stood in it twice), was the whole land of every place in the open: flat nothing on every side (11-19). The
place's picture is the place; its panorama is that picture carried on all the way round.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C  # noqa: E402

REFS = C.LIBRARY / "_refs"
CARD_SH = HERE.parent / "orbit" / "card.sh"
EDIT_PY = os.path.expanduser(os.environ.get("ASSETS_GUIDES_PY", "~/tts-ab/.venv-hyworld/bin/python"))
STEPS = 8                    # with the Lightning LoRA at 0.7 (world.py PANO_LIGHTNING_W: 1.0 painted a crackle)
SIZE = (2048, 1024)


def card() -> str:
    st = str(C.IMAGE_STATION)
    return st[4:] if st.startswith("card") and st[4:] else C.BLENDER_CARD


def turned(im):
    """The panorama with its middle (the place's picture) moved to its left and right edges."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.convert("RGB"))
    return Image.fromarray(np.roll(a, a.shape[1] // 2, axis=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("picture")
    ap.add_argument("ref_id")
    ap.add_argument("--desc", default="")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = REFS / a.ref_id / "picture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    repo = HERE.parents[2]
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "pano.png"
        cmd = (f"cd {shlex.quote(str(repo))} && WORLD_PANO_LIGHTNING=1 WORLD_PANO_LIGHTNING_W=0.7 "
               f"bash {shlex.quote(str(CARD_SH))} {card()} {shlex.quote(EDIT_PY)} -I tools/world.py _pano "
               f"{shlex.quote(str(a.picture))} {shlex.quote(str(raw))} {shlex.quote(a.desc)} {a.seed} {STEPS}")
        r = subprocess.run(["bash", "-c", cmd])
        if r.returncode or not raw.exists():
            print(f"[pano_from_picture] {time.strftime('%H:%M:%S')} {a.ref_id}: no panorama rc={r.returncode}", flush=True)
            return r.returncode or 1
        from PIL import Image
        turned(Image.open(raw)).resize(SIZE, Image.LANCZOS).save(out)
    print(f"[pano_from_picture] {time.strftime('%H:%M:%S')} {a.ref_id}: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
