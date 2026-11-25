"""11-24: the hand check reads the hands of the model it has just posed, not of the last armature by name."""
import sys
from pathlib import Path
src = (Path(__file__).resolve().parent.parent / "blockout_scene.py").read_text()
i = src.index("# the posed hands (humans)")
blk = src[i:src.index("BLOCKOUT_HANDS", i)]
ok = '_MODELS.get((m["blend"], m.get("who")))' in blk and '.get("rig")' in blk
print(("PASS " if ok else "FAIL ") + "the hands come from this model's own rig")
sys.exit(0 if ok else 1)
