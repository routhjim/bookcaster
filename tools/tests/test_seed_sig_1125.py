"""A seed's drawing is kept only when it was made from the very same request."""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import keyframe_draw as K

ok = True
def check(what, cond):
    global ok
    ok &= bool(cond)
    print("PASS" if cond else "FAIL", what)

d = Path(tempfile.mkdtemp())
still, mask = d / "start_color.png", d / "mask.png"
still.write_bytes(b"still one"); mask.write_bytes(b"mask")
wd = d / "key_seeds"; wd.mkdir()
def job(**k):
    j = {"id": "s11", "prompt": "Orla in the Hall", "seed": 11, "size": [64, 64], "path": str(wd / "s11.png"),
         "control": [{"image": str(still), "type": "depth", "scale": 1.0}],
         "init": {"image": str(still), "strength": 0.4, "mask": str(mask)}}
    j.update(k)
    return j
check("nothing drawn yet: to draw", not K.keep_or_drop(job()))
(wd / "s11.png").write_bytes(b"drawing"); (wd / "s11_likeness.json").write_text("{}"); (wd / "k003_s11_check.json").write_text("{}")
(wd / "k003_s7_check.json").write_text("{}")
check("the same request: kept", K.keep_or_drop(job()) and (wd / "s11.png").exists())
check("... under another list id too", K.keep_or_drop(job(id="k000_s11")))
os.utime(still, (1, 1)); os.utime(mask, (1, 1))
check("a picture written again with the same contents: kept", K.keep_or_drop(job()))
still.write_bytes(b"still two")
check("a new still: dropped", not K.keep_or_drop(job()) and not (wd / "s11.png").exists())
check("... with what was measured on it", not (wd / "s11_likeness.json").exists() and not (wd / "k003_s11_check.json").exists())
check("... and nothing of another seed", (wd / "k003_s7_check.json").exists())
(wd / "s11.png").write_bytes(b"drawing 2")
check("drawn from the new still: kept", K.keep_or_drop(job()))
check("other words: dropped", not K.keep_or_drop(job(prompt="Tam in the Hall")))
(wd / "s11.png").write_bytes(b"d3"); K.keep_or_drop(job(prompt="Tam in the Hall"))
check("another strength: dropped", not K.keep_or_drop(job(prompt="Tam in the Hall", init={"image": str(still), "strength": 0.6})))
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)
