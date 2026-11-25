"""A retrained LoRA's checkpoints are compared on their own test pictures."""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import charlora as C

ok = True
def check(what, cond):
    global ok
    ok &= bool(cond)
    print("PASS" if cond else "FAIL", what)

d = Path(tempfile.mkdtemp())
(d / "lora").mkdir(); (d / "pick").mkdir()
def put(p, t):
    p.write_bytes(b"x"); os.utime(p, (t, t))
for st in (100, 200):
    put(d / "lora" / f"step_{st:05d}.safetensors", 1000)
    for i in range(2):
        put(d / "pick" / f"step_{st:05d}_{i}.png", 2000)
check("pictures drawn after their checkpoints are current", C.pick_stale(d, 2) == [])
put(d / "lora" / "step_00200.safetensors", 3000)
bad = C.pick_stale(d, 2)
check("a checkpoint trained again makes its pictures stale", [f.name for _, f in bad] == ["step_00200_0.png", "step_00200_1.png"])
(d / "pick" / "step_00100_1.png").unlink()
check("a missing picture counts", len(C.pick_stale(d, 2)) == 3)
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)
