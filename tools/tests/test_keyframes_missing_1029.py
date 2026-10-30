"""10-29: a pace change deletes anim/ (the moment list is kept, re-timed); the shots flow then died reading
anim/keyframes.json. Its readers take an empty one."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import mannequin_flow as MF
vd = Path(tempfile.mkdtemp())
kf = MF._keyframes(vd)
ok = kf == {"frames": [], "props": {}}
(vd / "anim").mkdir(); (vd / "anim" / "keyframes.json").write_text(json.dumps({"frames": [{"shot": "a"}]}))
ok = ok and MF._keyframes(vd)["frames"] == [{"shot": "a"}]
print("PASS" if ok else "FAIL", "missing keyframes.json reads as empty; a present one as itself")
print(json.dumps({"fails": [] if ok else ["keyframes"]}))
sys.exit(0 if ok else 1)
