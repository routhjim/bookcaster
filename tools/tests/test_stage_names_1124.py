"""11-24: a figure is found in the cast index under any spelling of its title; a pose with no mid-spine joint still
interpolates."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import motion_batch as MB
import blockout_video as BV

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

with tempfile.TemporaryDirectory() as td:
    d = Path(td) / "characters" / "dr_orla"
    d.mkdir(parents=True)
    (d / "rigged.blend").write_bytes(b"x")
    idx = Path(td) / "characters" / "index.json"
    idx.write_text(json.dumps({"characters": {"Dr. Orla": {"kind": "human", "outfits": {"": {"dir": str(d)}}}}}))
    keep = MB.CAST["index"]
    MB.CAST["index"] = str(idx)
    try:
        check("the index's own spelling", MB.rig_for("Dr. Orla") == str(d / "rigged.blend"), MB.rig_for("Dr. Orla"))
        check("the title spelled out", MB.rig_for("Doctor Orla") == str(d / "rigged.blend"), MB.rig_for("Doctor Orla"))
        check("somebody else has no body", MB.rig_for("Doctor Tam") is None, MB.rig_for("Doctor Tam"))
    finally:
        MB.CAST["index"] = keep

v = lambda *a: np.array(a, float)
J0 = {"pelvis": v(0, 0, 1), "neck": v(0, 0, 1.5), "head": v(0, 0, 1.7)}
J1 = {"pelvis": v(1, 0, 1), "neck": v(1, 0, 1.5), "head": v(1, 0.2, 1.7)}
try:
    out = BV.interp_joints(J0, J1, 0.5)
    check("no mid-spine: the neck and head still move", abs(out["neck"][0] - 0.5) < 1e-6 and "head" in out, out)
    check("the poses given are not changed", "spine_mid" not in J0 and "spine_mid" not in J1)
except KeyError as exc:
    check("no mid-spine: the neck and head still move", False, repr(exc))
J0 = {"pelvis": v(0, 0, 1), "head": v(0, 0, 1.7)}
J1 = {"pelvis": v(1, 0, 1), "head": v(1, 0, 1.7)}
out = BV.interp_joints(J0, J1, 0.5)
check("a joint whose parent is missing moves in a straight line", abs(out["head"][0] - 0.5) < 1e-6, out.get("head"))

sys.exit(1 if fails else 0)
