"""Generic motion validators on per-frame joint dicts (blockout names), any story. CLI:
    python tools/motion_validate.py OUT_DIR      # every OUT/<id>/joints_<human>.npz -> offenders (JSON to stdout)
tpose_leak: both upper arms point SIDEWAYS and near-horizontal (abduction > 70 deg from hanging, > 70 % of the arm
along the body's left-right axis) for >= 6 frames, without an action that spreads the arms -- the signature of a
T/A rest pose leaking in (a failed or fallback clip)."""
import json
import os
import sys

import numpy as np

SPREAD_OK = ("spread", "arms out", "wide", "flap", "balance", "t-pose")


def _n(v):
    v = np.asarray(v, float)
    d = np.linalg.norm(v)
    return v / d if d > 1e-9 else v


def arm_abduction(J, s):
    """(angle from hanging down in deg, lateral fraction) of the upper arm (shoulder -> elbow)."""
    up = _n(np.asarray(J["neck"], float) - np.asarray(J["pelvis"], float))
    left = _n(np.asarray(J["l_shoulder"], float) - np.asarray(J["r_shoulder"], float))
    a = _n(np.asarray(J[f"{s}_elbow"], float) - np.asarray(J[f"{s}_shoulder"], float))
    ang = float(np.degrees(np.arccos(np.clip(np.dot(a, -up), -1, 1))))
    return ang, abs(float(np.dot(a, left)))


def tpose_leak(seq, prompt="", min_frames=6, deg=70.0, lateral=0.7):
    if any(k in (prompt or "").lower() for k in SPREAD_OK):
        return []
    bad = []
    for f, J in enumerate(seq):
        r = [arm_abduction(J, s) for s in ("l", "r")]
        if all(a > deg and lat > lateral for a, lat in r):
            bad.append(f)
    return bad if len(bad) >= min_frames else []


def scan(out_dir):
    rep = json.load(open(os.path.join(out_dir, "report.json")))["shots"]
    off = {}
    for sid, v in rep.items():
        for who, f in v.get("figures", {}).items():
            p = os.path.join(out_dir, sid, f"joints_{who}.npz")
            if not os.path.exists(p) or f.get("composite") or "verbs" in f and "prompt" not in f:
                continue
            z = np.load(p)
            names = [str(n) for n in z["names"]]
            if "l_elbow" not in names:
                continue
            seq = [{n: z["pos"][i, k] for k, n in enumerate(names)} for i in range(len(z["pos"]))]
            b = tpose_leak(seq, f.get("prompt", ""))
            if b:
                off[f"{sid}:{who}"] = {"frames": [b[0], b[-1], len(b)], "prompt": f.get("prompt")}
    return off


if __name__ == "__main__":
    print(json.dumps(scan(sys.argv[1]), indent=1))
