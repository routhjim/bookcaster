"""Regression test: the forging-strike recipe (approved 2026-10-19; the user's top priority) survives re-staging.

    BLOCKOUT_DRY=1 ~/tts-ab/.venv-pose/bin/python -I tools/tests/test_forge_recipe.py <book> <chapter> [<scene> <strike moment>]

Asserts on the re-staged scene: NO tongs unless the moment's text says tongs; every forging strike has the hammer
in the right hand and its strike face in contact with the work at impact (gap < 1 cm); where the work has a handle
(a sword blank), the LEFT hand holds it; sparks at the impact.
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ["BLOCKOUT_DRY"] = "1"
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import blockout as B  # noqa: E402
import scene_stage as S  # noqa: E402

book, chapter = sys.argv[1], sys.argv[2]
only = sys.argv[3] if len(sys.argv) > 3 else None
S.main([book, chapter] + (["--only", only] if only else []))
vd = S.CH.vdir_of(book)
shots = sorted((vd / chapter / "stage" / "shots").glob("*.json"))
fails, n_strikes = [], 0
for sp in shots:
    shot = json.loads(sp.read_text())
    if only and shot["scene"]["id"] != only:
        continue
    txt = (shot.get("source") or {}).get("prompt", "").lower()
    for f in shot["figures"]:
        if "tongs" in json.dumps(f.get("holds") or {}) and not re.search(r"\btongs\b", txt):
            fails.append(f"{sp.stem}: {f['who']} holds tongs the text never names")
    rec = shot.get("recipe")
    if not rec:
        continue
    if rec.get("mode") == "hold" or rec["name"] == "forge_hold":                    # a hold: NO impact contact, no sparks
        if shot.get("contacts") and any(c.get("point") == "strike_face" for c in shot["contacts"]):
            fails.append(f"{sp.stem}: a hold has a strike contact")
        if shot.get("fx"):
            fails.append(f"{sp.stem}: a hold has sparks")
        print(f"{sp.stem}: hold (no impact) ok")
        continue
    n_strikes += 1
    smith = next(f for f in shot["figures"] if (f.get("holds") or {}).get("right") == "hammer")
    if B.props.has_grip(rec["item"]) and (smith.get("holds") or {}).get("left") != rec["item"]:
        fails.append(f"{sp.stem}: the left hand does not hold the {rec['item']}")
    if not any(fx.get("type") == "sparks" for fx in shot.get("fx", [])):
        fails.append(f"{sp.stem}: no sparks at the impact")
    # the CONTACT must be in frame, in the upper 2/3, unoccluded by the smith (both keyframes)
    import copy as _c
    import numpy as _np
    import blockout_rig as _rig
    cp = shot["camera"].get("contact_point")
    if cp:
        st_ = B.stage_frames(shot, [("start", _c.deepcopy(shot["figures"]))])
        _, figs_, _, _, info_ = st_[0]
        cam_ = B.solve_camera(dict(shot["camera"]), figs_, info_, (1152, 640))
        q_ = _rig.project(cam_, [_np.asarray(cp, float)], 1152, 640)[0]
        if q_ is None or not (0 <= q_[0] <= 1152 and 0 <= q_[1] <= 640 * 2 / 3 + 1):
            fails.append(f"{sp.stem}: the contact is not in the upper 2/3 of the frame ({q_})")
        J_ = info_[smith["who"]]["joints"]
        p0, p1 = _np.asarray(cam_["pos"], float), _np.asarray(cp, float)
        for jn in ("pelvis", "spine_mid", "neck", "head"):
            if jn in J_:
                a_ = _np.asarray(J_[jn], float)
                t_ = _np.clip(_np.dot(a_ - p0, p1 - p0) / _np.dot(p1 - p0, p1 - p0), 0, 1)
                if t_ < 0.97 and _np.linalg.norm(a_ - (p0 + t_ * (p1 - p0))) < 0.18:
                    fails.append(f"{sp.stem}: the smith's {jn} hides the contact")
    res = B.run(json.loads(json.dumps(shot)), f"/tmp/forge_test/{sp.stem}", size=(640, 360))
    rows = [r for c in res["checks"].get("end", []) for r in c["rows"]]
    gaps = [r for r in rows if "hammer" in r["check"] and ("gap" in r["check"] or "contact" in r["check"]
                                                               or "deepest" in r["check"])]
    worst = max((abs(r.get("value_cm", 0.0)) for r in gaps), default=None)
    if worst is None or worst >= 1.0:
        fails.append(f"{sp.stem}: hammer-face contact gap at impact {worst} cm (needs < 1 cm)")
    print(f"{sp.stem}: work {rec['item']} left={smith['holds'].get('left')} hammer gap {worst} cm, "
          f"violations {res['violations']}")
print(json.dumps({"strikes": n_strikes, "fails": fails}, indent=1))
sys.exit(1 if fails or n_strikes == 0 else 0)
