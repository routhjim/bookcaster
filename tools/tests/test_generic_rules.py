"""Generic staging rules on SYNTHETIC data (no book, no story names, CPU only, seconds).

    ~/tts-ab/.venv-pose/bin/python -I tools/tests/test_generic_rules.py

Each check is one rule of eval/blockout/RULES.md; prints PASS/FAIL per rule, exit 1 on any failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import blockout_plan as BP  # noqa: E402
import props_build as PB  # noqa: E402
import scene_stage as S  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def room(walls):
    return {"walls": walls, "capture_xy": [0.0, 0.0], "floor_polygon_xy": [[-5, -5], [5, -5], [5, 5], [-5, 5]]}


def shot_with(props, figs=((0.0, 2.0),)):
    return {"set": {"props": [dict(p) for p in props]},
            "figures": [{"who": f"P{i}", "at": list(a)} for i, a in enumerate(figs)], "camera": {}}


# --- data-driven classes --------------------------------------------------------------------------------
fixed = set(S.FIXED_WORDS)
check("fixed props come from the prop-class mobility table",
      all(any(w in fixed for w in r["words"]) for r in PB.CLASSES["physical"] if r.get("mobility") == "fixed"))
for w in ("door", "window", "gate", "hearth"):
    check(f"'{w}' is architectural (class flag)", S._is_arch(w))
check("a movable prop is not architectural", not S._is_arch("sword") and not S._is_arch("cooking pot"))

# --- architecture: never free-standing ---------------------------------------------------------------------
# (1) the set's own anchor
anc = {"door": {"at": [1.0, 4.0, 0.0], "normal": [0.0, -1.0, 0.0], "tangent": [1.0, 0.0, 0.0], "width": 1.0,
                "height": 2.2, "lintel_z": 2.2}}
sc = {"anchors": anc, "room": room([{"n": [0.0, -1.0, 0.0], "d": 4.0}]), "capture_xy": [0, 0], "world_kind": "interior",
      "reference_camera": {"pos": [0.0, -1.0, 1.6]}}
sh = shot_with([{"name": "door", "box": [0, 0, 0, 1, 1, 1]}])
S.mount_arch(sh["set"]["props"][0], sc, sh)
pr = sh["set"]["props"][0]
check("arch: the set's own door is used (hidden, at its anchor)", pr.get("hidden") and
      np.allclose(pr["wall"]["centre"], [1.0, 4.0]), pr.get("anchored"))
# (2) no anchor, interior walls: flush in the nearest wall
sc2 = dict(sc, anchors={})
sh2 = shot_with([{"name": "wooden door", "box": [0, 2.5, 0, 1, 1, 1]}], figs=((0.5, 2.0),))
S.mount_arch(sh2["set"]["props"][0], sc2, sh2)
c2 = np.array(sh2["set"]["props"][0]["wall"]["centre"])
check("arch: no anchor -> flush on the nearest room wall", abs(c2[1] - 4.0) < 1e-6, f"centre {c2.tolist()}")
# (3) exterior: it brings a facade >= 3 m wide, and the validator passes on it
sc3 = {"anchors": {}, "room": None, "capture_xy": [0, 0], "world_kind": "exterior",
       "reference_camera": {"pos": [0.0, -1.0, 1.6]}}
sh3 = shot_with([{"name": "gate", "box": [0, 0, 0, 1, 1, 1]}], figs=((0.0, 3.0),))
S.mount_arch(sh3["set"]["props"][0], sc3, sh3)
fac = [p for p in sh3["set"]["props"] if p.get("prop") == "facade"]
check("arch: exterior -> a facade >= 3 m wide behind it", fac and fac[0]["box"][3] >= 3.0)
res = S.arch_wall_check(sh3, sc3, "synthetic")
check("validator: facade-backed architecture passes (>=80% wall behind)", res and res[0]["ok"], res)
# (4) validator rejects a free-standing one
sh4 = shot_with([{"name": "door", "box": [0, 3, 0, 1, 1, 1]}])
res4 = S.arch_wall_check(sh4, sc3, "synthetic")
check("validator: a free-standing door fails", res4 and not res4[0]["ok"], res4)
# (5) validator with a set depth (a synthetic point cloud wall) behind vs. in front of open space
with tempfile.TemporaryDirectory() as td:
    xs, zs = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(0, 3, 100))
    P = np.stack([xs, np.full_like(xs, 4.03), zs], -1)          # a wall 3 cm behind y = 4
    np.save(Path(td) / "points.npy", P)
    sc5 = dict(sc, world_dir=td)
    sh5 = shot_with([{"name": "door", "box": [0, 0, 0, 1, 1, 1]}])
    S.mount_arch(sh5["set"]["props"][0], sc5, sh5)
    r5 = S.arch_wall_check(sh5, sc5, "synthetic")
    check("validator: set depth 3 cm behind the outline passes", r5[0]["ok"], r5)
    np.save(Path(td) / "points.npy", P + np.array([0, 2.0, 0]))  # the wall 2 m behind: open space
    S._CLOUD.clear()
    r6 = S.arch_wall_check(sh5, sc5, "synthetic")
    check("validator: a wall 2 m behind (open space) fails", not r6[0]["ok"], r6)

# --- the set kind comes from the book data first -------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    (Path(td) / "stage").mkdir()
    (Path(td) / "stage" / "assets.json").write_text(json.dumps({"sets": [{"where": "The Old Hall", "kind": "exterior"}]}))
    check("set kind: assets.json beats the name regex", BP.set_kind({"where": "The Old Hall"}, {"book": td}) == "exterior")
    check("set kind: name fallback", BP.set_kind({"where": "Baker Street sitting room"}, {}) == "interior")

# --- contact action: roles by affordance / class flag ------------------------------------------------------
T = S.CONTACT_ACTIONS["strike_on_support"]
sup_classes = [r["base"] for r in PB.CLASSES["physical"] if r.get(T["support"]["class_flag"])]
check("contact action: a support class is flagged in data", bool(sup_classes), sup_classes)
import blockout_props as PR  # noqa: E402
strikers = [k for k, e in PR.library().items() if T["striker"]["affordance"] in (e.get("affordances") or {})]
check("contact action: strikers found by affordance", bool(strikers), strikers)
sup = sup_classes[0]
work = next(r["base"] for r in PB.CLASSES["physical"] if r.get("mobility") == "movable" and r["base"] not in strikers
            and not r.get("offhand_for") and not r.get("work_state"))
sh7 = {"set": {"props": [{"name": sup, "prop": sup, "box": [0, 3, 0, 0.7, 0.3, 0.8], "yaw": 0}]},
       "figures": [{"who": "P0", "at": [0, 2.3], "holds": {T["striker"]["hand"]: strikers[0]}, "pose": "standing"}],
       "camera": {}}
out = S.contact_action(sh7, {"prompt": f"P0 strikes the {work} on the {sup}.", "motion": "", "props": [work]}, {})
cts = sh7.get("contacts") or []
check("contact action: strike -> the striker face meets the work on the support",
      bool(out) and cts and cts[0]["point"] == T["striker"]["affordance"], out)
off = [r["base"] for r in PB.CLASSES["physical"] if r.get("offhand_for") == "strike_on_support"]
check("contact action: no offhand tool unless the text names it",
      sh7["figures"][0]["holds"].get(T["offhand_hand"]) not in off)

# --- the DIRECTOR's off-hand grasp makes the work piece (10-22: "Star-iron, cracked" left the left hand empty) ------
sh8 = {"set": {"props": [{"name": sup, "prop": sup, "box": [0, 3, 0, 0.7, 0.3, 0.8], "yaw": 0}]},
       "figures": [{"who": "P0", "at": [0, 2.3], "holds": {T["striker"]["hand"]: strikers[0], T["offhand_hand"]: "sword__blank"},
                    "pose": "standing"}], "camera": {}}
S.contact_action(sh8, {"prompt": f"P0 strikes the cracked star-iron on the {sup}.", "motion": "",
                       "props": ["Star-iron, cracked", "Hammer"]}, {})
lh = sh8["figures"][0]["holds"].get(T["offhand_hand"]) or ""
check("director: the off hand's grasped blade is the work (left hand keeps the sword)", lh.startswith("sword"),
      sh8["figures"][0]["holds"])
sh9 = {"set": {"props": [{"name": sup, "prop": sup, "box": [0, 3, 0, 0.7, 0.3, 0.8], "yaw": 0}]},
       "figures": [{"who": "P0", "at": [0, 2.3], "holds": {T["striker"]["hand"]: strikers[0]}, "free_hands": ["left"],
                    "pose": "standing"}], "camera": {}}
S.contact_action(sh9, {"prompt": f"P0 strikes the sword on the {sup}.", "motion": "", "props": ["sword"]}, {})
check("director: a hand directed free stays free", "left" not in sh9["figures"][0]["holds"], sh9["figures"][0]["holds"])

sh10 = {"set": {"props": [{"name": sup, "prop": sup, "box": [0, 3, 0, 0.7, 0.3, 0.8], "yaw": 0}]},
        "figures": [{"who": "P0", "at": [0, 2.3], "holds": {"right": "sword", "left": "sword"}, "pose": T["poses"]["raise"]}],
        "camera": {}}
out10 = S.contact_action(sh10, {"prompt": "P0 turns the finished sword in his hands.", "motion": "", "props": ["sword"],
                                "acts": [{"do": "grasp", "who": "P0", "hand": "both", "prop": "sword"}]}, {})
check("director: directed without a strike -> no strike template (hands kept)",
      not out10 and sh10["figures"][0]["holds"] == {"right": "sword", "left": "sword"}, sh10["figures"][0]["holds"])

print(json.dumps({"fails": FAILS}))
sys.exit(1 if FAILS else 0)
