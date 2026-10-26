"""The 10-21 evening fixes to the scene layer (synthetic, CPU only, < 1 s).

    ~/tts-ab/.venv-pose/bin/python -I tools/tests/test_scene_layer_1021.py

- keep_absent: a person the moment names but the planner gave no pose box keeps their place and pose from the
  nearest shot of the scene (the King on his throne -> DEFAULT STAGING -> 2 moments not rendered);
- insert_of: a close-up whose subject is hands or a held thing frames the hands + the prop, not the face.
Prints PASS/FAIL per check, exit 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import scene_stage as S  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def test_keep_absent():
    order = ["m1", "m2", "m3"]
    frames = {m: {"who": ["Edric", "King"]} for m in order}
    shots = {"m1": {"figures": [{"who": "Edric", "pose": "standing", "at": [0, 4]},
                                {"who": "King", "pose": "sitting_chair", "at": [9, 9], "facing": 225.0, "model": "k.blend"}]},
             "m2": {"figures": [{"who": "Edric", "pose": "standing", "at": [0, 4]}]},
             "m3": {"figures": [{"who": "Edric", "pose": "bowing", "at": [0, 4]}]}}
    scene = {"marks": {"Edric": {"at": [0, 4]}, "King": {"at": [-0.5, 4.6], "facing": 180.0}}}
    S.keep_absent(order, frames, shots, scene)
    for m in ("m2", "m3"):
        k = next((g for g in shots[m]["figures"] if g["who"] == "King"), None)
        check(f"keep_absent: King kept in {m}", k is not None)
        if k:
            check(f"keep_absent: {m} keeps his pose and model, at his scene mark",
                  k["pose"] == "sitting_chair" and k.get("model") == "k.blend" and k["at"] == [-0.5, 4.6], k)
    check("keep_absent: noted in plan_notes", shots["m2"].get("plan_notes", {}).get("kept") == ["King"])
    n = len(shots["m1"]["figures"])
    S.keep_absent(order, frames, shots, scene)
    check("keep_absent: idempotent", len(shots["m1"]["figures"]) == n and len(shots["m2"]["figures"]) == 2)
    frames2 = {"m1": {"who": ["Edric", "Stranger"]}}
    shots2 = {"m1": {"figures": [{"who": "Edric"}]}}
    S.keep_absent(["m1"], frames2, shots2, {"marks": {"Edric": {"at": [0, 0]}}})
    check("keep_absent: nobody invented without a mark or an earlier shot", len(shots2["m1"]["figures"]) == 1)


def test_insert_of():
    figs = [{"who": "Edric", "holds": {"right": "sword"}}, {"who": "King"}]
    cases = [("Close-up of Edric's hands holding the sheathed Star-iron Sword. The dark blade", "Edric"),
             ("Close-up of Edric's face, sweat on his brow.", None),
             ("Close-up of the sword in Edric's grip.", "Edric"),
             ("Close-up of Edric's face and the glowing sword. He looks down at the blade", None),
             ("Wide shot of the forge.", None),
             ("Close-up of the King's hands on the armrest.", "King")]
    for p, want in cases:
        got = S.insert_of(p, figs)
        check(f"insert_of: {p[:40]!r} -> {want}", got == want, got)


def test_closeup_props():
    props = [{"prop": "anvil"}, {"prop": "bellows"}, {"prop": "box", "name": "ingot on the anvil (contact proxy)"},
             {"prop": "ingot"}]
    cases = [("Close-up of the anvil. The hammer head has just struck the star-iron", ["anvil"]),
             ("Close-up of Edric's face and the glowing sword.", []),
             ("Close-up of Edric's hands and the anvil.", []),
             ("Close-up of the glowing ingot on the anvil.", ["anvil", "ingot"]),
             ("Wide shot of the anvil.", []),
             ("Close-up of Edric's upper body and the bellows. His arms are tense", [])]
    for p, want in cases:
        got = S.closeup_props(p, props, ["Edric", "Hulda"])
        check(f"closeup_props: {p[:40]!r} -> {want}", sorted(got) == sorted(want), got)


def main() -> int:
    test_keep_absent()
    test_insert_of()
    test_closeup_props()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
