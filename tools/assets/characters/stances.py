"""Stance variants chosen from the pose a shot names (stdlib).

A character whose loose cloth (a long skirt, a coat's skirt) was settled for a stance has one library entry per
stance (bl_settle.py bakes the cloth into that stance's bind pose). Each such entry carries
    "stance": {"base": "<the standing entry's id>", "pose": "seated" | "walk" | ..., "seat_asset": "<prop id>" | null,
               "seat_z_m": 0.506 | null, "params": {...the pose parameters it was settled for...}}
and the standing entry is its own base. resolve() picks the entry for a shot:

    resolve(library, char_id, body_pose, seat_asset=None) -> (entry id, note)

  - the pose family from the pose name: "sit*" / "seated" -> "seated", "walk*" -> "walk", anything else "stand";
  - "stand": the base entry;
  - otherwise the variants of the same base and family: the one settled for the same seat asset, else the one
    whose seat height is nearest (within 5 cm), else the base with a note that the loose cloth will hang as if
    standing (make the variant with bl_settle.py for that chair).
Any id of the family (the base or one of its variants) may be passed in.
"""
from __future__ import annotations

import json
from pathlib import Path


def family(pose_name: str | None) -> str:
    p = (pose_name or "").lower()
    if "sit" in p or "seat" in p:
        return "seated"
    if "walk" in p:
        return "walk"
    return "stand"


def _entries(library: Path):
    for p in (Path(library) / "character").glob("*/character.json"):
        try:
            yield json.loads(p.read_text())
        except (OSError, ValueError):
            continue


def resolve(library, char_id, body_pose=None, seat_asset=None):
    lib = Path(library)
    me = json.loads((lib / "character" / char_id / "character.json").read_text())
    base = (me.get("stance") or {}).get("base") or char_id
    fam = family((body_pose or {}).get("pose"))
    if fam == "stand":
        return base, f"{base}: standing entry"
    seat_z = ((body_pose or {}).get("params") or {}).get("seat_z_m")
    cands = [e for e in _entries(lib) if (e.get("stance") or {}).get("base") == base
             and (e.get("stance") or {}).get("pose") == fam]
    if seat_asset:
        same = [e for e in cands if e["stance"].get("seat_asset") == seat_asset]
        if same:
            return same[0]["id"], f"{same[0]['id']}: settled for {seat_asset}"
    if seat_z is not None:
        near = [e for e in cands if e["stance"].get("seat_z_m") is not None
                and abs(e["stance"]["seat_z_m"] - seat_z) <= 0.05]
        if near:
            best = min(near, key=lambda e: abs(e["stance"]["seat_z_m"] - seat_z))
            return best["id"], f"{best['id']}: nearest settled seat height ({best['stance']['seat_z_m']} m)"
    if cands and fam == "walk":
        return cands[0]["id"], f"{cands[0]['id']}: settled walking stance"
    return base, (f"{base}: no {fam} variant settled for this seat; the loose cloth hangs as if standing "
                  f"(settle one with img3d/bl_settle.py)")


if __name__ == "__main__":
    import sys
    print(resolve(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None,
                  sys.argv[4] if len(sys.argv) > 4 else None))
