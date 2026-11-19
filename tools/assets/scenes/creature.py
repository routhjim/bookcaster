"""A creature of the library staged like a person (stdlib): its rig's engine assets, poses and size, from its record.

A rigged creature is a library entry <library>/creature/<id>/ whose rig/bones.json (orbit/rig_fit.py, bl_rig.py,
ue_rig_poses.py) names its engine skeletal mesh, its pose clip and the poses the clip holds, one per frame:
    {"bones": [{name, parent, head, tail}], "ue": {"skeletal_mesh", "anim", "poses": [name, ...], "fps"?}}
The cast names it "creature/<id>"; the scene spec stages it as a figure ({"asset": "creature/<id>", "pose": name})
and the scene tool turns the pose name into the clip's frame.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RIG_FPS = 24.0          # the pose clip's frame rate when the rig record does not say (ue_rig_poses.py's default)
HEAD_BONE = "head"


def is_id(fid) -> bool:
    """A cast entry that names a library creature ("creature/<id>") rather than a character."""
    return isinstance(fid, str) and fid.startswith("creature/")


def rig(library: Path, asset: str) -> dict | None:
    """The staging facts of a rigged library creature ("creature/<id>" or "<id>"), or None when it has no engine rig:
    {"asset", "mesh", "anim", "poses": [names], "fps", "size_m", "radius_m", "head_len_m", "description"}."""
    aid = asset.split("/", 1)[1] if is_id(asset) else asset
    d = Path(library) / "creature" / aid
    ap, bp = d / "asset.json", d / "rig" / "bones.json"
    if not (ap.exists() and bp.exists()):
        return None
    aj, bj = json.loads(ap.read_text()), json.loads(bp.read_text())
    ue = bj.get("ue") or {}
    if not (ue.get("skeletal_mesh") and ue.get("anim") and ue.get("poses")):
        return None
    size = [float(x) for x in (aj.get("size_m") or [1.0, 1.0, 1.0])]
    head = next((b for b in bj.get("bones") or [] if b.get("name") == HEAD_BONE), None)
    hl = math.dist(head["head"], head["tail"]) if head else 0.12 * size[2]
    return {"asset": f"creature/{aid}", "mesh": ue["skeletal_mesh"], "anim": ue["anim"], "poses": list(ue["poses"]),
            "fps": float(ue.get("fps") or RIG_FPS), "size_m": size,
            # what others keep clear of round its middle: half its longer footprint side (it may face anyone: its
            # snout or tail reaches that far)
            "radius_m": round(0.5 * max(size[0], size[1]), 3), "head_len_m": round(hl, 3),
            "description": aj.get("description", "")}


def rigged(library: Path) -> list:
    """Every rigged creature of the library, by id."""
    root = Path(library) / "creature"
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        r = rig(library, d.name)
        if r:
            out.append(r)
    return out


def pose_time(r: dict, pose: str | None) -> float:
    """The clip time of a named pose (its frame / fps); an unknown name is the rest pose (the clip's first frame)."""
    i = r["poses"].index(pose) if pose in r["poses"] else 0
    return i / r["fps"]


def pose_words(name: str) -> str:
    """A pose name in plain words, as the directing model is offered it."""
    return name.replace("_", " ").strip()


def head_measure(base: list, facing: list, height: float, head: list | None, jaw: list | None, head_len: float) -> dict:
    """A creature's head as the camera rules read a face (set-frame metres): the eye line at the head joint, the head
    top and chin a share of the head's length over and under it, the head's forward from the head joint to the jaw
    joint (both carried by the head) in plan, else the body's facing. Without a head joint: the body's top."""
    fx, fy = facing
    if head is None:
        top = base[2] + height
        return {"eye_xy": [base[0] + fx * 0.3 * height, base[1] + fy * 0.3 * height], "eye_z": top - 0.1 * height,
                "head_top_z": top, "chin_z": top - 0.2 * height, "head_fwd_xy": [fx, fy]}
    fwd = [fx, fy]
    if jaw is not None:
        d = [jaw[0] - head[0], jaw[1] - head[1]]
        n = math.hypot(*d)
        if n > 0.02 * head_len:
            fwd = [d[0] / n, d[1] / n]
    return {"eye_xy": [round(head[0] + fwd[0] * 0.3 * head_len, 4), round(head[1] + fwd[1] * 0.3 * head_len, 4)],
            "eye_z": round(head[2], 4), "head_top_z": round(head[2] + 0.3 * head_len, 4),
            "chin_z": round(head[2] - 0.45 * head_len, 4), "head_fwd_xy": [round(fwd[0], 4), round(fwd[1], 4)]}
