"""A place's results kept: a copy before it is made again, an approved state that later runs leave alone, the way
back to either, and what a change of rules would move in an approved place before any engine work.

  snapshot(...)   the place's staging, its set and its keys copied into PLACE_DIR/history/TAG (the last KEEP kept)
  approve(...)    a snapshot marked as the place's approved state (PLACE_DIR/approved.json); story_keys.py runs no
                  step of an approved place until it is unlocked
  restore(...)    a snapshot copied back over the place, its set and its keys
  drift(...)      the planner's scene specs for the place as the rules now write them, against the approved ones:
                  who would stand elsewhere, sit elsewhere, face another way, which given cameras would move

(A place's keys were worked up over a day and judged good; rules written for another place then made its set and
its cameras again, and nothing held the good state or said it would move, 11-20.)

Only what the planner decides on its own is compared (scenes/from_plan.py: places, seats, facing, the cameras it
gives a position); the cameras the engine's staging solves are held by the approved copy itself.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

KEEP = 6                     # snapshots kept for a place (the approved one is never dropped)
MOVED_M = 0.25               # a person or a camera this far from where it was has moved
TURNED_DEG = 20.0            # ... a facing this far round has turned
PLACE_FILES = ("keys.json", "pass.json", "place_spec.json", "place_spec_c.json", "concept.png", "concept.json",
               "concept_depth.json", "cast.json", "carried.json", "shots.txt", "run.json")
PLACE_DIRS = ("specs",)
SEQ_KEEP = ("spec_round*.json", "speakers.json", "s*.png", "staged.json")


def _tag(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%m%d-%H%M%S')}"


def snapshot(place_dir: Path, keys_dir: Path, shots: list, set_dir: Path | None, tag: str | None = None) -> Path:
    """The place as it stands, copied to place_dir/history/tag: its specs, each staged scene's final specs and
    stills, its keys, its set. -> the snapshot's folder."""
    out = place_dir / "history" / (tag or _tag("before"))
    out.mkdir(parents=True, exist_ok=True)
    for n in PLACE_FILES:
        if (place_dir / n).exists():
            shutil.copy2(place_dir / n, out / n)
    for n in PLACE_DIRS:
        if (place_dir / n).is_dir():
            shutil.copytree(place_dir / n, out / n, dirs_exist_ok=True)
    for g in sorted((place_dir / "seq").glob("*")) if (place_dir / "seq").is_dir() else []:
        if g.is_dir():
            (out / "seq" / g.name).mkdir(parents=True, exist_ok=True)
            for pat in SEQ_KEEP:
                for f in g.glob(pat):
                    if f.is_file():
                        shutil.copy2(f, out / "seq" / g.name / f.name)
        elif g.suffix == ".json":
            (out / "seq").mkdir(exist_ok=True)
            shutil.copy2(g, out / "seq" / g.name)
    (out / "keys").mkdir(exist_ok=True)
    for sid in shots:
        for f in keys_dir.glob(f"{sid}_s0*"):
            shutil.copy2(f, out / "keys" / f.name)
        for f in keys_dir.glob(f".{sid}_s0*"):
            shutil.copy2(f, out / "keys" / f.name)
    if set_dir and set_dir.is_dir():
        shutil.copytree(set_dir, out / "set", dirs_exist_ok=True)
    (out / "snapshot.json").write_text(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "shots": len(shots),
                                                    "set": set_dir.name if set_dir else None}, indent=1))
    prune(place_dir)
    return out


def prune(place_dir: Path) -> None:
    held = (approved(place_dir) or {}).get("snapshot")
    snaps = sorted((p for p in (place_dir / "history").glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for p in snaps[:-KEEP]:
        if p.name != held:
            shutil.rmtree(p, ignore_errors=True)


def approved(place_dir: Path) -> dict | None:
    p = place_dir / "approved.json"
    return json.loads(p.read_text()) if p.exists() else None


def approve(place_dir: Path, keys_dir: Path, shots: list, set_dir: Path | None, note: str = "") -> Path:
    snap = snapshot(place_dir, keys_dir, shots, set_dir, _tag("approved"))
    (place_dir / "approved.json").write_text(json.dumps({"snapshot": snap.name, "note": note,
                                                         "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    return snap


def unapprove(place_dir: Path) -> None:
    (place_dir / "approved.json").unlink(missing_ok=True)


def snapshots(place_dir: Path) -> list:
    return [p.name for p in sorted((place_dir / "history").glob("*"), key=lambda p: p.stat().st_mtime) if p.is_dir()]


def restore(place_dir: Path, keys_dir: Path, set_dir: Path | None, tag: str | None = None) -> Path:
    """A snapshot (the approved one when no tag is given) copied back over the place, its set and its keys."""
    tag = tag or (approved(place_dir) or {}).get("snapshot")
    snap = place_dir / "history" / str(tag)
    if not tag or not snap.is_dir():
        raise SystemExit(f"{place_dir.name}: no snapshot {tag!r} (have: {', '.join(snapshots(place_dir)) or 'none'})")
    for n in PLACE_FILES:
        if (snap / n).exists():
            shutil.copy2(snap / n, place_dir / n)
    for n in PLACE_DIRS + ("seq",):
        if (snap / n).is_dir():
            shutil.copytree(snap / n, place_dir / n, dirs_exist_ok=True)
    for f in (snap / "keys").glob("*"):
        shutil.copy2(f, keys_dir / f.name)
    if set_dir and (snap / "set").is_dir():
        shutil.copytree(snap / "set", set_dir, dirs_exist_ok=True)
    return snap


def _specs(d: Path) -> dict:
    """{plan shot id: (its group's characters, its camera)} of a folder of the planner's scene specs."""
    out = {}
    for f in sorted(d.glob("pl_g*.json")):
        g = json.loads(f.read_text())
        for c in g.get("cameras") or []:
            out[c.get("plan_id") or c.get("name")] = (g.get("characters") or {}, c)
    return out


def _ang(a, b) -> float:
    if not a or not b or math.hypot(*a[:2]) < 1e-6 or math.hypot(*b[:2]) < 1e-6:
        return 0.0
    d = math.degrees(math.atan2(a[1], a[0]) - math.atan2(b[1], b[0]))
    return abs((d + 180.0) % 360.0 - 180.0)


def drift(was: Path, now: Path) -> list:
    """What differs between two folders of the planner's scene specs for one place, shot by shot, in words."""
    a, b = _specs(was), _specs(now)
    out = [f"{sid}: no longer staged" for sid in a if sid not in b] + [f"{sid}: newly staged" for sid in b if sid not in a]
    for sid in [s for s in a if s in b]:
        (ca, cama), (cb, camb) = a[sid], b[sid]
        for k in sorted(set(ca) | set(cb)):
            pa, pb = ca.get(k), cb.get(k)
            if not isinstance(pa, dict) or not isinstance(pb, dict):
                if (pa is None) != (pb is None):
                    out.append(f"{sid}: {k} {'is gone' if pb is None else 'is new'}")
                continue
            if pa.get("position") and pb.get("position") and math.dist(pa["position"][:2], pb["position"][:2]) > MOVED_M:
                out.append(f"{sid}: {k} stands {math.dist(pa['position'][:2], pb['position'][:2]):.1f} m from where "
                           f"they stood")
            if pa.get("seat") != pb.get("seat"):
                out.append(f"{sid}: {k} sits on {pb.get('seat') or 'nothing'}, was {pa.get('seat') or 'nothing'}")
            if _ang(pa.get("facing"), pb.get("facing")) > TURNED_DEG:
                out.append(f"{sid}: {k} is turned {_ang(pa.get('facing'), pb.get('facing')):.0f} degrees round")
            if (pa.get("body_pose") or {}).get("pose") != (pb.get("body_pose") or {}).get("pose"):
                out.append(f"{sid}: {k} is posed {(pb.get('body_pose') or {}).get('pose')}, "
                           f"was {(pa.get('body_pose') or {}).get('pose')}")
            if pa.get("work_at") != pb.get("work_at"):
                out.append(f"{sid}: {k} works at {pb.get('work_at') or 'nothing'}, was {pa.get('work_at') or 'nothing'}")
        if bool(cama.get("position")) != bool(camb.get("position")):
            out.append(f"{sid}: its camera is {'now given by the planner' if camb.get('position') else 'no longer given'}")
        elif cama.get("position") and math.dist(cama["position"], camb["position"]) > MOVED_M:
            out.append(f"{sid}: its camera stands {math.dist(cama['position'], camb['position']):.1f} m from where it "
                       f"stood")
        for key in ("kind", "subject", "show"):
            if cama.get(key) != camb.get(key):
                out.append(f"{sid}: the camera's {key} is {camb.get(key)}, was {cama.get(key)}")
    return out
