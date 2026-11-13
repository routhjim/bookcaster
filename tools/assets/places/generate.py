"""Make the pieces a place spec asks for that the library lacks: each generation request's own plan, run in order.

    python tools/assets/places/generate.py PLACE_SPEC.json [--library DIR] [--seed N] [--only KIND ...] [--dry]

For every piece of the spec that resolves to "generate" (places/pieces.py), the request's plan is run as it stands:
the reference drawing through the line (re-drawn with the next seed when it gives no three-quarter view, at most the
plan's max_redraws), the source spec written, the shape step through the line (picture to 3D, bake, engine import),
then the supports the kind needs measured from the exported mesh and written into the record (a seat box, a lie
box). A piece whose asset is already in the library is skipped. Pieces are made one after another: each step is a
line job, so the cards are shared with whatever else is queued. Nothing here decides how a piece looks: the look is
the spec's own words.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import config as C  # noqa: E402
import pieces as P  # noqa: E402

REFS = C.LIBRARY / "_refs"
CHAIN_WAIT_S = 2400                     # the bake and import chain is detached: the record is waited for this long
MEASURE = {"sit": [sys.executable, str(HERE.parent / "scenes" / "seatbox.py")],
           "seat": [sys.executable, str(HERE.parent / "scenes" / "seatbox.py")],
           "lie": [sys.executable, str(HERE.parent / "characters" / "liebox.py")]}


def log(m):
    print(f"[generate {time.strftime('%H:%M:%S')}] {m}", flush=True)


def record(aid: str, library: Path) -> Path | None:
    p = library / "prop" / aid / "asset.json"
    return p if p.exists() else None


def make(req: dict, library: Path, seed: int = 11, run=subprocess.run) -> dict:
    """One request -> {"asset_id", "ok", "why"}; `run` starts a process (tests pass a fake)."""
    aid, plan = req["asset_id"], req["plan"]
    steps = {s["step"]: s for s in plan["steps"]}
    did = plan["draw_id"]
    if record(aid, library):
        return {"asset_id": aid, "ok": True, "why": "already in the library"}
    pic = REFS / did / "three_quarter.png"
    for n in range(plan.get("max_redraws", 3)):
        if pic.exists():
            break
        argv = [a.replace("{seed}", str(seed + n)) for a in steps["draw"]["argv"]]
        r = run(argv)
        log(f"{did}: drawing seed {seed + n} rc={r.returncode}")
    if not pic.exists():
        return {"asset_id": aid, "ok": False, "why": "no drawing with a three-quarter view"}
    sp = library / "prop" / did / "source" / "spec.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(steps["spec"]["spec"], indent=1))
    r = run(steps["shape"]["argv"])
    log(f"{did}: shape rc={r.returncode}")
    if r.returncode:
        return {"asset_id": aid, "ok": False, "why": f"the shape step failed (rc {r.returncode})"}
    t_end = time.time() + CHAIN_WAIT_S
    while not record(aid, library) and time.time() < t_end:
        time.sleep(10)
    if not record(aid, library):
        return {"asset_id": aid, "ok": False, "why": "no library record after the bake and import chain"}
    for need in req.get("needs") or []:
        if need in MEASURE:
            r = run(MEASURE[need] + [str(library / "prop" / aid), "--write"])
            log(f"{aid}: {need} box rc={r.returncode}")
    # the record says which kind of piece it was made as and with what look (the lookup reads both)
    rp = record(aid, library)
    rec = json.loads(rp.read_text())
    rec.setdefault("piece_kind", req["kind"])
    rec.setdefault("look", req.get("look", ""))
    rp.write_text(json.dumps(rec, indent=1))
    return {"asset_id": aid, "ok": True, "why": "made"}


def main():
    a = sys.argv[1:]
    spec = json.loads(Path(a[0]).read_text())
    spec = spec.get("sheet", spec)
    library = Path(a[a.index("--library") + 1]) if "--library" in a else C.LIBRARY
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 11
    only = set(a[a.index("--only") + 1:]) - {"--dry"} if "--only" in a else None
    res = P.resolve(spec.get("pieces") or [], P.load_records(library))
    todo = [p for p in res["pieces"] if p["status"] == "generate" and (not only or p["kind"] in only)]
    log(f"{len(todo)} piece(s) to make: {[(p['kind'], p['request']['asset_id']) for p in todo]}")
    if "--dry" in a:
        return 0
    out = [dict(make(p["request"], library, seed), kind=p["kind"]) for p in todo]
    print(json.dumps(out, indent=1))
    return 0 if all(o["ok"] for o in out) else 1


if __name__ == "__main__":
    sys.exit(main())
