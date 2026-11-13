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
import os
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
SEAT_W_M = 0.56                         # the width between a seat's sides that an adult takes
CHAIN_WAIT_S = 2400                     # the bake and import chain is detached: the record is waited for this long
MEASURE = {"sit": [sys.executable, str(HERE.parent / "scenes" / "seatbox.py")],
           "seat": [sys.executable, str(HERE.parent / "scenes" / "seatbox.py")],
           "lie": [sys.executable, str(HERE.parent / "characters" / "liebox.py")]}


def log(m):
    print(f"[generate {time.strftime('%H:%M:%S')}] {m}", flush=True)


def record(aid: str, library: Path) -> Path | None:
    p = library / "prop" / aid / "asset.json"
    return p if p.exists() else None


def _clear(did: str, aid: str, library: Path) -> None:
    """A try that failed its gate is taken away whole: the next seed draws and shapes afresh."""
    import shutil
    for d in (REFS / did, library / "prop" / did, library / "prop" / aid):
        shutil.rmtree(d, ignore_errors=True)
    for f in (C.UE_HOME / "lab" / "img3d_props").glob(f"{did}*"):
        f.unlink() if f.is_file() else shutil.rmtree(f, ignore_errors=True)


def _try(req: dict, library: Path, seed: int, run) -> dict:
    aid, plan = req["asset_id"], req["plan"]
    steps = {s["step"]: s for s in plan["steps"]}
    did = plan["draw_id"]
    pic = REFS / did / "three_quarter.png"
    if not pic.exists():
        r = run([a.replace("{seed}", str(seed)) for a in steps["draw"]["argv"]])
        log(f"{did}: drawing seed {seed} rc={r.returncode}")
    if not pic.exists():
        return {"asset_id": aid, "ok": False, "why": "no drawing with a three-quarter view"}
    sp = library / "prop" / did / "source" / "spec.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(steps["spec"]["spec"], indent=1))
    argv = list(steps["shape"]["argv"])
    omp = os.environ.get("ASSETS_TRELLIS_OMP")          # the line runs the job in its own environment
    if omp and "--" in argv:
        i = argv.index("--") + 1
        argv[i:i] = ["env", f"ASSETS_TRELLIS_OMP={omp}"]
    r = run(argv)
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
            if r.returncode:
                return {"asset_id": aid, "ok": False, "why": f"no {need} box could be measured on it"}
    # a piece people sit INSIDE is sized by its seat, not by its kind's box (a cabin made to a fixed height came
    # out with a seat 0.3 m wide, the pilot taller than the whole cockpit): made again at the scale that gives the
    # measured seat a person's width, once
    seat = json.loads(record(aid, library).read_text()).get("seat") or {}
    if seat.get("inside") and seat.get("width_m") and not req.get("_resized"):
        k = max(1.0, min(2.5, SEAT_W_M / float(seat["width_m"])))
        if k > 1.1:
            spec = dict(steps["spec"]["spec"], size_m=[round(v * k, 3) for v in steps["spec"]["spec"]["size_m"]])
            sp.write_text(json.dumps(spec, indent=1))
            import shutil
            shutil.rmtree(library / "prop" / aid, ignore_errors=True)
            log(f"{aid}: its seat is {seat['width_m']} m wide; made again {k:.2f} times the size")
            r = run(argv)
            t_end = time.time() + CHAIN_WAIT_S
            while not record(aid, library) and time.time() < t_end:
                time.sleep(10)
            if not record(aid, library):
                return {"asset_id": aid, "ok": False, "why": "no library record after the resize"}
            for need in req.get("needs") or []:
                if need in MEASURE:
                    run(MEASURE[need] + [str(library / "prop" / aid), "--write"])
    # the record says which kind of piece it was made as and with what look (the lookup reads both)
    rp = record(aid, library)
    rec = json.loads(rp.read_text())
    rec.setdefault("piece_kind", req["kind"])
    rec.setdefault("look", req.get("look", ""))
    rp.write_text(json.dumps(rec, indent=1))
    return {"asset_id": aid, "ok": True, "why": "made"}


def make(req: dict, library: Path, seed: int = 11, run=subprocess.run) -> dict:
    """One request -> {"asset_id", "ok", "why", "seed"}: drawn, shaped and measured; a try that fails a gate is
    made again from the next seed's drawing, at most the plan's max_redraws tries. `run` starts a process."""
    aid, plan = req["asset_id"], req["plan"]
    if record(aid, library):
        return {"asset_id": aid, "ok": True, "why": "already in the library"}
    out = {}
    for n in range(plan.get("max_redraws", 3)):
        out = dict(_try(req, library, seed + n, run), seed=seed + n)
        if out["ok"]:
            return out
        log(f"{aid}: {out['why']}; the try is cleared" + ("" if n + 1 >= plan.get("max_redraws", 3) else ", next seed"))
        _clear(plan["draw_id"], aid, library)
    return out


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
