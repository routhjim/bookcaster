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
    rp0 = record(aid, library)                      # what it was made as, before anything is measured on it
    rec0 = json.loads(rp0.read_text())
    rec0.update(piece_kind=req["kind"], look=req.get("look", ""))
    if steps["spec"]["spec"].get("seats_abreast"):
        rec0["seats_abreast"] = steps["spec"]["spec"]["seats_abreast"]
    rp0.write_text(json.dumps(rec0, indent=1))
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
    if seat.get("inside") and seat.get("width_m"):
        k = max(1.0, min(2.5, SEAT_W_M / float(seat["width_m"])))
        k = k if k > 1.1 else 1.0
        spec = dict(steps["spec"]["spec"], size_m=[round(v * k, 3) for v in steps["spec"]["spec"]["size_m"]],
                    sight_cut_m=sight_cut(seat, k))
        sp.write_text(json.dumps(spec, indent=1))
        import shutil
        shutil.rmtree(library / "prop" / aid, ignore_errors=True)
        log(f"{aid}: its seat is {seat['width_m']} m wide; made again {k:.2f} times the size, opened ahead of the seats")
        r = run(argv)
        t_end = time.time() + CHAIN_WAIT_S
        while not record(aid, library) and time.time() < t_end:
            time.sleep(10)
        if not record(aid, library):
            return {"asset_id": aid, "ok": False, "why": "no library record after the second making"}
        rp1 = record(aid, library)
        rec1 = json.loads(rp1.read_text())
        rec1.update({k_: rec0[k_] for k_ in ("piece_kind", "look", "seats_abreast") if k_ in rec0})
        rp1.write_text(json.dumps(rec1, indent=1))
        for need in req.get("needs") or []:
            if need in MEASURE:
                run(MEASURE[need] + [str(library / "prop" / aid), "--write"])
    # a bunk is sized by the room over its lower tier: made again, once, at the scale that lets a person sit there
    sup_ = next((x for x in json.loads(record(aid, library).read_text()).get("supports") or []
                 if x.get("posture") == "lying"), None)
    if sup_ and sup_.get("headroom_m") and sup_["headroom_m"] < SIT_ROOM_M and not req.get("_tiered"):
        k = min(2.0, SIT_ROOM_M / float(sup_["headroom_m"]))
        spec = dict(steps["spec"]["spec"], size_m=[round(v * k, 3) for v in steps["spec"]["spec"]["size_m"]])
        sp.write_text(json.dumps(spec, indent=1))
        import shutil
        shutil.rmtree(library / "prop" / aid, ignore_errors=True)
        log(f"{aid}: {sup_['headroom_m']} m over its lower tier; made again {k:.2f} times the size")
        run(argv)
        t_end = time.time() + CHAIN_WAIT_S
        while not record(aid, library) and time.time() < t_end:
            time.sleep(10)
        if not record(aid, library):
            return {"asset_id": aid, "ok": False, "why": "no library record after the resize"}
        rp1 = record(aid, library)
        rec1 = json.loads(rp1.read_text())
        rec1.update({k_: rec0[k_] for k_ in ("piece_kind", "look", "seats_abreast") if k_ in rec0})
        rp1.write_text(json.dumps(rec1, indent=1))
        for need in req.get("needs") or []:
            if need in MEASURE:
                run(MEASURE[need] + [str(library / "prop" / aid), "--write"])
    rp_ = record(aid, library)
    rec_ = json.loads(rp_.read_text())
    es = edge_seats(rec_)
    if es:
        rec_.update(es)
        rp_.write_text(json.dumps(rec_, indent=1))
    # the record says which kind of piece it was made as and with what look (the lookup reads both)
    rp = record(aid, library)
    rec = json.loads(rp.read_text())
    rec.setdefault("piece_kind", req["kind"])
    rec.setdefault("look", req.get("look", ""))
    rp.write_text(json.dumps(rec, indent=1))
    return {"asset_id": aid, "ok": True, "why": "made"}


SIT_ROOM_M = 0.95                       # the room a person needs over a bunk's lower tier to sit on its edge
EDGE_DEPTH_M = 0.5                      # how far in from a bed's edge a sitter's back plane is taken
PLACE_W_M = 0.6                         # the width one person takes on a bed's edge


def edge_seats(rec: dict) -> dict:
    """A bed is also sat on: its long edges as seats, from its measured lie box (seat frame as scenes/seatbox.py:
    front -y). "seat" is the edge on the asset's -x side, "seat_other" the +x side; the scene takes the one that
    faces the room (scenes/seating.py)."""
    sup = next((x for x in rec.get("supports") or [] if x.get("posture") == "lying"), None)
    if not sup or rec.get("seat"):
        return {}
    (x0, x1), (y0, y1) = sup["box"]["x_m"], sup["box"]["y_m"]
    z = sup.get("cover_z_m") or sup["box"]["top_z_m"]
    n = max(1, int((y1 - y0 - 0.5) // PLACE_W_M))                # (the pillow's end is left free)

    def box(front_y, lo, hi, fa):
        mid = (lo + hi) / 2
        return {"seat_z_m": round(z, 3), "front_y_m": round(front_y, 3), "back_y_m": round(front_y + EDGE_DEPTH_M, 3),
                "inner_x_m": [round(lo, 3), round(hi, 3)], "arm_z_m": None, "width_m": round(hi - lo, 3),
                "depth_m": EDGE_DEPTH_M, "places": [round(mid + (i - (n - 1) / 2) * PLACE_W_M, 3) for i in range(n)],
                "front_asset": fa, "edge": True, "source": "places/generate.py from the lie box"}
    return {"seat": box(x0, -y1, -y0, [-1.0, 0.0]), "seat_other": box(-x1, y0, y1, [1.0, 0.0])}


EYE_UP_M = 0.78                         # a sitter's eyes above the seat


def sight_cut(seat: dict, k: float = 1.0) -> dict:
    """The box ahead of a cabin's seats that its people look out through (asset frame, metres, at k times the
    measured size): from the front end of the seats' lane to beyond the nose, across the seats and a little more,
    from just under eye height up to a standing head's height above it."""
    import math
    fa = seat.get("front_asset", [0.0, -1.0])
    t = math.radians(math.degrees(math.atan2(fa[1], fa[0])) + 90.0)        # the seat frame (front -y) in the asset's
    half = max(abs(v) for v in seat["inner_x_m"]) + 0.25
    pts = [(x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t))
           for x in (-half, half) for y in (seat["front_y_m"] + 0.1, seat["front_y_m"] - 50.0)]
    z0 = seat["seat_z_m"] + EYE_UP_M - 0.08
    return {"x": [round(min(p[0] for p in pts) * k, 3), round(max(p[0] for p in pts) * k, 3)],
            "y": [round(min(p[1] for p in pts) * k, 3), round(max(p[1] for p in pts) * k, 3)],
            "z": [round(z0 * k, 3), round((z0 + 0.75) * k, 3)]}


FACE_WORDS = ("{words} The bare surface of one wall seen straight on, flat and evenly lit, filling the whole picture "
              "from edge to edge and running off every edge. Nothing stands in front of it and nothing hangs on it: no "
              "floor, no ceiling, no door, no furniture, no pipes, no shadow, no perspective.")
FACE_TRIES = 3


def make_face(spec: dict, library: Path, seed: int = 11, run=subprocess.run) -> dict:
    """A wall's face panel (sets/place_build.py face_spec): its picture drawn through the line from the words the
    concept's walls were read into, looked at by the vision judge (a picture that shows anything but the bare wall
    is drawn again from the next seed), then the scripted panel's asset chain."""
    aid = spec["id"]
    if record(aid, library):
        return {"asset_id": aid, "ok": True, "why": "already in the library"}
    pic = REFS / aid / "picture.png"
    ra = [sys.executable, str(HERE.parent / "run_asset.py")]
    for n in range(FACE_TRIES):
        if pic.exists():
            break
        run(ra + ["--line-cmd", C.IMAGE_STATION, "refs", "--id", f"gen_face_{aid}_{n}", "--", sys.executable,
                  str(HERE.parent / "refs" / "ref_draw.py"), "picture", aid, "--desc",
                  FACE_WORDS.format(words=spec["wall_words"]), "--size", "1024", "1024", "--seed", str(seed + n)])
        if not pic.exists():
            continue
        r = run(ra + ["--line-cmd", "igpu", "choose", "--id", f"gen_face_look_{aid}_{n}", "--",
                      str(os.environ.get("STORY_VENV_PY") or sys.executable), str(HERE.parent / "scenes" / "concept.py"),
                      "--clean", str(pic)])
        log(f"{aid}: wall face picture, seed {seed + n}: {'bare' if r.returncode == 0 else 'shows other things'}")
        if r.returncode:
            pic.unlink()
    if not pic.exists():
        return {"asset_id": aid, "ok": False, "why": "no picture of the bare wall"}
    sp = library / "prop" / aid / "source" / "spec.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({k: v for k, v in spec.items() if k != "wall_words"}, indent=1))
    r = run(ra + [str(sp), str(HERE.parent / "models" / "backdrop_view" / "backdrop_view.py"), "--detach"])
    log(f"{aid}: chain rc={r.returncode}")
    t_end = time.time() + CHAIN_WAIT_S
    while not record(aid, library) and time.time() < t_end:
        time.sleep(10)
    return {"asset_id": aid, "ok": bool(record(aid, library)), "why": "made" if record(aid, library) else "no library record"}


def make_view(spec: dict, library: Path, seed: int = 11, run=subprocess.run) -> dict:
    """The backdrop of a place's view outside (sets/place_build.py view_spec): its picture drawn through the line from
    the view's own words, then the scripted panel's asset chain (bake, engine import)."""
    aid, words = spec["id"], spec["view_words"]
    if record(aid, library):
        return {"asset_id": aid, "ok": True, "why": "already in the library"}
    pic = REFS / aid / "picture.png"
    ra = [sys.executable, str(HERE.parent / "run_asset.py")]
    if not pic.exists():
        desc = (f"{words[0].upper() + words[1:]}. A wide open view of it, seen from far off, with nothing in the "
                "foreground: no window, no frame, no vehicle, no people, no writing.")
        r = run(ra + ["--line-cmd", C.IMAGE_STATION, "refs", "--id", f"gen_view_{aid}", "--", sys.executable,
                      str(HERE.parent / "refs" / "ref_draw.py"), "picture", aid, "--desc", desc, "--seed", str(seed)])
        log(f"{aid}: view picture rc={r.returncode}")
    if not pic.exists():
        return {"asset_id": aid, "ok": False, "why": "no picture of the view"}
    sp = library / "prop" / aid / "source" / "spec.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({k: v for k, v in spec.items() if k != "view_words"}, indent=1))
    r = run(ra + [str(sp), str(HERE.parent / "models" / "backdrop_view" / "backdrop_view.py"), "--detach"])
    log(f"{aid}: chain rc={r.returncode}")
    t_end = time.time() + CHAIN_WAIT_S
    while not record(aid, library) and time.time() < t_end:
        time.sleep(10)
    return {"asset_id": aid, "ok": bool(record(aid, library)), "why": "made" if record(aid, library) else "no library record"}


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
    ww = (((spec.get("_concept") or {}).get("bare") or {}).get("walls") or {}).get("words")
    if ww and not only:                                  # the place's own wall, as its concept shows it
        sys.path.insert(0, str(HERE.parent / "sets"))
        import place_build as PB
        out.append(dict(make_face(PB.face_spec(ww), library, seed), kind="wall_face"))
    if spec.get("outside") and not only:                 # what lies outside the place: its view's backdrop
        sys.path.insert(0, str(HERE.parent / "sets"))
        import place_build as PB
        out.append(dict(make_view(PB.view_spec(spec["outside"]), library, seed), kind="view"))
    print(json.dumps(out, indent=1))
    return 0 if all(o["ok"] for o in out) else 1


if __name__ == "__main__":
    sys.exit(main())
