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
SEAT_FIT_W_M = 0.48                     # a seat between arms narrower than this takes nobody: made again larger
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


LINE_ENV = ("ASSETS_TRELLIS_OMP", "ASSETS_SHELL_PY")   # what a shape job needs from this run's environment


def line_env() -> list:
    """`env K=V ...` for a line job: the line runs jobs in its own environment, so the shape step's settings are
    passed in the command (without ASSETS_SHELL_PY an outdoor world ran under a python with no trimesh, 11-17)."""
    kv = [f"{k}={os.environ[k]}" for k in LINE_ENV if os.environ.get(k)]
    return ["env"] + kv if kv else []


def _clear(did: str, aid: str, library: Path, refs: tuple = ()) -> None:
    """A try that failed its gate is taken away whole: the next seed draws and shapes afresh (refs: the try's own
    drawings kept under another id, a world's panorama; its ground map is kept, it failed no gate)."""
    import shutil
    for d in [REFS / did] + [REFS / r for r in refs if r] + [library / "prop" / did, library / "prop" / aid]:
        shutil.rmtree(d, ignore_errors=True)
    for f in (C.UE_HOME / "lab" / "img3d_props").glob(f"{did}*"):
        if f.is_file() and ".failed" in f.name:
            continue
        if f.is_file() and f.suffix == ".log":           # (the failed try's log is kept, renamed, to read why)
            f.replace(f.with_name(f"{f.stem}.failed{int(time.time())}.log"))
        else:
            f.unlink() if f.is_file() else shutil.rmtree(f, ignore_errors=True)


def tiered(aid: str, library: Path) -> bool:
    """The made bed has a tier above its mattress (its measured lying support says so)."""
    rp = record(aid, library)
    return bool(rp) and any(x.get("posture") == "lying" and x.get("tiers") for x in json.loads(rp.read_text()).get("supports") or [])


COUNT_SH = Path(os.environ.get("STORY_COUNT_SH", C.UE_HOME / "lab" / "redraw5" / "count.sh"))


def people_in(pics: dict, run, work: Path) -> dict:
    """{id: how many people each picture shows} (redraw/count_people.py through the line, one job for them all);
    {} when the count could not be made."""
    work.mkdir(parents=True, exist_ok=True)
    cj = work / "count.json"
    cj.write_text(json.dumps({"out": str(work / "people.json"),
                              "items": [{"id": k, "path": str(v)} for k, v in pics.items()]}, indent=1))
    (work / "people.json").unlink(missing_ok=True)
    r = run([sys.executable, str(HERE.parent / "run_asset.py"), "--line-cmd", C.IMAGE_STATION, "check", "--id",
             f"count_{work.name}", "--log", str(work / "count.log"), "--", "bash", str(COUNT_SH), str(cj)])
    if r.returncode or not (work / "people.json").exists():
        return {}
    return {k: len(v.get("boxes") or []) for k, v in json.loads((work / "people.json").read_text()).items()}


def nobody_seen(did: str, pic: Path, run) -> bool:
    """True when the vision judge sees a person in a place's panorama (scenes/concept.py --nobody, through the line).
    A person drawn into the panorama becomes a smeared sheet standing in the place (11-16: a man on the path of a
    river bank's panorama, with the people count not to hand)."""
    r = run([sys.executable, str(HERE.parent / "run_asset.py"), "--line-cmd", "igpu", "choose", "--id",
             f"gen_nobody_{did}", "--", str(os.environ.get("STORY_VENV_PY") or sys.executable),
             str(HERE.parent / "scenes" / "concept.py"), "--nobody", str(pic)])
    log(f"{did}: someone in the panorama: {'yes' if r.returncode == 1 else 'no' if r.returncode == 0 else 'not judged'}")
    return r.returncode == 1


def count_shells(library: Path, run) -> None:
    """Every cabin shell in the library whose panorama has not been counted gets its count ("panorama_people" in
    its record): the lookup passes over a shell that shows someone (a person drawn into the panorama is built into
    the cabin, a second pilot in every shot staged in it)."""
    todo = {}
    for rp in sorted((library / "prop").glob("*/asset.json")):
        rec = json.loads(rp.read_text())
        if "panorama_shell" in (rec.get("tags") or []) and "panorama_people" not in rec:
            pic = REFS / rp.parent.name.removesuffix("_tr") / "picture.png"
            if pic.exists():
                todo[rp.parent.name] = pic
    if not todo:
        return
    got = people_in(todo, run, C.UE_HOME / "lab" / "img3d_props" / "count_shells")
    for aid, n in got.items():
        rp = library / "prop" / aid / "asset.json"
        rec = json.loads(rp.read_text())
        rec["panorama_people"] = n
        rp.write_text(json.dumps(rec, indent=1))
        log(f"{aid}: {n} people in its panorama")


def reshape_shell(aid: str, library: Path, run) -> dict:
    """A library cabin shell of an older build (img3d/pano_shell.py SHELL_BUILD), made again from its own panorama:
    same picture, same seat, the current build's shell (the backing hull that closes its tears, 11-15). Its record
    keeps what was learned about it (kind, look, seats, the people count)."""
    import shutil
    sys.path.insert(0, str(HERE.parent / "img3d"))
    import pano_shell
    rp = record(aid, library)
    rec = json.loads(rp.read_text()) if rp else {}
    if rec.get("shell_build", 1) >= pano_shell.SHELL_BUILD:
        return {"asset_id": aid, "ok": True, "why": "the current build"}
    did = aid.removesuffix("_tr")
    keep = {k: rec[k] for k in ("piece_kind", "look", "seats_abreast", "panorama_people") if k in rec}
    for f in (C.UE_HOME / "lab" / "img3d_props").glob(f"{did}.*"):
        f.unlink()                                       # (the shell's mesh and facts: made again)
    shutil.rmtree(library / "prop" / aid, ignore_errors=True)
    env = line_env()
    r = run([sys.executable, str(HERE.parent / "run_asset.py"), "--line-cmd", C.IMAGE_STATION, "shape", "--id",
             f"gen_reshape_{did}", "--log", str(C.UE_HOME / "lab" / "img3d_props" / f"{did}.log"), "--"] + env
            + [sys.executable, str(HERE.parent / "img3d" / "shell_one.py"), did])
    t_end = time.time() + CHAIN_WAIT_S
    while not record(aid, library) and time.time() < t_end:
        time.sleep(10)
    if r.returncode or not record(aid, library):
        return {"asset_id": aid, "ok": False, "why": "the shell could not be made again"}
    rp = record(aid, library)
    rec = json.loads(rp.read_text())
    rec.update(keep, shell_build=pano_shell.SHELL_BUILD)
    rp.write_text(json.dumps(rec, indent=1))
    run(MEASURE["seat"] + [str(library / "prop" / aid), "--write"])
    log(f"{aid}: made again from its panorama as shell build {pano_shell.SHELL_BUILD}")
    return {"asset_id": aid, "ok": True, "why": f"made again as shell build {pano_shell.SHELL_BUILD}"}


def shell_too_big(req: dict, library: Path) -> str:
    """Why a made cabin shell is not the cabin asked for (pieces.shell_too_big), or ""."""
    rp = record(req["asset_id"], library)
    steps = {s_["step"]: s_ for s_ in (req.get("plan") or {}).get("steps") or []}
    if not rp or steps.get("spec", {}).get("spec", {}).get("made_as") != "shell":
        return ""
    got, want = json.loads(rp.read_text()).get("size_m") or [], req.get("size_m") or []
    if not P.shell_too_big(got, want):
        return ""
    return f"the shell is {max(got[:2]):.1f} m long, more than {P.SHELL_SPAN_K} times the {max(want[:2]):.1f} m asked for"


EDIT_PY = os.path.expanduser(os.environ.get("ASSETS_GUIDES_PY", "~/tts-ab/.venv-hyworld/bin/python"))   # the edit model's
CARD_SH = HERE.parent / "orbit" / "card.sh"                                                         # (as img3d/cast_char.py)


def edit_card() -> str:
    """The card the edit model runs on: the image station's when it is a card (the model needs a card's memory), else
    the Blender card."""
    st = str(C.IMAGE_STATION)
    return st[4:] if st.startswith("card") and st[4:] else C.BLENDER_CARD


def ground_map(steps: dict, seed: int, run) -> dict:
    """A place in the open's ground map (places/ground_map.py): drawn through the line, its water painted the key
    colour by the edit model (water_key.png), then read by the vision judge (scenes/concept.py --map) into map.json
    beside it. None of them stops the world: without the map its ground is the panorama's own, as before; without
    the key the judge's outline is its water; without the reading it has no water and its land places are the open
    ground. A reading of an older build (ground_map.READ_BUILD) is made again. -> {"map", "key", "read": bool}."""
    import shlex
    import ground_map as GM
    st, rd, ky = steps["map"], steps.get("map_read") or {}, steps.get("water_key") or {}
    d = REFS / st["ref_id"]
    pic = d / st.get("picture", "picture.png")
    if not pic.exists():
        r = run([a.replace("{seed}", str(seed)) for a in st["argv"]])
        log(f"{st['ref_id']}: ground map, seed {seed} rc={r.returncode}")
    if not pic.exists():
        return {"map": False, "key": False, "read": False}
    key = d / ky.get("picture", "water_key.png") if ky else None
    if key and not key.exists():
        jp = d / ky.get("jobs", "water_key_jobs.json")
        jp.write_text(json.dumps([GM.edit_job(str(pic), str(key), seed)], indent=1))
        repo = HERE.parents[2]
        r = run([sys.executable, str(HERE.parent / "run_asset.py"), "--line-cmd", ky.get("station", "image"),
                 ky.get("kind", "edit"), "--id", f"gen_waterkey_{st['ref_id']}", "--", "bash", "-c",
                 f"cd {shlex.quote(str(repo))} && bash {shlex.quote(str(CARD_SH))} {edit_card()} "
                 f"{shlex.quote(EDIT_PY)} -I tools/assets/qwen_edit.py {shlex.quote(str(jp))}"])
        log(f"{st['ref_id']}: the water painted by the edit model rc={r.returncode}")
    facts = d / rd.get("facts", "map.json")
    if facts.exists():
        try:
            old = int(json.loads(facts.read_text()).get("read_build") or 1) < GM.READ_BUILD
        except (OSError, ValueError):
            old = True
        if old:
            facts.unlink()
            log(f"{st['ref_id']}: a reading of an older build; read again")
    if rd and not facts.exists():
        pl = d / "land_places.json"
        pl.write_text(json.dumps({"land_places": rd.get("land_places") or {}}, indent=1))
        r = run([sys.executable, str(HERE.parent / "run_asset.py"), "--line-cmd", "igpu", "choose", "--id",
                 f"gen_mapread_{st['ref_id']}", "--", str(os.environ.get("STORY_VENV_PY") or sys.executable),
                 str(HERE.parent / "scenes" / "concept.py"), "--map", str(pic), str(pl), str(facts)]
                + ([str(key)] if key and key.exists() else []))
        log(f"{st['ref_id']}: the map read rc={r.returncode}")
    return {"map": True, "key": bool(key and key.exists()), "read": facts.exists()}


def _try(req: dict, library: Path, seed: int, run, unsourced: bool = False) -> dict:
    aid, plan = req["asset_id"], req["plan"]
    steps = {s["step"]: s for s in plan["steps"]}
    did = plan["draw_id"]
    # (a world's panorama keeps its own id, ref_id: a world made again for new map words takes the panorama drawn)
    relief = steps["spec"]["spec"].get("made_as") == "relief"     # (shaped already, from its place's picture)
    pic = Path(steps["spec"]["spec"]["relief_glb"]) if relief else \
        REFS / steps["draw"].get("ref_id", did) / steps["draw"].get("picture", "three_quarter.png")
    if not pic.exists() and not relief:
        dv = list(steps["draw"]["argv"])
        if unsourced and "--sources" in dv:              # (drawn from its look and its needs alone this time)
            dv = dv[:dv.index("--sources")]
        r = run([a.replace("{seed}", str(seed)) for a in dv])
        log(f"{did}: drawing seed {seed} rc={r.returncode}")
    if not pic.exists():
        return {"asset_id": aid, "ok": False, "why": f"no drawing ({pic.name})"}
    shell = steps["spec"]["spec"].get("made_as") == "shell"
    world = steps["spec"]["spec"].get("made_as") == "world"
    if shell or world:
        n = people_in({did: pic}, run, C.UE_HOME / "lab" / "img3d_props" / f"count_{did}").get(did)
        if n:
            return {"asset_id": aid, "ok": False, "why": f"its panorama shows {n} person(s) in the "
                                                         f"{'place' if world else 'cabin'}"}
    if world and nobody_seen(did, pic, run):
        return {"asset_id": aid, "ok": False, "why": "the vision judge sees someone in its panorama"}
    if world and "map" in steps:
        ground_map(steps, seed, run)
    sp = library / "prop" / did / "source" / "spec.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(steps["spec"]["spec"], indent=1))
    argv = list(steps["shape"]["argv"])
    if line_env() and "--" in argv:                     # the line runs the job in its own environment
        i = argv.index("--") + 1
        argv[i:i] = line_env()
    r = run(argv)
    log(f"{did}: shape rc={r.returncode}")
    if r.returncode:
        return {"asset_id": aid, "ok": False, "why": f"the shape step failed (rc {r.returncode})"}
    t_end = time.time() + CHAIN_WAIT_S
    while not record(aid, library) and time.time() < t_end:
        time.sleep(10)
    if not record(aid, library):
        return {"asset_id": aid, "ok": False, "why": "no library record after the bake and import chain"}
    big = shell_too_big(req, library)
    if big:
        return {"asset_id": aid, "ok": False, "why": big}
    rp0 = record(aid, library)                      # what it was made as, before anything is measured on it
    rec0 = json.loads(rp0.read_text())
    rec0.update(piece_kind=req["kind"], look=req.get("look", ""))
    if shell:                                       # (the engine's record does not carry the shell's own facts)
        facts = C.UE_HOME / "lab" / "img3d_props" / f"{did}.json"
        rec0["panorama_people"] = 0
        rec0["shell_build"] = json.loads(facts.read_text()).get("shell_build", 1) if facts.exists() else 1
    if world:                                       # (where people stand in it, and where its light comes from)
        facts = C.UE_HOME / "lab" / "img3d_props" / f"{did}.json"
        fj = json.loads(facts.read_text()) if facts.exists() else {}
        rec0["panorama_people"] = 0
        rec0["world"] = {k: fj.get(k) for k in ("eye_m", "stand_m", "spot_m", "spot", "marks_m", "flat_r_m", "sky_r_m", "sun",
                                                "world_build", "land", "ground_map", "water", "places_m", "water_region")}
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
    # (and any seat between arms too narrow for a person: a chair made to its kind's box had 0.3 m between its
    # arms, and the man named to it was sat on a bench across the room, 11-20)
    tight = bool(seat.get("arm_z_m")) and float(seat.get("width_m") or 9) < SEAT_FIT_W_M
    if (seat.get("inside") or tight) and seat.get("width_m") and steps["spec"]["spec"].get("made_as") != "shell":
        k = max(1.0, min(2.5, SEAT_W_M / float(seat["width_m"])))
        k = k if k > 1.1 else 1.0
        spec = dict(steps["spec"]["spec"], size_m=[round(v * k, 3) for v in steps["spec"]["spec"]["size_m"]],
                    **({"sight_cut_m": sight_cut(seat, k)} if seat.get("inside") else {}))
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
                # (a second making whose measure failed left a seat with no seat box: nobody could be sat on it)
                if run(MEASURE[need] + [str(library / "prop" / aid), "--write"]).returncode:
                    return {"asset_id": aid, "ok": False, "why": f"no {need} box could be measured on its second making"}
    # a bed is asked for with one open tier (people sit up on it, and a camera must see them): one that came out with
    # a tier above its mattress is not the piece asked for (seated on a concept's lower bunk a woman's head was up
    # inside the top one, and no camera saw a face)
    if "lie" in (req.get("needs") or []) and tiered(aid, library):
        return {"asset_id": aid, "ok": False, "why": "it has a tier above its mattress"}
    # a bunk is sized by the room over its lower tier: made again, once, at the scale that lets a person sit there
    sup_ = next((x for x in json.loads(record(aid, library).read_text()).get("supports") or []
                 if x.get("posture") == "lying"), None)
    if sup_ and sup_.get("headroom_m") and sup_["headroom_m"] < SIT_ROOM_M and not req.get("_tiered"):
        k = min(2.0, SIT_ROOM_M / float(sup_["headroom_m"]), BED_MAX_M / max(float(sup_.get("length_m") or 1.0), 0.5))
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


BED_MAX_M = 2.2                         # no bed is made longer than this
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


def face_fill(tex, px_per_m: float, tile=None, colour=None):
    """A wall's own picture (RGBA, alpha 0 where its concept picture does not see the wall) made whole: what is
    unknown is the bare wall's tile picture (tile: RGB, one FACE_M square, laid from the picture's top left at the
    picture's own texels a metre, toned to the known part's mean colour), else one colour (colour: "#rrggbb"; else the mean of what is known) -> RGB."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(HERE.parent / "sets"))
    import place_build as PB
    tex = np.asarray(tex)
    h, w = tex.shape[:2]
    known = tex[..., 3:4].astype(np.float32) / 255.0
    if tile is not None:
        n = max(1, int(round(PB.FACE_M * float(px_per_m))))
        t = np.asarray(Image.fromarray(np.asarray(tile)[..., :3]).resize((n, n), Image.LANCZOS)).astype(np.float32)
        back = np.tile(t, (h // n + 1, w // n + 1, 1))[:h, :w]
        k = tex[..., 3] > 0
        if k.any():                          # (as dark as the picture's own wall: the tile is drawn evenly lit)
            back = back * np.clip(tex[..., :3][k].mean(0) / np.maximum(back.reshape(-1, 3).mean(0), 1.0), 0.2, 1.5)
    else:
        if colour:
            c = [int(colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
        else:
            k = tex[..., 3] > 0
            c = tex[..., :3][k].mean(0) if k.any() else [128, 128, 128]
        back = np.broadcast_to(np.asarray(c, np.float32), (h, w, 3))
    out = tex[..., :3].astype(np.float32) * known + back * (1.0 - known)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


FLAT_PANEL_SCRIPT = """\"\"\"Model script (written by places/generate.py make_face_picture): a room's floor or ceiling as one
flat picture, lying in XY (the picture's top toward +Y, so the set's back wall), the middle of its underside at the
origin; seen from below (a ceiling) it shows the same picture. Spec: "backdrop": {"w", "h": metres along X, Y};
"keep_uvs": the picture laid over the whole texture (period_kit.picture_uvs).\"\"\"
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or %r)
import period_kit as K  # noqa: E402

B = SPEC["backdrop"]  # noqa: F821
LIB = os.environ.get("ASSETS_LIBRARY") or os.path.join(os.path.expanduser("~"), "ue", "library")
w, h = float(B["w"]), float(B["h"])
K.picture_uvs(K.box(f"{SPEC['id']}_pic", (w, h, 0.02), (0, 0, 0.01), bev=0.0,  # noqa: F821
                    mat=K.image_material(os.path.join(LIB, "_refs", SPEC["id"], "picture.png"),  # noqa: F821
                                         f"{SPEC['id']}_print", emit=0.0)))  # noqa: F821
"""
_WHOLE = {}                              # (the room's whole faces, made once for all its panels)


def face_picture_of(spec: dict, tile=None, colour=None):
    """A face panel's picture (sets/place_build.py face_pictures) -> (RGB array, how it was made) or (None, why).
    "face_whole": the room's face made whole from what its concept picture sees (img3d/concept_depth.py
    whole_faces), less the panel's inset at its ends; "face_picture": one face's own picture as it is. What is
    still unknown: the bare wall's tile, else the colour (face_fill)."""
    import numpy as np
    from PIL import Image
    fw = spec.get("face_whole")
    if fw:
        sys.path.insert(0, str(HERE.parent / "img3d"))
        import concept_depth as CD
        key = json.dumps([fw["faces"], fw["room_m"]], sort_keys=True)
        if key not in _WHOLE:
            _WHOLE.clear()
            _WHOLE[key] = CD.whole_faces(fw["faces"], fw["room_m"])
        got = _WHOLE[key].get(fw["face"])
        if got is None:
            return None, "nothing of the room's picture to make it from"
        tex, ppm, how = got
        iu, iv = (int(round(float(x) * ppm)) for x in (fw.get("inset_m") or [0, 0]))
        h, w = tex.shape[:2]
        if w > 2 * iu and h > 2 * iv:
            tex = tex[iv:h - iv, iu:w - iu]
        return face_fill(np.asarray(tex), ppm, tile, colour), how
    src = Path(spec.get("face_picture") or "")
    if not spec.get("face_picture") or not src.exists():
        return None, "no picture of the wall"
    return face_fill(Image.open(src).convert("RGBA"), spec.get("face_px_per_m") or 128.0, tile, colour), "its picture"


def make_face_picture(spec: dict, library: Path, run=subprocess.run, colour=None) -> dict:
    """A room's own wall, floor or ceiling as a face panel (sets/place_build.py face_pictures): its picture from the
    concept picture (face_picture_of), then the scripted panel's asset chain as for the tile (make_face): standing
    (models/backdrop_view), or lying flat (FLAT_PANEL_SCRIPT, "face_flat"). Nothing is drawn."""
    from PIL import Image
    aid = spec["id"]
    if record(aid, library):
        return {"asset_id": aid, "ok": True, "why": "already in the library"}
    tp = REFS / spec["face_fill"] / "picture.png" if spec.get("face_fill") else None
    tile = Image.open(tp).convert("RGB") if tp and tp.exists() else None
    rgb, how = face_picture_of(spec, tile, spec.get("face_colour") or colour)
    if rgb is None:
        return {"asset_id": aid, "ok": False, "why": how}
    pic = REFS / aid / "picture.png"
    pic.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(pic)
    model = HERE.parent / "models" / "backdrop_view" / "backdrop_view.py"
    if spec.get("face_flat"):
        model = REFS / aid / "flat_panel.py"
        model.write_text(FLAT_PANEL_SCRIPT % str(HERE.parent / "models"))
    sp = library / "prop" / aid / "source" / "spec.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({k: v for k, v in spec.items() if not k.startswith("face_")}, indent=1))
    r = run([sys.executable, str(HERE.parent / "run_asset.py"), str(sp), str(model), "--detach"])
    log(f"{aid}: room face picture ({how}; {'tile' if tile is not None else 'colour'} under what is still unknown), "
        f"chain rc={r.returncode}")
    t_end = time.time() + CHAIN_WAIT_S
    while not record(aid, library) and time.time() < t_end:
        time.sleep(10)
    return {"asset_id": aid, "ok": bool(record(aid, library)), "why": "made" if record(aid, library) else "no library record"}


def make_view(spec: dict, library: Path, seed: int = 11, run=subprocess.run, style: str = "") -> dict:
    """The backdrop of a place's view outside (sets/place_build.py view_spec): its picture drawn through the line from
    the view's own words, then the scripted panel's asset chain (bake, engine import)."""
    aid, words = spec["id"], spec["view_words"]
    if record(aid, library):
        return {"asset_id": aid, "ok": True, "why": "already in the library"}
    pic = REFS / aid / "picture.png"
    ra = [sys.executable, str(HERE.parent / "run_asset.py")]
    if not pic.exists():
        # (the view as its words give it, whole: "seen from far off" drew a far hill where the words had a thing
        # filling the view, 11-15)
        desc = (f"{words[0].upper() + words[1:].rstrip('.')}. The view alone, as one looks out at it, filling the "
                f"whole picture: no window, no frame, no vehicle, no people, no writing.{style}")
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


def stale_world(req: dict, library: Path) -> bool:
    """A place's world (made_as "world") of an older build (img3d/pano_shell.py WORLD_BUILD) is taken out of the
    library, its panorama kept: it is made again from the same picture."""
    import shutil
    steps = {s_["step"]: s_ for s_ in (req.get("plan") or {}).get("steps") or []}
    rp = record(req["asset_id"], library)
    if not rp or steps.get("spec", {}).get("spec", {}).get("made_as") != "world":
        return False
    sys.path.insert(0, str(HERE.parent / "img3d"))
    import pano_shell
    if ((json.loads(rp.read_text()).get("world") or {}).get("world_build") or 1) >= pano_shell.WORLD_BUILD:
        return False
    did = req["plan"]["draw_id"]
    for d in (library / "prop" / did, library / "prop" / req["asset_id"]):
        shutil.rmtree(d, ignore_errors=True)
    for f in (C.UE_HOME / "lab" / "img3d_props").glob(f"{did}*"):
        if f.is_file() and f.suffix in (".glb", ".json", ".png"):
            f.unlink()
    log(f"{req['asset_id']}: a world of an older build; made again from its panorama")
    return True


def draw_refs(plan: dict) -> tuple:
    """The drawing a try is made from, when it is kept under its own id (a world's panorama)."""
    return tuple(s_["ref_id"] for s_ in plan.get("steps") or [] if s_.get("step") == "draw" and s_.get("ref_id")
                 and s_["ref_id"] != plan.get("draw_id"))


def make(req: dict, library: Path, seed: int = 11, run=subprocess.run) -> dict:
    """One request -> {"asset_id", "ok", "why", "seed"}: drawn, shaped and measured; a try that fails a gate is
    made again from the next seed's drawing, at most the plan's max_redraws tries. `run` starts a process."""
    aid, plan = req["asset_id"], req["plan"]
    stale_world(req, library)
    big = shell_too_big(req, library)
    if record(aid, library) and not big and not ("lie" in (req.get("needs") or []) and tiered(aid, library)):
        return {"asset_id": aid, "ok": True, "why": "already in the library"}
    out, why = {}, ""
    if record(aid, library):
        _clear(plan["draw_id"], aid, library, draw_refs(plan))
        why = big or "it has a tier above its mattress"
        log(f"{aid}: in the library, but {why}; made again")
    # (a shell made too big is drawn again from the next seed: the same seed draws the same panorama)
    first = 1 if big else 0
    for n in range(first, first + plan.get("max_redraws", 3)):
        # (after a try with a tier above the mattress, the next is drawn without the piece's own picture: the
        # picture is what shows the tiers)
        out = dict(_try(req, library, seed + n, run, unsourced="tier" in why), seed=seed + n)
        why = out.get("why", "") or why
        if out["ok"]:
            return out
        log(f"{aid}: {out['why']}; the try is cleared" + ("" if n + 1 >= first + plan.get("max_redraws", 3) else ", next seed"))
        _clear(plan["draw_id"], aid, library, draw_refs(plan))
    return out


def main():
    a = sys.argv[1:]
    spec = json.loads(Path(a[0]).read_text())
    spec = spec.get("sheet", spec)
    library = Path(a[a.index("--library") + 1]) if "--library" in a else C.LIBRARY
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 11
    only = set(a[a.index("--only") + 1:]) - {"--dry"} if "--only" in a else None
    if "--dry" not in a:
        count_shells(library, subprocess.run)            # (before the lookup, which passes over shells with people)
    if "--dry" not in a:                                 # (a world of an older build is made again)
        for p in spec.get("pieces") or []:
            if (P.kinds().get(p.get("kind")) or {}).get("made_as") == "world":
                stale_world(P.generation_request(p), library)
    res = P.resolve(spec.get("pieces") or [], P.load_records(library))
    todo = [p for p in res["pieces"] if p["status"] == "generate" and (not only or p["kind"] in only)]
    # a bed of the library that has a tier above its mattress is made again as the one asked for
    by_id = {p["id"]: p for p in spec.get("pieces") or []}
    for p in res["pieces"]:
        if p["status"] == "library" and (P.kinds()[p["kind"]].get("needs") == "lie") and tiered(p["asset_id"], library) \
                and p["asset_id"].endswith("_tr") and "_gen_" in p["asset_id"]:
            todo.append(dict(p, request=P.generation_request(by_id[p["id"]]), _old=p["asset_id"]))
    log(f"{len(todo)} piece(s) to make: {[(p['kind'], p['request']['asset_id']) for p in todo]}")
    if "--dry" in a:
        return 0
    out = []
    for p in res["pieces"]:                              # a cabin shell found in the library, of an older build
        if p["status"] == "library" and P.kinds()[p["kind"]].get("made_as") == "shell":
            out.append(dict(reshape_shell(p["asset_id"], library, subprocess.run), kind=p["kind"]))
    for p in todo:
        o = dict(make(p["request"], library, seed), kind=p["kind"])
        out.append(o)
        old = p.get("_old")
        if o["ok"] and old and old != o["asset_id"] and record(old, library):     # (the tiered one serves no longer)
            rp_ = record(old, library)
            rj_ = json.loads(rp_.read_text())
            rj_["replaced_by"] = o["asset_id"]
            rp_.write_text(json.dumps(rj_, indent=1))
    ww = (((spec.get("_concept") or {}).get("bare") or {}).get("walls") or {}).get("words")
    if ww and not only and spec.get("kind") != "outdoors":   # the place's own wall, as its concept shows it
        sys.path.insert(0, str(HERE.parent / "sets"))
        import place_build as PB
        out.append(dict(make_face(PB.face_spec(ww), library, seed), kind="wall_face"))
    if spec.get("kind") != "outdoors" and not only:      # the room's walls as its concept picture shows them
        sys.path.insert(0, str(HERE.parent / "sets"))
        import place_build as PB
        col = (((spec.get("_concept") or {}).get("bare") or {}).get("walls") or {}).get("colour")
        for fs, _ in PB.face_pictures(spec):
            out.append(dict(make_face_picture(fs, library, colour=col), kind="wall_face"))
    if spec.get("outside") and not only:                 # what lies outside the place: its view's backdrop
        sys.path.insert(0, str(HERE.parent / "sets"))
        import place_build as PB
        each = list(dict.fromkeys(str(v).strip() for v in (spec.get("_views") or {}).values() if str(v).strip()))
        sty = f" {str(spec['_style']).strip().rstrip('.')}." if str(spec.get("_style") or "").strip() else ""
        for vw in [spec["outside"]] + [v for v in each if v != spec["outside"]]:
            out.append(dict(make_view(PB.view_spec(vw), library, seed, style=sty), kind="view"))
    print(json.dumps(out, indent=1))
    return 0 if all(o["ok"] for o in out) else 1


if __name__ == "__main__":
    sys.exit(main())
