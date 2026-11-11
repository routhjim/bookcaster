"""Set builder: set spec (JSON, see SETS.md) -> UE level /Game/Sets/<id>/L_<id> + validation stills.

    python tools/assets/sets/build_set.py SPEC.json                 # plan, check, build the level, render stills
    python tools/assets/sets/build_set.py SPEC.json --render-only   # stills of the saved level (no rebuild)
    python tools/assets/sets/build_set.py SPEC.json --cameras wide_south,low
    python tools/assets/sets/build_set.py SPEC.json --plan-only     # expand + checks, no UE
    python tools/assets/sets/build_set.py --stop-editor             # ask the set editor to quit

Stdlib only. Locations come from the asset chain's config.py (ASSETS_* variables); SETS_UE_PROJECT overrides the
project (default: the chain's), SETS_OUT the output root (default: <library>/set when building in the chain's
project, else $ASSETS_UE_HOME/lab/sets). Output per set: <out>/<id>/asset.json (manifest entry), spec.json,
scene.json (the expanded scene), ue/<id>_<camera>.png, logs/.

UE: one long-lived editor per project (started here, then fed jobs through <project>/Saved/SetServer/queue; it
quits after SETS_UE_IDLE_S without work or on --stop-editor). It holds one XTX's lock (/tmp/gpu-<bus>.lock, taken
with a polled non-blocking flock) and the project's lock (<project>/Saved/AssetServer/project.lock, the asset
chain's) for its whole life, and it never starts while the chain's own editor is alive in the same project: two
editors saving into one project can corrupt it. A detached guard stops it (by PID) when MemAvailable falls under
ASSETS_MIN_FREE_GB. The editor never inherits the lock descriptors: the guard holds them and exits when the
editor does, so no child process (zenserver, trace server) can keep a card locked. While <ASSETS_UE_HOME>/GPU_HOLD
exists no editor is started. SETS_UE_IDLE_S (60) is how long the editor waits for another job before quitting.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402  (the asset chain's locations)
import layout  # noqa: E402
import presets  # noqa: E402

PROJECT = Path(os.path.expanduser(os.environ.get("SETS_UE_PROJECT", str(C.UE_PROJECT))))
IDLE_S = float(os.environ.get("SETS_UE_IDLE_S", "20"))   # rules: no private editor kept for long
GPU_HOLD = Path(C.UE_HOME) / "GPU_HOLD"
MANNEQUIN = "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple"


def out_root():
    if os.environ.get("SETS_OUT"):
        return Path(os.path.expanduser(os.environ["SETS_OUT"]))
    if PROJECT.resolve() == Path(C.UE_PROJECT).resolve():
        return Path(C.LIBRARY) / "set"
    return Path(C.UE_HOME) / "lab" / "sets"


def log(msg):
    print(f"[sets {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 1024 / 1024
    return 0.0


# ------------------------------------------------------------------ plan
def library_assets():
    man = json.loads((Path(C.LIBRARY) / "manifest.json").read_text())
    out = {}
    for aid, a in man["assets"].items():
        p = Path(C.LIBRARY) / a["dir"] / "asset.json"
        if not p.exists():
            continue
        ent = json.loads(p.read_text())
        ue = ent.get("ue") or {}
        if ue.get("bounds_cm") and a.get("ue_path"):
            out[aid] = {"ue_path": a["ue_path"].split(".")[0], "bounds_cm": ue["bounds_cm"], "class": a["class"]}
    return out


def validate(spec):
    errs = []
    for k in ("id", "description", "size_m", "structure", "materials", "cameras"):
        if k not in spec:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    if not layout_id_ok(spec["id"]):
        errs.append("id must be lowercase letters, digits, underscore, 3-64 chars, starting with a letter")
    used = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith("material") and isinstance(v, str):
                    used.add(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(spec["structure"])
    walk(spec.get("lights", []))
    for m in sorted(used - set(spec["materials"])):
        errs.append(f"material {m!r} used but not defined in materials")
    return errs


def layout_id_ok(s):
    import re
    return bool(re.fullmatch(r"[a-z][a-z0-9_]{2,63}", s))


def plan(spec):
    t = time.time()
    errs = validate(spec)
    if errs:
        raise SystemExit("spec errors:\n  " + "\n  ".join(errs))
    assets = library_assets()
    missing = sorted({it["asset"] for k in ("fixtures", "props") for it in spec.get(k, [])} - set(assets))
    if missing:
        raise SystemExit(f"assets not in the library (or not imported to UE): {missing}")
    scene = layout.expand(spec, assets)
    scene["materials"] = {k: presets.resolve(v) for k, v in spec["materials"].items()}
    problems = layout.check(spec, assets, scene.pop("placed"))
    scene["plan_s"] = round(time.time() - t, 3)
    return scene, problems


# ------------------------------------------------------------------ editor
def proj_dir():
    return PROJECT.parent


def server_dir():
    d = proj_dir() / "Saved" / "SetServer"
    (d / "queue").mkdir(parents=True, exist_ok=True)
    return d


def alive(pid):
    try:
        return "UnrealEditor" in Path(f"/proc/{pid}/cmdline").read_text(errors="replace")
    except OSError:
        return False


def live_server():
    sp = server_dir() / "state.json"
    if not sp.exists():
        return None
    st = json.loads(sp.read_text())
    if not alive(st["pid"]) or (server_dir() / "queue" / "quit").exists():
        return None
    return st


def chain_editor_alive():
    sp = proj_dir() / "Saved" / "AssetServer" / "state.json"
    if not sp.exists():
        return None
    try:
        st = json.loads(sp.read_text())
    except ValueError:
        return None
    return st["pid"] if alive(st["pid"]) else None


def take_lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        return None


def acquire(wait_s):
    """Project lock (no chain editor alive in this project) and one free XTX, both polled non-blocking."""
    t0 = time.time()
    pl = proj_dir() / "Saved" / "AssetServer" / "project.lock"
    pl.parent.mkdir(parents=True, exist_ok=True)
    said = set()
    while True:
        if GPU_HOLD.exists():
            if "hold" not in said:
                log(f"{GPU_HOLD} exists: no new card job until it is gone")
                said.add("hold")
            time.sleep(10)
            continue
        pfd = take_lock(pl)
        if pfd is not None:
            ce = chain_editor_alive()
            if ce:
                os.close(pfd)
                pfd = None
                if "chain" not in said:
                    log(f"the asset chain's editor (pid {ce}) is using this project; waiting for it to quit")
                    said.add("chain")
            else:
                for bus in C.UE_CARDS:
                    cfd = take_lock(f"/tmp/gpu-{bus}.lock")
                    if cfd is not None:
                        return pfd, bus, cfd, round(time.time() - t0, 1)
                os.close(pfd)
                if "card" not in said:
                    log("both cards busy; polling")
                    said.add("card")
        if time.time() - t0 > wait_s:
            return None, None, None, round(time.time() - t0, 1)
        time.sleep(10)


def install_shim():
    dst = proj_dir() / "Content" / "Python" / "init_unreal.py"
    src = HERE.parent / "ue" / "init_unreal.py"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def start_editor(job, jp, wait_s):
    pfd, bus, cfd, waited = acquire(wait_s)
    if pfd is None:
        raise SystemExit(f"no project lock / card after {waited}s")
    install_shim()
    while mem_avail_gb() < C.MIN_FREE_GB + 10:
        log(f"waiting for memory: MemAvailable {mem_avail_gb():.1f} GiB")
        time.sleep(15)
    job["serve_dir"] = str(server_dir() / "queue")
    job["serve_idle_s"] = IDLE_S
    (server_dir() / "queue" / "quit").unlink(missing_ok=True)
    jp.write_text(json.dumps(job, indent=1))
    env = dict(os.environ, DRI_PRIME=f"pci-0000_{bus}_00_0", ASSETS_UE_JOB=str(jp))
    env.pop("MESA_VK_DEVICE_SELECT", None)
    cmd = [str(C.UE_EDITOR), str(PROJECT), "-RenderOffscreen", "-graphicsadapter=0", "-vulkan", "-sm6", "-windowed",
           "-resx=1152", "-resy=640", "-unattended", "-nosplash", "-nosound", "-stdout"]
    logf = Path(C.LOGS) / f"sets_ue_server_{time.strftime('%m%d_%H%M%S')}.log"
    f = open(logf, "w")
    # the editor does NOT inherit the lock fds (its zenserver / trace-server children would keep them after it
    # quits); the small guard process holds both locks until the editor is gone, then exits
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, close_fds=True, start_new_session=True)
    f.close()
    (server_dir() / "state.json").write_text(json.dumps({"pid": p.pid, "card": bus, "log": str(logf),
                                                         "started": time.ctime(), "project": str(PROJECT)}))
    subprocess.Popen([sys.executable, __file__, "--guard", str(p.pid)], start_new_session=True, pass_fds=(pfd, cfd),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(pfd)
    os.close(cfd)
    log(f"editor pid {p.pid} on card {bus} (lock wait {waited}s); log {logf}")
    return p.pid, bus, logf, waited


def guard_main(pid):
    while alive(pid):
        if mem_avail_gb() < C.MIN_FREE_GB:
            (server_dir() / "guard.txt").write_text(f"MemAvailable {mem_avail_gb():.1f} GiB at {time.ctime()}; "
                                                    f"editor {pid} stopped\n")
            os.kill(pid, 15)
        time.sleep(2)
    sp = server_dir() / "state.json"
    if sp.exists() and json.loads(sp.read_text()).get("pid") == pid:
        sp.unlink()
    (server_dir() / "queue" / "quit").unlink(missing_ok=True)


def run_job(job, adir, wait_s, timeout):
    """Hand the job to the live set editor or start one; wait for the result file."""
    res_p = Path(job["result"])
    res_p.unlink(missing_ok=True)
    t0 = time.time()
    srv = live_server()
    if srv:
        q = server_dir() / "queue" / f"{job['id']}.{os.getpid()}.{int(t0)}.job.json"
        tmp = q.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=1))
        tmp.replace(q)
        pid, bus, logf, waited, started = srv["pid"], srv["card"], Path(srv["log"]), 0.0, False
        log(f"job handed to the running set editor (pid {pid}, card {bus})")
    else:
        pid, bus, logf, waited = start_editor(job, adir / "logs" / "ue_job.json", wait_s)
        started = True
    while not res_p.exists():
        if not alive(pid):
            break
        if time.time() - t0 > timeout:
            log(f"no result after {timeout:.0f} s: stopping editor {pid}")
            os.kill(pid, 15)
            break
        time.sleep(0.25)
    dt = round(time.time() - t0, 1)
    res = json.loads(res_p.read_text()) if res_p.exists() else {"ok": False, "error": f"no result; see {logf}"}
    res.update({"wall_s": dt, "card": bus, "lock_wait_s": waited, "editor": "started" if started else "warm",
                "log": str(logf)})
    return res


# ------------------------------------------------------------------ scheduler (default)
def run_sched(job, adir, timeout):
    """Submit the set job to the asset scheduler's resident editor (run_asset.py --ue-job): no editor, guard or
    card lock of our own. The stills are the job's outputs."""
    ej = dict(job)
    ej.pop("script", None)
    want = ej.get("cameras")
    outs = [] if not ej.get("render") else [str(Path(ej["out"]) / f"{ej['id']}_{c['name']}.png")
                                           for c in json.loads(Path(ej["scene"]).read_text())["cameras"]
                                           if not want or c["name"] in want]
    if PROJECT.resolve() == Path(C.UE_PROJECT).resolve():
        ej["restore_map"] = C.UE_STAGE_MAP           # the chain's asset jobs expect their stage level loaded
    jd = {"project": str(PROJECT), "script": str(HERE / "ue" / "set_job.py"), "job": ej, "outputs": outs,
          "timeout_s": timeout, "id": f"set_{ej['id']}", "log_dir": str(adir)}
    jp = adir / "logs" / "ue_job_sched.json"
    jp.write_text(json.dumps(jd, indent=1))
    for o in outs:
        Path(o).unlink(missing_ok=True)
    t0 = time.time()
    for attempt in range(3):              # a scheduler restart (code update) drops queued jobs: resubmit
        p = subprocess.run([sys.executable, str(HERE.parent / "run_asset.py"), "--ue-job", str(jp)],
                           capture_output=True, text=True, env=dict(os.environ, ASSETS_SCHED="1"))
        if "scheduler went away" not in p.stderr:
            break
        log(f"the scheduler restarted under the job; resubmitting ({attempt + 1})")
        time.sleep(5)
    dt = round(time.time() - t0, 1)
    (adir / "logs" / "sched_client.log").write_text(p.stdout + "\n--- stderr\n" + p.stderr)
    last = None
    for ln in p.stdout.splitlines():
        if ln.startswith("{"):
            try:
                last = json.loads(ln)
            except ValueError:
                pass
    inner = (last or {}).get("result") or {}
    if isinstance(inner, dict) and "result" in inner and isinstance(inner["result"], dict):
        inner = inner["result"]
    res = dict(inner) if isinstance(inner, dict) else {}
    res.update({"ok": p.returncode == 0 and bool(res.get("ok")), "wall_s": dt, "editor": "scheduler",
                "lock_wait_s": None, "card": None, "client_rc": p.returncode,
                "error": None if p.returncode == 0 else ((last or {}).get("error") or p.stderr[-2000:] or p.stdout[-2000:])})
    return res


# ------------------------------------------------------------------ main
def write_manifest(spec, adir, scene, problems, res, times):
    shots = [str(Path(s["path"]).relative_to(adir)) for s in res.get("shots", [])]
    ent = {"id": spec["id"], "class": "set", "description": spec["description"], "tags": spec.get("tags", []),
           "size_m": spec["size_m"], "source": "scripted", "licence": "CC0 (original spec and code)",
           "level": f"/Game/Sets/{spec['id']}/L_{spec['id']}", "project": str(PROJECT),
           "spec": "spec.json", "scene": "scene.json", "counts": scene["counts"],
           "assets_used": sorted({a["id"] for a in scene["assets"]}), "plan_problems": problems,
           "thumbnails": {"ue": shots}, "render": {k: res.get(k) for k in ("card", "editor", "cvars", "log")},
           "times_s": times, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (adir / "asset.json").write_text(json.dumps(ent, indent=1))
    if out_root() == Path(C.LIBRARY) / "set":
        mp = Path(C.LIBRARY) / "manifest.json"
        fd = take_lock_blocking(Path(C.LIBRARY) / ".manifest.lock")
        try:
            man = json.loads(mp.read_text())
            man["assets"][spec["id"]] = {"class": "set", "dir": f"set/{spec['id']}", "ue_path": ent["level"],
                                         "checks_passed": not problems, "tags": ent["tags"],
                                         "description": spec["description"]}
            tmp = mp.with_name(f".manifest.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(man, indent=1))
            tmp.replace(mp)
        finally:
            os.close(fd)


def take_lock_blocking(p, wait_s=60):
    t0 = time.time()
    while True:
        fd = take_lock(p)
        if fd is not None or time.time() - t0 > wait_s:
            return fd if fd is not None else os.open(p, os.O_RDONLY)
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--cameras", default="")
    ap.add_argument("--warm-frames", type=int, default=240)
    ap.add_argument("--settle-frames", type=int, default=180)
    ap.add_argument("--lock-wait", type=float, default=C.LOCK_WAIT_S)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--stop-editor", action="store_true")
    ap.add_argument("--private-editor", action="store_true",
                    help="fallback: start a short-lived editor of our own instead of the scheduler's")
    ap.add_argument("--guard", type=int)
    a = ap.parse_args()
    if a.guard:
        return guard_main(a.guard)
    if a.stop_editor:
        if live_server():
            (server_dir() / "queue" / "quit").touch()
            log("quit requested")
        return
    t_all = time.time()
    spec = json.loads(Path(a.spec).read_text())
    scene, problems = plan(spec)
    adir = out_root() / spec["id"]
    (adir / "logs").mkdir(parents=True, exist_ok=True)
    (adir / "ue").mkdir(exist_ok=True)
    (adir / "spec.json").write_text(json.dumps(spec, indent=1))
    sp = adir / "scene.json"
    sp.write_text(json.dumps(scene))
    log(f"{spec['id']}: {len(scene['prims'])} primitives, {len(scene['assets'])} assets, {len(scene['lights'])} "
        f"lights; plan {scene['plan_s']} s")
    for p in problems:
        log(f"PLAN PROBLEM: {p}")
    if a.plan_only:
        return
    job = {"script": str(HERE / "ue" / "set_server.py"), "id": spec["id"], "scene": str(sp),
           "level": f"/Game/Sets/{spec['id']}/L_{spec['id']}", "build": not a.render_only,
           "render": not a.no_render, "cameras": [c for c in a.cameras.split(",") if c] or None,
           "out": str(adir / "ue"), "result": str(adir / "logs" / "ue_result.json"), "res": [1152, 640],
           "warm_frames": a.warm_frames, "settle_frames": a.settle_frames, "mannequin": MANNEQUIN}
    if a.private_editor:
        res = run_job(job, adir, a.lock_wait, a.timeout)
    else:
        res = run_sched(job, adir, a.timeout)
    times = {"plan": scene["plan_s"], "ue_wall": res["wall_s"], "lock_wait": res["lock_wait_s"],
             "ue": res.get("times_s"), "build": (res.get("build") or {}).get("times_s"),
             "editor": res["editor"], "total": round(time.time() - t_all, 1)}
    (adir / "logs" / "times.json").write_text(json.dumps(times, indent=1))
    if res.get("ok"):
        write_manifest(spec, adir, scene, problems, res, times)
    log(f"{spec['id']}: ok={res.get('ok')} total {times['total']} s (UE {res['wall_s']} s, editor {res['editor']})")
    if not res.get("ok"):
        log(f"error: {res.get('error')}")
        sys.exit(1)
    for s in res.get("shots", []):
        log(f"  {s['path']}")


if __name__ == "__main__":
    main()
