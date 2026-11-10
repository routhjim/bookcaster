"""One asset, end to end: model script + spec -> Blender (checked, baked, glTF) -> UE import -> UE validation stills.

    python tools/assets/run_asset.py SPEC.json MODEL.py [--skip-render] [--formats glb,fbx,usd] [--source-sheet]

Locations come from tools/assets/config.py (environment variables). Output: <library>/<class>/<id>/ with
asset.json (the manifest entry), source/, export/, textures/, sheet/, ue/ (validation stills) and logs/;
<library>/manifest.json indexes every asset. Exit 0 when every step ran (checks may still fail: see asset.json).

GPU rules: the UE render holds one card's lock (/tmp/gpu-<bus>.lock, polled with a non-blocking flock, given up
after ASSETS_LOCK_WAIT_S) for the editor's lifetime only; Blender runs on the CPU. A guard stops the editor (by its
own PID) when MemAvailable falls under ASSETS_MIN_FREE_GB.
"""

import argparse
import datetime
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
CLASSES = ("prop", "vehicle", "set", "outfit", "creature", "character")


def mem_avail_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def validate_spec(spec):
    errs = []
    if not ID_RE.match(spec.get("id", "")):
        errs.append("id: lowercase letters, digits, _; 3-64 chars, starts with a letter")
    if spec.get("class") not in CLASSES:
        errs.append(f"class: one of {CLASSES}")
    if spec.get("tier") not in C.TIERS:
        errs.append(f"tier: one of {list(C.TIERS)}")
    s = spec.get("size_m")
    if not (isinstance(s, list) and len(s) == 3 and all(isinstance(x, (int, float)) and x >= 0 for x in s)):
        errs.append("size_m: [x, y, z] metres (x = width left-right, y = depth front-back, z = height)")
    if not spec.get("description"):
        errs.append("description: required")
    if errs:
        raise SystemExit("bad spec:\n  " + "\n  ".join(errs))


def run_logged(cmd, logf, env=None, timeout=None):
    t = time.time()
    with open(logf, "w") as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return r.returncode, round(time.time() - t, 1)


# ---------------------------------------------------------------- Blender
def blender_step(spec, model, adir, args, mode="final"):
    """Run the harness. Device per ASSETS_BLENDER_DEVICE: xtx = wait for an XTX lock; igpu = the 8060S if idle;
    auto = an XTX whose lock is free right now, else the idle iGPU, else CPU; cpu = never a GPU. An XTX lock is
    held for the whole Blender run. A failed HIP run is rerun on the CPU."""
    job = {"spec": spec, "model": str(Path(model).resolve()), "out": str(adir), "mode": mode,
           "formats": args.formats.split(","), "source_sheet": args.source_sheet, "sheet": True}
    want = args.device or C.BLENDER_DEVICE
    bus, fd, waited = (None, None, 0.0)
    if want in ("xtx", "auto"):
        bus, fd, waited = take_card(C.LOCK_WAIT_S if want == "xtx" else 0)
    if bus is None and want in ("igpu", "auto") and igpu_free():
        bus = C.IGPU_BUS
    tries = [("HIP", bus)] if bus else []
    tries.append(("CPU", None))
    rep, dt = {}, 0.0
    try:
        for dev, b in tries:
            job.update(device=dev, hip_bus=b or "")
            jp = adir / "logs" / f"blender_{mode}_job.json"
            jp.write_text(json.dumps(job, indent=1))
            (adir / "blender_report.json").unlink(missing_ok=True)
            t = time.time()
            with open(adir / "logs" / f"blender_{mode}.log", "w") as f:
                r = subprocess.run([str(c) for c in (C.BLENDER, "-b", "--factory-startup", "-t", C.THREADS,
                                                     "--python", HERE / "bl_harness.py", "--", jp)],
                                   stdout=f, stderr=subprocess.STDOUT, timeout=3600, pass_fds=(fd,) if fd else ())
            dt = round(time.time() - t, 1)
            rp = adir / "blender_report.json"
            rep = json.loads(rp.read_text()) if rp.exists() else {}
            log(f"blender {mode} on {dev}{' card ' + b if b else ''}: rc={r.returncode} {dt}s "
                f"status={rep.get('status')}")
            if r.returncode == 0 and rep.get("status") == "ok":
                break
            if rep.get("error") and "model" not in rep.get("times_s", {}):
                break                                   # the model script itself failed: no point retrying
    finally:
        if fd:
            os.close(fd)
    if rep.get("status") != "ok":
        raise SystemExit(f"blender step failed; see {adir / 'logs' / f'blender_{mode}.log'}\n{rep.get('error', '')}")
    rep["wall_s"] = dt
    rep["startup_s"] = round(dt - rep["times_s"].get("total", 0), 1)
    rep["lock_wait_s"] = waited
    (adir / "blender_report.json").write_text(json.dumps(rep, indent=1))
    return rep, dt


# ---------------------------------------------------------------- UE commandlets (no GPU)
def ue_cmdlet(script, job, logf, timeout=1800):
    jp = Path(str(logf) + ".job.json")
    jp.write_text(json.dumps(job, indent=1))
    env = dict(os.environ, ASSETS_UE_JOB=str(jp))
    if mem_avail_gb() < C.MIN_FREE_GB + 4:
        raise SystemExit(f"MemAvailable {mem_avail_gb():.1f} GiB: too little for a UE commandlet")
    rc, dt = run_logged([C.UE_CMD, C.UE_PROJECT, "-run=pythonscript", f"-script={script}", "-nullrhi",
                         "-unattended", "-nosplash", "-nosound", "-stdout"], logf, env=env, timeout=timeout)
    res = json.loads(Path(job["result"]).read_text()) if Path(job["result"]).exists() else {}
    return rc, dt, res


def project_content() -> Path:
    return Path(C.UE_PROJECT).parent / "Content"


def ensure_stage(logdir, rebuild=False):
    umap = project_content() / (C.UE_STAGE_MAP.replace("/Game/", "") + ".umap")
    if umap.exists() and not rebuild:
        return {"built": False}
    log("building the stage level")
    rc, dt, res = ue_cmdlet(HERE / "ue" / "ue_stage.py", {"map": C.UE_STAGE_MAP, "result": str(logdir / "stage.json"),
                             "exposure_ev": C.STAGE_EXPOSURE_EV},
                            logdir / "ue_stage.log")
    if not res.get("ok"):
        raise SystemExit(f"stage build failed: {res.get('error')}; see {logdir / 'ue_stage.log'}")
    return {"built": True, "time_s": dt}


def ensure_mannequin():
    """Copy the template mannequin (mesh, materials, textures, rig assets) into the project once. None if absent."""
    src = Path(C.UE_ENGINE) / "Templates/TemplateResources/High/Characters/Content/Mannequins"
    dst = project_content() / "Characters" / "Mannequins"
    if not (dst / "Meshes" / "SKM_Manny_Simple.uasset").exists():
        if not (src / "Meshes" / "SKM_Manny_Simple.uasset").exists():
            return None
        for sub in ("Meshes", "Materials", "Textures", "Rigs"):
            if (src / sub).exists():
                shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    return "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple"


# ---------------------------------------------------------------- GPU lock + render
def take_card(wait_s):
    t0 = time.time()
    while True:
        for bus in C.GPU_CARDS:
            fd = os.open(f"/tmp/gpu-{bus}.lock", os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return bus, fd, round(time.time() - t0, 1)
            except BlockingIOError:
                os.close(fd)
        if time.time() - t0 > wait_s:
            return None, None, round(time.time() - t0, 1)
        time.sleep(15)


def igpu_free():
    """The iGPU draws the desktop and runs the language model during director passes: use it only when idle
    and when system RAM (its memory) has room."""
    try:
        busy = int(Path(f"/sys/bus/pci/devices/0000:{C.IGPU_BUS}:00.0/gpu_busy_percent").read_text())
    except (OSError, ValueError):
        return False
    return busy <= C.IGPU_MAX_BUSY and mem_avail_gb() >= C.MIN_FREE_GB + 8


def vram_used(bus):
    for d in Path("/sys/class/drm").glob("card[0-9]*"):
        if (d / "device").exists() and os.path.realpath(d / "device").endswith(f"0000:{bus}:00.0"):
            try:
                return int((d / "device" / "mem_info_vram_used").read_text()) // 2**20
            except OSError:
                return None
    return None


TS = re.compile(r"^\[(\d{4})\.(\d\d)\.(\d\d)-(\d\d)\.(\d\d)\.(\d\d):(\d{3})\]")


def ue_phases(text, t_start_utc):
    """Seconds from process start to each landmark of the render log (UE stamps its log in UTC)."""
    marks = {"python_ready": "ASSETS shim: running", "set_up": "ASSETS render: set up", "shot": "ASSETS render: shot ",
             "done": "ASSETS render: done", "exit_logged": "LogExit: Exiting."}
    out, last = {}, None
    for line in text.splitlines():
        m = TS.match(line)
        if not m:
            continue
        y, mo, d, h, mi, se, ms = map(int, m.groups())
        t = (datetime.datetime(y, mo, d, h, mi, se, ms * 1000, tzinfo=datetime.timezone.utc).timestamp()
             - t_start_utc)
        last = t
        for k, pat in marks.items():
            if pat in line:
                if k == "shot":
                    out.setdefault("shots", []).append(round(t, 1))
                else:
                    out.setdefault(k, round(t, 1))
    out["last_log_line"] = round(last, 1) if last is not None else None
    return out


def render_step(spec, sm_path, adir, args, mannequin):
    out = adir / "ue"
    out.mkdir(exist_ok=True)
    shim_dst = project_content() / "Python" / "init_unreal.py"
    shim_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "ue" / "init_unreal.py", shim_dst)
    job = {"script": str(HERE / "ue" / "ue_render.py"), "map": C.UE_STAGE_MAP, "mesh": sm_path, "id": spec["id"],
           "out": str(out), "res": [1152, 640], "warm_s": args.warm_s, "settle_s": args.settle_s,
           "ref_box_material": C.UE_STAGE_MAP.rsplit("/", 1)[0] + "/M_StageRef", "mannequin": mannequin,
           "result": str(adir / "logs" / "ue_render.json")}
    jp = adir / "logs" / "ue_render_job.json"
    jp.write_text(json.dumps(job, indent=1))
    Path(job["result"]).unlink(missing_ok=True)
    while mem_avail_gb() < C.MIN_FREE_GB + 10:
        log(f"waiting for memory: MemAvailable {mem_avail_gb():.1f} GiB")
        time.sleep(30)
    bus, fd, waited = take_card(C.LOCK_WAIT_S)
    if bus is None:
        return {"ok": False, "error": f"no card free after {waited}s; CPU-only parts done"}
    log(f"card {bus} locked after {waited}s")
    logf = C.LOGS / f"assets_{spec['id']}_render.log"
    env = dict(os.environ, ASSETS_UE_JOB=str(jp), DRI_PRIME=f"pci-0000_{bus}_00_0")
    env.pop("MESA_VK_DEVICE_SELECT", None)     # DRI_PRIME puts the locked card first; adapter 0 is then that card
    cmd = [C.UE_EDITOR, C.UE_PROJECT, C.UE_STAGE_MAP, "-RenderOffscreen", "-graphicsadapter=0", "-vulkan", "-sm6",
           "-windowed", "-resx=1152", "-resy=640", "-unattended", "-nosplash", "-nosound", "-stdout"]
    t0 = time.time()
    t0_utc = datetime.datetime.now(datetime.timezone.utc).timestamp()
    vram = {b: [] for b in C.GPU_CARDS}
    guard = {"stop": False, "tripped": None}
    with open(logf, "w") as f:
        p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, pass_fds=(fd,))

        def watch():
            while p.poll() is None and not guard["stop"]:
                a = mem_avail_gb()
                for b in C.GPU_CARDS:
                    vram[b].append(vram_used(b))
                if a < C.MIN_FREE_GB:
                    guard["tripped"] = f"MemAvailable {a:.1f} GiB at {time.time() - t0:.0f}s; editor stopped"
                    p.terminate()
                time.sleep(5)

        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            rc = p.wait(timeout=args.ue_timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                rc = p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait()
        guard["stop"] = True
    os.close(fd)                                          # release the card as soon as the editor is gone
    dt = round(time.time() - t0, 1)
    res = json.loads(Path(job["result"]).read_text()) if Path(job["result"]).exists() else {"ok": False}
    text = logf.read_text(errors="replace")
    gpu = re.findall(r'Metadata set : gpu="([^"]+)"', text)
    rt = re.findall(r'Metadata set : raytracing="(\d)"', text)
    res.update({"rc": rc, "time_s": dt, "card": bus, "lock_wait_s": waited, "log": str(logf),
                "gpu_name": gpu[-1] if gpu else None, "raytracing_metadata": rt[-1] if rt else None,
                "vram_peak_mib": {b: max([v for v in vram[b] if v is not None], default=None) for b in vram},
                "guard": guard["tripped"], "phases_s": ue_phases(text, t0_utc),
                "shader_jobs_logged": len(re.findall(r"LogShaderCompilers: Display: .*[Cc]ompil", text))})
    if not res.get("ok") and not res.get("error"):
        res["error"] = f"editor rc={rc}; no result; see {logf}"
    log(f"render rc={rc} {dt}s ok={res.get('ok')} card={bus} vram_peak={res['vram_peak_mib']}")
    return res


# ---------------------------------------------------------------- manifest
def write_manifest(spec, adir, blend, imp, render, model, times):
    entry = {
        "id": spec["id"], "class": spec["class"], "description": spec["description"], "tags": spec.get("tags", []),
        "tier": spec["tier"], "size_m": spec["size_m"], "placement": spec.get("placement", "floor"),
        "source": spec.get("source", "scripted"), "licence": spec.get("licence", "unknown"),
        "contract_version": 1,
        "blender": {"version": blend.get("blender"), "source_blend": rel(blend["files"].get("blend"), adir),
                    "model_script": rel(adir / "source" / Path(model).name, adir), "dims_m": blend.get("dims_m"),
                    "tris": blend.get("tris"), "parts": blend.get("parts"), "texture_px": blend.get("texture_px")},
        "export": {k: rel(v, adir) for k, v in blend["files"].get("export", {}).items()},
        "textures": {k: rel(v, adir) for k, v in blend["files"].get("textures", {}).items()},
        "ue": {"path": imp.get("static_mesh"), "materials": imp.get("materials"), "nanite": imp.get("nanite"),
               "bounds_cm": imp.get("bounds_cm"), "import_problems": imp.get("problems")},
        "thumbnails": {"sheet": rel(blend["files"].get("sheet"), adir),
                       "ue": [rel(s["path"], adir) for s in (render or {}).get("shots", [])]},
        "checks": {c["name"]: c["ok"] for c in blend["checks"]},
        "checks_passed": all(c["ok"] for c in blend["checks"]) and not imp.get("problems"),
        "check_details": blend["checks"],
        "rig": None,
        "render": {k: (render or {}).get(k) for k in ("ok", "card", "gpu_name", "raytracing_metadata", "cvars",
                                                      "time_s", "error")},
        "times_s": times,
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (adir / "asset.json").write_text(json.dumps(entry, indent=1))
    mp = C.LIBRARY / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"assets": {}}
    man["assets"][spec["id"]] = {"class": spec["class"], "tier": spec["tier"], "dir": rel(adir, C.LIBRARY),
                                 "ue_path": entry["ue"]["path"], "checks_passed": entry["checks_passed"],
                                 "tags": entry["tags"], "description": spec["description"]}
    tmp = mp.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1))
    tmp.replace(mp)
    return entry


def rel(p, base):
    if not p:
        return p
    try:
        return str(Path(p).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("model")
    ap.add_argument("--formats", default="glb")
    ap.add_argument("--source-sheet", action="store_true", help="also render the unbaked model (bake loss check)")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--rebuild-stage", action="store_true", help="rebuild the stage level (after changing it)")
    ap.add_argument("--preview", action="store_true",
                    help="modelling loop: model + checks + small unbaked sheet only (no bake, export or UE)")
    ap.add_argument("--device", choices=("auto", "xtx", "igpu", "cpu"), help="Blender bake/sheet device (default config)")
    ap.add_argument("--warm-s", type=float, default=20)
    ap.add_argument("--settle-s", type=float, default=3)
    ap.add_argument("--ue-timeout", type=float, default=2400)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    validate_spec(spec)
    adir = C.LIBRARY / spec["class"] / spec["id"]
    for d in ("source", "logs", "ue"):
        (adir / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.model, adir / "source" / Path(args.model).name)
    (adir / "source" / "spec.json").write_text(json.dumps(spec, indent=1))
    times = {}
    t0 = time.time()
    log(f"{spec['id']}: blender ({C.BLENDER})")
    if args.preview:
        rep, dt = blender_step(spec, args.model, adir, args, "preview")
        failed = [c["name"] for c in rep["checks"] if not c["ok"]]
        log(f"preview {dt}s; sheet {rep['files'].get('preview_sheet')}; failed checks: {failed or 'none'}")
        return 0
    blend, times["blender"] = blender_step(spec, args.model, adir, args)
    failed = [c["name"] for c in blend["checks"] if not c["ok"]]
    if failed:
        log(f"checks failed: {failed} (continuing; recorded in asset.json)")
    glb = blend["files"]["export"].get("glb")
    log("UE import")
    dest = f"{C.UE_ASSET_ROOT}/{spec['class'].capitalize()}s/{spec['id']}"   # Interchange adds <file name>/ inside

    rc, times["ue_import"], imp = ue_cmdlet(HERE / "ue" / "ue_import.py",
                                            {"file": glb, "dest": dest, "id": spec["id"], "nanite": True,
                                             "result": str(adir / "logs" / "ue_import.json")},
                                            adir / "logs" / "ue_import.log")
    log(f"import ok={imp.get('ok')} {times['ue_import']}s -> {imp.get('static_mesh')} problems={imp.get('problems')}")
    if not imp.get("ok"):
        raise SystemExit(f"UE import failed: {imp.get('error')}")
    render = None
    if not args.skip_render:
        st = ensure_stage(adir / "logs", args.rebuild_stage)
        if st.get("built"):
            times["ue_stage"] = st["time_s"]
        render = render_step(spec, imp["static_mesh"], adir, args, ensure_mannequin())
        times["ue_render"] = render.get("time_s")
    times["total"] = round(time.time() - t0, 1)
    entry = write_manifest(spec, adir, blend, imp, render, args.model, times)
    log(f"done in {times['total']}s; checks_passed={entry['checks_passed']}; render ok={(render or {}).get('ok')}")
    log(f"asset folder: {adir}")
    return 0 if (render is None or render.get("ok")) else 3


if __name__ == "__main__":
    sys.exit(main())
