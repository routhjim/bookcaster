"""One asset, end to end: model script + spec -> Blender (checked, baked, glTF) -> UE import -> UE validation stills.

    python tools/assets/run_asset.py SPEC.json MODEL.py [--skip-render] [--formats glb,fbx,usd] [--source-sheet]

Locations come from tools/assets/config.py (environment variables). Output: <library>/<class>/<id>/ with
asset.json (the manifest entry), source/, export/, textures/, sheet/, ue/ (validation stills) and logs/;
<library>/manifest.json indexes every asset. Exit 0 when every step ran (checks may still fail: see asset.json).

GPU rules: the UE render holds one card's lock (/tmp/gpu-<bus>.lock, polled with a non-blocking flock, given up
after ASSETS_LOCK_WAIT_S) for the editor's lifetime only; Blender bakes and renders on an XTX under its lock (else
the idle iGPU, else CPU). A guard stops the editor (by its own PID) when MemAvailable falls under ASSETS_MIN_FREE_GB.
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
def make_blender_job(spec, model, adir, args, mode, ue_follows=True):
    """The harness job. With UE stills following, the final pass has no Blender sheet unless --sheet (320 px,
    16 spp), --full-sheet (512 px, 32 spp), --sheet-px or --sheet-samples asks for one; without UE (--skip-render)
    the 512 px sheet is made as before; --no-sheet drops it always."""
    full = getattr(args, "full_sheet", False) or not ue_follows
    sheet_px = getattr(args, "sheet_px", None) or (512 if full else 320)
    # no denoiser by default: Blender runs OIDN on the CPU here, 0.4-4.4 s a tile under load (a 12 spp preview
    # took 12-15 s, 30 s of CPU); path tracing more samples on the card costs ~0.03 s a tile instead
    dn = bool(getattr(args, "denoise", False))
    sheet_samples = args.sheet_samples or ((32 if full else 16) if dn else (128 if full else 64))
    preview_samples = args.preview_samples or (12 if dn else 32)
    # with UE stills following, the sheet is off unless asked for (it cost 10-13 s per asset on a shared card)
    want_sheet = not args.no_sheet and bool(not ue_follows or getattr(args, "sheet", False) or full
                                            or getattr(args, "sheet_px", None) or args.sheet_samples)
    return {"spec": spec, "model": str(Path(model).resolve()), "out": str(adir), "mode": mode,
            "formats": args.formats.split(","), "source_sheet": args.source_sheet, "sheet": want_sheet,
            "hiprt": args.hiprt, "preview_px": args.preview_px, "preview_samples": preview_samples,
            "sheet_px": sheet_px, "sheet_samples": sheet_samples, "denoise": dn}


def blender_step(spec, model, adir, args, mode="final"):
    """Run the harness (the --local path). Device per ASSETS_BLENDER_DEVICE: a GPU (an XTX under its lock, or the
    iGPU), waited for; never the CPU unless --device cpu is asked for by hand."""
    job = make_blender_job(spec, model, adir, args, mode, ue_follows=not args.skip_render)
    want = args.device or C.BLENDER_DEVICE
    bus, fds, waited, how = take_bake_device(want, preview=(mode == "preview"))
    log(f"blender {mode}: {how} after {waited}s")
    if bus is None and want != "cpu":
        raise SystemExit(f"no GPU free after {waited}s (the chain never bakes on the CPU); try again later")
    tries = [("HIP", bus)] if bus else [("CPU", None)]
    rep, dt = {}, 0.0
    try:
        for dev, b in tries:
            job.update(device=dev, hip_bus=b or "")
            jp = adir / "logs" / f"blender_{mode}_job.json"
            jp.write_text(json.dumps(job, indent=1))
            (adir / "blender_report.json").unlink(missing_ok=True)
            t = time.time()
            sampler = LoadSampler(b)
            with open(adir / "logs" / f"blender_{mode}.log", "w") as f:
                r = subprocess.run([str(c) for c in (C.BLENDER, "-b", "--factory-startup", "-t", C.BLENDER_THREADS,
                                                     "--python", HERE / "bl_harness.py", "--", jp)],
                                   stdout=f, stderr=subprocess.STDOUT, timeout=3600, pass_fds=tuple(fds))
            dt = round(time.time() - t, 1)
            load = sampler.done()
            rp = adir / "blender_report.json"
            rep = json.loads(rp.read_text()) if rp.exists() else {}
            log(f"blender {mode} on {dev}{' card ' + b if b else ''}: rc={r.returncode} {dt}s "
                f"status={rep.get('status')}")
    finally:
        for x in fds:
            os.close(x)
    if rep.get("status") != "ok":
        raise SystemExit(f"blender step failed; see {adir / 'logs' / f'blender_{mode}.log'}\n{rep.get('error', '')}")
    rep["wall_s"] = dt
    rep["startup_s"] = round(dt - rep["times_s"].get("total", 0), 1)
    rep["lock_wait_s"] = waited
    rep["device_choice"] = how
    rep["gpu_load"] = load
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
    umap.unlink(missing_ok=True)          # new_level refuses to overwrite an existing map, even after delete_asset
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
def take_card(wait_s, cards=None):
    t0 = time.time()
    while True:
        for bus in cards or C.GPU_CARDS:
            fd = os.open(f"/tmp/gpu-{bus}.lock", os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return bus, fd, round(time.time() - t0, 1)
            except BlockingIOError:
                os.close(fd)
        if time.time() - t0 > wait_s:
            return None, None, round(time.time() - t0, 1)
        time.sleep(15)


def try_flock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        return None


def vram_free_gb(bus):
    d = _card_dir(bus)
    try:
        return (int((d / "mem_info_vram_total").read_text()) - int((d / "mem_info_vram_used").read_text())) / 2**30
    except (OSError, ValueError, TypeError):
        return 0.0


def take_bake_device(want, preview=False):
    """A device for a Blender bake/render: (bus | None, fds to hold, seconds waited, how).

    A card is used when its lock is free, or when its lock is held by this chain's warm UE editor and the card has
    ASSETS_BAKE_VRAM_GB free: then only the card's bake slot (/tmp/gpu-<bus>.bake.lock, one Blender job per card) is
    taken, so bakes are not starved by the idle editor (the editor uses ~5-6 GB of the 24). A job holding a card's
    main lock also takes its bake slot, so two bakes never share a card. Then the iGPU (its lock, when idle).
    Polled every 2 s for up to ASSETS_LOCK_WAIT_S; then (None, [], waited, "no GPU free"): the caller fails
    rather than bake on the CPU (the owner's rule). Previews try the iGPU first. Used by --local runs and
    paint.py; the scheduler (assetd.py) has its own fixed-home policy."""
    t0 = time.time()
    order = []
    if want in ("xtx", "auto"):
        order += [("card", b) for b in C.BLENDER_CARDS]
    if want in ("igpu", "auto"):
        order.insert(0 if preview else len(order), ("igpu", C.IGPU_BUS))
    if want == "cpu" or not order:
        return None, [], 0.0, "cpu (asked)"
    limit = C.LOCK_WAIT_S
    while True:
        srv = None
        for kind, b in order:
            if kind == "igpu":
                if igpu_free():
                    fd = try_flock(f"/tmp/gpu-{C.IGPU_LOCK_NAME}.lock")
                    if fd is not None:
                        return b, [fd], round(time.time() - t0, 1), "igpu"
                continue
            slot = try_flock(f"/tmp/gpu-{b}.bake.lock")
            if slot is None:
                continue                                   # a bake already runs on this card
            fd = try_flock(f"/tmp/gpu-{b}.lock")
            if fd is not None:
                return b, [fd, slot], round(time.time() - t0, 1), f"card {b}"
            if srv is None:
                srv = live_server() or {}
            if srv.get("card") == b and vram_free_gb(b) >= C.BAKE_VRAM_GB:
                return b, [slot], round(time.time() - t0, 1), f"card {b} beside the warm editor"
            os.close(slot)
        if time.time() - t0 > limit:
            return None, [], round(time.time() - t0, 1), "no GPU free"
        time.sleep(2)


def _card_dir(bus):
    for d in Path("/sys/class/drm").glob("card[0-9]*"):
        if (d / "device").exists() and os.path.realpath(d / "device").endswith(f"0000:{bus}:00.0"):
            return d / "device"
    return None


def gpu_load(bus):
    """(busy %, watts) of one card from sysfs."""
    d = _card_dir(bus)
    try:
        busy = int((d / "gpu_busy_percent").read_text())
        pw = next(iter(d.glob("hwmon/hwmon*/power1_average")), None) or next(iter(d.glob("hwmon/hwmon*/power1_input")),
                                                                          None)
        return busy, (int(pw.read_text()) / 1e6 if pw else None)
    except (OSError, ValueError, TypeError):
        return None, None


def summarize_load(samples):
    b = [x[0] for x in samples if x[0] is not None]
    w = [x[1] for x in samples if x[1] is not None]
    return {"samples": len(samples), "busy_mean": round(sum(b) / len(b), 1) if b else None, "busy_peak": max(b, default=None),
            "watts_mean": round(sum(w) / len(w), 1) if w else None, "watts_peak": max(w, default=None)}


class LoadSampler:
    def __init__(self, bus):
        self.bus, self.s, self.stop = bus, [], False
        self.th = threading.Thread(target=self.run, daemon=True)
        if bus:
            self.th.start()

    def run(self):
        while not self.stop:
            self.s.append(gpu_load(self.bus))
            time.sleep(0.25)

    def done(self):
        self.stop = True
        return summarize_load(self.s)


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


def server_dir() -> Path:
    d = Path(C.UE_PROJECT).parent / "Saved" / "AssetServer"
    (d / "queue").mkdir(parents=True, exist_ok=True)
    return d


class ProjectLock:
    """One editor per UE project: two editors saving into the same project can corrupt its asset registry and
    packages, so starting an editor (or the import commandlet) goes through this lock (non-blocking, polled)."""

    def __enter__(self):
        self.fd = os.open(server_dir() / "project.lock", os.O_RDWR | os.O_CREAT, 0o666)
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                time.sleep(2)

    def __exit__(self, *a):
        os.close(self.fd)


def live_server():
    sp = server_dir() / "state.json"
    if not sp.exists():
        return None
    st = json.loads(sp.read_text())
    try:
        alive = "UnrealEditor" in Path(f"/proc/{st['pid']}/cmdline").read_text(errors="replace")
    except OSError:
        alive = False
    if not alive or (server_dir() / "queue" / "quit").exists():
        sp.unlink(missing_ok=True)
        return None
    return st


def install_shim():
    dst = project_content() / "Python" / "init_unreal.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = (HERE / "ue" / "init_unreal.py").read_bytes()
    if not dst.exists() or dst.read_bytes() != src:
        tmp = dst.with_name(f".init_unreal.{os.getpid()}.tmp")
        tmp.write_bytes(src)
        tmp.replace(dst)


def ensure_rt_ini():
    """Write the ray-tracing renderer settings into the project's DefaultEngine.ini (ASSETS_UE_RAYTRACING: 1 =
    hardware ray tracing + Lumen hardware tracing, 0 = software Lumen). Called before an editor starts; an editor
    already running keeps what it started with. Returns True when the file changed (the start recompiles shaders)."""
    ini = Path(C.UE_PROJECT).parent / "Config" / "DefaultEngine.ini"
    want = dict(C.UE_RT_SETTINGS)
    if not C.UE_RAYTRACING:
        want.update({"r.RayTracing": "False", "r.Lumen.HardwareRayTracing": "False"})
    lines = ini.read_text().splitlines() if ini.exists() else []
    sec = "[/Script/Engine.RendererSettings]"
    if sec not in lines:
        lines += ["", sec]
    out, inside, done = [], False, set()
    for ln in lines:
        if ln.strip().startswith("["):
            if inside:                                     # leaving the section: add what was missing
                out += [f"{k}={v}" for k, v in want.items() if k not in done]
                done |= set(want)
            inside = ln.strip() == sec
        elif inside and "=" in ln and ln.split("=", 1)[0].strip() in want:
            k = ln.split("=", 1)[0].strip()
            out.append(f"{k}={want[k]}")
            done.add(k)
            continue
        out.append(ln)
    if inside:
        out += [f"{k}={v}" for k, v in want.items() if k not in done]
    new = "\n".join(out) + "\n"
    if ini.exists() and ini.read_text() == new:
        return False
    tmp = ini.with_name(f".DefaultEngine.{os.getpid()}.tmp")
    tmp.write_text(new)
    tmp.replace(ini)
    return True


def editor_cmd(bus):
    env = dict(os.environ, DRI_PRIME=f"pci-0000_{bus}_00_0")
    env.pop("MESA_VK_DEVICE_SELECT", None)     # DRI_PRIME puts the locked card first; adapter 0 is then that card
    cmd = [C.UE_EDITOR, C.UE_PROJECT, C.UE_STAGE_MAP, "-RenderOffscreen", "-graphicsadapter=0", "-vulkan", "-sm6",
           "-windowed", "-resx=1152", "-resy=640", "-unattended", "-nosplash", "-nosound", "-stdout",
           # never autosave the stage level the stills dirty (and no autosave files from a resident editor)
           "-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:bAutoSaveEnable=False"]
    return [str(c) for c in cmd], env


def ue_render_job(spec, imp_job, adir, args, mannequin):
    """The editor job for one asset: import + validation stills (ue/ue_render.py)."""
    return {"map": C.UE_STAGE_MAP, "import": imp_job, "id": spec["id"],
            "out": str(adir / "ue"), "res": [1152, 640], "warm_s": args.warm_s, "settle_s": args.settle_s,
            "warm_frames": args.warm_frames, "settle_frames": args.settle_frames,
            "ref_box_material": C.UE_STAGE_MAP.rsplit("/", 1)[0] + "/M_StageRef", "mannequin": mannequin,
            "cameras": {"preset": spec.get("camera_preset", spec["class"]), "custom": spec.get("cameras", []),
                        "close_target": spec.get("close_target"), "close_width": spec.get("close_width", 0.45)},
            "result": str(adir / "logs" / "ue_render.json")}


def sched_env():
    """The settings a client and the scheduler must share (else the client runs locally)."""
    return {k: v for k, v in sorted(os.environ.items())
            if (k.startswith("ASSETS_") or k == "BLENDER") and k not in ("ASSETS_SCHED", "ASSETS_UE_JOB")}


def guard_main(pid, bus):
    """Detached watchdog for a long-lived editor: stop it (by PID) when MemAvailable runs low; tidy up after."""
    while Path(f"/proc/{pid}").exists():
        if mem_avail_gb() < C.MIN_FREE_GB:
            (server_dir() / "guard.txt").write_text(f"MemAvailable {mem_avail_gb():.1f} GiB at {time.ctime()}; "
                                                    f"editor {pid} stopped\n")
            os.kill(pid, 15)
        time.sleep(2)
    sp = server_dir() / "state.json"
    if sp.exists() and json.loads(sp.read_text()).get("pid") == pid:
        sp.unlink()
    (server_dir() / "queue" / "quit").unlink(missing_ok=True)


def render_step(spec, imp_job, adir, args, mannequin):
    """Import + stills in one editor. Default: hand the job to the warm long-lived editor (start one if none; it
    holds its card's lock and quits after ASSETS_UE_IDLE_S without work). --fresh-editor: one editor per asset."""
    out = adir / "ue"
    out.mkdir(exist_ok=True)
    job = ue_render_job(spec, imp_job, adir, args, mannequin)
    Path(job["result"]).unlink(missing_ok=True)
    while mem_avail_gb() < C.MIN_FREE_GB + 10:
        log(f"waiting for memory: MemAvailable {mem_avail_gb():.1f} GiB")
        time.sleep(30)
    t0 = time.time()
    # The project lock is held only for the short hand-over / editor start, never while waiting for a card: a run
    # that waits for a card polls without it, and hands its job to an editor someone else started meanwhile.
    fd, bus, waited = None, None, 0.0
    while True:
        with ProjectLock():
            srv = None if args.fresh_editor else live_server()
            if srv:
                if fd is not None:
                    os.close(fd)
                    fd = None
                q = server_dir() / "queue" / f"{spec['id']}.{os.getpid()}.{int(t0)}.job.json"
                tmp = q.with_suffix(".tmp")
                tmp.write_text(json.dumps(job, indent=1))
                tmp.replace(q)
                log(f"job handed to the running editor (pid {srv['pid']}, card {srv['card']})")
                bus, p, logf, started = srv["card"], None, Path(srv["log"]), False
                break
            if fd is not None:
                install_shim()
                if ensure_rt_ini():
                    log(f"project renderer settings written (hardware ray tracing "
                        f"{'on' if C.UE_RAYTRACING else 'off'}); this start compiles shaders")
                log(f"card {bus} locked after {waited}s; starting an editor")
                job["script"] = str(HERE / "ue" / "ue_render.py")
                if not args.fresh_editor:
                    job["serve_dir"] = str(server_dir() / "queue")
                    job["serve_idle_s"] = C.UE_IDLE_S
                    (server_dir() / "queue" / "quit").unlink(missing_ok=True)
                jp = adir / "logs" / "ue_render_job.json"
                jp.write_text(json.dumps(job, indent=1))
                cmd, env = editor_cmd(bus)
                env["ASSETS_UE_JOB"] = str(jp)
                logf = C.LOGS / (f"assets_{spec['id']}_render.log" if args.fresh_editor
                                 else f"assets_ue_server_{time.strftime('%m%d_%H%M%S')}.log")
                f = open(logf, "w")
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, pass_fds=(fd,),
                                     start_new_session=not args.fresh_editor)
                f.close()
                os.close(fd)                               # the editor holds the card lock from here on
                started = True
                if not args.fresh_editor:
                    (server_dir() / "state.json").write_text(json.dumps({"pid": p.pid, "card": bus,
                                                                        "log": str(logf), "started": time.ctime()}))
                    subprocess.Popen([sys.executable, __file__, "--guard", str(p.pid), bus], start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
        bus, fd, _ = take_card(0, C.UE_CARDS)               # non-blocking, outside the project lock
        waited = round(time.time() - t0, 1)
        if fd is None:
            if waited > C.LOCK_WAIT_S:
                return {"ok": False, "error": f"no card free after {waited}s; CPU-only parts done"}
            time.sleep(3)
    t0_utc = datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).timestamp()
    util, vram = [], {b: [] for b in C.GPU_CARDS}
    rp = Path(job["result"])
    deadline = t0 + args.ue_timeout
    rc = None
    while not rp.exists():
        util.append(gpu_load(bus))
        for b in C.GPU_CARDS:
            vram[b].append(vram_used(b))
        if p is not None and args.fresh_editor:
            if p.poll() is not None:
                break
            if mem_avail_gb() < C.MIN_FREE_GB:
                p.terminate()
        elif not Path(f"/proc/{p.pid if p else srv['pid']}").exists():
            break                                          # the editor died: no result will come
        if time.time() > deadline:
            if p is not None:
                p.terminate()
            break
        time.sleep(0.25)
    if p is not None and args.fresh_editor:
        try:
            rc = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
    dt = round(time.time() - t0, 1)
    time.sleep(0.05)
    res = json.loads(rp.read_text()) if rp.exists() else {"ok": False}
    text = logf.read_text(errors="replace") if logf.exists() else ""
    gpu = re.findall(r'Metadata set : gpu="([^"]+)"', text)
    rt = re.findall(r'Metadata set : raytracing="(\d)"', text)
    res.update({"rc": rc, "time_s": dt, "card": bus, "lock_wait_s": waited, "log": str(logf),
                "editor": "fresh" if args.fresh_editor else ("started" if started else "warm"),
                "gpu_name": gpu[-1] if gpu else None, "raytracing_metadata": rt[-1] if rt else None,
                "vram_peak_mib": {b: max([v for v in vram[b] if v is not None], default=None) for b in vram},
                "gpu_load": summarize_load(util),
                "phases_s": ue_phases(text, t0_utc) if started else None})
    if not res.get("ok") and not res.get("error"):
        res["error"] = f"no result from the editor; see {logf}"
    log(f"render {dt}s ok={res.get('ok')} card={bus} editor={res['editor']}")
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
                                                      "time_s", "phases_s", "gpu_load", "error")},
        "blender_run": {k: blend.get(k) for k in ("device", "wall_s", "startup_s", "times_s", "gpu_load")},
        "times_s": times,
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (adir / "asset.json").write_text(json.dumps(entry, indent=1))
    mp = C.LIBRARY / "manifest.json"
    lf = os.open(C.LIBRARY / ".manifest.lock", os.O_RDWR | os.O_CREAT, 0o666)
    while True:                                   # read-modify-write under a lock (several runs finish at once)
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(0.05)
    try:
        man = json.loads(mp.read_text()) if mp.exists() else {"assets": {}}
        man["assets"][spec["id"]] = {"class": spec["class"], "tier": spec["tier"], "dir": rel(adir, C.LIBRARY),
                                     "ue_path": entry["ue"]["path"], "checks_passed": entry["checks_passed"],
                                     "tags": entry["tags"], "description": spec["description"]}
        tmp = mp.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(man, indent=1))
        tmp.replace(mp)
    finally:
        os.close(lf)
    return entry


def rel(p, base):
    if not p:
        return p
    try:
        return str(Path(p).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(p)


def build_parser():
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
    ap.add_argument("--warm-frames", type=int, default=90, help="frames after set-up before the first still")
    ap.add_argument("--settle-frames", type=int, default=30, help="frames after each camera move")
    ap.add_argument("--warm-s", type=float, default=0, help="minimum seconds on top of --warm-frames")
    ap.add_argument("--settle-s", type=float, default=0)
    ap.add_argument("--fresh-editor", action="store_true",
                    help="start and quit an editor for this asset instead of using the long-lived one")
    ap.add_argument("--preview-px", type=int, default=320, help="preview sheet tile size (px)")
    ap.add_argument("--preview-samples", type=int, default=None,
                    help="preview sheet Cycles samples (default 32; 12 with --denoise)")
    ap.add_argument("--denoise", action="store_true",
                    help="denoise the preview / sheet (OIDN runs on the CPU here: seconds per tile); default off")
    ap.add_argument("--sheet-samples", type=int, default=None,
                    help="final contact sheet Cycles samples (default 16 when UE stills follow, else 32)")
    ap.add_argument("--sheet-px", type=int, default=None,
                    help="final contact sheet tile size (default 320 when UE stills follow, else 512)")
    ap.add_argument("--sheet", action="store_true",
                    help="final pass with the 320 px, 16 spp contact sheet (off by default when UE stills follow)")
    ap.add_argument("--full-sheet", action="store_true", help="the 512 px, 32 spp contact sheet")
    ap.add_argument("--no-sheet", action="store_true", help="final pass without the Blender contact sheet")
    ap.add_argument("--local", action="store_true",
                    help="run in this process (debugging) instead of through the scheduler")
    ap.add_argument("--paint", action="store_true",
                    help="texture-painting stage (paint.py): image-model detail projected into a second map set")
    ap.add_argument("--hiprt", type=int, choices=(0, 1), default=C.HIPRT,
                    help="Cycles HIP RT (hardware ray tracing; default ASSETS_HIPRT=1)")
    ap.add_argument("--ue-timeout", type=float, default=2400)
    return ap


def main():
    a1 = sys.argv[1:]
    if len(a1) == 3 and a1[0] == "--guard":
        guard_main(int(a1[1]), a1[2])
        return 0
    if a1 and a1[0] in ("--status", "--stop-editor", "--release-card", "--release-cards", "--reclaim-cards",
                        "--ue-job"):
        import assetd as S
        if a1[0] == "--ue-job":
            if len(a1) != 2:
                raise SystemExit("usage: run_asset.py --ue-job JOB.json   (see assetd.client_ue_job)")
            return S.client_ue_job(a1[1])
        if a1[0] == "--status":
            return S.print_status()
        if a1[0] == "--release-cards":
            return S.release_cards()
        if a1[0] == "--reclaim-cards":
            return S.reclaim_cards()
        if a1[0] == "--release-card":
            if len(a1) != 2:
                raise SystemExit("usage: run_asset.py --release-card BUS   (the Blender card, e.g. "
                                 f"{C.BLENDER_CARD})")
            return S.release_card(a1[1])
        r = S.client_cmd("stop_editors")
        if r is not None:
            log(f"asked the scheduler's editor(s) to quit ({r.get('editors')} alive)")
        st = live_server()
        if st:
            (server_dir() / "queue" / "quit").touch()
            log(f"asked editor {st['pid']} to quit")
        return 0
    args = build_parser().parse_args()
    if not (args.local or args.fresh_editor) and os.environ.get("ASSETS_SCHED", "1") == "1":
        spec = json.loads(Path(args.spec).read_text())
        validate_spec(spec)                               # fail fast, before the scheduler
        import assetd as S
        rc = S.client_submit(sys.argv[1:])
        if rc is not None:
            return rc
    return main_local(args)


def main_local(args):
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
    paint_rep = None
    if args.paint:
        import paint                                       # the texture-painting stage (paint.py)
        log("paint: guide views -> image model -> projection")
        paint_rep = paint.run(adir, spec)
        times["paint"] = paint_rep["times_s"]
    painted_glb = adir / "export" / "painted" / f"{spec['id']}.glb"   # same file name: UE's folder is named after it
    tset = spec.get("texture_set") or ("painted" if paint_rep and (spec.get("paint") or {}).get("use", True)
                                       else "procedural")
    if tset == "painted":
        if not painted_glb.exists():
            raise SystemExit("texture_set painted but no painted maps: run with --paint")
        if not paint_rep:
            log("texture_set painted: using the maps of an earlier --paint run (same model script assumed)")
        glb = str(painted_glb)
    log(f"texture set for UE: {tset}")
    dest = f"{C.UE_ASSET_ROOT}/{spec['class'].capitalize()}s/{spec['id']}"   # Interchange adds <file name>/ inside
    # Nanite does not draw translucent sections: an asset with a glass slot imports without it
    imp_job = {"file": glb, "dest": dest, "id": spec["id"], "nanite": not blend.get("glass_materials"),
               "result": str(adir / "logs" / "ue_import.json")}
    Path(imp_job["result"]).unlink(missing_ok=True)
    render = None
    if args.skip_render:
        # import only: the no-GPU commandlet (a second editor start, ~7 s, so the render path imports in-process)
        log("UE import (commandlet)")
        with ProjectLock():
            if live_server():
                raise SystemExit("a long-lived editor owns the project; run without --skip-render to import through it")
            rc, times["ue_import"], imp = ue_cmdlet(HERE / "ue" / "ue_import.py", imp_job,
                                                    adir / "logs" / "ue_import.log")
    else:
        with ProjectLock():
            if args.rebuild_stage and live_server():
                raise SystemExit("stop the long-lived editor first (run_asset.py --stop-editor), then rebuild")
            st = ensure_stage(adir / "logs", args.rebuild_stage) if not live_server() else {"built": False}
            mannequin = ensure_mannequin()
        if st.get("built"):
            times["ue_stage"] = st["time_s"]
        log("UE import + render (one editor)")
        render = render_step(spec, imp_job, adir, args, mannequin)
        times["ue_render"] = render.get("time_s")
        rp = Path(imp_job["result"])
        imp = json.loads(rp.read_text()) if rp.exists() else {"ok": False, "error": render.get("error")}
        times["ue_import_in_editor"] = imp.get("time_s")
    log(f"import ok={imp.get('ok')} -> {imp.get('static_mesh')} problems={imp.get('problems')}")
    if not imp.get("ok"):
        raise SystemExit(f"UE import failed: {imp.get('error')}")
    times["total"] = round(time.time() - t0, 1)
    entry = write_manifest(spec, adir, blend, imp, render, args.model, times)
    entry["texture_set"] = tset
    pj = adir / "paint" / "paint.json"
    if pj.exists():
        pr = json.loads(pj.read_text())
        entry["textures_painted"] = (pr.get("files") or {}).get("textures")
        entry["export_painted"] = (pr.get("files") or {}).get("glb")
    (adir / "asset.json").write_text(json.dumps(entry, indent=1))
    log(f"done in {times['total']}s; checks_passed={entry['checks_passed']}; render ok={(render or {}).get('ok')}")
    log(f"asset folder: {adir}")
    return 0 if (render is None or render.get("ok")) else 3


if __name__ == "__main__":
    sys.exit(main())
