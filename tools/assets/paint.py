"""Texture painting: the local image model paints surface detail onto views of a baked asset; the paintings are
projected back onto the mesh and baked into a second texture set (the procedural maps are never touched).

    python tools/assets/paint.py ASSET_DIR                      (after a final run_asset.py pass of that asset)
    python tools/assets/run_asset.py SPEC.json MODEL.py --paint (the same, as a step of the chain; off by default)

Steps (each timed in <asset>/paint/paint.json):
1. guides  (Blender, bl_paint.py): N views around and above the asset (+ a nearer ring on the spec's close region),
   each a diffuse render of the baked BaseColor under even light on flat grey, plus mask, depth and the linear
   Image / DiffCol / Position passes. Views are sized so their covered pixels exceed the map's texels.
2. paint   (Qwen-Image-2.1, tools/qwen_image.py --serve, one session per device, model kept loaded): each view
   repainted by img2img from the guide (strength 0.65, the object mask as the per-pixel strength mask so the
   background and outline stay put; optional depth/canny ControlNet), with the spec's material prompt and a
   flat-light instruction.
   Method "edit" uses Qwen-Image-Edit-2509 (tools/assets/qwen_edit.py) on the guide instead.
3. project (Blender, bl_paint.py): painted / guide shading -> albedo per view, blended per texel by facing,
   pixel density and edge distance (low and high frequencies separately), gaps filled; detail normal and
   roughness from the painted colour; writes <asset>/textures/painted/T_<id>_*.png and export/painted/<id>.glb.

Spec block (all optional): "paint": {"prompt": material words, "views": {...}, "close": {"box": [...]},
"method": "img2img" | "edit", "strength": 0.5, "control_scale": 0.7, "seed": 7, "params": {projection},
"use": true}. "texture_set": "painted" | "procedural" picks what goes to UE (default: painted when painted
maps exist and paint.use is not false). Scratch (EXR passes, guide pictures) goes under ASSETS_PAINT_SCRATCH
(default $ASSETS_UE_HOME/lab/paint/<id>).

Devices: Blender on a free XTX (its lock and bake slot) or the idle iGPU (its lock), waiting until one is free:
never the CPU (-t 4 for Blender's own CPU-bound parts); the image model on whichever of the two XTX locks or the
iGPU lock comes free first (polled with flock -n), on a second device too when it is free at once.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

QWEN_PY = Path(os.path.expanduser(os.environ.get("ASSETS_QWEN_PY", "~/tts-ab/.venv-qwenimage/bin/python")))
# the iGPU (gfx1151) needs a torch built for it: the drawer venv's torch has no kernels for it
QWEN_PY_IGPU = Path(os.path.expanduser(os.environ.get("ASSETS_QWEN_PY_IGPU", "~/tts-ab/.venv-charlora-igpu/bin/python")))
QWEN_IMAGE = Path(os.environ.get("ASSETS_QWEN_IMAGE", HERE.parent / "qwen_image.py"))
EDIT_PY = Path(os.path.expanduser(os.environ.get("ASSETS_EDIT_PY", "~/tts-ab/.venv-hyworld/bin/python")))
HIP_INDEX = Path(os.path.expanduser(os.environ.get("ASSETS_HIP_INDEX", "~/bin/hip-index")))
PAINT_CARDS = [c.strip() for c in os.environ.get("ASSETS_PAINT_DEVICES", "05,65,igpu").split(",") if c.strip()]
PAINT_WAIT_S = float(os.environ.get("ASSETS_PAINT_WAIT_S", "1e9"))   # wait for a device (default: as long as it takes)
SCRATCH = Path(os.path.expanduser(os.environ.get("ASSETS_PAINT_SCRATCH", str(C.UE_HOME / "lab" / "paint"))))

DEFAULTS = {
    "views": {"ring": 8, "ring_el": 12, "upper": 4, "upper_el": 50, "top": True, "area_px": 1536 * 1024},
    # method test 11-10 (anvil, 3 views): no ControlNet at 0.65 gave the most surface character with the outline
    # held (silhouette IoU 0.989; the mask keeps the background); depth/canny control flattened the surface into
    # plain CG paint, and at 0.85 slid edges; the edit model invented holes and rivets (IoU 0.937) at ~80 s a view
    "method": "img2img", "strength": 0.65, "control": None, "control_scale": 0.7, "control_end": 1.0,
    "seed": 7, "samples": 48,
    "params": {"anchor": 0.0, "normal_strength": 1.5, "rough_strength": 0.25, "gain_match": True},
}
FLAT = ("Even, flat, shadowless studio lighting from every side; no cast shadows, no highlights, no reflections "
        "of the room. The same object, the same shape, outline, size and position as the input, on the same plain "
        "grey background.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] paint: {msg}", flush=True)


def merged(spec):
    p = json.loads(json.dumps(DEFAULTS))
    user = spec.get("paint") or {}
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    if user.get("close") and "close" not in user.get("views", {}):
        p["views"]["close"] = user["close"]
    return p


def prompt_for(spec, p):
    what = p.get("prompt") or spec["description"]
    return (f"A photograph of {spec['description'].rstrip('.')}. Surface: {what.rstrip('.')}. Rich, sharp, fine "
            f"surface detail across the whole object. {FLAT}")


# ---------------------------------------------------------------- locks
HOLD = Path(os.environ.get("ASSETS_GPU_HOLD", str(C.UE_HOME / "GPU_HOLD")))


def sched_active():
    """The asset scheduler (assetd.py) owns the cards while it runs (tenant files /tmp/gpu-<bus>.tenant); it reads
    any outside holder of a card for ~20 s as a release and stops its editor, so paint.py stays off the cards then."""
    dl = Path(os.environ.get("ASSETS_SCHED_DIR", str(C.UE_HOME / "sched"))) / "daemon.lock"
    if dl.exists():                          # the scheduler holds this flock while it lives
        fd = os.open(dl, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            return True
        finally:
            os.close(fd)
    for t in Path("/tmp").glob("gpu-*.tenant"):
        try:
            pid = json.loads(t.read_text()).get("sched_pid")
            if pid and Path(f"/proc/{pid}").exists():
                return True
        except (OSError, ValueError):
            pass
    return False


def card_hold():
    """No new card job: the machine's hold file exists (the iGPU stays usable), or the scheduler owns the cards."""
    return HOLD.exists() or sched_active()


def try_lock(dev):
    fd = os.open(f"/tmp/gpu-{dev}.lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        return None


def take_devices(order, want, wait_s, poll=10):
    """Poll the locks (flock -n) until one is free; then take up to `want` free at that moment."""
    t0 = time.time()
    while True:
        got = []
        for d in order:
            if d != "igpu" and card_hold():
                continue
            fd = try_lock(d)
            if fd is not None:
                got.append((d, fd))
                if len(got) >= want:
                    break
        if got:
            return got, round(time.time() - t0, 1)
        if time.time() - t0 > wait_s:
            return [], round(time.time() - t0, 1)
        time.sleep(poll)


def hip_env(dev):
    idx = subprocess.run([str(HIP_INDEX), "igpu" if dev == "igpu" else dev], capture_output=True, text=True,
                         check=True).stdout.strip()
    env = dict(os.environ, HIP_VISIBLE_DEVICES=idx, PYTHONUNBUFFERED="1", MIOPEN_FIND_MODE="FAST")
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)              # expandable segments corrupt results on this ROCm
    if dev == "igpu":
        env["HSA_CU_MASK"] = f"{idx}:0-38"                 # one CU left for the desktop
    return env


# ---------------------------------------------------------------- blender
def igpu_idle():
    try:
        busy = int(Path(f"/sys/bus/pci/devices/0000:{C.IGPU_BUS}:00.0/gpu_busy_percent").read_text())
    except (OSError, ValueError):
        return False
    return busy <= C.IGPU_MAX_BUSY


def editor_card():
    """The card of this chain's warm UE editor (it holds that card's lock but uses ~5-6 GB of the 24)."""
    sp = Path(C.UE_PROJECT).parent / "Saved" / "AssetServer" / "state.json"
    try:
        st = json.loads(sp.read_text())
        if "UnrealEditor" in Path(f"/proc/{st['pid']}/cmdline").read_text(errors="replace"):
            return st.get("card")
    except (OSError, ValueError, KeyError):
        pass
    return None


def vram_free_gb(bus):
    for d in Path("/sys/class/drm").glob("card[0-9]*"):
        if (d / "device").exists() and os.path.realpath(d / "device").endswith(f"0000:{bus}:00.0"):
            try:
                return (int((d / "device" / "mem_info_vram_total").read_text())
                        - int((d / "device" / "mem_info_vram_used").read_text())) / 2**30
            except (OSError, ValueError):
                return 0.0
    return 0.0


def gpu_for_blender():
    """(bus, fds, waited, how): the first free of the Blender cards (main lock + bake slot) or the idle iGPU."""
    t0 = time.time()
    said = False
    while True:
        for b in ([] if card_hold() else C.BLENDER_CARDS):
            slot = try_lock(f"{b}.bake")
            if slot is None:
                continue
            fd = try_lock(b)
            if fd is not None:
                return b, [fd, slot], round(time.time() - t0, 1), f"card {b}"
            if editor_card() == b and vram_free_gb(b) >= float(os.environ.get('ASSETS_BAKE_VRAM_GB', '8')):
                return b, [slot], round(time.time() - t0, 1), f"card {b} beside the warm editor"
            os.close(slot)
        if igpu_idle():
            fd = try_lock("igpu")
            if fd is not None:
                return C.IGPU_BUS, [fd], round(time.time() - t0, 1), "igpu"
        if not said:
            log("waiting for a GPU (cards and iGPU busy; never the CPU)")
            said = True
        time.sleep(3)


def blender(job, logf, cards=None):
    """Run bl_paint.py on a GPU, never the CPU (owner's rule): an XTX under its lock (and its bake slot), or the
    iGPU under /tmp/gpu-igpu.lock when it is idle; polled every 3 s until one is free, however long that takes."""
    bus, fds, waited, how = gpu_for_blender()
    job = dict(job, device="HIP", hip_bus=bus, hiprt=C.HIPRT)
    jp = Path(str(logf) + ".job.json")
    rp = Path(str(logf) + ".report.json")
    job["report"] = str(rp)
    jp.write_text(json.dumps(job, indent=1))
    rp.unlink(missing_ok=True)
    t = time.time()
    try:
        with open(logf, "w") as f:
            subprocess.run([str(C.BLENDER), "-b", "--factory-startup", "-t", "4", "--python",
                            str(HERE / "bl_paint.py"), "--", str(jp)], stdout=f, stderr=subprocess.STDOUT,
                           timeout=3600, pass_fds=tuple(fds))
    finally:
        for x in fds:
            os.close(x)
    rep = json.loads(rp.read_text()) if rp.exists() else {"status": "error", "error": f"no report; see {logf}"}
    rep["wall_s"] = round(time.time() - t, 1)
    rep["card"] = bus or "cpu"
    rep["device_choice"] = f"{how} after {waited}s"
    if rep.get("status") != "ok":
        raise SystemExit(f"bl_paint {job['mode']} failed: {rep.get('error')}")
    return rep


# ---------------------------------------------------------------- painting
class Server:
    """One qwen_image.py --serve process on one device; jobs in on stdin, one JSON line out per job."""

    def __init__(self, dev, fd, logf):
        self.dev, self.fd = dev, fd
        self.t0 = time.time()
        self.err = open(logf, "w")
        py = QWEN_PY_IGPU if dev == "igpu" else QWEN_PY
        self.p = subprocess.Popen([str(py), "-I", str(QWEN_IMAGE), "--serve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.err, text=True, env=hip_env(dev),
                                  pass_fds=(fd,))
        line = self.p.stdout.readline()
        self.load_s = round(time.time() - self.t0, 1)
        if not line:
            raise RuntimeError(f"drawer on {dev} did not start; see {logf}")

    def run(self, jobs, results):
        for j in jobs:
            self.p.stdin.write(json.dumps(j) + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
            r = json.loads(line) if line else {"id": j["id"], "error": "drawer died"}
            r["device"] = self.dev
            results.append(r)
            log(f"painted {j['id']} on {self.dev}: {r.get('seconds', r.get('error'))}")

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        self.err.close()
        os.close(self.fd)


def paint_views(views, out, spec, p, logdir, max_devices=2):
    """img2img + depth control for every view, spread over up to max_devices drawers."""
    jobs = []
    variants = p.get("variants") or [{}]     # method tests: [{"tag", ...overrides}] in one loaded session
    for var in variants:
        jobs += view_jobs(views, out.parent / f"views_{var['tag']}" if var.get("tag") else out, spec,
                          dict(p, **{k: v for k, v in var.items() if k != "tag"}), var.get("tag"))
    return draw_jobs(jobs, p, logdir, max_devices)


def view_jobs(views, out, spec, p, tag=None):
    prompt = prompt_for(spec, p)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for v in views:
        f = v["files"]
        path = str(out / f"{v['name']}.png")
        j = {"id": f"{tag}/{v['name']}" if tag else v["name"], "prompt": prompt, "seed": int(p["seed"]),
             "path": path, "size": v["size"],
             "init": {"image": f["guide"], "strength": float(p["strength"]), "mask": f["mask"],
                      "strength_bg": 0.0}}
        if p.get("controls"):              # several: [{"type": "depth" | "canny", "scale", "end"}] (canny of the guide)
            j["control"] = [{"image": f["depth"] if c["type"] == "depth" else f["guide"], "type": c["type"],
                             "scale": float(c.get("scale", 0.7)), "end": float(c.get("end", 1.0)),
                             **({"low": c["low"], "high": c["high"]} if "low" in c else {})} for c in p["controls"]]
        elif p.get("control"):
            j["control"] = {"image": f["depth"], "type": p["control"], "scale": float(p["control_scale"]),
                            "end": float(p.get("control_end", 1.0))}
        if p.get("refs"):
            j["refs"] = p["refs"]
        jobs.append(j)
    return jobs


def draw_jobs(jobs, p, logdir, max_devices):
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    times = {"jobs": len(jobs), "drawn": len(todo)}
    if not todo:
        return times
    got, waited = take_devices(PAINT_CARDS, min(max_devices, max(1, len(todo) // 4)), PAINT_WAIT_S)
    if not got:
        raise SystemExit(f"no device for the image model after {waited}s")
    times["lock_wait_s"] = waited
    log(f"drawing {len(todo)} views on {[d for d, _ in got]}")
    t = time.time()
    servers, threads, results = [], [], []
    try:
        starters = []
        for d, fd in got:
            def start(d=d, fd=fd):
                servers.append(Server(d, fd, logdir / f"qwen_{d}.log"))
            th = threading.Thread(target=start)
            th.start()
            starters.append(th)
        for th in starters:
            th.join()
        if not servers:
            raise SystemExit("no drawer started")
        times["load_s"] = {s.dev: s.load_s for s in servers}
        # the faster card gets more: XTX ~3x the iGPU
        share = [3.0 if s.dev != "igpu" else 1.0 for s in servers]
        queues = [[] for _ in servers]
        acc = [0.0] * len(servers)
        for j in todo:
            k = min(range(len(servers)), key=lambda i: (acc[i] + 1) / share[i])
            queues[k].append(j)
            acc[k] += 1
        for s, q in zip(servers, queues):
            th = threading.Thread(target=s.run, args=(q, results))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
    finally:
        for s in servers:
            s.close()
        for d, fd in got:
            if all(s.dev != d for s in servers):
                os.close(fd)
    times["paint_s"] = round(time.time() - t, 1)
    times["per_view_s"] = {r["id"]: r.get("seconds") for r in results}
    bad = [r for r in results if r.get("error")]
    if bad:
        raise SystemExit(f"painting failed: {bad[:3]}")
    return times


def edit_views(views, out, spec, p, logdir):
    """Qwen-Image-Edit-2509 on each guide (one card: encoder then transformer; a jobs file)."""
    words = p.get("prompt") or spec["description"]
    prompt = (f"Repaint the surface of this object as {words.rstrip('.')}, with rich, sharp, fine surface detail. "
              f"Keep the object's exact shape, outline, size and position and the plain grey background. {FLAT}")
    jobs = [{"id": v["name"], "prompt": prompt, "images": [v["files"]["guide"]], "seed": int(p["seed"]),
             "path": str(out / f"{v['name']}.png"), "size": v["size"]} for v in views]
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    times = {"jobs": len(jobs), "drawn": len(todo)}
    if not todo:
        return times
    got, waited = take_devices([d for d in PAINT_CARDS if d != "igpu"], 1, PAINT_WAIT_S)
    if not got:
        raise SystemExit(f"no card for the edit model after {waited}s")
    (d, fd), = got
    jp = logdir / "edit_jobs.json"
    jp.write_text(json.dumps(todo, indent=1))
    t = time.time()
    try:
        with open(logdir / f"edit_{d}.log", "w") as f:
            r = subprocess.run([str(EDIT_PY), "-I", str(HERE / "qwen_edit.py"), str(jp)], stdout=subprocess.PIPE,
                               stderr=f, text=True, env=hip_env(d), pass_fds=(fd,), timeout=7200)
    finally:
        os.close(fd)
    lines = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
    times.update(device=d, lock_wait_s=waited, paint_s=round(time.time() - t, 1),
                 per_view_s={x["id"]: x.get("seconds") for x in lines if "id" in x})
    bad = [x for x in lines if x.get("error")]
    if bad:
        raise SystemExit(f"editing failed: {bad[:3]}")
    return times


def contact(views, out_png, painted_dir):
    """guide | painted pairs, 4 pairs per row, 384 px tall (PIL from the drawer's venv)."""
    code = r"""
import json, sys
from PIL import Image
views, out, pd = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
H = 320
tiles = []
for v in views:
    g = Image.open(v["files"]["guide"]).convert("RGB")
    p = Image.open(f"{pd}/{v['name']}.png").convert("RGB")
    w = int(H * g.width / g.height)
    pair = Image.new("RGB", (2 * w + 6, H), (20, 20, 20))
    pair.paste(g.resize((w, H)), (0, 0)); pair.paste(p.resize((w, H)), (w + 6, 0))
    tiles.append(pair)
rows = [tiles[i:i + 3] for i in range(0, len(tiles), 3)]
W = max(sum(t.width + 10 for t in r) for r in rows)
sheet = Image.new("RGB", (W, len(rows) * (H + 10)), (40, 40, 40))
for ri, r in enumerate(rows):
    x = 0
    for t in r:
        sheet.paste(t, (x, ri * (H + 10))); x += t.width + 10
sheet.save(out)
"""
    subprocess.run([str(QWEN_PY), "-I", "-c", code, json.dumps(views), str(out_png), str(painted_dir)], check=True)


# ---------------------------------------------------------------- the stage
def run(adir, spec, stages=("guides", "paint", "project"), overrides=None):
    adir = Path(adir)
    aid = spec["id"]
    p = merged(spec)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    tag = p.get("tag", "")
    scratch = SCRATCH / aid / (tag or "run")
    gdir = scratch / "views"
    pdir = adir / "paint" / (f"views_{tag}" if tag else "views")
    logdir = adir / "logs"
    for d in (gdir, pdir, logdir):
        d.mkdir(parents=True, exist_ok=True)
    blend = adir / "source" / f"{aid}.blend"
    rep = {"id": aid, "params": p, "times_s": {}}
    tex_px = int(spec.get("texture_px", C.TIERS[spec["tier"]]["tex"]))
    t0 = time.time()
    if "guides" in stages:
        r = blender({"mode": "guides", "blend": str(blend), "id": aid, "out": str(gdir), "views": p["views"],
                     "samples": p["samples"]}, logdir / "paint_guides.log")
        rep["times_s"]["guides"] = r["wall_s"]
        rep["guides"] = {k: r.get(k) for k in ("views", "pixels", "exposure", "albedo_median", "card", "times_s")}
        if r.get("pixels", 0) < tex_px * tex_px * 0.6:
            log(f"warning: views cover {r.get('pixels')} px for a {tex_px}^2 map")
        log(f"guides: {r['views']} views, {r['pixels'] / 1e6:.1f} Mpx covered, {r['wall_s']} s on {r['card']}")
    gdir_src = Path(p["guides_from"]) if p.get("guides_from") else gdir     # reuse another run's guides
    V = json.loads((gdir_src / "views.json").read_text())
    if p.get("only"):                                       # a method test on a few views
        V["views"] = [v for v in V["views"] if v["name"] in p["only"]]
    if "paint" in stages:
        if p["method"] == "edit":
            r = edit_views(V["views"], pdir, spec, p, logdir)
        else:
            r = paint_views(V["views"], pdir, spec, p, logdir, int(p.get("devices", 2)))
        rep["times_s"]["paint"] = r.get("paint_s", 0)
        rep["paint"] = r
        if not p.get("variants"):
            contact(V["views"], adir / "paint" / f"{aid}_paint_views{'_' + tag if tag else ''}.jpg", pdir)
    if "project" in stages:
        tdir = adir / "textures" / (f"painted_{tag}" if tag else "painted")
        glb = adir / "export" / (f"painted_{tag}" if tag else "painted") / f"{aid}.glb"   # UE names its folder after the file
        glb.parent.mkdir(parents=True, exist_ok=True)
        r = blender({"mode": "project", "blend": str(blend), "id": aid, "views_json": str(gdir_src / "views.json"),
                     "painted": {v["name"]: str(pdir / f"{v['name']}.png") for v in V["views"]},
                     "res": tex_px, "out_dir": str(tdir), "glb": str(glb), "params": p["params"]},
                    logdir / "paint_project.log")
        rep["times_s"]["project"] = r["wall_s"]
        rep["project"] = {k: r.get(k) for k in ("texels", "seen_fraction", "seams", "bg_painted_fraction", "views_per_texel", "global_gain",
                                                "view_gains", "times_s", "card", "warnings")}
        rep["files"] = {"textures": {k: str(Path(v).relative_to(adir)) for k, v in r["files"].items()
                                     if k != "glb"}, "glb": str(glb.relative_to(adir))}
        log(f"project: seen {r.get('seen_fraction')}, {r['wall_s']} s on {r['card']}")
    if "stills" in stages:
        # before/after in Blender (Cycles, the real PBR maps, the harness's light): four sides and close-ups
        cb = (p["views"].get("close") or {}).get("box")
        ct = [(cb[0] + cb[3]) / 2, (cb[1] + cb[4]) / 2, (cb[2] + cb[5]) / 2] if cb else [0.5, 0.3, 0.8]
        shots = p.get("stills") or (
            [{"name": n, "az": az, "el": 10, "span": 1.05, "target": [0.5, 0.5, 0.5], "px": [1024, 1024]}
             for n, az in (("front", 0), ("right", 90), ("back", 180), ("left", 270))]
            + [{"name": "close", "az": 25, "el": 25, "span": 0.18, "target": ct, "px": [1280, 800], "lens": 85},
               {"name": "macro", "az": 25, "el": 25, "span": 0.07, "target": ct, "px": [1280, 800], "lens": 85}])
        t1 = time.time()
        for maps, sub in (("procedural", "procedural"),
                          (str(adir / "textures" / (f"painted_{tag}" if tag else "painted")), "painted")):
            if p.get("stills_only") and sub not in p["stills_only"]:
                continue
            out = adir / "paint" / f"stills_{sub}{'_' + tag if tag else ''}"
            blender({"mode": "stills", "blend": str(blend), "id": aid, "maps": maps, "out": str(out),
                     "shots": shots, "samples": int(p.get("stills_samples", 64))}, logdir / f"paint_stills_{sub}.log")
        rep["times_s"]["stills"] = round(time.time() - t1, 1)
    rep["times_s"]["total"] = round(time.time() - t0, 1)
    rp = adir / "paint" / f"paint{'_' + tag if tag else ''}.json"
    old = json.loads(rp.read_text()) if rp.exists() else {}
    times = dict(old.get("times_s", {}), **rep["times_s"])      # a run of some stages keeps the others' record
    old.update(rep)
    old["times_s"] = times
    rp.write_text(json.dumps(old, indent=1))
    return old


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("asset_dir")
    ap.add_argument("--stages", default="guides,paint,project")
    ap.add_argument("--set", action="append", default=[], help="override: key=json (e.g. strength=0.4, "
                    "params={\"anchor\":0.3}, tag=\"s04\")")
    a = ap.parse_args()
    adir = Path(a.asset_dir)
    spec = json.loads((adir / "source" / "spec.json").read_text())
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        ov[k] = json.loads(v)
    rep = run(adir, spec, tuple(a.stages.split(",")), ov)
    print(json.dumps(rep["times_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
