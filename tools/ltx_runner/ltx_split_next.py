"""LTX-2.5 distilled on two 7900 XTX cards (bus 05 = cuda:0, bus 65 = cuda:1).

  encode JOBS.json   Gemma4-12B bf16 split over both cards -> cached embeddings (emb/<name>.pt)
  gen    JOBS.json   DiT fp8-cast, blocks split over both cards; conv VAE; mp4 per job

Weights are streamed tensor by tensor from safetensors straight to the target card;
page cache is dropped as we go; the process exits if MemAvailable < 6 GiB.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import re
import sys
import threading
import types
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor-ltx"))  # survives reboots


def raise_nofile():
    """10-28: every cross-card hop sends fresh shared-memory tensors, each an open file until both sides free it;
    a long talk run held ~14 per piece and hit the 1024 soft limit after 71 pieces (EMFILE, then a dead thread and
    a 3 h hang). Lift the soft limit to the hard one; spawned workers inherit it."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


raise_nofile()

import safetensors  # noqa: E402
import torch  # noqa: E402

M = Path.home() / "models" / "ltx-2.5"
# 10-28: the weights every hold loads are on the internal NVMe (~4 GB/s; the USB root disk reads 0.7 GB/s, and the
# 75 s front/back load was mostly that read + the bf16->fp8 cast). The dev DiT there is pre-cast: the loader's own
# fp8 rule applied once (same names, metadata), so its load-time cast is a no-op. Anything missing: the USB original
MF_ = Path.home() / "models-fast" / "ltx-2.5"


def _w(rel, fast_rel=None):
    f = MF_ / (fast_rel or rel)
    return str(f if f.exists() else M / rel)


DIT = _w("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors")
TE = _w("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
VAE = _w("vae/ltx-2.5-video-vae-conv-bf16.safetensors")
AVAE = _w("vae/ltx-2.5-audio-vae-bf16.safetensors")
UPS = _w("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")
DUR = _w("model_patches/ltx-2.5-duration-head-bf16.safetensors")
DIT_DEV = _w("diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
             "diffusion_models/ltx-2.5-22b-dev-transformer-fp8cast.safetensors")
DLORA = _w("loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors")
D0, D1 = torch.device("cuda:0"), torch.device("cuda:1")
T0 = time.time()
PP_BACK_DEV = "1"
MAX_BATCH = 4                # guidance passes per DiT call (A2Vid CFG = 2, TI2Vid CFG + STG = 3)
TALK_CFG_CUT = 0.6           # talk (A2Vid) stage 1: guidance passes on the first 60% of steps only (see install_cfg_cut)
ACTION_CFG_CUT = 0.6         # action (TI2Vid) too (10-31 A/B, 6 clips, user: "it's the same"; card B -13%, A -7%)
COMPILE_MIN_JOBS = int(os.environ.get("LTX_COMPILE_MIN_JOBS", 20))       # --compile pays its ~80 s warm-up back after ~12-20 clips; smaller holds stay eager


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def mem_avail_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return 99.0


def ram_guard(min_gib: float = 6.0) -> None:
    def run():
        while True:
            a = mem_avail_gib()
            if a < min_gib:
                print(f"RAM GUARD: MemAvailable {a:.1f} GiB < {min_gib} -> abort", flush=True)
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=run, daemon=True).start()


def vram() -> str:
    out = []
    for d in range(torch.cuda.device_count()):
        free, tot = torch.cuda.mem_get_info(d)
        out.append(f"cuda:{d} alloc {torch.cuda.memory_allocated(d) / 2**30:.1f} "
                   f"peak {torch.cuda.max_memory_allocated(d) / 2**30:.1f} used {(tot - free) / 2**30:.1f} GiB")
    return " | ".join(out)


# ---------------------------------------------------------------- split loading
def split_load(paths, sd_ops, devfn):
    """{renamed_key: tensor on devfn(key)}; one tensor in RAM at a time, page cache dropped."""
    from ltx_core.loader.primitives import StateDict

    sd, size, dtypes = {}, 0, set()
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            with safetensors.safe_open(p, framework="pt", device="cpu") as f:
                for i, name in enumerate(f.keys()):
                    key = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if key is None or devfn(key) is None:
                        continue
                    v = f.get_tensor(name)
                    if v.dtype == torch.uint8 and v.ndim == 1:  # asset blob (tokenizer json etc.)
                        pairs = ((key, v),) if sd_ops is None else sd_ops.apply_to_key_value(key, v)
                    else:
                        v = v.to(devfn(key))
                        pairs = ((key, v),) if sd_ops is None else sd_ops.apply_to_key_value(key, v)
                    for k, t in pairs:
                        sd[k] = t
                        size += t.nbytes
                        dtypes.add(t.dtype)
                    del v
                    if i % 32 == 0:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return StateDict(sd=sd, device=D0, size=size, dtype=dtypes)


def place(model: torch.nn.Module, devfn) -> None:
    """Move every param/buffer not already on devfn(name) (catches buffers created on CPU)."""
    for mname, mod in model.named_modules():
        pre = f"{mname}." if mname else ""
        for n, t in list(mod._parameters.items()):
            if devfn(pre + n) is None:
                continue
            if t is not None and not t.is_meta and t.device != devfn(pre + n):
                mod._parameters[n] = torch.nn.Parameter(t.to(devfn(pre + n)), requires_grad=False)
        for n, t in list(mod._buffers.items()):
            if devfn(pre + n) is None:
                continue
            if t is not None and not t.is_meta and t.device != devfn(pre + n):
                mod._buffers[n] = t.to(devfn(pre + n))
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.is_meta and devfn(n) is not None]
    if meta:
        log("WARNING still-meta tensors:", meta[:10], len(meta))


def build_split(builder, devfn):
    from ltx_core.loader.helpers import as_path_list, assign_state_dict, create_meta_model

    metadata = builder.model_metadata()
    model = create_meta_model(builder._model_class_configurator, metadata, builder.module_ops)
    sd = split_load(as_path_list(builder.model_path), builder.model_sd_ops, devfn)
    assign_state_dict(model, sd.sd)
    del sd
    gc.collect()
    place(model, devfn)
    return model


def move(obj, dev):
    """Recursively move tensors inside dataclasses / tuples / lists / dicts."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj if obj.device == dev else obj.to(dev, non_blocking=False)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        ch = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            nv = move(v, dev)
            if nv is not v:
                ch[f.name] = nv
        return dataclasses.replace(obj, **ch) if ch else obj
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(move(v, dev) for v in obj))
    if isinstance(obj, (tuple, list)):
        return type(obj)(move(v, dev) for v in obj)
    if isinstance(obj, dict):
        return {k: move(v, dev) for k, v in obj.items()}
    return obj


def layer_devfn(pattern: str, split: int, part: str = "both"):
    """part='both': blocks >= split on cuda:1, rest cuda:0. 'front': only blocks < split + non-block params
    (cuda:0), back blocks skipped. 'back': only blocks >= split, on THIS process's cuda:0."""
    rx = re.compile(pattern)

    def devfn(key: str):
        m = rx.search(key)
        back = bool(m and int(m.group(1)) >= split)
        if part == "front":
            return None if back else D0
        if part == "back":
            return D0 if back else None
        return D1 if back else D0

    return devfn


# ---------------------------------------------------------------- process-per-card pipelining
HASHLOG = {"on": False}


_TM = threading.local()


def tm_reset():
    _TM.d = {"a_dit_s": 0.0, "a_copy_s": 0.0, "wait_b_s": 0.0, "b_dit_s": 0.0, "b_copy_s": 0.0}


def tm_add(k, v):
    d = getattr(_TM, "d", None)
    if d is not None:
        d[k] = d.get(k, 0.0) + v


def tm_snapshot():
    return {k: round(v, 2) for k, v in getattr(_TM, "d", {}).items()}


def gen_with_seam_retries(p, kw: dict, j: dict, dev, out_dir: Path):
    """Run the pipeline for job j; a chained chunk is re-rolled (seed + 1000) up to 3 tries while its seam check
    fails. All tries failing keeps the smoothest try and flags it (j["seam_flag"]) instead of raising
    (port of the remake agent's fix for an UnboundLocalError on `frames`). -> (frames, j)"""
    best = None
    for attempt in range(max(1, j.get("seam_tries", 3)) if j.get("chain_from") else 1):   # shot flow: 0 = no seam check
        with torch.inference_mode():
            res = p(**kw)
            vids = [v for v in res.video]
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)
        frames = torch.cat([v if v.ndim == 4 else v[0] for v in vids], 0)
        del res, vids
        if not j.get("chain_from") or j.get("seam_tries") == 0:
            best = None
            break
        ok, info = seam_ok(out_dir / f"{j['chain_from']}.mp4", j["chain_frame"], frames, j.get("overlap", 12))
        log(f"[{dev}] {j['name']} seam check (try {attempt + 1}): {info}")
        if ok:
            best = None
            break
        step = seam_step(info)
        if best is None or step < best[0]:
            best = (step, frames.cpu())
        kw["seed"] = kw["seed"] + 1000  # re-roll this chunk
        del frames
        frames = None
    if best is not None:
        frames = best[1]
        j = {**j, "seam_flag": f"all {attempt + 1} tries failed the seam check; kept step {best[0]:.2f}"}
        log(f"[{dev}] {j['name']} SEAM FLAG: {j['seam_flag']}")
    return frames, j


def gpu_waiters(locks=("/tmp/gpu-05.lock", "/tmp/gpu-65.lock")) -> bool:
    """True when a process outside our own ancestry has a GPU lock file open (waiting on it, with flock or gaprun).
    Off unless LTX_YIELD=1 (the remake driver sets it). Checked between jobs, at most every 30 s."""
    import os as _os
    import subprocess as _sp
    import time as _t
    if _os.environ.get("LTX_YIELD") != "1":
        return False
    now = _t.time()
    if now - getattr(gpu_waiters, "_t", 0) < 30:
        return getattr(gpu_waiters, "_v", False)
    gpu_waiters._t = now
    mine, pid = set(), _os.getpid()
    while pid > 1:
        mine.add(pid)
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    def tree_of(p):          # our children too (process B of --pp)
        try:
            return {int(x) for x in _sp.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()}
        except Exception:
            return set()
    kids, frontier = set(), {_os.getpid()}
    while frontier:
        nxt = set()
        for q in frontier:
            nxt |= tree_of(q) - kids
        kids |= nxt
        frontier = nxt
    mine |= kids
    r = _sp.run(["fuser", *locks], capture_output=True, text=True)
    pids = {int(x) for x in (r.stdout + " " + r.stderr).replace(":", " ").split() if x.isdigit()}
    others = pids - mine
    gpu_waiters._v = bool(others)
    return gpu_waiters._v


def seam_step(info: str) -> float:
    """'seam max step 5.74 vs prev median 5.10' -> 5.74 (inf when unparsable)."""
    m = re.search(r"step ([0-9.]+)", info or "")
    return float(m.group(1)) if m else float("inf")


def tmap(obj, fn):
    """fn applied to every tensor inside dataclasses / namedtuples / tuples / lists / dicts (as to_shm / move)."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        ch = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            nv = tmap(v, fn)
            if nv is not v:
                ch[f.name] = nv
        return dataclasses.replace(obj, **ch) if ch else obj
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(tmap(v, fn) for v in obj))
    if isinstance(obj, (tuple, list)):
        return type(obj)(tmap(v, fn) for v in obj)
    if isinstance(obj, dict):
        return {k: tmap(v, fn) for k, v in obj.items()}
    return obj


class _Ref:
    """A pool buffer the other process already has: sent by name only."""

    def __init__(self, name, buf=None):
        self.name, self.buf = name, buf


class ShmPool:
    """--pp-pool (10-24): the activations cross between the two processes through REUSED, PINNED shared-memory
    buffers (one per lane, direction, tensor slot and shape) instead of fresh pageable ones per hop: no allocation
    and page-faulting per hop, DMA straight from pinned pages; a buffer travels once, then only its name.
    Reuse is safe: a lane's next hop waits for the other side's reply, and the receiver copies a buffer onto its
    card (synchronously) before it replies."""

    def __init__(self):
        self.bufs = {}

    def _pin(self, t):
        try:
            torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        except Exception:                    # noqa: BLE001  (unpinned still works)
            pass
        return t

    def pack(self, obj, tag):
        n = [0]

        def put(t):
            n[0] += 1
            name = f"{tag}:{n[0]}:{tuple(t.shape)}:{t.dtype}"
            new = name not in self.bufs
            if new:
                self.bufs[name] = self._pin(torch.empty(t.shape, dtype=t.dtype).share_memory_())
            b = self.bufs[name]
            b.copy_(t, non_blocking=True)
            return _Ref(name, b if new else None)
        out = tmap(obj, put)
        torch.cuda.current_stream().synchronize()      # landed before the other side reads it
        return out

    def unpack(self, obj, dev):
        def walk(o):
            if isinstance(o, _Ref):
                if o.buf is not None:
                    self.bufs[o.name] = self._pin(o.buf)
                return self.bufs[o.name].to(dev, non_blocking=False)
            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                ch = {f.name: walk(getattr(o, f.name)) for f in dataclasses.fields(o)}
                return dataclasses.replace(o, **ch)
            if isinstance(o, tuple) and hasattr(o, "_fields"):
                return type(o)(*(walk(v) for v in o))
            if isinstance(o, (tuple, list)):
                return type(o)(walk(v) for v in o)
            if isinstance(o, dict):
                return {k: walk(v) for k, v in o.items()}
            return o
        return walk(obj)


POOL = None


def thash(t):
    """Exact-ish fingerprint of a tensor's bits (int64 sums of the raw 16-bit words, plain and position-weighted)."""
    if t is None:
        return "None"
    w = t.detach().contiguous().view(-1)
    w = w.view(torch.int16).to(torch.int64) if w.element_size() == 2 else w.view(torch.int32).to(torch.int64)
    idx = torch.arange(w.numel(), device=w.device, dtype=torch.int64) % 9973
    return f"{int(w.sum())}:{int((w * idx).sum())}"


def hlog(job, call, where, video):
    if HASHLOG["on"] and video is not None:
        print(f"HASH job={job} call={call} where={where} x={thash(video.x)}", flush=True)


def to_shm(obj):
    """GPU tensors (recursively) -> fresh CPU shared-memory tensors. The D2H copy is blocking, so the data
    has landed before the caller hands the object to the queue. Every hop gets NEW buffers: a slot is never
    reused, it is freed when both processes drop their reference (the strictest handshake)."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        if obj.device.type == "cpu":
            return obj.clone().share_memory_()
        t = torch.empty(obj.shape, dtype=obj.dtype).share_memory_()
        t.copy_(obj)
        return t
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        ch = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            nv = to_shm(v)
            if nv is not v:
                ch[f.name] = nv
        return dataclasses.replace(obj, **ch) if ch else obj
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(to_shm(v) for v in obj))
    if isinstance(obj, (tuple, list)):
        return type(obj)(to_shm(v) for v in obj)
    if isinstance(obj, dict):
        return {k: to_shm(v) for k, v in obj.items()}
    return obj


def ic_loras(args):
    """--ic-lora a.safetensors[,b.safetensors] with --lora-strength s[,t] -> [(path, strength)]."""
    paths = [x for x in args.ic_lora.split(",") if x]
    st = [float(x) for x in str(args.lora_strength).split(",")]
    st = st + [st[-1]] * (len(paths) - len(st))
    return list(zip(paths, st))


def prepared_builder(args):
    """The DiT builder (fp8 policy, module ops) for the current mode; no weights are touched."""
    from ltx_core.quantization.fp8_cast import build_policy
    from ltx_pipelines.utils.model_paths import ModelPaths
    if args.a2vid or args.ti2v:
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        paths = ModelPaths.from_split(transformer_path=DIT_DEV, text_encoder_path=TE, video_vae_path=VAE,
                                      audio_vae_path=AVAE, duration_head_path=DUR)
        dl = LoraPathStrengthAndSDOps(DLORA, args.dlora_strength, LTXV_LORA_COMFY_RENAMING_MAP)
        p = (TI2VidTwoStagesPipeline if args.ti2v else A2VidPipelineTwoStage)(
            model_paths=paths, distilled_lora=[dl], spatial_upsampler_path=UPS, loras=[], device=D0,
            quantization=build_policy(DIT_DEV))
        return p.stage_1._prepared_builder()
    from ltx_pipelines.distilled import DistilledPipeline
    base = DIT_DEV if args.base == "dev" else DIT
    paths = ModelPaths.from_split(transformer_path=base, text_encoder_path=TE, video_vae_path=VAE,
                                  audio_vae_path=AVAE, duration_head_path=DUR)
    p = DistilledPipeline(model_paths=paths, spatial_upsampler_path=UPS, loras=(), device=D0,
                          quantization=build_policy(base))
    return p.stage._prepared_builder()  # the IC pipeline's stage builds the same model (LoRA is runtime)


def _die_with_parent():
    """The back half must never outlive its parent (10-20: an orphan held 12 GB on XTX 65 and OOM'd two holds):
    PR_SET_PDEATHSIG SIGKILL + a parent-pid watchdog."""
    import ctypes
    import signal
    ppid = os.getppid()
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)       # PR_SET_PDEATHSIG
    except OSError:
        pass

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(3)
            time.sleep(2)
    threading.Thread(target=watch, daemon=True).start()


def vram_guard(limit_gb: float = 1.5):
    """Before a hold: every visible card must be (nearly) empty; else name the processes that have /dev/kfd open."""
    bad = []
    for d_ in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d_)
        used = (total - free - torch.cuda.memory_reserved(d_)) / 2**30 - 0.6   # minus our own + the HIP context
        if used > limit_gb:
            bad.append((d_, round(used, 1)))
    if bad:
        pids = []
        for pd in os.listdir("/proc"):
            if not pd.isdigit() or int(pd) == os.getpid():
                continue
            try:
                if any(os.readlink(f"/proc/{pd}/fd/{fd}") == "/dev/kfd" for fd in os.listdir(f"/proc/{pd}/fd")):
                    pids.append(f"{pd}:{open(f'/proc/{pd}/cmdline').read().replace(chr(0), ' ')[:80]}")
            except OSError:
                pass
        raise SystemExit(f"VRAM held before the hold: {bad} (GiB used per visible card); GPU processes: {pids}")


def back_server(args, inq, outqs, ready, decout=None):
    """Process B (its own Python, its own card as cuda:0): DiT blocks >= split, served FIFO for all lanes."""
    global T0
    T0 = time.time()
    _die_with_parent()
    try:
        torch.cuda.set_device(0)
        if not args.miopen:
            torch.backends.cudnn.enabled = False
        if args.blaslt:
            torch.backends.cuda.preferred_blas_library("cublaslt")
        from ltx_core.guidance.perturbations import PerturbationType as P
        ltx = build_split(prepared_builder(args), layer_devfn(r"^transformer_blocks\.(\d+)\.", args.split, "back")).eval()
        if args.fa_amd:
            install_fa_amd(ltx)
        state = types.SimpleNamespace(on=False)
        if args.ic_lora:
            for lp, ls in ic_loras(args):
                attach_runtime_lora(ltx, lp, ls, state)
        if args.a2vid or args.ti2v:
            attach_runtime_lora(ltx, DLORA, args.dlora_strength, state)
        elif args.base == "dev":
            attach_runtime_lora(ltx, DLORA, args.dlora_strength, types.SimpleNamespace(on=True))
        if args.compile:
            compile_blocks(ltx)
        nb = len(ltx.transformer_blocks)
        dec = None
        if args.decode_b or args.decode_pair:   # (10-24) clip decodes here too (10-31: --decode-pair, see PairSched)
            from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
            vd = VideoDecoder(VAE, torch.bfloat16, D0)
            dec = vd._prepared_builder().build(device=D0, dtype=torch.bfloat16).eval()
        log(f"[B] back half loaded: blocks {args.split}..{nb - 1}; alloc {torch.cuda.memory_allocated(0) / 2**30:.1f} GiB"
            + (" (+ VAE decoder)" if dec is not None else ""))
        ready.put(("ok", torch.cuda.memory_allocated(0)))
        busy = [0.0]
        bpool = ShmPool()
        import queue
        decq = queue.Queue()

        def decoder():
            """(10-31) Decodes on their own thread + stream: the other lane's DiT calls no longer queue behind a
            ~14 s decode (FIFO head-of-line block = the main idle on both cards)."""
            with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream(D0)):
                while (m := decq.get()) is not None:
                    if m[0] == "pairdec":     # --decode-pair: a parked clip's decode; frames to the collector on A
                        _, jid, latent, tiling, dt, gstate = m
                        t = time.time()
                        g = None
                        if gstate is not None:
                            g = torch.Generator(device=D0)
                            g.set_state(gstate)
                        fr = list(dec.decode_video(latent.to(D0, dtype=dt), tiling, g))
                        fr = fr[0] if len(fr) == 1 else torch.cat(fr, 0)
                        torch.cuda.current_stream().synchronize()
                        out = to_shm(fr)
                        busy[0] += time.time() - t
                        tl("B", -1, "decode", t, time.time())
                        decout.put((jid, out, time.time() - t))
                        del latent, fr, out, m
                        continue
                    _, lane, latent, tiling, dt, gstate = m
                    t = time.time()
                    g = None
                    if gstate is not None:
                        g = torch.Generator(device=D0)
                        g.set_state(gstate)
                    fr = list(dec.decode_video(latent.to(D0, dtype=dt), tiling, g))
                    fr = fr[0] if len(fr) == 1 else torch.cat(fr, 0)
                    torch.cuda.current_stream().synchronize()
                    out = to_shm(fr)
                    busy[0] += time.time() - t
                    tl("B", lane, "decode", t, time.time())
                    outqs[lane].put((out, time.time() - t))
                    del latent, fr, out, m
        dthr = threading.Thread(target=decoder, daemon=True)
        dthr.start()
        with torch.inference_mode():
            while True:
                msg = inq.get()
                if msg is None:
                    break
                if msg[0] == "pairdec" and ATOMIC:       # inline: card B also does one job at a time
                    _, jid, latent, tiling, dt, gstate = msg
                    t = time.time()
                    g = None
                    if gstate is not None:
                        g = torch.Generator(device=D0)
                        g.set_state(gstate)
                    fr = list(dec.decode_video(latent.to(D0, dtype=dt), tiling, g))
                    fr = fr[0] if len(fr) == 1 else torch.cat(fr, 0)
                    torch.cuda.current_stream().synchronize()
                    out = to_shm(fr)
                    busy[0] += time.time() - t
                    tl("B", -1, "decode", t, time.time())
                    decout.put((jid, out, time.time() - t))
                    del latent, fr, out, msg
                    continue
                if msg[0] in ("decode", "pairdec"):
                    decq.put(msg)
                    del msg
                    continue
                lane, video, audio, pert, on = msg
                t = time.time()
                state.on = on
                if args.pp_pool:
                    video, audio, pert = (bpool.unpack(x, D0) for x in (video, audio, pert))
                else:
                    video, audio, pert = move(video, D0), move(audio, D0), move(pert, D0)
                del msg
                torch.cuda.current_stream().synchronize()
                t_in = time.time()
                for i in range(args.split, nb):
                    if video is not None:
                        video = ltx.block_input_processor(video, pert, i, self_attn_type=P.SKIP_VIDEO_SELF_ATTN,
                                                          cross_attn_type=P.SKIP_A2V_CROSS_ATTN)
                    if audio is not None:
                        audio = ltx.block_input_processor(audio, pert, i, self_attn_type=P.SKIP_AUDIO_SELF_ATTN,
                                                          cross_attn_type=P.SKIP_V2A_CROSS_ATTN)
                    video, audio = ltx.transformer_blocks[i](video=video, audio=audio)
                torch.cuda.current_stream().synchronize()
                t_c = time.time()
                out = bpool.pack((video, audio), f"b{lane}") if args.pp_pool else (to_shm(video), to_shm(audio))
                t_out = time.time()
                tl("B", lane, "copyin", t, t_in)
                tl("B", lane, "dit", t_in, t_c)
                tl("B", lane, "copyout", t_c, t_out)
                busy[0] += t_out - t
                outqs[lane].put((*out, (t_c - t_in, (t_in - t) + (t_out - t_c))))
                del video, audio, pert, out
        decq.put(None)
        dthr.join()
        busy = busy[0]
        if args.compile:
            log("[B]", compile_stats())
        log(f"[B] done; busy {busy:.1f}s; peak {torch.cuda.max_memory_allocated(0) / 2**30:.1f} GiB")
        ready.put(("busy", busy, torch.cuda.max_memory_allocated(0)))
    except BaseException as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        ready.put(("error", repr(e)))


# ---------------------------------------------------------------- encode
def cmd_encode(args, jobs):
    from ltx_core.loader.registry import ModelRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
        GemmaTextEncoderConfigurator,
        get_gemma_ops,
        resolve_gemma_weight_paths,
    )

    emb_dir = Path(args.emb)
    emb_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (emb_dir / f"{j['name']}.pt").exists() or args.force]
    if not todo:
        log("all embeddings cached")
        return
    reg = ModelRegistry(cache_models=False, cache_weights=False)
    ops, mops = get_gemma_ops(TE)
    b = Builder(model_path=resolve_gemma_weight_paths(TE),
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(TE),
                model_sd_ops=ops, module_ops=mops, registry=reg)
    devfn = layer_devfn(r"layers\.(\d+)\.", 24)
    log("loading Gemma split 24/24 ...")
    te = build_split(b, devfn).to(torch.bfloat16).eval()
    log("Gemma loaded;", vram())
    lm = te.model.model
    # find the decoder layer list and hook every layer + final norm to pull inputs to its device
    layers = None
    for n, mod in lm.named_modules():
        if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) == 48:
            layers, lname = mod, n
    assert layers is not None
    log("decoder layers at", lname)

    def hook(mod, a, kw):
        d = next(mod.parameters()).device
        return move(a, d), move(kw, d)

    for L in layers:
        L.register_forward_pre_hook(hook, with_kwargs=True)
    parent = lm.get_submodule(lname.rsplit(".", 1)[0]) if "." in lname else lm
    if hasattr(parent, "norm"):
        parent.norm.register_forward_pre_hook(hook, with_kwargs=True)
    raw = []
    with torch.inference_mode():
        for j in todo:
            t = time.time()
            (hs, mask), = te.encode([j["prompt"]])
            # hidden states of all layers are ~0.4 GB per prompt: park them in host RAM, not on cuda:0
            # (52 prompts on the card = 20 GB -> OOM inside Gemma SDPA)
            raw.append((tuple(h.to("cpu") for h in hs), mask.to("cpu")))
            log(f"encoded {j['name']} in {time.time() - t:.1f}s")
    te.dispose() if hasattr(te, "dispose") else None
    del te, lm, layers
    gc.collect()
    torch.cuda.empty_cache()
    ep = Builder(model_path=(DIT_DEV if args.dev else DIT, TE),
                 model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(TE),
                 model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS, registry=reg)
    proc = build_split(ep, lambda k: D0).to(torch.bfloat16).eval()
    with torch.inference_mode():
        for j, (hs, mask) in zip(todo, raw):
            out = proc.process_hidden_states(tuple(h.to(D0) for h in hs), mask.to(D0))
            torch.save({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in out._asdict().items()},
                       emb_dir / f"{j['name']}.pt")
            log("saved", j["name"], {k: tuple(v.shape) for k, v in out._asdict().items() if v is not None})


# ---------------------------------------------------------------- gen
CARD_LOCKS = {0: threading.Lock(), 1: threading.Lock()}
# --pp: ONE lane at a time owns card A (process A's cuda:0). A lane gives the card up only while its activations are
# in process B (after a synchronous D2H), and takes it back when B's answer arrives. So card A never runs two
# lanes' kernels concurrently (bitwise-reproducible), while B's half of lane X overlaps A's work for lane Y.
class _TimedLock:
    """threading.Lock that accumulates the time it is held (= process A's card busy time under --pp)."""

    def __init__(self):
        self._l, self.held, self._t = threading.Lock(), 0.0, 0.0

    def acquire(self):
        self._l.acquire()
        self._t = time.time()

    def release(self):
        self.held += time.time() - self._t
        self._l.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()


class _SharedOwner(_TimedLock):
    """--pp-shared-a (10-21): no exclusive ownership of card A; each lane runs on its own stream, so one lane's VAE
    work overlaps the other lane's DiT steps instead of convoying (lanes waited 28-58 s per clip on B for 14-28 s of
    B work: both lanes in the DiT, then both in the VAE). Counts nothing: 'owned' then reads 0."""

    def acquire(self):
        pass

    def release(self):
        pass


A_OWNER = _TimedLock()
_TL_LOCK = threading.Lock()


def tl(card, lane, kind, t0, t1):
    """(10-31) Per-call timeline, [card, lane, kind, t0, t1] per line to $LTX_TIMELINE (default <out>/timeline.jsonl):
    the per-card occupancy the per-lane pp_timing timers can't give (with 2 lanes they count overlap twice).
    timeline_report.py reads it."""
    f = os.environ.get("LTX_TIMELINE")
    if f:
        with _TL_LOCK, open(f, "a") as fh:
            fh.write(json.dumps([card, lane, kind, round(t0, 4), round(t1, 4)]) + "\n")


def install_pp_forward(ltx, split, pp, get_state):
    """Process A side: blocks < split here (cuda:0), then hand the activations to process B and wait."""
    from ltx_core.guidance.perturbations import PerturbationType as P

    def proc(video, audio, perturbations):
        st = get_state()
        st.dcalls = getattr(st, "dcalls", 0) + 1
        hlog(getattr(st, "job", st.name), st.dcalls, "in", video)
        t0 = time.time()
        with CARD_LOCKS[0]:  # launch the whole front run at once (no op interleaving between lanes)
            for i in range(split):
                if video is not None:
                    video = ltx.block_input_processor(video, perturbations, i, self_attn_type=P.SKIP_VIDEO_SELF_ATTN,
                                                      cross_attn_type=P.SKIP_A2V_CROSS_ATTN)
                if audio is not None:
                    audio = ltx.block_input_processor(audio, perturbations, i, self_attn_type=P.SKIP_AUDIO_SELF_ATTN,
                                                      cross_attn_type=P.SKIP_V2A_CROSS_ATTN)
                video, audio = ltx.transformer_blocks[i](video=video, audio=audio)
            if ATOMIC:
                torch.cuda.current_stream().synchronize()     # inside the card lock: one job at a time on card A
        torch.cuda.current_stream().synchronize()     # this lane's stream (a parked decode may run on card A)
        t1 = time.time()
        tm_add("a_dit_s", t1 - t0)
        hlog(getattr(st, "job", st.name), st.dcalls, "handoff", video)
        if POOL is not None:
            msg = (st.lane, *POOL.pack((video, audio, perturbations), f"a{st.lane}"), st.on)
        else:
            msg = (st.lane, to_shm(video), to_shm(audio), to_shm(perturbations), st.on)  # blocking D2H: card A idle for us
        t2 = time.time()
        tm_add("a_copy_s", t2 - t1)
        A_OWNER.release()
        try:
            pp["inq"].put(msg)
            del msg
            if PAIR is not None:
                PAIR.a_turn(st.lane)
            v, a, bt = pp["outqs"][st.lane].get()
            if PAIR is not None:
                PAIR.after_b()
        finally:
            A_OWNER.acquire()
        t3 = time.time()
        tl("A", st.lane, "dit", t0, t1)
        tl("A", st.lane, "copy", t1, t2)
        tl("A", st.lane, "wait", t2, t3)
        tm_add("wait_b_s", t3 - t2)
        tm_add("b_dit_s", bt[0])
        tm_add("b_copy_s", bt[1])
        v, a = (POOL.unpack(v, D0), POOL.unpack(a, D0)) if POOL is not None else (move(v, D0), move(a, D0))
        tm_add("a_copy_s", time.time() - t3)
        tl("A", st.lane, "copyback", t3, time.time())
        hlog(getattr(st, "job", st.name), st.dcalls, "back", v)
        return v, a

    ltx._process_transformer_blocks = proc


class DeferredDecode(Exception):
    """Raised out of the pipeline when a clip's decode is parked for the pair scheduler (--decode-pair): the lane
    leaves this job here and takes its next clip; the decode finishes the clip later."""


PAIR = None     # --decode-pair scheduler
# 10-31 user: work on a card is ATOMIC - one job at a time, start to finish (a lane's front run, or a decode, holds
# card A until done; card B runs its parked decodes inline in its serial loop). Each lane on its own stream had let
# the lanes' kernels interleave on card A. Atomic, the two lanes fall into alternating by themselves, so the lane
# stagger is 0. 7 talk clips (paired decode, split 24): 422 -> 386.5 s; output identical to non-atomic.
ATOMIC = os.environ.get("LTX_ATOMIC", "1") != "0"


class PairSched:
    """--decode-pair (10-31, the user's scheme). Lanes never decode: a finished clip's latent is parked here and the
    lane starts its next clip. Card B decodes when it has completed its layers (a lane just got B's reply) and at
    least 2 latents are parked, so one is left for card A. Card A decodes when it has completed its layers (a lane
    handed its step to B and now waits) and sees card B decoding. Both cards decode in the same window, instead of
    one card's 14-25 s decode starving the other (timeline 10-31: card A idle ~50% with every decode on B).
    7 talk clips, 2 lanes: 463 -> 425 s (-8.4%), clips byte-identical. A parked clip's job stays 'running' until
    its mp4 is written, so a chained piece still waits for its source."""

    def __init__(self, pp):
        from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
        self.deca = VideoDecoder(VAE, torch.bfloat16, D0)._prepared_builder().build(device=D0, dtype=torch.bfloat16).eval()
        self.pp, self.lk, self.parked, self.b_busy, self.on_b, self.n, self.err = pp, threading.Condition(), [], False, {}, 0, []
        threading.Thread(target=self.collect, daemon=True).start()

    def park(self, item):
        with self.lk:
            self.parked.append(item)

    def take(self):
        with self.lk:
            return self.parked.pop(0) if self.parked else None

    def to_b(self, item):
        with self.lk:
            self.b_busy = True
            self.n += 1
            self.on_b[self.n] = item[4]
            jid = self.n
        latent, tiling, dt, gstate, _ = item
        self.pp["inq"].put(("pairdec", jid, latent, tiling, dt, gstate))

    def after_b(self):
        """Card B has completed its layers: it decodes if it is free and >= 2 latents are parked."""
        with self.lk:
            if self.b_busy or len(self.parked) < 2:
                return
            it = self.parked.pop(0)
        self.to_b(it)

    def a_turn(self, lane):
        """Card A has completed its layers: it decodes while card B decodes."""
        with self.lk:
            if not self.b_busy or not self.parked:
                return
            it = self.parked.pop(0)
        self.decode_a(it, lane)

    def decode_a(self, item, lane):
        latent, tiling, dt, gstate, fin = item
        t = time.time()
        try:
            with (CARD_LOCKS[0] if ATOMIC else nullcontext()), torch.inference_mode():
                g = None
                if gstate is not None:
                    g = torch.Generator(device=D0)
                    g.set_state(gstate)
                fr = list(self.deca.decode_video(latent.to(D0, dtype=dt), tiling, g))
                fr = fr[0] if len(fr) == 1 else torch.cat(fr, 0)
                torch.cuda.current_stream().synchronize()
            tl("A", lane, "decode", t, time.time())
            fin(fr, time.time() - t, "A")
        except BaseException as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.err.append(e)

    def collect(self):
        while True:
            jid, fr, sec = self.pp["decout"].get()
            try:
                self.on_b[jid](fr, sec, "B")
            except BaseException as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.err.append(e)
            finally:
                with self.lk:
                    del self.on_b[jid]
                    self.b_busy = False
                    self.lk.notify_all()

    def flush(self):
        """End of the hold: the parked rest, on whichever card is free; then wait for card B's."""
        while (it := self.take()) is not None:
            with self.lk:
                use_b = not self.b_busy
            self.to_b(it) if use_b else self.decode_a(it, -1)
        with self.lk:
            self.lk.wait_for(lambda: not self.on_b)


class RemoteDecoder:
    """--decode-b (10-24): a lane's VAE decode on process B's card (which waits while card A decodes: quiet clips
    spent 22 of their 45 s of work there). Same latent, dtype, tiling and generator state; keyframe decodes stay
    local."""

    def __init__(self, local, pp, get_state):
        self.local, self.pp, self.get_state = local, pp, get_state

    def __getattr__(self, k):
        return getattr(self.local, k)

    def __call__(self, latent, tiling_config=None, generator=None, *, dtype=None, keyframes=None, device_fn=None):
        if keyframes is not None:
            return self.local(latent, tiling_config, generator, dtype=dtype, keyframes=keyframes, device_fn=device_fn)
        st = self.get_state()
        dt = dtype or self.local._dtype
        if PAIR is not None and getattr(st, "defer", None) is not None:
            torch.cuda.current_stream().synchronize()
            st.defer(latent.to(dtype=dt).cpu(), tiling_config, dt, generator.get_state() if generator is not None else None)
            raise DeferredDecode()
        t0 = time.time()
        msg = ("decode", st.lane, to_shm(latent.to(dtype=dt)), tiling_config, dt,
               generator.get_state() if generator is not None else None)
        A_OWNER.release()
        try:
            self.pp["inq"].put(msg)
            del msg
            fr, bt = self.pp["outqs"][st.lane].get()
        finally:
            A_OWNER.acquire()
        tm_add("b_decode_s", bt)
        tm_add("decode_wait_s", time.time() - t0)
        return iter([fr.to(latent.device)])


def install_split_forward(ltx, devs, get_state=None):
    """Blocks run on their card. Each run of consecutive same-card blocks is LAUNCHED under that card's lock,
    so with two lanes (threads) one lane's whole front half is queued before the other's: the stream FIFO then
    executes them back to back and the lanes fall into a pipeline (A on cuda:1 while B on cuda:0) instead of a
    convoy (both interleaved on one card, then both on the other)."""
    from ltx_core.guidance.perturbations import PerturbationType as P

    runs, start = [], 0
    for i in range(1, len(devs) + 1):
        if i == len(devs) or devs[i] != devs[start]:
            runs.append((start, i, devs[start]))
            start = i

    def proc(video, audio, perturbations):
        blocks = ltx.transformer_blocks
        st = get_state() if get_state else None
        if st is not None:
            st.dcalls = getattr(st, "dcalls", 0) + 1
            hlog(getattr(st, "job", st.name), st.dcalls, "in", video)
        for ri, (a, b, d) in enumerate(runs):
            if st is not None and ri == 1:
                hlog(getattr(st, "job", st.name), st.dcalls, "handoff", video)
            with CARD_LOCKS[d.index]:
                for i in range(a, b):
                    block = blocks[i]
                    if video is not None:
                        video = ltx.block_input_processor(video, perturbations, i, self_attn_type=P.SKIP_VIDEO_SELF_ATTN,
                                                          cross_attn_type=P.SKIP_A2V_CROSS_ATTN)
                    if audio is not None:
                        audio = ltx.block_input_processor(audio, perturbations, i, self_attn_type=P.SKIP_AUDIO_SELF_ATTN,
                                                          cross_attn_type=P.SKIP_V2A_CROSS_ATTN)
                    video, audio = move(video, d), move(audio, d)
                    video, audio = block(video=video, audio=audio)
        video, audio = move(video, D0), move(audio, D0)
        if st is not None:
            hlog(getattr(st, "job", st.name), st.dcalls, "back", video)
        return video, audio

    ltx._process_transformer_blocks = proc


def stereo_audio(path, out_dir):
    """LTX's audio VAE wants 2 channels; TTS wavs are mono. Returns a stereo 48 kHz copy when needed."""
    import subprocess
    ch = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
                         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    if ch == "2":
        return path
    dst = Path(out_dir) / "audio_st" / (Path(path).stem + "_st48.wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "2", "-ar", "48000", str(dst)], check=True)
    log(f"audio {Path(path).name}: {ch or '?'} channel(s) -> stereo copy {dst}")
    return str(dst)


def seam_ok(prev_mp4, chain_frame, frames, overlap, factor=3.0):
    """Simulate the stitch's crossfade over `overlap` frames and compare the largest frame-to-frame step in
    the seam window with the previous chunk's median step. ok = max <= factor * median."""
    import av
    import numpy as np
    prev = np.stack([f.to_ndarray(format="rgb24") for f in av.open(str(prev_mp4)).decode(video=0)]).astype(np.float32)
    new = (frames.float().clamp(0, 1) * 255).cpu().numpy()
    a = prev[chain_frame:chain_frame + overlap]
    o = min(len(a), overlap)
    w = np.linspace(0, 1, o + 2)[1:-1][:, None, None, None]
    blend = (1 - w) * a[:o] + w * new[:o]
    seq = np.concatenate([prev[max(chain_frame - 3, 0):chain_frame], blend, new[o:o + 4]], 0)
    d = np.abs(np.diff(seq, axis=0)).mean(axis=(1, 2, 3))
    med = float(np.median(np.abs(np.diff(prev, axis=0)).mean(axis=(1, 2, 3))))
    return float(d.max()) <= factor * max(med, 0.5), f"seam max step {d.max():.2f} vs prev median {med:.2f}"


def save_last_png(frames: torch.Tensor, path) -> None:
    """The clip's last frame as a lossless PNG (the next chained piece's first frame, no mp4 round trip)."""
    import numpy as np
    from PIL import Image
    f = frames[-1:]
    if f.dtype != torch.uint8:
        f = f.float()
        if f.min() < -0.01:
            f = (f + 1) / 2
        f = (f.clamp(0, 1) * 255).round().to(torch.uint8)
    a = f[0].cpu().numpy()
    if a.shape[-1] != 3:
        a = np.ascontiguousarray(a.transpose(1, 2, 0))
    tmp = str(path) + ".part.png"
    Image.fromarray(a).save(tmp)
    __import__("os").replace(tmp, path)


def save_mp4(frames: torch.Tensor, path: str, fps: float, crf: int = 20):
    import av
    import numpy as np

    if frames.dtype != torch.uint8:
        f = frames.float()
        if f.min() < -0.01:
            f = (f + 1) / 2
        frames = (f.clamp(0, 1) * 255).round().to(torch.uint8)
    arr = frames.cpu().numpy()
    if arr.shape[-1] != 3:  # F,C,H,W -> F,H,W,C
        arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 1))
    h, w = arr.shape[1:3]
    c = av.open(path, "w")
    s = c.add_stream("libx264", rate=int(round(fps)), options={"crf": str(crf), "preset": "medium"})
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    for fr in arr:
        for pk in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            c.mux(pk)
    for pk in s.encode():
        c.mux(pk)
    c.close()
    return arr


def tstats(t):
    if t is None:
        return "None"
    f = t.float()
    return (f"{tuple(t.shape)} {t.dtype} mean {f.mean().item():.3f} std {f.std().item():.3f} "
            f"absmax {f.abs().max().item():.2f} nan {int(f.isnan().sum())}")


def install_debug(pipe, x0, ddir, name_fn=lambda: "job"):
    """Log latent stats per DiT call, stage-1/upsample/stage-2 latents, and decode output; save latents."""
    ddir.mkdir(parents=True, exist_ok=True)
    calls = {"n": 0, "t": time.time()}

    def post(mod, a, kw, out):
        torch.cuda.synchronize()
        v = out[0] if isinstance(out, (tuple, list)) else getattr(out, "video", out)
        calls["n"] += 1
        log(f"DiT call {calls['n']} {time.time() - calls['t']:.1f}s x0", tstats(v if isinstance(v, torch.Tensor) else None))
        calls["t"] = time.time()

    x0.register_forward_hook(post, with_kwargs=True)

    class Wrap:
        def __init__(self, inner, tag):
            self.inner, self.tag = inner, tag

        def __getattr__(self, k):
            return getattr(self.inner, k)

        def __call__(self, latent, *a, **kw):
            log(f"{self.tag} in", tstats(latent))
            torch.save(latent.detach().cpu(), ddir / f"{name_fn()}_{self.tag}_in.pt")
            t = time.time()
            out = self.inner(latent, *a, **kw)
            if isinstance(out, torch.Tensor):
                torch.cuda.synchronize()
                log(f"{self.tag} out {time.time() - t:.1f}s", tstats(out))
                return out

            def it():
                for c in out:
                    torch.cuda.synchronize()
                    log(f"{self.tag} chunk {time.time() - t:.1f}s", tstats(c))
                    yield c
            return it()

    pipe.upsampler = Wrap(pipe.upsampler, "upsample")
    pipe.video_decoder = Wrap(pipe.video_decoder, "vdecode")


def _lora_linear_forward(self, x):
    """Class-level forward of a Linear with runtime LoRAs (self._runtime_loras: (A, B, state)). Class-level, not an
    instance closure, so torch.compile traces it inline (a closure per module = a graph break and a recompile per
    block). 10-31: on a big input the active LoRAs are FOLDED into the weight upcast from fp8 anyway (W + B A, one
    out x r x in GEMM) and one GEMM runs; the unfused x A^T B^T cost two thin rank-450 GEMMs over every token
    (11% of a stage-2 call at 12k tokens). Small inputs (rows below in*out/(in+out)) keep the unfused sum."""
    on = [(A, B) for A, B, st in self._runtime_loras if st.on]
    if not on:
        return self._base_forward(x)
    F_ = torch.nn.functional
    n_in, n_out = self.weight.shape[1], self.weight.shape[0]
    if (x.numel() // n_in) * (n_in + n_out) <= n_in * n_out:
        out = self._base_forward(x)
        for A, B in on:
            out = out + F_.linear(F_.linear(x.to(A.dtype), A), B).to(out.dtype)
        return out
    # a weight already in x's dtype: .to() returns the parameter itself, so copy it (never fold into the parameter)
    w = self.weight.clone() if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
    for A, B in on:
        w.addmm_(B.to(w.dtype), A.to(w.dtype))
    return F_.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


_LORA_CLASSES = {}


def _lora_forward(m):
    """The runtime LoRAs on Linear m: a list of (A, B, state). m's class becomes a LoRA subclass of its own
    (one per base class) whose forward is _lora_linear_forward."""
    lst = getattr(m, "_runtime_loras", None)
    if lst is not None:
        return lst
    base = type(m)
    cls = _LORA_CLASSES.get(base)
    if cls is None:
        cls = _LORA_CLASSES[base] = type(f"Lora{base.__name__}", (base,),
                                         {"forward": _lora_linear_forward, "_base_forward": base.forward})
    lst = m._runtime_loras = []
    m.__class__ = cls
    return lst


def attach_runtime_lora(ltx, path, strength, state):
    """Runtime LoRA (see _lora_forward): adds strength * B A to each target Linear, on that Linear's card.
    One resident fp8 DiT serves both the LoRA stage (state.on=True) and the bare stage.
    Streams: each A/B pair goes to its card as soon as both halves are read (no full LoRA in RAM)."""
    n, nbytes, missing, pend = 0, 0, [], {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            name = k.removeprefix("diffusion_model.")
            mod_name, ab = name.rsplit(".lora_", 1)
            pend.setdefault(mod_name, {})[ab[0]] = k
            if len(pend[mod_name]) < 2:
                continue
            keys = pend.pop(mod_name)
            try:
                m = ltx.get_submodule(mod_name)
            except AttributeError:
                missing.append(mod_name)
                continue
            dev = m.weight.device
            if dev.type == "meta":
                continue
            A = f.get_tensor(keys["A"]).to(dev, torch.bfloat16)
            B = (f.get_tensor(keys["B"]).to(dev, torch.float32) * strength).to(torch.bfloat16)
            nbytes += A.nbytes + B.nbytes

            _lora_forward(m).append((A, B, state))
            n += 1
    log(f"runtime LoRA {Path(path).name} x{strength}: {n} linears hooked, {nbytes / 2**20:.0f} MiB, "
        f"missing {len(missing)} {missing[:5]}, unpaired {len(pend)}")
    return n


def serialize_sdpa():
    """ltx_core's PytorchAttention enters torch's sdpa_kernel(..., set_priority=True) on EVERY call. That mutates
    process-global backend flags and restores them on exit, so with two lanes (threads) one lane can launch its
    SDPA while the other has the flags half-restored, and get a different backend (tiny numeric differences,
    41 dB vs sequential). One lock around the call keeps every SDPA launch inside a consistent flag state."""
    import ltx_core.model.transformer.attention as LA
    if getattr(LA.PytorchAttention, "_serialized", False):
        return
    orig, lk = LA.PytorchAttention.__call__, threading.Lock()

    def call(self, *a, **k):
        with lk:
            return orig(self, *a, **k)

    LA.PytorchAttention.__call__ = call
    LA.PytorchAttention._serialized = True


ORDER_COST = {"a2vid": 1.0, "ti2v": 1.3}   # card time per frame vs talk (10-30 pp_timing: 1.30 vs 1.66 s/frame)


def order_jobs(todo: list, ti2v_default: bool = False) -> list:
    """--order cost (10-31 user): the hold's jobs from the most to the least estimated card time (frames x route
    factor). Neighbouring jobs - the ones the two lanes run side by side - then have similar half-steps, so the
    lanes keep alternating instead of colliding on one card, their paired decodes are of similar length (decode
    ~0.13 s/frame), and the hold ends on short clips instead of one long straggler. Order changes no output (each
    job has its own seed); a chained piece still waits for its source (lane_loop), retakes still join at the end."""
    def cost(j):
        route = "a2vid" if j.get("audio") else ("ti2v" if ti2v_default or j.get("neg_extra") is not None else "a2vid")
        return (j.get("frames") or 0) * ORDER_COST.get(route, 1.0)
    return sorted(todo, key=lambda j: -cost(j))     # stable: equal costs keep the plan's order


def cmd_gen(args, jobs):
    from ltx_core.quantization.fp8_cast import build_policy
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_pipelines.utils.blocks.diffusion_stage import DiffusionStage
    from ltx_core.model.transformer import X0Model
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput

    torch.cuda.set_device(0)
    if args.out:
        os.environ.setdefault("LTX_TIMELINE", str(Path(args.out) / "timeline.jsonl"))   # B inherits it (spawn)
    if args.compile and len(jobs) < COMPILE_MIN_JOBS:
        args.compile = False        # before B is spawned with these args: both halves stay eager
        log(f"--compile off: {len(jobs)} jobs < {COMPILE_MIN_JOBS} (its warm-up would not pay back)")
    HASHLOG["on"] = args.hashlog
    if args.lanes > 1:
        serialize_sdpa()
    if not args.miopen:  # MIOpen conv3d on gfx1100: VAE round trip 13.8 dB + 60x slower; off = 32.3 dB (= CPU)
        torch.backends.cudnn.enabled = False
        log("MIOpen (cudnn backend) disabled")
    if args.blaslt:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        log("preferred BLAS: hipBLASLt")
    BASE = DIT_DEV if args.base == "dev" else DIT
    paths = ModelPaths.from_split(transformer_path=BASE, text_encoder_path=TE, video_vae_path=VAE,
                                  audio_vae_path=AVAE, duration_head_path=DUR)
    ic = bool(args.ic_lora)
    loras_ic = [LoraPathStrengthAndSDOps(lp, ls, LTXV_LORA_COMFY_RENAMING_MAP) for lp, ls in ic_loras(args)] if ic else []

    a2v = bool(args.a2vid)
    # --ti2v (10-23): image-to-video on the same two stages as A2Vid (dev DiT + CFG/STG at half size, distilled
    # LoRA refine) with a negative prompt - the fight route; pipelined like A2Vid
    ti2v = bool(args.ti2v)
    if a2v or ti2v:
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.constants import detect_params
        a2v_params = detect_params(DIT_DEV)
        paths = ModelPaths.from_split(transformer_path=DIT_DEV, text_encoder_path=TE, video_vae_path=VAE,
                                      audio_vae_path=AVAE, duration_head_path=DUR)
        dlora = LoraPathStrengthAndSDOps(DLORA, args.dlora_strength, LTXV_LORA_COMFY_RENAMING_MAP)

    # 10-28 user: action (--ti2v) and talk (--a2vid) in ONE hold - the same dev DiT loaded once, both pipelines built
    # over it; each job takes the route its fields say (a talk job has "audio")
    mixed = a2v and ti2v

    def make_pipe(dev, kind=None):
        kind = kind or ("a2vid" if a2v else "ti2v" if ti2v else None)
        if kind == "ti2v":
            return TI2VidTwoStagesPipeline(model_paths=paths, distilled_lora=[dlora], spatial_upsampler_path=UPS,
                                           loras=[], device=dev, quantization=build_policy(DIT_DEV)), "stage_1"
        if kind == "a2vid":
            return A2VidPipelineTwoStage(model_paths=paths, distilled_lora=[dlora], spatial_upsampler_path=UPS,
                                         loras=[], device=dev, quantization=build_policy(DIT_DEV)), "stage_1"
        if ic:
            return ICLoraPipeline(model_paths=paths, spatial_upsampler_path=UPS, loras=loras_ic, device=dev,
                                  quantization=build_policy(BASE)), "_diffusion_stage"
        return DistilledPipeline(model_paths=paths, spatial_upsampler_path=UPS, loras=(), device=dev,
                                 quantization=build_policy(BASE)), "stage"

    pipe, stage_attr = make_pipe(D0)
    if ic:
        log("IC-LoRA pipeline; reference downscale", pipe.reference_downscale_factor,
            "temporal", pipe.reference_temporal_scale_factor)
    elif not (a2v or ti2v):
        log("ancestral sampler:", pipe.use_ancestral_sampler)
    builder = getattr(pipe, stage_attr)._prepared_builder()
    split = args.split
    pp = None
    global A_OWNER, POOL, PAIR
    if args.pp and args.pp_shared_a:
        A_OWNER = _SharedOwner()
        args.pp_streams = True       # lanes must not share one stream once they overlap on card A
    if args.pp:
        import torch.multiprocessing as tmp
        ctx = tmp.get_context("spawn")
        inq, ready, decout = ctx.Queue(), ctx.Queue(), ctx.Queue()
        outqs = [ctx.Queue() for _ in range(args.lanes)]
        env_keep = os.environ.get("HIP_VISIBLE_DEVICES")
        os.environ["HIP_VISIBLE_DEVICES"] = PP_BACK_DEV
        vram_guard()
        procB = ctx.Process(target=back_server, args=(args, inq, outqs, ready, decout), daemon=True)
        procB.start()
        import atexit
        atexit.register(lambda: (procB.terminate(), procB.join(10)) if procB.is_alive() else None)
        os.environ["HIP_VISIBLE_DEVICES"] = env_keep
        pp = dict(inq=inq, outqs=outqs, ready=ready, proc=procB, decout=decout)
        log(f"[A] spawned back-half process pid {procB.pid} on HIP device {PP_BACK_DEV}; loading front half ...")
        ltx = build_split(builder, layer_devfn(r"^transformer_blocks\.(\d+)\.", split, "front")).eval()
        r = ready.get()
        if r[0] != "ok":
            raise SystemExit(f"back-half process failed: {r}")
        nb = len(ltx.transformer_blocks)
        if args.pp_pool:
            POOL = ShmPool()
        install_pp_forward(ltx, split, pp, lambda: lstate)
    else:
        log(f"loading DiT fp8, blocks <{split} on cuda:0, rest on cuda:1 ...")
        ltx = build_split(builder, layer_devfn(r"^transformer_blocks\.(\d+)\.", split)).eval()
        nb = len(ltx.transformer_blocks)
        install_split_forward(ltx, [D0 if i < split else D1 for i in range(nb)], lambda: lstate)
    if args.fa_amd:
        install_fa_amd(ltx)
    x0 = X0Model(ltx).eval()

    class LoraState(threading.local):  # per lane (thread): LoRA on/off + current job
        on = False
        job_has_control = False
        name = None
        calls = 0       # DiT calls in the current stage
        stage_has_lora = False
        lane = 0
        gate = None     # stage-1 DiT calls with control; later calls cut the reference tokens off
        n_ref = 0       # reference tokens appended after the noisy tokens (stage 1)

    lstate = LoraState()
    if ic:
        for lp, ls in ic_loras(args):
            attach_runtime_lora(ltx, lp, ls, lstate)
    if a2v or ti2v:
        attach_runtime_lora(ltx, DLORA, args.dlora_strength, lstate)
    elif args.base == "dev":  # distilled behaviour = dev + distilled LoRA, always on
        attach_runtime_lora(ltx, DLORA, args.dlora_strength, types.SimpleNamespace(on=True))
    if args.compile:
        compile_blocks(ltx)
    install_cfg_cut()
    if pp and args.decode_pair:
        PAIR = PairSched(pp)
        log("paired decode: card B decodes at >= 2 parked, card A decodes while card B decodes")
    log("DiT loaded;", vram())

    def gate_hook(mod, a, kw):
        """Control gating: after `gate` stage-1 calls, LoRA off and noisy<->reference attention masked to 0."""
        lstate.calls += 1
        if lstate.gate is None or not lstate.stage_has_lora or not lstate.job_has_control:
            return None
        if lstate.calls <= lstate.gate:
            return None
        lstate.on = False
        video = a[0] if a else kw["video"]
        T = video.latent.shape[1]
        n = T - lstate.n_ref
        m = video.attention_mask
        m = torch.ones(video.latent.shape[0], T, T, device=video.latent.device, dtype=torch.float32) if m is None else m.clone()
        m[:, :n, n:] = 0
        m[:, n:, :n] = 0
        video = dataclasses.replace(video, attention_mask=m)
        if lstate.calls == lstate.gate + 1:
            log(f"control gated off from DiT call {lstate.calls} (T={T}, ref={lstate.n_ref})")
        if a:
            return (video, *a[1:]), kw
        kw["video"] = video
        return a, kw

    if ic:
        x0.register_forward_pre_hook(gate_hook, with_kwargs=True)

    class OnDevice(torch.nn.Module):
        """Lane on cuda:1: run the shared DiT (non-block params on cuda:0) and hand results back."""

        def __init__(self, inner, dev):
            super().__init__()
            self.inner, self.dev = inner, dev

        def forward(self, video, audio, perturbations):
            with torch.cuda.device(0):
                out = self.inner(move(video, D0), move(audio, D0), move(perturbations, D0))
            return move(out, self.dev)

    emb_dir = Path(args.emb)

    def prepare(p, dev):
        model = x0 if dev == D0 else OnDevice(x0, dev)

        class SplitStage(DiffusionStage):
            def _transformer_ctx(self, **kw):
                lstate.calls = 0
                lstate.stage_has_lora = bool(self._transformer_builder.loras)
                lstate.on = lstate.stage_has_lora and lstate.job_has_control
                log(f"[{dev}] stage ctx: LoRA {'ON' if lstate.on else 'off'}")
                return nullcontext(model)

        for attr in ([stage_attr, "stage_2"] if a2v or ti2v else [stage_attr]):
            st = getattr(p, attr)
            setattr(p, attr, SplitStage(st._transformer_builder, st._dtype, st._device,
                                        quantization=st._quantization, scale_factors=st.video_scale_factors))

        def fake_encoder(prompts, **kw):
            out = []
            for i in range(len(prompts)):
                d = torch.load(emb_dir / f"{lstate.name}{'' if i == 0 else '__neg'}.pt")
                out.append(EmbeddingsProcessorOutput(**{k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                                                        for k, v in d.items()}))
            return out

        p.prompt_encoder = fake_encoder
        if pp and (args.decode_b or args.decode_pair) and hasattr(p, "video_decoder"):
            p.video_decoder = RemoteDecoder(p.video_decoder, pp, lambda: lstate)
        return p

    def lane_pipes(dev, first=None):
        if not mixed:
            return prepare(first or make_pipe(dev)[0], dev)
        return {"a2vid": prepare(first or make_pipe(dev, "a2vid")[0], dev), "ti2v": prepare(make_pipe(dev, "ti2v")[0], dev)}

    lanes = [(lane_pipes(D0, pipe), D0)]
    if args.lanes >= 2 and pp:
        for _ in range(args.lanes - 1):   # every lane drives cuda:0; process B serves cuda:1 (FIFO over lanes)
            lanes.append((lane_pipes(D0), D0))
    elif args.lanes == 2:
        lanes.append((lane_pipes(D1), D1))
    if args.debug and not mixed:
        for i, (lp, _d) in enumerate(lanes):
            install_debug(lp, x0 if i == 0 else torch.nn.Module(), Path(args.out) / "dbg", name_fn=lambda: lstate.name)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for j in jobs:
        if args.only and j["name"] not in args.only.split(","):
            continue
        if (out_dir / f"{j['name']}.mp4").exists() and not args.force:
            log("skip existing", j["name"])
            continue
        todo.append(j)
    if pp and args.stagger and len(lanes) > 1:
        # 10-24: the two takes of a shot (same length) ran side by side in lockstep, so both lanes decoded on card A
        # at once while card B sat idle (0% for 6-8 s at a time); takes apart + a late second lane keep them out of step
        todo = [j for j in todo if not re.search(r"_t\d+$", j["name"])] + [j for j in todo if re.search(r"_t\d+$", j["name"])]
    if args.order == "cost":
        todo = order_jobs(todo, ti2v_default=bool(args.ti2v and not args.a2vid))
        log(f"job order: longest first ({len(todo)} jobs, {todo[0]['frames'] if todo else 0} -> "
            f"{todo[-1]['frames'] if todo else 0} frames)")
    timings, lock, errors = {}, threading.Lock(), []
    t_all = time.time()

    def run_job(p, dev, j):
        if isinstance(p, dict):             # mixed hold: the job's route
            kind = "a2vid" if j.get("audio") else "ti2v"
            p = p[kind]
        else:
            kind = "a2vid" if a2v else "ti2v" if ti2v else None
        j_a2v, j_ti2v = kind == "a2vid", kind == "ti2v"
        mp4 = out_dir / f"{j['name']}.mp4"
        lstate.name = j.get("emb", j["name"])
        lstate.dcalls = 0
        lstate.job = j["name"]
        if j.get("chain_from"):  # long-line chunk: frame 0 = a frame of the previous chunk's output
            import av as _av
            prev = out_dir / f"{j['chain_from']}.mp4"
            png = out_dir / f"{j['name']}_first.png"
            last = out_dir / f"{j['chain_from']}_last.png"   # 10-26: the previous piece's last frame, lossless
            if j.get("overlap", 12) <= 1 and last.exists():
                import shutil as _sh
                _sh.copy(last, png)
            else:
                fr = [f for i, f in enumerate(_av.open(str(prev)).decode(video=0)) if i == j["chain_frame"]][0]
                fr.to_image().save(png)
            j = {**j, "images": [{"path": str(png), "frame": 0, "strength": j.get("chain_strength", 1.0)}] + j.get("images", [])}
        if j.get("lock_end") and j.get("images"):  # camera lock: weak copy of frame 0 at the last frame
            j = {**j, "images": j["images"] + [{"path": j["images"][0]["path"], "frame": j["frames"] - 1,
                                                 "strength": j["lock_end"]}]}
        imgs = [ImageConditioningInput.at(im["path"], im["frame"], im.get("strength", 1.0), crf=im.get("crf"))
                for im in j.get("images", [])]
        kw = dict(prompt=j["prompt"], seed=j.get("seed", 42), height=j["height"], width=j["width"],
                  frame_rate=j.get("fps", 24), images=imgs, num_frames=j.get("frames"))
        ctl = j.get("control")
        if j_a2v:
            lstate.job_has_control = True  # = "LoRA allowed": distilled LoRA on in stage 2 only
            from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
            kw.update(negative_prompt=DEFAULT_NEGATIVE_PROMPT + j.get("neg_extra", ""), num_inference_steps=j.get("steps", a2v_params.num_inference_steps),
                      video_guider_params=dataclasses.replace(a2v_params.video_guider_params, **j.get("guider", {})), audio_path=stereo_audio(j["audio"], out_dir),
                      audio_start_time=j.get("audio_start", 0.0))
            kw["num_frames"] = j.get("frames")
        elif j_ti2v:
            lstate.job_has_control = True  # = "LoRA allowed": distilled LoRA on in stage 2 only
            from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
            kw.update(negative_prompt=DEFAULT_NEGATIVE_PROMPT + j.get("neg_extra", ""),
                      num_inference_steps=j.get("steps", a2v_params.num_inference_steps),
                      video_guider_params=dataclasses.replace(a2v_params.video_guider_params, **j.get("guider", {})),
                      audio_guider_params=dataclasses.replace(a2v_params.audio_guider_params, **j.get("aguider", {})))
        elif ic:
            lstate.job_has_control = bool(ctl)
            c0 = (ctl[0] if isinstance(ctl, list) else ctl) or {}
            lstate.gate = c0.get("gate_steps")
            lstate.n_ref = ((j["frames"] - 1) // 8 + 1) * (j["height"] // 128) * (j["width"] // 128) * (
                len(ctl) if isinstance(ctl, list) else 1)
            ctls = ctl if isinstance(ctl, list) else ([ctl] if ctl else [])
            kw["video_conditioning"] = [(c["path"], c.get("strength", 1.0)) for c in ctls]
            kw["conditioning_attention_strength"] = ctls[0].get("attn", 1.0) if ctls else 1.0
        elif ctl:
            raise SystemExit(f"job {j['name']} has control but --ic-lora not given")
        if j_a2v or j_ti2v:
            # 10-31: all guidance passes (cond + uncond [+ STG]) in ONE DiT call, as the pipeline batches them; its
            # default max_batch_size=1 split them back into sequential B=1 calls, so every step upcast the fp8
            # weights once per pass (11% of a stage-1 call). Talk 121 f: -2.4% card time, PSNR 43.5 dB vs B=1
            kw["max_batch_size"] = MAX_BATCH
        CFG_TL.cut = TALK_CFG_CUT if j_a2v else ACTION_CFG_CUT if j_ti2v else 1.0
        t = time.time()
        tm_reset()
        lstate.defer = None
        if PAIR is not None and not (j.get("chain_from") and j.get("seam_tries") != 0):
            def defer(latent, tiling, dt, gstate, j=j, t=t, kind=kind):
                t_dit, tm = time.time() - t, {**tm_snapshot(), "dit_calls": lstate.dcalls}

                def fin(frames, dec_s, card):
                    frames = frames if frames.ndim == 4 else frames[0]
                    finish_job(j, frames, t_dit, {**tm, "b_decode_s": round(dec_s, 2) if card == "B" else 0.0,
                                                  "a_decode_s": round(dec_s, 2) if card == "A" else 0.0,
                                                  "decode_wait_s": 0.0, "decode_card": card}, kind, dev)
                    with lock:
                        running.discard(j["name"])
                PAIR.park((latent, tiling, dt, gstate, fin))
            lstate.defer = defer
        try:
            frames, j = gen_with_seam_retries(p, kw, j, dev, out_dir)
        except DeferredDecode:
            return "deferred"
        finally:
            lstate.defer = None
        finish_job(j, frames, time.time() - t, {**tm_snapshot(), "dit_calls": lstate.dcalls}, kind, dev)

    def finish_job(j, frames, t_gen, tm, kind, dev):
        """A clip's frames -> mp4 (+ audio), job json, timings (run in the lane, or after a parked decode)."""
        mp4 = out_dir / f"{j['name']}.mp4"
        log(f"[{dev}] {j['name']}: video {tuple(frames.shape)} gen {t_gen:.1f}s ;", vram())
        part = str(mp4) + ".part.mp4"      # atomic: a clip killed mid-write never counts as done (10-21 dry run)
        if j.get("chunk"):                 # a talk piece: its last frame, lossless, starts the next piece (10-26)
            save_last_png(frames, out_dir / f"{j['name']}_last.png")
        save_mp4(frames, part, j.get("fps", 24))
        if j.get("audio"):
            import subprocess
            tmp = str(mp4) + ".noaudio.mp4"
            os.replace(part, tmp)
            # 10-28: ffmpeg's shortest-stream cut the video at a packet boundary - a line's last piece (its audio ending with its
            # picture) lost 4 frames; the audio padded, both cut at the picture's own length
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", str(j.get("audio_start", 0.0)), "-i", j.get("audio_mux", j["audio"]),
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-af", "apad",
                            "-t", f"{frames.shape[0] / j.get('fps', 24):.6f}", part], check=True)
            os.remove(tmp)
        os.replace(part, mp4)
        with lock:
            timings[j["name"]] = {"sec": round(t_gen, 1), "lane": str(dev), "vram": vram(),
                                  "shape": list(frames.shape)}
            (out_dir / f"{j['name']}.json").write_text(json.dumps({**j, **timings[j["name"]]}, indent=1))
        rec = {"job": j["name"], "route": kind or ("ic" if ic else "distilled"), "pp": bool(pp),
               "lanes": args.lanes, "split": split, "w": j["width"], "h": j["height"], "frames": int(frames.shape[0]),
               "wall_s": round(t_gen, 2), **tm}
        if pp:  # card A's non-DiT busy = the job's time on card A minus its DiT front and copies
            rec["a_other_s"] = round(t_gen - tm["wait_b_s"] - tm["a_dit_s"] - tm["a_copy_s"] - tm.get("decode_wait_s", 0), 2)
            rec["a_idle_s"] = round(tm["wait_b_s"], 2)  # this job's lane waiting on B (A may serve the other lane)
        with lock, open(out_dir / "pp_timing.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        del frames

    lane_ids = {}

    def lane(p, dev):
        torch.cuda.set_device(dev)
        with lock:
            lstate.lane = lane_ids.setdefault(threading.get_ident(), len(lane_ids))
        if pp and lstate.lane:
            time.sleep(args.stagger * lstate.lane)      # out of step with lane 0 (see the take order above)
        stream = torch.cuda.Stream(dev) if (pp and args.pp_streams) else None
        with (torch.cuda.stream(stream) if stream is not None else nullcontext()):
            lane_loop(p, dev)

    running: set = set()
    queued = {j["name"] for j in todo}

    def more_jobs() -> bool:
        """--more (10-24): jobs appended while we run (the take judge's retakes, judged on the iGPU as the clips
        land) join the queue on the loaded model; False once --more-done exists and nothing new came."""
        try:
            extra = json.loads(Path(args.more).read_text())
        except (OSError, ValueError):
            extra = []
        new = [x for x in extra if x["name"] not in queued and not (out_dir / f"{x['name']}.mp4").exists()]
        for x in new:
            queued.add(x["name"])
            todo.append(x)
        if new:
            log(f"{len(new)} more job(s) queued: {[x['name'] for x in new]}")
        return bool(new) or not Path(args.more_done).exists()

    def lane_loop(p, dev):
        while True:
            with lock:
                if errors:
                    return
                if not todo and not (args.more and more_jobs()):
                    return
                if time.time() - T0 > args.budget:
                    log("GPU-hold budget reached; lane", dev, "stops;", len(todo), "left")
                    return
                if len(timings) >= 2 and gpu_waiters():   # fairness (10-20 remake): another job waits on our GPU locks -> stop starting jobs
                    log("another GPU job is waiting on our locks; lane", dev, "yields;", len(todo), "left")
                    return
                # a chained chunk waits for its source (10-20: with 2 lanes, 382_c1 started beside c0 and found no
                # c0.mp4): take the first job whose chain source is neither queued nor running
                pend = {x["name"] for x in todo} | running
                k = next((i for i, x in enumerate(todo) if x.get("chain_from") not in pend), None)
                if not todo:
                    j = "wait"                  # --more: the judge may still add some
                elif k is None:
                    j = None
                else:
                    j = todo.pop(k)
                    running.add(j["name"])
            if j in (None, "wait") and PAIR is not None and (it := PAIR.take()) is not None:
                PAIR.decode_a(it, lstate.lane)    # nothing runnable (e.g. a chained piece waits on a parked decode)
                continue
            if j is None:
                time.sleep(2)
                continue
            if j == "wait":
                time.sleep(3)
                continue
            try:
                if pp:
                    with A_OWNER:
                        r = run_job(p, dev, j)
                        torch.cuda.current_stream().synchronize()
                else:
                    r = run_job(p, dev, j)
            except BaseException as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                with lock:
                    errors.append(e)
                    running.discard(j["name"])
                return
            if r != "deferred":            # a parked clip leaves 'running' once its decode has written it
                with lock:
                    running.discard(j["name"])
            gc.collect()

    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    threads = [threading.Thread(target=lane, args=lp) for lp in lanes]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if PAIR is not None:
        PAIR.flush()
        errors.extend(PAIR.err)
    wall = time.time() - t_all
    n = len(timings)
    if args.compile:
        log("[A]", compile_stats())
    log(f"done {n} jobs in {wall:.1f}s wall ({60 * n / max(wall, 1e-9):.2f} jobs/min, lanes={len(lanes)}); peak", vram())
    log(json.dumps(timings, indent=1))
    if pp:
        pp["inq"].put(None)
        r = pp["ready"].get(timeout=120)
        log(f"[B] busy {r[1]:.1f}s of {wall:.1f}s wall ({100 * r[1] / wall:.0f}%), peak {r[2] / 2**30:.1f} GiB; "
            f"[A] owned {A_OWNER.held:.1f}s ({100 * A_OWNER.held / wall:.0f}%)")
        pp["proc"].join(timeout=60)
    if errors:
        raise SystemExit(f"{len(errors)} lane error(s)")


CFG_TL = threading.local()      # per lane: .cut = the share of a guided stage's steps that run the guidance passes


def install_cfg_cut():
    """Guidance interval / CFG truncation (10-31 A/B, user OK): a guided stage runs its guidance passes (uncond,
    STG, modality) on the first CFG_TL.cut of its steps; the later steps run the conditioned pass alone. Talk
    at 0.6: 6 talk lines, SyncNet within +-0.15 of full guidance, card time -17% A / -13% B, wall -8%; 0.4 lost
    a face. Stage 2 (SimpleDenoiser) is unguided and untouched."""
    import ltx_pipelines.utils.denoisers as DN
    if getattr(DN, "_cfg_cut", False):
        return
    orig = DN._guided_denoise

    def off(g):
        return dataclasses.replace(g, params=dataclasses.replace(g.params, cfg_scale=1.0, stg_scale=0.0,
                                                                 modality_scale=1.0))

    def gd(*a, **k):
        n, cut = getattr(CFG_TL, "n", 0), getattr(CFG_TL, "cut", 1.0)
        if n and cut < 1.0 and k["step_index"] >= round(cut * n):
            k["video_guider"], k["audio_guider"] = off(k["video_guider"]), off(k["audio_guider"])
        return orig(*a, **k)
    DN._guided_denoise = gd
    for C in (DN.GuidedDenoiser, DN.FactoryGuidedDenoiser):
        def call(self, transformer, video_state, audio_state, sigmas, step_index, _oc=C.__call__):
            CFG_TL.n = len(sigmas) - 1
            return _oc(self, transformer, video_state, audio_state, sigmas, step_index)
        C.__call__ = call
    DN._cfg_cut = True


def compile_blocks(ltx):
    """--compile: torch.compile each resident block (regional: same-structure blocks share one graph). Fuses the
    norm / modulation / gate / residual elementwise ops around the GEMMs and attention."""
    import torch._dynamo as dyn
    dyn.config.cache_size_limit = 64
    n = 0
    for b in ltx.transformer_blocks:
        p = next(b.parameters(), None)
        if p is None or p.device.type != "cuda":
            continue
        b.compile(dynamic=True)
        n += 1
    log(f"torch.compile on {n} blocks")


def compile_stats():
    import torch._dynamo as dyn
    c = dyn.utils.counters
    return f"compile: {c['stats'].get('unique_graphs', 0)} graphs, {sum(c['graph_break'].values())} graph breaks, {dict(c['recompiles']) if 'recompiles' in c else ''}"


def install_fa_amd(ltx):
    """Unmasked attention -> AMD Triton flash kernel (~/src/attn-lab/fa_amd); masked calls stay on SDPA."""
    sys.path.insert(0, str(Path.home() / "src/attn-lab"))
    from fa_amd import interface_v2 as fa
    from ltx_core.model.transformer.attention import Attention

    @torch.library.custom_op("bookcaster::fa_amd", mutates_args=())
    def fa_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # an opaque op to torch.compile: the block stays one graph around the Triton kernel's Python launcher
        with torch.cuda.device(q.device):
            return fa.fwd(q, k, v, None, None, 0.0, q.shape[-1] ** -0.5, False, -1, -1, 0.0, False)[0]

    @fa_op.register_fake
    def _(q, k, v):
        return torch.empty_like(q)

    def fa_attn(q, k, v, heads):
        b, _, hd = q.shape
        d = hd // heads
        q, k, v = (t.reshape(b, -1, heads, d).contiguous() for t in (q, k.to(q.dtype), v.to(q.dtype)))
        return fa_op(q, k, v).reshape(b, -1, heads * d)

    n = 0
    for m in ltx.modules():
        if isinstance(m, Attention):
            m.attention_function = fa_attn
            n += 1
    log(f"fa_amd attention on {n} attention modules (unmasked path)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "gen"])
    ap.add_argument("jobs")
    ap.add_argument("--emb", default=str(Path(__file__).with_name("emb")))
    ap.add_argument("--out", default=str(Path(__file__).with_name("out")))
    ap.add_argument("--split", type=int, default=None,
                    help="first block on card 2 (blocks < split on card 1). Default per route: a2vid 24, else 17 "
                         "(card 05 also carries all VAE/upsampler work; pp2 data: A ~22.7 s vs B 18.6 s per job at 20)")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ic-lora", default="", help="IC-LoRA safetensors -> ICLoraPipeline (jobs may carry control)")
    ap.add_argument("--lora-strength", default="1.0", help="one value, or one per --ic-lora (comma list)")
    ap.add_argument("--dev", action="store_true", help="encode: use the dev checkpoint's embeddings connector")
    ap.add_argument("--a2vid", action="store_true", help="A2Vid two-stage: dev DiT + CFG/STG stage 1, distilled LoRA stage 2")
    ap.add_argument("--ti2v", action="store_true", help="image-to-video two-stage: dev DiT + CFG/STG stage 1, distilled LoRA stage 2")
    ap.add_argument("--dlora-strength", type=float, default=0.8)
    ap.add_argument("--lanes", type=int, default=1, choices=[1, 2, 3],
                    help="jobs in flight: 2 = lane 2 on cuda:1 (no --pp); with --pp all lanes on card A (3 needs --pp)")
    ap.add_argument("--pp-shared-a", action="store_true",
                    help="--pp: lanes overlap on card A (per-lane streams, no exclusive owner); prove identity first")
    ap.add_argument("--base", default="distilled", choices=["distilled", "dev"],
                    help="dev = dev DiT + distilled LoRA (always on) for distilled/IC jobs: one resident model for all job types")
    ap.add_argument("--hashlog", action="store_true", help="log bit fingerprints of the DiT activations per call")
    ap.add_argument("--pp-streams", action="store_true", help="per-lane CUDA streams in process A (default: one stream)")
    ap.add_argument("--pp", action="store_true", help="process-per-card pipelining (process B = back half on card 2)")
    ap.add_argument("--blaslt", action="store_true", help="prefer hipBLASLt for GEMMs")
    ap.add_argument("--compile", action="store_true",
                    help=f"torch.compile the DiT blocks (holds of {COMPILE_MIN_JOBS}+ jobs; 10-31: -18%% DiT time)")
    ap.add_argument("--fa-amd", action="store_true", help="fa_amd attention instead of SDPA")
    ap.add_argument("--miopen", action="store_true", help="re-enable MIOpen convs (BROKEN on gfx1100)")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="--pp: lane k starts k*S s late, takes ordered apart, so lanes don't decode on card A together")
    ap.add_argument("--pp-pool", action="store_true",
                    help="--pp: reused pinned shared-memory buffers for the activations between the processes")
    ap.add_argument("--order", choices=["plan", "cost"], default="plan",
                    help="job order: the plan's, or longest estimated card time first (order_jobs)")
    ap.add_argument("--decode-pair", action="store_true",
                    help="decodes parked and paired across both cards (PairSched; implies a decoder on each card)")
    ap.add_argument("--decode-b", action="store_true",
                    help="--pp: VAE decode on process B's card (B keeps a decoder loaded)")
    ap.add_argument("--more", default="", help="a jobs file read again whenever the queue runs dry")
    ap.add_argument("--more-done", default="", help="--more: this file existing = no more jobs will come")
    ap.add_argument("--budget", type=float, default=780.0, help="stop starting new jobs after this many s")
    args = ap.parse_args()
    if args.split is None:
        args.split = 24 if args.a2vid or args.ti2v else 17
    if args.pp:  # process A sees only the first card; process B gets the second
        global PP_BACK_DEV
        devs = os.environ.get("HIP_VISIBLE_DEVICES", "0,1").split(",")
        os.environ["HIP_VISIBLE_DEVICES"], PP_BACK_DEV = devs[0], devs[1]
    ram_guard()
    jobs = json.loads(Path(args.jobs).read_text())
    log("MemAvailable", f"{mem_avail_gib():.1f} GiB; devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    (cmd_encode if args.cmd == "encode" else cmd_gen)(args, jobs)


if __name__ == "__main__":
    main()
