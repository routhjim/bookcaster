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
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import safetensors  # noqa: E402
import torch  # noqa: E402

M = Path.home() / "models" / "ltx-2.5"
DIT = str(M / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors")
TE = str(M / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
VAE = str(M / "vae/ltx-2.5-video-vae-conv-bf16.safetensors")
AVAE = str(M / "vae/ltx-2.5-audio-vae-bf16.safetensors")
UPS = str(M / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")
DUR = str(M / "model_patches/ltx-2.5-duration-head-bf16.safetensors")
DIT_DEV = str(M / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
DLORA = str(M / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors")
D0, D1 = torch.device("cuda:0"), torch.device("cuda:1")
T0 = time.time()


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
    for d in (0, 1):
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


def install_split_forward(ltx, devs):
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
        for a, b, d in runs:
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
        return move(video, D0), move(audio, D0)

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


def gen_with_seam_retries(p, kw: dict, j: dict, dev, out_dir: Path):
    """Run the pipeline for job j; a chained chunk is re-rolled (seed + 1000) up to 3 tries while its seam check
    fails. All tries failing keeps the smoothest try and flags it in the job (j["seam_flag"]) instead of raising
    (an UnboundLocalError on `frames` killed whole talk holds, 10-20). -> (frames, j)"""
    import torch
    best = None                 # (seam step, frames): every try failing keeps the best one (was UnboundLocalError)
    for attempt in range(3 if j.get("chain_from") else 1):
        with torch.inference_mode():
            res = p(**kw)
            vids = [v for v in res.video]
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)
        frames = torch.cat([v if v.ndim == 4 else v[0] for v in vids], 0)
        del res, vids
        if not j.get("chain_from"):
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
    if best is not None:        # all tries failed the seam check: keep the smoothest, flag it, the hold goes on
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
    import re as _re
    m = _re.search(r"step ([0-9.]+)", info or "")
    return float(m.group(1)) if m else float("inf")


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


def install_debug(pipe, x0, ddir):
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
            torch.save(latent.detach().cpu(), ddir / f"{self.tag}_in.pt")
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


def attach_runtime_lora(ltx, path, strength, state):
    """Unfused LoRA: forward hooks add strength * (x A^T) B^T on each target Linear, on that Linear's card.
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

            def hook(mod, inp, out, A=A, B=B):
                if not state.on:
                    return out
                x = inp[0]
                return out + torch.nn.functional.linear(torch.nn.functional.linear(x.to(A.dtype), A), B).to(out.dtype)

            m.register_forward_hook(hook)
            n += 1
    log(f"runtime LoRA {Path(path).name} x{strength}: {n} linears hooked, {nbytes / 2**20:.0f} MiB, "
        f"missing {len(missing)} {missing[:5]}, unpaired {len(pend)}")
    return n


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
    if not args.miopen:  # MIOpen conv3d on gfx1100: VAE round trip 13.8 dB + 60x slower; off = 32.3 dB (= CPU)
        torch.backends.cudnn.enabled = False
        log("MIOpen (cudnn backend) disabled")
    if args.blaslt:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        log("preferred BLAS: hipBLASLt")
    paths = ModelPaths.from_split(transformer_path=DIT, text_encoder_path=TE, video_vae_path=VAE,
                                  audio_vae_path=AVAE, duration_head_path=DUR)
    ic = bool(args.ic_lora)
    lora = LoraPathStrengthAndSDOps(args.ic_lora, args.lora_strength, LTXV_LORA_COMFY_RENAMING_MAP) if ic else None

    a2v = bool(args.a2vid)
    if a2v:
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
        from ltx_pipelines.utils.constants import detect_params
        a2v_params = detect_params(DIT_DEV)
        paths = ModelPaths.from_split(transformer_path=DIT_DEV, text_encoder_path=TE, video_vae_path=VAE,
                                      audio_vae_path=AVAE, duration_head_path=DUR)
        dlora = LoraPathStrengthAndSDOps(DLORA, args.dlora_strength, LTXV_LORA_COMFY_RENAMING_MAP)
    # --ti2v: image-to-video on the same two stages as A2Vid (dev DiT + CFG/STG at half size, distilled LoRA
    # refine), with a negative prompt (10-23: the distilled action clips melted the dragon's head at contact)
    ti2v = bool(getattr(args, "ti2v", False))
    if ti2v:
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.constants import detect_params
        a2v_params = detect_params(DIT_DEV)
        paths = ModelPaths.from_split(transformer_path=DIT_DEV, text_encoder_path=TE, video_vae_path=VAE,
                                      audio_vae_path=AVAE, duration_head_path=DUR)
        dlora = LoraPathStrengthAndSDOps(DLORA, args.dlora_strength, LTXV_LORA_COMFY_RENAMING_MAP)

    def make_pipe(dev):
        if ti2v:
            return TI2VidTwoStagesPipeline(model_paths=paths, distilled_lora=[dlora], spatial_upsampler_path=UPS,
                                           loras=[], device=dev, quantization=build_policy(DIT_DEV)), "stage_1"
        if a2v:
            return A2VidPipelineTwoStage(model_paths=paths, distilled_lora=[dlora], spatial_upsampler_path=UPS,
                                         loras=[], device=dev, quantization=build_policy(DIT_DEV)), "stage_1"
        if ic:
            return ICLoraPipeline(model_paths=paths, spatial_upsampler_path=UPS, loras=[lora], device=dev,
                                  quantization=build_policy(DIT)), "_diffusion_stage"
        return DistilledPipeline(model_paths=paths, spatial_upsampler_path=UPS, loras=(), device=dev,
                                 quantization=build_policy(DIT)), "stage"

    pipe, stage_attr = make_pipe(D0)
    if ic:
        log("IC-LoRA pipeline; reference downscale", pipe.reference_downscale_factor,
            "temporal", pipe.reference_temporal_scale_factor)
    elif not (a2v or ti2v):
        log("ancestral sampler:", pipe.use_ancestral_sampler)
    builder = getattr(pipe, stage_attr)._prepared_builder()
    split = args.split
    log(f"loading DiT fp8, blocks <{split} on cuda:0, rest on cuda:1 ...")
    ltx = build_split(builder, layer_devfn(r"^transformer_blocks\.(\d+)\.", split)).eval()
    nb = len(ltx.transformer_blocks)
    install_split_forward(ltx, [D0 if i < split else D1 for i in range(nb)])
    if args.fa_amd:
        install_fa_amd(ltx)
    x0 = X0Model(ltx).eval()

    class LoraState(threading.local):  # per lane (thread): LoRA on/off + current job
        on = False
        job_has_control = False
        name = None
        calls = 0       # DiT calls in the current stage
        stage_has_lora = False
        gate = None     # stage-1 DiT calls with control; later calls cut the reference tokens off
        n_ref = 0       # reference tokens appended after the noisy tokens (stage 1)

    lstate = LoraState()
    if ic:
        attach_runtime_lora(ltx, args.ic_lora, args.lora_strength, lstate)
    if a2v or ti2v:
        attach_runtime_lora(ltx, DLORA, args.dlora_strength, lstate)
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
        return p

    lanes = [(prepare(pipe, D0), D0)]
    if args.lanes == 2:
        lanes.append((prepare(make_pipe(D1)[0], D1), D1))
    if args.debug:
        install_debug(pipe, x0, Path(args.out) / "dbg")
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
    timings, lock, errors = {}, threading.Lock(), []
    t_all = time.time()

    def run_job(p, dev, j):
        mp4 = out_dir / f"{j['name']}.mp4"
        lstate.name = j.get("emb", j["name"])
        if j.get("chain_from"):  # long-line chunk: frame 0 = a frame of the previous chunk's output
            import av as _av
            prev = out_dir / f"{j['chain_from']}.mp4"
            fr = [f for i, f in enumerate(_av.open(str(prev)).decode(video=0)) if i == j["chain_frame"]][0]
            png = out_dir / f"{j['name']}_first.png"
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
        if a2v:
            lstate.job_has_control = True  # = "LoRA allowed": distilled LoRA on in stage 2 only
            from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
            kw.update(negative_prompt=DEFAULT_NEGATIVE_PROMPT + j.get("neg_extra", ""), num_inference_steps=j.get("steps", a2v_params.num_inference_steps),
                      video_guider_params=dataclasses.replace(a2v_params.video_guider_params, **j.get("guider", {})), audio_path=stereo_audio(j["audio"], out_dir),
                      audio_start_time=j.get("audio_start", 0.0))
            kw["num_frames"] = j.get("frames")
        elif ti2v:
            lstate.job_has_control = True  # = "LoRA allowed": distilled LoRA on in stage 2 only
            from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
            kw.update(negative_prompt=DEFAULT_NEGATIVE_PROMPT + j.get("neg_extra", ""),
                      num_inference_steps=j.get("steps", a2v_params.num_inference_steps),
                      video_guider_params=dataclasses.replace(a2v_params.video_guider_params, **j.get("guider", {})),
                      audio_guider_params=a2v_params.audio_guider_params)
        elif ic:
            lstate.job_has_control = bool(ctl)
            lstate.gate = (ctl or {}).get("gate_steps")
            lstate.n_ref = ((j["frames"] - 1) // 8 + 1) * (j["height"] // 128) * (j["width"] // 128)
            kw["video_conditioning"] = [(ctl["path"], ctl.get("strength", 1.0))] if ctl else []
            kw["conditioning_attention_strength"] = (ctl or {}).get("attn", 1.0)
        elif ctl:
            raise SystemExit(f"job {j['name']} has control but --ic-lora not given")
        t = time.time()
        frames, j = gen_with_seam_retries(p, kw, j, dev, out_dir)
        t_gen = time.time() - t
        log(f"[{dev}] {j['name']}: video {tuple(frames.shape)} gen {t_gen:.1f}s ;", vram())
        save_mp4(frames, str(mp4), j.get("fps", 24))
        if j.get("audio"):
            import subprocess
            tmp = str(mp4) + ".noaudio.mp4"
            os.replace(mp4, tmp)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", str(j.get("audio_start", 0.0)), "-i", j.get("audio_mux", j["audio"]),
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(mp4)], check=True)
            os.remove(tmp)
        with lock:
            timings[j["name"]] = {"sec": round(t_gen, 1), "lane": str(dev), "vram": vram(),
                                  "shape": list(frames.shape)}
            (out_dir / f"{j['name']}.json").write_text(json.dumps({**j, **timings[j["name"]]}, indent=1))
        del frames

    def lane(p, dev):
        torch.cuda.set_device(dev)
        while True:
            with lock:
                if errors or not todo:
                    return
                if time.time() - T0 > args.budget:
                    log("GPU-hold budget reached; lane", dev, "stops;", len(todo), "left")
                    return
                if len(timings) >= 2 and gpu_waiters():   # fairness (10-20 remake): another job waits on our GPU locks -> stop starting jobs
                    log("another GPU job is waiting on our locks; lane", dev, "yields;", len(todo), "left")
                    return
                j = todo.pop(0)
            try:
                run_job(p, dev, j)
            except BaseException as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                with lock:
                    errors.append(e)
                return
            gc.collect()

    for d in (0, 1):
        torch.cuda.reset_peak_memory_stats(d)
    threads = [threading.Thread(target=lane, args=lp) for lp in lanes]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t_all
    n = len(timings)
    log(f"done {n} jobs in {wall:.1f}s wall ({60 * n / max(wall, 1e-9):.2f} jobs/min, lanes={len(lanes)}); peak", vram())
    log(json.dumps(timings, indent=1))
    if errors:
        raise SystemExit(f"{len(errors)} lane error(s)")


def install_fa_amd(ltx):
    """Unmasked attention -> AMD Triton flash kernel (~/src/attn-lab/fa_amd); masked calls stay on SDPA."""
    sys.path.insert(0, str(Path.home() / "src/attn-lab"))
    from fa_amd import interface_v2 as fa
    from ltx_core.model.transformer.attention import Attention

    def fa_attn(q, k, v, heads):
        b, _, hd = q.shape
        d = hd // heads
        q, k, v = (t.reshape(b, -1, heads, d).contiguous() for t in (q, k, v))
        with torch.cuda.device(q.device):
            out = fa.fwd(q, k.to(q.dtype), v.to(q.dtype), None, None, 0.0, d ** -0.5, False, -1, -1, 0.0, False)[0]
        return out.reshape(b, -1, heads * d)

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
    ap.add_argument("--split", type=int, default=20)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ic-lora", default="", help="IC-LoRA safetensors -> ICLoraPipeline (jobs may carry control)")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--dev", action="store_true", help="encode: use the dev checkpoint's embeddings connector")
    ap.add_argument("--ti2v", action="store_true", help="image-to-video two-stage: dev DiT + CFG/STG stage 1, distilled LoRA stage 2")
    ap.add_argument("--a2vid", action="store_true", help="A2Vid two-stage: dev DiT + CFG/STG stage 1, distilled LoRA stage 2")
    ap.add_argument("--dlora-strength", type=float, default=0.8)
    ap.add_argument("--lanes", type=int, default=1, choices=[1, 2], help="2 = two jobs in flight (lane 2 on cuda:1)")
    ap.add_argument("--blaslt", action="store_true", help="prefer hipBLASLt for GEMMs")
    ap.add_argument("--fa-amd", action="store_true", help="fa_amd attention instead of SDPA")
    ap.add_argument("--miopen", action="store_true", help="re-enable MIOpen convs (BROKEN on gfx1100)")
    ap.add_argument("--budget", type=float, default=780.0, help="stop starting new jobs after this many s")
    args = ap.parse_args()
    ram_guard()
    jobs = json.loads(Path(args.jobs).read_text())
    log("MemAvailable", f"{mem_avail_gib():.1f} GiB; devices", [torch.cuda.get_device_name(i) for i in (0, 1)])
    (cmd_encode if args.cmd == "encode" else cmd_gen)(args, jobs)


if __name__ == "__main__":
    main()
