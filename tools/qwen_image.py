"""Draw a list of pictures with Qwen-Image-2.1 + the 4-step Lightning LoRA (+ the Fun ControlNet Union).

A drop-in for tools/illustrate.py (FLUX.2 klein): the same jobs, the same --serve protocol, the same
JSON lines out, so video.DrawPool can launch either (BOOKCASTER_DRAWER=qwen). Licence: Qwen Research
(non-commercial). Needs its own venv (git diffusers, transformers>=5.17):

    ~/tts-ab/.venv-qwenimage/bin/python -I tools/qwen_image.py jobs.json OUT_DIR
    ... -I tools/qwen_image.py --serve          (jobs as JSON lines on stdin, the model stays loaded)

jobs: {"id": "p001", "prompt": "...", "seed": 1, "path": "/abs/out.png", "size": [w, h],
       "refs": ["/abs/sheet.png", ...]   up to 10 (the model's limit): character/outfit/place sheets,
                                         read by the vision encoder AND put in front of the picture as
                                         latent tokens (Qwen's multi-reference input),
       "control": {"image": "/abs/map.png", "type": "pose"|"depth"|"canny", "scale": 0.8},
       "loras": [["/abs/char.safetensors", 1.0, [x0, y0, x1, y1]], ...]   optional per-job LoRAs
                                         (tools/charlora.py), applied at run time on top of the merged
                                         Lightning LoRA (JobLoras); the optional 0-1 box = regional: the LoRA
                                         acts on the picture's tokens only there (no bleed onto others),
       "init": {"image": "/abs/start.png", "strength": 0.7}}   optional img2img: the schedule is scaled by
                                         strength and the init latents noised to the (shifted) first sigma
                                         structure control through the ControlNet Union; the image is a
                                         ready-made map (DWPose skeleton, depth map) or, for "canny", any
                                         picture (its edges are taken here unless "prepared": true).
One JSON line per picture: {"id", "refs", "seconds", "peak_gb", ...} or {"id", "error"}.

Memory on one 24 GB XTX (measured 10-19, nothing else on the card). The Qwen3-VL-8B encoder (LM int8,
vision tower bf16: 8.1 GiB) and the bf16 DiT (13.3 GiB) do not fit together with room to draw (OOM), so:
  --serve     both stay loaded, the DiT stored as int8 (6.6 GiB; same pictures as bf16 up to fine detail,
              steps 6% slower): 15.5 GiB loaded, 18-19.6 GiB peak with refs, 21.7 GiB with the control
              branch (int8, 3.5 GiB, loaded by the first job that asks for it); device peak 23.1 GB.
  jobs file   every prompt is encoded first, the encoder freed, then the bf16 DiT draws (14 GiB loaded,
              16.5-18 GiB peak, 20 GiB with control).
A CPU encoder was not an option: the drawer's RAM is capped at 8 GB by video.memory_scope(), and the models
disk reads ~0.8 GB/s, so swapping the encoder in per prompt would cost ~20 s a keyframe.

ROCm notes (see eval/qwen_vs_klein/NOTES.md): attention goes through AMD's Triton flash kernel (fa_amd)
where unmasked; GEMMs through hipBLASLt (+20%); the VAE decodes in 512 px tiles (with MIOpen off a conv
asked 6.4 GiB; with MIOpen on, each new shape cost ~10 min of tuning); the vision patch embedding is a
matmul (as a Conv3d it went to MIOpen: 56 s for two references); fp8 weights gave NaN under
expandable_segments, so quantized weights are int8.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOME = Path.home()
MODEL = Path(os.environ.get("QWEN_IMAGE_MODEL", HOME / "models" / "qwen-image-2.1")).expanduser()
LORA = Path(os.environ.get("QWEN_IMAGE_LORA", HOME / "models" / "qwen-image-2.1-lightning"
                           / "pytorch_lora_weights.safetensors")).expanduser()
CONTROL = Path(os.environ.get("QWEN_IMAGE_CONTROL", HOME / "models" / "qwen-image-2.1-controlnet-union"
                              / "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors")).expanduser()
ENC_Q = os.environ.get("QWEN_IMAGE_ENC", "int8")       # bf16 | int8 | fp8 | int4: the encoder's LM weights
VIS_Q = os.environ.get("QWEN_IMAGE_VIS", "bf16")       # bf16 | fp8: the encoder's vision tower
DIT_Q = os.environ.get("QWEN_IMAGE_DIT", "auto")       # auto | bf16 | int8: the 7B DiT (auto: int8 under --serve)
CTRL_Q = os.environ.get("QWEN_IMAGE_CTRL", "int8")     # bf16 | int8: the control branch
REF_RES = int(os.environ.get("QWEN_IMAGE_REF_RES", "640"))   # refs are shrunk to this many pixels squared
MAX_REFS = int(os.environ.get("QWEN_IMAGE_MAX_REFS", "10"))  # the model's documented limit
KV_CACHE = os.environ.get("QWEN_IMAGE_KV", "1") == "1"       # cache the prompt+refs keys after step 1
FA_AMD = Path(os.environ.get("QWEN_IMAGE_FA", HOME / "src" / "attn-lab")).expanduser()
SIGMAS = [1.0, 0.75, 0.5, 0.25]                        # the Lightning LoRA's 4-step training schedule


def log(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


# --- weight-only quantized layers -----------------------------------------------------------------

class QLinear(nn.Module):
    """A Linear stored as int8 or fp8 (one scale per output row) or int4 (asymmetric, groups of 64
    along the input) and widened to the activation dtype at each call. The widened copy lives only
    for that call (largest: 12288x4096 bf16 = 100 MB).

    Measured on the encoder against bf16 (10-19, one 97-token prompt and one with 2 references):
    int8 cos 0.998 text / 0.993 image tokens; fp8 0.996 / 0.989; int4-g64 0.93 / 0.81 (unusable).
    The vision tower must stay bf16 (fp8 there: image tokens cos 0.92). And fp8 gave all-NaN
    embeddings under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (what video.py launches the
    drawer with), finite without it; int8 is fine either way, so nothing here uses fp8 by default."""

    GROUP = 64

    def __init__(self, lin: nn.Linear, mode: str):
        super().__init__()
        self.in_features, self.out_features, self.mode = lin.in_features, lin.out_features, mode
        self.bias = lin.bias
        self.register_buffer("q", None)
        self.register_buffer("s", None)
        self.register_buffer("z", None)

    @torch.no_grad()
    def set_weight(self, w: torch.Tensor) -> None:
        w = w.float()
        if self.mode == "fp8":
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448
            # clamp: w/s can round just past 448, and e4m3fn has no inf (overflow = NaN)
            self.q, self.s = (w / s).clamp_(-448, 448).to(torch.float8_e4m3fn), s.to(torch.bfloat16)
        elif self.mode == "int8":
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
            self.q, self.s = (w / s).round_().clamp_(-127, 127).to(torch.int8), s.to(torch.bfloat16)
        elif self.mode == "int4":
            g = w.view(w.shape[0], -1, self.GROUP)
            lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
            s = ((hi - lo) / 15).clamp_min(1e-8)
            q = ((g - lo) / s).round_().clamp_(0, 15).to(torch.uint8).view(w.shape[0], -1)
            self.q = q[:, 0::2] | (q[:, 1::2] << 4)
            self.s, self.z = s.to(torch.bfloat16), lo.to(torch.bfloat16)
        else:
            raise ValueError(self.mode)

    def weight_as(self, dtype: torch.dtype) -> torch.Tensor:
        if self.mode in ("fp8", "int8"):
            return self.q.to(dtype) * self.s.to(dtype)
        q = torch.stack([self.q & 15, self.q >> 4], -1).view(self.out_features, -1, self.GROUP)
        return (q.to(dtype) * self.s.to(dtype) + self.z.to(dtype)).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_as(x.dtype), self.bias)


class QEmbedding(nn.Module):
    """The encoder's 151936-row token table as int8 rows (0.6 instead of 1.2 GB)."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.num_embeddings, self.embedding_dim = emb.num_embeddings, emb.embedding_dim
        self.register_buffer("q", None)
        self.register_buffer("s", None)

    @torch.no_grad()
    def set_weight(self, w: torch.Tensor) -> None:
        w = w.float()
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        self.q, self.s = (w / s).round_().clamp_(-127, 127).to(torch.int8), s.to(torch.bfloat16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.q[ids].to(torch.bfloat16) * self.s[ids]


class NoHead(nn.Module):
    """Stands in for lm_head: only the hidden states are wanted (saves 1.2 GB and a logits tensor)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :1]


def quantize_linears(model: nn.Module, prefixes: tuple[str, ...], mode: str) -> None:
    if mode == "bf16":
        return
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name.startswith(prefixes):
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, QLinear(mod, mode))


# --- loading straight to the GPU, checked ----------------------------------------------------------

def send(key: str, t: torch.Tensor, device: str) -> torch.Tensor:
    """One tensor to the GPU through ordinary memory, compared on arrival: copies to the dock-attached
    XTX are occasionally corrupted, and the pinned-memory paths fault there (see illustrate.load_direct)."""
    t = t.to(torch.bfloat16).clone()
    for _ in range(4):
        g = t.to(device)
        if torch.equal(g.cpu(), t):
            return g
        log(f"weight {key} arrived corrupted; resending")
    raise RuntimeError(f"weight {key} keeps arriving corrupted")


def fill(model: nn.Module, files: list[Path], device: str, rename=lambda k: k, extra=None) -> None:
    """Fill a meta-device model from safetensors, one tensor at a time (no CPU copy of the model:
    the drawer's RAM is capped at 8 GB). extra(key, gpu_tensor) may change a weight (LoRA merge)."""
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    for shard in files:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                name = rename(k)
                if name is None:
                    continue
                g = send(k, f.get_tensor(k), device)
                if extra:
                    g = extra(name, g)
                owner = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                if isinstance(owner, (QLinear, QEmbedding)) and name.endswith(".weight"):
                    owner.set_weight(g)
                else:
                    set_module_tensor_to_device(model, name, device, dtype=torch.bfloat16, value=g)
                del g
    left = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if left:
        raise RuntimeError(f"{len(left)} weights not in the checkpoint, e.g. {left[:3]}")


# --- the three parts -------------------------------------------------------------------------------

def load_encoder(device: str):
    """Qwen3-VL-8B (the prompt and the reference pictures together) as a pipeline that only encodes."""
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Pipeline
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    cfg = AutoConfig.from_pretrained(MODEL / "text_encoder")
    cfg._attn_implementation = "sdpa"
    with init_empty_weights():
        te = Qwen3VLForConditionalGeneration(cfg)
    te.lm_head = NoHead()
    if ENC_Q != "bf16":
        lm = te.model.language_model
        lm.embed_tokens = QEmbedding(lm.embed_tokens)
        quantize_linears(te, ("model.language_model.layers.",), ENC_Q)
        quantize_linears(te, ("model.visual.blocks.",), VIS_Q)
    files = sorted((MODEL / "text_encoder").glob("*.safetensors"))
    fill(te, files, device, rename=lambda k: None if k.startswith("lm_head.") else k)
    te.to(device).eval()
    # The vision patch embedding is a Conv3d whose kernel equals its stride, i.e. one matmul over each
    # flattened patch. As a conv on ROCm it went to MIOpen (a first-use search: 56 s for two references)
    # or, with MIOpen off, to the slow direct conv: do the matmul.
    pe = te.model.visual.patch_embed
    w2 = pe.proj.weight.reshape(pe.embed_dim, -1)
    pe.forward = lambda x: F.linear(x.reshape(-1, w2.shape[1]).to(w2.dtype), w2, pe.proj.bias)
    proc = Qwen3VLProcessor.from_pretrained(MODEL / "processor")
    return QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=te, processor=proc, transformer=None)


def load_drawer(device: str, processor, dit_q: str = "bf16"):
    """The DiT (Lightning LoRA merged in, bf16) + VAE + the Lightning scheduler, as a pipeline
    whose helpers (latent packing, VAE encode) the draw loop uses."""
    from accelerate import init_empty_weights
    from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline,
                           QwenImage21Transformer2DModel)
    from safetensors import safe_open

    with init_empty_weights():
        tr = QwenImage21Transformer2DModel.from_config(QwenImage21Transformer2DModel.load_config(MODEL / "transformer"))
    quantize_linears(tr, ("transformer_blocks.",), dit_q)
    lora = {}
    if LORA.exists():
        meta = {}
        with safe_open(str(LORA), framework="pt", device="cpu") as f:
            meta = json.loads((f.metadata() or {}).get("lora_adapter_metadata", "{}"))
            for k in f.keys():
                base, part = k.removeprefix("transformer.").split(".lora.")
                lora.setdefault(base + ".weight", {})[part.split(".")[0]] = f.get_tensor(k)
        scale = float(meta.get("transformer.lora_alpha", 16)) / float(meta.get("transformer.r", 16))

        def merge(name, g):
            ab = lora.pop(name, None)
            if ab is None:
                return g
            up, down = ab["up"].to(g.device, torch.float32), ab["down"].to(g.device, torch.float32)
            return (g.float() + scale * (up @ down)).to(torch.bfloat16)
    else:
        log(f"no Lightning LoRA at {LORA}: drawing with the base model (needs ~40 steps)")
        merge = None
    fill(tr, sorted((MODEL / "transformer").glob("*.safetensors")), device, extra=merge)
    if lora:
        raise RuntimeError(f"{len(lora)} LoRA weights matched nothing, e.g. {list(lora)[:2]}")
    tr.to(device).eval()
    vae = AutoencoderKLQwenImage21.from_pretrained(MODEL / "vae", torch_dtype=torch.bfloat16).to(device).eval()
    # Tiled: with MIOpen off (as illustrate.py runs) a conv falls back to im2col, which asked 6.4 GiB at
    # 1536x864; with MIOpen on, each new conv shape cost ~10 min of tuning and the decode peaked at 13.6 GiB.
    # Tiles of VAE_TILE px (default 512, stride 3/4): ~1.6 s and ~3 GiB, no tuning (10-19).
    tile = int(os.environ.get("QWEN_IMAGE_VAE_TILE", "512"))
    vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                      tile_sample_stride_height=tile * 3 // 4, tile_sample_stride_width=tile * 3 // 4)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL / "scheduler")
    # The Lightning card's schedule: dynamic exponential shift, no terminal stretch.
    sched = FlowMatchEulerDiscreteScheduler.from_config(sched.config, shift_terminal=None)
    return QwenImage21Pipeline(scheduler=sched, vae=vae, text_encoder=None, processor=processor, transformer=tr)


class ControlBlock(nn.Module):
    """One block of the VACE-style control chain (VideoX-Fun QwenImage21ControlTransformerBlock)."""

    def __init__(self, first: bool):
        super().__init__()
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

        self.block = QwenImage21TransformerBlock(dim=4096, num_attention_heads=32, attention_head_dim=128,
                                                 mlp_ratio=3, eps=1e-6)
        self.before_proj = nn.Linear(4096, 4096) if first else None
        self.after_proj = nn.Linear(4096, 4096)


class Control(nn.Module):
    """The Fun ControlNet Union branch: control_img_in + 16 control blocks, each adding a skip after
    base block 0, 2, ..., 30. The chain only reads the base stream at block 0, so its blocks run
    interleaved with the base blocks (one live skip instead of 16 stacked, ~1 GB less at 8k tokens)."""

    LAYERS = list(range(0, 32, 2))

    def __init__(self):
        super().__init__()
        self.control_img_in = nn.Linear(129, 4096)
        self.control_blocks = nn.ModuleList([ControlBlock(i == 0) for i in range(len(self.LAYERS))])
        self.context, self.scale, self.c, self.kw = None, 1.0, None, None
        self.extra = []          # more (context, scale) pairs: multi-control (depth + canny), summed skips

    def attach(self, tr: nn.Module) -> None:
        tr.transformer_blocks[0].register_forward_pre_hook(self._start, with_kwargs=True)
        for n, layer in enumerate(self.LAYERS):
            tr.transformer_blocks[layer].register_forward_hook(
                lambda mod, args, kwargs, out, n=n: self._skip(n, out), with_kwargs=True)

    def _start(self, mod, args, kwargs):
        if self.context is None:
            return None
        x = kwargs["hidden_states"]
        self.kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        self.kw.update(layer_cache=None, kv_cache_mode=None, cache_write_slice=None)
        self.c = []
        for ctx, _ in [(self.context, self.scale)] + self.extra:
            cj = torch.zeros_like(x)
            cj[:, -ctx.shape[1]:] = self.control_img_in(ctx)                    # target tokens are last
            self.c.append(self.control_blocks[0].before_proj(cj) + x)
        return None

    def _skip(self, n, out):
        if self.context is None:
            return out
        cb = self.control_blocks[n]
        for i, (_, sc) in enumerate([(self.context, self.scale)] + self.extra):
            self.c[i] = cb.block(self.c[i], **self.kw)
            out = out + cb.after_proj(self.c[i]) * sc
        if n == len(self.LAYERS) - 1:
            self.c = self.kw = None
        return out


def load_control(tr: nn.Module, device: str) -> Control:
    from accelerate import init_empty_weights

    with init_empty_weights():
        ctl = Control()
    quantize_linears(ctl, ("control_blocks.",), CTRL_Q)

    def rename(k):          # checkpoint: control_blocks.N.attn... -> our control_blocks.N.block.attn...
        p = k.split(".")
        if p[0] == "control_blocks" and p[2] not in ("before_proj", "after_proj"):
            return ".".join(p[:2] + ["block"] + p[2:])
        return k
    fill(ctl, [CONTROL], device, rename=rename)
    ctl.to(device).eval()
    ctl.attach(tr)
    return ctl


# --- per-job LoRAs (character LoRAs from tools/charlora.py) -----------------------------------------

class JobLoras:
    """LoRAs applied per job at run time (not merged), so one loaded drawer serves jobs with different
    character LoRAs: job "loras": [[path, scale], ...]. Each LoRA file (peft/diffusers keys
    'transformer.<module>.lora_A.weight' / 'lora_B', alpha in the 'lora_adapter_metadata') is read once
    and kept on the GPU (rank 16 on attention + MLP of 32 blocks: ~0.2 GB bf16). A forward hook on each
    targeted Linear / QLinear adds scale * alpha/r * B(A(x)) while the job runs; nothing otherwise."""

    def __init__(self, model: nn.Module, device: str):
        self.model, self.device = model, device
        self.files: dict = {}          # path -> {module name: (A, B, s)}
        self.hooked: set = set()
        self.active: list = []         # [(deltas dict, scale)]

    def _load(self, path: str) -> dict:
        from safetensors import safe_open
        if path in self.files:
            return self.files[path]
        ab: dict = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = json.loads((f.metadata() or {}).get("lora_adapter_metadata", "{}") or "{}")
            for k in f.keys():
                for suf, part in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
                    if k.endswith(suf):
                        ab.setdefault(k[: -len(suf)].removeprefix("transformer."), {})[part] = f.get_tensor(k)
        r = float(meta.get("r") or meta.get("transformer.r") or next(iter(ab.values()))["A"].shape[0])
        s = float(meta.get("lora_alpha") or meta.get("transformer.lora_alpha") or r) / r
        mods = dict(self.model.named_modules())
        out = {}
        for name, d in ab.items():
            if name not in mods:
                raise RuntimeError(f"LoRA {path}: module {name} not in the DiT")
            out[name] = (d["A"].to(self.device, torch.bfloat16), d["B"].to(self.device, torch.bfloat16), s)
            if name not in self.hooked:
                mods[name].register_forward_hook(lambda m, args, y, n=name: self._add(n, args[0], y))
                self.hooked.add(name)
        self.files[path] = out
        log(f"LoRA {Path(path).name}: {len(out)} modules, alpha/r {s:g}")
        return out

    def _add(self, name, x, y):
        for deltas, scale, mask in self.active:
            d = deltas.get(name)
            if d is not None:
                a, b, s = d
                dy = F.linear(F.linear(x.to(a.dtype), a), b) * (s * scale)
                if mask is not None and dy.shape[1] >= mask.shape[0]:
                    # regional: on the picture's own tokens (always the LAST h*w of the joint stream, also
                    # under the KV cache) only inside the character's box; text and references unmasked
                    n = mask.shape[0]
                    dy = torch.cat([dy[:, :-n], dy[:, -n:] * mask[:, None].to(dy.dtype)], 1)
                y = y + dy.to(y.dtype)
        return y

    def set(self, loras, grid=None) -> None:
        """loras: [[path, scale] or [path, scale, [x0, y0, x1, y1] (0-1 box: apply only there)], ...];
        grid = (h/16, w/16) of the picture being drawn."""
        self.active = []
        for l in loras or []:
            mask = None
            if len(l) > 2 and l[2] and grid:
                gh, gw = grid
                x0, y0, x1, y1 = l[2]
                ys = (torch.arange(gh, device=self.device).float() + 0.5) / gh
                xs = (torch.arange(gw, device=self.device).float() + 0.5) / gw
                # soft edge, ~1.5 tokens wide, so the box border does not show
                f = lambda v, lo, hi, n: ((v - lo) * n / 1.5 + 0.5).clamp(0, 1) * ((hi - v) * n / 1.5 + 0.5).clamp(0, 1)
                mask = (f(ys, y0, y1, gh)[:, None] * f(xs, x0, x1, gw)[None, :]).reshape(-1)
            self.active.append((self._load(str(Path(l[0]).expanduser())), float(l[1]), mask))


# --- attention: AMD's Triton flash kernel where the call allows it ----------------------------------

def use_fa_amd() -> bool:
    """The DiT's unmasked attention calls (every image block and the target image) through AMD's
    Triton flash-attention (~2.5x torch SDPA on the XTX, memory xtx-attention-kernels); masked calls
    (the causal text runs, padded prompts) stay on SDPA."""
    if not (FA_AMD / "fa_amd").exists() or os.environ.get("QWEN_IMAGE_FA_AMD", "1") != "1":
        return False
    sys.path.insert(0, str(FA_AMD))
    try:
        from fa_amd import interface_v2 as fa
    except Exception as exc:                 # noqa: BLE001 - optional speed-up
        log(f"fa_amd unavailable ({exc}); using torch SDPA")
        return False
    import diffusers.models.transformers.transformer_qwenimage21 as m

    orig = m.dispatch_attention_fn

    def attn(query, key, value, attn_mask=None, dropout_p=0.0, backend=None, parallel_config=None, **kw):
        if attn_mask is None and not dropout_p and query.dtype in (torch.bfloat16, torch.float16) and not kw:
            with torch.cuda.device(query.device):
                return fa.fwd(query.contiguous(), key.contiguous(), value.contiguous(), None, None, 0.0,
                              query.shape[-1] ** -0.5, False, -1, -1, 0.0, False)[0]
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, backend=backend,
                    parallel_config=parallel_config, **kw)
    m.dispatch_attention_fn = attn
    return True


# --- one picture -----------------------------------------------------------------------------------

def ref_image(path: str, res: int = REF_RES):
    """A reference as the pipeline would size it at output_resolution=res (area res^2, sides /32),
    RGBA as the VAE reads it."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    from PIL import Image

    im = Image.open(path)
    im = im.convert("RGBA") if im.mode != "RGBA" else im
    w, h, _ = calculate_dimensions(res * res, im.width / im.height)
    return im.resize((w, h), Image.LANCZOS)


def control_map(c: dict, w: int, h: int):
    """The control picture at the canvas size. "canny" takes the edges of any picture."""
    import numpy as np
    from PIL import Image, ImageFilter

    im = Image.open(c["image"]).convert("RGB").resize((w, h), Image.LANCZOS)
    if c.get("type") == "canny" and not c.get("prepared"):
        try:
            import cv2
            g = cv2.Canny(np.asarray(im.convert("L")), int(c.get("low", 100)), int(c.get("high", 200)))
        except ImportError:                  # no OpenCV: a Sobel-ish edge map, thresholded
            e = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
            g = ((e > float(c.get("threshold", 40))) * 255).astype(np.uint8)
        im = Image.fromarray(g).convert("RGB")
    return im


class Drawer:
    """resident=True (--serve): encoder and DiT both stay loaded; they only fit one card with the DiT
    as int8 (encoder int8+bf16 vision 8.1 GiB + bf16 DiT 13.3 GiB left no room to draw: OOM, 10-19).
    resident=False (a jobs file): every prompt is encoded first, the encoder freed, then the bf16 DiT
    loaded (encode_all / start_drawing)."""

    def __init__(self, device: str, resident: bool = True):
        from transformers import Qwen3VLProcessor

        self.device, self.ctl, self.pipe = device, None, None
        self.cache: dict = {}                # best-of-N: a keyframe's candidates share prompt and refs
        self.ref_cache: dict = {}
        self.cache_max = 8 if resident else 10**6
        t = time.time()
        self.fa = use_fa_amd()
        self.processor = Qwen3VLProcessor.from_pretrained(MODEL / "processor")
        self.enc = load_encoder(device)
        self.t_enc = time.time() - t
        if resident:
            self.start_drawing("int8" if DIT_Q == "auto" else DIT_Q)
        self.t_load = time.time() - t

    def start_drawing(self, dit_q: str) -> None:
        t = time.time()
        self.dit_q = dit_q
        self.pipe = load_drawer(self.device, self.processor, dit_q)
        log(f"drawer loaded in {time.time() - t:.1f} s (ENC={ENC_Q if self.enc else 'freed'} DIT={dit_q}, "
            f"fa_amd={self.fa}), {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")

    def free_encoder(self) -> None:
        self.enc = None
        gc.collect()
        torch.cuda.empty_cache()

    def encode(self, prompt: str, refs: list[str]):
        """Prompt + reference pictures through Qwen3-VL -> (embeddings, mask, image-slot mask, ref sizes)."""
        key = (prompt, tuple((r, os.path.getmtime(r)) for r in refs))
        if key in self.cache:
            return self.cache[key]
        if self.enc is None:
            raise RuntimeError("prompt not encoded before the encoder was freed")
        if len(self.cache) >= self.cache_max:
            self.cache.pop(next(iter(self.cache)))
        ims = [ref_image(r) for r in refs]
        with torch.inference_mode():
            emb, mask, pad = self.enc.encode_prompt(prompt=prompt, image=ims or None, device=torch.device(self.device))
        if not torch.isfinite(emb).all():
            raise RuntimeError("the text encoder produced NaN/inf")
        out = (emb.cpu(), None if mask is None else mask.cpu(), pad.cpu(), [(im.width, im.height) for im in ims])
        self.cache[key] = out
        return out

    def ref_latents(self, refs: list[str]):
        """The references as VAE latent tokens, put in front of the picture's tokens."""
        if not refs:
            return None
        key = tuple((r, os.path.getmtime(r)) for r in refs)
        if key not in self.ref_cache:
            if len(self.ref_cache) >= 8:
                self.ref_cache.pop(next(iter(self.ref_cache)))
            p, packed = self.pipe, []
            g = torch.Generator(self.device).manual_seed(0)
            with torch.inference_mode():
                for r in refs:                       # [1, 64, 1, h/16, w/16] -> [1, h/16*w/16, 64]
                    im = ref_image(r)
                    v = p.image_processor.preprocess(im, width=im.width, height=im.height).unsqueeze(2)
                    e = p._encode_vae_image(v.to(self.device, torch.bfloat16), g)
                    packed.append(p._pack_latents(e, 1, e.shape[1], e.shape[3], e.shape[4]))
            self.ref_cache[key] = torch.cat(packed, dim=1).cpu()
        return self.ref_cache[key]

    def control(self, c, w: int, h: int):
        """c: one control dict, or a LIST of them (multi-control: each runs the control chain, the skips
        are summed with their own scales; +~50% step time per extra control)."""
        if self.ctl:
            self.ctl.extra = []
        if isinstance(c, list):
            if not c:
                return self.control(None, w, h)
            first = self.control(c[0], w, h)
            ex = []
            for cc in c[1:]:
                self.control(cc, w, h)
                ex.append((self.ctl.context, self.ctl.scale))
            self.control(c[0], w, h)
            self.ctl.extra = ex
            return first
        if not c:
            if self.ctl:
                self.ctl.context = None
            return None
        if self.ctl is None:
            t = time.time()
            self.ctl = load_control(self.pipe.transformer, self.device)
            log(f"control branch loaded in {time.time() - t:.1f} s ({CTRL_Q}), "
                f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")
        p = self.pipe
        im = control_map(c, w, h)
        with torch.inference_mode():
            x = p.image_processor.preprocess(im, height=h, width=w)
            x = torch.cat([x, torch.ones_like(x[:, :1])], 1).unsqueeze(2).to(self.device, torch.bfloat16)
            lat = p._encode_vae_image(x, torch.Generator(self.device).manual_seed(0))      # [1,64,1,h/16,w/16]
            ctx = torch.cat([lat, torch.zeros_like(lat[:, :1]), torch.zeros_like(lat)], 1)  # | mask 0 | no inpaint
            self.ctl.context = p._pack_latents(ctx, 1, ctx.shape[1], ctx.shape[3], ctx.shape[4])
        self.ctl.scale = float(c.get("scale", 1.0))
        return im

    def draw(self, j: dict, steps_out: list | None = None):
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        dev, p = self.device, self.pipe
        tr, sched = p.transformer, p.scheduler
        w, h = j.get("size") or (1536, 864)
        w, h = int(w) // 32 * 32, int(h) // 32 * 32
        refs = [r for r in j.get("refs", []) if Path(r).exists()][:MAX_REFS]
        if len(j.get("refs", [])) > MAX_REFS:
            log(f"{j['id']}: {len(j['refs'])} references, the model takes {MAX_REFS}: using the first")
        emb, mask, pad, ref_sizes = self.encode(j["prompt"], refs)
        ref_lat = self.ref_latents(refs)
        ctl_on = self.control(j.get("control"), w, h) is not None
        if j.get("loras") or getattr(self, "loras", None):
            if getattr(self, "loras", None) is None:
                self.loras = JobLoras(tr, dev)
            self.loras.set(j.get("loras"), (h // 16, w // 16))
        g = torch.Generator(dev).manual_seed(int(j.get("seed", 0)))
        with torch.inference_mode():
            emb = emb.to(dev)
            mask = None if mask is None else mask.to(dev)
            shape = (1, 1, 64, h // 16, w // 16)
            from diffusers.utils.torch_utils import randn_tensor
            lat = randn_tensor(shape, generator=g, device=dev, dtype=torch.float32)
            lat = p._pack_latents(lat, 1, 64, h // 16, w // 16)        # fp32 latents, as Lightning trained
            ref = None if ref_lat is None else ref_lat.to(dev)
            img_shapes = [[*[(1, rh // 16, rw // 16) for rw, rh in ref_sizes], (1, h // 16, w // 16)]]
            mu = calculate_shift(lat.shape[1], sched.config.get("base_image_seq_len", 256),
                                 sched.config.get("max_image_seq_len", 4096), sched.config.get("base_shift", 0.5),
                                 sched.config.get("max_shift", 1.15))
            init = j.get("init")         # img2img: {"image": path, "strength": 0..1} (1 = pure noise)
            st = float(init.get("strength", 0.7)) if init else 1.0
            # per-pixel strength (differential diffusion, 10-20): init "mask" (L, 255 = `strength`) + "strength_bg"
            # elsewhere; the run starts at the larger one and each token is held on the noised-init path until the
            # schedule reaches its own start sigma
            dmask = None
            if init and init.get("mask") and init.get("strength_bg") is not None and Path(init["mask"]).exists():
                st_in, st_bg = st, float(init["strength_bg"])
                st = max(st_in, st_bg)
                from PIL import Image as _I, ImageFilter as _F
                gh, gw = h // 16, w // 16      # the packed latent's token grid (_pack_latents at h//16, w//16)
                m = _I.open(init["mask"]).convert("L").filter(_F.GaussianBlur(3)).resize((gw, gh), _I.BILINEAR)
                mm = torch.tensor(list(m.getdata()), dtype=torch.float32, device=dev).view(1, gh * gw, 1) / 255.0
                dmask = (mm, st_in, st_bg)
            timesteps, _ = retrieve_timesteps(sched, None, dev, sigmas=[s_ * st for s_ in SIGMAS], mu=mu)
            if init:
                from PIL import Image as _I
                im0 = _I.open(init["image"]).convert("RGB").resize((w, h), _I.LANCZOS)
                v = p.image_processor.preprocess(im0, height=h, width=w)
                v = torch.cat([v, torch.ones_like(v[:, :1])], 1).unsqueeze(2)      # RGB + alpha, as control()
                x0 = p._encode_vae_image(v.to(dev, torch.bfloat16), torch.Generator(dev).manual_seed(0))
                x0 = p._pack_latents(x0, 1, 64, h // 16, w // 16).float()
                s0 = float(sched.sigmas[0])            # the shifted start sigma actually used
                noise = lat.clone()
                lat = s0 * lat + (1.0 - s0) * x0
                j["_sigma0"] = round(s0, 3)
                if dmask is not None and dmask[0].shape[1] == lat.shape[1]:
                    import math as _m

                    def _shift(sg):        # the scheduler's exponential time shift (checked against sigmas[0] below)
                        return _m.exp(mu) / (_m.exp(mu) + (1 / max(sg, 1e-6) - 1)) if sg > 0 else 0.0
                    mm, st_in, st_bg = dmask
                    a_in, a_bg = _shift(SIGMAS[0] * st_in), _shift(SIGMAS[0] * st_bg)
                    j["_dd"] = {"start_in": round(a_in, 3), "start_bg": round(a_bg, 3), "shift_check": round(_shift(SIGMAS[0] * st) - s0, 4)}
                    dmask = (mm * a_in + (1 - mm) * a_bg, noise, x0)      # per-token start sigma
                elif dmask is not None:     # never silently paint the whole frame at the inside strength (10-20)
                    raise RuntimeError(f"{j.get('id')}: per-pixel mask has {dmask[0].shape[1]} tokens, the latent "
                                       f"{lat.shape[1]} ({w}x{h})")
            pad = torch.cat([pad.to(dev), pad.new_ones(1, lat.shape[1] // 4).to(dev)], dim=1)
            prefix = int(emb.shape[1] - pad[0, :emb.shape[1]].sum()) + 4 * int(pad[0, :emb.shape[1]].sum())
            use_kv = KV_CACHE and not ctl_on and self._kv_fits(prefix)
            cache = QwenImage21KVCache(len(tr.transformer_blocks)) if use_kv else None
            sched.set_begin_index(0)
            last = time.time()
            # control ENDS part-way (10-22 mannequin flow: pose/depth fix the structure early, the LoRAs and prompt
            # paint the rest): "end" = the fraction of the steps that get control (the smallest of a multi-control)
            cl_ = j.get("control")
            ends_ = [float(c_.get("end", 1.0)) for c_ in (cl_ if isinstance(cl_, list) else [cl_] if cl_ else [])]
            ctl_steps = math.ceil(min(ends_) * len(timesteps)) if ends_ else len(timesteps)
            ctl_saved = None
            for i, t in enumerate(timesteps):
                if self.ctl and i == ctl_steps and self.ctl.context is not None:
                    ctl_saved = (self.ctl.context, self.ctl.extra)      # control off from here on
                    self.ctl.context, self.ctl.extra = None, []
                if init and dmask is not None:          # tokens not started yet stay on the noised init path
                    sg = float(sched.sigmas[i])
                    hold = (dmask[0] < sg - 1e-4).float()
                    lat = hold * (sg * dmask[1] + (1 - sg) * dmask[2]) + (1 - hold) * lat
                x = lat.to(torch.bfloat16)
                if ref is not None:
                    x = torch.cat([ref, x], dim=1)
                kv_mode = ("extract" if i == 0 else "cached") if use_kv else None
                pred = tr(hidden_states=x, timestep=(t.expand(1) / 1000).to(torch.bfloat16),
                          encoder_hidden_states=emb, encoder_hidden_states_mask=mask, img_shapes=img_shapes,
                          img_mask=pad, kv_cache=cache, kv_cache_mode=kv_mode, return_dict=False)[0]
                pred = pred[:, -lat.shape[1]:].float()
                lat = sched.step(pred, t, lat, return_dict=False)[0]
                if steps_out is not None:
                    torch.cuda.synchronize()
                    steps_out.append(round(time.time() - last, 2))
                    last = time.time()
            del cache
            if self.ctl:
                self.ctl.context = None
                self.ctl.extra = []
            if not torch.isfinite(lat).all():
                raise RuntimeError("the denoiser produced NaN/inf (bad GPU state or a broken kernel)")
            z = p._unpack_latents(lat.to(torch.bfloat16), h, w, 16)
            vae = p.vae
            mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1).to(dev, z.dtype)
            std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1).to(dev, z.dtype)
            img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
            img = p.image_processor.postprocess(img, output_type="pil")[0]
            if steps_out is not None:
                torch.cuda.synchronize()
                steps_out.append(round(time.time() - last, 2))
        img = img.convert("RGB") if img.mode != "RGB" else img
        dest = Path(j["path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}")
        img.save(tmp, format="PNG")
        tmp.replace(dest)
        j["_refs"], j["_kv"] = len(refs), use_kv
        return img

    def _kv_fits(self, prefix_tokens: int) -> bool:
        """The prefix KV cache costs 32 blocks x K,V x 4096 x bf16 = 512 KiB a token: only when it fits."""
        free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        # + the uncached forward's own peak: 2.6 GiB at 4 refs (measured 10-19), and a margin
        return prefix_tokens * 512 * 1024 + 3 * 2**30 < free


# --- protocol (as illustrate.py) -------------------------------------------------------------------

def run_job(d: Drawer, j: dict) -> dict:
    t = time.time()
    torch.cuda.reset_peak_memory_stats()
    if not Path(j["path"]).exists():
        steps: list = []
        d.draw(j, steps)
        log(f"{j['id']}: {time.time() - t:.1f} s (steps+decode {steps}, refs {j.get('_refs', 0)}, "
            f"kv {j.get('_kv')}, control {bool(j.get('control'))}, sigma0 {j.get('_sigma0', 1.0)}"
            + (f", per-pixel {j['_dd']}" if j.get("_dd") else "") + ")")
    st = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(("VmHWM", "RssAnon"))}
    return {"id": j["id"], "refs": j.get("_refs", 0), "seconds": round(time.time() - t, 1),
            "sigma0": j.get("_sigma0"), "dd": j.get("_dd"),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "rss_anon_gb": round(st.get("RssAnon", 0) / 2**20, 2),       # VmHWM also counts mmapped weight
            "rss_peak_gb": round(st.get("VmHWM", 0) / 2**20, 2)}          # pages (page cache) touched at load


def serve(device: str) -> int:
    t = time.time()
    d = Drawer(device)
    print(json.dumps({"phase": "load drawing model (kept)", "seconds": round(time.time() - t, 1)}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        j = json.loads(line)
        try:
            print(json.dumps(run_job(d, j)), flush=True)
        except Exception as exc:              # a bad job (an unreadable reference): report, keep serving
            import traceback
            traceback.print_exc()
            print(json.dumps({"id": j.get("id"), "error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
            if d.ctl:
                d.ctl.context = None
            gc.collect()
            torch.cuda.empty_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--steps", type=int, default=4, help="ignored: the Lightning LoRA draws in 4")
    ap.add_argument("--serve", action="store_true", help="keep the model loaded; jobs as JSON lines on stdin")
    a = ap.parse_args()
    # As illustrate.py / ace_serve.py: MIOpen's conv search costs more than it saves for a few VAE passes.
    torch.backends.cudnn.enabled = os.environ.get("ILLUSTRATE_MIOPEN", "0") == "1"
    if os.environ.get("QWEN_IMAGE_BLASLT", "1") == "1":     # hipBLASLt GEMMs (SkyReels: +35% over rocBLAS)
        torch.backends.cuda.preferred_blas_library("cublaslt")
    if a.serve:
        return serve("cuda")
    if not (a.jobs and a.out):
        ap.error("jobs and out are required without --serve")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(a.jobs).read_text())
    for j in jobs:
        j["path"] = j.get("path") or str(out / f"{j['id']}.png")
        j.setdefault("size", [a.width, a.height])
    jobs = [j for j in jobs if not Path(j["path"]).exists()]
    if not jobs:
        return 0
    # Encode every prompt first, free the encoder, then draw with the bf16 DiT (illustrate.py's two
    # halves): the encoder and the bf16 DiT do not fit one card together.
    t = time.time()
    d = Drawer("cuda", resident=False)
    bad = {}
    for j in jobs:
        try:
            d.encode(j["prompt"], [r for r in j.get("refs", []) if Path(r).exists()][:MAX_REFS])
        except Exception as exc:
            bad[j["id"]] = f"{type(exc).__name__}: {exc}"[:300]
    print(json.dumps({"phase": f"read {len(jobs)} prompts", "seconds": round(time.time() - t, 1)}), flush=True)
    d.free_encoder()
    t = time.time()
    d.start_drawing("bf16" if DIT_Q == "auto" else DIT_Q)
    print(json.dumps({"phase": "load drawing model", "seconds": round(time.time() - t, 1)}), flush=True)
    for j in jobs:
        if j["id"] in bad:
            print(json.dumps({"id": j["id"], "error": bad[j["id"]]}), flush=True)
            continue
        try:
            print(json.dumps(run_job(d, j)), flush=True)
        except Exception as exc:
            print(json.dumps({"id": j.get("id"), "error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
