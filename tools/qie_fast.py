"""Fast Qwen-Image-Edit-2509 transformer for HY-Pano on one 24 GB XTX (ROCm), used by tools/world.py.

The diffusers GGUF path dequantises every Q6_K weight in torch on every pass. Here each Linear is
dequantised ONCE at load, the LoRAs (HY-Pano, optionally Lightning) are merged in at full precision, and
the result is stored as int8 with one scale per output row (tools/qwen_image.QLinear, the keyframe
drawer's layer: widened to bf16 per call). Plus the keyframe drawer's ROCm settings: hipBLASLt GEMMs
and AMD's Triton flash attention (fa_amd) for the unmasked joint attention.

    tr = load_int8_transformer(gguf, base_dir, loras=[(path, weight), ...], device="cuda")
    install_fa_amd()          # patches diffusers' QwenImage attention dispatch; returns False if unavailable

There is no bf16 Edit-2509 transformer on disk (deleted to save 40 GB), so the source is the Q6_K GGUF;
int8 per-row on top of Q6_K adds little error (checked against the GGUF path on the same seed).
Never run under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (silent corruption on this ROCm).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

TOOLS = Path(__file__).resolve().parent
FA_AMD = Path(os.environ.get("QWEN_IMAGE_FA", Path.home() / "src" / "attn-lab")).expanduser()


def log(msg: str) -> None:
    sys.stderr.write(f"[qie_fast] {msg}\n")
    sys.stderr.flush()


def _lora_deltas(loras):
    """[(path, weight)] -> {module_name: [(B, A, scale)]} from diffusers/peft-style LoRA files
    (keys '[transformer.]<module>.lora_A.weight' / 'lora_B', or '.lora_down'/'.lora_up', or '.lora.down/up')."""
    from safetensors import safe_open
    out: dict = {}
    for path, weight in loras:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = json.loads((f.metadata() or {}).get("lora_adapter_metadata", "{}") or "{}")
            keys = list(f.keys())
            parts: dict = {}
            for k in keys:
                kk = k.removeprefix("transformer.").removeprefix("diffusion_model.")
                for a, b in ((".lora_A.weight", "A"), (".lora_B.weight", "B"), (".lora_down.weight", "A"),
                             (".lora_up.weight", "B"), (".lora.down.weight", "A"), (".lora.up.weight", "B"),
                             (".alpha", "alpha")):
                    if kk.endswith(a):
                        parts.setdefault(kk[: -len(a)], {})[b] = f.get_tensor(k)
                        break
            r_meta = meta.get("transformer.r") or meta.get("r")
            a_meta = meta.get("transformer.lora_alpha") or meta.get("lora_alpha")
            for mod, p in parts.items():
                if "A" not in p or "B" not in p:
                    continue
                r = p["A"].shape[0]
                if "alpha" in p:
                    alpha = float(p["alpha"])
                elif a_meta is not None:
                    alpha = float(a_meta) * (r / float(r_meta) if r_meta else 1.0)
                else:
                    alpha = float(r)            # no alpha anywhere: scale 1
                out.setdefault(mod, []).append((p["B"], p["A"], weight * alpha / r))
    return out


def _qlinear_class():
    """tools/qwen_image.QLinear, loaded by file path: HY-World's panogen also has a package named qwen_image."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bookcaster_qwen_image", TOOLS / "qwen_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.QLinear


def _send_checked(t: torch.Tensor, device: str) -> torch.Tensor:
    """CPU -> GPU, compared on arrival (copies to the dock-attached XTX are occasionally corrupted)."""
    for _ in range(4):
        g = t.to(device)
        if torch.equal(g.cpu(), t):
            return g
        log("a weight arrived corrupted; resending")
    raise RuntimeError("weight keeps arriving corrupted")


@torch.no_grad()
def load_int8_transformer(gguf: str | Path, base_dir: str | Path, loras=(), device: str = "cuda",
                          mode: str = "int8"):
    """GGUF Edit-2509 transformer -> int8 per-row QLinear on `device`, LoRAs merged. Returns the
    diffusers QwenImageTransformer2DModel (its Linear layers replaced) and a stats dict."""
    from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    QLinear = _qlinear_class()

    t0 = time.time()
    tr = QwenImageTransformer2DModel.from_single_file(
        str(gguf), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=str(Path(base_dir) / "transformer"), torch_dtype=torch.bfloat16)
    t_read = time.time() - t0
    deltas = _lora_deltas(loras)
    used = set()
    n_lin = 0
    t1 = time.time()
    for name, mod in list(tr.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight
        if isinstance(w, GGUFParameter) or getattr(w, "quant_type", None) is not None:
            g = GGUFParameter(_send_checked(w.data, device), quant_type=w.quant_type)
            wf = dequantize_gguf_tensor(g).float().reshape(mod.out_features, mod.in_features)
            del g
        else:
            wf = _send_checked(w.data, device).float()
        for B, A, sc in deltas.get(name, []):
            wf += sc * (B.to(device, torch.float32) @ A.to(device, torch.float32))
            used.add(name)
        q = QLinear(mod, mode)
        q.set_weight(wf)
        q.bias = nn.Parameter(_send_checked(mod.bias.data, device).to(torch.bfloat16), requires_grad=False) \
            if mod.bias is not None else None
        parent, attr = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(tr.get_submodule(parent) if parent else tr, attr, q)
        del wf
        n_lin += 1
    missing = set(deltas) - used
    if missing:
        raise RuntimeError(f"{len(missing)} LoRA modules matched nothing, e.g. {sorted(missing)[:3]}")
    # everything else (norms, embedders' non-linear params, buffers) to the GPU in bf16
    for name, p in list(tr.named_parameters()):
        if p.device.type != "cuda" or (p.is_floating_point() and p.dtype != torch.bfloat16):
            owner = tr.get_submodule(name.rsplit(".", 1)[0]) if "." in name else tr
            setattr(owner, name.rsplit(".", 1)[-1],
                    nn.Parameter(_send_checked(p.data, device).to(torch.bfloat16) if p.is_floating_point()
                                 else _send_checked(p.data, device), requires_grad=False))
    tr.to(device)
    tr.eval()
    stats = {"read_s": round(t_read, 1), "convert_s": round(time.time() - t1, 1), "linears": n_lin,
             "lora_modules": len(used), "gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
    log(f"int8 transformer ready {stats}")
    return tr, stats


def install_fa_amd() -> bool:
    """Route diffusers' QwenImage joint attention (unmasked, bf16) through fa_amd's Triton flash kernel."""
    if os.environ.get("QIE_FA_AMD", "1") != "1" or not (FA_AMD / "fa_amd").exists():
        return False
    sys.path.insert(0, str(FA_AMD))
    try:
        from fa_amd import interface_v2 as fa
    except Exception as exc:                 # noqa: BLE001 - optional speed-up
        log(f"fa_amd unavailable ({exc}); torch SDPA")
        return False
    import diffusers.models.transformers.transformer_qwenimage as m
    if getattr(m.dispatch_attention_fn, "_fa_amd", False):
        return True
    orig = m.dispatch_attention_fn
    calls = {"fa": 0, "sdpa": 0}

    def attn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None,
             parallel_config=None, **kw):
        if attn_mask is None and not dropout_p and not is_causal and not kw \
                and query.dtype in (torch.bfloat16, torch.float16):
            calls["fa"] += 1
            with torch.cuda.device(query.device):
                return fa.fwd(query.contiguous(), key.contiguous(), value.contiguous(), None, None, 0.0,
                              query.shape[-1] ** -0.5, False, -1, -1, 0.0, False)[0]
        calls["sdpa"] += 1
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                    backend=backend, parallel_config=parallel_config, **kw)
    attn._fa_amd = True
    attn.calls = calls
    m.dispatch_attention_fn = attn
    return True
