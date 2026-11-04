"""11-04 eager fusion for the LTX DiT block (Triton kernels; binaries cache on disk, no torch.compile tracing).
Block profile (block 0, 16560 video tokens, LoRA on): eager 216.1 ms v compiled 190.7 ms; ~90% of the gap is the
copies / elementwise ops around the GEMMs and attention. Each kernel here replaces one such pattern; 1-5 reproduce
eager's bf16 rounding step by step (block outputs bit-identical to eager), 6 (norm fusion) is within one bf16 step.
9-clip Acts replay, 3 lanes: compiled 446.0 s, plain eager 415.3 s, fused 1-5 374.5 s (clips frame-identical to eager).

1. RoPE (split type): apply_split_rotary_emb transposes, splits, mul + 2x addcmul, cat, transposes back
   (~6 kernels + copies per q / k). One kernel reads x (B, T, H*D) and cos / sin (B|1, H, T, D/2) by stride.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_split_kernel(x_ptr, c_ptr, s_ptr, o_ptr, T, H: tl.constexpr, R: tl.constexpr,
                       sxb, sxt, sxh, scb, sch, sct, ssb, ssh, sst, sob, sot, soh,
                       BH: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T
    h = tl.arange(0, BH)[:, None]
    r = tl.arange(0, BR)[None, :]
    m = (h < H) & (r < R)
    xb = x_ptr + b * sxb + t * sxt + h * sxh
    x1 = tl.load(xb + r, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(xb + R + r, mask=m, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + b * scb + h * sch + t * sct + r, mask=m, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + b * ssb + h * ssh + t * sst + r, mask=m, other=0.0).to(tl.float32)
    # eager: out = split * cos (bf16 round), then addcmul(out_half, -/+sin, other_half) (fp32 math, one round)
    a1 = (x1 * c).to(tl.bfloat16).to(tl.float32)
    a2 = (x2 * c).to(tl.bfloat16).to(tl.float32)
    o1 = (a1 + (-s) * x2).to(tl.bfloat16)
    o2 = (a2 + s * x1).to(tl.bfloat16)
    ob = o_ptr + b * sob + t * sot + h * soh
    tl.store(ob + r, o1, mask=m)
    tl.store(ob + R + r, o2, mask=m)


def rope_split(x, cos, sin):
    """x (B, T, H*D) bf16; cos / sin (B or 1, H, T, D/2) bf16 -> (B, T, H*D), as apply_split_rotary_emb."""
    B, T, HD = x.shape
    _, H, Tc, R = cos.shape
    assert Tc == T and HD == H * 2 * R and sin.shape == cos.shape
    xv = x.view(B, T, H, 2 * R) if x.stride(-1) == 1 else x.contiguous().view(B, T, H, 2 * R)
    out = torch.empty(B, T, H, 2 * R, device=x.device, dtype=x.dtype)
    cb = 0 if cos.shape[0] == 1 else cos.stride(0)
    sb = 0 if sin.shape[0] == 1 else sin.stride(0)
    _rope_split_kernel[(B * T,)](xv, cos, sin, out, T, H, R,
                                 xv.stride(0), xv.stride(1), xv.stride(2),
                                 cb, cos.stride(1), cos.stride(2), sb, sin.stride(1), sin.stride(2),
                                 out.stride(0), out.stride(1), out.stride(2),
                                 BH=triton.next_power_of_2(H), BR=triton.next_power_of_2(R), num_warps=4)
    return out.view(B, T, H * 2 * R)


def install_rope():
    """Route LTX's split RoPE through the fused kernel when it applies (3-D bf16 input, 4-D bf16 cos / sin, last
    dims contiguous); anything else falls back to the original function."""
    import ltx_core.model.transformer.rope as rope
    if getattr(rope, "_eager_fused", False):
        return
    orig = rope.apply_split_rotary_emb

    def fused(input_tensor, cos_freqs, sin_freqs):
        if (input_tensor.ndim == 3 and cos_freqs.ndim == 4 and input_tensor.is_cuda
                and input_tensor.dtype == cos_freqs.dtype == sin_freqs.dtype == torch.bfloat16
                and cos_freqs.stride(-1) == 1 and sin_freqs.stride(-1) == 1
                and cos_freqs.shape[0] in (1, input_tensor.shape[0])):
            return rope_split(input_tensor, cos_freqs, sin_freqs)
        return orig(input_tensor, cos_freqs, sin_freqs)
    rope.apply_split_rotary_emb = fused
    rope._eager_fused = True
    rope._orig_apply_split_rotary_emb = orig


# 2. AdaLN modulate + gated residual. Eager writes `rms_norm(x) * (1 + scale) + shift` (3 elementwise kernels after
#    the norm) and `x = x + y * gate` (2 kernels). The norm stays PyTorch's (a Triton reduction would sum in another
#    order); these kernels do the elementwise tail in one pass, rounding to bf16 where eager does.
@triton.jit
def _mod_kernel(n_ptr, sc_ptr, sh_ptr, o_ptr, T, D, snb, snt, scb, sct, shb, sht, sob, sot, BD: tl.constexpr):
    row = tl.program_id(0)
    b = row // T
    t = row % T
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    m = d < D
    n = tl.load(n_ptr + b * snb + t * snt + d, mask=m).to(tl.float32)
    sc = tl.load(sc_ptr + b * scb + t * sct + d, mask=m).to(tl.float32)
    sh = tl.load(sh_ptr + b * shb + t * sht + d, mask=m).to(tl.float32)
    one_sc = (1.0 + sc).to(tl.bfloat16).to(tl.float32)
    u = (n * one_sc).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + b * sob + t * sot + d, (u + sh).to(tl.bfloat16), mask=m)


@triton.jit
def _gate_kernel(x_ptr, y_ptr, g_ptr, o_ptr, T, D, sxb, sxt, syb, syt, sgb, sgt, sob, sot, BD: tl.constexpr):
    row = tl.program_id(0)
    b = row // T
    t = row % T
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    m = d < D
    x = tl.load(x_ptr + b * sxb + t * sxt + d, mask=m).to(tl.float32)
    y = tl.load(y_ptr + b * syb + t * syt + d, mask=m).to(tl.float32)
    g = tl.load(g_ptr + b * sgb + t * sgt + d, mask=m).to(tl.float32)
    p = (y * g).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + b * sob + t * sot + d, (x + p).to(tl.bfloat16), mask=m)


def _rows(t, B, T, D):
    """(B, T, D)-broadcastable bf16 tensor -> (tensor, batch stride, token stride) with unit last stride, or None."""
    if t.dtype != torch.bfloat16 or t.ndim != 3 or t.stride(-1) != 1 or t.shape[-1] != D:
        return None                 # a broadcast last dim would get stride 0; the kernels assume unit stride there
    t = t.expand(B, T, D)
    return t, t.stride(0), t.stride(1)


def _ok(*ts):
    return all(t.is_cuda and t.dtype == torch.bfloat16 for t in ts)


def modulate(x, scale, shift, eps):
    """rms_norm(x, eps) * (1 + scale) + shift, eager-exact."""
    from ltx_core.utils import rms_norm
    n = rms_norm(x, eps=eps)
    if n.ndim == 3 and _ok(n, scale, shift) and n.stride(-1) == 1:
        B, T, D = n.shape
        sc, sh = _rows(scale, B, T, D), _rows(shift, B, T, D)
        if sc and sh:
            out = torch.empty_like(n)
            BD = 1024
            _mod_kernel[(B * T, triton.cdiv(D, BD))](n, sc[0], sh[0], out, T, D, n.stride(0), n.stride(1),
                                                    sc[1], sc[2], sh[1], sh[2], out.stride(0), out.stride(1),
                                                    BD=BD, num_warps=4)
            return out
    return n * (1 + scale) + shift


def gate_add(x, y, g):
    """x + y * g, eager-exact."""
    if x.ndim == 3 and _ok(x, y, g) and x.stride(-1) == 1 and y.shape == x.shape and y.stride(-1) == 1:
        B, T, D = x.shape
        gg = _rows(g, B, T, D)
        if gg:
            out = torch.empty_like(x)
            BD = 1024
            _gate_kernel[(B * T, triton.cdiv(D, BD))](x, y, gg[0], out, T, D, x.stride(0), x.stride(1),
                                                     y.stride(0), y.stride(1), gg[1], gg[2],
                                                     out.stride(0), out.stride(1), BD=BD, num_warps=4)
            return out
    return x + y * g


def install_block(log=print):
    """Rewrite BasicAVTransformerBlock.forward's source: each `rms_norm(X, eps=self.norm_eps) * (1 + S) + SH` ->
    _ef_modulate(X, S, SH, self.norm_eps), each line `v = v + EXPR * GATE` -> v = _ef_gate_add(v, EXPR, GATE).
    Logs the counts; if the source does not parse or nothing matched, the original stays."""
    import inspect
    import re
    import textwrap
    import ltx_core.model.transformer.transformer as tr
    cls = next(c for c in vars(tr).values() if inspect.isclass(c) and c.__module__ == tr.__name__
               and "scale_shift_table" in inspect.getsource(c) and hasattr(c, "get_ada_values"))
    if getattr(cls, "_eager_fused", False):
        return
    src = textwrap.dedent(inspect.getsource(cls.forward))
    src, n_mod = re.subn(r"rms_norm\((\w+), eps=self\.norm_eps\) \* \(1 \+ (\w+)\) \+ (\w+)",
                         r"_ef_modulate(\1, \2, \3, self.norm_eps)", src)
    src, n_gate = re.subn(r"^(\s*)(\w+) = \2 \+ (.+) \* (\w+)$", r"\1\2 = _ef_gate_add(\2, \3, \4)", src, flags=re.M)
    if not (n_mod or n_gate):
        log("eager_fuse: block forward patterns not found; unchanged")
        return
    g = dict(tr.__dict__, _ef_modulate=modulate, _ef_gate_add=gate_add)
    ns = {}
    exec(compile(src, tr.__file__, "exec"), g, ns)
    cls.forward = ns["forward"]
    cls._eager_fused = True
    cls._eager_src = src
    log(f"eager_fuse: {cls.__name__}.forward -> {n_mod} modulate + {n_gate} gated-residual sites fused")


# 3. AdaLN values: LTX gathers the compact per-timestep modulation out to every token, THEN adds the block's small
#    scale_shift_table to that full (B, T, k, D) tensor. Adding the table to the compact rows first and gathering
#    after gives the same bf16 values (each element is the same two numbers through the same rounding) while the add
#    touches a few rows instead of T. Measured: the full-size adds were ~5 ms of a 216 ms block call.
def install_ada(log=print):
    import ltx_core.model.transformer.transformer as tr
    cls = next(c for c in vars(tr).values() if isinstance(c, type) and c.__module__ == tr.__name__
               and hasattr(c, "get_ada_values"))
    if getattr(cls, "_eager_fused_ada", False):
        return

    def get_ada_values(self, scale_shift_table, batch_size, timestep_embedding, indices, modulation=None):
        num_ada_params = scale_shift_table.shape[0]
        hidden_dim = scale_shift_table.shape[1]
        modulation = timestep_embedding.modulation if modulation is None else modulation
        modulation = modulation.reshape(batch_size, -1, num_ada_params, hidden_dim)[:, :, indices, :]
        table = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=modulation.device, dtype=modulation.dtype)
        return timestep_embedding.gather(table + modulation).unbind(dim=2)

    cls.get_ada_values = get_ada_values
    cls._eager_fused_ada = True
    log(f"eager_fuse: {cls.__name__}.get_ada_values -> add on the compact table, then gather")


# 4. More of the same patterns: `x + (attn(...) * gate * mask)` (audio<->video cross-attention residual, eager: two
#    rounded muls then the add) and `x * (1 + scale) + shift` without the norm (text cross-attention's AdaLN).
@triton.jit
def _gate2_kernel(x_ptr, y_ptr, g_ptr, k_ptr, o_ptr, T, D, sxb, sxt, syb, syt, sgb, sgt, skb, skt, skd, sob, sot,
                  BD: tl.constexpr):
    row = tl.program_id(0)
    b = row // T
    t = row % T
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    m = d < D
    x = tl.load(x_ptr + b * sxb + t * sxt + d, mask=m).to(tl.float32)
    y = tl.load(y_ptr + b * syb + t * syt + d, mask=m).to(tl.float32)
    g = tl.load(g_ptr + b * sgb + t * sgt + d, mask=m).to(tl.float32)
    k = tl.load(k_ptr + b * skb + t * skt + d * skd, mask=m).to(tl.float32)
    p = (y * g).to(tl.bfloat16).to(tl.float32)
    q = (p * k).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + b * sob + t * sot + d, (x + q).to(tl.bfloat16), mask=m)


def gate2_add(x, y, g, k):
    """x + (y * g * k), eager-exact (k: the perturbation mask, broadcast)."""
    if (x.ndim == 3 and torch.is_tensor(k) and _ok(x, y, g, k) and x.stride(-1) == 1 and y.shape == x.shape
            and y.stride(-1) == 1 and k.ndim <= 3):
        B, T, D = x.shape
        gg = _rows(g, B, T, D)
        k3 = k.reshape((1,) * (3 - k.ndim) + tuple(k.shape)).expand(B, T, D)     # mask: any broadcast, any strides
        kk = (k3, k3.stride(0), k3.stride(1), k3.stride(2))
        if gg:
            out = torch.empty_like(x)
            BD = 1024
            _gate2_kernel[(B * T, triton.cdiv(D, BD))](x, y, gg[0], kk[0], out, T, D, x.stride(0), x.stride(1),
                                                      y.stride(0), y.stride(1), gg[1], gg[2], kk[1], kk[2], kk[3],
                                                      out.stride(0), out.stride(1), BD=BD, num_warps=4)
            return out
    return x + (y * g * k)


def affine(x, scale, shift):
    """x * (1 + scale) + shift, eager-exact (x already normalized)."""
    if x.ndim == 3 and _ok(x, scale, shift) and x.stride(-1) == 1:
        B, T, D = x.shape
        sc, sh = _rows(scale, B, T, D), _rows(shift, B, T, D)
        if sc and sh:
            out = torch.empty_like(x)
            BD = 1024
            _mod_kernel[(B * T, triton.cdiv(D, BD))](x, sc[0], sh[0], out, T, D, x.stride(0), x.stride(1),
                                                    sc[1], sc[2], sh[1], sh[2], out.stride(0), out.stride(1),
                                                    BD=BD, num_warps=4)
            return out
    return x * (1 + scale) + shift


def install_more(log=print):
    """Second source pass: the multi-line cross-attention residual in the block's forward, and the affine lines in
    apply_cross_attention_adaln."""
    import inspect
    import re
    import textwrap
    import ltx_core.model.transformer.transformer as tr
    if getattr(tr, "_eager_fused_more", False):
        return
    cls = next(c for c in vars(tr).values() if isinstance(c, type) and c.__module__ == tr.__name__
               and hasattr(c, "get_ada_values"))
    n2 = 0
    if getattr(cls, "_eager_fused", False):        # forward already rewritten once: rewrite the current source text
        src = cls._eager_src
    else:
        src = textwrap.dedent(inspect.getsource(cls.forward))
    src, n2 = re.subn(r"(\w+) = \1 \+ \(\n(\s+)(self\.\w+\((?:.|\n)*?\n\s+\))\n\s+\* (\w+)\n\s+\* ([\w.]+)\n\s+\)",
                      r"\1 = _ef_gate2_add(\1, \3, \4, \5)", src)
    g = dict(tr.__dict__, _ef_modulate=modulate, _ef_gate_add=gate_add, _ef_gate2_add=gate2_add)
    ns = {}
    exec(compile(src, tr.__file__, "exec"), g, ns)
    cls.forward = ns["forward"]
    cls._eager_src = src
    fsrc = textwrap.dedent(inspect.getsource(tr.apply_cross_attention_adaln))
    fsrc, na = re.subn(r"(\w+) \* \(1 \+ (\w+)\) \+ (\w+)", r"_ef_affine(\1, \2, \3)", fsrc)
    ns = {}
    exec(compile(fsrc, tr.__file__, "exec"), dict(tr.__dict__, _ef_affine=affine), ns)
    tr.apply_cross_attention_adaln = ns["apply_cross_attention_adaln"]
    tr._eager_fused_more = True
    log(f"eager_fuse: {n2} cross-attention residuals + {na} text cross-attention affines fused")


# 5. fp8 weight upcast. Fp8CastLinear upcasts every weight on every call with weight.to(bf16) - PyTorch's generic
#    (non-vectorized) copy kernel: 3.2 ms per block call, the same at any token count (compiled: Triton casts, ~1.2 ms).
#    e4m3 -> bf16 is exact, so a vectorized Triton cast gives identical weights.
@triton.jit
def _cast_kernel(src_ptr, dst_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(dst_ptr + i, tl.load(src_ptr + i, mask=m).to(dst_ptr.dtype.element_ty), mask=m)


# Triton on gfx11 converts e4m3's NaN codes (0x7f / 0xff) as +-480 instead of NaN (all 254 finite codes are
# identical, checked exhaustively). Only e4m3, and only tensors scanned NaN-free once (install_cast) take this path.
_NAN_FREE = set()      # data_ptr of fp8 weights verified to hold no NaN code


def fp8_upcast(w, dtype):
    """w.to(dtype) for a NaN-free contiguous CUDA e4m3 tensor via a vectorized Triton cast; else w.to(dtype)."""
    if (w.dtype == torch.float8_e4m3fn and w.is_cuda and w.is_contiguous() and dtype in (torch.bfloat16, torch.float16)
            and w.data_ptr() in _NAN_FREE):
        out = torch.empty(w.shape, device=w.device, dtype=dtype)
        n = w.numel()
        BLOCK = 4096
        _cast_kernel[(triton.cdiv(n, BLOCK),)](w, out, n, BLOCK=BLOCK, num_warps=8)
        return out
    return w.to(dtype)


def install_cast(model, log=print):
    """Scan model's e4m3 weights / biases once for NaN codes, then route the NaN-free ones through fp8_upcast."""
    import ltx_core.quantization.fp8_cast as fc
    n_ok = n_bad = 0
    with torch.inference_mode():
        for p in model.parameters():
            if p.dtype == torch.float8_e4m3fn and p.is_cuda and p.is_contiguous():
                if bool(((p.view(torch.uint8) & 0x7F) == 0x7F).any()):
                    n_bad += 1
                else:
                    _NAN_FREE.add(p.data_ptr())
                    n_ok += 1
    if getattr(fc, "_eager_fused", False):
        log(f"eager_fuse: fp8 cast scan +{n_ok} NaN-free tensors ({n_bad} with NaN codes keep torch's cast)")
        return
    orig = fc._upcast_and_round

    def upcast(weight, dtype, with_stochastic_rounding=False, seed=0):
        if not with_stochastic_rounding or not fc._use_triton_fp8_kernel(weight.device):
            return fp8_upcast(weight, dtype)
        return orig(weight, dtype, with_stochastic_rounding, seed)
    fc._upcast_and_round = upcast
    fc._eager_fused = True
    log(f"eager_fuse: fp8 weight upcast -> Triton cast for {n_ok} NaN-free tensors ({n_bad} with NaN codes keep torch's cast)")


# 6. The RMS norm inside the modulate kernel (one read of x instead of norm write + read). On by default
#    (LTX_EAGER_FUSE_NORM=0 = off). NOT bit-exact: the sum of squares is reduced in Triton's order, not PyTorch's;
#    measured on real block inputs ~1% of outputs move one bf16 step v plain eager (compiled code: ~50%, larger).
@triton.jit
def _normmod_kernel(x_ptr, sc_ptr, sh_ptr, o_ptr, T, D, eps, sxb, sxt, scb, sct, shb, sht, sob, sot,
                    BD: tl.constexpr):
    row = tl.program_id(0)
    b = row // T
    t = row % T
    d = tl.arange(0, BD)
    m = d < D
    x = tl.load(x_ptr + b * sxb + t * sxt + d, mask=m, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    n = (x * rstd).to(tl.bfloat16).to(tl.float32)
    sc = tl.load(sc_ptr + b * scb + t * sct + d, mask=m).to(tl.float32)
    sh = tl.load(sh_ptr + b * shb + t * sht + d, mask=m).to(tl.float32)
    one_sc = (1.0 + sc).to(tl.bfloat16).to(tl.float32)
    u = (n * one_sc).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + b * sob + t * sot + d, (u + sh).to(tl.bfloat16), mask=m)


def modulate_norm(x, scale, shift, eps):
    if x.ndim == 3 and _ok(x, scale, shift) and x.stride(-1) == 1 and x.shape[-1] <= 8192:
        B, T, D = x.shape
        sc, sh = _rows(scale, B, T, D), _rows(shift, B, T, D)
        if sc and sh:
            out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            _normmod_kernel[(B * T,)](x, sc[0], sh[0], out, T, D, float(eps), x.stride(0), x.stride(1),
                                      sc[1], sc[2], sh[1], sh[2], out.stride(0), out.stride(1),
                                      BD=triton.next_power_of_2(D), num_warps=8)
            return out
    return modulate(x, scale, shift, eps)


def install_norm(log=print):
    """Swap the fused forward's modulate for the norm-fused one (re-exec the stored source with it)."""
    import ltx_core.model.transformer.transformer as tr
    cls = next(c for c in vars(tr).values() if isinstance(c, type) and c.__module__ == tr.__name__
               and hasattr(c, "get_ada_values"))
    if not getattr(cls, "_eager_fused", False):
        return
    g = dict(tr.__dict__, _ef_modulate=modulate_norm, _ef_gate_add=gate_add, _ef_gate2_add=gate2_add)
    ns = {}
    exec(compile(cls._eager_src, tr.__file__, "exec"), g, ns)
    cls.forward = ns["forward"]
    log("eager_fuse: RMS norm fused into modulate (within one bf16 step of eager; LTX_EAGER_FUSE_NORM=0 = exact)")
