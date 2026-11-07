"""One Triton kernel for the Qwen-Image DiT's q/k norm + RoPE (E19). Eager runs about nine kernels and copies per q or k:
float cast, pow, mean, add eps, rsqrt, promote, mul, cast, mul by the per-head weight, cast, complex mul, cast back.
Here: read once, write once, rounding to bf16 where eager does. Not bit-exact, the mean sums in Triton's order
(tools/tests/test_qwen_fuse.py bounds it at one bf16 step). QWEN_IMAGE_FUSE=0 = off.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_kernel(x_ptr, w_ptr, f_ptr, o_ptr, H, D, eps, BR: tl.constexpr):
    row = tl.program_id(0)                               # one (token, head); x and o [1, S, H, D] contiguous
    s = row // H
    r = tl.arange(0, BR)
    base = row * D + 2 * r
    re = tl.load(x_ptr + base).to(tl.float32)
    im = tl.load(x_ptr + base + 1).to(tl.float32)
    rs = tl.rsqrt((tl.sum(re * re + im * im, axis=0)) / D + eps)
    wr = tl.load(w_ptr + 2 * r).to(tl.float32)
    wi = tl.load(w_ptr + 2 * r + 1).to(tl.float32)
    a = ((re * rs).to(tl.bfloat16).to(tl.float32) * wr).to(tl.bfloat16).to(tl.float32)
    b = ((im * rs).to(tl.bfloat16).to(tl.float32) * wi).to(tl.bfloat16).to(tl.float32)
    fr = tl.load(f_ptr + s * D + 2 * r)                  # the complex64 [S, D/2] as its fp32 pairs
    fi = tl.load(f_ptr + s * D + 2 * r + 1)
    tl.store(o_ptr + base, (a * fr - b * fi).to(tl.bfloat16))
    tl.store(o_ptr + base + 1, (a * fi + b * fr).to(tl.bfloat16))


def norm_rope(x: torch.Tensor, norm, freqs: torch.Tensor) -> torch.Tensor:
    """apply_rotary_emb_qwen(norm(x), freqs, use_real=False) for a bf16 [1, S, H, D] x, RMSNorm `norm`, complex [S, D/2]."""
    _, S, H, D = x.shape
    x = x.contiguous()
    f = torch.view_as_real(freqs)
    if not f.is_contiguous():
        f = f.contiguous()
    w = norm.weight
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _norm_rope_kernel[(S * H,)](x, w, f, out, H, D, float(norm.eps), BR=D // 2, num_warps=4)
    return out


def applies(x: torch.Tensor, norm, freqs) -> bool:
    return (x.ndim == 4 and x.shape[0] == 1 and x.dtype == torch.bfloat16 and x.is_cuda and x.shape[-1] in (64, 128, 256)
            and torch.is_tensor(freqs) and freqs.is_complex() and freqs.ndim == 2 and freqs.shape == (x.shape[1], x.shape[-1] // 2)
            and norm.weight is not None and norm.bias is None and norm.weight.dtype == torch.bfloat16
            and norm.weight.is_contiguous())


# The rest of a KV-cached step's block, where every token is the target's: x * (1 + scale) after a LayerNorm and the two
# gated residuals, one read and one write each instead of 2-3 kernels with bf16 scratch; and no full-size torch.where
# for the modulation rows, which are all the first row there. Rounding as eager (gate_add bit-exact on test data); the
# norm's reduction runs in Triton's order, so that one is within a bf16 step. silu(a) * b stayed eager: a kernel for it
# was slower (1.88 v 1.54 ms).
@triton.jit
def _ln_mod_kernel(x_ptr, s_ptr, o_ptr, D, eps, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    m = d < D
    x = tl.load(x_ptr + row * D + d, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    c = tl.where(m, x - mean, 0.0)
    r = tl.rsqrt(tl.sum(c * c, axis=0) / D + eps)
    n = (c * r).to(tl.bfloat16).to(tl.float32)
    s = tl.load(s_ptr + d, mask=m, other=0.0).to(tl.float32)
    one = (1.0 + s).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + row * D + d, (n * one).to(tl.bfloat16), mask=m)


@triton.jit
def _gate_add_kernel(x_ptr, y_ptr, g_ptr, o_ptr, D, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    m = d < D
    x = tl.load(x_ptr + row * D + d, mask=m, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * D + d, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + d, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * D + d, (x + (g * y).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), mask=m)


def ln_mod(x, scale, eps):
    """LayerNorm(no affine)(x) * (1 + scale) for bf16 x [1, S, D], scale [D]."""
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    x = x.contiguous()
    _ln_mod_kernel[(x.shape[1],)](x, scale, out, x.shape[2], float(eps), BD=triton.next_power_of_2(x.shape[2]), num_warps=8)
    return out


def gate_add(x, y, g):
    """x + g * y for bf16 x, y [1, S, D], g [D]."""
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    x, y = x.contiguous(), y.contiguous()
    _gate_add_kernel[(x.shape[1],)](x, y, g, out, x.shape[2], BD=triton.next_power_of_2(x.shape[2]), num_warps=8)
    return out


def _install_block(m) -> None:
    Block = m.QwenImage21TransformerBlock
    orig = Block.forward

    def forward(self, hidden_states, modulation, rotary_emb=None, attention_mask=None, target_token_mask=None,
                layer_cache=None, kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        x = hidden_states
        if (kv_cache_mode != "cached" or segments is not None or x.dtype != torch.bfloat16 or x.ndim != 3 or x.shape[0] != 1
                or modulation.shape[0] != 2):
            return orig(self, hidden_states, modulation, rotary_emb, attention_mask, target_token_mask, layer_cache,
                        kv_cache_mode, cache_write_slice, segments, key_valid)
        m1, m2 = modulation[0].chunk(2, dim=-1)         # the first row: every token here is the target's
        s1, g1 = m1.chunk(2, dim=-1)
        s2, g2 = m2.chunk(2, dim=-1)
        a = self.attn(hidden_states=ln_mod(x, s1, self.img_norm1.eps), attention_mask=attention_mask, rotary_emb=rotary_emb,
                      layer_cache=layer_cache, kv_cache_mode=kv_cache_mode, cache_write_slice=cache_write_slice,
                      segments=segments, key_valid=key_valid)
        x = gate_add(x, a, g1.tanh())
        y = ln_mod(x, s2, self.img_norm2.eps)
        mlp = self.img_mlp
        y = mlp.out(mlp.activation_fn(mlp.gate_layer(y)) * mlp.proj(y))
        return gate_add(x, y, g2.tanh())
    Block.forward = forward


def install(log=print) -> bool:
    """Route the DiT's q/k through norm_rope; anything else takes the original function's path."""
    if os.environ.get("QWEN_IMAGE_FUSE", "1") != "1":
        return False
    import diffusers.models.transformers.transformer_qwenimage21 as m

    if getattr(m, "_qwen_fused", False):
        return True

    def prepare(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        head_dim = attn.inner_dim // attn.heads
        query = query.unflatten(-1, (-1, head_dim))
        key = key.unflatten(-1, (-1, head_dim))
        value = value.unflatten(-1, (-1, head_dim))
        if rotary_emb is not None and applies(query, attn.norm_q, rotary_emb) and applies(key, attn.norm_k, rotary_emb):
            query, key = norm_rope(query, attn.norm_q, rotary_emb), norm_rope(key, attn.norm_k, rotary_emb)
        else:
            query = attn.norm_q(query).to(value.dtype)
            key = attn.norm_k(key).to(value.dtype)
            if rotary_emb is not None:
                query = m.apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
                key = m.apply_rotary_emb_qwen(key, rotary_emb, use_real=False)
        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get()
                key = torch.cat([cached_k, key], dim=1)
                value = torch.cat([cached_v, value], dim=1)
        return query, key, value, query.shape[1]

    m._qwenimage21_prepare_qkv = prepare
    if os.environ.get("QWEN_IMAGE_FUSE_BLOCK", "1") == "1":
        _install_block(m)
    m._qwen_fused = True
    log("q/k norm + RoPE fused")
    return True


# The VAE decoder's 3x3 convs, as tools/ltx_runner/vae_conv3d.py does the LTX decoder's. With MIOpen off a bf16 conv is
# PyTorch's im2col: most of the decode's 2.6 GiB peak and small copies around one GEMM. This is an implicit GEMM: each
# program computes a BLOCK_M (output pixels) x BLOCK_N (out channels) tile over the 9 taps x Cin, out-of-range taps read
# as zero (the pad), fp32 accumulation, bias fused. Output within the conv's own bf16 error (tests).
@triton.jit
def _conv3x3_kernel(x_ptr, w_ptr, b_ptr, y_ptr, H, W, Cin, Cout, sxc, sxh, sxw, syc, syh, syw,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_ok = offs_m < H * W
    h = offs_m // W
    w = offs_m % W
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < Cout
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tap in tl.static_range(9):
        hh = h + tap // 3 - 1
        ww = w + tap % 3 - 1
        ok = m_ok & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
        row = hh.to(tl.int64) * sxh + ww.to(tl.int64) * sxw
        w_tap = w_ptr + tap * Cin * Cout
        for k0 in range(0, Cin, BLOCK_K):
            kk = k0 + offs_k
            k_ok = kk < Cin
            a = tl.load(x_ptr + row[:, None] + kk[None, :].to(tl.int64) * sxc, mask=ok[:, None] & k_ok[None, :], other=0.0)
            bw = tl.load(w_tap + kk[:, None] * Cout + offs_n[None, :], mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            acc = tl.dot(a, bw, acc)
    acc = acc + tl.load(b_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
    y_off = (h.to(tl.int64)[:, None] * syh + w.to(tl.int64)[:, None] * syw + offs_n[None, :].to(tl.int64) * syc)
    tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


CONV_BLOCKS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4}    # best of 4 on the iGPU, 3 rounds (E19)


def _packed(conv: torch.nn.Conv2d) -> torch.Tensor:
    """[Cout, Cin, 3, 3] -> [9, Cin, Cout], made once; install_vae then drops the original (0.46 GiB over the 40 convs)."""
    if getattr(conv, "_packed_w", None) is None:
        conv._packed_w = conv.weight.detach().permute(2, 3, 1, 0).reshape(9, conv.in_channels, conv.out_channels).contiguous()
    return conv._packed_w


def conv3x3(x: torch.Tensor, conv: torch.nn.Conv2d) -> torch.Tensor:
    """conv(x) for bf16 x [1, Cin, H, W], kernel 3x3, stride 1, one pixel of zero pad around x."""
    _, Cin, H, W = x.shape
    Cout = conv.out_channels
    wp = _packed(conv)
    x = x[0]
    y = torch.empty((Cout, H, W), device=x.device, dtype=x.dtype)
    bm, bn = CONV_BLOCKS["BLOCK_M"], CONV_BLOCKS["BLOCK_N"]
    _conv3x3_kernel[((H * W + bm - 1) // bm, (Cout + bn - 1) // bn)](x, wp, conv.bias, y, H, W, Cin, Cout, *x.stride(), *y.stride(),
                                                                     BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=CONV_BLOCKS["BLOCK_K"],
                                                                     num_warps=CONV_BLOCKS["num_warps"])
    return y[None]


def _conv_supported(c: torch.nn.Module) -> bool:
    pad = getattr(c, "_padding", None)            # QwenImage21CausalConv3d pads one pixel itself (F.pad, padding 0)
    return (isinstance(c, torch.nn.Conv2d) and c.kernel_size == (3, 3) and c.stride == (1, 1) and c.dilation == (1, 1)
            and c.groups == 1 and c.bias is not None and c.padding_mode == "zeros" and c.weight.dtype == torch.bfloat16
            and (tuple(pad) == (1, 1, 1, 1) if pad is not None else c.padding == (1, 1)))


def install_vae(decoder: torch.nn.Module, log=print) -> int:
    """Route the decoder's 3x3 convs through conv3x3; anything else in it keeps torch's conv. QWEN_IMAGE_FUSE=0 = off."""
    if os.environ.get("QWEN_IMAGE_FUSE", "1") != "1":
        return 0
    n = 0
    for c in decoder.modules():
        if not _conv_supported(c):
            continue
        if hasattr(c, "_padding"):
            def fwd(x, cache_x=None, c=c):
                if cache_x is not None:
                    raise ValueError("no feature cache in the image VAE")
                return conv3x3(x.squeeze(2), c).unsqueeze(2)
        else:
            def fwd(x, c=c):
                return conv3x3(x, c)
        c.forward = fwd
        _packed(c)
        c._parameters["weight"] = None
        n += 1
    log(f"VAE: {n} 3x3 convs fused")
    return n
