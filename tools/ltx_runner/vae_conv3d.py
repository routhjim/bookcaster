"""Fused 3x3x3 Conv3d for the LTX VAE decoder (11-03).

Every Conv3d in the LTX-2.5 video decoder is kernel 3x3x3, stride 1, padding (0,1,1) (time is padded by the
caller's workspace), zero spatial padding, with bias, on channels_last_3d bf16 activations. With MIOpen off
(its conv3d is wrong/slow on gfx1100) PyTorch runs these as slow_conv_dilated3d: an explicit im2col buffer
(27 x Cin x T x H x W -- 2 GiB for one 16-frame chunk at 128 ch, 104x192) plus ~130 small copy kernels per
call, then one GEMM. Profile of a 177-frame decode: GEMM 36% of GPU time, the rest is that overhead, and the
2 GiB scratch is allocated/freed ~1900 times per decode.

This is an implicit GEMM: each program computes a BLOCK_M (output voxels) x BLOCK_N (out channels) tile,
looping over the 27 taps x Cin, loading the input straight from the (possibly strided) channels-last view.
No im2col, no layout copies, fp32 accumulation, bias fused. Output is a fresh channels_last_3d tensor.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _configs():
    out = []
    for bm, bn, bk, w in [(128, 128, 32, 8), (128, 64, 32, 4), (64, 128, 32, 4), (64, 64, 64, 4),
                          (128, 128, 64, 8), (256, 64, 32, 8), (64, 256, 32, 8)]:
        out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=2))
    return out


@triton.autotune(configs=_configs(), key=["Cin", "Cout", "HW_BUCKET"], cache_results=True)  # tuned once, on disk
@triton.jit
def _conv3x3x3_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    T, H, W, Cin, Cout, HW_BUCKET,
    sxb, sxc, sxt, sxh, sxw,
    syb, syc, syt, syh, syw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = tl.program_id(2)

    HWo = H * W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_ok = offs_m < T * HWo
    t = offs_m // HWo
    hw = offs_m % HWo
    h = hw // W
    w = hw % W
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < Cout
    offs_k = tl.arange(0, BLOCK_K)

    x_b = x_ptr + b.to(tl.int64) * sxb
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tap in tl.static_range(27):
        dt = tap // 9
        dh = (tap // 3) % 3 - 1
        dw = tap % 3 - 1
        hh = h + dh
        ww = w + dw
        ok = m_ok & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
        row = (t + dt).to(tl.int64) * sxt + hh.to(tl.int64) * sxh + ww.to(tl.int64) * sxw
        w_tap = w_ptr + tap * Cin * Cout
        for k0 in range(0, Cin, BLOCK_K):
            kk = k0 + offs_k
            k_ok = kk < Cin
            a = tl.load(x_b + row[:, None] + kk[None, :].to(tl.int64) * sxc, mask=ok[:, None] & k_ok[None, :], other=0.0)
            bw = tl.load(w_tap + kk[:, None] * Cout + offs_n[None, :], mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            acc = tl.dot(a, bw, acc)
    bias = tl.load(b_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    y_off = (b.to(tl.int64) * syb + t.to(tl.int64)[:, None] * syt + h.to(tl.int64)[:, None] * syh
             + w.to(tl.int64)[:, None] * syw + offs_n[None, :].to(tl.int64) * syc)
    tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


_WCACHE: dict[int, tuple[torch.Tensor, int]] = {}


def _packed_weight(conv: torch.nn.Conv3d) -> torch.Tensor:
    """[Cout, Cin, 3, 3, 3] -> [27, Cin, Cout] contiguous, cached per module (invalidated if the weight changes)."""
    key = id(conv)
    hit = _WCACHE.get(key)
    if hit is not None and hit[1] == conv.weight._version:
        return hit[0]
    wp = conv.weight.detach().permute(2, 3, 4, 1, 0).reshape(27, conv.in_channels, conv.out_channels).contiguous()
    _WCACHE[key] = (wp, conv.weight._version)
    return wp


def conv3x3x3(x: torch.Tensor, conv: torch.nn.Conv3d) -> torch.Tensor:
    """y = conv(x) for x [B, Cin, T+2, H, W] (any strides), conv 3x3x3 / stride 1 / padding (0,1,1) / zeros."""
    B, Cin, Tin, H, W = x.shape
    T = Tin - 2
    Cout = conv.out_channels
    y = torch.empty((B, Cout, T, H, W), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
    wp = _packed_weight(conv)
    bias = conv.bias if conv.bias is not None else torch.zeros(Cout, device=x.device, dtype=x.dtype)
    hw_bucket = 1 << max(0, (H * W - 1).bit_length())  # autotune key: a few buckets, not every tile shape

    def grid(meta):
        return (triton.cdiv(T * H * W, meta["BLOCK_M"]), triton.cdiv(Cout, meta["BLOCK_N"]), B)

    _conv3x3x3_kernel[grid](x, wp, bias, y, T, H, W, Cin, Cout, hw_bucket,
                            *x.stride(), *y.stride())
    return y


def supported(conv: torch.nn.Module) -> bool:
    return (isinstance(conv, torch.nn.Conv3d) and conv.kernel_size == (3, 3, 3) and conv.stride == (1, 1, 1)
            and conv.padding == (0, 1, 1) and conv.dilation == (1, 1, 1) and conv.groups == 1
            and conv.padding_mode == "zeros")


def install(decoder: torch.nn.Module) -> int:
    """Route every supported Conv3d inside `decoder` through the fused kernel. LTX_VAE_FUSED_CONV=0 = off."""
    if os.environ.get("LTX_VAE_FUSED_CONV", "1") == "0":
        return 0
    n = 0
    for m in decoder.modules():
        if supported(m) and not getattr(m, "_fused", False):
            m.forward = (lambda mod: (lambda inp: conv3x3x3(inp, mod)))(m)
            m._fused = True
            n += 1
    return n
