# suite: manual
"""E19: the fused q/k norm + RoPE stays within one bf16 step of eager's expression, on real-shaped data, and refuses
what it cannot do. Run with a ROCm venv that has diffusers and Triton; skips (exit 0) without a GPU.

    HIP_VISIBLE_DEVICES=$(~/bin/hip-index igpu) ~/tts-ab/.venv-charlora-igpu/bin/python -I tools/tests/test_qwen_fuse.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError("no GPU")
    import qwen_fuse
    from diffusers.models.normalization import RMSNorm
    from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen
except ImportError as exc:
    print(f"skipped: {exc}")
    sys.exit(0)

torch.manual_seed(0)
fails = []
S, H, D = 2880, 32, 128
norm = RMSNorm(D, eps=1e-6).to("cuda", torch.bfloat16)
norm.weight.data = (torch.randn(D, device="cuda") * 0.3 + 1).to(torch.bfloat16)
x = (torch.randn(1, S, H, D, device="cuda") * torch.exp(torch.randn(1, S, H, 1, device="cuda") * 2)).to(torch.bfloat16)
ang = torch.rand(S, D // 2, device="cuda") * 6.283
freqs = torch.polar(torch.ones_like(ang), ang)
if not qwen_fuse.applies(x, norm, freqs):
    fails.append("applies() refused the DiT's shape")
with torch.inference_mode():
    ref = apply_rotary_emb_qwen(norm(x).to(x.dtype), freqs, use_real=False)
    got = qwen_fuse.norm_rope(x, norm, freqs)
diff = (got != ref)
frac = float(diff.float().mean())
# the normed values sit around 1, where one bf16 step is 2^-7 (a ratio to ref itself would blow up at the rotation's zeros)
dmax = float((got.float() - ref.float()).abs().max())
print({"moved": round(frac, 6), "max_abs": round(dmax, 5)})
if dmax > 2 ** -5:
    fails.append(f"too far from eager: {dmax}")
for bad in (x[0], x.float()):
    if qwen_fuse.applies(bad, norm, freqs):
        fails.append(f"applies() took {bad.shape} {bad.dtype}")

# the cached block's elementwise pieces, against eager's expressions
S4, D4 = 2880, 4096
x4 = (torch.randn(1, S4, D4, device="cuda") * 2).to(torch.bfloat16)
y4 = torch.randn(1, S4, D4, device="cuda").to(torch.bfloat16)
s4 = (torch.randn(D4, device="cuda") * 0.2).to(torch.bfloat16)
g4 = (torch.randn(D4, device="cuda") * 0.5).to(torch.bfloat16)
ln = torch.nn.LayerNorm(D4, elementwise_affine=False, eps=1e-6)
with torch.inference_mode():
    checks = {"ln_mod": (qwen_fuse.ln_mod(x4, s4, 1e-6), ln(x4) * (1 + s4)),
              "gate_add": (qwen_fuse.gate_add(x4, y4, g4), x4 + g4 * y4)}
for name, (got4, ref4) in checks.items():
    moved4 = float((got4 != ref4).float().mean())
    err4 = float((got4.float() - ref4.float()).abs().max())
    print({name: {"moved": round(moved4, 5), "max_abs": round(err4, 5)}})
    if err4 > 2 ** -3:
        fails.append(f"{name}: {err4}")

# the VAE's 3x3 conv: at least as close to an fp32 conv as torch's own bf16 conv, with the pad and with a causal one
import torch.nn.functional as F
for cin, cout, h, w in ((96, 96, 40, 72), (384, 192, 20, 36), (64, 3, 9, 14)):
    torch.manual_seed(1)
    conv = torch.nn.Conv2d(cin, cout, 3, padding=1).to("cuda", torch.bfloat16)
    y = torch.randn(1, cin, h, w, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        exact = F.conv2d(y.float(), conv.weight.float(), conv.bias.float(), padding=1)
        got = qwen_fuse.conv3x3(y, conv)
        ref = F.conv2d(y, conv.weight, conv.bias, padding=1)
    err_new, err_ref = (got.float() - exact).abs().max().item(), (ref.float() - exact).abs().max().item()
    print({"conv": (cin, cout, h, w), "err_fused": round(err_new, 4), "err_torch": round(err_ref, 4)})
    if got.shape != exact.shape or err_new > err_ref * 1.5 + 1e-3:
        fails.append(f"conv {cin}->{cout}: {err_new} v torch {err_ref}")
if qwen_fuse._conv_supported(torch.nn.Conv2d(4, 4, 3, stride=2, padding=1).to(torch.bfloat16)):
    fails.append("took a stride-2 conv")
if qwen_fuse._conv_supported(torch.nn.Conv2d(4, 4, 1).to(torch.bfloat16)):
    fails.append("took a 1x1 conv")
print({"fails": fails})
sys.exit(1 if fails else 0)
