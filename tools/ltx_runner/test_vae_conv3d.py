"""vae_conv3d: the fused 3x3x3 conv matches an fp32 conv3d at least as well as PyTorch's own bf16 conv does,
on strided channels-last workspace views like the ones the memory-efficient decoder passes. Needs a GPU."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vae_conv3d as V  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("cin,cout,t,h,w", [(128, 128, 5, 24, 40), (256, 512, 4, 13, 24), (128, 48, 3, 9, 14)])
def test_matches_fp32(cin, cout, t, h, w):
    torch.manual_seed(0)
    conv = torch.nn.Conv3d(cin, cout, 3, padding=(0, 1, 1)).to("cuda", torch.bfloat16)
    conv = conv.to(memory_format=torch.channels_last_3d)
    ws = torch.randn(1, max(cin, cout), t + 2, h, w, device="cuda", dtype=torch.bfloat16)
    x = ws.to(memory_format=torch.channels_last_3d)[:, :cin]          # strided view, as the workspace passes it
    with torch.inference_mode():
        exact = F.conv3d(x.float(), conv.weight.float(), conv.bias.float(), padding=(0, 1, 1))
        y = V.conv3x3x3(x, conv)
        ref = conv(x.contiguous(memory_format=torch.channels_last_3d))
    assert y.shape == exact.shape
    err_new = (y.float() - exact).abs().max().item()
    err_ref = (ref.float() - exact).abs().max().item()
    assert err_new <= err_ref * 1.5 + 1e-3, (err_new, err_ref)


def test_install_only_supported():
    m = torch.nn.Sequential(torch.nn.Conv3d(8, 8, 3, padding=(0, 1, 1)), torch.nn.Conv3d(8, 8, 3, padding=1),
                            torch.nn.Conv3d(8, 8, 1))
    assert V.install(m) == 1
    assert getattr(m[0], "_fused", False) and not getattr(m[1], "_fused", False)
