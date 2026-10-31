"""10-31: the runtime-LoRA forward folds active LoRAs into the weight on big inputs and adds them unfused on small
ones; both must equal the reference x W^T + b + sum (x A^T) B^T, an inactive LoRA must not count, and a bf16
weight must never be modified in place. CPU, fp32: run with the LTX venv python."""
import sys, types
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from ltx_split_next import _lora_forward

torch.manual_seed(0)
fails = []
for dtype in (torch.float32, torch.bfloat16):
    m = torch.nn.Linear(64, 48).to(dtype)
    w0 = m.weight.detach().clone()
    on, off = types.SimpleNamespace(on=True), types.SimpleNamespace(on=False)
    A1, B1, A2, B2 = (torch.randn(8, 64), torch.randn(48, 8) * 0.1, torch.randn(4, 64), torch.randn(48, 4) * 0.1)
    _lora_forward(m).append((A1.to(dtype), B1.to(dtype), on))
    _lora_forward(m).append((A2.to(dtype), B2.to(dtype), off))
    tol = 1e-4 if dtype == torch.float32 else 0.15
    for rows in (2, 200):        # 64*48/(64+48) = 27.4: 2 rows unfused, 200 rows folded
        x = torch.randn(rows, 64).to(dtype)
        ref = (x.float() @ w0.float().T + m.bias.float() + (x.float() @ A1.T) @ B1.T)
        got = m(x).float()
        err = (got - ref).abs().max().item()
        if err > tol:
            fails.append(f"{dtype} rows {rows}: max err {err:.4g}")
    if not torch.equal(m.weight.detach(), w0):
        fails.append(f"{dtype}: weight changed in place")
    on.on = False
    x = torch.randn(200, 64).to(dtype)
    if (m(x).float() - (x.float() @ w0.float().T + m.bias.float())).abs().max().item() > tol:
        fails.append(f"{dtype}: LoRA off still applied")
print({"fails": fails})
sys.exit(1 if fails else 0)
