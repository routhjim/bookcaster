"""11-01: torch.compile reads a threading.local attribute as its class default, not the lane thread's value, so a
compiled block's runtime LoRA must read its card's gate (BLOCK_GATES), synced from the lane state by gate_from under
the card lock. Checks: attach puts block Linears on the gate and leaves non-block Linears on the lane state; compiled
in a worker thread, LoRA ON == eager ON and OFF == eager OFF. CPU only: HIP_VISIBLE_DEVICES= ~/tts-ab/.venv-ltx/bin/python"""
import sys
import tempfile
import threading
from pathlib import Path
import torch
import safetensors.torch as stt
sys.path.insert(0, str(Path(__file__).parent))
import ltx_split_next as R

fails = []


class Lane(threading.local):
    on = False


lane = Lane()


class Blk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64, 48)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([Blk()])
        self.head = torch.nn.Linear(48, 48)


torch.manual_seed(0)
toy = Toy()
with tempfile.TemporaryDirectory() as td:
    fp = f"{td}/l.safetensors"
    stt.save_file({"diffusion_model.transformer_blocks.0.lin.lora_A.weight": torch.randn(8, 64).to(torch.bfloat16),
                   "diffusion_model.transformer_blocks.0.lin.lora_B.weight": torch.randn(48, 8).to(torch.bfloat16),
                   "diffusion_model.head.lora_A.weight": torch.randn(8, 48).to(torch.bfloat16),
                   "diffusion_model.head.lora_B.weight": torch.randn(48, 8).to(torch.bfloat16)}, fp)
    R.attach_runtime_lora(toy, fp, 0.8, lane, block_gate=True)
if toy.transformer_blocks[0].lin._runtime_loras[0][2] is not R.BLOCK_GATES[0]:
    fails.append("a block Linear is not on its card's gate")
if toy.head._runtime_loras[0][2] is not lane:
    fails.append("a non-block Linear left the lane state")

blk = toy.transformer_blocks[0].lin
cb = torch.compile(blk, dynamic=True)
x = torch.randn(200, 64)
res = {}
with torch.inference_mode():         # dynamo's FIRST trace from a worker thread segfaults on CPU (fake-tensor setup);
    cb(torch.randn(100, 64))         # warm it in the main thread (gate off), the worker then switches it on


lane2 = Lane()
m = torch.nn.Linear(64, 48)
R._lora_forward(m).append((torch.randn(8, 64), torch.randn(48, 8), lane2))
cm = torch.compile(m, dynamic=True)
with torch.inference_mode():
    cm(torch.randn(100, 64))


def worker():
    with torch.inference_mode():
        lane.on = True
        R.gate_from(lane, 0)
        res["eager_on"], res["comp_on"] = blk(x), cb(x)
        lane.on = False
        R.gate_from(lane, 0)
        res["eager_off"], res["comp_off"] = blk(x), cb(x)


th = threading.Thread(target=worker)
th.start()
th.join()
eff = (res["eager_on"] - res["eager_off"]).abs().max().item()
d_on = (res["comp_on"] - res["eager_on"]).abs().max().item()
d_off = (res["comp_off"] - res["eager_off"]).abs().max().item()
# (the LoRA dropped = a diff the size of its effect; this fp32 toy's fused bf16 -> fp32 cast differs ~0.2%)
if eff == 0 or d_on > 0.01 * eff:
    fails.append(f"compiled block LoRA ON in a lane thread != eager ON (diff {d_on:.3g}, LoRA effect {eff:.3g})")
if d_off > 0.01 * eff:
    fails.append(f"compiled block LoRA OFF != eager OFF (diff {d_off:.3g})")

# the bug itself, for the record: a compiled Linear reading the threading.local directly ignores the lane's value
bug = {}


def worker2():
    with torch.inference_mode():
        lane2.on = True
        bug["eager"], bug["comp"] = m(x), cm(x)
        lane2.on = False
        bug["off"] = m(x)


th = threading.Thread(target=worker2)
th.start()
th.join()
print({"gate_on_diff": round(d_on, 4), "lora_effect": round(eff, 2),
       "threading.local compiled == eager OFF (the bug)": torch.allclose(bug["comp"], bug["off"], atol=1e-3)})
print({"fails": fails})
sys.exit(1 if fails else 0)
