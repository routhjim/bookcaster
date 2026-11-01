"""11-01: quiet clips in the two-stage hold. (1) job_route: a quiet job ("route": "i2v") takes the distilled route
only where the hold has it, audio -> a2vid, else ti2v; (2) the distilled LoRA attached once at 1.0 and scaled per job
gives the SAME bits as the old pre-scaled B (0.8 two-stage, 1.0 quiet), folded and unfused; (3) under torch.compile
a strength switch (0.8 <-> 1.0) compiles no new graph. CPU only: HIP_VISIBLE_DEVICES= ~/tts-ab/.venv-ltx/bin/python"""
import sys
import types
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import ltx_split_next as R

fails = []
# (1) routes
q, t, a = {"route": "i2v", "frames": 65}, {"audio": "x.wav", "frames": 97}, {"route": "ti2v", "neg_extra": "", "frames": 49}
ALL = ["a2vid", "ti2v", "distilled"]
for j, routes, want in ((q, ALL, "distilled"), (t, ALL, "a2vid"), (a, ALL, "ti2v"), (q, ["a2vid", "ti2v"], "ti2v"),
                        (q, ["ti2v", "distilled"], "distilled"), (a, ["ti2v", "distilled"], "ti2v"),
                        (q, ["a2vid", "distilled"], "distilled"), (t, ["a2vid", "distilled"], "a2vid")):
    if R.job_route(j, routes) != want:
        fails.append(f"route {j} in {routes}: {R.job_route(j, routes)} != {want}")
if R.job_route(q, None) != "distilled" or R.job_route(a, None, True) != "ti2v":
    fails.append("order_jobs routes")

# (2) runtime scale == pre-scaled B, bit for bit
torch.manual_seed(0)
for dtype in (torch.bfloat16, torch.float32):
    m_old, m_new = torch.nn.Linear(64, 48).to(dtype), torch.nn.Linear(64, 48).to(dtype)
    m_new.load_state_dict(m_old.state_dict())
    # the LoRA file is bf16: attach read B exactly (bf16 -> fp32 -> x strength -> bf16)
    A, B = torch.randn(8, 64), (torch.randn(48, 8) * 0.1).to(torch.bfloat16).float()
    for s in (0.8, 1.0):
        old = types.SimpleNamespace(on=True)
        new = types.SimpleNamespace(on=True, scales=R.scales_for(s))
        for m_ in (m_old, m_new):
            if getattr(m_, "_runtime_loras", None) is not None:
                m_._runtime_loras.clear()
        R._lora_forward(m_old).append((A.to(torch.bfloat16), (B.float() * s).to(torch.bfloat16), old))   # old attach
        R._lora_forward(m_new).append((A.to(torch.bfloat16), B.to(torch.bfloat16), new))                 # at 1.0
        for rows in (2, 200):        # 64*48/(64+48) = 27.4: 2 rows unfused, 200 rows folded
            x = torch.randn(rows, 64).to(dtype)
            if not torch.equal(m_old(x), m_new(x)):
                fails.append(f"{dtype} s={s} rows {rows}: runtime scale != pre-scaled B")

# (3) torch.compile: a strength switch is a graph input, not a guard
try:
    import torch._dynamo as dyn
    dyn.reset()
    m = torch.nn.Linear(64, 48).to(torch.bfloat16)
    st = types.SimpleNamespace(on=True, scales=R.scales_for(0.8))
    R._lora_forward(m).append((torch.randn(8, 64).to(torch.bfloat16), (torch.randn(48, 8) * 0.1).to(torch.bfloat16), st))
    f = torch.compile(m, dynamic=True)
    x = torch.randn(200, 64).to(torch.bfloat16)
    with torch.inference_mode():
        y8 = f(x)
        g0 = dyn.utils.counters["stats"].get("unique_graphs", 0)
        st.scales = R.scales_for(1.0)
        y1 = f(x)
        st.scales = R.scales_for(0.8)
        y8b = f(x)
    g1 = dyn.utils.counters["stats"].get("unique_graphs", 0)
    if g1 != g0:
        fails.append(f"compile: {g1 - g0} new graph(s) on a strength switch")
    if torch.equal(y8, y1) or not torch.equal(y8, y8b):
        fails.append("compile: the switched strength was not applied")
    print({"graphs": g1, "recompiles": dict(dyn.utils.counters.get("recompiles", {}))})
except Exception as e:                 # noqa: BLE001  (no C compiler etc.: report, don't hide)
    fails.append(f"compile test could not run: {e!r}"[:300])
# (4) 11-01: torch.compile reads a threading.local attribute as the class default, not the lane's value - a block's
# LoRA reads its card's gate (BLOCK_GATES), synced from the lane state under the card lock; compiled ON in a worker
# thread must equal eager ON
try:
    import tempfile
    import threading
    import safetensors.torch as stt

    class Lane(threading.local):
        on = False
        scales = None
        unit = False

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

    toy = Toy()
    with tempfile.TemporaryDirectory() as td:
        fp = f"{td}/l.safetensors"
        stt.save_file({"diffusion_model.transformer_blocks.0.lin.lora_A.weight": torch.randn(8, 64).to(torch.bfloat16),
                       "diffusion_model.transformer_blocks.0.lin.lora_B.weight": torch.randn(48, 8).to(torch.bfloat16),
                       "diffusion_model.head.lora_A.weight": torch.randn(8, 48).to(torch.bfloat16),
                       "diffusion_model.head.lora_B.weight": torch.randn(48, 8).to(torch.bfloat16)}, fp)
        R.attach_runtime_lora(toy, fp, 1.0, lane, block_gate=True)
    if toy.transformer_blocks[0].lin._runtime_loras[0][2] is not R.BLOCK_GATES[0]:
        fails.append("a block Linear is not on its card's gate")
    if toy.head._runtime_loras[0][2] is not lane:
        fails.append("a non-block Linear left the lane state")
    blk = toy.transformer_blocks[0].lin
    cb = torch.compile(blk, dynamic=True)
    x = torch.randn(200, 64)
    res = {}

    def worker():
        with torch.inference_mode():
            lane.on, lane.scales, lane.unit = True, R.scales_for(0.8), False
            R.gate_from(lane, 0)
            res["eager_on"], res["comp_on"] = blk(x), cb(x)
            lane.on = False
            R.gate_from(lane, 0)
            res["eager_off"], res["comp_off"] = blk(x), cb(x)
    th = threading.Thread(target=worker)
    th.start()
    th.join()
    d_on = (res["comp_on"] - res["eager_on"]).abs().max().item()
    eff = (res["eager_on"] - res["eager_off"]).abs().max().item()
    # (the LoRA dropped = a diff the size of its effect; a fused bf16 -> fp32 cast without the intermediate rounding
    # on this fp32 toy = ~0.2%: the GPU path is bf16 end to end, bit-exactness there is the A/B's job)
    if d_on > 0.01 * eff or eff == 0:
        fails.append(f"compiled block LoRA ON in a lane thread != eager ON (diff {d_on:.3g}, LoRA effect {eff:.3g})")
    if not torch.allclose(res["comp_off"], res["eager_off"], atol=1e-3):
        fails.append("compiled block LoRA OFF != eager OFF")
except Exception as e:                 # noqa: BLE001
    fails.append(f"gate test could not run: {e!r}"[:300])
print({"fails": fails})
sys.exit(1 if fails else 0)
