"""Draw a list of pictures with FLUX.2 [klein] 4B (Apache 2.0, 4-step distilled).

Runs as a one-shot batch on the GPU that ACE-Step uses, and fits beside it
(~12 GB free on a 24 GB card) by never holding both halves of the model:
the text encoder (Qwen3 4B, 8 GB) encodes every prompt first and is freed,
then the transformer (7.8 GB) + VAE draw them. Weights load straight to the
GPU (no CPU copy), so system RAM stays low.

    ~/tts-ab/.venv-ace/bin/python -I tools/illustrate.py jobs.json OUT_DIR

jobs.json: [{"id": "p001", "prompt": "...", "seed": 1,
             "path": "/abs/out.png" (default OUT_DIR/<id>.png),
             "size": [w, h] (default --width/--height),
             "refs": ["/abs/ref.png", ...] (reference pictures to keep a
                     character or place looking the same; jobs run in order,
                     so a reference drawn earlier in the list can be used)}]
Skips jobs whose picture exists. One JSON line per picture on stdout.

    ... -I tools/illustrate.py --serve      (jobs as JSON lines on stdin)
keeps both halves loaded for a whole keyframe phase (video.drawing_session).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL = Path(os.environ.get("ILLUSTRATE_MODEL", Path.home() / "models" / "flux2-klein-4b")).expanduser()


def load_direct(model: torch.nn.Module, folder: Path, device: str) -> torch.nn.Module:
    """Fill a meta-device model from its safetensors onto the GPU one tensor
    at a time, so no full CPU copy of the weights exists (the box has only a
    few GB to spare while Strata is up). Each tensor is first copied into
    ordinary memory: .to(device) straight from the mmapped file makes ROCm
    keep those pages (+3.5 GB per 60 tensors, measured), and the pinned-
    memory paths (safetensors device="cuda", a pinned staging buffer) hit
    SDMA permission faults on the dock-attached XTX, corrupting the weights
    and eventually resetting the GPU."""
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    for shard in sorted(folder.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k).to(torch.bfloat16).clone()
                # Copies to the dock-attached XTX are occasionally corrupted
                # (caught by verify() once): check each tensor, resend if wrong.
                for attempt in range(4):
                    g = t.to(device)
                    if torch.equal(g.cpu(), t):
                        break
                    sys.stderr.write(f"weight {k} arrived corrupted; resending\n")
                else:
                    raise RuntimeError(f"weight {k} keeps arriving corrupted")
                # dtype=: the meta model was built in float32; without it every
                # weight is widened back to 4 bytes (8 GB -> 16 GB).
                set_module_tensor_to_device(model, k, device, dtype=torch.bfloat16, value=g)
                del t, g
    return model


def verify(model: torch.nn.Module, folder: Path, n: int = 24) -> None:
    """Spot-check that weights arrived intact (a bad DMA path corrupts silently)."""
    from safetensors import safe_open

    params = dict(model.named_parameters())
    for shard in sorted(folder.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for k in keys[:: max(1, len(keys) // n)]:
                if k in params and not torch.equal(params[k].detach().cpu(),
                                                   f.get_tensor(k).to(torch.bfloat16)):
                    raise RuntimeError(f"weight {k} differs on the GPU: refusing to draw")


def load_encoder(device: str):
    """The text half (Qwen3 4B, 8 GB) as a pipeline that only encodes prompts."""
    from accelerate import init_empty_weights
    from diffusers import Flux2KleinPipeline
    from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL / "text_encoder")
    with init_empty_weights():
        te = Qwen3ForCausalLM(cfg)
    te = load_direct(te, MODEL / "text_encoder", device)
    te.tie_weights()
    te.to(device).eval()
    verify(te, MODEL / "text_encoder")      # buffers (RoPE inv_freq) stay float32, as stock
    tok = AutoTokenizer.from_pretrained(MODEL / "tokenizer")
    return Flux2KleinPipeline(scheduler=None, vae=None, text_encoder=te, tokenizer=tok,
                              transformer=None, is_distilled=True)


def encode(enc, prompt: str, device: str) -> torch.Tensor:
    with torch.inference_mode():
        emb, _ = enc.encode_prompt(prompt=prompt, device=torch.device(device))
    return emb.to("cpu")


def encode_all(prompts: list[str], device: str) -> list[torch.Tensor]:
    enc = load_encoder(device)
    out = [encode(enc, p, device) for p in prompts]
    del enc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def load_drawer(device: str):
    """The picture half: transformer (7.8 GB) + VAE + scheduler."""
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
    from diffusers import Flux2Transformer2DModel

    with init_empty_weights():
        tr = Flux2Transformer2DModel.from_config(
            Flux2Transformer2DModel.load_config(MODEL / "transformer"))
    tr = load_direct(tr, MODEL / "transformer", device).to(device).eval()
    verify(tr, MODEL / "transformer")
    vae = AutoencoderKLFlux2.from_pretrained(MODEL / "vae", torch_dtype=torch.bfloat16).to(device)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL / "scheduler")
    return Flux2KleinPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None,
                              transformer=tr, is_distilled=True)


def draw_one(pipe, j: dict, emb: torch.Tensor, a, device: str) -> list[float]:
    """Draw job *j* and save it (atomically); returns the step times."""
    g = torch.Generator(device).manual_seed(int(j.get("seed", 0)))
    w, h = j.get("size") or (a.width, a.height)
    refs = [ref_image(r) for r in j.get("refs", []) if Path(r).exists()]
    steps_t, last = [], [time.time()]

    def tick(_p, i, _t, kw):
        torch.cuda.synchronize()
        now = time.time()
        steps_t.append(round(now - last[0], 1))
        last[0] = now
        return kw
    with torch.inference_mode():
        img = pipe(image=refs or None, callback_on_step_end=tick, prompt_embeds=emb.to(device), width=w, height=h,
                   num_inference_steps=a.steps, guidance_scale=1.0, generator=g).images[0]
    dest = Path(j["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}")
    img.save(tmp, format="PNG")
    tmp.replace(dest)
    j["_refs"] = len(refs)
    return steps_t


def serve(a, device: str) -> int:
    """--serve: both halves stay loaded (~17 GB: the music server must be
    stopped meanwhile) and jobs arrive one JSON object per line on stdin, so a
    keyframe phase with its check-and-redraw rounds loads the model once
    instead of once a round (~45 s each, 10-18). One JSON line out per job:
    {"id", "seconds"} or {"id", "error"}."""
    t = time.time()
    enc = load_encoder(device)
    pipe = load_drawer(device)
    print(json.dumps({"phase": "load drawing model (kept)", "seconds": round(time.time() - t, 1)}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        j = json.loads(line)
        t = time.time()
        try:
            if not Path(j["path"]).exists():
                draw_one(pipe, j, encode(enc, j["prompt"], device), a, device)
            print(json.dumps({"id": j["id"], "refs": j.get("_refs", 0), "seconds": round(time.time() - t, 1)}),
                  flush=True)
        except Exception as exc:              # a bad job (an unreadable reference): report, keep serving
            print(json.dumps({"id": j["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
            torch.cuda.empty_cache()
    return 0


REF_AREA = 512 * 640     # each reference adds (w/16)*(h/16) tokens; ~1300 at this size


def ref_image(path: str):
    """A reference picture, shrunk so it conditions without doubling the cost."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    scale = min(1.0, (REF_AREA / (im.width * im.height)) ** 0.5)
    w, h = (max(16, int(im.width * scale) // 16 * 16), max(16, int(im.height * scale) // 16 * 16))
    return im.resize((w, h), Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--serve", action="store_true", help="keep the model loaded; jobs as JSON lines on stdin")
    a = ap.parse_args()
    # As in ace_serve.py: MIOpen's conv search on ROCm costs far more than it
    # saves for a few VAE decodes.
    torch.backends.cudnn.enabled = os.environ.get("ILLUSTRATE_MIOPEN", "0") == "1"
    if a.serve:
        return serve(a, "cuda")
    if not (a.jobs and a.out):
        ap.error("jobs and out are required without --serve")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for j in (jobs := json.loads(Path(a.jobs).read_text())):
        j["path"] = j.get("path") or str(out / f"{j['id']}.png")
    jobs = [j for j in jobs if not Path(j["path"]).exists()]
    if not jobs:
        return 0
    device = "cuda"
    t = time.time()
    embeds = encode_all([j["prompt"] for j in jobs], device)
    print(json.dumps({"phase": f"read {len(jobs)} prompts", "seconds": round(time.time() - t, 1)}), flush=True)

    t = time.time()
    pipe = load_drawer(device)
    print(json.dumps({"phase": "load drawing model", "seconds": round(time.time() - t, 1)}), flush=True)
    from PIL import Image

    def rss_gb() -> float:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return round(int(line.split()[1]) / 1048576, 2)
        return 0.0

    def io() -> tuple[int, int]:
        rb = next(int(l.split()[1]) for l in open("/proc/self/io") if l.startswith("read_bytes"))
        return rb, int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[9])

    io0 = io()
    for n, (j, emb) in enumerate(zip(jobs, embeds)):
        t = time.time()
        steps_t = draw_one(pipe, j, emb, a, device)
        print(json.dumps({"id": j["id"], "refs": j["_refs"], "seconds": round(time.time() - t, 1),
                          "rss_gb": rss_gb()}), flush=True)
        if n == 0:
            sys.stderr.write(f"first picture {time.time() - t:.1f} s (steps {steps_t}, "
                             f"rest = refs+decode), rss {rss_gb()} GB, disk read "
                             f"{(io()[0] - io0[0]) / 2**20:.0f} MiB, major faults {io()[1] - io0[1]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
