"""Edit pictures with Qwen-Image-Edit-2509 (multi-image). Copied from the E35 keys branch for the asset chain's
texture-painting stage (tools/assets/paint.py, method "edit"); the loader is tools/qie_fast.py.

    ~/tts-ab/.venv-hyworld/bin/python -I tools/assets/qwen_edit.py jobs.json       (on one card, under its flock)
    ~/tts-ab/.venv-hyworld/bin/python -I tools/assets/qwen_edit.py jobs.json --dry (CPU: inputs, sizes, tokens)

jobs: [{"id", "prompt", "images": ["/abs/frame.png", "/abs/sheet.png", ...], "seed", "path", "size": [w, h]}, ...]
  images[0] is the picture edited; the rest are references (the model's "Picture 2", "Picture 3", ...; 2509 is
  trained on 1-3 input images). One JSON line out per job: {"id", "seconds", ...} or {"id", "error"}; a job whose
  path exists is skipped (a rerun resumes).

As tools/world.py's panorama stage (the same weights, the same loader): the Qwen2.5-VL encoder (bf16, 16.6 GB) and
the 20B transformer (int8 per row from the Q6_K GGUF, ~19 GiB: tools/qie_fast.py) do not fit one 24 GB card
together, so every job's prompt is encoded first (with its pictures), the encoder freed, then the transformer
loaded once and every job drawn. Lightning (8 steps, CFG off) at 0.7: world.py's 10-21 A/B (1.0 painted a crackle).
The character LoRAs are Qwen-Image 2.1's (32 blocks, 4096 wide) and cannot load on this 60-block 3072-wide model:
identity comes from the reference pictures only.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

HOME = Path.home()
TOOLS = Path(__file__).resolve().parent.parent          # tools/ (qie_fast.py)
MODELS = Path(os.environ.get("WORLD_MODELS", HOME / "models" / "hy-world"))
BASE = Path(os.environ.get("QIE_EDIT_BASE", MODELS / "Qwen-Image-Edit-2509"))
GGUF = Path(os.environ.get("QIE_EDIT_GGUF", MODELS / "qie2509-gguf" / "Qwen-Image-Edit-2509-Q6_K.gguf"))
LIGHTNING = Path(os.environ.get("QIE_EDIT_LIGHTNING", MODELS / "lightning" / "Qwen-Image-Edit-2509"
                                / "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"))
LIGHT_W = float(os.environ.get("QIE_EDIT_LIGHTNING_W", "0.7"))     # 0 = the 40-step true-CFG path
STEPS = int(os.environ.get("QIE_EDIT_STEPS", "8" if LIGHT_W > 0 else "40"))
CFG = float(os.environ.get("QIE_EDIT_CFG", "1.0" if LIGHT_W > 0 else "4.0"))
NEG = " "
# every input picture goes to the VAE at this many pixels (the pipeline's default is 1024x1024 for each): the key's
# own size, so the frame is not upscaled and three inputs cost ~3x the picture's tokens, not ~4.5x
VAE_AREA = int(os.environ.get("QIE_EDIT_VAE_AREA", "0"))
# the scheduler the Lightning LoRA was distilled with (lightx2v's Qwen-Image-Lightning README)
LIGHT_SCHED = {"base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
               "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0,
               "shift_terminal": None, "stochastic_sampling": False, "time_shift_type": "exponential",
               "use_beta_sigmas": False, "use_dynamic_shifting": True, "use_exponential_sigmas": False,
               "use_karras_sigmas": False}
IMG_TAG = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


def log(msg: str) -> None:
    sys.stderr.write(f"[qwen_edit] {msg}\n")
    sys.stderr.flush()


def sizes(img_wh: tuple, vae_area: int) -> dict:
    """The condition (vision encoder) and VAE sizes the pipeline gives one input picture."""
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as M
    w, h = img_wh
    cw, ch = M.calculate_dimensions(M.CONDITION_IMAGE_SIZE, w / h)
    vw, vh = M.calculate_dimensions(vae_area or M.VAE_IMAGE_SIZE, w / h)
    return {"cond": [cw, ch], "vae": [vw, vh], "vae_tokens": (vw // 16) * (vh // 16)}


def check_job(j: dict) -> list:
    """What is wrong with a job before anything loads."""
    bad = [k for k in ("id", "prompt", "images", "seed", "path", "size") if k not in j]
    if bad:
        return [f"missing {bad}"]
    out = []
    if not 1 <= len(j["images"]) <= int(os.environ.get("QIE_EDIT_MAX_IMAGES", "3")):
        out.append(f"{len(j['images'])} input pictures (1-3)")
    out += [f"no picture {p}" for p in j["images"] if not Path(p).exists()]
    w, h = j["size"]
    if w % 16 or h % 16:
        out.append(f"size {w}x{h} is not a multiple of 16")
    return out


def dry(jobs: list) -> int:
    """CPU only: each job's inputs, the sizes the pipeline gives them, the encoder's text and its token count."""
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(BASE / "processor"))
    rc = 0
    for j in jobs:
        errs = check_job(j)
        if errs:
            print(json.dumps({"id": j.get("id"), "error": "; ".join(errs)}), flush=True)
            rc = 1
            continue
        w, h = j["size"]
        ims = [Image.open(p).convert("RGB") for p in j["images"]]
        sz = [sizes(im.size, VAE_AREA or w * h) for im in ims]
        conds = [im.resize(tuple(s["cond"])) for im, s in zip(ims, sz)]
        txt = "".join(IMG_TAG.format(i + 1) for i in range(len(ims))) + j["prompt"]
        enc = proc(text=[txt], images=conds, padding=True, return_tensors="pt")
        print(json.dumps({"id": j["id"], "inputs": [list(im.size) for im in ims], "sizes": sz,
                          "encoder_tokens": int(enc.input_ids.shape[1]),
                          "dit_tokens": (w // 16) * (h // 16) + sum(s["vae_tokens"] for s in sz),
                          "prompt_words": len(j["prompt"].split())}), flush=True)
    return rc


def draw(jobs: list) -> int:
    import torch
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as M
    sys.path.insert(0, str(TOOLS))
    import qie_fast
    jobs = [j for j in jobs if not Path(j["path"]).exists()]
    for j in list(jobs):
        errs = check_job(j)
        if errs:
            print(json.dumps({"id": j.get("id"), "error": "; ".join(errs)}), flush=True)
            jobs.remove(j)
    if not jobs:
        return 0
    dev = "cuda"
    torch.backends.cuda.preferred_blas_library("cublaslt")
    fa = qie_fast.install_fa_amd()
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(BASE), transformer=None, torch_dtype=torch.bfloat16)
    if LIGHT_W > 0:
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHT_SCHED)
    pipe.text_encoder.to(dev)
    pipe.vae.to(dev)
    pipe.vae.enable_tiling()
    enc = {}
    with torch.no_grad():
        for j in jobs:
            try:
                ims = [Image.open(p).convert("RGB") for p in j["images"]]
                conds = [pipe.image_processor.resize(im, *reversed(sizes(im.size, 0)["cond"])) for im in ims]
                pe, pm = pipe.encode_prompt(image=conds, prompt=j["prompt"], device=dev)
                ne = nm = None
                if CFG > 1:
                    ne, nm = pipe.encode_prompt(image=conds, prompt=NEG, device=dev)
                enc[j["id"]] = [x.cpu() if x is not None else None for x in (pe, pm, ne, nm)]
            except Exception as exc:          # noqa: BLE001 - a bad job is reported, the rest drawn
                print(json.dumps({"id": j["id"], "error": f"encode: {type(exc).__name__}: {exc}"}), flush=True)
    t_enc = time.time() - t0
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    loras = [(LIGHTNING, LIGHT_W)] if LIGHT_W > 0 else []
    tr, st = qie_fast.load_int8_transformer(GGUF, BASE, loras=loras, device=dev)
    pipe.register_modules(transformer=tr)
    print(json.dumps({"phase": "loaded", "encode_s": round(t_enc, 1), "fa_amd": fa, **st}), flush=True)
    for j in jobs:
        if j["id"] not in enc:
            continue
        t1 = time.time()
        try:
            w, h = j["size"]
            M.VAE_IMAGE_SIZE = VAE_AREA or w * h
            pe, pm, ne, nm = (x.to(dev) if x is not None else None for x in enc[j["id"]])
            ims = [Image.open(p).convert("RGB") for p in j["images"]]
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                im = pipe(image=ims, prompt_embeds=pe, prompt_embeds_mask=pm, negative_prompt_embeds=ne,
                          negative_prompt_embeds_mask=nm, true_cfg_scale=CFG, num_inference_steps=STEPS,
                          guidance_scale=1.0, height=h, width=w,
                          generator=torch.Generator(device="cpu").manual_seed(int(j["seed"]))).images[0]
            Path(j["path"]).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(j["path"]).with_suffix(".part.png")
            im.save(tmp)
            os.replace(tmp, j["path"])
            print(json.dumps({"id": j["id"], "seconds": round(time.time() - t1, 1), "steps": STEPS, "cfg": CFG,
                              "lightning": LIGHT_W, "inputs": len(ims),
                              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
        except Exception as exc:              # noqa: BLE001
            print(json.dumps({"id": j["id"], "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            if "out of memory" in str(exc).lower():
                gc.collect()
                torch.cuda.empty_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--dry", action="store_true", help="CPU only: check the jobs, print sizes and token counts")
    a = ap.parse_args()
    jobs = json.loads(Path(a.jobs).read_text())
    return dry(jobs) if a.dry else draw(jobs)


if __name__ == "__main__":
    sys.exit(main())
