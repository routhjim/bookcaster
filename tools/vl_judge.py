"""Ask a vision-language model short questions about pictures (generic picture checks; any book).

Uses the Qwen3-VL-8B that ships as Qwen-Image-2.1's text encoder (full model incl. lm_head), so no
extra download. Batch mode: JSON lines on stdin -> JSON lines on stdout.

    HIP_VISIBLE_DEVICES=$(~/bin/hip-index igpu) HSA_CU_MASK=<igpu>:0-38 \
    ~/tts-ab/.venv-charlora-igpu/bin/python -I tools/vl_judge.py < q.jsonl > a.jsonl
    {"id", "image": path, "crop": [x0,y0,x1,y1] (0-1, optional), "ask": {"age": "How old ...? number"}}
    -> {"id", "answers": {"age": "72"}}

Python: Judge().ask(pil_image, question, max_new_tokens=8) -> str.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODEL = Path(os.environ.get("QWEN_IMAGE_MODEL", Path.home() / "models" / "qwen-image-2.1")).expanduser()


class Judge:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        torch.backends.cudnn.enabled = False          # no MIOpen tuning (iGPU with a desktop up)
        self.proc = AutoProcessor.from_pretrained(MODEL / "processor")
        self.m = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL / "text_encoder", torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.device = device
        # the vision patch embedding is a Conv3d with kernel == stride: one matmul (as a conv it is MIOpen
        # tuning or, with MIOpen off, the slow direct conv); as in tools/qwen_image.py
        pe = self.m.model.visual.patch_embed
        w2 = pe.proj.weight.reshape(pe.embed_dim, -1)
        import torch.nn.functional as F
        pe.forward = lambda x: F.linear(x.reshape(-1, w2.shape[1]).to(w2.dtype), w2, pe.proj.bias)

    def ask(self, im, question: str, max_new_tokens: int = 8, max_side: int = 640, prefix: str = "") -> str:
        sc = max_side / max(im.size)          # vision tokens grow with the area: 1536 px cost ~10 s a question
        if sc < 1:
            im = im.resize((max(28, round(im.width * sc)), max(28, round(im.height * sc))))
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": question}]}]
        if prefix:                            # the answer's start, written for the model (fewer tokens to decode)
            text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + prefix
            x = self.proc(text=[text], images=[im], return_tensors="pt").to(self.device)
        else:
            x = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                              return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.m.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return prefix + self.proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def box(self, im, what: str):
        """[x0, y0, x1, y1] in 0-1 of *what*, or None (Qwen3-VL grounding, 0-1000 coordinates)."""
        import re
        a = self.ask(im, f"Locate {what} in the image and output the bbox coordinates in JSON format.", 26,
                     prefix='```json\n[\n\t{"bbox_2d": [')
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", a)
        if not m:
            return None
        b = [int(v) / 1000 for v in m.groups()]
        return b if b[2] > b[0] and b[3] > b[1] else None


def crop(path: str, box=None, max_side: int = 768):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if box:
        W, H = im.size
        im = im.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)))
    return im


def main() -> int:
    j = Judge()
    for line in sys.stdin:
        if not line.strip():
            continue
        q = json.loads(line)
        im = crop(q["image"], q.get("crop"))
        ans = {k: j.ask(im, v, int(q.get("max_new_tokens", 8))) for k, v in q["ask"].items()}
        print(json.dumps({"id": q.get("id"), "answers": ans}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
