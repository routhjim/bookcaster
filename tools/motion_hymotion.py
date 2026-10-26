"""HY-Motion 1.0 backend runner (runs in ~/tts-ab/.venv-hymotion; tools/motion_source.py drives it).

    python tools/motion_hymotion.py encode   JOBS.json   # text encoders (Qwen3-8B + CLIP-L), no GPU lock
    python tools/motion_hymotion.py generate JOBS.json   # the DiT, on one GPU (caller holds the flock)

JOBS {"model": "<dir with config.yml + latest.ckpt>", "te_device": "cpu"|"cuda", "steps": 50, "cfg": 5.0,
      "jobs": [{"text": str, "dur": seconds, "seed": int, "fps": 24, "out": "/abs/base"}]}
encode caches each text's embeddings in $HYMOTION_HOME/emb_cache/<sha1>.pt; generate needs them and writes
each job as a CanonicalMotion (out.npz + out.json, tools/motion_canonical.py), resampled from the model's
native 30 fps to the job's fps, turned to face +X at frame 0. Timings and peak VRAM go to the header.

Licence: Tencent HY-Motion 1.0 Community License (not valid in the EU, UK, South Korea); see
eval/motion/STATE.md.
"""

import gc
import hashlib
import json
import os
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
HOME = os.environ.get("HYMOTION_HOME", os.path.expanduser("~/models/hy-motion"))
REPO = os.environ.get("HYMOTION_REPO", os.path.expanduser("~/src/HY-Motion-1.0"))
QWEN = os.path.join(HOME, "Qwen3-8B")
CLIP = os.path.join(HOME, "clip-vit-large-patch14")
CACHE = os.path.join(HOME, "emb_cache")
MAX_LLM = 128          # config text_encoder_cfg.max_length_llm

sys.path.insert(0, TOOLS)
sys.path.insert(0, REPO)
if len(sys.argv) > 2:
    sys.argv[2] = os.path.abspath(sys.argv[2])
os.chdir(REPO)        # the pipeline reads ./stats and the wooden body model by relative path

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cudnn.enabled = False     # MIOpen off (broke other models on this box; nothing here needs it)


TE_MODE = os.environ.get("HYMOTION_TE", "int8")   # int8 (default) | bf16 (reference): Qwen3-8B weights


def key(text, mode=None):
    mode = TE_MODE if mode is None else mode
    tag = "" if mode == "ref" else f"|{mode}|nopad"     # "ref" = the upstream bf16, padded-to-128 path
    return hashlib.sha1(f"qwen3-8b|clipl|{MAX_LLM}{tag}|{text}".encode()).hexdigest()


def build_te(dev, mode):
    """HY-Motion's text model (Qwen3-8B LLM + CLIP-L). mode: ref | bf16 | int8. Non-ref modes drop the
    LM head (only hidden states are used); int8 stores every decoder Linear and the token table as int8
    rows (tools/qwen_image.py QLinear/QEmbedding, one scale per output row)."""
    from hymotion.network.text_encoders import text_encoder as te
    te.LLM_ENCODER_LAYOUT["qwen3"]["module_path"] = QWEN
    te.SENTENCE_EMB_LAYOUT["clipl"]["module_path"] = CLIP
    model = te.HYTextModel(llm_type="qwen3", max_length_llm=MAX_LLM)
    if mode != "ref":
        from qwen_image import NoHead, QEmbedding, QLinear
        lm = model.llm_text_encoder
        lm.lm_head = NoHead()
        if mode == "int8":
            for name, mod in list(lm.model.named_modules()):
                if isinstance(mod, torch.nn.Linear):
                    parent, attr = name.rsplit(".", 1)
                    q = QLinear(mod, "int8")
                    q.set_weight(mod.weight.data)
                    setattr(lm.model.get_submodule(parent), attr, q)
            qe = QEmbedding(lm.model.embed_tokens)
            qe.set_weight(lm.model.embed_tokens.weight.data)
            lm.model.embed_tokens = qe
            gc.collect()
    return model.to(dev).eval()


@torch.no_grad()
def encode_texts(model, texts, mode):
    """-> list of {text_vec_raw (1,1,768), text_ctxt_raw (1,128,4096), text_ctxt_raw_length (1,)} (float32, cpu).
    ref: upstream encode, one prompt at a time, padded to crop_start+128. Others: the whole batch in one
    forward, right-padded to the longest prompt only (causal attention: real tokens are unchanged), and the
    128-token window zero-filled past each prompt's length (masked out by the DiT)."""
    if mode == "ref":
        out = []
        for t in texts:
            v, c, n = model.encode([t])
            out.append({"text_vec_raw": v.float().cpu(), "text_ctxt_raw": c.float().cpu(), "text_ctxt_raw_length": n.cpu()})
        return out
    from hymotion.network.text_encoders.text_encoder import LLM_ENCODER_LAYOUT
    tok, lm = model.llm_tokenizer, model.llm_text_encoder
    dev = next(lm.parameters()).device if any(True for _ in lm.parameters()) else torch.device("cpu")
    chat = [tok.apply_chat_template(model.apply_text_to_template(t, LLM_ENCODER_LAYOUT["qwen3"]["template"]),
                                    tokenize=False, add_generation_prompt=False, enable_thinking=False) for t in texts]
    enc = tok(chat, truncation=True, max_length=model.max_length_llm, padding=True, return_tensors="pt")
    hs = lm(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
            output_hidden_states=True).hidden_states[-1].float().cpu()
    s = model.crop_start
    vtxt = model.encode_sentence_emb(texts).float().cpu()
    out = []
    for i in range(len(texts)):
        n = int(enc["attention_mask"][i].sum()) - s
        n = max(0, min(n, MAX_LLM))
        c = torch.zeros(1, MAX_LLM, hs.shape[-1])
        c[0, :n] = hs[i, s:s + n]
        out.append({"text_vec_raw": vtxt[i:i + 1], "text_ctxt_raw": c, "text_ctxt_raw_length": torch.tensor([n])})
    return out


def encode(J):
    os.makedirs(CACHE, exist_ok=True)
    mode = J.get("te_mode", TE_MODE)
    todo = sorted({j["text"] for j in J["jobs"]
                   if not j.get("null_text") and not os.path.exists(os.path.join(CACHE, key(j["text"], mode) + ".pt"))})
    if not todo:
        print("ENCODE all cached")
        return
    dev = torch.device(J.get("te_device", "cpu"))
    t0 = time.time()
    model = build_te(dev, mode)
    t1 = time.time()
    print(f"ENCODE load {t1 - t0:.1f}s on {dev} ({mode})")
    res = encode_texts(model, todo, mode)
    t2 = time.time()
    for text, r in zip(todo, res):
        torch.save(r, os.path.join(CACHE, key(text, mode) + ".pt"))
    print(f"ENCODE {len(todo)} prompts {t2 - t1:.1f}s ({(t2 - t1) / len(todo):.1f}s each)")
    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def generate(J):
    import yaml
    from hymotion.utils.loaders import load_object
    from hymotion.utils.geometry import rot6d_to_rotation_matrix
    from motion_canonical import C_SMPL, from_smpl

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(os.path.join(J["model"], "config.yml")))
    t0 = time.time()
    pipe = load_object(cfg["train_pipeline"], cfg["train_pipeline_args"], network_module=cfg["network_module"],
                       network_module_args=cfg["network_module_args"])
    pipe.load_in_demo(os.path.join(J["model"], "latest.ckpt"), build_text_encoder=False)
    pipe.to(dev).eval()
    pipe.validation_steps = int(J.get("steps", 50))
    load_s = time.time() - t0
    print(f"GEN load {load_s:.1f}s on {dev}")
    body = pipe.body_model
    rest_smpl = body.j_template[:22].cpu().numpy()
    ground_y = float(body.v_template[:, 1].min())
    for j in J["jobs"]:
        if j.get("null_text"):          # smoke test: unconditional features, no text encoder needed
            hs = {"text_vec_raw": pipe.null_vtxt_feat.detach().cpu(), "text_ctxt_raw": pipe.null_ctxt_input.detach().cpu(),
                  "text_ctxt_raw_length": torch.tensor([1])}
        else:
            cands = [os.path.join(CACHE, key(j["text"], m) + ".pt") for m in (J.get("te_mode", TE_MODE), "int8", "bf16", "ref")]
            hs = torch.load(next(c for c in cands if os.path.exists(c)))   # any cached encoding of this text
        hs = {k: v.to(dev) for k, v in hs.items()}
        hs["text_vec_raw"] = hs["text_vec_raw"].to(next(pipe.parameters()).dtype)
        hs["text_ctxt_raw"] = hs["text_ctxt_raw"].to(next(pipe.parameters()).dtype)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        out = pipe.generate(j["text"], [int(j.get("seed", 0))], float(j["dur"]), cfg_scale=float(J.get("cfg", 5.0)),
                            hidden_state_dict=hs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        gen_s = time.time() - t
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        R = rot6d_to_rotation_matrix(out["rot6d"][0][:, :22]).cpu().numpy()          # (L, 22, 3, 3) local
        # out["keypoints3d"] is root-relative (LBS joints without the translation); rebuild world joints
        # and ground on the posed mesh's lowest vertex over the whole clip
        with torch.no_grad():
            tr = out["transl"][0].to(dev)
            res = body({"rot6d": out["rot6d"][0].to(dev), "trans": tr})
            vmin = float(res["vertices"][..., 1].min())
            K = (res["keypoints3d"][:, :22] + tr[:, None]).cpu().numpy()
        K[..., 1] -= vmin
        meta = {"backend": "hymotion", "model": os.path.basename(os.path.normpath(J["model"])),
                "text": j["text"], "seed": int(j.get("seed", 0)), "cfg": float(J.get("cfg", 5.0)),
                "steps": pipe.validation_steps, "native_fps": float(pipe.output_mesh_fps),
                "gen_s": round(gen_s, 2), "load_s": round(load_s, 2), "peak_vram_gib": round(peak, 2),
                "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
        cm = from_smpl(R, K, rest_smpl, float(pipe.output_mesh_fps), ground_y, meta)
        # sanity: our FK must reproduce the model's own joints (mm-level)
        err = float(np.abs(cm.pos - K @ C_SMPL.T).max())
        cm.meta["fk_check_m"] = round(err, 4)
        cm = cm.normalise().resample(float(j.get("fps", 24)))
        cm.save(j["out"])
        print(f"GEN {gen_s:.2f}s peak {peak:.2f} GiB fk_err {err * 1000:.1f} mm frames {cm.frames}  -> {j['out']}")


if __name__ == "__main__":
    op, path = sys.argv[1], sys.argv[2]
    J = json.load(open(path))
    {"encode": encode, "generate": generate}[op](J)
