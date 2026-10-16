"""Batched decode for Fish Audio S2 Pro: several lines share each weight read.

Fish decodes one audio frame per step: the 36-layer slow transformer once,
then the 4-layer fast transformer once per codebook (10x). At batch 1 that is
~16 GB of weights read per frame, so the 7900 XTX is bandwidth-bound at ~41
frames/s (~1.9x real time). Running B lines together reads the same weights
once for all B rows, so throughput scales almost with B.

Each row is its own sequence at its own position (lines have different
prompt lengths), so the slow transformer's attention is written here for
per-row positions; the fast transformer's positions are shared (codebook
index) and run batched as-is.

Pieces:
  setup(model, rows, seq_len)  - per-row KV caches (+ sliced head, int8 opt)
  prefill(model, row, prompt)  - one line's prompt into its cache row
  step(model, tokens, pos, ...) - one frame for every row at once
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- per-row slow-transformer attention -------------------------------------

def _rope(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """x: (B, 1, H, D); f: (B, D/2, 2) — each row rotated at its own position."""
    xs = x.float().reshape(*x.shape[:-1], -1, 2)
    f = f.view(x.shape[0], 1, 1, xs.shape[-2], 2).float()
    out = torch.stack([xs[..., 0] * f[..., 0] - xs[..., 1] * f[..., 1],
                       xs[..., 1] * f[..., 0] + xs[..., 0] * f[..., 1]], -1)
    return out.flatten(3).type_as(x)


def _attend_rows(att, x, f, pos, kv_len: int, kc, vc):
    """Decode attention for B rows at positions *pos* (B,) over the first
    *kv_len* cache slots of each row (grouped heads, no repeat_interleave)."""
    B = x.shape[0]
    H, Hkv, D = att.n_head, att.n_local_heads, att.head_dim
    q, k, v = att.wqkv(x).split([H * D, Hkv * D, Hkv * D], dim=-1)
    q, k, v = q.view(B, 1, H, D), k.view(B, 1, Hkv, D), v.view(B, 1, Hkv, D)
    if att.attention_qk_norm:
        q, k = att.q_norm(q), att.k_norm(k)
    q, k = _rope(q, f), _rope(k, f)
    rows = torch.arange(B, device=x.device)
    kc[rows, :, pos] = k[:, 0]
    vc[rows, :, pos] = v[:, 0]
    K, V = kc[:B, :, :kv_len], vc[:B, :, :kv_len]               # (B, Hkv, n, D)
    qg = q.view(B, Hkv, H // Hkv, D)                            # (B, Hkv, g, D)
    s = (qg @ K.transpose(-1, -2)).float() / math.sqrt(D)       # (B, Hkv, g, n)
    live = torch.arange(kv_len, device=x.device)[None] <= pos[:, None]
    s = s.masked_fill(~live[:, None, None], float("-inf"))
    y = s.softmax(-1).to(V.dtype) @ V                           # (B, Hkv, g, D)
    return att.wo(y.reshape(B, 1, H * D))


def slow_forward_rows(model, x: torch.Tensor, pos: torch.Tensor, kv_len: int):
    """Slow transformer for one decode token per row; returns (logits over
    the sliced head rows, hidden state for the fast transformer)."""
    f = model.freqs_cis[pos]
    for layer, (kc, vc) in zip(model.layers, model._row_caches):
        h = x + _attend_rows(layer.attention, layer.attention_norm(x), f, pos, kv_len, kc, vc)
        x = h + layer.feed_forward(layer.ffn_norm(h))
    out = model.norm(x)
    logits = F.linear(out, model._head_w)                       # (B, 1, n_sem + 1)
    hidden = out if getattr(model.config, "norm_fastlayer_input", False) else x
    return logits, model.fast_project_in(hidden)


def embed_rows(model, tok: torch.Tensor) -> torch.Tensor:
    """tok: (B, 11) = [main token, 10 codebook codes] -> (B, 1, dim)."""
    cfg = model.config
    offs = torch.arange(cfg.num_codebooks, device=tok.device) * cfg.codebook_size
    vq = model.codebook_embeddings(tok[:, 1:] + offs).sum(1)
    sem = (tok[:, 0] >= cfg.semantic_begin_id) & (tok[:, 0] <= cfg.semantic_end_id)
    vq = vq * sem[:, None].to(vq.dtype)
    x = model.embeddings(tok[:, 0]) + vq
    if cfg.scale_codebook_embeddings:
        x = torch.where(sem[:, None], x / math.sqrt(cfg.num_codebooks + 1), x)
    return x[:, None]


# --- sampling (per row, over small vocabularies) ----------------------------

def sample_rows(logits: torch.Tensor, temp: torch.Tensor, top_p: torch.Tensor,
                top_k: int) -> torch.Tensor:
    """logits (B, V) -> (B,) ids; top-k then nucleus, Gumbel-free exponential
    race like fish's multinomial_sample_one_no_sync (no host sync)."""
    k = min(top_k, logits.shape[-1])
    vals, idx = logits.float().topk(k, dim=-1)                  # sorted desc
    cum = vals.softmax(-1).cumsum(-1)
    drop = cum > top_p[:, None] if top_p.dim() else cum > top_p
    drop[:, 0] = False
    vals = vals.masked_fill(drop, float("-inf")) / temp.clamp(min=1e-5)
    probs = vals.softmax(-1)
    q = -torch.log(torch.rand_like(probs))
    pick = torch.argmax(probs / q, dim=-1, keepdim=True)
    return idx.gather(-1, pick)[:, 0]


RAS_WIN = 10


def step(model, tok, pos, kv_len: int, prev, temp, top_p, top_k: int = 30):
    """One frame for every row. tok (B, 11) last frame, pos (B,) positions to
    write, prev (B, RAS_WIN) recent main tokens. Returns the new frame (B, 11)."""
    cfg = model.config
    x = embed_rows(model, tok)
    logits, hidden = slow_forward_rows(model, x, pos, kv_len)
    logits = logits[:, 0]
    main = sample_rows(logits, temp, top_p, top_k)
    # Repetition-aware sampling: a semantic token repeated within the window
    # is re-drawn at temperature 1.0 / top-p 0.9 (fish's RAS).
    hi = sample_rows(logits, torch.ones_like(temp), torch.full_like(top_p, 0.9), top_k)
    rep = (prev == main[:, None]).any(-1) & (main < model._n_sem)
    main = torch.where(rep, hi, main)
    main_id = model._head_ids[main]                             # back to vocab ids
    codes = [main_id]
    zero = torch.zeros(1, dtype=torch.long, device=tok.device)
    model.forward_generate_fast(hidden, zero)
    a = (main_id - cfg.semantic_begin_id).clamp(0, cfg.codebook_size - 1)
    h = model.fast_embeddings(a)
    codes.append(a)
    for i in range(1, cfg.num_codebooks):
        lg = model.forward_generate_fast(h, zero + i)[:, 0]
        a = sample_rows(lg, temp, top_p, top_k)
        h = model.fast_embeddings(a)
        codes.append(a)
    return torch.stack(codes, 1), main


# --- setup / prefill --------------------------------------------------------

def setup(model, rows: int, seq_len: int) -> None:
    """Per-row KV caches for the slow transformer, a fast-transformer cache
    with *rows* rows, and the sliced output head (semantic tokens + im_end)."""
    cfg = model.config
    dev = model.embeddings.weight.device
    dtype = model.embeddings.weight.dtype
    model._row_caches = [
        (torch.zeros(rows, cfg.n_local_heads, seq_len, cfg.head_dim, device=dev, dtype=dtype),
         torch.zeros(rows, cfg.n_local_heads, seq_len, cfg.head_dim, device=dev, dtype=dtype))
        for _ in model.layers]
    with torch.device(dev):
        model.setup_caches(max_batch_size=rows, max_seq_len=seq_len, dtype=dtype)
    # The stock setup made full per-layer caches for the slow layers too;
    # drop them (prefill uses its own path below).
    for layer in model.layers:
        layer.attention.kv_cache = None
    torch.cuda.empty_cache()
    if not hasattr(model, "_head_w"):
        im_end = model.tokenizer.get_token_id("<|im_end|>")
        ids = torch.cat([torch.arange(cfg.semantic_begin_id, cfg.semantic_end_id + 1),
                         torch.tensor([im_end])]).to(dev)
        w = model.embeddings.weight if cfg.tie_word_embeddings else model.output.weight
        model._head_ids = ids
        model._head_w = w[ids].contiguous()
        model._n_sem = cfg.semantic_end_id - cfg.semantic_begin_id + 1
        model._im_end_row = len(ids) - 1
    model._rows, model._seq_len = rows, seq_len


@torch.inference_mode()
def prefill(model, row: int, prompt: torch.Tensor, audio_masks=None, audio_parts=None):
    """Run a line's prompt (11, T) through the slow transformer into cache
    row *row*; returns (logits (n_sem+1,), hidden) for its first frame."""
    T = prompt.shape[1]
    dev = prompt.device
    x_in = prompt[None]
    # Reuse the model's own embedding (handles audio parts) via forward on a
    # temporary single-row cache that we then copy into our row.
    from fish_speech.models.text2semantic.llama import KVCache

    cfg = model.config
    tmp = []
    for layer in model.layers:
        layer.attention.kv_cache = KVCache(1, T, cfg.n_local_heads, cfg.head_dim,
                                           dtype=model._row_caches[0][0].dtype).to(dev)
        tmp.append(layer.attention.kv_cache)
    saved_len, saved_bs = model.max_seq_len, model.max_batch_size
    model.max_seq_len = T
    try:
        res = model.forward_generate(x_in, torch.arange(T, device=dev),
                                     audio_masks=audio_masks, audio_parts=audio_parts, kv_len=T)
    finally:
        model.max_seq_len, model.max_batch_size = saved_len, saved_bs
        for layer in model.layers:
            layer.attention.kv_cache = None
    for (kc, vc), c in zip(model._row_caches, tmp):
        kc[row, :, :T] = c.k_cache[0]
        vc[row, :, :T] = c.v_cache[0]
    # forward_generate's last-token logits are over the full vocab (or the
    # model's own head); take the sliced rows.
    logits = res.logits[0, -1][model._head_ids]
    return logits, res.hidden_states[:, -1:]


class Int8Linear(nn.Module):
    """Weight-only int8 Linear (per-output-channel scales)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        self.register_buffer("weight", torch.round(w / scale[:, None]).to(torch.int8))
        self.register_buffer("scales", scale.to(lin.weight.dtype))
        self.bias = lin.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight.to(x.dtype)) * self.scales
        return y if self.bias is None else y + self.bias


def quantize_int8(model, slow: bool = True, fast: bool = True) -> int:
    names = ("wqkv", "wo", "w1", "w2", "w3")
    stacks = ([model.layers] if slow else []) + ([model.fast_layers] if fast else [])
    n = 0
    for stack in stacks:
        for mod in stack.modules():
            for name in names:
                lin = getattr(mod, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(mod, name, Int8Linear(lin))
                    n += 1
    torch.cuda.empty_cache()
    return n
