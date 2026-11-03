# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch DeepSeek V4.1 model for transformers library.

Packed-sequence native training implementation. The packed inputs follow the
engine convention: ``attention_mask`` carries per-token sequence ids (0 for
padding, 1..N inside each packed segment) and ``position_ids`` carries
per-segment positions. Extra packed metadata keys (``cu_seqlens``,
``max_length_q`` ...) are absorbed via ``**kwargs``.

v4.1-specific mechanisms implemented here:

- CSA2 cross-layer sharing: ``SharedAttentionRuntime`` is a per-forward
  container of slots (compressed KV, index K, top-k indices, candidate
  blocks) published by kv/index source layers and consumed by Reindex/Reuse
  layers. Tensors keep their autograd graph, so consumer layers backprop into
  the source layer's compressor.
- Because of that cross-layer graph, the kv-source publish step
  (``DecoderLayer.publish_shared``) runs before the layer call, outside the
  engine's per-layer activation checkpointing; a checkpointed publish would
  orphan the shared tensors' graph.
- Engram conditional memory: the n-gram hash is computed from ``input_ids``
  inside the forward (deterministic buffers, no learned parameters). The
  full-table ``DeepseekV41Engram`` is the HF/reference variant;
  ``RowShardedEngram`` (built by the parallel plan) holds a per-rank row
  slice as a plain tensor so the 2x183GiB tables never enter FSDP.

Deliberate differences vs deepseek_v4 modeling: delayed hyper-connections
(coefficient sets apply one sublayer later; no learned hc_head parameters),
no extra per-head q norm, no hadamard rotation in the indexer, no ape/overlap
in the compressor, no hash routing, and RMSNorm casts back to the input
dtype. The indexer auxiliary loss (DSAIndexerLoss in v4) is not ported yet.

Cross-layer tensors (CSA2 shared slots, delayed hc pre-mix) flow between
per-layer checkpoint regions; this requires NON-REENTRANT activation
checkpointing (the engine default). kv-source layers additionally publish
their shared slots outside the checkpointed region for robustness.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HCSplitSinkhorn,
    HcPost,
)
from ..deepseek_v4 import packing_utils as pu
from .configuration_deepseek_v41 import DeepseekV41Config


# ---------------------------------------------------------------------------
# Rotary helpers (computed from positions on the fly; no 1M-row tables).
# ---------------------------------------------------------------------------


def _yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    original_seq_len: int,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """Inverted frequencies with YaRN correction (v4 formula)."""

    def correction_dim(num_rotations):
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
    smooth = 1 - ramp

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return freqs / factor * (1 - smooth) + freqs * smooth


def apply_rope_at_positions(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    positions: torch.LongTensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the last dim of ``x`` (the rope part) at the given positions.

    ``x``: [b, s, d] or [b, s, h, d]; ``positions``: [b, s].
    """
    freqs = positions.unsqueeze(-1).float() * inv_freq.float().to(x.device)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    if inverse:
        freqs_cis = freqs_cis.conj()
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(x_complex.size(0), x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(x_complex.size(0), x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(x.dtype)


def _rope_tail(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    positions: torch.LongTensor,
    rope_head_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate only the trailing rope dims of the last dimension, in place of
    the full-tensor in-place style used by the reference implementation."""
    x_nope, x_rope = torch.split(x, [x.size(-1) - rope_head_dim, rope_head_dim], dim=-1)
    x_rope = apply_rope_at_positions(x_rope, inv_freq, positions, inverse=inverse)
    return torch.cat([x_nope, x_rope], dim=-1)


class DeepseekV41RMSNorm(nn.Module):
    """RMSNorm with fp32 compute and cast back to the input dtype."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(dtype)


# ---------------------------------------------------------------------------
# Delayed hyper-connections.
# ---------------------------------------------------------------------------


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Initial one-hot mix: collapse onto copy 0."""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


def collapse_hc(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the hc copies into one, weighted by pre_mix.

    x: [b, s, hc, dim]; pre_mix: [b, s, hc].
    """
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)


class DeepseekV41HyperConnections(nn.Module):
    """v4.1 hyper-connections with delayed application.

    Coefficient sets are computed from the stream at each sublayer entry but
    applied one sublayer later: the attention input is collapsed with the
    previous layer's ffn pre-mix, the ffn input with this layer's attention
    pre-mix, and the final collapse with the last layer's ffn pre-mix. This
    is why v4.1 has no learned hc_head parameters (unlike v4).
    """

    def __init__(self, hc_mult: int, sinkhorn_iters: int, hc_eps: float, norm_eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_split_sinkhorn = DeepseekV4HCSplitSinkhorn(hc_mult, sinkhorn_iters, hc_eps)

    def mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b, s, hc, dim] -> (pre, post, comb) coefficient sets."""
        x_flat = x.flatten(2).float()
        # Normalized over the whole flattened hc*d stream, one statistic per
        # token.
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        return self.hc_split_sinkhorn(mixes, hc_scale, hc_base)


class DeepseekV41Head(nn.Module):
    """Output head: fp32 projection over the normed collapsed stream."""

    def __init__(self, text_config):
        super().__init__()
        self.vocab_size = text_config.vocab_size
        self.dim = text_config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        return F.linear(norm(x).float(), self.weight)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)


# ---------------------------------------------------------------------------
# Packed sliding-window indices.
# ---------------------------------------------------------------------------


def get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    device: torch.device,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Sliding-window candidate positions per query, packed-aware.

    Rows pack multiple segments; window slots belonging to another segment
    (or to padding) are set to -1 via the sequence-id check. Invalid slots
    stay -1 because the selection ``where(mask, window_topk, -1)`` uses the
    original (already -1) values, never the clamped gather indices.
    """
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    window_topk = (base - window_size + 1).clamp(0) + torch.arange(
        min(seqlen, window_size), device=device
    )
    window_topk = torch.where(window_topk > base, -1, window_topk)  # [s, w]
    window_topk = window_topk.unsqueeze(0).expand(bsz, -1, -1)

    window_token_indices = window_topk.clamp_min(0).reshape(bsz, -1)
    batch_index = torch.arange(bsz, device=device).unsqueeze(1)
    window_ids = attention_mask[batch_index, window_token_indices].view_as(window_topk)
    query_ids = attention_mask.unsqueeze(-1)
    mask = (window_ids == query_ids) & (query_ids > 0)
    return torch.where(mask, window_topk, torch.full_like(window_topk, -1))


# ---------------------------------------------------------------------------
# Two-level candidate pre-filtering (hierarchical sparse indexer level 1).
# ---------------------------------------------------------------------------


def select_candidate_blocks_packed(
    index_score: torch.Tensor,
    block_ids: torch.Tensor,
    block_positions: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    ratio: int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Keep the ``topk_blocks`` best candidate blocks per query.

    Blocks group ``block_size`` consecutive row-blocks of the packed block
    list. A block becomes visible to a query once the query has passed its
    last token; invisible entries are already -inf in ``index_score``, which
    also makes an all-invisible candidate block score -inf (dropped). The
    candidate block holding the query's newest visible block is pinned in.
    Returns a bool mask shaped like ``index_score`` (True = inside the pool).
    """
    bsz, seqlen, num_blocks = index_score.shape
    pad = -num_blocks % block_size
    padded = F.pad(index_score, (0, pad), value=float("-inf"))
    block_scores = padded.view(bsz, seqlen, -1, block_size).amax(dim=-1)
    num_cand_blocks = block_scores.size(-1)

    # Index of the last block visible to each query: blocks of the query's
    # own segment whose start position is before the next block boundary.
    boundary = ((position_ids + 1) // ratio) * ratio  # [b, s]
    visible = (block_ids.unsqueeze(1) == attention_mask.unsqueeze(-1)) & (
        block_positions.unsqueeze(1) < boundary.unsqueeze(-1)
    )  # [b, s, T]
    last = visible.sum(dim=-1, keepdim=True) - 1  # [b, s, 1]

    cand_index = torch.arange(num_cand_blocks, device=index_score.device)
    block_scores = block_scores.masked_fill(cand_index == last, float("inf"))

    top = block_scores.topk(min(topk_blocks, num_cand_blocks), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > float("-inf")
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :num_blocks]


# ---------------------------------------------------------------------------
# CSA2 runtime shared across layers of one forward.
# ---------------------------------------------------------------------------


class SharedAttentionRuntime:
    """Per-forward slots handed down the layer stack (inference's
    SharedAttentionRuntime, training edition). Sources write before their
    consumers read; every forward starts from a fresh instance, and
    ``published`` records layers whose shared tensors were produced outside
    activation checkpointing (see DecoderLayer.publish_shared)."""

    def __init__(self):
        self.compress_kv: Optional[torch.Tensor] = None
        self.index_k: Optional[torch.Tensor] = None
        self.topk_idxs: Optional[torch.Tensor] = None
        self.candidates: Optional[torch.Tensor] = None
        self.published: set = set()


# ---------------------------------------------------------------------------
# Compressor and indexer.
# ---------------------------------------------------------------------------


class DeepseekV41Compressor(nn.Module):
    """Packed KV compressor.

    ratio == 1 is a plain bf16 projection (no gate); ratio > 1 pools ratio
    tokens with a learned softmax gate in fp32 (wkv/wgate stored fp32). Both
    paths emit the PRE-rope latent for the packed complete blocks, ordered by
    the block list built by ``packing_utils``.
    """

    def __init__(self, config, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if compress_ratio > 1:
            self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.float32)
            self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.float32)
        else:
            self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        """Return the pre-rope latent [b, T, head_dim] or None (no block)."""
        dtype = x.dtype
        if self.compress_ratio > 1:
            # Softmax pooling runs in fp32; the checkpoint stores fp32
            # weights to match. The pooled latent returns to bf16 before the
            # norm so downstream concat with window KV stays single-dtype.
            x = x.float()
            kv = self.wkv(x)
            score = self.wgate(x)
        else:
            kv = self.wkv(x)
            score = torch.zeros_like(kv)
        # v4.1 has no learned position bias; a zero ape keeps the shared
        # packed pooling helper on the plain softmax path.
        ape = torch.zeros(
            self.compress_ratio, self.head_dim, dtype=score.dtype, device=x.device
        )
        latent, _ = pu.compress_packed_kv(
            kv,
            score,
            ape,
            attention_mask,
            position_ids,
            self.compress_ratio,
            self.head_dim,
            overlap=False,
        )
        if latent is None:
            return None
        return self.norm(latent.to(dtype))

    def init_weights(self, _init_std: float):
        for linear in [self.wkv] + ([self.wgate] if self.compress_ratio > 1 else []):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.norm.weight, mean=1, std=0.02)


class DeepseekV41Indexer(nn.Module):
    """Lightning indexer over the shared compressed KV (no hadamard, no fp4).

    ``owns_k`` layers (kv sources) project their own compressor latent into
    index keys and publish them; the rest score against ``shared.index_k``.
    The candidate-source layer publishes the level-1 block pool; later index
    sources mask their scores into that pool.
    """

    def __init__(self, config, compress_ratio: int, owns_k: bool, is_candidate_source: bool):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.is_candidate_source = is_candidate_source
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim ** -0.5
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: Optional[torch.Tensor],
        inv_freq: torch.Tensor,
        shared: SharedAttentionRuntime,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        """Return (topk row-block indices [b, s, k] with -1 padding,
        index_score [b, s, T] for the future indexer aux loss)."""
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim

        block_ids, block_positions = pu.packed_compress_kv_ids_and_positions(
            attention_mask, self.compress_ratio, position_ids
        )
        num_blocks = block_ids.size(1)

        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.head_dim)
        q = _rope_tail(q, inv_freq, position_ids, rd)

        if self.owns_k:
            if latent is None:
                return None, None
            k = self.k_norm(self.wk(latent))
            k = _rope_tail(k, inv_freq, block_positions, rd)
            shared.index_k = k
        else:
            k = shared.index_k
            if k is None or k.size(1) != num_blocks:
                return None, None

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        invisible = pu.packed_compress_mask(
            attention_mask,
            block_ids.unsqueeze(1),
            block_positions.unsqueeze(1),
            self.compress_ratio,
            position_ids,
        )
        index_score = index_score + torch.where(
            invisible, torch.finfo(index_score.dtype).min, torch.zeros((), dtype=index_score.dtype, device=x.device)
        )

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks_packed(
                index_score,
                block_ids,
                block_positions,
                attention_mask,
                position_ids,
                self.compress_ratio,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif shared.candidates is not None:
            index_score = index_score.masked_fill(~shared.candidates, torch.finfo(index_score.dtype).min)

        topk = min(self.index_topk, num_blocks)
        if topk == 0:
            return index_score.new_full((bsz, seqlen, 0), -1, dtype=torch.long), index_score
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values

        batch_index = torch.arange(bsz, device=x.device).unsqueeze(1)
        topk_ids = block_ids[batch_index, idxs]
        topk_pos = block_positions[batch_index, idxs]
        invalid = pu.packed_compress_mask(
            attention_mask, topk_ids, topk_pos, self.compress_ratio, position_ids
        )
        return torch.where(invalid, -1, idxs), index_score

    def init_weights(self, _init_std: float):
        for linear in [self.wq_b, self.weights_proj]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        if self.owns_k:
            nn.init.trunc_normal_(self.wk.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.k_norm.weight, mean=1, std=0.02)


# ---------------------------------------------------------------------------
# Attention: pre / inner / post structure (kept for SP patching, see v4).
# ---------------------------------------------------------------------------


class DeepseekV41PreAttentionOps(nn.Module):
    """Projections, window KV, and the shared compressed KV production."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.eps = config.rms_norm_eps
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.window_size = config.sliding_window

    def forward(
        self,
        x: torch.Tensor,
        freqs_inv_plain: torch.Tensor,
        freqs_inv_yarn: torch.Tensor,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor,
        indexer,
        shared: SharedAttentionRuntime,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        bsz, seqlen, _ = x.size()
        # Layers with compressed KV rope everything with the compress table
        # (YaRN + compress_rope_theta); pure sliding-window layers use the
        # plain table (matches the reference implementation).
        inv_freq = freqs_inv_yarn if self.compress_ratio > 0 else freqs_inv_plain
        rd = self.rope_head_dim

        qr = q_norm(wq_a(x))
        q = wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = _rope_tail(q, inv_freq, position_ids, rd)

        kv = kv_norm(wkv(x))
        kv = _rope_tail(kv, inv_freq, position_ids, rd)

        window_topk = get_window_topk_idxs(
            self.window_size, bsz, seqlen, x.device, attention_mask
        )

        kv_compress = None
        compress_topk = None
        if self.compress_ratio > 0:
            if self.layer_idx in shared.published:
                # Shared tensors were produced outside checkpointing by
                # publish_shared; just read the slots.
                kv_compress = shared.compress_kv
                compress_topk = shared.topk_idxs
            else:
                latent = None
                if compressor is not None:
                    latent = compressor(x, attention_mask, position_ids)
                if indexer is not None:
                    compress_topk, _ = indexer(
                        x, qr, latent, inv_freq, shared, attention_mask, position_ids
                    )
                    shared.topk_idxs = compress_topk
                else:
                    compress_topk = shared.topk_idxs
                if latent is not None:
                    # Rope the latent at its block-start positions and
                    # publish for this group's consumer layers.
                    block_ids, block_positions = pu.packed_compress_kv_ids_and_positions(
                        attention_mask, self.compress_ratio, position_ids
                    )
                    kv_compress = _rope_tail(latent, inv_freq, block_positions, rd)
                    shared.compress_kv = kv_compress
                else:
                    kv_compress = shared.compress_kv

        return q, kv, window_topk, kv_compress, compress_topk


class DeepseekV41InnerAttentionOps(nn.Module):
    """Sparse attention over window KV plus (optionally) compressed KV."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.softmax_scale = config.head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        window_topk: torch.Tensor,
        kv_compress: Optional[torch.Tensor],
        compress_topk: Optional[torch.Tensor],
        attn_sink: torch.Tensor,
    ):
        bsz, seqlen, n_heads, head_dim = q.size()

        if kv_compress is not None and compress_topk is not None:
            offset = kv.size(1)
            topk_idxs = torch.cat(
                [window_topk, compress_topk + offset], dim=-1
            ).int()
            kv_states = torch.cat([kv, kv_compress], dim=1)
        else:
            topk_idxs = window_topk.int()
            kv_states = kv

        q_t = q.transpose(1, 2)  # [b, h, s, d]
        attn_weights = torch.matmul(q_t, kv_states.unsqueeze(1).transpose(2, 3))
        attn_weights = attn_weights * self.softmax_scale

        # +1 column for the attention sink; kv_states is [b, kv_len, d].
        index_mask = torch.full(
            (bsz, 1, seqlen, kv_states.size(1) + 1),
            fill_value=torch.finfo(attn_weights.dtype).min,
            dtype=attn_weights.dtype,
            device=q.device,
        )
        valid_mask = topk_idxs >= 0
        if valid_mask.any():
            b_indices, s_indices, k_indices = torch.where(valid_mask)
            valid_topk = topk_idxs[b_indices, s_indices, k_indices]
            index_mask[b_indices, 0, s_indices, valid_topk] = 0
        attn_weights = attn_weights + index_mask[..., :-1]

        sinks = attn_sink.reshape(1, -1, 1, 1).to(attn_weights.dtype)
        sinks = sinks.expand(bsz, -1, seqlen, -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

        probs = F.softmax(combined_logits.float(), dim=-1, dtype=attn_weights.dtype)
        attn_output = torch.matmul(probs[..., :-1], kv_states)  # [b, h, s, d]
        return attn_output.transpose(1, 2).contiguous()


class DeepseekV41PostAttentionOps(nn.Module):
    """Inverse rope on the output tail plus the grouped low-rank output
    projection."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        self.rope_head_dim = config.qk_rope_head_dim
        self.head_dim = config.head_dim

    def forward(
        self,
        o: torch.Tensor,
        inv_freq: torch.Tensor,
        position_ids: torch.LongTensor,
        bsz: int,
        seqlen: int,
        n_local_groups: int,
        wo_a,
        wo_b,
    ):
        o = _rope_tail(o, inv_freq, position_ids, self.rope_head_dim, inverse=True)
        o = o.view(bsz, seqlen, n_local_groups, -1)
        wo_a_weight = wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a_weight)
        return wo_b(o.reshape(bsz, seqlen, -1))


class DeepseekV41Attention(nn.Module):
    """Latent attention with sliding window plus CSA2 compressed KV.

    Parameters are defined here directly (checkpoint-compatible names);
    the compressor exists only on kv-source layers and the indexer only on
    index-source layers.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.eps = config.rms_norm_eps
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids

        self.q_lora_rank = config.q_lora_rank
        self.wq_a = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DeepseekV41RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, self.eps)

        self.o_lora_rank = config.o_lora_rank
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, config.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))

        if self.is_kv_source:
            self.compressor = DeepseekV41Compressor(config, self.compress_ratio)
        if self.is_index_source:
            self.indexer = DeepseekV41Indexer(
                config,
                self.compress_ratio,
                owns_k=self.is_kv_source,
                is_candidate_source=layer_idx == config.candidate_source_layer_id,
            )

        self.pre_attention_ops = DeepseekV41PreAttentionOps(config, layer_idx)
        self.inner_attention_ops = DeepseekV41InnerAttentionOps(config, layer_idx)
        self.post_attention_ops = DeepseekV41PostAttentionOps(config)

        # Lazy rope frequencies: plain (theta, no YaRN) and compress
        # (YaRN + compress theta). Built on first use; None buffers are
        # skipped by FSDP/to_empty.
        self.register_buffer("inv_freq_plain", None, persistent=False)
        self.register_buffer("inv_freq_yarn", None, persistent=False)

    def _ensure_inv_freq(self, device: torch.device):
        if self.inv_freq_plain is None:
            dim = self.rope_head_dim
            freqs = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            self.inv_freq_plain = freqs.to(device)
        if self.inv_freq_yarn is None:
            self.inv_freq_yarn = _yarn_inv_freq(
                self.rope_head_dim,
                self.config.compress_rope_theta,
                self.config.rope_factor,
                self.config.original_max_position_embeddings,
                self.config.beta_fast,
                self.config.beta_slow,
            ).to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared: SharedAttentionRuntime,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        **kwargs,
    ):
        bsz, seqlen, _ = hidden_states.size()
        self._ensure_inv_freq(hidden_states.device)

        compressor = self.compressor if self.is_kv_source else None
        indexer = self.indexer if self.is_index_source else None

        q, kv, window_topk, kv_compress, compress_topk = self.pre_attention_ops(
            hidden_states,
            self.inv_freq_plain,
            self.inv_freq_yarn,
            self.wq_a,
            self.q_norm,
            self.wq_b,
            self.wkv,
            self.kv_norm,
            compressor,
            indexer,
            shared,
            attention_mask,
            position_ids,
        )

        o = self.inner_attention_ops(
            q, kv, window_topk, kv_compress, compress_topk, self.attn_sink
        )

        inv_freq = self.inv_freq_yarn if self.compress_ratio > 0 else self.inv_freq_plain
        # Groups shrink when SP splits heads; SP=1 keeps all of them.
        n_local_groups = self.n_groups // (self.n_heads // q.shape[2])
        return self.post_attention_ops(
            o,
            inv_freq,
            position_ids,
            bsz,
            seqlen,
            n_local_groups,
            self.wo_a,
            self.wo_b,
        )

    def init_weights(self, init_std: float):
        for linear in [self.wq_a, self.wq_b, self.wkv, self.wo_a, self.wo_b]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.q_norm.weight, mean=1, std=0.02)
        nn.init.trunc_normal_(self.kv_norm.weight, mean=1, std=0.02)
        nn.init.trunc_normal_(self.attn_sink, mean=0.0, std=init_std)
        if self.is_kv_source:
            self.compressor.init_weights(init_std)
        if self.is_index_source:
            self.indexer.init_weights(init_std)


# ---------------------------------------------------------------------------
# Engram conditional memory.
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def _find_next_prime(start: int, seen: set) -> int:
    candidate = start + 1
    while candidate in seen or not _is_prime(candidate):
        candidate += 1
    return candidate


def _build_engram_primes(
    num_layers: int,
    max_ngram_size: int,
    n_heads: int,
    engram_vocab_size: int,
) -> torch.Tensor:
    """Bucket modulus per (layer, n-gram size, head); primes are drawn in
    order from a global pool so bucket ranges never overlap."""
    seen: set = set()
    primes = []
    for _ in range(num_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes = []
            current = engram_vocab_size - 1
            for _ in range(n_heads):
                current = _find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    # [num_layers, max_ngram_size - 1, n_heads]
    return torch.tensor(primes, dtype=torch.int64)


def _compute_engram_multipliers(
    layer_ids,
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback); per-layer RNG so layers hash
    differently. Matches the reference (numpy PCG64, seed 10007 * layer)."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)  # [num_layers, max_ngram_size]


def build_compressed_token_map(tokenizer) -> tuple:
    """Map token ids onto the compressed id space where surface variants
    collapse together ("The"/"the"/"THE" hash identically).

    Ported from the reference implementation; the returned compressed vocab
    size is asserted against the config because a mismatch would silently
    rehash the whole engram table.
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


class NgramHashState(nn.Module):
    """Deterministic n-gram hash for the engram lookup (no parameters).

    Buffers are built lazily because tokenizer access is only possible once
    the model path is known (engine path passes it via ``build``; the HF
    reference path falls back to ``config._name_or_path``).
    """

    def __init__(self, top_config, text_config):
        super().__init__()
        self.layer_ids = tuple(text_config.engram_layer_ids)
        self.max_ngram_size = text_config.engram_max_ngram_size
        self.n_heads = text_config.engram_n_heads
        self.pad_token_id = text_config.engram_pad_token_id
        self.compressed_vocab_size = text_config.engram_compressed_vocab_size
        self.engram_vocab_size = text_config.engram_vocab_size
        self.vocab_size = text_config.vocab_size
        self._top_config = top_config

        self.register_buffer("token_map", None, persistent=False)
        self.register_buffer("multipliers", None, persistent=False)
        self.register_buffer("primes", None, persistent=False)
        self.register_buffer("offsets", None, persistent=False)

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    def build(self, device: torch.device, model_path: Optional[str] = None):
        if self.token_map is not None:
            return
        from transformers import AutoTokenizer

        path = model_path or getattr(self._top_config, "_name_or_path", None)
        if not path:
            raise RuntimeError(
                "NgramHashState needs a tokenizer path: pass model_path to "
                "build() or set config._name_or_path."
            )
        tokenizer = AutoTokenizer.from_pretrained(path)
        token_map, compressed_vocab_size = build_compressed_token_map(tokenizer)
        if compressed_vocab_size != self.compressed_vocab_size:
            raise RuntimeError(
                f"engram compressed vocab size mismatch: tokenizer gives "
                f"{compressed_vocab_size}, config expects "
                f"{self.compressed_vocab_size} - the table would be silently "
                f"rehashed. Check the tokenizer against the checkpoint."
            )

        primes = _build_engram_primes(
            len(self.layer_ids), self.max_ngram_size, self.n_heads, self.engram_vocab_size
        )
        multipliers = _compute_engram_multipliers(
            self.layer_ids, self.max_ngram_size, compressed_vocab_size
        )
        # Offsets: cumulative start index of each (n-gram, head) bucket range
        # inside the layer's table.
        offsets = []
        for layer_primes in primes.tolist():
            flat_sizes = [p for per_ngram in layer_primes for p in per_ngram]
            starts = [0]
            for size in flat_sizes[:-1]:
                starts.append(starts[-1] + size)
            offsets.append(starts)

        self.token_map = torch.tensor(token_map, dtype=torch.int64).to(device)
        self.multipliers = multipliers.to(device)
        self.primes = primes.to(device)
        self.offsets = torch.tensor(offsets, dtype=torch.int64).to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """hash ids [b, s, n_engram_layers, n_hash_cols]."""
        if self.token_map is None:
            self.build(input_ids.device)

        bsz, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]  # [b, s]
        seq = attention_mask

        positions = torch.arange(seqlen, device=input_ids.device)
        blocked = torch.zeros(bsz, seqlen, dtype=torch.bool, device=input_ids.device)
        tokens = []
        for shift in range(self.max_ngram_size):
            src_pos = (positions - shift).clamp_min(0)
            source = compressed.gather(1, src_pos.unsqueeze(0).expand(bsz, -1))
            src_seq = seq.gather(1, src_pos.unsqueeze(0).expand(bsz, -1))
            # An n-gram never crosses a segment boundary or touches padding.
            blocked = blocked | (positions - shift < 0) | (src_seq != seq) | (src_seq == 0) | (seq == 0)
            tokens.append(torch.where(blocked, self.pad_token_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [b, s, max_ngram_size]

        # XOR the multiplied ids one lookback at a time; each running value
        # is the hash of an (i+1)-gram landing in its own prime bucket.
        products = tokens.unsqueeze(2) * self.multipliers  # [b, s, L, n]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        hash_cols = torch.cat(hashes, dim=-1)  # [b, s, L, (n-1)*H]
        return hash_cols + self.offsets


class DeepseekV41Engram(nn.Module):
    """Gated n-gram lookup written into the residual stream.

    The full-table variant used by the HF/reference path; the training path
    swaps in RowShardedEngram (same param names, row-sharded table).
    """

    def __init__(self, text_config, layer_idx: int, layer_hash_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_hash_index = layer_hash_index
        self.dim = text_config.hidden_size
        self.hc_mult = text_config.hc_mult
        self.clamp_value = 1e-6
        self.eps = text_config.rms_norm_eps

        num_rows = text_config.engram_num_embeddings[layer_hash_index]
        n_hash_cols = (text_config.engram_max_ngram_size - 1) * text_config.engram_n_heads
        self.n_hash_cols = n_hash_cols
        self.embed = nn.Embedding(num_rows, text_config.engram_head_dim)
        self.wkv = nn.Linear(
            n_hash_cols * text_config.engram_head_dim,
            text_config.hidden_size * (text_config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))

    def _embed_lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(hash_ids)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        """x: [b, s, hc, dim]; hash_ids: [b, s, n_hash_cols]."""
        kv = self.wkv(self._embed_lookup(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        # Normalized per (token, hc copy) over dim, NOT jointly over copies.
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim ** -0.5
        # Signed sqrt before the sigmoid, matching the training kernel.
        gate = torch.sigmoid(
            torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)
        )
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

    def init_weights(self, _init_std: float):
        nn.init.trunc_normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wkv.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)


class RowShardedEngram(DeepseekV41Engram):
    """Training-path engram with the table row-sharded across ranks.

    The table is a plain tensor attribute (NOT an nn.Parameter): FSDP2 would
    all-gather a 183GiB parameter on every forward, and the generic weight
    loader broadcasts full tensors, so the shard is read directly from the
    safetensors file via ``load_table``. Lookup masks out-of-shard rows and
    all_reduces, so each hash id is summed exactly once (frozen phase 1:
    no gradient synchronization is involved).
    """

    def __init__(self, text_config, layer_idx: int, layer_hash_index: int):
        super().__init__(text_config, layer_idx, layer_hash_index)
        del self.embed  # full table never materializes on the training path
        self.table: Optional[torch.Tensor] = None
        self.table_key = f"layers.{layer_idx}.engram.embed.weight"
        self.num_rows = text_config.engram_num_embeddings[layer_hash_index]
        self.head_dim = text_config.engram_head_dim

    def init_weights(self, init_std: float):
        # No self.embed here; the table comes from load_table/init_table_random.
        nn.init.trunc_normal_(self.wkv.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)

    def load_table(self, model_path: str, device=None, group=dist.group.WORLD):
        """Read this rank's row slice from the checkpoint shard by shard.

        The table lives on the compute device (frozen, resident); slices are
        streamed in ~512MiB chunks so neither host nor device ever holds the
        full table.
        """
        from torchturbo.runtime import get_device_runtime

        if device is None:
            device = get_device_runtime().device
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        shard_rows = math.ceil(self.num_rows / world_size)
        row_start = rank * shard_rows
        row_end = min(row_start + shard_rows, self.num_rows)

        index_path = Path(model_path) / "model.safetensors.index.json"
        with open(index_path) as f:
            filename = json.load(f)["weight_map"][self.table_key]

        local = torch.empty(shard_rows, self.head_dim, dtype=torch.bfloat16, device=device)
        chunk_rows = max(1, (1 << 29) // (self.head_dim * 2))
        with safe_open(Path(model_path) / filename, framework="pt") as f:
            table_slice = f.get_slice(self.table_key)
            for start in range(row_start, row_end, chunk_rows):
                end = min(start + chunk_rows, row_end)
                local[start - row_start : end - row_start] = table_slice[start:end].to(
                    device
                )
        if row_end - row_start < shard_rows:
            local[row_end - row_start :].zero_()
        self.table = local

    def init_table_random(self):
        """Fallback for train-from-scratch runs without a checkpoint."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        shard_rows = math.ceil(self.num_rows / world_size)
        device = next(self.parameters()).device
        table = torch.empty(shard_rows, self.head_dim, dtype=torch.float32, device=device)
        nn.init.trunc_normal_(table, mean=0.0, std=0.02)
        self.table = table.to(torch.bfloat16)

    def _embed_lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        if self.table is None:
            raise RuntimeError(
                f"RowShardedEngram table not loaded for {self.table_key}; "
                f"call load_table() (the Turbo wrapper does this in "
                f"on_materialized)."
            )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        shard_rows = self.table.size(0)
        row_start = dist.get_rank() * shard_rows if dist.is_initialized() else 0
        mask = (hash_ids < row_start) | (hash_ids >= row_start + shard_rows)
        local = (hash_ids - row_start).masked_fill(mask, 0)
        values = F.embedding(local, self.table)
        values = values.masked_fill(mask.unsqueeze(-1), 0)
        if world_size > 1:
            dist.all_reduce(values)
        return values


# ---------------------------------------------------------------------------
# MoE (HF/reference path; the training path replaces ffn with MoELayer).
# ---------------------------------------------------------------------------


class DeepseekV41Gate(nn.Module):
    """noaux_tc gating: the bias steers selection only; routing weights come
    from the unbiased scores. bias_vl (image-span routing bias) is loaded
    from the checkpoint but only consumed on multimodal inputs."""

    def __init__(self, text_config, vision_enabled: bool):
        super().__init__()
        self.topk = text_config.num_experts_per_tok
        self.score_func = text_config.scoring_func
        self.norm_topk_prob = text_config.norm_topk_prob
        self.route_scale = text_config.routed_scaling_factor

        self.weight = nn.Parameter(
            torch.empty(text_config.n_routed_experts, text_config.hidden_size)
        )
        self.register_buffer(
            "bias", torch.zeros(text_config.n_routed_experts, dtype=torch.float32)
        )
        if vision_enabled:
            self.register_buffer(
                "bias_vl", torch.zeros(text_config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.bias_vl = None

    def forward(self, hidden_states: torch.Tensor):
        scores = F.linear(hidden_states.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()

        indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices

    def init_weights(self, _init_std: float):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=_init_std)


class DeepseekV41Expert(nn.Module):
    """SwiGLU FFN with the training clamps (up clamped both sides, gate from
    above)."""

    def __init__(self, text_config):
        super().__init__()
        self.swiglu_limit = text_config.swiglu_limit
        self.w1 = nn.Linear(text_config.hidden_size, text_config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(text_config.moe_intermediate_size, text_config.hidden_size, bias=False)
        self.w3 = nn.Linear(text_config.hidden_size, text_config.moe_intermediate_size, bias=False)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None):
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = x * weights
        return self.w2(x.to(dtype))


class DeepseekV41MoE(nn.Module):
    """Reference MoE (per-expert loop). The parallel plan replaces this with
    the fused MoELayer + GroupedMLP and reuses shared_experts."""

    def __init__(self, text_config, layer_idx: int, vision_enabled: bool):
        super().__init__()
        self.dim = text_config.hidden_size
        self.n_routed_experts = text_config.n_routed_experts
        self.gate = DeepseekV41Gate(text_config, vision_enabled)
        self.experts = nn.ModuleList(
            [DeepseekV41Expert(text_config) for _ in range(self.n_routed_experts)]
        )
        assert text_config.n_shared_experts == 1
        self.shared_experts = DeepseekV41Expert(text_config)

    def forward(self, hidden_states: torch.Tensor):
        shape = hidden_states.size()
        x = hidden_states.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        for i in range(self.n_routed_experts):
            mask = indices == i
            if mask.any():
                idx, top = torch.where(mask)
                y[idx] += self.experts[i](x[idx], weights[idx, top].unsqueeze(-1))
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    def init_weights(self, init_std: float):
        self.gate.init_weights(init_std)
        for expert in self.experts:
            for linear in [expert.w1, expert.w2, expert.w3]:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for linear in [self.shared_experts.w1, self.shared_experts.w2, self.shared_experts.w3]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(self, text_config, layer_idx: int, vision_enabled: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_eps = text_config.rms_norm_eps
        self.hc_mult = text_config.hc_mult
        self.is_kv_source = layer_idx in text_config.kv_source_layer_ids

        self.attn = DeepseekV41Attention(text_config, layer_idx)
        self.ffn = DeepseekV41MoE(text_config, layer_idx, vision_enabled)
        self.attn_norm = DeepseekV41RMSNorm(text_config.hidden_size, self.norm_eps)
        self.ffn_norm = DeepseekV41RMSNorm(text_config.hidden_size, self.norm_eps)

        self.engram = None
        self.engram_hash_index = -1
        if layer_idx in text_config.engram_layer_ids:
            self.engram_hash_index = text_config.engram_layer_ids.index(layer_idx)
            self.engram = DeepseekV41Engram(text_config, layer_idx, self.engram_hash_index)

        hc_mult = self.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * text_config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        self.hc = DeepseekV41HyperConnections(
            hc_mult, text_config.hc_sinkhorn_iters, text_config.hc_eps, self.norm_eps
        )
        self.hc_post = HcPost()

    def publish_shared(
        self,
        hidden_states: torch.Tensor,
        shared: SharedAttentionRuntime,
        pre_mix: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        """Run the compressor + indexer OUTSIDE activation checkpointing.

        The shared tensors (compressed KV, index K) are consumed by later
        layers, so their autograd graph must stay alive; a checkpointed
        publish would orphan it (re-entrant mode) or force a region
        recompute mid-backward. Cheap relative to the layer itself.
        """
        if not self.is_kv_source:
            return
        attn = self.attn
        attn._ensure_inv_freq(hidden_states.device)
        x = collapse_hc(hidden_states, pre_mix)
        x = self.attn_norm(x)
        latent = attn.compressor(x, attention_mask, position_ids)
        inv_freq = attn.inv_freq_yarn
        if attn.is_index_source:
            qr = attn.q_norm(attn.wq_a(x))
            topk, _ = attn.indexer(
                x, qr, latent, inv_freq, shared, attention_mask, position_ids
            )
            shared.topk_idxs = topk
        if latent is not None:
            block_ids, block_positions = pu.packed_compress_kv_ids_and_positions(
                attention_mask, attn.compress_ratio, position_ids
            )
            shared.compress_kv = _rope_tail(
                latent, inv_freq, block_positions, attn.rope_head_dim
            )
        shared.published.add(self.layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared: SharedAttentionRuntime,
        pre_mix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        engram_hashes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.engram is not None:
            if engram_hashes is None:
                raise ValueError(
                    f"layer {self.layer_idx} has an engram but engram_hashes "
                    f"is None; input_ids is required when engram layers exist."
                )
            hash_ids = engram_hashes[:, :, self.engram_hash_index, :]
            hidden_states = self.engram(hidden_states, hash_ids)

        # Attention sublayer: mixes from the attention-entry stream, applied
        # to the ffn input; the attention input uses the incoming pre-mix
        # (the previous layer's ffn pre).
        residual = hidden_states
        attn_pre, attn_post, attn_comb = self.hc.mixes(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden_states = collapse_hc(hidden_states, pre_mix)
        hidden_states = self.attn_norm(hidden_states)
        attn_output = self.attn(hidden_states, shared, attention_mask, position_ids)
        hidden_states = self.hc_post(attn_output, residual, attn_post, attn_comb)

        # FFN sublayer: the ffn pre-mix becomes the next layer's attention
        # input mix (returned to the caller).
        residual = hidden_states
        ffn_pre, ffn_post, ffn_comb = self.hc.mixes(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = collapse_hc(hidden_states, attn_pre)
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre

    def init_weights(self, init_std: float):
        for norm in [self.attn_norm, self.ffn_norm]:
            nn.init.trunc_normal_(norm.weight, mean=1, std=0.02)
        self.attn.init_weights(init_std)
        self.ffn.init_weights(init_std)
        if self.engram is not None:
            self.engram.init_weights(init_std)
        for param in [
            self.hc_attn_fn,
            self.hc_attn_base,
            self.hc_attn_scale,
            self.hc_ffn_fn,
            self.hc_ffn_base,
            self.hc_ffn_scale,
        ]:
            nn.init.trunc_normal_(param, mean=0.0, std=0.02)


# ---------------------------------------------------------------------------
# Vision tower and aligner (weights load; forward is exercised from the
# multimodal data path only).
# ---------------------------------------------------------------------------


class DeepseekV41VisionAttention(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.n_heads = vision_config.num_attention_heads
        self.head_dim = vision_config.hidden_size // self.n_heads
        self.wqkv = nn.Linear(vision_config.hidden_size, 3 * vision_config.hidden_size)
        self.wo = nn.Linear(vision_config.hidden_size, vision_config.hidden_size)

    def forward(self, x, cos, sin):
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q, k = (
            _apply_vision_rotary(t, cos, sin) for t in (q, k)
        )
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(o.transpose(0, 1).reshape(n, -1))


def _apply_vision_rotary(x, cos, sin):
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(dtype)


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.w1 = nn.Linear(
            vision_config.hidden_size, 2 * vision_config.intermediate_size, bias=False
        )
        self.w2 = nn.Linear(vision_config.intermediate_size, vision_config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.norm1 = DeepseekV41RMSNorm(vision_config.hidden_size)
        self.attn = DeepseekV41VisionAttention(vision_config)
        self.norm2 = DeepseekV41RMSNorm(vision_config.hidden_size)
        self.mlp = DeepseekV41VisionMLP(vision_config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.proj = nn.Linear(
            3 * vision_config.patch_size ** 2, vision_config.hidden_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DeepseekV41VisionTower(nn.Module):
    """DeepSeek ViT: bidirectional attention over one image with 2D RoPE."""

    def __init__(self, vision_config):
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.n_heads = vision_config.num_attention_heads
        self.rope_theta = vision_config.rope_theta
        self.patch_embed = _PatchEmbed(vision_config)
        self.blocks = nn.ModuleList(
            [DeepseekV41VisionBlock(vision_config) for _ in range(vision_config.num_hidden_layers)]
        )
        self.norm = DeepseekV41RMSNorm(vision_config.hidden_size)

    def _cos_sin(self, n_h, n_w, device):
        rope_dim = self.hidden_size // self.n_heads // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        inv_freq = inv_freq.to(device)
        hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
        wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
        freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
        freqs = freqs.flatten(1)
        return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed.proj(patches.flatten(1))
        cos, sin = self._cos_sin(n_h, n_w, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV41Aligner(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.downsample_ratio = vision_config.downsample_ratio
        in_dim = vision_config.hidden_size * self.downsample_ratio ** 2
        self.w1 = nn.Linear(in_dim, text_config.hidden_size)
        self.w2 = nn.Linear(text_config.hidden_size, text_config.hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class DeepseekV41PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV41Config
    base_model_prefix = ""  # checkpoint weights have no "model." prefix
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", None)
        if std is None:
            std = self.config.text_config.initializer_range
        if isinstance(module, DeepseekV41DecoderLayer):
            module.init_weights(std)
        elif isinstance(module, DeepseekV41Head):
            module.init_weights(std)
        elif isinstance(module, (DeepseekV41Engram, RowShardedEngram)):
            module.init_weights(std)
        elif isinstance(module, DeepseekV41RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    """DeepSeek V4.1 for causal LM (flat layout, checkpoint-compatible)."""

    _tied_weights_keys = ["head.weight"]

    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        text_config = config.text_config
        if text_config.num_nextn_predict_layers > 0:
            raise NotImplementedError(
                "DSpark MTP layers are not implemented yet (phase 1 trains "
                "the backbone only); enable the engine MTP config once "
                "dspark support lands."
            )
        self.vocab_size = text_config.vocab_size
        self.padding_idx = config.pad_token_id

        self.embed = nn.Embedding(
            text_config.vocab_size, text_config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(text_config, i, config.vision_enabled)
                for i in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV41RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.head = DeepseekV41Head(text_config)

        if config.vision_enabled:
            self.vision = DeepseekV41VisionTower(config.vision_config)
            self.aligner = DeepseekV41Aligner(config.vision_config, text_config)
            self.image_start = nn.Parameter(torch.empty(text_config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(text_config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(text_config.hidden_size))

        self.engram_hash = (
            NgramHashState(config, text_config)
            if text_config.engram_layer_ids
            else None
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head.weight = (
            new_embeddings.weight if hasattr(new_embeddings, "weight") else new_embeddings
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": kwargs.get("attention_mask"),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """The packed metadata keys added by the engine (``cu_seqlens``,
        ``cu_seq_lens_q``/``k``, ``max_length_q``/``k``, ``max_seqlen``) are
        absorbed by ``**kwargs`` and unused here: the packed semantics live
        in ``attention_mask`` (sequence ids) and ``position_ids``."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        batch_size, seq_length = inputs_embeds.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0).expand(batch_size, -1)

        if self.engram_hash is not None and input_ids is None:
            raise ValueError("input_ids is required when the model has engram layers")

        engram_hashes = None
        if self.engram_hash is not None:
            engram_hashes = self.engram_hash(input_ids, attention_mask)

        hc_mult = self.config.text_config.hc_mult
        hidden_states = inputs_embeds.unsqueeze(2).repeat(1, 1, hc_mult, 1)
        pre_mix = make_identity_pre_mix(hidden_states, hc_mult)

        all_hidden_states = () if output_hidden_states else None
        shared = SharedAttentionRuntime()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # kv-source layers always publish their shared tensors before the
            # layer call: the engine owns activation checkpointing (it wraps
            # whole layers via the parallel plan), and a checkpointed layer
            # would otherwise orphan the cross-layer shared autograd graph.
            if decoder_layer.is_kv_source:
                decoder_layer.publish_shared(
                    hidden_states, shared, pre_mix, attention_mask, position_ids
                )
            hidden_states, pre_mix = decoder_layer(
                hidden_states,
                shared,
                pre_mix,
                attention_mask=attention_mask,
                position_ids=position_ids,
                engram_hashes=engram_hashes,
            )

        # Final collapse with the last layer's ffn pre-mix (delayed HC), then
        # norm + fp32 projection.
        hidden_states = collapse_hc(hidden_states, pre_mix)
        logits = self.head(hidden_states, self.norm)

        if output_hidden_states:
            all_hidden_states += (hidden_states.mean(dim=2),)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )
