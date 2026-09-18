"""TTTTransformer: a Llama-3 style decoder split into a frozen prefix and a TTT suffix.

Layout mirrors e2e's BlockCollectionSplit (github.com/test-time-training/e2e,
ttt/model/transformer.py), adapted to PyTorch:

    blocks[0 : first_fast_layer]      PREFIX  - no fast weights below, so the
                                               meta-gradient reaches these blocks
                                               as a single VJP. Fused attention OK.
    blocks[first_fast_layer : L]      SUFFIX  - carry the second-order path.
                                               Math SDPA backend required.

The prefix runs once over the whole sequence. The suffix runs chunk by chunk,
threading a rolling KV cache and the current fast weights W_i.

Fast weights are supplied as a dict of tensors and injected with
`torch.func.functional_call`, which overrides only the named parameters and
leaves the rest of the module bound to its own nn.Parameters. That keeps W_i in
the autograd graph (an in-place optimizer step would sever it).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ttt.config import ModelConfig
from ttt.model.attention import KVCache
from ttt.model.block import TransformerBlock
from ttt.model.rope import build_rope_cache


class TTTTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, *, max_seq_len: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = max_seq_len

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # Blocks below first_fast_layer never see a fast weight -> fused attention.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg, use_math_backend=(i >= cfg.first_fast_layer))
                for i in range(cfg.num_layers)
            ]
        )
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # One learned log-multiplier per fast tensor (Meta-SGD / MAML++ LSLR, and LaCT's
        # per-matrix learned inner LR). It is a SLOW parameter: the outer loop learns how
        # fast each fast tensor should move. exp() keeps it positive; init 0 => multiplier 1.
        self.inner_lr_log = nn.ParameterDict(
            {self._mangle(k): nn.Parameter(torch.zeros(())) for k in self.fast_param_names()}
        )

        cos, sin = build_rope_cache(cfg.head_dim, max_seq_len, cfg.rope)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------ utils

    @property
    def first_fast_layer(self) -> int:
        return self.cfg.first_fast_layer

    @staticmethod
    def _mangle(name: str) -> str:
        """nn.ParameterDict keys cannot contain '.'"""
        return name.replace(".", "__")

    def fast_param_names(self) -> list[str]:
        """The MLP projections of the last `fast_blocks` blocks, in a fixed order.

        Must agree with ttt.model.naming.is_fast_param; tested in test_trainer.py.
        """
        return [
            f"blocks.{i}.mlp.{w}.weight"
            for i in range(self.cfg.first_fast_layer, self.cfg.num_layers)
            for w in ("w1", "w2", "w3")
        ]

    def inner_lr_multipliers(self) -> dict[str, Tensor]:
        """exp(inner_lr_log), keyed by fast-parameter name. Stays in the graph: it is
        meta-learned, so detaching it would silently freeze the learned inner LR."""
        return {k: torch.exp(self.inner_lr_log[self._mangle(k)]) for k in self.fast_param_names()}

    def _rope_slice(self, start: int, length: int) -> tuple[Tensor, Tensor]:
        assert start + length <= self.max_seq_len, (
            f"positions [{start}, {start + length}) exceed max_seq_len {self.max_seq_len}"
        )
        return self.rope_cos[start : start + length], self.rope_sin[start : start + length]

    def _project_logits(self, h: Tensor) -> Tensor:
        h = self.norm(h)
        weight = self.embed_tokens.weight if self.cfg.tie_word_embeddings else self.lm_head.weight
        return torch.nn.functional.linear(h, weight)

    # ----------------------------------------------------------------- caches

    def init_caches(self, *, batch: int, device, dtype) -> list[KVCache]:
        """One cache per SUFFIX block. Zero-initialised; invalid positions are
        excluded by the sliding-window mask (absolute index >= 0), exactly as in
        e2e's SWA.sw_causal_mask."""
        return [
            KVCache.empty(batch, self.cfg.window_size, self.cfg.num_kv_heads, self.cfg.head_dim,
                          device=device, dtype=dtype)
            for _ in range(self.cfg.fast_blocks)
        ]

    @staticmethod
    def flatten_caches(caches: list[KVCache]) -> tuple[Tensor, ...]:
        """(k0, v0, k1, v1, ...) so the carry can cross a checkpoint boundary.

        `length` is deliberately NOT carried: it is a deterministic function of the
        chunk index, min(window, (i+1)*chunk_size), so recomputing it avoids an
        int tensor in the carry."""
        out: list[Tensor] = []
        for c in caches:
            out.extend((c.k, c.v))
        return tuple(out)

    def unflatten_caches(self, flat: tuple[Tensor, ...], chunk_index: int) -> list[KVCache]:
        """Rebuild caches at the START of chunk `chunk_index`.

        length = min(window_size, chunk_index * chunk_size): the number of valid
        cached positions is a deterministic function of how many chunks have been
        consumed, so it is recomputed here rather than carried as an int tensor
        across the checkpoint boundary.
        """
        assert len(flat) == 2 * self.cfg.fast_blocks, (
            f"expected {2 * self.cfg.fast_blocks} cache tensors, got {len(flat)}"
        )
        seen = chunk_index * self.cfg.chunk_size
        length = torch.tensor(min(self.cfg.window_size, seen), dtype=torch.int64, device=flat[0].device)
        return [
            KVCache(k=flat[2 * i], v=flat[2 * i + 1], length=length)
            for i in range(self.cfg.fast_blocks)
        ]

    # ---------------------------------------------------------------- forward

    def prefix_forward(self, input_ids: Tensor) -> Tensor:
        """Run the frozen prefix over the whole sequence in one shot. [B, T] -> [B, T, d]."""
        b, t = input_ids.shape
        h = self.embed_tokens(input_ids)
        cos, sin = self._rope_slice(0, t)
        for i in range(self.first_fast_layer):
            h, _ = self.blocks[i](h, cos, sin, None)
        return h

    def suffix_forward(
        self,
        h: Tensor,  # [B, b, d] prefix output for this chunk
        *,
        fast: dict[str, Tensor],
        caches: list[KVCache],
        chunk_index: int,
    ) -> tuple[Tensor, list[KVCache]]:
        """Run the TTT suffix on one chunk with fast weights `fast`.

        `fast` is keyed by full model parameter names ("blocks.12.mlp.w1.weight");
        each block receives only its own entries, stripped of the "blocks.{i}." prefix.
        """
        chunk = self.cfg.chunk_size
        start = chunk_index * chunk
        assert h.shape[1] == chunk, f"expected chunk of {chunk} tokens, got {h.shape[1]}"
        cos, sin = self._rope_slice(start, chunk)

        new_caches: list[KVCache] = []
        for j, layer in enumerate(range(self.first_fast_layer, self.cfg.num_layers)):
            block = self.blocks[layer]
            prefix = f"blocks.{layer}."
            overrides = {k[len(prefix) :]: v for k, v in fast.items() if k.startswith(prefix)}
            assert overrides, f"no fast weights supplied for suffix block {layer}"
            h, cache = torch.func.functional_call(block, overrides, (h, cos, sin, caches[j]))
            new_caches.append(cache)
        return self._project_logits(h), new_caches

    def forward(self, *args, **kwargs):  # pragma: no cover - the model is driven by the two methods above
        raise RuntimeError("Use prefix_forward / suffix_forward; TTTTransformer has no single forward.")
