"""The TTT inner loop: chunked test-time training with checkpointing through time.

This is the core of the method. For one sequence of T tokens split into N = T/b
chunks, with fast weights W (the MLP matrices of the last `fast_blocks` blocks):

    W_0 = pretrained MLP weights                       (reset at every sequence)
    for i = 1..N:
        l_i = CE( f(x_i ; W_{i-1}, theta) , y_i )      # loss BEFORE the update
        g_i = d l_i / d W_{i-1}                        # create_graph=True
        W_i = InnerOpt(W_{i-1}, g_i)
    L(theta) = (1/N) sum_i l_i                         # TTT-E2E Eq. 6

The outer loop differentiates L w.r.t. the slow parameters theta, which requires
gradients of gradients through the whole chain W_0 -> ... -> W_N.

Memory. Materialising every W_i costs N copies of the fast weights. We instead
checkpoint through time (TTT-E2E footnote 9; TTT arXiv:2407.04620 Appendix C):
chunks are processed in groups of `remat_group`, only the carry (fast weights +
KV caches) is stored at group boundaries, and the interior of a group is
recomputed during backward. Peak fast-weight storage is

    (N / g + g) copies      instead of      N copies

Measured caveat: because the outer pass is a double backward, a checkpointed
group is re-executed 2-3.5x rather than once, and the per-chunk recompute cost
grows with g. The optimum is therefore below sqrt(N); `remat_group=0` picks
round(sqrt(N)) as a starting point and Task 9 tunes it against measured memory.

Prefix / suffix split (mirrors e2e's BlockCollectionSplit). Blocks below the
first fast block cannot depend on any fast weight, so their forward is run once
over the whole sequence and needs only a FIRST-order backward: the meta-gradient
reaches them as dL/dP . dP/dtheta, a plain vector-Jacobian product. They may
therefore use fused attention kernels. Blocks at or above the first fast block
carry the second-order path and must use the math SDPA backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ttt.config import Config
from ttt.optim.inner import InnerOptimizer


@dataclass
class SequenceOutput:
    """Result of running TTT over one sequence."""

    loss: Tensor  # scalar, differentiable: mean over chunks of loss-before-update
    per_chunk_loss: Tensor  # [N], detached
    token_nll: Tensor  # [T], detached
    fast_final: dict[str, Tensor]  # W_N, differentiable (used by the forgetting probe)


def resolve_remat_group(num_chunks: int, remat_group: int) -> int:
    """Group size g for checkpointing through time. 0 -> round(sqrt(N)), clamped to [1, N]."""
    g = round(sqrt(num_chunks)) if remat_group == 0 else remat_group
    g = max(1, min(g, num_chunks))
    assert num_chunks % g == 0 or g == num_chunks, (
        f"remat_group {g} must divide num_chunks {num_chunks} (or equal it); "
        f"ragged final groups would change the recompute pattern silently"
    )
    return g


class TTTInnerLoop:
    """Runs the chunked inner loop for a single sequence.

    micro_batch must be 1: the fast weights are per-sequence state, so batching
    would require a leading batch dimension on every fast tensor (a vmap). We
    serialise sequences and accumulate outer gradients instead.
    """

    def __init__(self, model, cfg: Config, inner_opt: InnerOptimizer) -> None:
        self.model = model
        self.cfg = cfg
        self.inner_opt = inner_opt
        self.num_chunks = cfg.num_chunks
        self.chunk_size = cfg.model.chunk_size
        self.group = resolve_remat_group(self.num_chunks, cfg.train.remat_group)

    # ---------------------------------------------------------------- helpers

    def _fast_keys(self, fast: dict[str, Tensor]) -> list[str]:
        """Deterministic ordering so flatten/unflatten round-trips exactly."""
        return sorted(fast)

    def _decay_toward_init(self, fast: dict[str, Tensor], fast0: dict[str, Tensor]) -> dict[str, Tensor]:
        """W <- W_0 + (1 - lambda) (W - W_0): pull the fast weights back toward their
        pretrained values before each step, bounding drift over a long sequence."""
        lam = self.cfg.inner.delta_decay
        if lam == 0.0:
            return fast
        return {k: fast0[k] + (1.0 - lam) * (fast[k] - fast0[k]) for k in fast}

    # ------------------------------------------------------------- inner step

    def _chunk_step(
        self,
        fast: dict[str, Tensor],
        fast0: dict[str, Tensor],
        opt_state: dict,
        caches: list,
        prefix_chunk: Tensor,
        targets: Tensor,
        loss_mask: Tensor,
        chunk_index: int,
        lr_scale: Tensor | float,
        lr_mult: dict[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict, list, Tensor, Tensor]:
        """One TTT step. Returns (W_i, opt_state, caches, loss_before_update, token_nll)."""
        fast = self._decay_toward_init(fast, fast0)

        logits, caches = self.model.suffix_forward(
            prefix_chunk, fast=fast, caches=caches, chunk_index=chunk_index
        )
        loss, token_nll = masked_cross_entropy(logits, targets, loss_mask)

        keys = self._fast_keys(fast)
        grads = torch.autograd.grad(
            loss, [fast[k] for k in keys], create_graph=True, allow_unused=False
        )
        grad_dict = dict(zip(keys, grads, strict=True))

        if not opt_state and self.inner_opt.needs_first_grad:
            # AdamW warm start: m_0 = g_1, v_0 = g_1^2 removes the step-1 singularity.
            opt_state = self.inner_opt.init_state(fast, first_grad=grad_dict)

        fast, opt_state = self.inner_opt.step(
            fast, grad_dict, opt_state, lr_scale=lr_scale, lr_mult=lr_mult
        )
        return fast, opt_state, caches, loss, token_nll

    # -------------------------------------------------------------- sequence

    def run_sequence(
        self,
        input_ids: Tensor,  # [1, T]
        targets: Tensor,  # [1, T]
        loss_mask: Tensor,  # [1, T]
        fast0: dict[str, Tensor],
        *,
        lr_scale: Tensor | float = 1.0,
        lr_mult: dict[str, Tensor] | None = None,
    ) -> SequenceOutput:
        b, t = input_ids.shape
        assert b == 1, f"inner loop requires micro_batch=1 (fast weights are per-sequence), got {b}"
        assert t == self.cfg.train.seq_len, f"expected seq_len {self.cfg.train.seq_len}, got {t}"
        assert t % self.chunk_size == 0, f"seq_len {t} not divisible by chunk_size {self.chunk_size}"

        # Prefix: frozen-block forward over the whole sequence, computed once.
        # Gradient to slow prefix params is first order only (see module docstring).
        prefix_out = self.model.prefix_forward(input_ids)  # [1, T, d]

        fast = dict(fast0)
        opt_state: dict = {} if self.inner_opt.needs_first_grad else self.inner_opt.init_state(fast)
        caches = self.model.init_caches(batch=1, device=input_ids.device, dtype=prefix_out.dtype)

        keys = self._fast_keys(fast)
        n_cache = len(caches)
        chunk_losses: list[Tensor] = []
        token_nlls: list[Tensor] = []

        def run_group(start: int, *flat: Tensor):
            """Checkpointed region covering `self.group` consecutive chunks.

            Flat layout: [fast..., cache_k/v interleaved...]. Losses are returned as
            tensors so they stay in the graph across the checkpoint boundary.
            """
            f = dict(zip(keys, flat[: len(keys)], strict=True))
            c = self.model.unflatten_caches(flat[len(keys) :])
            st = opt_state_ref[0]
            losses, nlls = [], []
            for j in range(self.group):
                idx = start + j
                sl = slice(idx * self.chunk_size, (idx + 1) * self.chunk_size)
                f, st, c, loss_i, nll_i = self._chunk_step(
                    f, fast0, st, c,
                    prefix_out[:, sl], targets[:, sl], loss_mask[:, sl],
                    idx, lr_scale, lr_mult,
                )
                losses.append(loss_i)
                nlls.append(nll_i)
            opt_state_ref[0] = st
            return (*[f[k] for k in keys], *self.model.flatten_caches(c), torch.stack(losses), torch.cat(nlls, dim=-1))

        opt_state_ref = [opt_state]
        flat = (*[fast[k] for k in keys], *self.model.flatten_caches(caches))

        for start in range(0, self.num_chunks, self.group):
            out = checkpoint(run_group, start, *flat, use_reentrant=False)
            flat = out[: len(keys) + n_cache * 2]
            chunk_losses.append(out[-2])
            token_nlls.append(out[-1])

        fast_final = dict(zip(keys, flat[: len(keys)], strict=True))
        per_chunk = torch.cat(chunk_losses)
        assert per_chunk.numel() == self.num_chunks, (
            f"expected {self.num_chunks} chunk losses, got {per_chunk.numel()}"
        )
        return SequenceOutput(
            loss=per_chunk.mean(),
            per_chunk_loss=per_chunk.detach(),
            token_nll=torch.cat(token_nlls, dim=-1).detach().reshape(-1),
            fast_final=fast_final,
        )


def masked_cross_entropy(logits: Tensor, targets: Tensor, loss_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Mean CE over unmasked positions, plus the per-token NLL.

    loss = sum_t mask_t * nll_t / max(sum_t mask_t, 1)
    Logits are cast to float32 before the softmax: bf16 log-softmax loses ~2 decimal
    digits, which matters at the 0.001-nat resolution the paper reports.
    """
    logits = logits.float()
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    mask = loss_mask.to(nll.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (nll * mask).sum() / denom, nll.detach()
