"""Held-out evaluation: mean loss, the per-token-index NLL curve, and the forgetting probe.

Evaluation is the same computation as a training step minus the outer backward:
TTT runs over each held-out sequence exactly as it does at train time, the fast
weights are reset to W_0 at every sequence boundary, and the reported loss is the
paper's Eq. 6 objective (mean over chunks of the loss BEFORE the update).

``token_nll`` is the paper's Fig. 6 curve: NLL averaged across sequences at each
position index 0..T-1. It is what shows TTT working -- loss should fall along the
sequence as the fast weights adapt, beyond what a sliding-window model gets from
context alone.

    THE ONE WAY TO BREAK THIS
    -------------------------
    Do not run the inner loop under ``torch.no_grad()``. ``TTTInnerLoop.run_sequence``
    calls ``torch.autograd.grad(..., create_graph=True)`` on every chunk loss --
    that IS the method, not an artifact of training -- and under ``no_grad`` the
    chunk loss has no grad_fn, so the call raises. The decorator below is kept
    because everything AROUND the loop (probe scoring, accumulation) genuinely
    wants no_grad; the loop itself is re-enabled with ``torch.enable_grad()`` and
    every tensor taken out of it is detached immediately, so no graph survives
    past one sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from ttt.eval.forgetting import lr_multipliers, probe_delta_nll
from ttt.train.inner_loop import TTTInnerLoop

__all__ = ["EvalResult", "evaluate"]

_BATCH_KEYS = ("input_ids", "targets", "loss_mask")


@dataclass
class EvalResult:
    loss: float  # mean over sequences of the mean-over-chunks loss (Eq. 6)
    token_nll: np.ndarray  # [seq_len] mean per-token-index NLL over sequences (Fig. 6 curve)
    num_sequences: int
    forgetting_delta_nll: float | None  # None unless a probe batch was supplied


@torch.no_grad()
def evaluate(
    loop: TTTInnerLoop,
    split,
    dataloader: Iterable[dict],
    *,
    max_sequences: int | None = None,
    probe_batch: dict | None = None,
    device=None,
) -> EvalResult:
    """Run TTT over the held-out sequences and aggregate.

    ``split`` is the ``ParamSplit`` of the model behind ``loop``; ``dict(split.fast)``
    is passed fresh for every sequence, which is what resets the fast weights to
    W_0 at each sequence boundary (train and eval alike, per plan §0.2 "Reset").

    Deterministic: no sampling, no dropout, and the per-sequence order of the
    dataloader is the only thing that fixes the result. Sequences are processed
    one at a time because fast weights are per-sequence state -- a batched
    dataloader is unrolled along dim 0.

    See the module docstring for why the inner loop is run under
    ``torch.enable_grad()`` inside a ``no_grad`` function.
    """
    seq_len = loop.cfg.train.seq_len
    lr_mult = lr_multipliers(loop)
    probe = None if probe_batch is None else _move(probe_batch, device)

    loss_sum: Tensor | None = None
    nll_sum: Tensor | None = None
    forget_sum: Tensor | None = None
    count = 0

    for batch in dataloader:
        ids, targets, loss_mask = _prepare(batch, device)
        for i in range(ids.shape[0]):
            if max_sequences is not None and count >= max_sequences:
                break

            with torch.enable_grad():
                out = loop.run_sequence(
                    ids[i : i + 1], targets[i : i + 1], loss_mask[i : i + 1],
                    dict(split.fast), lr_scale=1.0, lr_mult=lr_mult,
                )
                # Detach everything we keep: the graph dies with `out` at the end
                # of this iteration, so peak memory is one sequence, not the split.
                seq_loss = out.loss.detach()
                seq_nll = out.token_nll.detach()
                fast_final = {k: v.detach() for k, v in out.fast_final.items()}
            del out

            assert seq_nll.shape == (seq_len,), (
                f"token_nll must be [seq_len] = [{seq_len}], got {tuple(seq_nll.shape)}"
            )
            # Accumulate as tensors; .item() is paid once, after the loop.
            loss_sum = seq_loss if loss_sum is None else loss_sum + seq_loss
            nll_sum = seq_nll if nll_sum is None else nll_sum + seq_nll
            if probe_batch is not None:
                delta = probe_delta_nll(loop, split, fast_final, probe)
                forget_sum = delta if forget_sum is None else forget_sum + delta
            count += 1
        if max_sequences is not None and count >= max_sequences:
            break

    assert count > 0, "evaluate() saw no sequences: the dataloader was empty"
    assert loss_sum is not None and nll_sum is not None

    token_nll = (nll_sum / count).double().cpu().numpy()
    assert token_nll.shape == (seq_len,), f"aggregated token_nll has shape {token_nll.shape}"
    return EvalResult(
        loss=float(loss_sum / count),
        token_nll=token_nll,
        num_sequences=count,
        forgetting_delta_nll=None if forget_sum is None else float(forget_sum / count),
    )


def _prepare(batch: dict, device) -> tuple[Tensor, Tensor, Tensor]:
    """Validate one dataloader batch and move it to ``device``."""
    missing = [k for k in _BATCH_KEYS if k not in batch]
    assert not missing, f"batch is missing {missing}; expected keys {_BATCH_KEYS}"
    moved = _move(batch, device)
    ids, targets, loss_mask = (moved[k] for k in _BATCH_KEYS)
    assert ids.ndim == 2, f"input_ids must be [B, T], got {tuple(ids.shape)}"
    assert targets.shape == ids.shape and loss_mask.shape == ids.shape, (
        f"targets {tuple(targets.shape)} and loss_mask {tuple(loss_mask.shape)} "
        f"must match input_ids {tuple(ids.shape)}"
    )
    return ids, targets, loss_mask


def _move(batch: dict, device) -> dict:
    if device is None:
        return batch
    return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
