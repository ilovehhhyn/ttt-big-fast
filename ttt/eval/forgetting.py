"""The forgetting probe: how much did test-time training cost us elsewhere?

Protocol (plan 2026-09-17 §0.2, "Forgetting probe"):

    1. Run TTT over one sequence to obtain W_T (the fast weights after the last
       chunk). The slow parameters are untouched.
    2. Score a FRESH held-out chunk -- unrelated text the model never adapted to --
       under W_T and under W_0, with NO inner step in either pass.
    3. Report  mean NLL(W_T) - mean NLL(W_0).

    Positive  => the model got worse on unrelated text: it forgot.
    Negative  => adaptation generalised off-distribution.
    Zero      => the fast weights did not move (the lr = 0 control).

The sign is the experimental question, not an invariant, so nothing in this
module assumes one.

Two details that decide whether the number means anything:

* The probe is scored with a plain chunked forward, threading the same rolling
  KV cache the inner loop uses but never calling the inner optimizer. Taking an
  inner step on the probe would measure "can TTT fit the probe", which is a
  different question and would make the probe non-reusable.
* The mean is a token-weighted mean over the whole probe (sum of masked NLL over
  sum of mask), not a mean of per-chunk means. With a ragged loss mask those two
  differ, and the token-weighted one is the quantity the paper's per-token curve
  integrates to.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ttt.train.inner_loop import TTTInnerLoop, masked_cross_entropy

__all__ = ["forgetting_delta_nll", "lr_multipliers", "probe_delta_nll", "score_with_weights"]


def _one_sequence(batch: dict, key: str) -> Tensor:
    x = batch[key]
    assert x.ndim == 2 and x.shape[0] == 1, (
        f"the forgetting probe scores one sequence at a time; {key} has shape {tuple(x.shape)}"
    )
    return x


@torch.no_grad()
def score_with_weights(loop: TTTInnerLoop, batch: dict, fast: dict[str, Tensor]) -> Tensor:
    """Token-weighted mean NLL of ``batch`` under fast weights ``fast``. No inner step.

    Returns a 0-dim tensor so callers can accumulate without an ``.item()`` per
    chunk. Runs under ``no_grad``: unlike the inner loop, a scoring pass takes no
    gradient of anything, so there is no graph to keep.
    """
    model, cfg = loop.model, loop.cfg
    input_ids = _one_sequence(batch, "input_ids")
    targets = _one_sequence(batch, "targets")
    loss_mask = _one_sequence(batch, "loss_mask")
    chunk = cfg.model.chunk_size
    t = input_ids.shape[1]
    assert t % chunk == 0, f"probe length {t} is not divisible by chunk_size {chunk}"
    assert targets.shape == input_ids.shape and loss_mask.shape == input_ids.shape

    prefix_out = model.prefix_forward(input_ids)
    caches = model.init_caches(batch=1, device=input_ids.device, dtype=prefix_out.dtype)

    nll_sum: Tensor | None = None
    denom: Tensor | None = None
    for i in range(t // chunk):
        sl = slice(i * chunk, (i + 1) * chunk)
        logits, caches = model.suffix_forward(
            prefix_out[:, sl], fast=fast, caches=caches, chunk_index=i
        )
        _, token_nll = masked_cross_entropy(logits, targets[:, sl], loss_mask[:, sl])
        mask = loss_mask[:, sl].to(token_nll.dtype)
        chunk_sum = (token_nll * mask).sum()
        chunk_denom = mask.sum()
        nll_sum = chunk_sum if nll_sum is None else nll_sum + chunk_sum
        denom = chunk_denom if denom is None else denom + chunk_denom

    assert nll_sum is not None and denom is not None
    return nll_sum / denom.clamp_min(1.0)


def probe_delta_nll(loop: TTTInnerLoop, split, fast_final: dict[str, Tensor], probe_batch: dict) -> Tensor:
    """mean NLL(W_T) - mean NLL(W_0) on ``probe_batch``, as a 0-dim tensor.

    Takes W_T directly so a caller that has already run the inner loop (the
    evaluator) does not pay for a second pass. ``fast_final`` is detached here:
    the probe is a measurement, never a place the meta-gradient flows through.
    """
    w_t = {k: v.detach() for k, v in fast_final.items()}
    w_0 = {k: v.detach() for k, v in split.fast.items()}
    assert set(w_t) == set(w_0), (
        f"W_T and W_0 must have the same keys; "
        f"missing={sorted(set(w_0) - set(w_t))} extra={sorted(set(w_t) - set(w_0))}"
    )
    return score_with_weights(loop, probe_batch, w_t) - score_with_weights(loop, probe_batch, w_0)


def forgetting_delta_nll(loop: TTTInnerLoop, split, seq_batch: dict, probe_batch: dict) -> float:
    """Run TTT on ``seq_batch``, then score ``probe_batch`` under W_T and W_0.

    Returns mean NLL(W_T) - mean NLL(W_0). Positive means the model got worse on
    the unrelated probe, i.e. it forgot.

    ``torch.enable_grad()`` around ``run_sequence`` is NOT optional: the inner
    loop calls ``torch.autograd.grad`` on every chunk loss, so it raises under
    ``no_grad``. The outputs are detached immediately afterwards.
    """
    lr_mult = lr_multipliers(loop)
    with torch.enable_grad():
        out = loop.run_sequence(
            _one_sequence(seq_batch, "input_ids"),
            _one_sequence(seq_batch, "targets"),
            _one_sequence(seq_batch, "loss_mask"),
            dict(split.fast),
            lr_scale=1.0,
            lr_mult=lr_mult,
        )
        fast_final = {k: v.detach() for k, v in out.fast_final.items()}
    del out
    return float(probe_delta_nll(loop, split, fast_final, probe_batch))


def lr_multipliers(loop: TTTInnerLoop) -> dict[str, Tensor] | None:
    """exp(inner_lr_log) when the learned inner LR is on, else None.

    Evaluation must use the SAME inner LR the run was meta-trained with: dropping
    the learned multiplier here would silently evaluate a different optimizer
    than the one that was trained.
    """
    if not loop.cfg.inner.learned_lr:
        return None
    return {k: v.detach() for k, v in loop.model.inner_lr_multipliers().items()}
