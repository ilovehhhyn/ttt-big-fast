"""Outer loop: meta-learning the slow parameters through the TTT inner loop.

One outer step processes `tokens_per_step / seq_len` sequences. Because the fast
weights are per-sequence state, sequences are run one at a time and their
gradients accumulated (e2e uses either a vmap over the batch or `accum_steps`;
we always take the second route, which keeps fast-weight memory independent of
batch size).

    for each of S sequences:
        L_s = (1/N) sum_i l_i(W_{i-1})       # inner loop, second-order
        backward(L_s / S)                    # accumulate into the slow params
    clip_grad_norm_(slow, 1.0)
    AdamW.step()

Only the slow parameters are optimized. The fast weights are reset to W_0 at
every sequence, and W_0 itself is never written by the inner loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ttt.config import Config
from ttt.model.naming import ParamSplit
from ttt.optim.outer import inner_lr_scale_at_step, lr_at_step, set_lr
from ttt.train.inner_loop import TTTInnerLoop


@dataclass
class StepMetrics:
    step: int
    loss: float
    grad_norm: float
    lr: float
    inner_lr_scale: float
    seconds: float
    extra: dict = field(default_factory=dict)

    def as_log(self) -> dict:
        d = {"step": self.step, "loss": self.loss, "grad_norm": self.grad_norm,
             "lr": self.lr, "inner_lr_scale": self.inner_lr_scale, "sec_per_step": self.seconds}
        d.update(self.extra)
        return d


class Trainer:
    def __init__(self, cfg: Config, model, split: ParamSplit, loop: TTTInnerLoop,
                 optimizer: torch.optim.Optimizer, train_iter, *, device: torch.device,
                 empty_cache: bool = False) -> None:
        self.cfg = cfg
        self.model = model
        self.split = split
        self.loop = loop
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.device = device
        self.empty_cache = empty_cache and device.type == "cuda"
        self.slow_params: list[Tensor] = [v for _, v in sorted(split.slow.items())]
        self.seqs_per_step = cfg.train.seqs_per_step
        assert self.seqs_per_step >= 1

    def _lr_mult(self) -> dict[str, Tensor] | None:
        """Learned per-tensor inner LR, or None when disabled."""
        if not self.cfg.inner.learned_lr:
            return None
        return self.model.inner_lr_multipliers()

    def train_step(self, step: int) -> StepMetrics:
        t0 = time.perf_counter()
        lr = lr_at_step(step, self.cfg.outer)
        set_lr(self.optimizer, lr)
        scale = inner_lr_scale_at_step(step, self.cfg.inner, self.cfg.outer.total_steps)

        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(self.seqs_per_step):
            batch = next(self.train_iter)
            ids = batch["input_ids"].to(self.device)
            tgt = batch["targets"].to(self.device)
            mask = batch["loss_mask"].to(self.device)
            out = self.loop.run_sequence(ids, tgt, mask, dict(self.split.fast),
                                         lr_scale=scale, lr_mult=self._lr_mult())
            # Divide before backward so the accumulated gradient is the mean over
            # sequences, matching a single large batch.
            (out.loss / self.seqs_per_step).backward()
            total += out.loss.detach().item()
            if self.empty_cache:
                # Each sequence's second-order graph is released by backward(), but the
                # caching allocator keeps those blocks. Returning them to the driver
                # between sequences trades a little speed for headroom, which is what
                # decides whether a long-context step fits at all.
                del out
                torch.cuda.empty_cache()

        gnorm = torch.nn.utils.clip_grad_norm_(self.slow_params, self.cfg.outer.grad_clip)
        self.optimizer.step()

        extra: dict = {}
        if self.cfg.inner.learned_lr:
            # exp(inner_lr_log) multiplies the inner step per fast tensor. It starts at
            # exactly 1.0. Whether the outer loop drives it toward 0 (learning to switch
            # TTT off) or keeps it near 1 (learning to use TTT) is the single most
            # diagnostic quantity in this experiment, so it is logged every step.
            with torch.no_grad():
                mult = torch.stack([m.detach().reshape(()) for m in self.model.inner_lr_multipliers().values()])
            extra = {"inner_lr_mult_mean": float(mult.mean()),
                     "inner_lr_mult_min": float(mult.min()),
                     "inner_lr_mult_max": float(mult.max())}
        return StepMetrics(step=step, loss=total / self.seqs_per_step, grad_norm=float(gnorm),
                           lr=lr, inner_lr_scale=scale, seconds=time.perf_counter() - t0, extra=extra)
