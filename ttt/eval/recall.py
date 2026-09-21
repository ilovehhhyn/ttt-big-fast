"""Recall probe: what does the model retrieve from BEYOND its attention window?

Language-model loss cannot separate memory from repair: test-time training lowers the loss of
a windowed model both by carrying information forward and by repairing the damage the window
does to an un-tuned model (docs/results/results.md, "The sliding window breaks the un-tuned
model"). This probe measures memory alone. A passage P of n tokens from ANOTHER document is
written into a real validation sequence ("carrier") and scored where it is repeated:

    PRESENT   carrier with P at [a, a+n)  and again at [b, b+n)
    ABSENT    carrier with P only at [b, b+n)            (the carrier's own text stays at a)

    recall = NLL_absent(P at b) - NLL_present(P at b)        nats per token, positive = recalled

The two inputs are identical except for tokens [a, a+n), so everything that is not information
about P carried from a to b cancels: the damage done by the window, its repair, the abrupt
topic change at b. Reference points:

    floor     a windowed model without test-time training cannot see P at all once the gap
              g = b - (a+n) exceeds its receptive field, L * (k-1) tokens for L layers of
              window k (each layer looks back k-1 positions). Then recall == 0 EXACTLY, which
              doubles as a check of the probe itself. Inside the receptive field it is measured.
    ceiling   full attention (window = T) on the same pairs.

The first `cue` tokens of the repeat are not scored: they are what tells the model that P has
started again. Overwriting (never inserting) keeps every sequence at T tokens with all
positions aligned between the two conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ttt.train.inner_loop import TTTInnerLoop

__all__ = ["RecallSpec", "RecallPair", "plant", "choose_pairs", "score_pair"]


@dataclass(frozen=True)
class RecallSpec:
    """Where the passage is planted and where its repeat is scored (token positions)."""

    source_start: int  # a: P overwrites tokens [a, a+n) in the PRESENT condition
    gap: int  # g: tokens between the end of the planted passage and the start of its repeat
    length: int  # n: passage length
    cue: int  # first `cue` tokens of the repeat are context, not scored

    @property
    def target_start(self) -> int:
        """b = a + n + g."""
        return self.source_start + self.length + self.gap

    def scored_targets(self) -> slice:
        """Indices into `targets` (and token_nll) that predict the scored tokens P[cue:].

        targets[t] is the token at position t+1, so token P[j] at position b+j is predicted at
        index b+j-1. Scored: j in [cue, n)  ->  t in [b+cue-1, b+n-1).
        """
        b = self.target_start
        return slice(b + self.cue - 1, b + self.length - 1)

    def check(self, seq_len: int) -> None:
        a, g, n, c = self.source_start, self.gap, self.length, self.cue
        assert a >= 1, f"source_start must be >= 1 (position 0 may be a BOS), got {a}"
        assert n >= 2 and 1 <= c < n, f"need length >= 2 and 1 <= cue < length, got length={n} cue={c}"
        assert g >= 0, f"gap must be >= 0, got {g}"
        assert self.target_start + n <= seq_len, (
            f"the repeat [{self.target_start}, {self.target_start + n}) does not fit in {seq_len} tokens: "
            f"reduce --source-start, --gap or --length"
        )

    def beyond_receptive_field(self, num_layers: int, window: int) -> bool:
        """True iff no scored prediction can see a planted token through attention.

        Layer by layer a query reaches back window-1 positions, so the prediction at index t
        depends on tokens >= t - L*(k-1). The earliest scored index is b+cue-1 and the last
        planted token sits at a+n-1:  unreachable  iff  b + cue - 1 - L*(k-1) > a + n - 1.
        """
        reach = num_layers * (window - 1)
        return self.target_start + self.cue - 1 - reach > self.source_start + self.length - 1


@dataclass(frozen=True)
class RecallPair:
    carrier: int  # dataset index of the sequence that is modified
    donor: int  # dataset index of the sequence the passage is taken from
    carrier_doc: int
    donor_doc: int


def plant(window: Tensor, passage: Tensor, spec: RecallSpec, *, present: bool) -> dict[str, Tensor]:
    """Build one probe input from a carrier's raw token window.

    window:  [T+1] the carrier's tokens (input_ids followed by the final target)
    passage: [n]   the donor passage P
    Returns input_ids/targets/loss_mask, each [1, T], consistent with each other: targets are
    the inputs shifted by one AFTER planting, so the repeat is scored against P itself.
    """
    assert window.ndim == 1 and passage.shape == (spec.length,), (tuple(window.shape), tuple(passage.shape))
    seq_len = window.shape[0] - 1
    spec.check(seq_len)
    a, b, n = spec.source_start, spec.target_start, spec.length
    w = window.clone()
    w[b : b + n] = passage
    if present:
        w[a : a + n] = passage
    ids, tgt = w[:-1].unsqueeze(0), w[1:].unsqueeze(0)
    return {"input_ids": ids, "targets": tgt, "loss_mask": torch.ones_like(tgt)}


def choose_pairs(tokens: np.ndarray, bos_token_id: int, seq_len: int, order: list[int],
                 spec: RecallSpec, count: int) -> tuple[list[RecallPair], int]:
    """Pick `count` (carrier, donor) pairs by walking the evaluation order. Deterministic.

    Documents are delimited by BOS; the document of a token region is the one that holds its
    first token (regions with a BOS inside are never used, so that is the whole region).

    A carrier is usable iff no document boundary lies in [a, b+n]: a boundary between the
    planted passage and its repeat would tell any model that the earlier text is over. The
    donor is the next sequence of the order whose passage region [a, a+n) holds no BOS and
    belongs to a DIFFERENT document, so P cannot be predicted from the carrier itself.
    Returns the pairs and the number of carriers skipped for containing a boundary.
    """
    spec.check(seq_len)
    a, b, n = spec.source_start, spec.target_start, spec.length
    assert len(order) >= 2, "need at least two sequences: a carrier and a donor"
    bos_pos = np.flatnonzero(np.asarray(tokens) == bos_token_id)
    assert bos_pos.size > 0 and bos_pos[0] == 0, "token stream must start with BOS"

    def has_bos(idx: int, lo: int, hi: int) -> bool:  # any BOS at stream positions idx*T + [lo, hi)?
        left, right = np.searchsorted(bos_pos, [idx * seq_len + lo, idx * seq_len + hi], side="left")
        return bool(right > left)

    def doc_at(idx: int, pos: int) -> int:  # side="right": a position on a BOS belongs to the document it opens
        return int(np.searchsorted(bos_pos, idx * seq_len + pos, side="right")) - 1

    pairs, skipped = [], 0
    for i, carrier in enumerate(order):
        if len(pairs) == count:
            break
        if has_bos(carrier, a, b + n + 1):
            skipped += 1
            continue
        carrier_doc = doc_at(carrier, a)
        later_then_earlier = order[i + 1 :] + order[:i]
        donor = next((d for d in later_then_earlier
                      if not has_bos(d, a, a + n) and doc_at(d, a) != carrier_doc), None)
        assert donor is not None, f"no donor passage from another document for carrier {carrier}"
        pairs.append(RecallPair(carrier=carrier, donor=donor, carrier_doc=carrier_doc, donor_doc=doc_at(donor, a)))
    assert len(pairs) == count, (
        f"only {len(pairs)} usable carriers among {len(order)} sequences ({skipped} hold a document "
        f"boundary inside [{a}, {b + n}]); asked for {count}"
    )
    return pairs, skipped


def score_pair(loop: TTTInnerLoop, split, lr_mult, window: Tensor, passage: Tensor, spec: RecallSpec,
               *, device=None) -> dict[str, np.ndarray]:
    """NLL [n - cue] of the scored tokens of the repeat, with P present and absent earlier.

    Runs the SAME path as evaluation (ttt/eval/evaluator.py): fast weights start from W_0 for
    each input, inference mode, learned inner-LR multipliers applied.
    """
    scored = spec.scored_targets()
    out = {}
    for name, present in (("present", True), ("absent", False)):
        batch = plant(window, passage, spec, present=present)
        ids, tgt, mask = (batch[k] if device is None else batch[k].to(device) for k in ("input_ids", "targets", "loss_mask"))
        with torch.enable_grad():  # the inner loop differentiates the chunk loss (see evaluator.py)
            res = loop.run_sequence(ids, tgt, mask, dict(split.fast),
                                    lr_scale=1.0, lr_mult=lr_mult, inference=True)
            nll = res.token_nll.detach()
        del res
        assert nll.shape == (ids.shape[1],), tuple(nll.shape)
        out[name] = nll[scored].double().cpu().numpy()
    return out
