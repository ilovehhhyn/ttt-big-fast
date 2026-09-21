"""The recall probe (ttt/eval/recall.py) must measure information carried from the planted
passage to its repeat, and nothing else.

The sharpest check is the floor. With L layers of window k, a prediction at index t depends on
tokens >= t - L*(k-1). So without test-time training:

    gap one token too short  ->  the first scored prediction still sees the last planted token:
                                 PRESENT and ABSENT differ (the probe can see information flow)
    gap at the boundary      ->  nothing planted is visible: the two agree EXACTLY

and with test-time training on, the same out-of-reach gap gives a non-zero difference, which
can only have travelled through the fast weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, RopeConfig, TrainConfig
from ttt.eval.recall import RecallSpec, choose_pairs, plant, score_pair
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import build_inner_optimizer
from ttt.train.inner_loop import TTTInnerLoop

LAYERS, WINDOW, CHUNK, SEQ_LEN, VOCAB = 3, 8, 4, 64, 32
A, N, CUE = 4, 8, 2  # planted at [4, 12); both chunk-aligned
# Out of reach iff  b + cue - 1 - L*(k-1) > a + n - 1  <=>  b > 31  <=>  gap = b - (a+n) >= 20.
GAP_OUT_OF_REACH = 20


def build(inner: InnerConfig):
    """Tiny float64 CPU model; mirrors tests/test_eval.py::build."""
    mcfg = ModelConfig(vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_layers=LAYERS,
                       num_heads=4, num_kv_heads=2, window_size=WINDOW, chunk_size=CHUNK, fast_blocks=1,
                       rope=RopeConfig(theta=10000.0, scaling="none"), lora=LoRAConfig(rank=2, alpha=4.0))
    cfg = Config(model=mcfg, inner=inner,
                 train=TrainConfig(seq_len=SEQ_LEN, tokens_per_step=SEQ_LEN, micro_batch=1, dtype="fp32"))
    torch.manual_seed(0)
    model = TTTTransformer(mcfg, max_seq_len=SEQ_LEN).double()
    split = split_parameters(model, mcfg, cfg.train)
    return split, TTTInnerLoop(model, cfg, build_inner_optimizer(inner))


def carrier_and_passage(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(1, VOCAB, (SEQ_LEN + 1,), generator=g),  # [T+1]
            torch.randint(1, VOCAB, (N,), generator=g))  # [n]


def max_abs_difference(inner: InnerConfig, gap: int) -> float:
    split, loop = build(inner)
    window, passage = carrier_and_passage()
    scored = score_pair(loop, split, None, window, passage, RecallSpec(A, gap, N, CUE))
    assert scored["present"].shape == scored["absent"].shape == (N - CUE,)
    return float(np.abs(scored["absent"] - scored["present"]).max())


# ---------------------------------------------------------------- geometry


def test_plant_changes_exactly_the_planted_span():
    window, passage = carrier_and_passage()
    spec = RecallSpec(A, GAP_OUT_OF_REACH, N, CUE)
    b = spec.target_start
    present, absent = (plant(window, passage, spec, present=flag) for flag in (True, False))

    for batch in (present, absent):
        assert batch["input_ids"].shape == batch["targets"].shape == batch["loss_mask"].shape == (1, SEQ_LEN)
        assert torch.equal(batch["input_ids"][0, b : b + N], passage)  # the repeat is in BOTH conditions
        assert torch.equal(batch["input_ids"][0, 1:], batch["targets"][0, :-1])  # targets = inputs shifted by one
    assert torch.equal(present["input_ids"][0, A : A + N], passage)
    assert torch.equal(absent["input_ids"][0, A : A + N], window[A : A + N])  # carrier's own text stays
    differs = (present["input_ids"] != absent["input_ids"])[0]
    assert not differs[:A].any() and not differs[A + N :].any()  # the ONLY difference is the planted span
    assert torch.equal(window, carrier_and_passage()[0])  # the carrier itself is not modified in place


def test_scored_targets_are_the_passage_after_the_cue():
    window, passage = carrier_and_passage()
    spec = RecallSpec(A, GAP_OUT_OF_REACH, N, CUE)
    batch = plant(window, passage, spec, present=True)
    assert torch.equal(batch["targets"][0, spec.scored_targets()], passage[CUE:])


@pytest.mark.parametrize("spec, message", [
    (RecallSpec(A, SEQ_LEN, N, CUE), "does not fit"),
    (RecallSpec(A, 8, N, N), "cue < length"),
    (RecallSpec(0, 8, N, CUE), "source_start must be >= 1"),
    (RecallSpec(A, -1, N, CUE), "gap must be >= 0"),
])
def test_spec_rejects_bad_geometry(spec: RecallSpec, message: str):
    with pytest.raises(AssertionError, match=message):
        spec.check(SEQ_LEN)


def test_receptive_field_boundary_formula():
    assert not RecallSpec(A, GAP_OUT_OF_REACH - 1, N, CUE).beyond_receptive_field(LAYERS, WINDOW)
    assert RecallSpec(A, GAP_OUT_OF_REACH, N, CUE).beyond_receptive_field(LAYERS, WINDOW)


# ---------------------------------------------------------------- floor, and what TTT adds


def test_without_ttt_the_floor_is_exact_and_the_boundary_is_tight():
    no_ttt = InnerConfig(optimizer="none", lr_warmup_frac=0.0)
    assert max_abs_difference(no_ttt, GAP_OUT_OF_REACH - 1) > 0.0  # one token inside reach: visible
    assert max_abs_difference(no_ttt, GAP_OUT_OF_REACH) == 0.0  # out of reach: exactly nothing


def test_ttt_carries_the_passage_beyond_attentions_reach():
    on = InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False)
    off = InnerConfig(optimizer="normalized_sgd", lr_rms=0.0, learned_lr=False)
    assert max_abs_difference(on, GAP_OUT_OF_REACH) > 0.0  # only the fast weights can have carried it
    assert max_abs_difference(off, GAP_OUT_OF_REACH) == 0.0  # same rule with a zero step: nothing


# ---------------------------------------------------------------- choosing pairs


def stream(doc_lengths: list[int], bos: int = 0) -> np.ndarray:
    """Documents of the given lengths, each opened by BOS and filled with its own id + 1."""
    return np.concatenate([np.r_[bos, np.full(n - 1, d + 1)] for d, n in enumerate(doc_lengths)])


def test_choose_pairs_skips_boundaries_and_takes_donors_from_other_documents():
    T, spec = 16, RecallSpec(source_start=2, gap=4, length=4, cue=1)  # plant [2, 6), repeat [10, 14)
    # BOS at stream positions 0, 40, 52. Sequences of 16 tokens:
    #   0: doc 0    1: doc 0    2: boundary at 8 (after the passage region)
    #   3: boundary at 4 (INSIDE the passage region [2, 6))    4: doc 2
    tokens = stream([40, 12, 29])
    assert len(tokens) == 81  # 5 sequences of 16 tokens need 81
    pairs, skipped = choose_pairs(tokens, 0, T, [0, 1, 2, 3, 4], spec, count=3)

    assert skipped == 2 and [q.carrier for q in pairs] == [0, 1, 4]  # 2 and 3 hold a boundary in [2, 14]
    # Carrier 0 walks 1 (same document), 2 (its passage is still document 0), 3 (BOS inside the
    # passage) and takes 4. Carrier 4 wraps around to sequence 0.
    assert [q.donor for q in pairs] == [4, 4, 0]
    assert [(q.carrier_doc, q.donor_doc) for q in pairs] == [(0, 2), (0, 2), (2, 0)]


def test_choose_pairs_refuses_when_too_few_carriers_are_usable():
    T, spec = 16, RecallSpec(source_start=2, gap=4, length=4, cue=1)
    with pytest.raises(AssertionError, match="usable carriers"):
        choose_pairs(stream([40, 12, 29]), 0, T, [0, 1, 2, 3, 4], spec, count=4)
