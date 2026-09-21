"""Tests for the paired on/off analysis: document assignment and clustered statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ttt.eval.paired import cluster_means, clustered_paired_stats, ols_fit, paired_stats, sequence_documents

BOS = 99


def _stream(doc_lengths: list[int]) -> np.ndarray:
    """[BOS, d0 tokens..., BOS, d1 tokens..., ...]; doc k's tokens all equal k, so a wrong
    boundary shows up as the wrong value."""
    out: list[int] = []
    for k, n in enumerate(doc_lengths):
        out.append(BOS)
        out.extend([k] * n)
    return np.asarray(out, dtype=np.uint32)


def test_sequence_inside_one_document():
    # doc 0 occupies positions [0, 21): BOS + 20 tokens. seq_len 8 -> sequences 0 and 1
    # lie entirely inside it.
    toks = _stream([20, 20])
    assert sequence_documents(toks, BOS, 8, [0, 1]) == [0, 0]


def test_sequence_spanning_two_documents_goes_to_the_majority():
    # doc 0 = [0, 11), doc 1 = [11, 42). Sequence 1 covers [8, 16): 3 tokens of doc 0,
    # 5 tokens of doc 1 -> doc 1. Sequence 0 covers [0, 8) -> doc 0.
    toks = _stream([10, 30])
    assert sequence_documents(toks, BOS, 8, [0, 1, 2]) == [0, 1, 1]


def test_boundary_exactly_on_a_sequence_edge():
    # doc 0 = [0, 8) exactly (BOS + 7 tokens), doc 1 starts at 8. Sequence 0 = [0, 8) is
    # all doc 0; sequence 1 = [8, 16) is all doc 1. An off-by-one would mix them.
    toks = _stream([7, 20])
    assert sequence_documents(toks, BOS, 8, [0, 1]) == [0, 1]


def test_order_of_indices_is_preserved():
    toks = _stream([10, 30])
    assert sequence_documents(toks, BOS, 8, [2, 0, 1]) == [1, 0, 1]


def test_stream_must_start_with_bos():
    with pytest.raises(AssertionError, match="start with BOS"):
        sequence_documents(np.asarray([5, BOS, 1, 2], dtype=np.uint32), BOS, 2, [0])


def test_paired_stats_matches_a_hand_computation():
    d = [0.1, 0.3, 0.2, 0.4]                      # mean 0.25, sample sd = sqrt(0.05/3)
    s = paired_stats(d)
    sd = math.sqrt(sum((x - 0.25) ** 2 for x in d) / 3)
    assert s["n"] == 4 and s["mean"] == pytest.approx(0.25)
    assert s["sd"] == pytest.approx(sd) and s["se"] == pytest.approx(sd / 2)
    assert s["t"] == pytest.approx(0.25 / (sd / 2))
    # Student t, 3 degrees of freedom, two-sided 95%: 3.182446
    assert s["ci95"][0] == pytest.approx(0.25 - 3.182446 * sd / 2, rel=1e-6)
    assert s["ci95"][1] == pytest.approx(0.25 + 3.182446 * sd / 2, rel=1e-6)
    assert s["positive"] == 4


def test_clustering_collapses_sequences_from_one_document():
    """Four sequences but only two documents: the effective sample size is 2, and the
    clustered interval must be the wider, honest one."""
    diffs = [0.10, 0.12, 0.30, 0.32]
    docs = [7, 7, 9, 9]
    c = clustered_paired_stats(diffs, docs)
    assert c["n"] == 2
    assert c["mean"] == pytest.approx((0.11 + 0.31) / 2)
    naive = paired_stats(diffs)
    assert (c["ci95"][1] - c["ci95"][0]) > (naive["ci95"][1] - naive["ci95"][0])


def test_clustered_equals_plain_when_every_sequence_is_its_own_document():
    diffs = [0.1, 0.3, 0.2, 0.4]
    assert clustered_paired_stats(diffs, [0, 1, 2, 3]) == paired_stats(diffs)


# ---------------------------------------------------------------- forgetting-probe choice
def test_probe_comes_from_a_document_no_evaluated_sequence_touches():
    from ttt.eval.paired import select_probe_position

    #            evaluated (n_eval=3)   candidates ...
    docs = [4, 4, 7,                    7, 4, 9, 2]
    # position 3 shares book 7 and position 4 shares book 4 with evaluated sequences, so
    # the first genuinely unrelated candidate is position 5 (book 9).
    assert select_probe_position(docs, n_eval=3) == 5


def test_probe_selection_fails_loudly_when_every_candidate_is_contaminated():
    from ttt.eval.paired import select_probe_position

    with pytest.raises(AssertionError, match="no held-out sequence"):
        select_probe_position([1, 2, 1, 2, 2], n_eval=2)
    with pytest.raises(AssertionError, match="no held-out sequence"):
        select_probe_position([1, 2], n_eval=2)   # nothing left after the evaluated ones


# ---------------------------------------------------------------- regression across documents


def test_cluster_means_orders_clusters_and_averages_within():
    assert cluster_means([1.0, 3.0, 10.0, 5.0], [7, 7, 2, 9]) == [10.0, 2.0, 5.0]  # clusters 2, 7, 9


def test_ols_recovers_known_coefficients_exactly_without_noise():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=40), rng.normal(size=40)
    fit = ols_fit({"a": a.tolist(), "b": b.tolist()}, (0.5 + 2.0 * a - 3.0 * b).tolist())
    got = {k: v["b"] for k, v in fit["coef"].items()}
    assert got == pytest.approx({"intercept": 0.5, "a": 2.0, "b": -3.0}, abs=1e-10)
    assert fit["r2"] == pytest.approx(1.0) and fit["n"] == 40


def test_ols_standard_error_matches_the_textbook_simple_regression():
    # One regressor: se(b) = s / sqrt(sum (x - xbar)^2), s^2 = RSS / (n - 2).
    rng = np.random.default_rng(1)
    x = rng.normal(size=30)
    y = 1.0 + 0.7 * x + rng.normal(scale=0.3, size=30)
    fit = ols_fit({"x": x.tolist()}, y.tolist())
    b = fit["coef"]["x"]["b"]
    resid = y - (fit["coef"]["intercept"]["b"] + b * x)
    se = math.sqrt(float(resid @ resid) / 28 / float(((x - x.mean()) ** 2).sum()))
    assert fit["coef"]["x"]["se"] == pytest.approx(se, rel=1e-10)
    lo, hi = fit["coef"]["x"]["ci95"]
    assert lo < 0.7 < hi  # the interval covers the truth in this draw


def test_ols_refuses_collinear_columns_and_too_few_observations():
    x = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(AssertionError, match="collinear"):
        ols_fit({"x": x, "twice": [2 * v for v in x]}, [1.0, 2.0, 2.5, 4.0])
    with pytest.raises(AssertionError, match="more observations"):
        ols_fit({"x": [1.0, 2.0]}, [1.0, 2.0])
