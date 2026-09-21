"""Paired analysis of an on/off evaluation: is a between-condition difference real?

Both conditions are evaluated on the SAME sequences, so the unit of analysis is the
per-sequence difference d_i = loss_off_i - loss_on_i, which removes the large
sequence-to-sequence variation in absolute loss. For n differences:

    mean = (1/n) sum d_i        sd = sqrt( sum (d_i - mean)^2 / (n - 1) )
    se   = sd / sqrt(n)         t  = mean / se          (n - 1 degrees of freedom)
    CI95 = mean +/- t_{0.975, n-1} * se

Independence. Evaluation sequences are fixed-length windows of a token stream, so several
can fall inside one long document (a PG-19 book spans many 32K windows). Those are not
independent draws, and treating them as such makes the interval too narrow. The
clustered version averages d_i within each document first and runs the same test across
documents, which is conservative: the effective sample size becomes the document count.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import stats


def sequence_documents(tokens: np.ndarray, bos_id: int, seq_len: int, seq_indices: list[int]) -> list[int]:
    """Document id of each evaluation sequence (the document covering most of it).

    Documents are delimited by BOS: document k occupies [bos_pos[k], bos_pos[k + 1]).
    Sequence i covers the half-open token range [i * L, (i + 1) * L), the positions it is
    scored on. A sequence that straddles a boundary goes to the document it overlaps most.
    """
    assert seq_len >= 1
    bos_pos = np.flatnonzero(np.asarray(tokens) == bos_id)
    assert bos_pos.size > 0 and bos_pos[0] == 0, "token stream must start with BOS"
    # One extra edge closes the last document at the end of the stream.
    edges = np.append(bos_pos, len(tokens))

    out: list[int] = []
    for i in seq_indices:
        start, end = i * seq_len, (i + 1) * seq_len
        assert 0 <= start and end <= len(tokens), f"sequence {i} = [{start}, {end}) outside the stream"
        # side="right": a position exactly on a BOS belongs to the document it opens.
        first = int(np.searchsorted(bos_pos, start, side="right")) - 1
        last = int(np.searchsorted(bos_pos, end - 1, side="right")) - 1
        overlap = {
            d: min(end, int(edges[d + 1])) - max(start, int(edges[d]))
            for d in range(first, last + 1)
        }
        assert sum(overlap.values()) == seq_len, f"overlaps {overlap} do not tile sequence {i}"
        out.append(max(overlap, key=lambda d: (overlap[d], -d)))  # ties -> earlier document
    return out


def paired_stats(diffs: list[float]) -> dict:
    """Student-t summary of paired differences (see module docstring for the formulas)."""
    n = len(diffs)
    assert n >= 2, f"need at least 2 paired differences, got {n}"
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        # se == 0 means every difference is identical: t is infinite for a nonzero mean
        # and undefined (0/0) when they are all zero.
        "t": mean / se if se > 0 else (math.copysign(math.inf, mean) if mean != 0 else math.nan),
        "ci95": (mean - half, mean + half),
        "positive": sum(1 for x in diffs if x > 0),
    }


def clustered_paired_stats(diffs: list[float], clusters: list[int]) -> dict:
    """The same test with one observation per cluster: the mean difference within it."""
    return paired_stats(cluster_means(diffs, clusters))


def cluster_means(values: list[float], clusters: list[int]) -> list[float]:
    """Mean of `values` within each cluster, clusters in sorted order."""
    assert len(values) == len(clusters), f"{len(values)} values for {len(clusters)} cluster labels"
    by_cluster: dict[int, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters, strict=True):
        by_cluster[c].append(v)
    return [sum(v) / len(v) for _, v in sorted(by_cluster.items())]


def ols_fit(columns: dict[str, list[float]], y: list[float]) -> dict:
    """Ordinary least squares with an intercept:  y_i = b_0 + sum_j b_j x_ij + e_i.

        b = (X'X)^-1 X'y     s^2 = RSS / (n - p)     se(b_j) = sqrt(s^2 [(X'X)^-1]_jj)
        CI95 = b_j +/- t_{0.975, n-p} se(b_j)        (n observations, p = 1 + number of columns)

    Classical (homoskedastic) standard errors; pass one observation per independent unit
    (cluster means), as everywhere else in this module. Used to ask which per-document
    quantity a TTT gain follows: what out-of-window context is worth, or how much the window
    damages the model (scripts/ttt_vs_context_value.py).
    """
    names = ["intercept", *columns]
    n, p = len(y), len(names)
    assert all(len(c) == n for c in columns.values()), "every column must have one value per observation"
    assert n > p, f"need more observations ({n}) than coefficients ({p})"
    X = np.column_stack([np.ones(n), *[np.asarray(c, dtype=np.float64) for c in columns.values()]])  # [n, p]
    yv = np.asarray(y, dtype=np.float64)  # [n]
    assert np.linalg.matrix_rank(X) == p, "collinear columns: the coefficients are not identified"
    xtx_inv = np.linalg.inv(X.T @ X)  # [p, p]
    beta = xtx_inv @ X.T @ yv  # [p]
    resid = yv - X @ beta
    s2 = float(resid @ resid) / (n - p)
    se = np.sqrt(s2 * np.diag(xtx_inv))
    half = float(stats.t.ppf(0.975, n - p)) * se
    tss = float(((yv - yv.mean()) ** 2).sum())
    return {"n": n, "r2": 1.0 - float(resid @ resid) / tss if tss > 0 else math.nan,
            "coef": {k: {"b": float(b), "se": float(e), "ci95": (float(b - h), float(b + h))}
                     for k, b, e, h in zip(names, beta, se, half, strict=True)}}


def select_probe_position(docs_in_eval_order: list[int], n_eval: int) -> int:
    """Position (in evaluation order) of the sequence to use as the forgetting probe.

    The probe must be text the model did not adapt to, so it has to come from a document
    that NONE of the `n_eval` evaluated sequences touches: the first such position at or
    after `n_eval`. A probe sharing a book with an evaluated sequence would measure
    adaptation to that book, not forgetting.
    """
    assert 0 < n_eval, f"n_eval must be positive, got {n_eval}"
    seen = set(docs_in_eval_order[:n_eval])
    for pos in range(n_eval, len(docs_in_eval_order)):
        if docs_in_eval_order[pos] not in seen:
            return pos
    raise AssertionError(
        f"no held-out sequence from an unseen document after the first {n_eval}: the "
        f"validation split has too few documents for an uncontaminated forgetting probe"
    )
