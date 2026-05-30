"""Bootstrap uncertainty summaries for paired model comparisons."""

from __future__ import annotations

import numpy as np

from fraud_model.metrics import average_precision, brier_score, roc_auc

DELTA_KEYS = [
    "auc_delta_mean",
    "auc_delta_low",
    "auc_delta_high",
    "ap_delta_mean",
    "ap_delta_low",
    "ap_delta_high",
    "brier_delta_mean",
    "brier_delta_low",
    "brier_delta_high",
]


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    anchor_score: np.ndarray,
    candidate_score: np.ndarray,
    repeats: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate paired label-stratified metric deltas for candidate minus anchor."""

    y, anchor, candidate = _prepare_inputs(y_true, anchor_score, candidate_score)
    repeat_count = int(repeats)
    if repeat_count < 1:
        raise ValueError("repeats must be positive")

    positive_idx = np.flatnonzero(y == 1)
    negative_idx = np.flatnonzero(y == 0)
    if positive_idx.shape[0] == 0 or negative_idx.shape[0] == 0:
        raise ValueError("y_true must contain both classes")

    rng = np.random.default_rng(seed)
    auc_deltas = np.empty(repeat_count, dtype=np.float64)
    ap_deltas = np.empty(repeat_count, dtype=np.float64)
    brier_deltas = np.empty(repeat_count, dtype=np.float64)

    for repeat_idx in range(repeat_count):
        sampled_idx = np.concatenate(
            [
                rng.choice(positive_idx, size=positive_idx.shape[0], replace=True),
                rng.choice(negative_idx, size=negative_idx.shape[0], replace=True),
            ]
        )
        sampled_y = y[sampled_idx]
        sampled_anchor = anchor[sampled_idx]
        sampled_candidate = candidate[sampled_idx]

        auc_deltas[repeat_idx] = roc_auc(sampled_y, sampled_candidate) - roc_auc(sampled_y, sampled_anchor)
        ap_deltas[repeat_idx] = average_precision(sampled_y, sampled_candidate) - average_precision(
            sampled_y,
            sampled_anchor,
        )
        brier_deltas[repeat_idx] = brier_score(sampled_y, sampled_candidate) - brier_score(sampled_y, sampled_anchor)

    return {
        "auc_delta_mean": float(np.mean(auc_deltas)),
        "auc_delta_low": float(np.quantile(auc_deltas, 0.025)),
        "auc_delta_high": float(np.quantile(auc_deltas, 0.975)),
        "ap_delta_mean": float(np.mean(ap_deltas)),
        "ap_delta_low": float(np.quantile(ap_deltas, 0.025)),
        "ap_delta_high": float(np.quantile(ap_deltas, 0.975)),
        "brier_delta_mean": float(np.mean(brier_deltas)),
        "brier_delta_low": float(np.quantile(brier_deltas, 0.025)),
        "brier_delta_high": float(np.quantile(brier_deltas, 0.975)),
    }


def _prepare_inputs(
    y_true: np.ndarray,
    anchor_score: np.ndarray,
    candidate_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1)
    anchor = np.asarray(anchor_score).reshape(-1)
    candidate = np.asarray(candidate_score).reshape(-1)
    if y.shape[0] != anchor.shape[0] or y.shape[0] != candidate.shape[0]:
        raise ValueError("y_true, anchor_score, and candidate_score must have the same length")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_true must contain finite values")
    if not np.all(np.isfinite(anchor)):
        raise ValueError("anchor_score must contain finite values")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("candidate_score must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y_true must contain only 0/1 labels")
    return (
        y.astype(np.float64, copy=False),
        anchor.astype(np.float64, copy=False),
        candidate.astype(np.float64, copy=False),
    )
