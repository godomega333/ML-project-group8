"""From-scratch binary classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_inputs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1)
    score = np.asarray(y_score).reshape(-1)
    if y.shape[0] != score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_true must contain finite values")
    if not np.all(np.isfinite(score)):
        raise ValueError("y_score must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y_true must contain only 0/1 labels")
    return y.astype(np.float64, copy=False), score.astype(np.float64, copy=False)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1

        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end

    return ranks


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y, score = _prepare_inputs(y_true, y_score)
    positives = y == 1
    n_pos = int(np.sum(positives))
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(score)
    positive_rank_sum = float(np.sum(ranks[positives]))
    auc = (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y, score = _prepare_inputs(y_true, y_score)
    n_pos = int(np.sum(y == 1))
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order]
    sorted_score = score[order]

    tp = 0.0
    seen = 0.0
    ap = 0.0
    start = 0
    while start < sorted_y.shape[0]:
        end = start + 1
        while end < sorted_y.shape[0] and sorted_score[end] == sorted_score[start]:
            end += 1

        group_pos = float(np.sum(sorted_y[start:end] == 1))
        tp += group_pos
        seen += end - start
        if group_pos > 0:
            ap += (group_pos / n_pos) * (tp / seen)
        start = end

    return float(ap)


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, int]:
    y, score = _prepare_inputs(y_true, y_score)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    pred_pos = score >= float(threshold)
    actual_pos = y == 1
    tp = int(np.sum(pred_pos & actual_pos))
    fp = int(np.sum(pred_pos & ~actual_pos))
    tn = int(np.sum(~pred_pos & ~actual_pos))
    fn = int(np.sum(~pred_pos & actual_pos))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def binary_scores_from_confusion(cm: dict[str, int]) -> dict[str, float]:
    tp = int(cm["tp"])
    fp = int(cm["fp"])
    tn = int(cm["tn"])
    fn = int(cm["fn"])

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def threshold_summary(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    threshold_values = np.asarray(thresholds).reshape(-1).astype(np.float64, copy=False)
    if not np.all(np.isfinite(threshold_values)):
        raise ValueError("thresholds must contain finite values")

    rows = []
    for threshold in threshold_values:
        cm = confusion_at_threshold(y_true, y_score, float(threshold))
        scores = binary_scores_from_confusion(cm)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": scores["accuracy"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "tp": cm["tp"],
                "fp": cm["fp"],
                "tn": cm["tn"],
                "fn": cm["fn"],
            }
        )

    return pd.DataFrame(
        rows,
        columns=["threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"],
    )


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y, score = _prepare_inputs(y_true, y_score)
    return float(np.mean((score - y) ** 2))


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y, score = _prepare_inputs(y_true, y_score)
    if bins < 1:
        raise ValueError("bins must be positive")

    clipped = np.clip(score, 0.0, 1.0)
    bin_ids = np.minimum((clipped * bins).astype(np.int64), bins - 1)

    rows = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        count = int(np.sum(mask))
        if count:
            mean_pred = float(np.mean(clipped[mask]))
            frac_positive = float(np.mean(y[mask]))
        else:
            mean_pred = float("nan")
            frac_positive = float("nan")
        rows.append(
            {
                "bin": bin_id,
                "count": count,
                "mean_pred": mean_pred,
                "frac_positive": frac_positive,
            }
        )

    return pd.DataFrame(rows, columns=["bin", "count", "mean_pred", "frac_positive"])
