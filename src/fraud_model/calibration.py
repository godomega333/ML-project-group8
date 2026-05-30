"""From-scratch calibration and threshold policy helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraud_model.metrics import threshold_summary


_THRESHOLD_COLUMNS = ["threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]


@dataclass
class PlattCalibrator:
    learning_rate: float = 0.05
    epochs: int = 500

    a_: float = field(init=False, default=0.0)
    b_: float = field(init=False, default=0.0)
    score_mean_: float = field(init=False, default=0.0)
    score_std_: float = field(init=False, default=1.0)
    _is_fit: bool = field(init=False, default=False)

    def fit(self, logits_or_scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        scores = self._prepare_scores(logits_or_scores)
        target = self._prepare_target(y, scores.shape[0])
        epochs = self._validate_hyperparameters()
        learning_rate = float(self.learning_rate)

        self.score_mean_, self.score_std_ = self._score_location_scale(scores)
        standardized_scores = self._standardize_scores(scores)
        self.a_ = 0.0
        self.b_ = self._initial_bias(target)

        for _ in range(epochs):
            proba = self._sigmoid(self.a_ * standardized_scores + self.b_)
            error = proba - target
            grad_a = float(np.mean(error * standardized_scores))
            grad_b = float(np.mean(error))
            self.a_ -= learning_rate * grad_a
            self.b_ -= learning_rate * grad_b

        self._is_fit = True
        return self

    def predict_proba(self, logits_or_scores: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise ValueError("calibrator must be fit before prediction")
        scores = self._prepare_scores(logits_or_scores)
        standardized_scores = self._standardize_scores(scores)
        return self._sigmoid(self.a_ * standardized_scores + self.b_)

    def _validate_hyperparameters(self) -> int:
        try:
            learning_rate = float(self.learning_rate)
        except (TypeError, ValueError):
            raise ValueError("learning_rate must be finite and positive") from None
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")

        try:
            epochs = float(self.epochs)
        except (TypeError, ValueError):
            raise ValueError("epochs must be a non-negative integer") from None
        if not np.isfinite(epochs) or epochs < 0.0 or epochs != np.floor(epochs):
            raise ValueError("epochs must be a non-negative integer")
        return int(epochs)

    @staticmethod
    def _initial_bias(y: np.ndarray) -> float:
        positive_rate = (float(np.sum(y)) + 0.5) / (float(y.shape[0]) + 1.0)
        return float(np.log(positive_rate / (1.0 - positive_rate)))

    @staticmethod
    def _prepare_scores(logits_or_scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(logits_or_scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] == 0:
            raise ValueError("logits_or_scores must contain at least one value")
        if not np.all(np.isfinite(scores)):
            raise ValueError("logits_or_scores must contain finite values")
        return scores

    @staticmethod
    def _prepare_target(y: np.ndarray, n_rows: int) -> np.ndarray:
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        if target.shape[0] != n_rows:
            raise ValueError("y must have the same length as logits_or_scores")
        if not np.all(np.isfinite(target)):
            raise ValueError("y must contain finite values")
        if not np.all((target == 0.0) | (target == 1.0)):
            raise ValueError("y must contain only 0/1 labels")
        return target

    @staticmethod
    def _score_location_scale(scores: np.ndarray) -> tuple[float, float]:
        score_mean = float(np.mean(scores))
        if not np.isfinite(score_mean):
            score_mean = float(np.median(scores))

        score_std = float(np.std(scores))
        if not np.isfinite(score_std) or score_std <= 0.0:
            score_std = 1.0
        return score_mean, score_std

    def _standardize_scores(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.score_mean_) / self.score_std_

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))


def choose_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    prior: float,
    recall_target: float = 0.80,
) -> dict[str, dict[str, float | int]]:
    if not np.isfinite(prior) or not 0.0 <= prior <= 1.0:
        raise ValueError("prior must be finite and within [0, 1]")
    if not np.isfinite(recall_target) or not 0.0 <= recall_target <= 1.0:
        raise ValueError("recall_target must be finite and within [0, 1]")

    summary = threshold_summary(y_true, y_score, thresholds=_threshold_grid(y_score, float(prior)))
    return {
        "prior": _row_to_policy(_closest_threshold_row(summary, float(prior))),
        "max_f1": _row_to_policy(_best_row(summary, primary="f1", eligible=np.ones(len(summary), dtype=bool))),
        "recall_target": _row_to_policy(_recall_target_row(summary, float(recall_target))),
    }


def _threshold_grid(y_score: np.ndarray, prior: float) -> np.ndarray:
    score_thresholds = np.asarray(y_score, dtype=np.float64).reshape(-1)
    grid = np.concatenate(
        (
            np.linspace(0.0, 1.0, 101, dtype=np.float64),
            np.array([prior], dtype=np.float64),
            score_thresholds,
        )
    )
    return np.unique(grid)


def _closest_threshold_row(summary: pd.DataFrame, threshold: float) -> pd.Series:
    distances = np.abs(summary["threshold"].to_numpy(dtype=np.float64) - threshold)
    return summary.iloc[int(np.argmin(distances))]


def _recall_target_row(summary: pd.DataFrame, recall_target: float) -> pd.Series:
    recall = summary["recall"].to_numpy(dtype=np.float64)
    eligible = recall >= recall_target
    if not np.any(eligible):
        eligible = recall == np.max(recall)
    return _best_row(summary, primary="f1", eligible=eligible)


def _best_row(summary: pd.DataFrame, primary: str, eligible: np.ndarray) -> pd.Series:
    table = summary.loc[eligible, _THRESHOLD_COLUMNS].copy()
    table = table.sort_values(
        by=[primary, "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return table.iloc[0]


def _row_to_policy(row: pd.Series) -> dict[str, float | int]:
    return {
        "threshold": float(row["threshold"]),
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "tp": int(row["tp"]),
        "fp": int(row["fp"]),
        "tn": int(row["tn"]),
        "fn": int(row["fn"]),
    }
