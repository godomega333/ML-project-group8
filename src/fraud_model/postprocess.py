"""Compliant prediction post-processing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def smooth_by_group(
    submission: pd.DataFrame,
    keys: pd.DataFrame,
    group_column: str,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shrink each submission probability toward its group mean."""

    if not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and within [0, 1]")
    _require_columns(submission, ["TransactionID", "isFraud"], "submission")
    _require_columns(keys, ["TransactionID", group_column], "keys")

    try:
        merged = submission.merge(
            keys[["TransactionID", group_column]],
            on="TransactionID",
            how="left",
            validate="one_to_one",
        )
    except pd.errors.MergeError as exc:
        raise ValueError(f"submission and keys must have a one-to-one TransactionID merge: {exc}") from exc

    if merged[group_column].isna().any():
        missing = int(merged[group_column].isna().sum())
        raise ValueError(f"keys are missing group values for {missing} submission rows")

    scores = pd.to_numeric(merged["isFraud"], errors="coerce").astype("float64")
    if scores.isna().any() or not ((scores >= 0.0) & (scores <= 1.0)).all():
        raise ValueError("submission probabilities must be numeric and within [0, 1]")

    group_mean = scores.groupby(merged[group_column]).transform("mean")
    smoothed_scores = (1.0 - float(alpha)) * scores + float(alpha) * group_mean
    smoothed = submission[["TransactionID"]].copy()
    smoothed["isFraud"] = np.clip(smoothed_scores.to_numpy(dtype=np.float64), 0.0, 1.0)

    diagnostics = {
        "rows": int(len(smoothed)),
        "groups": int(merged[group_column].nunique(dropna=False)),
        "alpha": float(alpha),
        "group_column": group_column,
        "changed_rows": int(np.sum(np.abs(smoothed["isFraud"].to_numpy() - scores.to_numpy()) > 1e-15)),
        "before_min": float(scores.min()),
        "before_max": float(scores.max()),
        "after_min": float(smoothed["isFraud"].min()),
        "after_max": float(smoothed["isFraud"].max()),
    }
    return smoothed, diagnostics


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
