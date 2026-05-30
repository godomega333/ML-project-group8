"""Experiment orchestration helpers for fraud modeling demos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_model.calibration import choose_thresholds
from fraud_model.manifest import ensure_new_output_dir
from fraud_model.metrics import average_precision, brier_score, roc_auc


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    seed: int = 42
    sample_rows: int | None = None
    valid_fraction: float = 0.2
    model: str = "both"
    run_name: str = "demo"

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload


def chronological_split(
    df: pd.DataFrame,
    y: np.ndarray,
    valid_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if "TransactionDT" not in df.columns:
        raise ValueError("df must contain TransactionDT for chronological splitting")
    if not np.isfinite(valid_fraction) or not 0.0 < float(valid_fraction) < 1.0:
        raise ValueError("valid_fraction must be finite and within (0, 1)")

    target = np.asarray(y).reshape(-1)
    if target.shape[0] != len(df):
        raise ValueError("y must have the same number of rows as df")
    if len(df) < 2:
        raise ValueError("chronological_split requires at least two rows")

    transaction_dt = pd.to_numeric(df["TransactionDT"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(transaction_dt)):
        raise ValueError("TransactionDT must contain finite values")

    order = np.argsort(transaction_dt, kind="mergesort")
    sorted_df = df.iloc[order].reset_index(drop=True)
    sorted_y = target[order]

    valid_rows = int(np.ceil(len(sorted_df) * float(valid_fraction)))
    valid_rows = min(max(valid_rows, 1), len(sorted_df) - 1)
    split_at = len(sorted_df) - valid_rows
    train_df = sorted_df.iloc[:split_at].reset_index(drop=True)
    valid_df = sorted_df.iloc[split_at:].reset_index(drop=True)
    return train_df, valid_df, sorted_y[:split_at].copy(), sorted_y[split_at:].copy()


def ensure_output_dir(path: str | Path, *, allow_existing: bool = False) -> Path:
    output_dir = Path(path)
    if allow_existing:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    return ensure_new_output_dir(output_dir)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_metrics_csv(path: str | Path, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False)


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, prior: float) -> dict[str, Any]:
    return {
        "roc_auc": roc_auc(y_true, y_score),
        "average_precision": average_precision(y_true, y_score),
        "brier_score": brier_score(y_true, y_score),
        "thresholds": choose_thresholds(y_true, y_score, prior=prior),
    }


def write_presenter_notes(path: str | Path, run_summary: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = str(run_summary.get("command", ""))
    model = str(run_summary.get("model", ""))
    metrics = run_summary.get("metrics", {})
    interpretation = str(run_summary.get("interpretation", ""))

    lines = [
        "# Presenter Notes",
        "",
        "## Command",
        "",
        f"`{command}`" if command else "Command not recorded.",
        "",
        "## Model",
        "",
        f"Requested model setting: `{model}`.",
        "",
        "## Metrics",
        "",
    ]

    if isinstance(metrics, dict) and metrics:
        for model_name, model_metrics in metrics.items():
            if not isinstance(model_metrics, dict):
                continue
            lines.extend(
                [
                    f"- `{model_name}`: ROC-AUC {_format_metric(model_metrics.get('roc_auc'))}, "
                    f"average precision {_format_metric(model_metrics.get('average_precision'))}, "
                    f"Brier score {_format_metric(model_metrics.get('brier_score'))}.",
                ]
            )
    else:
        lines.append("- No metrics were recorded.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation
            or "Use ROC-AUC for ranking quality, average precision for rare-fraud retrieval, and Brier score for calibration error.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _format_metric(value: Any) -> str:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(metric):
        return "n/a"
    return f"{metric:.4f}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
