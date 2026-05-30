#!/usr/bin/env python3
"""Run a fast local modeling demo on IEEE-CIS fraud data."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_model.boosting import HistogramGradientBoostingClassifier
from fraud_model.data import load_train_data
from fraud_model.experiment import (
    ExperimentConfig,
    chronological_split,
    ensure_output_dir,
    evaluate_predictions,
    write_json,
    write_metrics_csv,
    write_presenter_notes,
)
from fraud_model.features import FeaturePipeline
from fraud_model.logistic import MatrixLogisticRegression
from fraud_model.metrics import calibration_table, threshold_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small deterministic IEEE-CIS fraud modeling demo.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Kaggle competition CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for demo artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization.")
    parser.add_argument("--sample-rows", type=int, default=20000, help="Maximum train rows to load.")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="Chronological validation fraction.")
    parser.add_argument("--model", choices=["lr", "boosting", "both"], default="both", help="Model path to run.")
    parser.add_argument("--run-name", default="demo", help="Run name stored in metrics artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args = parse_args(argv)
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        sample_rows=args.sample_rows,
        valid_fraction=args.valid_fraction,
        model=args.model,
        run_name=args.run_name,
    )

    output_dir = ensure_output_dir(config.output_dir)
    df, y = load_train_data(config.data_dir, nrows=config.sample_rows)
    train_df, valid_df, y_train, y_valid = chronological_split(df, y, config.valid_fraction)
    prior = float(np.mean(y_train))

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    model_names = ["lr", "boosting"] if config.model == "both" else [config.model]
    metrics_by_model: dict[str, dict[str, Any]] = {}
    threshold_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        y_score = _fit_and_predict(model_name, pipeline, train_df, valid_df, y_train, y_valid, config.seed)
        model_metrics = evaluate_predictions(y_valid, y_score, prior=prior)
        metrics_by_model[model_name] = model_metrics

        thresholds = threshold_summary(y_valid, y_score)
        thresholds.insert(0, "model", model_name)
        threshold_rows.extend(thresholds.to_dict(orient="records"))

        calibration = calibration_table(y_valid, y_score, bins=10)
        calibration.insert(0, "model", model_name)
        calibration_rows.extend(calibration.to_dict(orient="records"))

    metrics_payload = {
        "config": config.to_json_dict(),
        "rows": {
            "loaded": int(len(df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "train_positive_rate": prior,
            "valid_positive_rate": float(np.mean(y_valid)),
        },
        "metrics": metrics_by_model,
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    write_metrics_csv(output_dir / "threshold_summary.csv", threshold_rows)
    write_metrics_csv(output_dir / "calibration_table.csv", calibration_rows)
    write_presenter_notes(
        output_dir / "presenter_notes.md",
        {
            "command": " ".join(["python", "experiments/run_demo.py", *sys.argv[1:]]),
            "model": config.model,
            "metrics": metrics_by_model,
            "interpretation": (
                "This demo uses a chronological holdout, fit-on-train preprocessing, and small from-scratch "
                "model settings. Treat the numbers as a quick code-path check, not a final model comparison."
            ),
        },
    )

    print(_summary_text(config, metrics_by_model, output_dir))
    return 0


def _fit_and_predict(
    model_name: str,
    pipeline: FeaturePipeline,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
) -> np.ndarray:
    if model_name == "lr":
        train_x = pipeline.transform(train_df, model="lr")
        valid_x = pipeline.transform(valid_df, model="lr")
        model = MatrixLogisticRegression(
            learning_rate=0.05,
            l2=0.05,
            epochs=12,
            batch_size=2048,
            class_weight=_balanced_class_weight(y_train),
            seed=seed,
            tolerance=0.0,
        )
        model.fit(train_x, y_train, valid_x=valid_x, valid_y=y_valid)
        return model.predict_proba(valid_x)

    if model_name == "boosting":
        train_x = pipeline.transform(train_df, model="boosting")
        valid_x = pipeline.transform(valid_df, model="boosting")
        model = HistogramGradientBoostingClassifier(
            n_estimators=5,
            max_depth=2,
            learning_rate=0.10,
            n_bins=32,
            l2=1.0,
            min_child_weight=1.0,
            positive_weight=_positive_weight(y_train),
            seed=seed,
        )
        model.fit(train_x, y_train, x_valid=valid_x, y_valid=y_valid)
        return model.predict_proba(valid_x)

    raise ValueError("model_name must be 'lr' or 'boosting'")


def _balanced_class_weight(y: np.ndarray) -> dict[int, float] | None:
    target = np.asarray(y).reshape(-1)
    positives = float(np.sum(target == 1))
    negatives = float(np.sum(target == 0))
    if positives <= 0.0 or negatives <= 0.0:
        return None
    total = positives + negatives
    return {0: total / (2.0 * negatives), 1: total / (2.0 * positives)}


def _positive_weight(y: np.ndarray) -> float | None:
    target = np.asarray(y).reshape(-1)
    positives = float(np.sum(target == 1))
    negatives = float(np.sum(target == 0))
    if positives <= 0.0 or negatives <= 0.0:
        return None
    return negatives / positives


def _summary_text(config: ExperimentConfig, metrics_by_model: dict[str, dict[str, Any]], output_dir: Path) -> str:
    lines = [
        f"Demo run: {config.run_name}",
        f"Output dir: {output_dir}",
    ]
    for model_name, metrics in metrics_by_model.items():
        lines.append(
            f"{model_name}: ROC-AUC={float(metrics['roc_auc']):.4f} "
            f"AP={float(metrics['average_precision']):.4f} "
            f"Brier={float(metrics['brier_score']):.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
