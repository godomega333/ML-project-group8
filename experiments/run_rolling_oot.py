#!/usr/bin/env python3
"""Run expanding-window chronological out-of-time validation."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_model.configs import config_hash, get_model_config
from fraud_model.data import load_train_data
from fraud_model.experiment import ensure_output_dir, evaluate_predictions, write_json
from fraud_model.manifest import build_run_manifest, write_manifest

if __package__:
    from .run_oot import (
        fit_model_on_frames,
        recorded_command,
        reviewed_model_configs,
        runner_identity_slug,
        runner_manifest_config,
        selected_model_names,
    )
else:
    from run_oot import (
        fit_model_on_frames,
        recorded_command,
        reviewed_model_configs,
        runner_identity_slug,
        runner_manifest_config,
        selected_model_names,
    )


ROLLING_COLUMNS = [
    "fold",
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "model",
    "roc_auc",
    "average_precision",
    "brier",
    "max_f1",
    "best_threshold",
    "train_seconds",
]

FOLD_SPECS = [
    (1, 0.60, 0.70),
    (2, 0.70, 0.80),
    (3, 0.80, 1.00),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IEEE-CIS rolling chronological OOT validation.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Kaggle competition CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for rolling OOT artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Maximum train rows to load for debug runs.")
    parser.add_argument("--model", choices=["lr", "boosting", "both"], default="both", help="Model path to run.")
    parser.add_argument("--config-id", default=None, help="Canonical config id from fraud_model.configs.")
    parser.add_argument(
        "--feature-profile",
        choices=["baseline", "uid_d", "uid_agg"],
        default="baseline",
        help="Feature engineering profile to use.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    model_config = get_model_config(args.config_id) if args.config_id else None
    if model_config is not None:
        args.model = model_config.model_family
        args.feature_profile = model_config.feature_profile
    output_dir = ensure_output_dir(args.output_dir)
    total_start = time.perf_counter()
    command = recorded_command("experiments/run_rolling_oot.py", raw_argv)

    df, y = load_train_data(args.data_dir, nrows=args.sample_rows)
    sorted_df, sorted_y = sort_chronologically(df, y)
    model_names = selected_model_names(args.model)

    rows: list[dict[str, Any]] = []
    for fold_number, train_fraction, valid_end_fraction in FOLD_SPECS:
        train_df, valid_df, y_train, y_valid = rolling_fold(
            sorted_df,
            sorted_y,
            train_fraction=train_fraction,
            valid_end_fraction=valid_end_fraction,
        )
        prior = float(np.mean(y_train))
        for model_name in model_names:
            result = fit_model_on_frames(
                model_name=model_name,
                train_df=train_df,
                y_train=y_train,
                valid_df=valid_df,
                y_valid=y_valid,
                seed=args.seed,
                feature_profile=args.feature_profile,
                model_config=model_config,
            )
            metrics = evaluate_predictions(y_valid, result.y_score, prior=prior)
            max_f1_policy = metrics["thresholds"]["max_f1"]
            rows.append(
                {
                    "fold": fold_number,
                    "train_start": _range_start(train_df),
                    "train_end": _range_end(train_df),
                    "valid_start": _range_start(valid_df),
                    "valid_end": _range_end(valid_df),
                    "model": model_name,
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "brier": metrics["brier_score"],
                    "max_f1": max_f1_policy["f1"],
                    "best_threshold": max_f1_policy["threshold"],
                    "train_seconds": result.train_seconds,
                }
            )

    metrics = pd.DataFrame(rows, columns=ROLLING_COLUMNS)
    metrics.to_csv(output_dir / "rolling_metrics.csv", index=False)
    write_json(
        output_dir / "config.json",
        {
            "command": command,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "sample_rows": args.sample_rows,
            "model": args.model,
            "feature_profile": args.feature_profile,
            "config_id": model_config.config_id if model_config else None,
            "config_hash": config_hash(model_config) if model_config else None,
            "folds": [
                {"fold": fold, "train_fraction": train, "valid_end_fraction": valid_end}
                for fold, train, valid_end in FOLD_SPECS
            ],
            "model_configs": reviewed_model_configs(),
        },
    )
    write_json(
        output_dir / "runtime.json",
        {
            "total_seconds": time.perf_counter() - total_start,
            "rows_loaded": int(len(sorted_df)),
            "models": model_names,
        },
    )
    write_manifest(
        output_dir / "manifest.json",
        build_run_manifest(
            candidate_id=f"rolling_oot_{runner_identity_slug(args.model, args.feature_profile, model_config)}",
            config=runner_manifest_config(args.model, args.feature_profile, model_config),
            split_policy="rolling_oot",
            source_run_id=f"rolling_oot_{runner_identity_slug(args.model, args.feature_profile, model_config)}",
            command=command,
            output_dir=output_dir,
            artifact_role="local_validation",
            sample_rows=args.sample_rows,
            train_seconds=float(metrics["train_seconds"].sum()) if not metrics.empty else 0.0,
            extra={
                "rows_loaded": int(len(sorted_df)),
                "folds": [
                    {"fold": fold, "train_fraction": train, "valid_end_fraction": valid_end}
                    for fold, train, valid_end in FOLD_SPECS
                ],
                "models": model_names,
            },
        ),
    )

    print(f"Rolling OOT rows: {len(metrics)}")
    print(f"Output dir: {output_dir}")
    return 0


def sort_chronologically(df: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if "TransactionDT" not in df.columns:
        raise ValueError("df must contain TransactionDT for rolling OOT")
    target = np.asarray(y).reshape(-1)
    if len(df) != target.shape[0]:
        raise ValueError("y must have the same number of rows as df")
    transaction_dt = pd.to_numeric(df["TransactionDT"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(transaction_dt)):
        raise ValueError("TransactionDT must contain finite values")
    order = np.argsort(transaction_dt, kind="mergesort")
    return df.iloc[order].reset_index(drop=True), target[order].copy()


def rolling_fold(
    df: pd.DataFrame,
    y: np.ndarray,
    train_fraction: float,
    valid_end_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    n_rows = len(df)
    train_end = int(np.floor(n_rows * train_fraction))
    valid_end = n_rows if valid_end_fraction >= 1.0 else int(np.floor(n_rows * valid_end_fraction))
    if train_end <= 0 or valid_end <= train_end or valid_end > n_rows:
        raise ValueError("rolling fold fractions produce an empty train or validation split")
    train_df = df.iloc[:train_end].reset_index(drop=True)
    valid_df = df.iloc[train_end:valid_end].reset_index(drop=True)
    return train_df, valid_df, y[:train_end].copy(), y[train_end:valid_end].copy()


def _range_start(df: pd.DataFrame) -> int:
    return int(pd.to_numeric(df["TransactionDT"], errors="raise").iloc[0])


def _range_end(df: pd.DataFrame) -> int:
    return int(pd.to_numeric(df["TransactionDT"], errors="raise").iloc[-1])


if __name__ == "__main__":
    raise SystemExit(main())
