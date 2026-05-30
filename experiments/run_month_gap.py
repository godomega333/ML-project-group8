#!/usr/bin/env python3
"""Run gap-based chronological validation for IEEE-CIS fraud experiments."""

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


METRIC_COLUMNS = [
    "run",
    "feature_profile",
    "model",
    "rows_train",
    "rows_valid",
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "roc_auc",
    "average_precision",
    "brier",
    "max_f1",
    "best_threshold",
    "train_seconds",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IEEE-CIS month-gap chronological validation.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Kaggle competition CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for month-gap artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Maximum train rows to load for debug runs.")
    parser.add_argument("--model", choices=["lr", "boosting", "both"], default="both", help="Model path to run.")
    parser.add_argument("--config-id", default=None, help="Canonical config id from fraud_model.configs.")
    parser.add_argument("--feature-profile", choices=["baseline", "uid_d", "uid_agg"], default="baseline")
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--gap-fraction", type=float, default=0.10)
    parser.add_argument("--valid-fraction", type=float, default=0.30)
    parser.add_argument("--run-name", default="month-gap")
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
    start = time.perf_counter()
    command = recorded_command("experiments/run_month_gap.py", raw_argv)

    df, y = load_train_data(args.data_dir, nrows=args.sample_rows)
    train_df, valid_df, y_train, y_valid = month_gap_split(
        df,
        y,
        train_fraction=args.train_fraction,
        gap_fraction=args.gap_fraction,
        valid_fraction=args.valid_fraction,
    )
    prior = float(np.mean(y_train))
    rows: list[dict[str, Any]] = []
    for model_name in selected_model_names(args.model):
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
                "run": args.run_name,
                "feature_profile": args.feature_profile,
                "model": model_name,
                "rows_train": int(len(train_df)),
                "rows_valid": int(len(valid_df)),
                "train_start": _range_start(train_df),
                "train_end": _range_end(train_df),
                "valid_start": _range_start(valid_df),
                "valid_end": _range_end(valid_df),
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "brier": metrics["brier_score"],
                "max_f1": max_f1_policy["f1"],
                "best_threshold": max_f1_policy["threshold"],
                "train_seconds": result.train_seconds,
            }
        )

    pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(output_dir / "month_gap_metrics.csv", index=False)
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
            "train_fraction": args.train_fraction,
            "gap_fraction": args.gap_fraction,
            "valid_fraction": args.valid_fraction,
            "model_configs": reviewed_model_configs(),
        },
    )
    write_json(
        output_dir / "runtime.json",
        {
            "total_seconds": time.perf_counter() - start,
            "rows_loaded": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
        },
    )
    write_manifest(
        output_dir / "manifest.json",
        build_run_manifest(
            candidate_id=f"{args.run_name}_{runner_identity_slug(args.model, args.feature_profile, model_config)}",
            config=runner_manifest_config(args.model, args.feature_profile, model_config),
            split_policy="month_gap",
            source_run_id=f"{args.run_name}_{runner_identity_slug(args.model, args.feature_profile, model_config)}_month_gap",
            command=command,
            output_dir=output_dir,
            artifact_role="local_validation",
            sample_rows=args.sample_rows,
            train_seconds=sum(float(row["train_seconds"]) for row in rows),
            extra={
                "rows_loaded": int(len(df)),
                "rows_train": int(len(train_df)),
                "rows_valid": int(len(valid_df)),
                "train_fraction": args.train_fraction,
                "gap_fraction": args.gap_fraction,
                "valid_fraction": args.valid_fraction,
                "models": selected_model_names(args.model),
            },
        ),
    )
    print(f"Month-gap rows: {len(rows)}")
    print(f"Output dir: {output_dir}")
    return 0


def month_gap_split(
    df: pd.DataFrame,
    y: np.ndarray,
    train_fraction: float,
    gap_fraction: float,
    valid_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if "TransactionDT" not in df.columns:
        raise ValueError("df must contain TransactionDT for month-gap validation")
    fractions = [float(train_fraction), float(gap_fraction), float(valid_fraction)]
    if any((not np.isfinite(value) or value < 0.0) for value in fractions):
        raise ValueError("fractions must be finite and non-negative")
    if train_fraction <= 0.0 or valid_fraction <= 0.0 or sum(fractions) > 1.0 + 1e-12:
        raise ValueError("train_fraction and valid_fraction must be positive and fractions must sum to <= 1")
    target = np.asarray(y).reshape(-1)
    if target.shape[0] != len(df):
        raise ValueError("y must have the same number of rows as df")
    transaction_dt = pd.to_numeric(df["TransactionDT"], errors="raise").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(transaction_dt)):
        raise ValueError("TransactionDT must contain finite values")
    order = np.argsort(transaction_dt, kind="mergesort")
    sorted_df = df.iloc[order].reset_index(drop=True)
    sorted_y = target[order].copy()
    n_rows = len(sorted_df)
    train_end = int(np.floor(n_rows * train_fraction))
    valid_rows = max(1, int(np.ceil(n_rows * valid_fraction)))
    gap_rows = int(np.floor(n_rows * gap_fraction))
    valid_start = train_end + gap_rows
    valid_end = min(n_rows, valid_start + valid_rows)
    if train_end <= 0 or valid_start >= n_rows or valid_end <= valid_start:
        raise ValueError("fractions produce an empty train or validation split")
    return (
        sorted_df.iloc[:train_end].reset_index(drop=True),
        sorted_df.iloc[valid_start:valid_end].reset_index(drop=True),
        sorted_y[:train_end].copy(),
        sorted_y[valid_start:valid_end].copy(),
    )


def _range_start(df: pd.DataFrame) -> int:
    return int(pd.to_numeric(df["TransactionDT"], errors="raise").iloc[0])


def _range_end(df: pd.DataFrame) -> int:
    return int(pd.to_numeric(df["TransactionDT"], errors="raise").iloc[-1])


if __name__ == "__main__":
    raise SystemExit(main())
