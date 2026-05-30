#!/usr/bin/env python3
"""Run feature-group ablations for chronological fraud validation."""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_model.configs import config_hash, get_model_config
from fraud_model.data import load_train_data
from fraud_model.experiment import chronological_split, ensure_output_dir, evaluate_predictions, write_json
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


ABLATION_COLUMNS = [
    "run",
    "enabled_groups",
    "model",
    "roc_auc",
    "average_precision",
    "max_f1",
    "train_seconds",
]

ABLATION_RUNS = [
    ("transaction_core", ["transaction_core"]),
    ("plus_counts_and_d", ["transaction_core", "counts_and_d"]),
    ("plus_m_and_v", ["transaction_core", "counts_and_d", "m_and_v"]),
    ("all_groups", ["transaction_core", "counts_and_d", "m_and_v", "identity"]),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IEEE-CIS feature-group ablations.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Kaggle competition CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for ablation artifacts.")
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
    command = recorded_command("experiments/run_ablation.py", raw_argv)

    df, y = load_train_data(args.data_dir, nrows=args.sample_rows)
    train_df, valid_df, y_train, y_valid = chronological_split(df, y, valid_fraction=0.2)
    model_names = selected_model_names(args.model)
    groups = feature_group_columns(df)

    rows: list[dict[str, Any]] = []
    for run_name, enabled_groups in ABLATION_RUNS:
        train_subset = select_groups(train_df, groups, enabled_groups)
        valid_subset = select_groups(valid_df, groups, enabled_groups)
        prior = float(np.mean(y_train))
        for model_name in model_names:
            result = fit_model_on_frames(
                model_name=model_name,
                train_df=train_subset,
                y_train=y_train,
                valid_df=valid_subset,
                y_valid=y_valid,
                seed=args.seed,
                feature_profile=args.feature_profile,
                model_config=model_config,
            )
            metrics = evaluate_predictions(y_valid, result.y_score, prior=prior)
            max_f1_policy = metrics["thresholds"]["max_f1"]
            rows.append(
                {
                    "run": run_name,
                    "enabled_groups": "+".join(enabled_groups),
                    "model": model_name,
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "max_f1": max_f1_policy["f1"],
                    "train_seconds": result.train_seconds,
                }
            )

    summary = pd.DataFrame(rows, columns=ABLATION_COLUMNS)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
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
            "runs": [{"run": run, "enabled_groups": enabled} for run, enabled in ABLATION_RUNS],
            "groups": groups,
            "model_configs": reviewed_model_configs(),
        },
    )
    write_json(
        output_dir / "runtime.json",
        {
            "total_seconds": time.perf_counter() - total_start,
            "rows_loaded": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_valid": int(len(valid_df)),
            "models": model_names,
        },
    )
    write_manifest(
        output_dir / "manifest.json",
        build_run_manifest(
            candidate_id=f"ablation_{runner_identity_slug(args.model, args.feature_profile, model_config)}",
            config=runner_manifest_config(args.model, args.feature_profile, model_config),
            split_policy="ablation",
            source_run_id=f"ablation_{runner_identity_slug(args.model, args.feature_profile, model_config)}",
            command=command,
            output_dir=output_dir,
            artifact_role="local_validation",
            sample_rows=args.sample_rows,
            train_seconds=float(summary["train_seconds"].sum()) if not summary.empty else 0.0,
            extra={
                "rows_loaded": int(len(df)),
                "rows_train": int(len(train_df)),
                "rows_valid": int(len(valid_df)),
                "runs": [{"run": run, "enabled_groups": enabled} for run, enabled in ABLATION_RUNS],
                "models": model_names,
            },
        ),
    )

    print(f"Ablation rows: {len(summary)}")
    print(f"Output dir: {output_dir}")
    return 0


def feature_group_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = set(df.columns)
    transaction_core = [
        column
        for column in [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "dist1",
            "dist2",
            "P_emaildomain",
            "R_emaildomain",
        ]
        if column in columns
    ]
    counts_and_d = [column for column in df.columns if re.fullmatch(r"C\d+", column) or re.fullmatch(r"D\d+", column)]
    m_and_v = [column for column in df.columns if re.fullmatch(r"M\d+", column) or re.fullmatch(r"V\d+", column)]
    identity = [
        column
        for column in df.columns
        if column.startswith("id_") or column in {"DeviceType", "DeviceInfo"}
    ]
    return {
        "transaction_core": transaction_core,
        "counts_and_d": counts_and_d,
        "m_and_v": m_and_v,
        "identity": identity,
    }


def select_groups(df: pd.DataFrame, groups: dict[str, list[str]], enabled_groups: list[str]) -> pd.DataFrame:
    selected: list[str] = []
    seen: set[str] = set()
    for group_name in enabled_groups:
        for column in groups[group_name]:
            if column not in seen and column in df.columns:
                selected.append(column)
                seen.add(column)
    return df.loc[:, selected].copy()


if __name__ == "__main__":
    raise SystemExit(main())
