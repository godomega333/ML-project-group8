#!/usr/bin/env python3
"""Run primary chronological out-of-time fraud experiments."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_model.boosting import HistogramGradientBoostingClassifier
from fraud_model.configs import ModelConfig, class_weight_from_alpha, config_hash, get_model_config
from fraud_model.data import load_train_data
from fraud_model.experiment import (
    ExperimentConfig,
    ensure_output_dir,
    evaluate_predictions,
    write_json,
    write_metrics_csv,
    write_presenter_notes,
)
from fraud_model.features import FeaturePipeline
from fraud_model.logistic import MatrixLogisticRegression
from fraud_model.manifest import build_run_manifest, write_manifest
from fraud_model.metrics import calibration_table, threshold_summary
from fraud_model.splits import final_oot_split, inner_tuning_split


LR_CONFIG: dict[str, Any] = {
    "learning_rate": 0.05,
    "l2": 0.05,
    "epochs": 12,
    "batch_size": 2048,
    "tolerance": 0.0,
}

BOOSTING_CONFIG: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.08,
    "n_bins": 96,
    "l2": 1.0,
    "gamma": 0.0,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample": 0.75,
}


@dataclass
class ModelRunResult:
    model_name: str
    estimator: Any
    y_score: np.ndarray
    history: dict[str, list[float]]
    train_seconds: float
    config: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IEEE-CIS chronological out-of-time experiments.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Kaggle competition CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for experiment artifacts.")
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
    parser.add_argument(
        "--split-policy",
        choices=["final_oot", "inner_tuning"],
        default="final_oot",
        help="Chronological split policy. inner_tuning keeps final 20 percent untouched.",
    )
    parser.add_argument("--run-name", default="oot", help="Run name stored in metrics artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    model_config = get_model_config(args.config_id) if args.config_id else None
    if model_config is not None:
        args.model = model_config.model_family
        args.feature_profile = model_config.feature_profile
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        sample_rows=args.sample_rows,
        valid_fraction=0.2,
        model=args.model,
        run_name=args.run_name,
    )

    total_start = time.perf_counter()
    output_dir = ensure_output_dir(config.output_dir)
    model_names = selected_model_names(config.model)
    command = recorded_command("experiments/run_oot.py", raw_argv)

    load_start = time.perf_counter()
    df, y = load_train_data(config.data_dir, nrows=config.sample_rows)
    load_seconds = time.perf_counter() - load_start
    if args.split_policy == "inner_tuning":
        train_df, valid_df, y_train, y_valid = inner_tuning_split(df, y)
    else:
        train_df, valid_df, y_train, y_valid = final_oot_split(df, y, valid_fraction=config.valid_fraction)
    prior = float(np.mean(y_train))

    base_predictions = _validation_prediction_frame(valid_df, y_valid)
    metrics_by_model: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    fit_curve_rows: list[dict[str, Any]] = []
    runtime_by_model: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        result = fit_model_on_frames(
            model_name=model_name,
            train_df=train_df,
            y_train=y_train,
            valid_df=valid_df,
            y_valid=y_valid,
            seed=config.seed,
            feature_profile=args.feature_profile,
            model_config=model_config,
        )
        base_predictions[f"{model_name}_score"] = result.y_score

        metrics = evaluate_predictions(y_valid, result.y_score, prior=prior)
        metrics_by_model[model_name] = metrics
        max_f1_policy = metrics["thresholds"]["max_f1"]
        metric_rows.append(
            {
                "run": config.run_name,
                "feature_profile": args.feature_profile,
                "model": model_name,
                "rows_train": int(len(train_df)),
                "rows_valid": int(len(valid_df)),
                "positive_rate_train": prior,
                "positive_rate_valid": float(np.mean(y_valid)),
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "brier": metrics["brier_score"],
                "max_f1": max_f1_policy["f1"],
                "best_threshold": max_f1_policy["threshold"],
                "train_seconds": result.train_seconds,
            }
        )

        thresholds = threshold_summary(y_valid, result.y_score)
        thresholds.insert(0, "model", model_name)
        threshold_rows.extend(thresholds.to_dict(orient="records"))

        calibration = calibration_table(y_valid, result.y_score, bins=10)
        calibration.insert(0, "model", model_name)
        calibration_rows.extend(calibration.to_dict(orient="records"))

        fit_curve_rows.extend(history_rows(model_name, result.history))
        runtime_by_model[model_name] = {
            "train_seconds": result.train_seconds,
            "config": result.config,
            "fit_iterations": int(max((len(values) for values in result.history.values()), default=0)),
        }

    total_seconds = time.perf_counter() - total_start
    config_payload = {
        **config.to_json_dict(),
        "feature_profile": args.feature_profile,
        "split_policy": args.split_policy,
        "command": command,
        "config_id": model_config.config_id if model_config else None,
        "config_hash": config_hash(model_config) if model_config else None,
        "model_configs": reviewed_model_configs(),
    }
    metrics_payload = {
        "config": config_payload,
        "rows": {
            "loaded": int(len(df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "train_positive_rate": prior,
            "valid_positive_rate": float(np.mean(y_valid)),
        },
        "metrics": metrics_by_model,
    }
    runtime_payload = {
        "command": command,
        "load_seconds": load_seconds,
        "total_seconds": total_seconds,
        "models": runtime_by_model,
    }

    write_json(output_dir / "config.json", config_payload)
    write_metrics_csv(output_dir / "metrics.csv", metric_rows)
    write_json(output_dir / "metrics.json", metrics_payload)
    base_predictions.to_csv(output_dir / "predictions_valid.csv", index=False)
    write_metrics_csv(output_dir / "threshold_summary.csv", threshold_rows)
    write_metrics_csv(output_dir / "calibration_table.csv", calibration_rows)
    write_metrics_csv(output_dir / "fit_curve.csv", fit_curve_rows)
    write_json(output_dir / "runtime.json", runtime_payload)
    write_manifest(
        output_dir / "manifest.json",
        build_run_manifest(
            candidate_id=f"{config.run_name}_{runner_identity_slug(config.model, args.feature_profile, model_config)}",
            config=runner_manifest_config(config.model, args.feature_profile, model_config),
            split_policy=args.split_policy,
            source_run_id=(
                f"{config.run_name}_{runner_identity_slug(config.model, args.feature_profile, model_config)}_"
                f"{args.split_policy}"
            ),
            command=command,
            output_dir=output_dir,
            artifact_role="local_validation",
            sample_rows=config.sample_rows,
            train_seconds=sum(float(row["train_seconds"]) for row in metric_rows),
            extra={
                "rows_loaded": int(len(df)),
                "rows_train": int(len(train_df)),
                "rows_valid": int(len(valid_df)),
                "models": model_names,
            },
        ),
    )
    write_presenter_notes(
        output_dir / "presenter_notes.md",
        {
            "command": command,
            "model": config.model,
            "metrics": metrics_by_model,
            "interpretation": split_policy_interpretation(args.split_policy),
        },
    )

    print(_summary_text(config.run_name, metric_rows, output_dir))
    return 0


def selected_model_names(model_choice: str) -> list[str]:
    if model_choice == "both":
        return ["lr", "boosting"]
    if model_choice in {"lr", "boosting"}:
        return [model_choice]
    raise ValueError("model must be 'lr', 'boosting', or 'both'")


def reviewed_model_configs() -> dict[str, dict[str, Any]]:
    return {
        "lr": dict(LR_CONFIG),
        "boosting": dict(BOOSTING_CONFIG),
    }


def runner_manifest_config(
    model_choice: str,
    feature_profile: str,
    model_config: ModelConfig | None = None,
) -> ModelConfig:
    if model_config is not None:
        return model_config
    if model_choice == "lr":
        return ModelConfig(
            config_id="lr_baseline",
            model_family="lr",
            feature_profile=feature_profile,
            params=dict(LR_CONFIG),
            class_weight_alpha=1.0,
            notes="Runner logistic baseline until Task 4 config-id routing.",
        )
    if model_choice == "boosting":
        return ModelConfig(
            config_id="runner_boosting_200iter_d",
            model_family="boosting",
            feature_profile=feature_profile,
            params=dict(BOOSTING_CONFIG),
            positive_weight="balanced",
            notes="Runner boosting config until Task 4 config-id routing.",
        )
    if model_choice == "both":
        return ModelConfig(
            config_id="runner_both_reviewed",
            model_family="both",
            feature_profile=feature_profile,
            params=reviewed_model_configs(),
            notes="Combined runner config until Task 4 single-config routing.",
        )
    raise ValueError("model_choice must be 'lr', 'boosting', or 'both'")


def runner_identity_slug(
    model_choice: str,
    feature_profile: str,
    model_config: ModelConfig | None = None,
) -> str:
    if model_config is not None:
        return model_config.config_id
    return f"{model_choice}_{feature_profile}"


def recorded_command(script_path: str, argv: list[str] | None = None) -> str:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    return " ".join(shlex.quote(str(part)) for part in ["python", script_path, *raw_argv])


def split_policy_interpretation(split_policy: str) -> str:
    if split_policy == "inner_tuning":
        return (
            "This run trains and tunes on the protected inner window while leaving the final 20% untouched. "
            "The split, preprocessing fit, metrics, threshold grid, calibration table, and fit curve are "
            "recorded for report-ready tuning analysis."
        )
    return (
        "This run uses the earliest 80% of transactions for fitting and the latest 20% for validation. "
        "The split, preprocessing fit, metrics, threshold grid, calibration table, and fit curve are "
        "recorded for report-ready OOT analysis."
    )


def fit_model_on_frames(
    model_name: str,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    valid_df: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    seed: int = 42,
    feature_profile: str = "baseline",
    model_config: ModelConfig | None = None,
) -> ModelRunResult:
    if model_config is not None:
        if model_name != model_config.model_family:
            raise ValueError("model_name must match model_config.model_family")
        feature_profile = model_config.feature_profile

    pipeline = FeaturePipeline(feature_profile=feature_profile)
    pipeline.fit(train_df)
    train_x = pipeline.transform(train_df, model=model_name)
    valid_x = pipeline.transform(valid_df, model=model_name) if valid_df is not None else None

    result = fit_model_on_matrices(
        model_name=model_name,
        train_x=train_x,
        y_train=y_train,
        valid_x=valid_x,
        y_valid=y_valid,
        seed=seed,
        model_config=model_config,
    )
    return result


def fit_model_on_matrices(
    model_name: str,
    train_x: np.ndarray,
    y_train: np.ndarray,
    valid_x: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    seed: int = 42,
    model_config: ModelConfig | None = None,
) -> ModelRunResult:
    if model_config is not None and model_name != model_config.model_family:
        raise ValueError("model_name must match model_config.model_family")

    if model_name == "lr":
        model_params = dict(model_config.params) if model_config else dict(LR_CONFIG)
        class_weight = (
            class_weight_from_alpha(float(np.mean(y_train)), model_config.class_weight_alpha)
            if model_config
            else balanced_class_weight(y_train)
        )
        estimator = MatrixLogisticRegression(
            **model_params,
            class_weight=class_weight,
            seed=seed,
        )
        fit_start = time.perf_counter()
        history = estimator.fit(train_x, y_train, valid_x=valid_x, valid_y=y_valid)
        train_seconds = time.perf_counter() - fit_start
        y_score = estimator.predict_proba(valid_x) if valid_x is not None else np.array([], dtype=np.float64)
        return ModelRunResult(model_name, estimator, y_score, history, train_seconds, _json_ready_config(model_params))

    if model_name == "boosting":
        model_params = dict(model_config.params) if model_config else dict(BOOSTING_CONFIG)
        positive_weight_value = resolved_boosting_positive_weight(y_train, model_config)
        estimator = HistogramGradientBoostingClassifier(
            **model_params,
            positive_weight=positive_weight_value,
            seed=seed,
        )
        fit_start = time.perf_counter()
        estimator.fit(train_x, y_train, x_valid=valid_x, y_valid=y_valid)
        train_seconds = time.perf_counter() - fit_start
        y_score = estimator.predict_proba(valid_x) if valid_x is not None else np.array([], dtype=np.float64)
        history = getattr(estimator, "history_", {})
        return ModelRunResult(model_name, estimator, y_score, history, train_seconds, _json_ready_config(model_params))

    raise ValueError("model_name must be 'lr' or 'boosting'")


def balanced_class_weight(y: np.ndarray) -> dict[int, float] | None:
    target = np.asarray(y).reshape(-1)
    positives = float(np.sum(target == 1))
    negatives = float(np.sum(target == 0))
    if positives <= 0.0 or negatives <= 0.0:
        return None
    total = positives + negatives
    return {0: total / (2.0 * negatives), 1: total / (2.0 * positives)}


def positive_weight(y: np.ndarray) -> float | None:
    target = np.asarray(y).reshape(-1)
    positives = float(np.sum(target == 1))
    negatives = float(np.sum(target == 0))
    if positives <= 0.0 or negatives <= 0.0:
        return None
    return negatives / positives


def resolved_boosting_positive_weight(y: np.ndarray, model_config: ModelConfig | None) -> float | None:
    if model_config is None or model_config.positive_weight == "balanced":
        return positive_weight(y)
    if model_config.positive_weight is None:
        return None
    try:
        return float(model_config.positive_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("positive_weight must be 'balanced', None, or numeric") from exc


def history_rows(model_name: str, history: dict[str, list[float]]) -> list[dict[str, Any]]:
    columns = ["train_loss", "valid_loss", "train_auc", "valid_auc"]
    iterations = max((len(history.get(column, [])) for column in columns), default=0)
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        row: dict[str, Any] = {"model": model_name, "iteration": iteration + 1}
        for column in columns:
            values = history.get(column, [])
            row[column] = values[iteration] if iteration < len(values) else np.nan
        rows.append(row)
    return rows


def _validation_prediction_frame(valid_df: pd.DataFrame, y_valid: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "TransactionID": valid_df["TransactionID"].to_numpy() if "TransactionID" in valid_df else np.arange(len(valid_df)),
            "TransactionDT": valid_df["TransactionDT"].to_numpy()
            if "TransactionDT" in valid_df
            else np.arange(len(valid_df)),
            "isFraud": np.asarray(y_valid).reshape(-1),
        }
    )
    return frame


def _summary_text(run_name: str, metric_rows: list[dict[str, Any]], output_dir: Path) -> str:
    lines = [f"OOT run: {run_name}", f"Output dir: {output_dir}"]
    for row in metric_rows:
        lines.append(
            f"{row['model']}: ROC-AUC={float(row['roc_auc']):.4f} "
            f"AP={float(row['average_precision']):.4f} "
            f"Brier={float(row['brier']):.4f} "
            f"maxF1={float(row['max_f1']):.4f}"
        )
    return "\n".join(lines)


def _json_ready_config(config: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, np.generic):
            ready[key] = value.item()
        else:
            ready[key] = value
    return ready


if __name__ == "__main__":
    raise SystemExit(main())
