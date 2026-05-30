#!/usr/bin/env python3
"""Build small report-ready CSV artifacts from experiment outputs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_model.metrics import calibration_table, threshold_summary


EXPERIMENT_INPUTS = {
    "metrics": "metrics.csv",
    "threshold": "threshold_summary.csv",
    "calibration": "calibration_table.csv",
    "fit_curve": "fit_curve.csv",
}
ROLLING_INPUT = "rolling_metrics.csv"
MONTH_GAP_INPUT = "month_gap_metrics.csv"
ABLATION_INPUT = "ablation_summary.csv"

METRIC_COLUMNS = [
    "run",
    "model",
    "rows_train",
    "rows_valid",
    "positive_rate_train",
    "positive_rate_valid",
    "roc_auc",
    "average_precision",
    "brier",
    "max_f1",
    "best_threshold",
    "train_seconds",
]
THRESHOLD_COLUMNS = ["model", "threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
CALIBRATION_COLUMNS = ["model", "bin", "count", "mean_pred", "frac_positive"]
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
ABLATION_COLUMNS = ["run", "enabled_groups", "model", "roc_auc", "average_precision", "max_f1", "train_seconds"]
MONTH_GAP_COLUMNS = [
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
BOOTSTRAP_COLUMNS = [
    "anchor",
    "candidate",
    "repeats",
    "seed",
    "n_rows",
    "n_pos",
    "n_neg",
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
FINAL_CONTENDER_COLUMNS = [
    "candidate_id",
    "role",
    "model_family",
    "config_id",
    "config_hash",
    "feature_profile",
    "split_policy",
    "source_run_id",
    "source_run_dir",
    "submission_artifact",
    "sample_rows",
    "local_oot_auc",
    "local_oot_ap",
    "local_oot_brier",
    "max_f1",
    "train_seconds",
    "observed_status",
    "decision",
    "paper_claim",
    "limitation_tag",
]
KAGGLE_SUMMARY_COLUMNS = [
    "candidate_id",
    "source_run_id",
    "source_run_dir",
    "submission_artifact",
    "identity_gate_status",
    "schema_validation_status",
    "ref",
    "status",
    "public_score",
    "private_score",
    "observed_or_projected",
    "operator",
    "submitted_at",
    "decision",
    "notes",
]
MODEL_SELECTION_COLUMNS = [
    "candidate_id",
    "selected_as",
    "reason",
    "local_oot_rank",
    "kaggle_rank",
    "robustness_status",
    "calibration_status",
    "runtime_status",
    "evidence_status",
    "paper_claim",
    "limitations",
]
FIT_CURVE_COLUMNS = ["model", "iteration", "train_loss", "valid_loss", "train_auc", "valid_auc"]
FEATURE_PROFILE_COLUMNS = [
    "run",
    "feature_profile",
    "model",
    "roc_auc",
    "average_precision",
    "brier",
    "max_f1",
    "train_seconds",
]
FIT_SUMMARY_COLUMNS = [
    "model",
    "iterations",
    "last_iteration",
    "best_valid_auc",
    "best_valid_auc_iteration",
    "last_valid_auc",
    "min_valid_loss",
    "min_valid_loss_iteration",
    "last_valid_loss",
    "last_train_loss",
    "last_train_auc",
]
CONTRIBUTION_COLUMNS = ["member", "contribution", "notes", "status"]
PREDICTIONS_INPUT = "predictions_valid.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final-report CSV artifacts from Task 10 outputs.")
    parser.add_argument("--experiment-dir", type=Path, required=True, help="Directory containing OOT artifacts.")
    parser.add_argument(
        "--comparison-experiment-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional OOT directory whose metrics should be included in feature profile comparison output.",
    )
    parser.add_argument("--rolling-dir", type=Path, required=True, help="Directory containing rolling OOT artifacts.")
    parser.add_argument("--month-gap-dir", type=Path, required=True, help="Directory containing month-gap artifacts.")
    parser.add_argument("--ablation-dir", type=Path, required=True, help="Directory containing ablation artifacts.")
    parser.add_argument("--bootstrap-file", type=Path, required=True, help="Paired bootstrap uncertainty CSV.")
    parser.add_argument("--final-contenders-file", type=Path, required=True, help="Final contender summary CSV.")
    parser.add_argument("--kaggle-summary-file", type=Path, required=True, help="Observed/projected Kaggle summary CSV.")
    parser.add_argument("--lr-audit-summary-file", type=Path, required=True, help="LR audit summary CSV.")
    parser.add_argument("--boosting-search-summary-file", type=Path, required=True, help="Boosting search summary CSV.")
    parser.add_argument("--model-selection-file", type=Path, required=True, help="Model selection decision CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for report figure data CSVs.")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("final-report/tables"),
        help="Directory for report table CSV drafts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = input_paths(args.experiment_dir, args.rolling_dir, args.month_gap_dir, args.ablation_dir)
    comparison_dirs = args.comparison_experiment_dir
    comparison_paths = [directory / EXPERIMENT_INPUTS["metrics"] for directory in comparison_dirs]
    extra_paths = [
        args.bootstrap_file,
        args.final_contenders_file,
        args.kaggle_summary_file,
        args.lr_audit_summary_file,
        args.boosting_search_summary_file,
        args.model_selection_file,
    ]
    required_paths = [paths["metrics"], paths["rolling"], paths["month_gap"], paths["ablation"], *extra_paths]
    optional_or_derivable_paths = [paths["threshold"], paths["calibration"], paths["fit_curve"]]
    require_files(required_paths)
    require_derivable_experiment_files(args.experiment_dir, optional_or_derivable_paths)
    require_files(comparison_paths)
    for directory in comparison_dirs:
        require_derivable_experiment_files(
            directory,
            [
                directory / EXPERIMENT_INPUTS["threshold"],
                directory / EXPERIMENT_INPUTS["calibration"],
            ],
        )

    all_metrics = pd.read_csv(paths["metrics"])
    require_columns(all_metrics, paths["metrics"], METRIC_COLUMNS)
    comparison_metrics = read_comparison_metrics(comparison_paths)
    metric_frames = [all_metrics, *comparison_metrics]
    metrics = pd.concat(
        [frame.loc[:, METRIC_COLUMNS].copy() for frame in metric_frames],
        ignore_index=True,
    )
    threshold_frames, calibration_frames = read_threshold_and_calibration_frames(
        args.experiment_dir,
        paths["threshold"],
        paths["calibration"],
        comparison_dirs,
    )
    threshold = pd.concat(threshold_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    rolling = read_required_columns(paths["rolling"], ROLLING_COLUMNS)
    month_gap = read_required_columns(paths["month_gap"], MONTH_GAP_COLUMNS)
    ablation = read_required_columns(paths["ablation"], ABLATION_COLUMNS)
    bootstrap = read_required_columns(args.bootstrap_file, BOOTSTRAP_COLUMNS)
    final_contenders = read_required_columns(args.final_contenders_file, FINAL_CONTENDER_COLUMNS)
    kaggle = read_required_columns(args.kaggle_summary_file, KAGGLE_SUMMARY_COLUMNS)
    lr_audit = pd.read_csv(args.lr_audit_summary_file)
    boosting_search = pd.read_csv(args.boosting_search_summary_file)
    selection = read_required_columns(args.model_selection_file, MODEL_SELECTION_COLUMNS)
    fit_curve = read_optional_fit_curve(paths["fit_curve"])

    output_dir = args.output_dir
    tables_dir = args.tables_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(output_dir / "model_metric_comparison.csv", index=False)
    write_feature_profile_comparison(
        [all_metrics, *comparison_metrics],
        output_dir,
        infer_profiles=bool(comparison_metrics),
    )
    threshold.to_csv(output_dir / "threshold_curve.csv", index=False)
    calibration_with_error(calibration).to_csv(output_dir / "calibration_reliability.csv", index=False)
    rolling.to_csv(output_dir / "rolling_oot_stability.csv", index=False)
    month_gap.to_csv(output_dir / "month_gap_stability.csv", index=False)
    ablation.to_csv(output_dir / "ablation_summary.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_uncertainty.csv", index=False)
    final_contenders.to_csv(output_dir / "final_contender_summary.csv", index=False)
    kaggle.to_csv(output_dir / "kaggle_submission_summary.csv", index=False)
    lr_audit.to_csv(output_dir / "lr_audit_summary.csv", index=False)
    boosting_search.to_csv(output_dir / "boosting_search_summary.csv", index=False)
    selection.to_csv(output_dir / "model_selection_decision.csv", index=False)
    fit_curve_summary(fit_curve).to_csv(output_dir / "fit_curve_summary.csv", index=False)
    contribution_draft().to_csv(tables_dir / "contribution-draft.csv", index=False)

    print(f"Report figure data written to {output_dir}")
    print(f"Contribution draft written to {tables_dir / 'contribution-draft.csv'}")
    return 0


def input_paths(experiment_dir: Path, rolling_dir: Path, month_gap_dir: Path, ablation_dir: Path) -> dict[str, Path]:
    return {
        "metrics": experiment_dir / EXPERIMENT_INPUTS["metrics"],
        "threshold": experiment_dir / EXPERIMENT_INPUTS["threshold"],
        "calibration": experiment_dir / EXPERIMENT_INPUTS["calibration"],
        "fit_curve": experiment_dir / EXPERIMENT_INPUTS["fit_curve"],
        "rolling": rolling_dir / ROLLING_INPUT,
        "month_gap": month_gap_dir / MONTH_GAP_INPUT,
        "ablation": ablation_dir / ABLATION_INPUT,
    }


def require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Missing required input file: {path}")


def require_derivable_experiment_files(experiment_dir: Path, paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    predictions = experiment_dir / PREDICTIONS_INPUT
    only_fit_curve_missing = all(path.name == EXPERIMENT_INPUTS["fit_curve"] for path in missing)
    if only_fit_curve_missing:
        return
    if not predictions.is_file():
        raise SystemExit(f"Missing required input file: {missing[0]}")


def read_required_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, path, columns)
    return df.loc[:, columns].copy()


def read_threshold_and_calibration(
    experiment_dir: Path,
    threshold_path: Path,
    calibration_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if threshold_path.is_file() and calibration_path.is_file():
        return (
            read_required_columns(threshold_path, THRESHOLD_COLUMNS),
            read_required_columns(calibration_path, CALIBRATION_COLUMNS),
        )
    predictions = pd.read_csv(experiment_dir / PREDICTIONS_INPUT)
    require_columns(predictions, experiment_dir / PREDICTIONS_INPUT, ["isFraud"])
    score_columns = [column for column in predictions.columns if column.endswith("_score")]
    if not score_columns:
        raise SystemExit(f"Input file {experiment_dir / PREDICTIONS_INPUT} is missing a *_score column")
    model_name = score_columns[0].removesuffix("_score")
    y_true = predictions["isFraud"].to_numpy()
    y_score = predictions[score_columns[0]].to_numpy()
    threshold = threshold_summary(y_true, y_score)
    threshold.insert(0, "model", model_name)
    calibration = calibration_table(y_true, y_score, bins=10)
    calibration.insert(0, "model", model_name)
    return threshold.loc[:, THRESHOLD_COLUMNS].copy(), calibration.loc[:, CALIBRATION_COLUMNS].copy()


def read_threshold_and_calibration_frames(
    experiment_dir: Path,
    threshold_path: Path,
    calibration_path: Path,
    comparison_dirs: Iterable[Path],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    threshold_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    for current_dir, current_threshold, current_calibration in [
        (experiment_dir, threshold_path, calibration_path),
        *[
            (
                directory,
                directory / EXPERIMENT_INPUTS["threshold"],
                directory / EXPERIMENT_INPUTS["calibration"],
            )
            for directory in comparison_dirs
        ],
    ]:
        threshold, calibration = read_threshold_and_calibration(current_dir, current_threshold, current_calibration)
        threshold_frames.append(threshold)
        calibration_frames.append(calibration)
    return threshold_frames, calibration_frames


def read_optional_fit_curve(path: Path) -> pd.DataFrame:
    if path.is_file():
        return read_required_columns(path, FIT_CURVE_COLUMNS)
    return pd.DataFrame(columns=FIT_CURVE_COLUMNS)


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise SystemExit(f"Input file {path} is missing required columns: {joined}")


def read_comparison_metrics(paths: Iterable[Path]) -> list[pd.DataFrame]:
    metrics = []
    for path in paths:
        df = pd.read_csv(path)
        require_columns(df, path, METRIC_COLUMNS)
        metrics.append(df)
    return metrics


def write_feature_profile_comparison(
    metric_frames: list[pd.DataFrame],
    output_dir: Path,
    *,
    infer_profiles: bool = False,
) -> None:
    path = output_dir / "feature_profile_comparison.csv"
    if infer_profiles:
        metric_frames = [ensure_feature_profile(frame) for frame in metric_frames]
    complete_frames = [
        frame
        for frame in metric_frames
        if {"run", "feature_profile", "model"}.issubset(frame.columns)
    ]
    if not complete_frames:
        path.unlink(missing_ok=True)
        return
    comparison = pd.concat([frame.loc[:, FEATURE_PROFILE_COLUMNS] for frame in complete_frames], ignore_index=True)
    comparison.to_csv(path, index=False)


def ensure_feature_profile(metrics: pd.DataFrame) -> pd.DataFrame:
    if "feature_profile" in metrics.columns:
        return metrics
    result = metrics.copy()
    result.insert(1, "feature_profile", [infer_feature_profile(run_name) for run_name in result["run"]])
    return result


def infer_feature_profile(run_name: object) -> str:
    normalized = str(run_name).lower().replace("-", "_")
    if "uid_agg" in normalized:
        return "uid_agg"
    if "uid_d" in normalized:
        return "uid_d"
    return "baseline"


def calibration_with_error(calibration: pd.DataFrame) -> pd.DataFrame:
    result = calibration.copy()
    result["calibration_error"] = (result["mean_pred"] - result["frac_positive"]).abs()
    return result.loc[:, [*CALIBRATION_COLUMNS, "calibration_error"]]


def fit_curve_summary(fit_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, model_curve in fit_curve.groupby("model", sort=False):
        ordered = model_curve.sort_values("iteration", kind="mergesort").reset_index(drop=True)
        last = ordered.iloc[-1]
        best_auc = best_row(ordered, "valid_auc", maximize=True)
        min_loss = best_row(ordered, "valid_loss", maximize=False)
        rows.append(
            {
                "model": model,
                "iterations": int(len(ordered)),
                "last_iteration": int(last["iteration"]),
                "best_valid_auc": row_value(best_auc, "valid_auc"),
                "best_valid_auc_iteration": row_iteration(best_auc),
                "last_valid_auc": last["valid_auc"],
                "min_valid_loss": row_value(min_loss, "valid_loss"),
                "min_valid_loss_iteration": row_iteration(min_loss),
                "last_valid_loss": last["valid_loss"],
                "last_train_loss": last["train_loss"],
                "last_train_auc": last["train_auc"],
            }
        )
    return pd.DataFrame(rows, columns=FIT_SUMMARY_COLUMNS)


def best_row(df: pd.DataFrame, column: str, maximize: bool) -> pd.Series | None:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return None
    index = values.idxmax() if maximize else values.idxmin()
    return df.loc[index]


def row_value(row: pd.Series | None, column: str) -> Any:
    if row is None:
        return pd.NA
    return row[column]


def row_iteration(row: pd.Series | None) -> int | Any:
    if row is None:
        return pd.NA
    return int(row["iteration"])


def contribution_draft() -> pd.DataFrame:
    rows = [
        {
            "member": "Team Member 1",
            "contribution": "",
            "notes": "Replace placeholder with agreed final-report contribution.",
            "status": "draft",
        },
        {
            "member": "Team Member 2",
            "contribution": "",
            "notes": "Replace placeholder with agreed final-report contribution.",
            "status": "draft",
        },
        {
            "member": "Team Member 3",
            "contribution": "",
            "notes": "Replace placeholder with agreed final-report contribution.",
            "status": "draft",
        },
        {
            "member": "Team Member 4",
            "contribution": "",
            "notes": "Replace placeholder with agreed final-report contribution.",
            "status": "draft",
        },
    ]
    return pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS)


if __name__ == "__main__":
    raise SystemExit(main())
