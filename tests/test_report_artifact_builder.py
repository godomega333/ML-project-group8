#!/usr/bin/env python3
"""Regression tests for Task 11 report artifact builder."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd


EXPECTED_REPORT_FILES = [
    "ablation_summary.csv",
    "boosting_search_summary.csv",
    "bootstrap_uncertainty.csv",
    "calibration_reliability.csv",
    "feature_profile_comparison.csv",
    "final_contender_summary.csv",
    "fit_curve_summary.csv",
    "kaggle_submission_summary.csv",
    "lr_audit_summary.csv",
    "model_metric_comparison.csv",
    "model_selection_decision.csv",
    "month_gap_stability.csv",
    "rolling_oot_stability.csv",
    "threshold_curve.csv",
]


class ReportArtifactBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.experiment_dir = self.root / "oot"
        self.rolling_dir = self.root / "rolling"
        self.month_gap_dir = self.root / "month gap"
        self.ablation_dir = self.root / "ablation"
        self.bootstrap_file = self.root / "bootstrap.csv"
        self.final_contenders_file = self.root / "final contenders.csv"
        self.kaggle_summary_file = self.root / "kaggle summary.csv"
        self.lr_audit_summary_file = self.root / "lr audit summary.csv"
        self.boosting_search_summary_file = self.root / "boosting search summary.csv"
        self.model_selection_file = self.root / "model selection.csv"
        self.output_dir = self.root / "report data"
        self.tables_dir = self.root / "tables"
        self._write_inputs()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_input_exits_with_file_name(self) -> None:
        builder = importlib.import_module("experiments.build_report_artifacts")
        missing_file = self.experiment_dir / "calibration_table.csv"
        missing_file.unlink()

        with self.assertRaises(SystemExit) as raised:
            builder.main(
                [
                    "--experiment-dir",
                    str(self.experiment_dir),
                    "--rolling-dir",
                    str(self.rolling_dir),
                    "--month-gap-dir",
                    str(self.month_gap_dir),
                    "--ablation-dir",
                    str(self.ablation_dir),
                    "--bootstrap-file",
                    str(self.bootstrap_file),
                    "--final-contenders-file",
                    str(self.final_contenders_file),
                    "--kaggle-summary-file",
                    str(self.kaggle_summary_file),
                    "--lr-audit-summary-file",
                    str(self.lr_audit_summary_file),
                    "--boosting-search-summary-file",
                    str(self.boosting_search_summary_file),
                    "--model-selection-file",
                    str(self.model_selection_file),
                    "--output-dir",
                    str(self.output_dir),
                    "--tables-dir",
                    str(self.tables_dir),
                ]
            )

        self.assertIn(str(missing_file), str(raised.exception))
        self.assertFalse(self.output_dir.exists())

    def test_missing_column_exits_with_file_and_column_name(self) -> None:
        builder = importlib.import_module("experiments.build_report_artifacts")
        metrics_file = self.experiment_dir / "metrics.csv"
        metrics = pd.read_csv(metrics_file).drop(columns=["roc_auc"])
        metrics.to_csv(metrics_file, index=False)

        with self.assertRaises(SystemExit) as raised:
            builder.main(
                [
                    "--experiment-dir",
                    str(self.experiment_dir),
                    "--rolling-dir",
                    str(self.rolling_dir),
                    "--month-gap-dir",
                    str(self.month_gap_dir),
                    "--ablation-dir",
                    str(self.ablation_dir),
                    "--bootstrap-file",
                    str(self.bootstrap_file),
                    "--final-contenders-file",
                    str(self.final_contenders_file),
                    "--kaggle-summary-file",
                    str(self.kaggle_summary_file),
                    "--lr-audit-summary-file",
                    str(self.lr_audit_summary_file),
                    "--boosting-search-summary-file",
                    str(self.boosting_search_summary_file),
                    "--model-selection-file",
                    str(self.model_selection_file),
                    "--output-dir",
                    str(self.output_dir),
                    "--tables-dir",
                    str(self.tables_dir),
                ]
            )

        message = str(raised.exception)
        self.assertIn(str(metrics_file), message)
        self.assertIn("roc_auc", message)
        self.assertFalse(self.output_dir.exists())

    def test_builder_writes_report_ready_schemas(self) -> None:
        builder = importlib.import_module("experiments.build_report_artifacts")
        optimization_dir = self.root / "oot optimized"
        optimization_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "run": "optimized",
                    "feature_profile": "uid_agg",
                    "model": "lr",
                    "rows_train": 8,
                    "rows_valid": 2,
                    "positive_rate_train": 0.25,
                    "positive_rate_valid": 0.5,
                    "roc_auc": 0.72,
                    "average_precision": 0.45,
                    "brier": 0.22,
                    "max_f1": 0.55,
                    "best_threshold": 0.3,
                    "train_seconds": 0.2,
                }
            ]
        ).to_csv(optimization_dir / "metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "lr",
                    "threshold": 0.5,
                    "accuracy": 0.9,
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.75,
                    "tp": 4,
                    "fp": 1,
                    "tn": 5,
                    "fn": 1,
                }
            ]
        ).to_csv(optimization_dir / "threshold_summary.csv", index=False)
        pd.DataFrame(
            [{"model": "lr", "bin": 0, "count": 12, "mean_pred": 0.20, "frac_positive": 0.25}]
        ).to_csv(optimization_dir / "calibration_table.csv", index=False)

        exit_code = builder.main(
            [
                "--experiment-dir",
                str(self.experiment_dir),
                "--comparison-experiment-dir",
                str(optimization_dir),
                "--rolling-dir",
                str(self.rolling_dir),
                "--month-gap-dir",
                str(self.month_gap_dir),
                "--ablation-dir",
                str(self.ablation_dir),
                "--bootstrap-file",
                str(self.bootstrap_file),
                "--final-contenders-file",
                str(self.final_contenders_file),
                "--kaggle-summary-file",
                str(self.kaggle_summary_file),
                "--lr-audit-summary-file",
                str(self.lr_audit_summary_file),
                "--boosting-search-summary-file",
                str(self.boosting_search_summary_file),
                "--model-selection-file",
                str(self.model_selection_file),
                "--output-dir",
                str(self.output_dir),
                "--tables-dir",
                str(self.tables_dir),
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(sorted(path.name for path in self.output_dir.iterdir()), EXPECTED_REPORT_FILES)
        comparison = pd.read_csv(self.output_dir / "model_metric_comparison.csv")
        self.assertEqual(comparison["model"].tolist(), ["lr", "lr"])
        self.assertEqual(comparison["run"].tolist(), ["synthetic", "optimized"])
        self.assertEqual(
            list(comparison.columns),
            [
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
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "feature_profile_comparison.csv").columns),
            [
                "run",
                "feature_profile",
                "model",
                "roc_auc",
                "average_precision",
                "brier",
                "max_f1",
                "train_seconds",
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "threshold_curve.csv").columns),
            ["model", "threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"],
        )
        self.assertEqual(len(pd.read_csv(self.output_dir / "threshold_curve.csv")), 2)
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "calibration_reliability.csv").columns),
            ["model", "bin", "count", "mean_pred", "frac_positive", "calibration_error"],
        )
        self.assertEqual(len(pd.read_csv(self.output_dir / "calibration_reliability.csv")), 2)
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "rolling_oot_stability.csv").columns),
            [
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
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "month_gap_stability.csv").columns),
            [
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
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "ablation_summary.csv").columns),
            [
                "run",
                "enabled_groups",
                "model",
                "roc_auc",
                "average_precision",
                "max_f1",
                "train_seconds",
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "bootstrap_uncertainty.csv").columns),
            [
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
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "final_contender_summary.csv").columns),
            [
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
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "kaggle_submission_summary.csv").columns),
            [
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
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "model_selection_decision.csv").columns),
            [
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
            ],
        )
        self.assertEqual(list(pd.read_csv(self.output_dir / "lr_audit_summary.csv").columns), ["rank", "candidate_id"])
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "boosting_search_summary.csv").columns),
            ["rank", "candidate_id"],
        )
        self.assertEqual(
            list(pd.read_csv(self.output_dir / "fit_curve_summary.csv").columns),
            [
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
            ],
        )
        self.assertEqual(
            list(pd.read_csv(self.tables_dir / "contribution-draft.csv").columns),
            ["member", "contribution", "notes", "status"],
        )
        profile_comparison = pd.read_csv(self.output_dir / "feature_profile_comparison.csv")
        self.assertEqual(profile_comparison.loc[0, "feature_profile"], "uid_d")
        self.assertEqual(profile_comparison.loc[1, "feature_profile"], "uid_agg")
        calibration = pd.read_csv(self.output_dir / "calibration_reliability.csv")
        self.assertAlmostEqual(calibration.loc[0, "calibration_error"], 0.20)
        fit_summary = pd.read_csv(self.output_dir / "fit_curve_summary.csv")
        self.assertEqual(fit_summary.loc[0, "best_valid_auc_iteration"], 2)
        self.assertAlmostEqual(fit_summary.loc[0, "min_valid_loss"], 0.40)
        boosting_summary = fit_summary.loc[fit_summary["model"] == "boosting"].iloc[0]
        self.assertEqual(boosting_summary["best_valid_auc_iteration"], 2)
        self.assertAlmostEqual(boosting_summary["last_valid_auc"], 0.64)
        self.assertTrue(pd.isna(boosting_summary["min_valid_loss"]))
        self.assertTrue(pd.isna(boosting_summary["min_valid_loss_iteration"]))

    def test_builder_skips_profile_comparison_when_feature_profile_missing(self) -> None:
        builder = importlib.import_module("experiments.build_report_artifacts")
        metrics_file = self.experiment_dir / "metrics.csv"
        metrics = pd.read_csv(metrics_file).drop(columns=["feature_profile"])
        metrics.to_csv(metrics_file, index=False)
        self.output_dir.mkdir(parents=True)
        stale_profile = self.output_dir / "feature_profile_comparison.csv"
        stale_profile.write_text("stale,artifact\n1,2\n", encoding="utf-8")

        exit_code = builder.main(
            [
                "--experiment-dir",
                str(self.experiment_dir),
                "--rolling-dir",
                str(self.rolling_dir),
                "--month-gap-dir",
                str(self.month_gap_dir),
                "--ablation-dir",
                str(self.ablation_dir),
                "--bootstrap-file",
                str(self.bootstrap_file),
                "--final-contenders-file",
                str(self.final_contenders_file),
                "--kaggle-summary-file",
                str(self.kaggle_summary_file),
                "--lr-audit-summary-file",
                str(self.lr_audit_summary_file),
                "--boosting-search-summary-file",
                str(self.boosting_search_summary_file),
                "--model-selection-file",
                str(self.model_selection_file),
                "--output-dir",
                str(self.output_dir),
                "--tables-dir",
                str(self.tables_dir),
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertFalse(stale_profile.exists())

    def test_builder_derives_threshold_and_calibration_from_predictions(self) -> None:
        builder = importlib.import_module("experiments.build_report_artifacts")
        (self.experiment_dir / "threshold_summary.csv").unlink()
        (self.experiment_dir / "calibration_table.csv").unlink()
        (self.experiment_dir / "fit_curve.csv").unlink()
        pd.DataFrame(
            [
                {"TransactionID": 1, "TransactionDT": 100, "isFraud": 0, "boosting_score": 0.1},
                {"TransactionID": 2, "TransactionDT": 101, "isFraud": 1, "boosting_score": 0.9},
                {"TransactionID": 3, "TransactionDT": 102, "isFraud": 0, "boosting_score": 0.4},
                {"TransactionID": 4, "TransactionDT": 103, "isFraud": 1, "boosting_score": 0.8},
            ]
        ).to_csv(self.experiment_dir / "predictions_valid.csv", index=False)

        exit_code = builder.main(
            [
                "--experiment-dir",
                str(self.experiment_dir),
                "--rolling-dir",
                str(self.rolling_dir),
                "--month-gap-dir",
                str(self.month_gap_dir),
                "--ablation-dir",
                str(self.ablation_dir),
                "--bootstrap-file",
                str(self.bootstrap_file),
                "--final-contenders-file",
                str(self.final_contenders_file),
                "--kaggle-summary-file",
                str(self.kaggle_summary_file),
                "--lr-audit-summary-file",
                str(self.lr_audit_summary_file),
                "--boosting-search-summary-file",
                str(self.boosting_search_summary_file),
                "--model-selection-file",
                str(self.model_selection_file),
                "--output-dir",
                str(self.output_dir),
                "--tables-dir",
                str(self.tables_dir),
            ]
        )

        self.assertEqual(exit_code, 0)
        threshold = pd.read_csv(self.output_dir / "threshold_curve.csv")
        calibration = pd.read_csv(self.output_dir / "calibration_reliability.csv")
        fit_summary = pd.read_csv(self.output_dir / "fit_curve_summary.csv")
        self.assertEqual(threshold["model"].unique().tolist(), ["boosting"])
        self.assertEqual(len(calibration), 10)
        self.assertEqual(list(fit_summary.columns), builder.FIT_SUMMARY_COLUMNS)
        self.assertTrue(fit_summary.empty)

    def _write_inputs(self) -> None:
        self.experiment_dir.mkdir(parents=True)
        self.rolling_dir.mkdir(parents=True)
        self.month_gap_dir.mkdir(parents=True)
        self.ablation_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "run": "synthetic",
                    "feature_profile": "uid_d",
                    "model": "lr",
                    "rows_train": 8,
                    "rows_valid": 2,
                    "positive_rate_train": 0.25,
                    "positive_rate_valid": 0.5,
                    "roc_auc": 0.75,
                    "average_precision": 0.5,
                    "brier": 0.2,
                    "max_f1": 0.6,
                    "best_threshold": 0.4,
                    "train_seconds": 0.1,
                }
            ]
        ).to_csv(self.experiment_dir / "metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "lr",
                    "threshold": 0.4,
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.6,
                    "f1": 0.64,
                    "tp": 3,
                    "fp": 1,
                    "tn": 5,
                    "fn": 2,
                }
            ]
        ).to_csv(self.experiment_dir / "threshold_summary.csv", index=False)
        pd.DataFrame(
            [{"model": "lr", "bin": 0, "count": 10, "mean_pred": 0.30, "frac_positive": 0.10}]
        ).to_csv(self.experiment_dir / "calibration_table.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "lr",
                    "iteration": 1,
                    "train_loss": 0.6,
                    "valid_loss": 0.5,
                    "train_auc": 0.7,
                    "valid_auc": 0.8,
                },
                {
                    "model": "lr",
                    "iteration": 2,
                    "train_loss": 0.5,
                    "valid_loss": 0.4,
                    "train_auc": 0.8,
                    "valid_auc": 0.85,
                },
                {
                    "model": "boosting",
                    "iteration": 1,
                    "train_loss": None,
                    "valid_loss": None,
                    "train_auc": 0.70,
                    "valid_auc": 0.60,
                },
                {
                    "model": "boosting",
                    "iteration": 2,
                    "train_loss": None,
                    "valid_loss": None,
                    "train_auc": 0.75,
                    "valid_auc": 0.64,
                },
            ]
        ).to_csv(self.experiment_dir / "fit_curve.csv", index=False)
        pd.DataFrame(
            [
                {
                    "fold": 1,
                    "train_start": 100,
                    "train_end": 200,
                    "valid_start": 201,
                    "valid_end": 250,
                    "model": "lr",
                    "roc_auc": 0.7,
                    "average_precision": 0.4,
                    "brier": 0.2,
                    "max_f1": 0.5,
                    "best_threshold": 0.4,
                    "train_seconds": 0.2,
                }
            ]
        ).to_csv(self.rolling_dir / "rolling_metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run": "month-gap",
                    "feature_profile": "baseline",
                    "model": "lr",
                    "rows_train": 8,
                    "rows_valid": 2,
                    "train_start": 100,
                    "train_end": 200,
                    "valid_start": 230,
                    "valid_end": 300,
                    "roc_auc": 0.7,
                    "average_precision": 0.4,
                    "brier": 0.2,
                    "max_f1": 0.5,
                    "best_threshold": 0.4,
                    "train_seconds": 0.2,
                }
            ]
        ).to_csv(self.month_gap_dir / "month_gap_metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run": "all_groups",
                    "enabled_groups": "transaction_core+identity",
                    "model": "lr",
                    "roc_auc": 0.7,
                    "average_precision": 0.4,
                    "max_f1": 0.5,
                    "train_seconds": 0.2,
                }
            ]
        ).to_csv(self.ablation_dir / "ablation_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "anchor": "lr_anchor_score",
                    "candidate": "boosting_score",
                    "repeats": 1000,
                    "seed": 42,
                    "n_rows": 10,
                    "n_pos": 2,
                    "n_neg": 8,
                    "auc_delta_mean": 0.1,
                    "auc_delta_low": 0.05,
                    "auc_delta_high": 0.15,
                    "ap_delta_mean": 0.2,
                    "ap_delta_low": 0.1,
                    "ap_delta_high": 0.3,
                    "brier_delta_mean": -0.01,
                    "brier_delta_low": -0.02,
                    "brier_delta_high": -0.005,
                }
            ]
        ).to_csv(self.bootstrap_file, index=False)
        pd.DataFrame(
            [
                {
                    "candidate_id": "boosting_final",
                    "role": "final_boosting_contender",
                    "model_family": "boosting",
                    "config_id": "boosting_final",
                    "config_hash": "abc123",
                    "feature_profile": "baseline",
                    "split_policy": "final_oot",
                    "source_run_id": "run-final",
                    "source_run_dir": "outputs/run-final",
                    "submission_artifact": "outputs/submission/submission.csv",
                    "sample_rows": pd.NA,
                    "local_oot_auc": 0.9,
                    "local_oot_ap": 0.5,
                    "local_oot_brier": 0.08,
                    "max_f1": 0.5,
                    "train_seconds": 1.0,
                    "observed_status": "observed",
                    "decision": "selected",
                    "paper_claim": "claim",
                    "limitation_tag": "none",
                    "needs_bootstrap": 1,
                    "needs_rolling": 1,
                    "needs_month_gap": 1,
                    "needs_ablation": 1,
                }
            ]
        ).to_csv(self.final_contenders_file, index=False)
        pd.DataFrame(
            [
                {
                    "candidate_id": "boosting_final",
                    "source_run_id": "run-final",
                    "source_run_dir": "outputs/run-final",
                    "submission_artifact": "outputs/submission/submission.csv",
                    "identity_gate_status": "passed",
                    "schema_validation_status": "passed",
                    "ref": 123,
                    "status": "SubmissionStatus.COMPLETE",
                    "public_score": 0.9,
                    "private_score": 0.88,
                    "observed_or_projected": "observed",
                    "operator": "local_kaggle_cli",
                    "submitted_at": "2026-05-29T00:00:00+08:00",
                    "decision": "candidate",
                    "notes": "notes",
                }
            ]
        ).to_csv(self.kaggle_summary_file, index=False)
        pd.DataFrame([{"rank": 1, "candidate_id": "lr"}]).to_csv(self.lr_audit_summary_file, index=False)
        pd.DataFrame([{"rank": 1, "candidate_id": "boosting"}]).to_csv(
            self.boosting_search_summary_file,
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "candidate_id": "boosting_final",
                    "selected_as": "final_boosting_contender",
                    "reason": "best local OOT",
                    "local_oot_rank": 1,
                    "kaggle_rank": "public=1;private=1",
                    "robustness_status": "passed",
                    "calibration_status": "pending",
                    "runtime_status": "recorded",
                    "evidence_status": "complete",
                    "paper_claim": "claim",
                    "limitations": "none",
                }
            ]
        ).to_csv(self.model_selection_file, index=False)


if __name__ == "__main__":
    unittest.main()
