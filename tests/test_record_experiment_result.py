#!/usr/bin/env python3
"""Tests for converting experiment outputs into ledger records."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER_SCRIPT = REPO_ROOT / "scripts" / "record_experiment_result.py"


def load_recorder_module():
    if not RECORDER_SCRIPT.is_file():
        raise AssertionError(f"Recorder script is missing: {RECORDER_SCRIPT}")
    spec = importlib.util.spec_from_file_location("record_experiment_result", RECORDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {RECORDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExperimentResultRecorderTest(unittest.TestCase):
    def test_builds_oot_record_from_metrics_csv(self) -> None:
        recorder = load_recorder_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "oot"
            output_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "model": "lr",
                        "roc_auc": 0.71,
                        "average_precision": 0.18,
                        "brier": 0.24,
                        "max_f1": 0.30,
                        "best_threshold": 0.8,
                        "train_seconds": 3.0,
                    },
                    {
                        "model": "boosting",
                        "roc_auc": 0.76,
                        "average_precision": 0.24,
                        "brier": 0.09,
                        "max_f1": 0.35,
                        "best_threshold": 0.6,
                        "train_seconds": 7.0,
                    },
                ]
            ).to_csv(output_dir / "metrics.csv", index=False)

            record = recorder.build_record(
                kind="oot",
                run_id="oot-test",
                output_dir=output_dir,
                command="python experiments/run_oot.py",
                data_scope="unit",
                decision="promote",
                reason="unit test",
            )

        self.assertEqual(record["run_id"], "oot-test")
        self.assertAlmostEqual(record["metrics"]["boosting"]["roc_auc"], 0.76)
        self.assertAlmostEqual(record["metrics"]["lr"]["brier"], 0.24)

    def test_builds_rolling_record_with_mean_metrics(self) -> None:
        recorder = load_recorder_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "rolling"
            output_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "model": "lr",
                        "roc_auc": 0.70,
                        "average_precision": 0.10,
                        "brier": 0.20,
                        "max_f1": 0.30,
                        "best_threshold": 0.5,
                        "train_seconds": 1.0,
                    },
                    {
                        "model": "lr",
                        "roc_auc": 0.80,
                        "average_precision": 0.20,
                        "brier": 0.30,
                        "max_f1": 0.40,
                        "best_threshold": 0.7,
                        "train_seconds": 2.0,
                    },
                ]
            ).to_csv(output_dir / "rolling_metrics.csv", index=False)

            record = recorder.build_record(
                kind="rolling",
                run_id="rolling-test",
                output_dir=output_dir,
                command="python experiments/run_rolling_oot.py",
                data_scope="unit",
                decision="reference-only",
                reason="unit test",
            )

        self.assertAlmostEqual(record["metrics"]["lr"]["roc_auc"], 0.75)
        self.assertAlmostEqual(record["metrics"]["lr"]["train_seconds"], 3.0)

    def test_oot_record_preserves_feature_profile_scope(self) -> None:
        recorder = load_recorder_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "oot"
            output_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "run": "profile",
                        "feature_profile": "uid_d",
                        "model": "lr",
                        "roc_auc": 0.71,
                        "average_precision": 0.18,
                        "brier": 0.24,
                        "max_f1": 0.30,
                        "best_threshold": 0.8,
                        "train_seconds": 3.0,
                    }
                ]
            ).to_csv(output_dir / "metrics.csv", index=False)

            record = recorder.build_record(
                kind="oot",
                run_id="oot-profile",
                output_dir=output_dir,
                command="python experiments/run_oot.py --feature-profile uid_d",
                data_scope="sample_rows=unit feature_profile=uid_d",
                decision="reference-only",
                reason="unit test",
            )

        self.assertEqual(record["data_scope"], "sample_rows=unit feature_profile=uid_d")
        self.assertIn("--feature-profile uid_d", record["command"])
        self.assertAlmostEqual(record["metrics"]["lr"]["roc_auc"], 0.71)

    def test_builds_month_gap_record_with_mean_metrics(self) -> None:
        recorder = load_recorder_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "month-gap"
            output_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "run": "month-gap",
                        "feature_profile": "uid_d",
                        "model": "lr",
                        "rows_train": 10,
                        "rows_valid": 5,
                        "train_start": 100,
                        "train_end": 500,
                        "valid_start": 800,
                        "valid_end": 1000,
                        "roc_auc": 0.70,
                        "average_precision": 0.10,
                        "brier": 0.20,
                        "max_f1": 0.30,
                        "best_threshold": 0.5,
                        "train_seconds": 1.0,
                    },
                    {
                        "run": "month-gap",
                        "feature_profile": "uid_d",
                        "model": "boosting",
                        "rows_train": 10,
                        "rows_valid": 5,
                        "train_start": 100,
                        "train_end": 500,
                        "valid_start": 800,
                        "valid_end": 1000,
                        "roc_auc": 0.80,
                        "average_precision": 0.20,
                        "brier": 0.10,
                        "max_f1": 0.40,
                        "best_threshold": 0.7,
                        "train_seconds": 2.0,
                    },
                    {
                        "run": "month-gap",
                        "feature_profile": "uid_d",
                        "model": "lr",
                        "rows_train": 20,
                        "rows_valid": 5,
                        "train_start": 200,
                        "train_end": 600,
                        "valid_start": 900,
                        "valid_end": 1100,
                        "roc_auc": 0.80,
                        "average_precision": 0.20,
                        "brier": 0.30,
                        "max_f1": 0.40,
                        "best_threshold": 0.7,
                        "train_seconds": 3.0,
                    },
                ]
            ).to_csv(output_dir / "month_gap_metrics.csv", index=False)

            record = recorder.build_record(
                kind="month-gap",
                run_id="month-gap-test",
                output_dir=output_dir,
                command="python experiments/run_month_gap.py --feature-profile uid_d",
                data_scope="sample_rows=unit month-gap feature_profile=uid_d",
                decision="reference-only",
                reason="unit test",
            )

        self.assertEqual(record["data_scope"], "sample_rows=unit month-gap feature_profile=uid_d")
        self.assertAlmostEqual(record["metrics"]["lr"]["roc_auc"], 0.75)
        self.assertAlmostEqual(record["metrics"]["lr"]["train_seconds"], 4.0)
        self.assertAlmostEqual(record["metrics"]["boosting"]["brier"], 0.10)

    def test_main_writes_json_and_appends_ledger(self) -> None:
        recorder = load_recorder_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "ablation"
            ledger = root / "ledger.md"
            record_json = root / "record.json"
            output_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "run": "base",
                        "model": "lr",
                        "roc_auc": 0.61,
                        "average_precision": 0.08,
                        "max_f1": 0.12,
                        "train_seconds": 1.0,
                    },
                    {
                        "run": "all_groups",
                        "model": "boosting",
                        "roc_auc": 0.72,
                        "average_precision": 0.20,
                        "max_f1": 0.30,
                        "train_seconds": 2.0,
                    },
                ]
            ).to_csv(output_dir / "ablation_summary.csv", index=False)

            exit_code = recorder.main(
                [
                    "--kind",
                    "ablation",
                    "--run-id",
                    "ablation-test",
                    "--output-dir",
                    str(output_dir),
                    "--command",
                    "python experiments/run_ablation.py",
                    "--data-scope",
                    "unit",
                    "--decision",
                    "reference-only",
                    "--reason",
                    "unit test",
                    "--ledger",
                    str(ledger),
                    "--record-json",
                    str(record_json),
                ]
            )
            payload = json.loads(record_json.read_text(encoding="utf-8"))
            content = ledger.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["run_id"], "ablation-test")
        self.assertIn("| ablation-test |", content)
        self.assertIn("| all_groups:boosting | 0.720000 | 0.200000 |", content)
        self.assertIn("python experiments/run_ablation.py", content)

    def test_ablation_record_keeps_both_models_for_same_run(self) -> None:
        recorder = load_recorder_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "ablation"
            output_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "run": "all_groups",
                        "enabled_groups": "transaction_core+identity",
                        "model": "lr",
                        "roc_auc": 0.62,
                        "average_precision": 0.12,
                        "max_f1": 0.20,
                        "train_seconds": 1.5,
                    },
                    {
                        "run": "all_groups",
                        "enabled_groups": "transaction_core+identity",
                        "model": "boosting",
                        "roc_auc": 0.74,
                        "average_precision": 0.22,
                        "max_f1": 0.33,
                        "train_seconds": 2.5,
                    },
                ]
            ).to_csv(output_dir / "ablation_summary.csv", index=False)

            record = recorder.build_record(
                kind="ablation",
                run_id="ablation-both",
                output_dir=output_dir,
                command="python experiments/run_ablation.py --model both",
                data_scope="unit",
                decision="reference-only",
                reason="unit test",
            )

        self.assertIn("all_groups:lr", record["metrics"])
        self.assertIn("all_groups:boosting", record["metrics"])
        self.assertAlmostEqual(record["metrics"]["all_groups:lr"]["roc_auc"], 0.62)
        self.assertAlmostEqual(record["metrics"]["all_groups:boosting"]["roc_auc"], 0.74)
        self.assertIsNone(record["metrics"]["all_groups:lr"]["brier"])
        self.assertIsNone(record["metrics"]["all_groups:boosting"]["best_threshold"])

    def test_missing_columns_include_path_and_column_names(self) -> None:
        recorder = load_recorder_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "oot"
            output_dir.mkdir()
            metrics_path = output_dir / "metrics.csv"
            pd.DataFrame(
                [
                    {
                        "model": "lr",
                        "roc_auc": 0.71,
                        "average_precision": 0.18,
                        "max_f1": 0.30,
                        "train_seconds": 3.0,
                    }
                ]
            ).to_csv(metrics_path, index=False)

            with self.assertRaises(SystemExit) as raised:
                recorder.build_record(
                    kind="oot",
                    run_id="bad-schema",
                    output_dir=output_dir,
                    command="python experiments/run_oot.py",
                    data_scope="unit",
                    decision="reject",
                    reason="unit test",
                )

        self.assertIn(str(metrics_path), str(raised.exception))
        self.assertIn("brier", str(raised.exception))
        self.assertIn("best_threshold", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
