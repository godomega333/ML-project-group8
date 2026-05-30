#!/usr/bin/env python3
"""Tests for month-gap validation runner."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.test_task10_runners import _write_synthetic_ieee_data


class MonthGapValidationTest(unittest.TestCase):
    def test_make_month_gap_split_skips_middle_period(self) -> None:
        runner = importlib.import_module("experiments.run_month_gap")
        df = pd.DataFrame({"TransactionDT": np.arange(100, 1000, 100), "TransactionID": np.arange(9)})
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])

        train_df, valid_df, y_train, y_valid = runner.month_gap_split(
            df,
            y,
            train_fraction=0.50,
            gap_fraction=0.25,
            valid_fraction=0.25,
        )

        self.assertEqual(len(train_df), 4)
        self.assertEqual(len(valid_df), 3)
        self.assertLess(train_df["TransactionDT"].max(), valid_df["TransactionDT"].min())
        np.testing.assert_array_equal(y_train, np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(y_valid, np.array([0, 1, 0]))

    def test_month_gap_split_rejects_nonfinite_transaction_dt(self) -> None:
        runner = importlib.import_module("experiments.run_month_gap")
        df = pd.DataFrame({"TransactionDT": [100.0, np.nan, 300.0], "TransactionID": [1, 2, 3]})
        y = np.array([0, 1, 0])

        with self.assertRaisesRegex(ValueError, "TransactionDT must contain finite values"):
            runner.month_gap_split(
                df,
                y,
                train_fraction=0.5,
                gap_fraction=0.0,
                valid_fraction=0.3,
            )

    def test_runner_writes_metrics_and_config(self) -> None:
        runner = importlib.import_module("experiments.run_month_gap")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            output_dir = root / "month-gap"
            _write_synthetic_ieee_data(data_dir, train_rows=40, test_rows=6)

            exit_code = runner.main(
                [
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--model",
                    "lr",
                    "--feature-profile",
                    "uid_d",
                    "--train-fraction",
                    "0.5",
                    "--gap-fraction",
                    "0.25",
                    "--valid-fraction",
                    "0.25",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "month_gap_metrics.csv").is_file())
            self.assertTrue((output_dir / "config.json").is_file())
            metrics = pd.read_csv(output_dir / "month_gap_metrics.csv")
            self.assertEqual(
                list(metrics.columns),
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
            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["feature_profile"], "uid_d")


if __name__ == "__main__":
    unittest.main()
