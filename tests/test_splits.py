#!/usr/bin/env python3
"""Tests for final-report chronological split policies."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments import run_oot
from fraud_model.splits import final_oot_split, inner_tuning_split, sorted_by_time


def _write_synthetic_ieee_data(root: Path, train_rows: int = 30) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train_ids = [1000 + idx for idx in range(train_rows)]
    test_ids = [2000 + idx for idx in range(4)]

    train_rows_payload = [
        {
            "TransactionID": transaction_id,
            "TransactionDT": 10000 + idx * 100,
            "TransactionAmt": 10.0 + idx,
            "ProductCD": "W",
            "card1": 1000 + idx % 3,
            "isFraud": idx % 2,
        }
        for idx, transaction_id in enumerate(train_ids)
    ]
    test_rows_payload = [
        {
            "TransactionID": transaction_id,
            "TransactionDT": 20000 + idx * 100,
            "TransactionAmt": 20.0 + idx,
            "ProductCD": "W",
            "card1": 2000 + idx % 3,
        }
        for idx, transaction_id in enumerate(test_ids)
    ]
    pd.DataFrame(train_rows_payload).to_csv(root / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": train_ids}).to_csv(root / "train_identity.csv", index=False)
    pd.DataFrame(test_rows_payload).to_csv(root / "test_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": test_ids}).to_csv(root / "test_identity.csv", index=False)
    pd.DataFrame({"TransactionID": test_ids, "isFraud": [0.5] * len(test_ids)}).to_csv(
        root / "sample_submission.csv",
        index=False,
    )


class SplitPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({"TransactionDT": [50, 10, 40, 20, 30], "x": [5, 1, 4, 2, 3]})
        self.y = np.array([1, 0, 1, 0, 1])

    def test_final_oot_uses_last_fraction_as_validation(self) -> None:
        train_df, valid_df, y_train, y_valid = final_oot_split(self.df, self.y, valid_fraction=0.4)

        self.assertEqual(train_df["TransactionDT"].tolist(), [10, 20, 30])
        self.assertEqual(valid_df["TransactionDT"].tolist(), [40, 50])
        self.assertEqual(y_train.tolist(), [0, 0, 1])
        self.assertEqual(y_valid.tolist(), [1, 1])

    def test_inner_tuning_holds_final_window_out(self) -> None:
        train_df, tune_df, y_train, y_tune = inner_tuning_split(
            self.df,
            self.y,
            train_fraction=0.6,
            tune_fraction=0.2,
        )

        self.assertEqual(train_df["TransactionDT"].tolist(), [10, 20, 30])
        self.assertEqual(tune_df["TransactionDT"].tolist(), [40])
        self.assertEqual(y_train.tolist(), [0, 0, 1])
        self.assertEqual(y_tune.tolist(), [1])

    def test_inner_tuning_rejects_invalid_fraction_sum(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum"):
            inner_tuning_split(self.df, self.y, train_fraction=0.8, tune_fraction=0.3)

    def test_sorted_by_time_rejects_nonfinite_transaction_dt(self) -> None:
        frame = pd.DataFrame({"TransactionDT": [10, np.nan], "x": [1, 2]})

        with self.assertRaisesRegex(ValueError, "finite"):
            sorted_by_time(frame, np.array([0, 1]))

    def test_final_oot_rejects_single_row(self) -> None:
        frame = pd.DataFrame({"TransactionDT": [10]})

        with self.assertRaisesRegex(ValueError, "at least one row"):
            final_oot_split(frame, np.array([1]))


class OOTRunnerSplitPolicyTest(unittest.TestCase):
    def test_parse_args_help_handles_split_policy_percent_wording(self) -> None:
        with self.assertRaises(SystemExit) as context:
            run_oot.parse_args(["--help"])

        self.assertEqual(context.exception.code, 0)

    def test_inner_tuning_presenter_notes_describe_protected_final_window(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "outputs"
            _write_synthetic_ieee_data(data_dir)

            exit_code = run_oot.main(
                [
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--model",
                    "lr",
                    "--split-policy",
                    "inner_tuning",
                ]
            )

            self.assertEqual(exit_code, 0)
            notes = (output_dir / "presenter_notes.md").read_text(encoding="utf-8")
            self.assertIn("protected inner window", notes)
            self.assertIn("final 20% untouched", notes)
            self.assertNotIn("latest 20% for validation", notes)


if __name__ == "__main__":
    unittest.main()
