#!/usr/bin/env python3
"""Tests for experiment orchestration helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_model.experiment import (
    chronological_split,
    ensure_output_dir,
    evaluate_predictions,
    write_json,
    write_metrics_csv,
    write_presenter_notes,
)


class ExperimentHelperTest(unittest.TestCase):
    def test_chronological_split_sorts_and_aligns_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "TransactionID": [10, 11, 12, 13, 14],
                "TransactionDT": [50, 10, 40, 20, 30],
                "TransactionAmt": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ).set_index(pd.Index([100, 101, 102, 103, 104]))
        y = np.array([0, 1, 0, 1, 0], dtype=np.int64)

        train_df, valid_df, y_train, y_valid = chronological_split(frame, y, valid_fraction=0.4)

        self.assertEqual(list(train_df["TransactionDT"]), [10, 20, 30])
        self.assertEqual(list(valid_df["TransactionDT"]), [40, 50])
        self.assertEqual(list(y_train), [1, 1, 0])
        self.assertEqual(list(y_valid), [0, 0])

    def test_evaluate_predictions_returns_metrics_and_named_thresholds(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        score = np.array([0.05, 0.20, 0.65, 0.95], dtype=np.float64)

        result = evaluate_predictions(y, score, prior=0.25)

        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertIn("average_precision", result)
        self.assertIn("brier_score", result)
        self.assertEqual(set(result["thresholds"]), {"prior", "max_f1", "recall_target"})
        self.assertIn("threshold", result["thresholds"]["max_f1"])

    def test_writers_create_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = ensure_output_dir(root / "nested" / "outputs")
            json_path = output_dir / "metrics.json"
            csv_path = output_dir / "metrics.csv"
            notes_path = output_dir / "presenter_notes.md"

            write_json(json_path, {"b": 2, "a": 1})
            write_metrics_csv(csv_path, [{"model": "lr", "roc_auc": 0.75}])
            write_presenter_notes(
                notes_path,
                {
                    "command": "python experiments/run_demo.py --model lr",
                    "model": "lr",
                    "metrics": {"lr": {"roc_auc": 0.75, "average_precision": 0.12, "brier_score": 0.08}},
                    "interpretation": "Small chronological validation demo.",
                },
            )

            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), {"a": 1, "b": 2})
            self.assertTrue(json_path.read_text(encoding="utf-8").startswith("{\n  \"a\""))

            write_json(json_path, {"nan_metric": float("nan"), "inf_metric": float("inf")})
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), {"inf_metric": None, "nan_metric": None})

            csv = pd.read_csv(csv_path)
            self.assertEqual(list(csv.columns), ["model", "roc_auc"])
            notes = notes_path.read_text(encoding="utf-8")
            self.assertIn("python experiments/run_demo.py", notes)
            self.assertIn("lr", notes)
            self.assertIn("ROC-AUC", notes)


if __name__ == "__main__":
    unittest.main()
