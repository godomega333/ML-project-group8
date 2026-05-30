#!/usr/bin/env python3
"""Tests for compliant submission post-processing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_model.postprocess import smooth_by_group


class PostprocessTest(unittest.TestCase):
    def test_smooth_by_group_preserves_schema_and_probability_range(self) -> None:
        submission = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3, 4],
                "isFraud": [0.1, 0.9, 0.2, 0.8],
            }
        )
        keys = pd.DataFrame({"TransactionID": [1, 2, 3, 4], "UID_D1": ["a", "a", "b", "c"]})

        smoothed, diagnostics = smooth_by_group(submission, keys, group_column="UID_D1", alpha=0.5)

        self.assertEqual(list(smoothed.columns), ["TransactionID", "isFraud"])
        self.assertTrue(((smoothed["isFraud"] >= 0.0) & (smoothed["isFraud"] <= 1.0)).all())
        np.testing.assert_allclose(smoothed["isFraud"].to_numpy(), np.array([0.3, 0.7, 0.2, 0.8]))
        self.assertEqual(diagnostics["rows"], 4)
        self.assertEqual(diagnostics["groups"], 3)
        self.assertEqual(diagnostics["changed_rows"], 2)
        self.assertEqual(diagnostics["group_column"], "UID_D1")

    def test_smooth_by_group_rejects_invalid_alpha(self) -> None:
        submission = pd.DataFrame({"TransactionID": [1], "isFraud": [0.2]})
        keys = pd.DataFrame({"TransactionID": [1], "UID_D1": ["a"]})

        with self.assertRaisesRegex(ValueError, "alpha"):
            smooth_by_group(submission, keys, group_column="UID_D1", alpha=1.5)
        with self.assertRaisesRegex(ValueError, "alpha"):
            smooth_by_group(submission, keys, group_column="UID_D1", alpha=-0.1)

    def test_smooth_by_group_rejects_invalid_probability_inputs(self) -> None:
        keys = pd.DataFrame({"TransactionID": [1, 2], "UID_D1": ["a", "a"]})

        for value in [-0.01, 1.01, "not-a-number"]:
            with self.subTest(value=value):
                submission = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.2, value]})
                with self.assertRaisesRegex(ValueError, "probabilities"):
                    smooth_by_group(submission, keys, group_column="UID_D1", alpha=0.5)

    def test_smooth_by_group_requires_one_to_one_keys(self) -> None:
        submission = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.2, 0.8]})

        duplicate_keys = pd.DataFrame({"TransactionID": [1, 1], "UID_D1": ["a", "a"]})
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            smooth_by_group(submission, duplicate_keys, group_column="UID_D1", alpha=0.5)

        missing_keys = pd.DataFrame({"TransactionID": [1], "UID_D1": ["a"]})
        with self.assertRaisesRegex(ValueError, "missing group values"):
            smooth_by_group(submission, missing_keys, group_column="UID_D1", alpha=0.5)

    def test_smooth_submission_cli_writes_output_and_diagnostics(self) -> None:
        import experiments.smooth_submission as smooth_cli

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            submission_path = root / "submission.csv"
            keys_path = root / "keys.csv"
            output_path = root / "smoothed.csv"
            diagnostics_path = root / "diagnostics.json"
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.1, 0.9]}).to_csv(submission_path, index=False)
            pd.DataFrame({"TransactionID": [1, 2], "UID_D1": ["a", "a"]}).to_csv(keys_path, index=False)

            exit_code = smooth_cli.main(
                [
                    "--submission",
                    str(submission_path),
                    "--keys",
                    str(keys_path),
                    "--group-column",
                    "UID_D1",
                    "--alpha",
                    "0.5",
                    "--output",
                    str(output_path),
                    "--diagnostics",
                    str(diagnostics_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            result = pd.read_csv(output_path)
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(list(result.columns), ["TransactionID", "isFraud"])
            self.assertTrue(((result["isFraud"] >= 0.0) & (result["isFraud"] <= 1.0)).all())
            self.assertEqual(diagnostics["rows"], 2)
            self.assertEqual(diagnostics["changed_rows"], 2)


if __name__ == "__main__":
    unittest.main()
