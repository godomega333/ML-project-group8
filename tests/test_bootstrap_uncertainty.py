from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments import run_bootstrap_uncertainty
from fraud_model.bootstrap import paired_bootstrap_deltas


EXPECTED_DELTA_KEYS = [
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


class PairedBootstrapDeltasTest(unittest.TestCase):
    def test_returns_expected_keys_in_order_with_ordered_intervals(self) -> None:
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        anchor_score = [0.10, 0.70, 0.30, 0.60, 0.20, 0.75, 0.40, 0.55]
        candidate_score = [0.05, 0.85, 0.20, 0.80, 0.25, 0.90, 0.35, 0.65]

        summary = paired_bootstrap_deltas(y_true, anchor_score, candidate_score, repeats=50, seed=7)

        self.assertEqual(list(summary.keys()), EXPECTED_DELTA_KEYS)
        for metric in ["auc", "ap", "brier"]:
            self.assertGreaterEqual(summary[f"{metric}_delta_high"], summary[f"{metric}_delta_low"])

    def test_mismatched_lengths_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            paired_bootstrap_deltas([0, 1, 0], [0.1, 0.9], [0.2, 0.8, 0.3], repeats=50)

    def test_single_class_labels_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "both classes"):
            paired_bootstrap_deltas([1, 1, 1], [0.6, 0.7, 0.8], [0.7, 0.8, 0.9], repeats=50)


class BootstrapUncertaintyCliTest(unittest.TestCase):
    def test_cli_rejects_repeats_below_final_evidence_floor_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions = root / "predictions.csv"
            output = root / "bootstrap.csv"
            pd.DataFrame(
                {
                    "isFraud": [0, 1, 0, 1],
                    "anchor_score": [0.1, 0.8, 0.3, 0.7],
                    "candidate_score": [0.2, 0.9, 0.25, 0.85],
                }
            ).to_csv(predictions, index=False)

            with self.assertRaises(SystemExit) as exc:
                run_bootstrap_uncertainty.main(
                    [
                        "--predictions",
                        str(predictions),
                        "--anchor-column",
                        "anchor_score",
                        "--candidate-column",
                        "candidate_score",
                        "--output",
                        str(output),
                        "--repeats",
                        "999",
                    ]
                )

            self.assertNotEqual(exc.exception.code, 0)
            self.assertFalse(output.exists())

    def test_cli_writes_candidate_delta_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions = root / "predictions.csv"
            output = root / "nested" / "bootstrap.csv"
            pd.DataFrame(
                {
                    "isFraud": [0, 1, 0, 1, 0, 1],
                    "anchor_score": [0.1, 0.8, 0.3, 0.7, 0.2, 0.9],
                    "candidate_a": [0.2, 0.9, 0.25, 0.85, 0.15, 0.95],
                    "candidate_b": [0.4, 0.6, 0.35, 0.75, 0.30, 0.80],
                }
            ).to_csv(predictions, index=False)

            exit_code = run_bootstrap_uncertainty.main(
                [
                    "--predictions",
                    str(predictions),
                    "--anchor-column",
                    "anchor_score",
                    "--candidate-column",
                    "candidate_a",
                    "--candidate-column",
                    "candidate_b",
                    "--output",
                    str(output),
                    "--repeats",
                    "1000",
                    "--seed",
                    "11",
                ]
            )

            self.assertEqual(exit_code, 0)
            rows = pd.read_csv(output)
            self.assertEqual(rows.shape[0], 2)
            self.assertEqual(
                list(rows.columns),
                [
                    "anchor",
                    "candidate",
                    "repeats",
                    "seed",
                    "n_rows",
                    "n_pos",
                    "n_neg",
                    *EXPECTED_DELTA_KEYS,
                ],
            )
            self.assertEqual(rows["anchor"].tolist(), ["anchor_score", "anchor_score"])
            self.assertEqual(rows["candidate"].tolist(), ["candidate_a", "candidate_b"])
            self.assertEqual(rows["repeats"].tolist(), [1000, 1000])
            self.assertEqual(rows["seed"].tolist(), [11, 11])
            self.assertEqual(rows["n_rows"].tolist(), [6, 6])
            self.assertEqual(rows["n_pos"].tolist(), [3, 3])
            self.assertEqual(rows["n_neg"].tolist(), [3, 3])


if __name__ == "__main__":
    unittest.main()
