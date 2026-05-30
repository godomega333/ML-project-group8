#!/usr/bin/env python3
"""Tests for from-scratch binary classification metrics."""

from __future__ import annotations

import unittest

import numpy as np

from fraud_model.metrics import (
    average_precision,
    brier_score,
    calibration_table,
    confusion_at_threshold,
    roc_auc,
    threshold_summary,
)


class MetricsTest(unittest.TestCase):
    def test_roc_auc_perfect_and_worst(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        self.assertAlmostEqual(roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_roc_auc_ties_use_average_ranks(self) -> None:
        y = np.array([0, 1, 0, 1], dtype=np.float64)
        score = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        self.assertAlmostEqual(roc_auc(y, score), 0.5)

    def test_average_precision_known_order(self) -> None:
        y = np.array([1, 0, 1, 0], dtype=np.float64)
        score = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
        self.assertAlmostEqual(average_precision(y, score), (1.0 + 2.0 / 3.0) / 2.0)

    def test_confusion_and_threshold_summary(self) -> None:
        y = np.array([0, 1, 1, 0], dtype=np.int64)
        score = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float64)
        cm = confusion_at_threshold(y, score, 0.5)
        self.assertEqual(cm["tp"], 1)
        self.assertEqual(cm["fp"], 1)
        self.assertEqual(cm["tn"], 1)
        self.assertEqual(cm["fn"], 1)
        summary = threshold_summary(y, score, thresholds=np.array([0.3, 0.5, 0.8]))
        self.assertEqual(list(summary.columns), ["threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"])
        self.assertEqual(len(summary), 3)

    def test_brier_and_calibration_table(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        score = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float64)
        self.assertAlmostEqual(brier_score(y, score), np.mean((score - y) ** 2))
        table = calibration_table(y, score, bins=2)
        self.assertEqual(list(table.columns), ["bin", "count", "mean_pred", "frac_positive"])
        self.assertEqual(int(table["count"].sum()), 4)


if __name__ == "__main__":
    unittest.main()
