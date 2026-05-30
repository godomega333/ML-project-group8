#!/usr/bin/env python3
"""Tests for from-scratch calibration and threshold helpers."""

from __future__ import annotations

import unittest

import numpy as np

from fraud_model.calibration import PlattCalibrator, choose_thresholds


class CalibrationTest(unittest.TestCase):
    def test_platt_calibrator_outputs_probabilities(self) -> None:
        logits = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        cal = PlattCalibrator(learning_rate=0.1, epochs=200)
        cal.fit(logits, y)
        p = cal.predict_proba(logits)
        self.assertEqual(p.shape, (4,))
        self.assertTrue(np.all((0.0 <= p) & (p <= 1.0)))
        self.assertLess(p[0], p[-1])

    def test_platt_calibrator_handles_large_score_magnitudes(self) -> None:
        logits = np.array([-3000.0, -1000.0, 1000.0, 3000.0], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        cal = PlattCalibrator(learning_rate=0.1, epochs=200)
        cal.fit(logits, y)
        self.assertAlmostEqual(cal.score_mean_, 0.0)
        self.assertGreater(cal.score_std_, 1.0)
        p = cal.predict_proba(logits)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all((0.0 <= p) & (p <= 1.0)))
        self.assertLess(p[0], p[-1])

    def test_platt_calibrator_rejects_nonfinite_epochs_cleanly(self) -> None:
        logits = np.array([-1.0, 1.0], dtype=np.float64)
        y = np.array([0, 1], dtype=np.float64)
        cal = PlattCalibrator(learning_rate=0.1, epochs=np.inf)
        with self.assertRaisesRegex(ValueError, "epochs"):
            cal.fit(logits, y)

    def test_choose_thresholds_returns_named_policies(self) -> None:
        y = np.array([0, 1, 1, 0], dtype=np.int64)
        p = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float64)
        result = choose_thresholds(y, p, prior=0.25, recall_target=0.5)
        self.assertIn("prior", result)
        self.assertIn("max_f1", result)
        self.assertIn("recall_target", result)
        self.assertGreaterEqual(result["recall_target"]["recall"], 0.5)

    def test_choose_thresholds_uses_score_derived_candidates(self) -> None:
        y = np.array([0, 1], dtype=np.int64)
        p = np.array([0.004, 0.006], dtype=np.float64)
        result = choose_thresholds(y, p, prior=0.25)
        self.assertAlmostEqual(result["max_f1"]["f1"], 1.0)
        self.assertGreater(result["max_f1"]["threshold"], 0.0)
        self.assertLessEqual(result["max_f1"]["threshold"], 0.006)


if __name__ == "__main__":
    unittest.main()
