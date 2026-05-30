#!/usr/bin/env python3
"""Tests for boosting early stopping."""

from __future__ import annotations

import unittest

import numpy as np

from fraud_model.boosting import HistogramGradientBoostingClassifier


class BoostingEarlyStoppingTest(unittest.TestCase):
    def test_early_stopping_records_best_iteration(self) -> None:
        x = np.array([[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]], dtype=np.float64)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
        model = HistogramGradientBoostingClassifier(
            n_estimators=30,
            max_depth=1,
            learning_rate=0.2,
            n_bins=8,
            early_stopping_rounds=3,
            min_delta=0.0,
            seed=7,
        )

        model.fit(x, y, x_valid=x, y_valid=y)

        self.assertLessEqual(len(model.trees_), 30)
        self.assertGreaterEqual(model.best_iteration_, 1)
        self.assertIn("valid_auc", model.history_)
        self.assertIn("best_valid_auc", model.history_)

    def test_no_early_stopping_without_validation_pair(self) -> None:
        x = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        model = HistogramGradientBoostingClassifier(n_estimators=5, max_depth=1, n_bins=4, early_stopping_rounds=2)

        model.fit(x, y)

        self.assertEqual(len(model.trees_), 5)
        self.assertEqual(model.best_iteration_, 5)

    def test_invalid_early_stopping_arguments_are_rejected(self) -> None:
        x = np.array([[0.0], [1.0]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.float64)
        invalid_models = [
            HistogramGradientBoostingClassifier(early_stopping_rounds=0),
            HistogramGradientBoostingClassifier(early_stopping_rounds=-1),
            HistogramGradientBoostingClassifier(early_stopping_rounds=True),  # type: ignore[arg-type]
            HistogramGradientBoostingClassifier(early_stopping_rounds=1.0),  # type: ignore[arg-type]
            HistogramGradientBoostingClassifier(early_stopping_rounds=np.float64(1.0)),  # type: ignore[arg-type]
            HistogramGradientBoostingClassifier(early_stopping_rounds=1.5),  # type: ignore[arg-type]
            HistogramGradientBoostingClassifier(early_stopping_rounds=float("nan")),  # type: ignore[arg-type]
            HistogramGradientBoostingClassifier(early_stopping_rounds=float("inf")),  # type: ignore[arg-type]
            HistogramGradientBoostingClassifier(min_delta=-1e-9),
            HistogramGradientBoostingClassifier(min_delta=float("nan")),
            HistogramGradientBoostingClassifier(min_delta=float("inf")),
        ]

        for model in invalid_models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    model.fit(x, y)

    def test_early_stopping_truncates_to_best_iteration(self) -> None:
        x = np.array(
            [[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]],
            dtype=np.float64,
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
        model = HistogramGradientBoostingClassifier(
            n_estimators=20,
            max_depth=1,
            learning_rate=0.2,
            n_bins=8,
            early_stopping_rounds=2,
            min_delta=0.0,
            seed=7,
        )

        model.fit(x, y, x_valid=x, y_valid=y)

        self.assertTrue(model.early_stopped_)
        self.assertEqual(len(model.trees_), model.best_iteration_)
        self.assertLess(len(model.trees_), len(model.history_["valid_auc"]))
        self.assertEqual(model.best_valid_auc_, max(model.history_["valid_auc"]))
        np.testing.assert_allclose(model.history_["best_valid_auc"][-1], model.best_valid_auc_)


if __name__ == "__main__":
    unittest.main()
