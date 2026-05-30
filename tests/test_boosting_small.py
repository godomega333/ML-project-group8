#!/usr/bin/env python3
"""Small deterministic tests for sparsity-aware histogram boosting."""

from __future__ import annotations

import unittest

import numpy as np

from fraud_model.boosting import HistogramGradientBoostingClassifier


class HistogramBoostingTest(unittest.TestCase):
    def test_predict_proba_shape_and_range(self) -> None:
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=np.float32)
        model = HistogramGradientBoostingClassifier(
            n_estimators=5,
            max_depth=1,
            learning_rate=0.5,
            n_bins=8,
            min_child_weight=0.0,
            seed=1,
        )
        model.fit(x, y, x, y)
        p = model.predict_proba(x)
        self.assertEqual(p.shape, (4,))
        self.assertTrue(np.all((0.0 <= p) & (p <= 1.0)))
        self.assertGreater(p[-1], p[0])

    def test_nan_default_routing_is_recorded(self) -> None:
        x = np.array([[np.nan], [np.nan], [0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
        model = HistogramGradientBoostingClassifier(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.5,
            n_bins=8,
            min_child_weight=0.0,
            seed=2,
        )
        model.fit(x, y)
        root = model.trees_[0]
        self.assertIn(root.nan_go_left, (True, False))
        self.assertGreaterEqual(root.feature_index, 0)

    def test_deterministic_seed(self) -> None:
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=np.float32)
        a = HistogramGradientBoostingClassifier(n_estimators=3, max_depth=1, seed=9).fit(x, y).predict_proba(x)
        b = HistogramGradientBoostingClassifier(n_estimators=3, max_depth=1, seed=9).fit(x, y).predict_proba(x)
        np.testing.assert_allclose(a, b)

    def test_sparse_binary_values_keep_separate_bins(self) -> None:
        x = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
        y = np.array([0, 0, 0, 1], dtype=np.float32)
        model = HistogramGradientBoostingClassifier(
            n_estimators=1,
            max_depth=1,
            n_bins=8,
            min_child_weight=0.0,
            seed=4,
        )
        model.fit(x, y)
        bins = model._bin_matrix(np.array([[0.0], [1.0]], dtype=np.float32)).reshape(-1)
        self.assertNotEqual(int(bins[0]), int(bins[1]))

    def test_min_child_weight_uses_hessian_not_row_count(self) -> None:
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=np.float32)
        model = HistogramGradientBoostingClassifier(
            n_estimators=1,
            max_depth=1,
            n_bins=8,
            min_child_weight=0.6,
            seed=5,
        )
        model.fit(x, y)
        root = model.trees_[0]
        self.assertIsNotNone(root.value)
        self.assertEqual(root.feature_index, -1)


if __name__ == "__main__":
    unittest.main()
