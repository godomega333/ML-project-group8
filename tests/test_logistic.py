#!/usr/bin/env python3
"""Tests for from-scratch matrix logistic regression."""

from __future__ import annotations

import unittest

import numpy as np

from fraud_model.logistic import MatrixLogisticRegression


class MatrixLogisticRegressionTest(unittest.TestCase):
    def test_predict_proba_shape_and_range(self) -> None:
        x = np.array([[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=np.float32)
        model = MatrixLogisticRegression(learning_rate=0.2, l2=0.0, epochs=80, seed=7)
        history = model.fit(x, y, x, y)
        p = model.predict_proba(x)
        self.assertEqual(p.shape, (4,))
        self.assertTrue(np.all((0.0 <= p) & (p <= 1.0)))
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])

    def test_deterministic_seed(self) -> None:
        x = np.array([[1.0, -1.0], [1.0, 1.0], [1.0, 2.0], [1.0, -2.0]], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.float32)
        a = MatrixLogisticRegression(learning_rate=0.1, epochs=20, seed=123).fit(x, y)
        b = MatrixLogisticRegression(learning_rate=0.1, epochs=20, seed=123).fit(x, y)
        np.testing.assert_allclose(a["train_loss"], b["train_loss"])

    def test_mini_batch_deterministic_seed(self) -> None:
        x = np.array(
            [[1.0, -2.0], [1.0, -1.0], [1.0, -0.5], [1.0, 0.5], [1.0, 1.0], [1.0, 2.0]],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        model_a = MatrixLogisticRegression(learning_rate=0.1, epochs=25, batch_size=2, seed=321)
        model_b = MatrixLogisticRegression(learning_rate=0.1, epochs=25, batch_size=2, seed=321)
        history_a = model_a.fit(x, y)
        history_b = model_b.fit(x, y)
        np.testing.assert_allclose(history_a["train_loss"], history_b["train_loss"])
        np.testing.assert_allclose(model_a.coef_, model_b.coef_)

    def test_invalid_class_weights_raise_before_training(self) -> None:
        x = np.array([[1.0, -1.0], [1.0, 1.0], [1.0, 2.0], [1.0, -2.0]], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.float32)
        for weight in [0.0, -1.0, np.nan, np.inf, -np.inf]:
            with self.subTest(weight=weight):
                model = MatrixLogisticRegression(class_weight={1: weight})
                with self.assertRaisesRegex(ValueError, "class_weight values must be positive"):
                    model.fit(x, y)
                self.assertIsNone(model.coef_)


if __name__ == "__main__":
    unittest.main()
