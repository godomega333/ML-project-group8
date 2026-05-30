#!/usr/bin/env python3
"""Focused Task 4 runner config wiring tests."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.run_oot import fit_model_on_matrices
from fraud_model.configs import ModelConfig, get_model_config


class Task4RunnerConfigTest(unittest.TestCase):
    def _tiny_boosting_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
        )

    def test_numeric_positive_weight_reaches_boosting_estimator(self) -> None:
        config = ModelConfig(
            config_id="unit_boosting_numeric_weight",
            model_family="boosting",
            feature_profile="baseline",
            params={
                "n_estimators": 1,
                "max_depth": 1,
                "learning_rate": 0.1,
                "n_bins": 8,
                "l2": 1.0,
                "gamma": 0.0,
                "min_child_weight": 1.0,
                "subsample": 1.0,
                "colsample": 1.0,
            },
            positive_weight=3.5,
        )

        result = fit_model_on_matrices(
            model_name="boosting",
            train_x=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
            y_train=np.array([0, 1, 0, 1], dtype=np.int8),
            seed=7,
            model_config=config,
        )

        self.assertEqual(result.estimator.positive_weight, 3.5)

    def test_boosting_config_d_registry_params_are_runnable(self) -> None:
        train_x, y_train = self._tiny_boosting_matrix()
        result = fit_model_on_matrices(
            model_name="boosting",
            train_x=train_x,
            y_train=y_train,
            valid_x=np.array([[0.0, 0.0], [2.0, 1.0]], dtype=np.float32),
            y_valid=np.array([0, 1], dtype=np.int8),
            seed=11,
            model_config=get_model_config("boosting_config_d"),
        )

        self.assertEqual(result.estimator.early_stopping_rounds, 25)
        self.assertEqual(result.estimator.min_delta, 0.0001)
        self.assertTrue(result.estimator.early_stopping_enabled_)
        self.assertGreater(len(result.history["valid_auc"]), 0)
        self.assertGreater(len(result.history["train_auc"]), 0)

    def test_boosting_config_d_without_validation_keeps_params_but_disables_stopping(self) -> None:
        train_x, y_train = self._tiny_boosting_matrix()
        result = fit_model_on_matrices(
            model_name="boosting",
            train_x=train_x,
            y_train=y_train,
            seed=11,
            model_config=get_model_config("boosting_config_d"),
        )

        self.assertEqual(result.estimator.early_stopping_rounds, 25)
        self.assertEqual(result.estimator.min_delta, 0.0001)
        self.assertFalse(result.estimator.early_stopping_enabled_)
        self.assertFalse(result.estimator.early_stopped_)
        self.assertEqual(len(result.history["train_auc"]), get_model_config("boosting_config_d").params["n_estimators"])
        self.assertEqual(result.history["valid_auc"], [])


if __name__ == "__main__":
    unittest.main()
