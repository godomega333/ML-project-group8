#!/usr/bin/env python3
"""Tests for leakage-safe preprocessing."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fraud_model.features import FeaturePipeline


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [86400, 90000, 172800, 200000],
            "TransactionAmt": [10.10, 20.20, 10.10, 99.99],
            "ProductCD": ["W", "W", "C", "S"],
            "card1": [100, 100, 200, 300],
            "addr1": [1.0, 1.0, np.nan, 4.0],
            "D1": [1.0, 1.0, np.nan, 3.0],
            "C1": [1.0, 2.0, 3.0, 4.0],
        }
    )


class FeaturePipelineTest(unittest.TestCase):
    def test_fit_transform_and_transform_have_same_columns(self) -> None:
        train = sample_frame().iloc[:3].copy()
        valid = sample_frame().iloc[3:].copy()
        pipe = FeaturePipeline()
        x_train_lr = pipe.fit_transform(train, model="lr")
        x_valid_lr = pipe.transform(valid, model="lr")
        self.assertEqual(x_train_lr.shape[1], x_valid_lr.shape[1])
        self.assertTrue(np.isfinite(x_train_lr).all())
        self.assertTrue(np.isfinite(x_valid_lr).all())

    def test_boosting_preserves_nan_before_binning(self) -> None:
        train = sample_frame().iloc[:3].copy()
        pipe = FeaturePipeline()
        x_boost = pipe.fit_transform(train, model="boosting")
        self.assertEqual(x_boost.dtype, np.float32)
        self.assertTrue(np.isnan(x_boost).any())

    def test_unknown_category_uses_frequency_fallback(self) -> None:
        train = sample_frame().iloc[:3].copy()
        valid = sample_frame().iloc[3:].copy()
        valid.loc[:, "ProductCD"] = "UNKNOWN_CATEGORY"
        pipe = FeaturePipeline()
        pipe.fit(train)
        transformed = pipe.transform_frame(valid)
        self.assertEqual(float(transformed["ProductCD"].iloc[0]), 1.0)

    def test_lr_handles_nullable_integer_and_boolean_columns(self) -> None:
        train = sample_frame().copy()
        train["nullable_int"] = pd.Series([1, 2, pd.NA, pd.NA], dtype="Int64")
        train["nullable_bool"] = pd.Series([True, False, pd.NA, True], dtype="boolean")
        valid = sample_frame().iloc[:2].copy()
        valid["nullable_int"] = pd.Series([pd.NA, 3], dtype="Int64")
        valid["nullable_bool"] = pd.Series([pd.NA, False], dtype="boolean")

        pipe = FeaturePipeline()
        x_train_lr = pipe.fit_transform(train, model="lr")
        x_valid_lr = pipe.transform(valid, model="lr")

        self.assertTrue(np.isfinite(x_train_lr).all())
        self.assertTrue(np.isfinite(x_valid_lr).all())

    def test_baseline_profile_keeps_existing_feature_contract(self) -> None:
        train = sample_frame().iloc[:3].copy()
        baseline = FeaturePipeline(feature_profile="baseline")
        implicit = FeaturePipeline()

        baseline.fit(train)
        implicit.fit(train)

        self.assertEqual(baseline.feature_columns, implicit.feature_columns)
        np.testing.assert_allclose(
            baseline.transform(train, model="lr"),
            implicit.transform(train, model="lr"),
            rtol=0.0,
            atol=0.0,
        )

    def test_uid_d_profile_adds_d_normalized_and_uid_frequency_features(self) -> None:
        train = sample_frame().iloc[:3].copy()
        valid = sample_frame().iloc[3:].copy()
        pipe = FeaturePipeline(feature_profile="uid_d")

        pipe.fit(train)
        transformed = pipe.transform_frame(valid)
        raw_keys = pipe.transform_keys(valid)

        self.assertIn("D1n", transformed.columns)
        self.assertIn("UID_D1", transformed.columns)
        self.assertIn("UID_Email", transformed.columns)
        self.assertIn("card1_addr1", transformed.columns)
        self.assertIn("UID_D1", raw_keys.columns)
        self.assertIn("UID_Email", raw_keys.columns)
        self.assertEqual(float(transformed["UID_D1"].iloc[0]), 1.0)
        self.assertEqual(float(transformed["UID_Email"].iloc[0]), 1.0)
        self.assertEqual(str(raw_keys["UID_D1"].iloc[0]), "300_4.0_-1.0")
        self.assertEqual(pipe.transform(train, model="lr").shape[1], pipe.transform(valid, model="lr").shape[1])

    def test_uid_agg_profile_adds_train_fit_aggregates_with_unseen_fallback(self) -> None:
        train = sample_frame().copy()
        valid = sample_frame().iloc[[3]].copy()
        valid.loc[:, "card1"] = 9999
        valid.loc[:, "addr1"] = 9999
        valid.loc[:, "D1"] = 9999
        pipe = FeaturePipeline(feature_profile="uid_agg")

        pipe.fit(train)
        transformed = pipe.transform_frame(valid)

        for column in ["UID_Count", "UID_TransactionAmt_Mean", "UID_TransactionAmt_Std", "UID_C1_Mean"]:
            self.assertIn(column, transformed.columns)
            self.assertTrue(np.isfinite(float(transformed[column].iloc[0])))
        self.assertEqual(float(transformed["UID_Count"].iloc[0]), 0.0)

    def test_rejects_unknown_feature_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature_profile"):
            FeaturePipeline(feature_profile="unknown").fit(sample_frame())


if __name__ == "__main__":
    unittest.main()
