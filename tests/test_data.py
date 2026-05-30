#!/usr/bin/env python3
"""Tests for IEEE-CIS data loading helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fraud_model.data import (
    REQUIRED_FILES,
    discover_competition_files,
    merge_transaction_identity,
    normalize_identity_columns,
)


class DataHelperTest(unittest.TestCase):
    def test_discover_competition_files_nested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "competitions" / "ieee-fraud-detection"
            nested.mkdir(parents=True)
            for name in REQUIRED_FILES:
                (nested / name).write_text("x\n", encoding="utf-8")
            found = discover_competition_files(root)
        self.assertEqual(set(found), set(REQUIRED_FILES))

    def test_normalize_identity_columns(self) -> None:
        df = pd.DataFrame({"TransactionID": [1], "id-01": [2.0], "id_02": [3.0]})
        normalized = normalize_identity_columns(df)
        self.assertIn("id_01", normalized.columns)
        self.assertIn("id_02", normalized.columns)
        self.assertNotIn("id-01", normalized.columns)

    def test_merge_transaction_identity_left_join_and_sort(self) -> None:
        tx = pd.DataFrame({"TransactionID": [2, 1], "TransactionDT": [20, 10], "TransactionAmt": [5.0, 9.0]})
        ident = pd.DataFrame({"TransactionID": [1], "id-01": [7.0]})
        merged = merge_transaction_identity(tx, ident)
        self.assertEqual(list(merged["TransactionID"]), [1, 2])
        self.assertIn("id_01", merged.columns)
        self.assertTrue(pd.isna(merged.loc[merged["TransactionID"] == 2, "id_01"]).iloc[0])


if __name__ == "__main__":
    unittest.main()
