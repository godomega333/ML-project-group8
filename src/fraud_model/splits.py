"""Chronological split policies for final-report experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sorted_by_time(df: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Return rows and labels sorted by TransactionDT with stable tie ordering."""
    if "TransactionDT" not in df.columns:
        raise ValueError("df must contain TransactionDT")

    target = np.asarray(y).reshape(-1)
    if target.shape[0] != len(df):
        raise ValueError("y must have the same number of rows as df")

    transaction_dt = pd.to_numeric(df["TransactionDT"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(transaction_dt)):
        raise ValueError("TransactionDT must contain finite values")

    order = np.argsort(transaction_dt, kind="mergesort")
    return df.iloc[order].reset_index(drop=True), target[order].copy()


def final_oot_split(
    df: pd.DataFrame,
    y: np.ndarray,
    valid_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Use the latest fraction of transactions as the final OOT validation window."""
    sorted_df, sorted_y = sorted_by_time(df, y)
    valid_fraction = float(valid_fraction)
    if not np.isfinite(valid_fraction) or not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be finite and within (0, 1)")
    if len(sorted_df) < 2:
        raise ValueError("final_oot_split requires at least one row in train and valid")

    valid_rows = int(np.ceil(len(sorted_df) * valid_fraction))
    valid_rows = min(max(valid_rows, 1), len(sorted_df) - 1)
    split_at = len(sorted_df) - valid_rows
    return (
        sorted_df.iloc[:split_at].reset_index(drop=True),
        sorted_df.iloc[split_at:].reset_index(drop=True),
        sorted_y[:split_at].copy(),
        sorted_y[split_at:].copy(),
    )


def inner_tuning_split(
    df: pd.DataFrame,
    y: np.ndarray,
    train_fraction: float = 0.64,
    tune_fraction: float = 0.16,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return early train and tuning windows while leaving the final window untouched."""
    sorted_df, sorted_y = sorted_by_time(df, y)
    train_fraction = float(train_fraction)
    tune_fraction = float(tune_fraction)
    if not np.isfinite(train_fraction) or not np.isfinite(tune_fraction):
        raise ValueError("train_fraction and tune_fraction must be finite")
    if train_fraction <= 0.0 or tune_fraction <= 0.0 or train_fraction + tune_fraction >= 1.0:
        raise ValueError("train_fraction and tune_fraction must be positive and sum to less than 1")

    train_end = int(np.floor(len(sorted_df) * train_fraction))
    tune_end = int(np.floor(len(sorted_df) * (train_fraction + tune_fraction)))
    if train_end <= 0 or tune_end <= train_end or tune_end >= len(sorted_df):
        raise ValueError("fractions must leave at least one row in train, tune, and final windows")

    return (
        sorted_df.iloc[:train_end].reset_index(drop=True),
        sorted_df.iloc[train_end:tune_end].reset_index(drop=True),
        sorted_y[:train_end].copy(),
        sorted_y[train_end:tune_end].copy(),
    )
