"""IEEE-CIS data discovery and merge helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
)


def discover_competition_files(root: Path) -> dict[str, Path]:
    root_path = Path(root)
    found: dict[str, Path] = {}
    required = set(REQUIRED_FILES)

    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.name in required and path.name not in found:
            found[path.name] = path

    missing = [name for name in REQUIRED_FILES if name not in found]
    if missing:
        raise FileNotFoundError(f"Missing competition files under {root_path}: {', '.join(missing)}")

    return found


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {f"id-{idx:02d}": f"id_{idx:02d}" for idx in range(1, 39)}
    return df.copy().rename(columns=renames)


def merge_transaction_identity(transaction: pd.DataFrame, identity: pd.DataFrame | None) -> pd.DataFrame:
    merged = transaction.copy()
    if identity is not None:
        identity_normalized = normalize_identity_columns(identity)
        merged = merged.merge(identity_normalized, on="TransactionID", how="left")

    if "TransactionDT" in merged.columns:
        merged = merged.sort_values("TransactionDT", kind="mergesort")

    return merged.reset_index(drop=True)


def load_train_data(data_dir: Path, nrows: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    files = discover_competition_files(data_dir)
    transaction = pd.read_csv(files["train_transaction.csv"], nrows=nrows)
    identity = pd.read_csv(files["train_identity.csv"], nrows=nrows)
    merged = merge_transaction_identity(transaction, identity)
    y = merged.pop("isFraud").to_numpy()
    return merged, y


def load_test_data(data_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    files = discover_competition_files(data_dir)
    transaction = pd.read_csv(files["test_transaction.csv"], nrows=nrows)
    identity = pd.read_csv(files["test_identity.csv"], nrows=nrows)
    return merge_transaction_identity(transaction, identity)
