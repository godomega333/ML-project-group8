#!/usr/bin/env python3
"""Validate IEEE-CIS Kaggle submission files without using ML libraries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = ["TransactionID", "isFraud"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an IEEE-CIS Fraud Detection submission CSV.")
    parser.add_argument("--submission", required=True, type=Path, help="Path to submission CSV.")
    parser.add_argument("--sample", required=True, type=Path, help="Path to Kaggle sample_submission.csv.")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def validate_csv_shape(path: Path, label: str) -> None:
    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != EXPECTED_COLUMNS:
                raise SystemExit(f"{label} columns must be {EXPECTED_COLUMNS}, got {header}")

            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(EXPECTED_COLUMNS):
                    raise SystemExit(
                        f"{label} row {row_number} has {len(row)} fields; "
                        f"expected {len(EXPECTED_COLUMNS)}"
                    )
    except StopIteration:
        raise SystemExit(f"{label} could not be read as CSV: No columns to parse from file")
    except (csv.Error, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(f"{label} could not be read as CSV: {exc}") from exc


def read_csv_checked(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(f"{label} could not be read as CSV: {exc}") from exc


def validate_submission(submission_path: Path, sample_path: Path) -> None:
    require_file(submission_path, "submission")
    require_file(sample_path, "sample")

    validate_csv_shape(submission_path, "submission")
    validate_csv_shape(sample_path, "sample")

    submission = read_csv_checked(submission_path, "submission")
    sample = read_csv_checked(sample_path, "sample")

    if list(submission.columns) != EXPECTED_COLUMNS:
        raise SystemExit(
            f"submission columns must be {EXPECTED_COLUMNS}, got {list(submission.columns)}"
        )

    if list(sample.columns) != EXPECTED_COLUMNS:
        raise SystemExit(f"sample columns must be {EXPECTED_COLUMNS}, got {list(sample.columns)}")

    if len(sample) == 0:
        raise SystemExit("sample must contain at least one row")

    if sample["TransactionID"].isna().any():
        raise SystemExit("sample TransactionID contains missing values")

    if sample["TransactionID"].duplicated().any():
        raise SystemExit("sample TransactionID values must be unique")

    if len(submission) == 0:
        raise SystemExit("submission must contain at least one row")

    if len(submission) != len(sample):
        raise SystemExit(f"row count mismatch: submission={len(submission)} sample={len(sample)}")

    if submission["TransactionID"].isna().any():
        raise SystemExit("submission TransactionID contains missing values")

    if not submission["TransactionID"].equals(sample["TransactionID"]):
        raise SystemExit("TransactionID values must exactly match sample_submission.csv order")

    probabilities = pd.to_numeric(submission["isFraud"], errors="coerce")
    if probabilities.isna().any():
        raise SystemExit("isFraud contains non-numeric values")

    if not ((0.0 <= probabilities) & (probabilities <= 1.0)).all():
        raise SystemExit("isFraud probabilities must be within [0, 1]")

    print(
        "submission OK:",
        f"rows={len(submission)}",
        f"min={probabilities.min():.6f}",
        f"max={probabilities.max():.6f}",
    )


def main() -> None:
    args = parse_args()
    validate_submission(args.submission, args.sample)


if __name__ == "__main__":
    main()
