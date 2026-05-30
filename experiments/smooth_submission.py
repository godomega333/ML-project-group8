#!/usr/bin/env python3
"""Smooth a Kaggle submission by pseudo-UID groups without labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_model.experiment import write_json
from fraud_model.postprocess import smooth_by_group


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth submission probabilities by pseudo-UID groups.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--group-column", default="UID_D1")
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    submission = pd.read_csv(args.submission)
    keys = pd.read_csv(args.keys)
    smoothed, diagnostics = smooth_by_group(
        submission=submission,
        keys=keys,
        group_column=args.group_column,
        alpha=args.alpha,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    smoothed.to_csv(args.output, index=False)
    write_json(args.diagnostics, diagnostics)
    print(f"Smoothed rows: {len(smoothed)}")
    print(f"Output path: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
