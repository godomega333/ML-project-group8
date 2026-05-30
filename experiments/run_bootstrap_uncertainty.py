#!/usr/bin/env python3
"""Write paired bootstrap uncertainty artifacts for final-report comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_model.bootstrap import paired_bootstrap_deltas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap uncertainty deltas.")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV with isFraud and model score columns.")
    parser.add_argument("--anchor-column", required=True, help="Anchor model score column.")
    parser.add_argument(
        "--candidate-column",
        action="append",
        required=True,
        help="Candidate model score column. Repeat for multiple candidates.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV for bootstrap delta summaries.")
    parser.add_argument("--repeats", type=int, default=1000, help="Bootstrap repeat count; final evidence requires >=1000.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for paired bootstrap sampling.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1000:
        raise SystemExit("--repeats must be at least 1000 for final-report evidence; lower values are smoke/debug only.")

    predictions = pd.read_csv(args.predictions)
    required_columns = ["isFraud", args.anchor_column, *args.candidate_column]
    missing = [column for column in required_columns if column not in predictions.columns]
    if missing:
        raise ValueError(f"predictions CSV is missing required column(s): {', '.join(missing)}")

    y_true = predictions["isFraud"].to_numpy()
    anchor_score = predictions[args.anchor_column].to_numpy()
    n_pos = int((predictions["isFraud"] == 1).sum())
    n_rows = int(predictions.shape[0])
    n_neg = n_rows - n_pos

    rows = []
    for candidate_column in args.candidate_column:
        summary = paired_bootstrap_deltas(
            y_true,
            anchor_score,
            predictions[candidate_column].to_numpy(),
            repeats=args.repeats,
            seed=args.seed,
        )
        rows.append(
            {
                "anchor": args.anchor_column,
                "candidate": candidate_column,
                "repeats": args.repeats,
                "seed": args.seed,
                "n_rows": n_rows,
                "n_pos": n_pos,
                "n_neg": n_neg,
                **summary,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} bootstrap comparison(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
