#!/usr/bin/env python3
"""Convert experiment artifacts into modeling ledger records."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = ["roc_auc", "average_precision", "brier", "max_f1", "best_threshold", "train_seconds"]
OOT_REQUIRED_COLUMNS = ["model", *METRIC_COLUMNS]
ROLLING_REQUIRED_COLUMNS = ["model", *METRIC_COLUMNS]
MONTH_GAP_REQUIRED_COLUMNS = ["model", *METRIC_COLUMNS]
ABLATION_REQUIRED_COLUMNS = ["run", "model", "roc_auc", "average_precision", "max_f1", "train_seconds"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an experiment output directory in the modeling ledger.")
    parser.add_argument("--kind", choices=["oot", "rolling", "month-gap", "ablation", "decision"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--data-scope", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--ledger", type=Path, default=Path("docs/modeling-results-ledger.md"))
    parser.add_argument("--record-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_record(
        kind=args.kind,
        run_id=args.run_id,
        output_dir=args.output_dir,
        command=args.command,
        data_scope=args.data_scope,
        decision=args.decision,
        reason=args.reason,
    )
    if args.record_json is not None:
        args.record_json.parent.mkdir(parents=True, exist_ok=True)
        args.record_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from append_modeling_ledger import append_record

    append_record(args.ledger, record)
    return 0


def build_record(
    kind: str,
    run_id: str,
    output_dir: Path,
    command: str,
    data_scope: str,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    metrics = metrics_for_kind(kind, output_dir)
    return {
        "kind": "experiment",
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "data_scope": data_scope,
        "output_dir": str(output_dir),
        "metrics": metrics,
        "decision": decision,
        "reason": reason,
    }


def metrics_for_kind(kind: str, output_dir: Path) -> dict[str, dict[str, float | None]]:
    if kind == "oot":
        return oot_metrics(output_dir / "metrics.csv")
    if kind == "rolling":
        return rolling_metrics(output_dir / "rolling_metrics.csv")
    if kind == "month-gap":
        return month_gap_metrics(output_dir / "month_gap_metrics.csv")
    if kind == "ablation":
        return ablation_metrics(output_dir / "ablation_summary.csv")
    if kind == "decision":
        return {"summary": {}}
    raise SystemExit(f"Unsupported kind: {kind}")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing experiment artifact: {path}")


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {path}: {', '.join(missing)}")


def oot_metrics(path: Path) -> dict[str, dict[str, float | None]]:
    require_file(path)
    df = pd.read_csv(path)
    require_columns(df, path, OOT_REQUIRED_COLUMNS)
    rows: dict[str, dict[str, float | None]] = {}
    for _, row in df.iterrows():
        rows[str(row["model"])] = metric_payload(row)
    return rows


def rolling_metrics(path: Path) -> dict[str, dict[str, float | None]]:
    require_file(path)
    df = pd.read_csv(path)
    require_columns(df, path, ROLLING_REQUIRED_COLUMNS)
    return grouped_metric_means(df)


def month_gap_metrics(path: Path) -> dict[str, dict[str, float | None]]:
    require_file(path)
    df = pd.read_csv(path)
    require_columns(df, path, MONTH_GAP_REQUIRED_COLUMNS)
    return grouped_metric_means(df)


def grouped_metric_means(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    for model_name, group in df.groupby("model", sort=False):
        rows[str(model_name)] = {
            "roc_auc": float(group["roc_auc"].mean()),
            "average_precision": float(group["average_precision"].mean()),
            "brier": float(group["brier"].mean()),
            "max_f1": float(group["max_f1"].mean()),
            "best_threshold": float(group["best_threshold"].mean()),
            "train_seconds": float(group["train_seconds"].sum()),
        }
    return rows


def ablation_metrics(path: Path) -> dict[str, dict[str, float | None]]:
    require_file(path)
    df = pd.read_csv(path)
    require_columns(df, path, ABLATION_REQUIRED_COLUMNS)
    rows: dict[str, dict[str, float | None]] = {}
    for _, row in df.iterrows():
        rows[f"{row['run']}:{row['model']}"] = {
            "roc_auc": float(row["roc_auc"]),
            "average_precision": float(row["average_precision"]),
            "brier": None,
            "max_f1": float(row["max_f1"]),
            "best_threshold": None,
            "train_seconds": float(row["train_seconds"]),
        }
    return rows


def metric_payload(row: pd.Series) -> dict[str, float | None]:
    return {
        "roc_auc": float(row["roc_auc"]),
        "average_precision": float(row["average_precision"]),
        "brier": float(row["brier"]),
        "max_f1": float(row["max_f1"]),
        "best_threshold": float(row["best_threshold"]),
        "train_seconds": float(row["train_seconds"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
