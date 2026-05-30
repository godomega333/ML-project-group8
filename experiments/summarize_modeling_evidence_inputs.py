#!/usr/bin/env python3
"""Summarize manifest-backed LR and boosting evidence inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manifest-backed LR and boosting evidence inputs.")
    parser.add_argument("--lr-audit-root", type=Path, required=True)
    parser.add_argument("--boosting-search-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lr = summarize_root(args.lr_audit_root, expected_family="lr")
    boosting = summarize_root(args.boosting_search_root, expected_family="boosting")
    if lr.empty:
        raise SystemExit("lr_audit_summary is empty")
    if boosting.empty:
        raise SystemExit("boosting_search_summary is empty")
    reject_sample_rows(lr, "lr_audit_summary")
    reject_sample_rows(boosting, "boosting_search_summary")
    lr.to_csv(args.output_dir / "lr_audit_summary.csv", index=False)
    boosting.to_csv(args.output_dir / "boosting_search_summary.csv", index=False)
    print(f"Summarized LR={len(lr)} boosting={len(boosting)}")
    return 0


def summarize_root(root: Path, *, expected_family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        family = manifest.get("model_family")
        if family != expected_family:
            raise SystemExit(f"{manifest_path} model_family is {family}, expected {expected_family}")
        metrics_path = manifest_path.parent / "metrics.csv"
        if not metrics_path.exists():
            raise SystemExit(f"Missing metrics file for {manifest_path}: {metrics_path}")
        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            raise SystemExit(f"Metrics file is empty: {metrics_path}")
        metric_row = metrics.iloc[0].to_dict()
        config = manifest.get("config") or {}
        rows.append(
            {
                "candidate_id": manifest.get("candidate_id"),
                "config_id": manifest.get("config_id"),
                "config_hash": manifest.get("config_hash"),
                "model_family": family,
                "feature_profile": manifest.get("feature_profile"),
                "split_policy": manifest.get("split_policy"),
                "source_run_id": manifest.get("source_run_id"),
                "source_run_dir": str(manifest_path.parent),
                "manifest_path": str(manifest_path),
                "sample_rows": manifest.get("sample_rows"),
                "params_json": json.dumps(config.get("params") or {}, sort_keys=True),
                "positive_weight": config.get("positive_weight"),
                "class_weight_alpha": config.get("class_weight_alpha"),
                "roc_auc": metric_row.get("roc_auc"),
                "average_precision": metric_row.get("average_precision"),
                "brier": metric_row.get("brier"),
                "max_f1": metric_row.get("max_f1"),
                "best_threshold": metric_row.get("best_threshold"),
                "train_seconds": metric_row.get("train_seconds", manifest.get("train_seconds")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["roc_auc", "average_precision", "brier"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def reject_sample_rows(frame: pd.DataFrame, name: str) -> None:
    if "sample_rows" in frame.columns and frame["sample_rows"].notna().any():
        raise SystemExit(f"{name} contains sample_rows evidence")


if __name__ == "__main__":
    raise SystemExit(main())
