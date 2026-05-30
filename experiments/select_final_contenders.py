#!/usr/bin/env python3
"""Select final contender configurations from protected tuning runs."""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd


AUC_TIE_BAND = 0.002
AUC_TIE_BAND_DECIMAL = Decimal("0.002")
REQUIRED_METRIC_COLUMNS = ("roc_auc", "average_precision", "brier", "train_seconds", "max_f1")
OUTPUT_COLUMNS = (
    "rank",
    "run_dir",
    "candidate_id",
    "config_id",
    "config_hash",
    "source_run_id",
    "source_run_dir",
    "model_family",
    "feature_profile",
    "split_policy",
    "sample_rows",
    "roc_auc",
    "average_precision",
    "brier",
    "train_seconds",
    "max_f1",
)
IDENTITY_FIELDS = (
    "candidate_id",
    "config_id",
    "config_hash",
    "source_run_id",
    "model_family",
    "feature_profile",
    "split_policy",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank final contender runs from protected inner-tuning metrics.")
    parser.add_argument("--search-root", type=Path, required=True, help="Root containing per-candidate run directories.")
    parser.add_argument("--output", type=Path, required=True, help="CSV path for the selected top contenders.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of contenders to write.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    contenders = select_contenders(args.search_root, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contenders.to_csv(args.output, index=False)
    print(f"Wrote {len(contenders)} contender(s) to {args.output}")
    return 0


def select_contenders(search_root: Path, *, top_k: int = 20) -> pd.DataFrame:
    metric_paths = sorted(search_root.glob("*/metrics.csv"), key=_path_sort_key)
    if not metric_paths:
        raise SystemExit(f"No metrics.csv files found under {search_root}")

    records = [_load_candidate(metrics_path) for metrics_path in metric_paths]
    ranked = _rank_records(records)
    return ranked.loc[:, list(OUTPUT_COLUMNS)].head(top_k).reset_index(drop=True)


def _load_candidate(metrics_path: Path) -> dict[str, Any]:
    run_dir = metrics_path.parent
    metrics = pd.read_csv(metrics_path)
    _require_columns(metrics, metrics_path, REQUIRED_METRIC_COLUMNS)
    if len(metrics) != 1:
        raise SystemExit(f"{metrics_path} must contain exactly one candidate row; found {len(metrics)}")

    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    manifest_present = manifest_path.is_file()
    manifest = _read_json_if_present(manifest_path)
    config = _read_json_if_present(config_path)
    _reject_sample_rows(manifest_path, manifest)
    _reject_sample_rows(config_path, config)

    row = metrics.iloc[0].to_dict()
    identity = _candidate_identity(
        run_dir=run_dir,
        manifest=manifest,
        manifest_present=manifest_present,
        config=config,
        metric_row=row,
    )
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "source_run_dir": str(run_dir),
        "sample_rows": _sample_rows(manifest, config),
        **identity,
    }
    for column in REQUIRED_METRIC_COLUMNS:
        record[column] = _finite_number(row[column], path=metrics_path, column=column)
    return record


def _rank_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(records)
    best_auc = float(table["roc_auc"].max())
    top_tie_mask = table["roc_auc"].map(lambda value: _is_in_top_auc_tie_band(best_auc, value))
    deterministic = ["run_dir", "candidate_id", "config_id", "config_hash", "source_run_id"]
    top_tie_band = table.loc[top_tie_mask].sort_values(
        by=["average_precision", "brier", "train_seconds", *deterministic],
        ascending=[False, True, True, True, True, True, True, True],
        kind="mergesort",
    )
    remaining = table.loc[~top_tie_mask].sort_values(
        by=["roc_auc", "average_precision", "brier", "train_seconds", *deterministic],
        ascending=[False, False, True, True, True, True, True, True, True],
        kind="mergesort",
    )
    table = pd.concat([top_tie_band, remaining], ignore_index=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def _is_in_top_auc_tie_band(best_auc: Any, value: Any) -> bool:
    return Decimal(str(best_auc)) - Decimal(str(value)) < AUC_TIE_BAND_DECIMAL


def _candidate_identity(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    manifest_present: bool,
    config: dict[str, Any],
    metric_row: dict[str, Any],
) -> dict[str, str]:
    identity = {
        "candidate_id": _first_present(manifest, config, metric_row, "candidate_id") or run_dir.name,
        "config_id": _first_present(manifest, config, _nested_config(manifest), _nested_config(config), "config_id"),
        "config_hash": _first_present(manifest, config, "config_hash"),
        "source_run_id": (
            _first_present(manifest, "source_run_id")
            if manifest_present
            else _first_present(config, _nested_config(config), metric_row, "source_run_id") or run_dir.name
        ),
        "model_family": _first_present(
            manifest,
            config,
            _nested_config(manifest),
            _nested_config(config),
            metric_row,
            "model_family",
            aliases=("model",),
        ),
        "feature_profile": _first_present(
            manifest,
            config,
            _nested_config(manifest),
            _nested_config(config),
            metric_row,
            "feature_profile",
        ),
        "split_policy": _first_present(manifest, config, "split_policy"),
    }
    missing = [field for field in IDENTITY_FIELDS if _is_nullish(identity.get(field))]
    if missing:
        raise SystemExit(f"{run_dir} is missing required identity field(s): {', '.join(missing)}")
    return {field: str(identity[field]) for field in IDENTITY_FIELDS}


def _require_columns(table: pd.DataFrame, path: Path, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise SystemExit(f"{path} is missing required metric column(s): {', '.join(missing)}")


def _reject_sample_rows(path: Path, payload: dict[str, Any]) -> None:
    if not payload or "sample_rows" not in payload:
        return
    if payload["sample_rows"] is not None:
        raise SystemExit(f"{path} has non-null sample_rows; final contender evidence must use full data")


def _sample_rows(*payloads: dict[str, Any]) -> Any:
    for payload in payloads:
        if isinstance(payload, dict) and "sample_rows" in payload:
            return payload["sample_rows"]
    return None


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Unable to parse JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _nested_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload, dict) else None
    return config if isinstance(config, dict) else {}


def _first_present(*sources: Any, aliases: tuple[str, ...] = ()) -> Any:
    key = sources[-1]
    dictionaries = sources[:-1]
    keys = (key, *aliases)
    for source in dictionaries:
        if not isinstance(source, dict):
            continue
        for candidate_key in keys:
            value = source.get(candidate_key)
            if not _is_nullish(value):
                return value
    return None


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _finite_number(value: Any, *, path: Path, column: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path} column {column} must be numeric; found {value!r}") from exc
    if not math.isfinite(number):
        raise SystemExit(f"{path} column {column} must be finite; found {value!r}")
    return number


def _path_sort_key(path: Path) -> str:
    return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
