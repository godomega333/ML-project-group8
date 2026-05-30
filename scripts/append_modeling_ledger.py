#!/usr/bin/env python3
"""Append experiment and Kaggle submission records to the modeling ledger."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HEADER = """# Modeling Results Ledger

This ledger records real experiment and Kaggle submission evidence for the final report and presentation preparation phase.

## Review Checklist

Every update must preserve these constraints:

- Re-check `project-requirement.md` and `rubric-for-project.md`.
- Keep LR and boosting as from-scratch primary models.
- Do not introduce forbidden high-level ML APIs into fitting, metrics, calibration, or validation.
- Keep raw data, generated outputs, credentials, tokens, cookies, and large artifacts out of git.
- Record enough local and Kaggle evidence to support final report and presentation writing.

## Experiment Records

| Run ID | Time | Scope | Model | ROC-AUC | AP | Brier | Max-F1 | Threshold | Train Seconds | Decision | Reason | Output | Command |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|

## Submission Records

| Candidate | Source Run | Time | File | Validation | Ref | Status | Public | Private | Decision | Notes | Kaggle Command |
|---|---|---|---|---|---|---|---:|---:|---|---|---|
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a JSON record to docs/modeling-results-ledger.md.")
    parser.add_argument("--ledger", type=Path, default=Path("docs/modeling-results-ledger.md"))
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = json.loads(args.record.read_text(encoding="utf-8"))
    append_record(args.ledger, record)
    return 0


def append_record(ledger_path: Path, record: dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    content = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else HEADER
    kind = str(record.get("kind", ""))
    if kind == "experiment":
        content = insert_after_table(content, "## Experiment Records", experiment_rows(record))
    elif kind == "submission":
        content = insert_after_table(content, "## Submission Records", submission_row(record))
    else:
        raise SystemExit(f"Unknown record kind: {kind}")
    ledger_path.write_text(content, encoding="utf-8")


def insert_after_table(content: str, heading: str, rows: str) -> str:
    lines = content.rstrip().splitlines()
    heading_index = next((index for index, line in enumerate(lines) if line == heading), None)
    if heading_index is None:
        raise SystemExit(f"Ledger missing heading: {heading}")
    insert_index = heading_index + 1
    while insert_index < len(lines) and lines[insert_index].strip() == "":
        insert_index += 1
    while insert_index < len(lines) and lines[insert_index].startswith("|"):
        insert_index += 1
    new_lines = lines[:insert_index] + rows.rstrip().splitlines() + lines[insert_index:]
    return "\n".join(new_lines).rstrip() + "\n"


def experiment_rows(record: dict[str, Any]) -> str:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return experiment_row(record, "summary", {}) + "\n"
    return "\n".join(
        experiment_row(record, str(model_name), model_metrics if isinstance(model_metrics, dict) else {})
        for model_name, model_metrics in metrics.items()
    ) + "\n"


def experiment_row(record: dict[str, Any], model_name: str, metrics: dict[str, Any]) -> str:
    return "| {run_id} | {timestamp} | {scope} | {model} | {auc} | {ap} | {brier} | {f1} | {threshold} | {seconds} | {decision} | {reason} | {output} | {command} |".format(
        run_id=cell(record.get("run_id")),
        timestamp=cell(record.get("timestamp")),
        scope=cell(record.get("data_scope")),
        model=cell(model_name),
        auc=number(metrics.get("roc_auc")),
        ap=number(metrics.get("average_precision")),
        brier=number(metrics.get("brier")),
        f1=number(metrics.get("max_f1")),
        threshold=number(metrics.get("best_threshold")),
        seconds=number(metrics.get("train_seconds")),
        decision=cell(record.get("decision")),
        reason=cell(record.get("reason")),
        output=cell(record.get("output_dir")),
        command=cell(record.get("command")),
    )


def submission_row(record: dict[str, Any]) -> str:
    return "| {candidate} | {source} | {timestamp} | {file} | {validation} | {ref} | {status} | {public} | {private} | {decision} | {notes} | {command} |\n".format(
        candidate=cell(record.get("candidate_id")),
        source=cell(record.get("source_run_id")),
        timestamp=cell(record.get("timestamp")),
        file=cell(record.get("submission_file")),
        validation=cell(record.get("validation_status")),
        ref=cell(record.get("submission_ref")),
        status=cell(record.get("kaggle_status")),
        public=number(record.get("public_score")),
        private=number(record.get("private_score")),
        decision=cell(record.get("decision")),
        notes=cell(record.get("notes")),
        command=cell(record.get("kaggle_command")),
    )


def cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def number(value: Any) -> str:
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number_value):
        return ""
    return f"{number_value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
