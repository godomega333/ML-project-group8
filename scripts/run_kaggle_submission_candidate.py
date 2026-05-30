#!/usr/bin/env python3
"""Validate, submit, poll, and ledger a Kaggle submission candidate."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "ieee-fraud-detection"
PENDING_STATUSES = {"SubmissionStatus.PENDING", "SubmissionStatus.RUNNING"}
FAILED_STATUS_MARKERS = ("fail", "error", "cancel")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit one validated IEEE-CIS Kaggle candidate.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--submission-file", type=Path, default=None)
    parser.add_argument("--sample-file", type=Path, default=Path("ieee-fraud-detection-dataset/sample_submission.csv"))
    parser.add_argument("--message", required=True)
    parser.add_argument("--ledger", type=Path, default=Path("docs/modeling-results-ledger.md"))
    parser.add_argument("--notes", default="")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-polls", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_candidate_identity(args.source_run_dir, args.submission_dir)
    args.submission_file = resolve_gated_submission_file(args.submission_dir, args.submission_file)
    validate_submission(args.submission_file, args.sample_file)
    submit_command = kaggle_submit_command(args.submission_file, args.message)
    subprocess.run(submit_command, check=True)
    row = poll_for_submission(args.message, args.poll_seconds, args.max_polls)
    append_submission_record(args, submit_command, row)
    print(
        "Kaggle submission recorded:",
        f"candidate={args.candidate_id}",
        f"ref={row.get('ref', '')}",
        f"status={row.get('status', '')}",
        f"public={row.get('publicScore', '')}",
    )
    return 0


def validate_candidate_identity(source_run_dir: Path, submission_dir: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_candidate_identity.py"),
        "--source-run-dir",
        str(source_run_dir),
        "--submission-dir",
        str(submission_dir),
    ]
    subprocess.run(command, check=True)


def resolve_gated_submission_file(submission_dir: Path, submission_file: Path | None) -> Path:
    gated_submission = submission_dir / "submission.csv"
    manifest_path = submission_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"submission manifest is not valid JSON: {manifest_path}") from exc
    except OSError as exc:
        raise SystemExit(f"submission manifest could not be read: {manifest_path}") from exc

    manifest_artifact = manifest.get("submission_artifact")
    if not isinstance(manifest_artifact, str) or not manifest_artifact:
        raise SystemExit(f"submission manifest missing submission_artifact: {manifest_path}")

    expected = gated_submission.resolve()
    actual_manifest_artifact = Path(manifest_artifact).resolve()
    if actual_manifest_artifact != expected:
        raise SystemExit(
            "submission manifest submission_artifact must match gated artifact: "
            f"{actual_manifest_artifact} != {expected}"
        )

    if submission_file is not None and submission_file.resolve() != expected:
        raise SystemExit(
            "--submission-file must match gated artifact: "
            f"{submission_file.resolve()} != {expected}"
        )

    return gated_submission


def validate_submission(submission_file: Path, sample_file: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_submission.py"),
        "--submission",
        str(submission_file),
        "--sample",
        str(sample_file),
    ]
    subprocess.run(command, check=True)


def kaggle_submit_command(submission_file: Path, message: str) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        "mlw-project",
        "kaggle",
        "competitions",
        "submit",
        COMPETITION,
        "-f",
        str(submission_file),
        "-m",
        message,
    ]


def kaggle_submissions_command() -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        "mlw-project",
        "kaggle",
        "competitions",
        "submissions",
        COMPETITION,
        "--csv",
        "--page-size",
        "20",
    ]


def poll_for_submission(message: str, poll_seconds: float, max_polls: int) -> dict[str, str]:
    row: dict[str, str] | None = None
    for attempt in range(max(1, max_polls)):
        result = subprocess.run(kaggle_submissions_command(), check=True, capture_output=True, text=True)
        row = latest_matching_submission(result.stdout, message)
        if row and not is_pending_status(row.get("status", "")):
            return row
        if attempt < max_polls - 1:
            time.sleep(max(0.0, poll_seconds))
    if row:
        raise SystemExit(
            "Kaggle submission unresolved after polling: "
            f"message={message!r} status={row.get('status', '')!r}"
        )
    raise SystemExit(f"No Kaggle submission found with message: {message}")


def latest_matching_submission(csv_text: str, message: str) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(csv_text))
    matches = [row for row in reader if row.get("description") == message]
    if not matches:
        return {}
    dated_matches = [(parse_kaggle_date(row.get("date", "")), row) for row in matches]
    if any(parsed_date is None for parsed_date, _ in dated_matches):
        return matches[0]
    return max(dated_matches, key=lambda item: item[0] or datetime.min)[1]


def parse_kaggle_date(value: str) -> datetime | None:
    for date_format in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    return None


def is_pending_status(status: str) -> bool:
    return status in PENDING_STATUSES


def submission_decision(status: str) -> str:
    normalized = status.lower()
    if status == "SubmissionStatus.COMPLETE":
        return "candidate"
    if any(marker in normalized for marker in FAILED_STATUS_MARKERS):
        return "rejected"
    return "candidate"


def append_submission_record(args: argparse.Namespace, command: list[str], row: dict[str, str]) -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from append_modeling_ledger import append_record

    record = {
        "kind": "submission",
        "candidate_id": args.candidate_id,
        "source_run_id": args.source_run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "submission_file": str(args.submission_file),
        "validation_status": "passed",
        "kaggle_command": shlex.join(command),
        "submission_ref": row.get("ref", ""),
        "kaggle_status": row.get("status", ""),
        "public_score": row.get("publicScore", ""),
        "private_score": row.get("privateScore", ""),
        "decision": submission_decision(row.get("status", "")),
        "notes": args.notes,
    }
    append_record(args.ledger, record)


if __name__ == "__main__":
    raise SystemExit(main())
