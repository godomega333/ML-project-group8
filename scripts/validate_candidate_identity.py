#!/usr/bin/env python3
"""Validate exact source-run/submission candidate identity before Kaggle upload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fraud_model.manifest import IDENTITY_FIELDS, candidate_identity, read_manifest


PASS_VALUES = {"pass", "passed", "ok", "success", "true"}
MANIFEST_SCHEMA_VALIDATION_PASS = "pass"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate candidate identity between source run and submission.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_manifest = read_manifest(args.source_run_dir / "manifest.json")
    submission_manifest = read_manifest(args.submission_dir / "manifest.json")

    require_role(source_manifest, "source", "local_validation")
    require_role(submission_manifest, "submission", "submission")
    require_manifest_schema_validation_passed(submission_manifest)
    require_submission_csv(args.submission_dir)
    require_schema_validation_sidecar_passed(args.submission_dir / "schema_validation.json")
    require_matching_identity(source_manifest, submission_manifest)

    print("candidate identity OK")
    return 0


def require_role(manifest: dict[str, Any], label: str, expected: str) -> None:
    actual = manifest.get("artifact_role")
    if actual != expected:
        raise SystemExit(f"{label} artifact_role must be {expected}, got {actual!r}")


def require_submission_csv(submission_dir: Path) -> None:
    submission_csv = submission_dir / "submission.csv"
    if not submission_csv.is_file():
        raise SystemExit(f"missing submission.csv: {submission_csv}")


def require_manifest_schema_validation_passed(submission_manifest: dict[str, Any]) -> None:
    actual = submission_manifest.get("schema_validation_status")
    if actual != MANIFEST_SCHEMA_VALIDATION_PASS:
        raise SystemExit(
            "submission manifest schema_validation_status must be "
            f"{MANIFEST_SCHEMA_VALIDATION_PASS!r}, got {actual!r}"
        )


def require_schema_validation_sidecar_passed(schema_path: Path) -> None:
    if not schema_path.exists():
        return
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"schema validation file is not valid JSON: {schema_path}") from exc
    if not schema_validation_passed(payload):
        raise SystemExit(f"schema validation did not pass: {schema_path}")


def schema_validation_passed(payload: Any) -> bool:
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, str):
        return payload.strip().lower() in PASS_VALUES
    if isinstance(payload, dict):
        for key in ("passed", "pass", "ok", "valid"):
            if key in payload:
                return schema_validation_passed(payload[key])
        for key in ("status", "schema_validation_status", "validation_status"):
            if key in payload:
                return schema_validation_passed(payload[key])
    return False


def require_matching_identity(source_manifest: dict[str, Any], submission_manifest: dict[str, Any]) -> None:
    source_identity = candidate_identity(source_manifest)
    submission_identity = candidate_identity(submission_manifest)
    mismatches = [
        field
        for field in IDENTITY_FIELDS
        if source_identity.get(field) != submission_identity.get(field)
    ]
    if not mismatches:
        return
    details = ", ".join(
        f"{field}: {source_identity.get(field)!r} != {submission_identity.get(field)!r}"
        for field in mismatches
    )
    raise SystemExit(f"candidate identity mismatch: {details}")


if __name__ == "__main__":
    raise SystemExit(main())
