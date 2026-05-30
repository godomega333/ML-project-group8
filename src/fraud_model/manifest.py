"""Manifest contracts for experiment and submission artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fraud_model.configs import config_hash


IDENTITY_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "source_run_id",
    "model_family",
    "config_id",
    "config_hash",
    "feature_profile",
    "split_policy",
)


def ensure_new_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path already exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_run_manifest(
    candidate_id: str,
    config: Any,
    split_policy: str,
    source_run_id: str,
    command: str,
    output_dir: str | Path,
    artifact_role: str,
    sample_rows: int | None = None,
    train_seconds: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_payload = config.to_json_dict()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "candidate_id": str(candidate_id),
        "source_run_id": str(source_run_id),
        "artifact_role": str(artifact_role),
        "model_family": str(config_payload["model_family"]),
        "config_id": str(config_payload["config_id"]),
        "config_hash": config_hash(config),
        "feature_profile": str(config_payload["feature_profile"]),
        "split_policy": str(split_policy),
        "output_dir": str(output_dir),
        "sample_rows": sample_rows if sample_rows is None else int(sample_rows),
        "train_seconds": train_seconds if train_seconds is None else float(train_seconds),
        "command": str(command),
        "config": config_payload,
    }
    if extra is not None:
        manifest["extra"] = _json_clean(dict(extra))
    return _json_clean(manifest)


def candidate_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {field: manifest[field] for field in IDENTITY_FIELDS}


def manifests_match_for_submission(local: Mapping[str, Any], submission: Mapping[str, Any]) -> bool:
    return candidate_identity(local) == candidate_identity(submission)


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_clean(dict(manifest)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
