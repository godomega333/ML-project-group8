#!/usr/bin/env python3
"""Train a reviewed model on full training data and write a Kaggle submission."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_model.configs import ModelConfig, config_hash, get_model_config
from fraud_model.data import discover_competition_files, load_test_data, load_train_data
from fraud_model.experiment import ensure_output_dir, write_json
from fraud_model.features import FeaturePipeline
from fraud_model.manifest import build_run_manifest, read_manifest, write_manifest

if __package__:
    from .run_oot import (
        fit_model_on_matrices,
        recorded_command,
        reviewed_model_configs,
        runner_identity_slug,
        runner_manifest_config,
    )
else:
    from run_oot import (
        fit_model_on_matrices,
        recorded_command,
        reviewed_model_configs,
        runner_identity_slug,
        runner_manifest_config,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an IEEE-CIS Kaggle submission from a from-scratch model.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Kaggle competition CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for submission artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization.")
    parser.add_argument("--model", choices=["lr", "boosting"], default="boosting", help="Single model path to train.")
    parser.add_argument("--config-id", default=None, help="Canonical config id from fraud_model.configs.")
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Local validation run directory whose manifest should bind this submission identity.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=["baseline", "uid_d", "uid_agg"],
        default="baseline",
        help="Feature engineering profile to use.",
    )
    parser.add_argument(
        "--write-postprocess-keys",
        action="store_true",
        help="Write pseudo-UID keys for compliant smoothing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    source_manifest = read_manifest(args.source_run_dir / "manifest.json") if args.source_run_dir else None
    model_config = (
        _model_config_from_source_manifest(source_manifest)
        if source_manifest is not None
        else get_model_config(args.config_id)
        if args.config_id
        else None
    )
    if model_config is not None:
        args.model = model_config.model_family
        args.feature_profile = model_config.feature_profile
    if args.model not in {"lr", "boosting"}:
        raise ValueError("submission model must be 'lr' or 'boosting'")
    output_dir = ensure_output_dir(args.output_dir)
    total_start = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    command = recorded_command("experiments/make_submission.py", raw_argv)
    files = discover_competition_files(args.data_dir)

    load_start = time.perf_counter()
    train_df, y_train = load_train_data(args.data_dir)
    test_df = load_test_data(args.data_dir)
    load_seconds = time.perf_counter() - load_start

    transform_start = time.perf_counter()
    pipeline = FeaturePipeline(feature_profile=args.feature_profile)
    pipeline.fit(train_df)
    test_features_frame = pipeline.transform_frame(test_df) if args.write_postprocess_keys else None
    postprocess_keys = pipeline.transform_keys(test_df) if args.write_postprocess_keys else None
    train_x = pipeline.transform(train_df, model=args.model)
    test_x = (
        pipeline._matrix_from_frame(test_features_frame, model=args.model)
        if test_features_frame is not None
        else pipeline.transform(test_df, model=args.model)
    )
    transform_seconds = time.perf_counter() - transform_start

    result = fit_model_on_matrices(
        model_name=args.model,
        train_x=train_x,
        y_train=y_train,
        valid_x=None,
        y_valid=None,
        seed=args.seed,
        model_config=model_config,
    )
    probabilities = np.clip(result.estimator.predict_proba(test_x), 0.0, 1.0)
    submission = pd.DataFrame(
        {
            "TransactionID": test_df["TransactionID"].to_numpy(),
            "isFraud": probabilities,
        }
    )

    sample_path = files.get("sample_submission.csv")
    if sample_path is not None and sample_path.is_file():
        sample = pd.read_csv(sample_path, usecols=["TransactionID"])
        submission = sample.merge(submission, on="TransactionID", how="left", validate="one_to_one")
        if submission["isFraud"].isna().any():
            missing = int(submission["isFraud"].isna().sum())
            raise RuntimeError(f"submission is missing predictions for {missing} sample rows")

    submission_path = output_dir / "submission.csv"
    staged_submission_path = output_dir / "submission.unvalidated.csv"
    submission.loc[:, ["TransactionID", "isFraud"]].to_csv(staged_submission_path, index=False)

    validation: dict[str, Any] = {"ran": False, "status": "not_run"}
    if sample_path is not None and sample_path.is_file():
        validate_command = [
            sys.executable,
            str(repo_root / "scripts" / "validate_submission.py"),
            "--submission",
            str(staged_submission_path),
            "--sample",
            str(sample_path),
        ]
        try:
            subprocess.run(validate_command, check=True)
        except subprocess.CalledProcessError:
            staged_submission_path.unlink(missing_ok=True)
            raise
        validation = {"ran": True, "status": "pass", "command": validate_command}

    staged_submission_path.replace(submission_path)
    if postprocess_keys is not None:
        postprocess_keys.to_csv(output_dir / "postprocess_keys.csv", index=False)

    write_json(
        output_dir / "config.json",
        {
            "command": command,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "model": args.model,
            "feature_profile": args.feature_profile,
            "source_run_dir": args.source_run_dir,
            "config_id": model_config.config_id if model_config else None,
            "config_hash": config_hash(model_config) if model_config else None,
            "model_configs": reviewed_model_configs(),
        },
    )
    write_json(
        output_dir / "runtime.json",
        {
            "load_seconds": load_seconds,
            "transform_seconds": transform_seconds,
            "train_seconds": result.train_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "validation": validation,
        },
    )
    manifest = build_run_manifest(
        candidate_id=(
            str(source_manifest["candidate_id"])
            if source_manifest is not None
            else f"submission_{runner_identity_slug(args.model, args.feature_profile, model_config)}"
        ),
        config=runner_manifest_config(args.model, args.feature_profile, model_config),
        split_policy=str(source_manifest["split_policy"]) if source_manifest is not None else "submission",
        source_run_id=(
            str(source_manifest["source_run_id"])
            if source_manifest is not None
            else f"submission_{runner_identity_slug(args.model, args.feature_profile, model_config)}"
        ),
        command=command,
        output_dir=output_dir,
        artifact_role="submission",
        train_seconds=result.train_seconds,
        extra={
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "validation": validation,
        },
    )
    if source_manifest is not None:
        for field in [
            "candidate_id",
            "source_run_id",
            "split_policy",
            "config_id",
            "config_hash",
            "model_family",
            "feature_profile",
        ]:
            manifest[field] = str(source_manifest[field])
    manifest["source_run_dir"] = str(args.source_run_dir) if args.source_run_dir else None
    manifest["submission_artifact"] = str(submission_path)
    manifest["schema_validation_status"] = str(validation["status"])
    write_manifest(
        output_dir / "manifest.json",
        manifest,
    )

    print(f"Submission rows: {len(submission)}")
    print(f"Submission path: {submission_path}")
    return 0


def _model_config_from_source_manifest(source_manifest: dict[str, Any]) -> ModelConfig:
    config_payload = source_manifest.get("config")
    if not isinstance(config_payload, dict):
        return get_model_config(str(source_manifest["config_id"]))
    params = config_payload.get("params")
    if not isinstance(params, dict):
        raise ValueError("source manifest config must include params")
    return ModelConfig(
        config_id=str(config_payload.get("config_id", source_manifest["config_id"])),
        model_family=str(config_payload.get("model_family", source_manifest["model_family"])),
        feature_profile=str(config_payload.get("feature_profile", source_manifest["feature_profile"])),
        params=params,
        class_weight_alpha=config_payload.get("class_weight_alpha"),
        positive_weight=config_payload.get("positive_weight"),
        notes=str(config_payload.get("notes", "")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
