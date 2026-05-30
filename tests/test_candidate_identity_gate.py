#!/usr/bin/env python3
"""Tests for exact candidate identity validation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fraud_model.configs import get_model_config
from fraud_model.manifest import build_run_manifest, write_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_candidate_identity.py"


class CandidateIdentityGateTest(unittest.TestCase):
    def test_matching_manifests_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            submission_dir = root / "submission"
            config = get_model_config("lr_baseline")
            source = build_run_manifest(
                candidate_id="lr_baseline_final_oot",
                config=config,
                split_policy="final_oot",
                source_run_id="oot-lr",
                command="python experiments/run_oot.py",
                output_dir=source_dir,
                artifact_role="local_validation",
            )
            submission = dict(source)
            submission["artifact_role"] = "submission"
            submission["source_run_dir"] = str(source_dir)
            submission["submission_artifact"] = str(submission_dir / "submission.csv")
            submission["schema_validation_status"] = "pass"
            write_manifest(source_dir / "manifest.json", source)
            write_manifest(submission_dir / "manifest.json", submission)
            (submission_dir / "submission.csv").write_text("TransactionID,isFraud\n1,0.1\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-run-dir",
                    str(source_dir),
                    "--submission-dir",
                    str(submission_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("candidate identity OK", result.stdout)

    def test_mismatched_feature_profile_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            submission_dir = root / "submission"
            config = get_model_config("boosting_config_d")
            source = build_run_manifest(
                candidate_id="boosting_config_d_final_oot",
                config=config,
                split_policy="final_oot",
                source_run_id="oot-d",
                command="python experiments/run_oot.py",
                output_dir=source_dir,
                artifact_role="local_validation",
            )
            submission = dict(source)
            submission["artifact_role"] = "submission"
            submission["feature_profile"] = "uid_agg"
            submission["source_run_dir"] = str(source_dir)
            submission["submission_artifact"] = str(submission_dir / "submission.csv")
            submission["schema_validation_status"] = "pass"
            write_manifest(source_dir / "manifest.json", source)
            write_manifest(submission_dir / "manifest.json", submission)
            (submission_dir / "submission.csv").write_text("TransactionID,isFraud\n1,0.1\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-run-dir",
                    str(source_dir),
                    "--submission-dir",
                    str(submission_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("feature_profile", result.stderr)

    def test_missing_submission_csv_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            submission_dir = root / "submission"
            config = get_model_config("lr_baseline")
            source = build_run_manifest(
                candidate_id="lr_baseline_final_oot",
                config=config,
                split_policy="final_oot",
                source_run_id="oot-lr",
                command="python experiments/run_oot.py",
                output_dir=source_dir,
                artifact_role="local_validation",
            )
            submission = dict(source)
            submission["artifact_role"] = "submission"
            submission["source_run_dir"] = str(source_dir)
            submission["submission_artifact"] = str(submission_dir / "submission.csv")
            submission["schema_validation_status"] = "pass"
            write_manifest(source_dir / "manifest.json", source)
            write_manifest(submission_dir / "manifest.json", submission)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-run-dir",
                    str(source_dir),
                    "--submission-dir",
                    str(submission_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("submission.csv", result.stderr)

    def test_manifest_schema_validation_status_must_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            submission_dir = root / "submission"
            config = get_model_config("lr_baseline")
            source = build_run_manifest(
                candidate_id="lr_baseline_final_oot",
                config=config,
                split_policy="final_oot",
                source_run_id="oot-lr",
                command="python experiments/run_oot.py",
                output_dir=source_dir,
                artifact_role="local_validation",
            )
            submission = dict(source)
            submission["artifact_role"] = "submission"
            submission["source_run_dir"] = str(source_dir)
            submission["submission_artifact"] = str(submission_dir / "submission.csv")
            submission["schema_validation_status"] = "fail"
            write_manifest(source_dir / "manifest.json", source)
            write_manifest(submission_dir / "manifest.json", submission)
            (submission_dir / "submission.csv").write_text("TransactionID,isFraud\n1,0.1\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-run-dir",
                    str(source_dir),
                    "--submission-dir",
                    str(submission_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("schema_validation_status", result.stderr)


if __name__ == "__main__":
    unittest.main()
