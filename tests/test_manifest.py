#!/usr/bin/env python3
"""Tests for manifest-backed experiment output contracts."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from fraud_model.configs import config_hash, get_model_config
from fraud_model.manifest import (
    IDENTITY_FIELDS,
    candidate_identity,
    build_run_manifest,
    ensure_new_output_dir,
    manifests_match_for_submission,
    read_manifest,
    write_manifest,
)


class ManifestContractTest(unittest.TestCase):
    def test_identity_fields_are_immutable(self) -> None:
        self.assertIsInstance(IDENTITY_FIELDS, tuple)

    def test_ensure_new_output_dir_rejects_existing_non_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            (output_dir / "existing.txt").write_text("already here\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                ensure_new_output_dir(output_dir)

    def test_manifest_round_trips_lr_baseline_local_validation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "lr outputs"
            output_dir.mkdir()
            config = get_model_config("lr_baseline")

            manifest = build_run_manifest(
                candidate_id="lr_baseline_final_oot_baseline",
                config=config,
                split_policy="final_oot",
                source_run_id="unit-lr-final-oot",
                command="python experiments/run_oot.py --model lr",
                output_dir=output_dir,
                artifact_role="local_validation",
                sample_rows=100,
                train_seconds=1.25,
            )
            write_manifest(output_dir / "manifest.json", manifest)

            loaded = read_manifest(output_dir / "manifest.json")

        self.assertEqual(loaded["candidate_id"], "lr_baseline_final_oot_baseline")
        self.assertEqual(loaded["config_id"], "lr_baseline")
        self.assertEqual(loaded["config_hash"], config_hash(config))
        self.assertEqual(len(loaded["config_hash"]), 16)
        self.assertEqual(loaded["artifact_role"], "local_validation")
        self.assertEqual(loaded["config"], config.to_json_dict())
        self.assertEqual(candidate_identity(loaded), {field: loaded[field] for field in IDENTITY_FIELDS})

    def test_manifest_identity_match_for_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = get_model_config("lr_baseline")
            local = build_run_manifest(
                candidate_id="lr_baseline_final_oot_baseline",
                config=config,
                split_policy="final_oot",
                source_run_id="unit-lr-final-oot",
                command="python experiments/run_oot.py --model lr",
                output_dir=Path(tmp) / "local",
                artifact_role="local_validation",
            )
            submission = copy.deepcopy(local)
            submission["artifact_role"] = "submission"
            submission["output_dir"] = str(Path(tmp) / "submission")

        self.assertTrue(manifests_match_for_submission(local, submission))

        changed = copy.deepcopy(submission)
        changed["feature_profile"] = "uid_d"
        self.assertFalse(manifests_match_for_submission(local, changed))


if __name__ == "__main__":
    unittest.main()
