#!/usr/bin/env python3
"""Tests for Kaggle submission candidate automation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from fraud_model.configs import get_model_config
from fraud_model.manifest import build_run_manifest, write_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_kaggle_submission_candidate.py"


def load_submission_module():
    if not SCRIPT.is_file():
        raise AssertionError(f"Submission candidate script is missing: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("run_kaggle_submission_candidate", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_candidate_manifests(root: Path, submission: Path) -> tuple[Path, Path]:
    source_run_dir = root / "source-run"
    submission_dir = submission.parent
    config = get_model_config("lr_baseline")
    source_manifest = build_run_manifest(
        candidate_id="S1",
        config=config,
        split_policy="final_oot",
        source_run_id="oot-full-both",
        command="python experiments/run_oot.py --config-id lr_baseline",
        output_dir=source_run_dir,
        artifact_role="local_validation",
        train_seconds=0.1,
    )
    submission_manifest = dict(source_manifest)
    submission_manifest["artifact_role"] = "submission"
    submission_manifest["source_run_dir"] = str(source_run_dir)
    submission_manifest["submission_artifact"] = str(submission)
    submission_manifest["schema_validation_status"] = "pass"
    write_manifest(source_run_dir / "manifest.json", source_manifest)
    write_manifest(submission_dir / "manifest.json", submission_manifest)
    return source_run_dir, submission_dir


class KaggleSubmissionCandidateTest(unittest.TestCase):
    def test_parses_latest_matching_submission_csv(self) -> None:
        module = load_submission_module()
        csv_text = (
            "ref,fileName,date,description,status,publicScore,privateScore\n"
            "111,old.csv,2026-05-17 01:00:00,S1 old,SubmissionStatus.COMPLETE,0.700000,0.690000\n"
            "222,submission.csv,2026-05-17 02:00:00,S1 boosting default,SubmissionStatus.COMPLETE,0.812345,\n"
        )

        row = module.latest_matching_submission(csv_text, "S1 boosting default")

        self.assertEqual(row["ref"], "222")
        self.assertEqual(row["status"], "SubmissionStatus.COMPLETE")
        self.assertEqual(row["publicScore"], "0.812345")

    def test_latest_matching_submission_returns_empty_dict_without_match(self) -> None:
        module = load_submission_module()
        csv_text = (
            "ref,fileName,date,description,status,publicScore,privateScore\n"
            "111,old.csv,2026-05-17 01:00:00,S1 old,SubmissionStatus.COMPLETE,0.700000,0.690000\n"
        )

        self.assertEqual(module.latest_matching_submission(csv_text, "S1 boosting default"), {})

    def test_latest_matching_submission_returns_latest_duplicate_by_date(self) -> None:
        module = load_submission_module()
        csv_text = (
            "ref,fileName,date,description,status,publicScore,privateScore\n"
            "111,old.csv,2026-05-16 03:46:49.600000,S1 boosting default,SubmissionStatus.COMPLETE,0.700000,0.690000\n"
            "222,submission.csv,2026-05-16 03:47:49.600000,S1 boosting default,SubmissionStatus.COMPLETE,0.812345,\n"
        )

        row = module.latest_matching_submission(csv_text, "S1 boosting default")

        self.assertEqual(row["ref"], "222")

    def test_poll_for_submission_waits_for_completed_row(self) -> None:
        module = load_submission_module()
        outputs = [
            (
                "ref,fileName,date,description,status,publicScore,privateScore\n"
                "333,submission.csv,2026-05-17 02:00:00,S1 boosting default,SubmissionStatus.PENDING,,\n"
            ),
            (
                "ref,fileName,date,description,status,publicScore,privateScore\n"
                "333,submission.csv,2026-05-17 02:01:00,S1 boosting default,SubmissionStatus.COMPLETE,0.812345,\n"
            ),
        ]

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout=outputs.pop(0), stderr="")

        with (
            mock.patch.object(module.subprocess, "run", side_effect=fake_run),
            mock.patch.object(module.time, "sleep") as sleep,
        ):
            row = module.poll_for_submission("S1 boosting default", poll_seconds=5.0, max_polls=2)

        self.assertEqual(row["status"], "SubmissionStatus.COMPLETE")
        self.assertEqual(row["publicScore"], "0.812345")
        sleep.assert_called_once_with(5.0)

    def test_main_raises_without_ledger_when_submission_stays_pending(self) -> None:
        module = load_submission_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            submission = root / "submission.csv"
            sample = root / "sample_submission.csv"
            ledger = root / "ledger.md"
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.1, 0.9]}).to_csv(submission, index=False)
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.5, 0.5]}).to_csv(sample, index=False)
            source_run_dir, submission_dir = write_candidate_manifests(root, submission)

            def fake_run(command, **kwargs):
                if "submissions" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=(
                            "ref,fileName,date,description,status,publicScore,privateScore\n"
                            "333,submission.csv,2026-05-17 02:00:00,S1 boosting default,SubmissionStatus.PENDING,,\n"
                        ),
                        stderr="",
                    )
                return subprocess.CompletedProcess(command, 0, stdout="submission OK\n", stderr="")

            with (
                mock.patch.object(module.subprocess, "run", side_effect=fake_run),
                mock.patch.object(module.time, "sleep"),
                self.assertRaisesRegex(SystemExit, "unresolved|pending|running|timeout"),
            ):
                module.main(
                    [
                        "--candidate-id",
                        "S1",
                        "--source-run-id",
                        "oot-full-both",
                        "--source-run-dir",
                        str(source_run_dir),
                        "--submission-dir",
                        str(submission_dir),
                        "--submission-file",
                        str(submission),
                        "--sample-file",
                        str(sample),
                        "--message",
                        "S1 boosting default",
                        "--ledger",
                        str(ledger),
                        "--poll-seconds",
                        "0",
                        "--max-polls",
                        "2",
                    ]
                )

            self.assertFalse(ledger.exists())

    def test_validate_submit_and_record_uses_expected_commands(self) -> None:
        module = load_submission_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            submission = root / "candidate outputs" / "submission.csv"
            sample = root / "sample file.csv"
            ledger = root / "ledger.md"
            submission.parent.mkdir()
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.1, 0.9]}).to_csv(submission, index=False)
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.5, 0.5]}).to_csv(sample, index=False)
            source_run_dir, submission_dir = write_candidate_manifests(root, submission)

            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append([str(part) for part in command])
                if "submissions" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=(
                            "ref,fileName,date,description,status,publicScore,privateScore\n"
                            "333,submission.csv,2026-05-17 02:00:00,S1 boosting default,SubmissionStatus.COMPLETE,0.812345,\n"
                        ),
                        stderr="",
                    )
                return subprocess.CompletedProcess(command, 0, stdout="submission OK\n", stderr="")

            with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                exit_code = module.main(
                    [
                        "--candidate-id",
                        "S1",
                        "--source-run-id",
                        "oot-full-both",
                        "--source-run-dir",
                        str(source_run_dir),
                        "--submission-dir",
                        str(submission_dir),
                        "--sample-file",
                        str(sample),
                        "--message",
                        "S1 boosting default",
                        "--ledger",
                        str(ledger),
                        "--notes",
                        "unit test candidate",
                        "--poll-seconds",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            identity_call = calls[0]
            validation_call = next(call for call in calls if "validate_submission.py" in " ".join(call))
            submit_call = next(call for call in calls if "submit" in call)
            poll_call = next(call for call in calls if "submissions" in call)
            self.assertEqual(
                identity_call,
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "validate_candidate_identity.py"),
                    "--source-run-dir",
                    str(source_run_dir),
                    "--submission-dir",
                    str(submission_dir),
                ],
            )
            self.assertEqual(
                validation_call,
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "validate_submission.py"),
                    "--submission",
                    str(submission),
                    "--sample",
                    str(sample),
                ],
            )
            self.assertEqual(
                submit_call,
                [
                    "conda",
                    "run",
                    "-n",
                    "mlw-project",
                    "kaggle",
                    "competitions",
                    "submit",
                    "ieee-fraud-detection",
                    "-f",
                    str(submission),
                    "-m",
                    "S1 boosting default",
                ],
            )
            self.assertEqual(
                poll_call,
                [
                    "conda",
                    "run",
                    "-n",
                    "mlw-project",
                    "kaggle",
                    "competitions",
                    "submissions",
                    "ieee-fraud-detection",
                    "--csv",
                    "--page-size",
                    "20",
                ],
            )
            self.assertTrue(ledger.is_file())
            content = ledger.read_text(encoding="utf-8")
            self.assertIn("| S1 | oot-full-both |", content)
            self.assertIn("333", content)
            self.assertIn("0.812345", content)
            self.assertIn(f"'{submission}'", content)
            self.assertIn("-m 'S1 boosting default'", content)

    def test_mismatched_submission_file_is_rejected_before_submit(self) -> None:
        module = load_submission_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gated_submission = root / "candidate outputs" / "submission.csv"
            other_submission = root / "other outputs" / "submission.csv"
            sample = root / "sample file.csv"
            ledger = root / "ledger.md"
            gated_submission.parent.mkdir()
            other_submission.parent.mkdir()
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.1, 0.9]}).to_csv(gated_submission, index=False)
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.2, 0.8]}).to_csv(other_submission, index=False)
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.5, 0.5]}).to_csv(sample, index=False)
            source_run_dir, submission_dir = write_candidate_manifests(root, gated_submission)

            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append([str(part) for part in command])
                return subprocess.CompletedProcess(command, 0, stdout="OK\n", stderr="")

            with (
                mock.patch.object(module.subprocess, "run", side_effect=fake_run),
                self.assertRaisesRegex(SystemExit, "submission-file"),
            ):
                module.main(
                    [
                        "--candidate-id",
                        "S1",
                        "--source-run-id",
                        "oot-full-both",
                        "--source-run-dir",
                        str(source_run_dir),
                        "--submission-dir",
                        str(submission_dir),
                        "--submission-file",
                        str(other_submission),
                        "--sample-file",
                        str(sample),
                        "--message",
                        "S1 boosting default",
                        "--ledger",
                        str(ledger),
                        "--poll-seconds",
                        "0",
                    ]
                )

            self.assertEqual(len(calls), 1)
            self.assertIn("validate_candidate_identity.py", " ".join(calls[0]))
            self.assertFalse(any("submit" in call for call in calls))
            self.assertFalse(ledger.exists())

    def test_failed_terminal_submission_records_rejected_decision(self) -> None:
        module = load_submission_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            submission = root / "submission.csv"
            sample = root / "sample_submission.csv"
            ledger = root / "ledger.md"
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.1, 0.9]}).to_csv(submission, index=False)
            pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0.5, 0.5]}).to_csv(sample, index=False)
            source_run_dir, submission_dir = write_candidate_manifests(root, submission)

            def fake_run(command, **kwargs):
                if "submissions" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=(
                            "ref,fileName,date,description,status,publicScore,privateScore\n"
                            "444,submission.csv,2026-05-17 02:00:00,S2 failed candidate,SubmissionStatus.ERROR,,\n"
                        ),
                        stderr="",
                    )
                return subprocess.CompletedProcess(command, 0, stdout="submission OK\n", stderr="")

            with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                self.assertEqual(
                    module.main(
                        [
                            "--candidate-id",
                            "S2",
                            "--source-run-id",
                            "oot-full-both",
                            "--source-run-dir",
                            str(source_run_dir),
                            "--submission-dir",
                            str(submission_dir),
                            "--submission-file",
                            str(submission),
                            "--sample-file",
                            str(sample),
                            "--message",
                            "S2 failed candidate",
                            "--ledger",
                            str(ledger),
                            "--poll-seconds",
                            "0",
                        ]
                    ),
                    0,
                )

            content = ledger.read_text(encoding="utf-8")
            self.assertIn("| S2 | oot-full-both |", content)
            self.assertIn("SubmissionStatus.ERROR", content)
            self.assertIn("| rejected |", content)


if __name__ == "__main__":
    unittest.main()
