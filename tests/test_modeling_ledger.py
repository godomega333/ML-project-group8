#!/usr/bin/env python3
"""Tests for modeling results ledger rendering."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_SCRIPT = REPO_ROOT / "scripts" / "append_modeling_ledger.py"


def load_ledger_module():
    if not LEDGER_SCRIPT.is_file():
        raise AssertionError(f"Ledger script is missing: {LEDGER_SCRIPT}")
    spec = importlib.util.spec_from_file_location("append_modeling_ledger", LEDGER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {LEDGER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelingLedgerTest(unittest.TestCase):
    def test_appends_experiment_and_submission_records(self) -> None:
        ledger = load_ledger_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger_path = root / "ledger.md"
            experiment_json = root / "experiment.json"
            submission_json = root / "submission.json"
            invalid_score_submission_json = root / "invalid-score-submission.json"
            experiment_json.write_text(
                json.dumps(
                    {
                        "kind": "experiment",
                        "run_id": "oot-100k-both",
                        "timestamp": "2026-05-17T12:00:00+08:00",
                        "command": "python experiments/run_oot.py --model both",
                        "data_scope": "sample_rows=100000",
                        "output_dir": "outputs/experiments/oot-100k-both",
                        "metrics": {
                            "lr": {
                                "roc_auc": 0.72,
                                "average_precision": 0.18,
                                "brier": 0.25,
                                "max_f1": 0.31,
                                "best_threshold": 0.7,
                                "train_seconds": 10.5,
                            },
                            "boosting": {
                                "roc_auc": 0.75,
                                "average_precision": 0.24,
                                "brier": 0.09,
                                "max_f1": 0.36,
                                "best_threshold": 0.6,
                                "train_seconds": 20.5,
                            },
                        },
                        "decision": "promote",
                        "reason": "boosting improves AP and Brier in OOT",
                    }
                ),
                encoding="utf-8",
            )
            submission_json.write_text(
                json.dumps(
                    {
                        "kind": "submission",
                        "candidate_id": "S1",
                        "source_run_id": "oot-100k-both",
                        "timestamp": "2026-05-17T12:30:00+08:00",
                        "submission_file": "outputs/experiments/submission-s1/submission.csv",
                        "validation_status": "passed",
                        "kaggle_command": "kaggle competitions submit ieee-fraud-detection -f submission.csv -m S1",
                        "submission_ref": "123456",
                        "kaggle_status": "SubmissionStatus.COMPLETE",
                        "public_score": "0.8123",
                        "private_score": 0.7345,
                        "decision": "candidate",
                        "notes": "first boosting candidate",
                    }
                ),
                encoding="utf-8",
            )
            invalid_score_submission_json.write_text(
                json.dumps(
                    {
                        "kind": "submission",
                        "candidate_id": "S2",
                        "source_run_id": "oot-100k-both",
                        "timestamp": "2026-05-17T13:00:00+08:00",
                        "submission_file": "outputs/experiments/submission-s2/submission.csv",
                        "validation_status": "passed",
                        "kaggle_command": "kaggle competitions submit ieee-fraud-detection -f submission.csv -m S2",
                        "submission_ref": "123457",
                        "kaggle_status": "SubmissionStatus.COMPLETE",
                        "public_score": "nan",
                        "decision": "rejected",
                        "notes": "invalid scores blanked",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(ledger.main(["--ledger", str(ledger_path), "--record", str(experiment_json)]), 0)
            self.assertEqual(ledger.main(["--ledger", str(ledger_path), "--record", str(submission_json)]), 0)
            self.assertEqual(
                ledger.main(["--ledger", str(ledger_path), "--record", str(invalid_score_submission_json)]),
                0,
            )
            content = ledger_path.read_text(encoding="utf-8")

        self.assertIn("# Modeling Results Ledger", content)
        self.assertIn("| oot-100k-both |", content)
        self.assertIn("| boosting | 0.750000 | 0.240000 | 0.090000 |", content)
        self.assertIn("| S1 | oot-100k-both |", content)
        self.assertIn("python experiments/run_oot.py --model both", content)
        self.assertIn("kaggle competitions submit ieee-fraud-detection -f submission.csv -m S1", content)
        self.assertIn("0.812300", content)
        self.assertIn("0.734500", content)
        self.assertIn("## Review Checklist", content)
        self.assertIn(
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|\n"
            "| oot-100k-both |",
            content,
        )
        self.assertIn(
            "| boosting | 0.750000 | 0.240000 | 0.090000 | 0.360000 | 0.600000 | 20.500000 | promote | boosting improves AP and Brier in OOT | outputs/experiments/oot-100k-both | python experiments/run_oot.py --model both |\n"
            "\n"
            "## Submission Records",
            content,
        )
        self.assertIn(
            "|---|---|---|---|---|---|---|---:|---:|---|---|---|\n"
            "| S1 | oot-100k-both |",
            content,
        )
        self.assertIn(
            "| S1 | oot-100k-both | 2026-05-17T12:30:00+08:00 | outputs/experiments/submission-s1/submission.csv | passed | 123456 | SubmissionStatus.COMPLETE | 0.812300 | 0.734500 | candidate | first boosting candidate | kaggle competitions submit ieee-fraud-detection -f submission.csv -m S1 |",
            content,
        )
        self.assertIn(
            "| S2 | oot-100k-both | 2026-05-17T13:00:00+08:00 | outputs/experiments/submission-s2/submission.csv | passed | 123457 | SubmissionStatus.COMPLETE |  |  | rejected | invalid scores blanked | kaggle competitions submit ieee-fraud-detection -f submission.csv -m S2 |",
            content,
        )


if __name__ == "__main__":
    unittest.main()
