#!/usr/bin/env python3
"""Regression tests for Task 10 experiment runner contracts."""

from __future__ import annotations

import importlib
import json
import shlex
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fraud_model.configs import get_model_config
from fraud_model.manifest import build_run_manifest, write_manifest


def _write_synthetic_ieee_data(root: Path, train_rows: int = 30, test_rows: int = 6) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train_ids = [1000 + idx for idx in range(train_rows)]
    test_ids = [2000 + idx for idx in range(test_rows)]

    def transaction_rows(ids: list[int], include_target: bool) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for idx, transaction_id in enumerate(ids):
            row: dict[str, object] = {
                "TransactionID": transaction_id,
                "TransactionDT": 10000 + idx * 100,
                "TransactionAmt": 10.0 + (idx % 7) + 0.25,
                "ProductCD": "W" if idx % 2 == 0 else "C",
                "card1": 1000 + idx % 5,
                "card2": 200 + idx % 4,
                "card3": 150,
                "card4": "visa" if idx % 2 == 0 else "mastercard",
                "card5": 100 + idx % 3,
                "card6": "debit" if idx % 2 == 0 else "credit",
                "addr1": 300 + idx % 4,
                "addr2": 87,
                "dist1": float(idx % 6),
                "dist2": float((idx + 2) % 5),
                "P_emaildomain": "gmail.com" if idx % 2 == 0 else "yahoo.com",
                "R_emaildomain": "gmail.com" if idx % 3 == 0 else "hotmail.com",
                "C1": float(idx % 4),
                "D1": float(1 + idx % 6),
                "M1": "T" if idx % 2 == 0 else "F",
                "V1": float((idx % 5) / 10.0),
            }
            if include_target:
                row["isFraud"] = idx % 2
            rows.append(row)
        return rows

    def identity_rows(ids: list[int]) -> list[dict[str, object]]:
        return [
            {
                "TransactionID": transaction_id,
                "id-01": float(idx % 5),
                "DeviceType": "desktop" if idx % 2 == 0 else "mobile",
                "DeviceInfo": f"device-{idx % 3}",
            }
            for idx, transaction_id in enumerate(ids)
        ]

    pd.DataFrame(transaction_rows(train_ids, include_target=True)).to_csv(root / "train_transaction.csv", index=False)
    pd.DataFrame(identity_rows(train_ids)).to_csv(root / "train_identity.csv", index=False)
    pd.DataFrame(transaction_rows(test_ids, include_target=False)).to_csv(root / "test_transaction.csv", index=False)
    pd.DataFrame(identity_rows(test_ids)).to_csv(root / "test_identity.csv", index=False)
    pd.DataFrame({"TransactionID": test_ids, "isFraud": [0.5] * len(test_ids)}).to_csv(
        root / "sample_submission.csv",
        index=False,
    )


def _output_filenames(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.iterdir() if path.is_file())


def _read_config(output_dir: Path) -> dict[str, object]:
    return json.loads((output_dir / "config.json").read_text(encoding="utf-8"))


def _read_manifest(output_dir: Path) -> dict[str, object]:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def _assert_minimal_manifest(
    test_case: unittest.TestCase,
    output_dir: Path,
    *,
    artifact_role: str,
    split_policy: str,
) -> None:
    manifest = _read_manifest(output_dir)
    test_case.assertEqual(manifest["artifact_role"], artifact_role)
    test_case.assertEqual(manifest["split_policy"], split_policy)
    test_case.assertIsInstance(manifest["candidate_id"], str)
    test_case.assertTrue(manifest["candidate_id"])
    test_case.assertIsInstance(manifest["config_hash"], str)
    test_case.assertEqual(len(str(manifest["config_hash"])), 16)


class Task10RunnerImportTest(unittest.TestCase):
    def test_experiment_runners_are_importable_as_modules(self) -> None:
        for module_name in [
            "experiments.run_ablation",
            "experiments.run_month_gap",
            "experiments.run_rolling_oot",
            "experiments.make_submission",
        ]:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


class Task10RunnerArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_dir = self.root / "synthetic ieee data"
        _write_synthetic_ieee_data(self.data_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_oot_runner_writes_expected_artifacts(self) -> None:
        runner = importlib.import_module("experiments.run_oot")
        output_dir = self.root / "oot outputs"

        exit_code = runner.main(
            [
                "--data-dir",
                str(self.data_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "lr",
                "--feature-profile",
                "uid_d",
                "--run-name",
                "unit oot",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            _output_filenames(output_dir),
            [
                "calibration_table.csv",
                "config.json",
                "fit_curve.csv",
                "manifest.json",
                "metrics.csv",
                "metrics.json",
                "predictions_valid.csv",
                "presenter_notes.md",
                "runtime.json",
                "threshold_summary.csv",
            ],
        )
        self.assertEqual(
            list(pd.read_csv(output_dir / "metrics.csv").columns),
            [
                "run",
                "feature_profile",
                "model",
                "rows_train",
                "rows_valid",
                "positive_rate_train",
                "positive_rate_valid",
                "roc_auc",
                "average_precision",
                "brier",
                "max_f1",
                "best_threshold",
                "train_seconds",
            ],
        )
        self.assertEqual(
            list(pd.read_csv(output_dir / "predictions_valid.csv").columns),
            ["TransactionID", "TransactionDT", "isFraud", "lr_score"],
        )
        self.assertEqual(
            list(pd.read_csv(output_dir / "threshold_summary.csv").columns),
            ["model", "threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"],
        )
        self.assertEqual(
            list(pd.read_csv(output_dir / "calibration_table.csv").columns),
            ["model", "bin", "count", "mean_pred", "frac_positive"],
        )
        self.assertEqual(
            list(pd.read_csv(output_dir / "fit_curve.csv").columns),
            ["model", "iteration", "train_loss", "valid_loss", "train_auc", "valid_auc"],
        )
        command = str(_read_config(output_dir)["command"])
        self.assertIn(shlex.quote(str(output_dir)), command)
        self.assertIn(shlex.quote("unit oot"), command)
        config = _read_config(output_dir)
        self.assertEqual(config["feature_profile"], "uid_d")
        self.assertIn("--feature-profile", str(config["command"]))
        _assert_minimal_manifest(self, output_dir, artifact_role="local_validation", split_policy="final_oot")

    def test_oot_runner_accepts_registry_config_id(self) -> None:
        runner = importlib.import_module("experiments.run_oot")
        output_dir = self.root / "oot config id outputs"

        try:
            exit_code = runner.main(
                [
                    "--data-dir",
                    str(self.data_dir),
                    "--output-dir",
                    str(output_dir),
                    "--config-id",
                    "lr_alpha_0",
                    "--run-name",
                    "unit config id",
                ]
            )
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 1

        self.assertEqual(exit_code, 0)
        config = _read_config(output_dir)
        self.assertEqual(config["config_id"], "lr_alpha_0")
        self.assertEqual(config["model"], "lr")
        manifest = _read_manifest(output_dir)
        self.assertEqual(manifest["config_id"], "lr_alpha_0")
        self.assertEqual(manifest["model_family"], "lr")

    def test_rolling_oot_runner_writes_expected_artifacts(self) -> None:
        runner = importlib.import_module("experiments.run_rolling_oot")
        output_dir = self.root / "rolling outputs"

        exit_code = runner.main(
            ["--data-dir", str(self.data_dir), "--output-dir", str(output_dir), "--model", "lr"]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(_output_filenames(output_dir), ["config.json", "manifest.json", "rolling_metrics.csv", "runtime.json"])
        self.assertEqual(
            list(pd.read_csv(output_dir / "rolling_metrics.csv").columns),
            [
                "fold",
                "train_start",
                "train_end",
                "valid_start",
                "valid_end",
                "model",
                "roc_auc",
                "average_precision",
                "brier",
                "max_f1",
                "best_threshold",
                "train_seconds",
            ],
        )
        self.assertIn(shlex.quote(str(output_dir)), str(_read_config(output_dir)["command"]))
        self.assertEqual(_read_config(output_dir)["feature_profile"], "baseline")
        _assert_minimal_manifest(self, output_dir, artifact_role="local_validation", split_policy="rolling_oot")

    def test_month_gap_runner_writes_expected_artifacts(self) -> None:
        runner = importlib.import_module("experiments.run_month_gap")
        output_dir = self.root / "month gap outputs"

        exit_code = runner.main(
            [
                "--data-dir",
                str(self.data_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "lr",
                "--sample-rows",
                "24",
                "--train-fraction",
                "0.5",
                "--gap-fraction",
                "0.25",
                "--valid-fraction",
                "0.25",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(_output_filenames(output_dir), ["config.json", "manifest.json", "month_gap_metrics.csv", "runtime.json"])
        self.assertEqual(
            list(pd.read_csv(output_dir / "month_gap_metrics.csv").columns),
            [
                "run",
                "feature_profile",
                "model",
                "rows_train",
                "rows_valid",
                "train_start",
                "train_end",
                "valid_start",
                "valid_end",
                "roc_auc",
                "average_precision",
                "brier",
                "max_f1",
                "best_threshold",
                "train_seconds",
            ],
        )
        metrics = pd.read_csv(output_dir / "month_gap_metrics.csv")
        self.assertEqual(tuple(metrics[["rows_train", "rows_valid"]].iloc[0]), (12, 6))
        self.assertIn(shlex.quote(str(output_dir)), str(_read_config(output_dir)["command"]))
        manifest = _read_manifest(output_dir)
        self.assertEqual(manifest["sample_rows"], 24)
        self.assertIsInstance(manifest["train_seconds"], float)
        self.assertEqual(manifest["extra"]["rows_loaded"], 24)
        _assert_minimal_manifest(self, output_dir, artifact_role="local_validation", split_policy="month_gap")

    def test_ablation_runner_writes_expected_artifacts(self) -> None:
        runner = importlib.import_module("experiments.run_ablation")
        output_dir = self.root / "ablation outputs"

        exit_code = runner.main(
            ["--data-dir", str(self.data_dir), "--output-dir", str(output_dir), "--model", "lr"]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(_output_filenames(output_dir), ["ablation_summary.csv", "config.json", "manifest.json", "runtime.json"])
        self.assertEqual(
            list(pd.read_csv(output_dir / "ablation_summary.csv").columns),
            ["run", "enabled_groups", "model", "roc_auc", "average_precision", "max_f1", "train_seconds"],
        )
        self.assertIn(shlex.quote(str(output_dir)), str(_read_config(output_dir)["command"]))
        _assert_minimal_manifest(self, output_dir, artifact_role="local_validation", split_policy="ablation")

    def test_submission_runner_writes_expected_artifacts(self) -> None:
        runner = importlib.import_module("experiments.make_submission")
        output_dir = self.root / "submission outputs"

        exit_code = runner.main(
            [
                "--data-dir",
                str(self.data_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "lr",
                "--feature-profile",
                "uid_agg",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(_output_filenames(output_dir), ["config.json", "manifest.json", "runtime.json", "submission.csv"])
        self.assertEqual(list(pd.read_csv(output_dir / "submission.csv").columns), ["TransactionID", "isFraud"])
        self.assertIn(shlex.quote(str(output_dir)), str(_read_config(output_dir)["command"]))
        config = _read_config(output_dir)
        self.assertEqual(config["feature_profile"], "uid_agg")
        self.assertIn("--feature-profile", str(config["command"]))
        _assert_minimal_manifest(self, output_dir, artifact_role="submission", split_policy="submission")

    def test_submission_runner_source_run_dir_carries_manifest_identity(self) -> None:
        runner = importlib.import_module("experiments.make_submission")
        source_run_dir = self.root / "source oot run"
        output_dir = self.root / "submission from source outputs"
        source_run_dir.mkdir()
        source_config = get_model_config("lr_alpha_0")
        source_manifest = build_run_manifest(
            candidate_id="unit_lr_alpha_0_final_oot",
            config=source_config,
            split_policy="final_oot",
            source_run_id="unit-lr-alpha-0-final-oot",
            command="python experiments/run_oot.py --config-id lr_alpha_0",
            output_dir=source_run_dir,
            artifact_role="local_validation",
            train_seconds=0.25,
        )
        write_manifest(source_run_dir / "manifest.json", source_manifest)

        exit_code = runner.main(
            [
                "--data-dir",
                str(self.data_dir),
                "--output-dir",
                str(output_dir),
                "--source-run-dir",
                str(source_run_dir),
            ]
        )

        self.assertEqual(exit_code, 0)
        manifest = _read_manifest(output_dir)
        self.assertEqual(manifest["artifact_role"], "submission")
        for field in [
            "candidate_id",
            "source_run_id",
            "split_policy",
            "config_id",
            "config_hash",
            "model_family",
            "feature_profile",
        ]:
            with self.subTest(field=field):
                self.assertEqual(manifest[field], source_manifest[field])
        self.assertEqual(manifest["source_run_dir"], str(source_run_dir))
        self.assertEqual(manifest["submission_artifact"], str(output_dir / "submission.csv"))
        config = _read_config(output_dir)
        self.assertEqual(config["model"], "lr")
        self.assertEqual(config["config_id"], "lr_alpha_0")

    def test_submission_runner_can_write_postprocess_keys(self) -> None:
        runner = importlib.import_module("experiments.make_submission")
        normal_output_dir = self.root / "submission normal outputs"
        output_dir = self.root / "submission keys outputs"

        normal_exit_code = runner.main(
            [
                "--data-dir",
                str(self.data_dir),
                "--output-dir",
                str(normal_output_dir),
                "--model",
                "lr",
                "--feature-profile",
                "uid_d",
            ]
        )
        exit_code = runner.main(
            [
                "--data-dir",
                str(self.data_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "lr",
                "--feature-profile",
                "uid_d",
                "--write-postprocess-keys",
            ]
        )

        self.assertEqual(normal_exit_code, 0)
        self.assertEqual(exit_code, 0)
        self.assertTrue((output_dir / "postprocess_keys.csv").is_file())
        keys = pd.read_csv(output_dir / "postprocess_keys.csv")
        self.assertIn("TransactionID", keys.columns)
        self.assertIn("UID_D1", keys.columns)
        self.assertEqual(str(keys["UID_D1"].iloc[0]), "1000_300_-1.0")
        pd.testing.assert_frame_equal(
            pd.read_csv(output_dir / "submission.csv"),
            pd.read_csv(normal_output_dir / "submission.csv"),
        )


if __name__ == "__main__":
    unittest.main()
