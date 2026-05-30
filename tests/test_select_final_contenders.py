from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import experiments.select_final_contenders as selector


class FinalContenderSelectorTest(unittest.TestCase):
    def test_ranks_tie_band_and_carries_config_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_run(
                root / "candidate-a",
                candidate_id="candidate-a",
                config_id="cfg_a",
                config_hash="hash-a",
                source_run_id="run-a",
                roc_auc=0.9020,
                average_precision=0.980,
                brier=0.090,
                train_seconds=5.0,
                max_f1=0.440,
            )
            _write_run(
                root / "candidate-b",
                candidate_id="candidate-b",
                config_id="cfg_b",
                config_hash="hash-b",
                source_run_id="run-b",
                roc_auc=0.9030,
                average_precision=0.400,
                brier=0.080,
                train_seconds=2.0,
                max_f1=0.410,
            )
            _write_run(
                root / "candidate-c",
                candidate_id="candidate-c",
                config_id="cfg_c",
                config_hash="hash-c",
                source_run_id="run-c",
                roc_auc=0.9010,
                average_precision=0.990,
                brier=0.070,
                train_seconds=1.0,
                max_f1=0.460,
            )
            _write_run(
                root / "candidate-d",
                candidate_id="candidate-d",
                config_id="cfg_d",
                config_hash="hash-d",
                source_run_id="run-d",
                roc_auc=0.8999,
                average_precision=0.999,
                brier=0.010,
                train_seconds=1.0,
                max_f1=0.470,
            )
            output = root / "top20-inner-tuning.csv"

            exit_code = selector.main(["--search-root", str(root), "--output", str(output), "--top-k", "3"])

            self.assertEqual(exit_code, 0)
            selected = pd.read_csv(output)
            self.assertEqual(
                list(selected.columns),
                [
                    "rank",
                    "run_dir",
                    "candidate_id",
                    "config_id",
                    "config_hash",
                    "source_run_id",
                    "source_run_dir",
                    "model_family",
                    "feature_profile",
                    "split_policy",
                    "sample_rows",
                    "roc_auc",
                    "average_precision",
                    "brier",
                    "train_seconds",
                    "max_f1",
                ],
            )
            self.assertEqual(selected["rank"].tolist(), [1, 2, 3])
            self.assertEqual(selected["config_id"].tolist(), ["cfg_a", "cfg_b", "cfg_c"])
            self.assertEqual(selected["candidate_id"].tolist(), ["candidate-a", "candidate-b", "candidate-c"])
            self.assertEqual(selected["source_run_id"].tolist(), ["run-a", "run-b", "run-c"])
            self.assertEqual(
                selected["source_run_dir"].tolist(),
                [
                    str(root / "candidate-a"),
                    str(root / "candidate-b"),
                    str(root / "candidate-c"),
                ],
            )
            self.assertEqual(selected["model_family"].tolist(), ["boosting", "boosting", "boosting"])
            self.assertEqual(selected["feature_profile"].tolist(), ["baseline", "baseline", "baseline"])
            self.assertEqual(selected["split_policy"].tolist(), ["inner_tuning", "inner_tuning", "inner_tuning"])
            self.assertTrue(selected["sample_rows"].isna().all())
            self.assertEqual(selected["max_f1"].tolist(), [0.44, 0.41, 0.46])

    def test_auc_tie_band_boundary_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_run(
                root / "candidate-a",
                candidate_id="candidate-a",
                config_id="cfg_a",
                config_hash="hash-a",
                source_run_id="run-a",
                roc_auc=0.9030,
                average_precision=0.400,
                brier=0.080,
                train_seconds=2.0,
                max_f1=0.410,
            )
            _write_run(
                root / "candidate-b",
                candidate_id="candidate-b",
                config_id="cfg_b",
                config_hash="hash-b",
                source_run_id="run-b",
                roc_auc=0.9010,
                average_precision=0.999,
                brier=0.010,
                train_seconds=1.0,
                max_f1=0.470,
            )
            _write_run(
                root / "candidate-c",
                candidate_id="candidate-c",
                config_id="cfg_c",
                config_hash="hash-c",
                source_run_id="run-c",
                roc_auc=0.9011,
                average_precision=0.500,
                brier=0.090,
                train_seconds=3.0,
                max_f1=0.420,
            )

            selected = selector.select_contenders(root)

            self.assertEqual(selected["candidate_id"].tolist(), ["candidate-c", "candidate-a", "candidate-b"])

    def test_auc_tie_band_exact_decimal_boundary_is_not_hidden_by_float_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_run(
                root / "candidate-a",
                candidate_id="candidate-a",
                config_id="cfg_a",
                config_hash="hash-a",
                source_run_id="run-a",
                roc_auc=0.0024,
                average_precision=0.400,
                brier=0.080,
                train_seconds=2.0,
                max_f1=0.410,
            )
            _write_run(
                root / "candidate-b",
                candidate_id="candidate-b",
                config_id="cfg_b",
                config_hash="hash-b",
                source_run_id="run-b",
                roc_auc=0.0004,
                average_precision=0.999,
                brier=0.010,
                train_seconds=1.0,
                max_f1=0.470,
            )

            selected = selector.select_contenders(root)

            self.assertEqual(selected["candidate_id"].tolist(), ["candidate-a", "candidate-b"])

    def test_missing_required_metric_columns_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "candidate"
            run_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "model": "boosting",
                        "average_precision": 0.42,
                        "brier": 0.10,
                        "train_seconds": 3.0,
                        "max_f1": 0.45,
                    }
                ]
            ).to_csv(run_dir / "metrics.csv", index=False)
            _write_manifest(run_dir / "manifest.json", candidate_id="candidate", config_id="cfg", source_run_id="run")

            with self.assertRaises(SystemExit) as raised:
                selector.main(["--search-root", str(root), "--output", str(root / "selected.csv")])

            message = str(raised.exception)
            self.assertIn(str(run_dir / "metrics.csv"), message)
            self.assertIn("roc_auc", message)

    def test_reads_identity_from_config_when_manifest_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "candidate"
            run_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "model": "boosting",
                        "roc_auc": 0.88,
                        "average_precision": 0.44,
                        "brier": 0.12,
                        "train_seconds": 4.0,
                        "max_f1": 0.46,
                    }
                ]
            ).to_csv(run_dir / "metrics.csv", index=False)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "config_id": "cfg_from_config",
                        "config_hash": "hash-from-config",
                        "model": "boosting",
                        "feature_profile": "uid_agg",
                        "split_policy": "inner_tuning",
                        "sample_rows": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            selected = selector.select_contenders(root)

            self.assertEqual(selected.loc[0, "candidate_id"], "candidate")
            self.assertEqual(selected.loc[0, "config_id"], "cfg_from_config")
            self.assertEqual(selected.loc[0, "config_hash"], "hash-from-config")
            self.assertEqual(selected.loc[0, "feature_profile"], "uid_agg")
            self.assertEqual(selected.loc[0, "source_run_id"], "candidate")
            self.assertEqual(selected.loc[0, "source_run_dir"], str(run_dir))
            self.assertTrue(pd.isna(selected.loc[0, "sample_rows"]))

    def test_non_null_sample_rows_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_run(
                root / "candidate",
                candidate_id="candidate",
                config_id="cfg",
                config_hash="hash",
                source_run_id="run",
                roc_auc=0.80,
                average_precision=0.30,
                brier=0.20,
                train_seconds=1.0,
                max_f1=0.35,
                sample_rows=1000,
            )

            with self.assertRaises(SystemExit) as raised:
                selector.main(["--search-root", str(root), "--output", str(root / "selected.csv")])

            message = str(raised.exception)
            self.assertIn("sample_rows", message)
            self.assertIn(str(root / "candidate" / "manifest.json"), message)

    def test_partial_manifest_without_source_run_id_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_run(
                root / "candidate",
                candidate_id="candidate",
                config_id="cfg",
                config_hash="hash",
                source_run_id="run",
                roc_auc=0.80,
                average_precision=0.30,
                brier=0.20,
                train_seconds=1.0,
                max_f1=0.35,
            )
            manifest_path = root / "candidate" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["source_run_id"]
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                selector.select_contenders(root)

            message = str(raised.exception)
            self.assertIn(str(root / "candidate"), message)
            self.assertIn("source_run_id", message)

    def test_missing_max_f1_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_run(
                root / "candidate",
                candidate_id="candidate",
                config_id="cfg",
                config_hash="hash",
                source_run_id="run",
                roc_auc=0.80,
                average_precision=0.30,
                brier=0.20,
                train_seconds=1.0,
                max_f1=0.35,
            )
            metrics_path = root / "candidate" / "metrics.csv"
            metrics = pd.read_csv(metrics_path)
            metrics = metrics.drop(columns=["max_f1"])
            metrics.to_csv(metrics_path, index=False)

            with self.assertRaises(SystemExit) as raised:
                selector.select_contenders(root)

            message = str(raised.exception)
            self.assertIn(str(metrics_path), message)
            self.assertIn("max_f1", message)


def _write_run(
    run_dir: Path,
    *,
    candidate_id: str,
    config_id: str,
    config_hash: str,
    source_run_id: str,
    roc_auc: float,
    average_precision: float,
    brier: float,
    train_seconds: float,
    max_f1: float,
    sample_rows: int | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run": candidate_id,
                "model": "boosting",
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "brier": brier,
                "train_seconds": train_seconds,
                "max_f1": max_f1,
            }
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)
    _write_manifest(
        run_dir / "manifest.json",
        candidate_id=candidate_id,
        config_id=config_id,
        config_hash=config_hash,
        source_run_id=source_run_id,
        sample_rows=sample_rows,
    )


def _write_manifest(
    path: Path,
    *,
    candidate_id: str,
    config_id: str,
    config_hash: str = "hash",
    source_run_id: str = "run",
    sample_rows: int | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "config_id": config_id,
                "config_hash": config_hash,
                "source_run_id": source_run_id,
                "model_family": "boosting",
                "feature_profile": "baseline",
                "split_policy": "inner_tuning",
                "sample_rows": sample_rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
