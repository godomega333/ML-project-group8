from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import run_lr_audit


class LRAuditRunnerTest(unittest.TestCase):
    def test_plan_only_writes_inner_tuning_commands_for_audit_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "lr-audit"
            commands_path = output_root / "commands.jsonl"

            exit_code = run_lr_audit.main(
                [
                    "--data-dir",
                    str(root / "data"),
                    "--output-root",
                    str(output_root),
                    "--commands",
                    str(commands_path),
                    "--plan-only",
                ]
            )

            self.assertEqual(exit_code, 0)
            rows = [
                json.loads(line)
                for line in commands_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(rows), 10)

            first_command = rows[0]["command"]
            self.assertIn("--config-id", first_command)
            self.assertIn("--split-policy", first_command)
            self.assertIn("inner_tuning", first_command)
            self.assertNotIn("--sample-rows", first_command)

    def test_non_plan_mode_delegates_to_batch_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "lr-audit"
            commands_path = root / "commands.jsonl"

            with mock.patch.object(run_lr_audit.subprocess, "call", return_value=0) as call:
                exit_code = run_lr_audit.main(
                    [
                        "--data-dir",
                        str(root / "data"),
                        "--output-root",
                        str(output_root),
                        "--commands",
                        str(commands_path),
                        "--max-workers",
                        "3",
                    ]
                )

            self.assertEqual(exit_code, 0)
            call.assert_called_once()
            batch_command = call.call_args.args[0]
            self.assertIn("scripts/run_experiment_batch.py", batch_command)
            self.assertIn("--commands", batch_command)
            self.assertIn(str(commands_path), batch_command)
            self.assertIn("--report", batch_command)
            self.assertIn(str(output_root / "batch-report.json"), batch_command)
            self.assertIn("--max-workers", batch_command)
            self.assertIn("3", batch_command)


if __name__ == "__main__":
    unittest.main()
