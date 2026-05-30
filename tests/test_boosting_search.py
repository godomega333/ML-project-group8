from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import experiments.run_boosting_search as run_boosting_search


class BoostingSearchRunnerTest(unittest.TestCase):
    def assertCommandOptionValue(self, command: list[str], option: str, expected: str) -> None:
        self.assertEqual(command.count(option), 1)
        option_index = command.index(option)
        self.assertLess(option_index + 1, len(command))
        self.assertEqual(command[option_index + 1], expected)

    def test_plan_only_writes_protected_inner_tuning_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "boosting-search"
            commands_path = output_root / "commands.jsonl"

            exit_code = run_boosting_search.main(
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
            self.assertGreaterEqual(len(rows), 40)
            self.assertLessEqual(len(rows), 60)
            self.assertEqual(
                [row["name"] for row in rows],
                [config.config_id for config in run_boosting_search.boosting_search_configs()],
            )

            for row in rows:
                command = row["command"]
                name = row["name"]
                self.assertIn("experiments/run_oot.py", command)
                self.assertCommandOptionValue(command, "--config-id", name)
                self.assertCommandOptionValue(command, "--run-name", name)
                self.assertCommandOptionValue(command, "--output-dir", str(output_root / name))
                self.assertCommandOptionValue(command, "--split-policy", "inner_tuning")
                self.assertNotIn("--sample-rows", json.dumps(row))

    def test_non_plan_mode_delegates_to_batch_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "boosting-search"
            commands_path = root / "commands.jsonl"

            with mock.patch.object(run_boosting_search.subprocess, "call", return_value=0) as call:
                exit_code = run_boosting_search.main(
                    [
                        "--data-dir",
                        str(root / "data"),
                        "--output-root",
                        str(output_root),
                        "--commands",
                        str(commands_path),
                        "--max-workers",
                        "4",
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
            self.assertIn("4", batch_command)


if __name__ == "__main__":
    unittest.main()
