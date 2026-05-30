#!/usr/bin/env python3
"""Tests for the bounded experiment batch runner."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPT = REPO_ROOT / "scripts" / "run_experiment_batch.py"


def load_batch_module():
    if not BATCH_SCRIPT.is_file():
        raise AssertionError(f"Batch runner script is missing: {BATCH_SCRIPT}")
    spec = importlib.util.spec_from_file_location("run_experiment_batch", BATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BATCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExperimentBatchRunnerTest(unittest.TestCase):
    def test_dry_run_writes_report_without_executing_commands(self) -> None:
        runner = load_batch_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "should-not-exist.txt"
            commands_path = root / "commands.jsonl"
            report_path = root / "reports" / "batch-report.json"
            commands_path.write_text(
                json.dumps(
                    {
                        "name": "would-write-marker",
                        "command": [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    [
                        "--commands",
                        str(commands_path),
                        "--report",
                        str(report_path),
                        "--dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(marker.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["max_workers"], 2)
            self.assertEqual(report["commands"][0]["name"], "would-write-marker")
            self.assertEqual(report["commands"][0]["command"][0], sys.executable)
            self.assertEqual(report["results"], [])

    def test_default_max_workers_is_two(self) -> None:
        runner = load_batch_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            commands_path = Path(temp_dir) / "commands.jsonl"
            commands_path.write_text(
                json.dumps({"name": "noop", "command": [sys.executable, "-c", ""]}) + "\n",
                encoding="utf-8",
            )

            args = runner.parse_args(["--commands", str(commands_path), "--dry-run"])

        self.assertEqual(args.max_workers, 2)

    def test_accepts_six_workers(self) -> None:
        runner = load_batch_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            commands_path = Path(temp_dir) / "commands.jsonl"
            commands_path.write_text(
                json.dumps({"name": "noop", "command": [sys.executable, "-c", ""]}) + "\n",
                encoding="utf-8",
            )

            args = runner.parse_args(["--commands", str(commands_path), "--max-workers", "6"])

        self.assertEqual(args.max_workers, 6)

    def test_rejects_max_workers_above_concurrency_cap(self) -> None:
        runner = load_batch_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            commands_path = Path(temp_dir) / "commands.jsonl"
            commands_path.write_text(
                json.dumps({"name": "noop", "command": [sys.executable, "-c", ""]}) + "\n",
                encoding="utf-8",
            )

            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                runner.parse_args(["--commands", str(commands_path), "--max-workers", "7"])

        self.assertNotEqual(exc.exception.code, 0)

    def test_non_dry_run_writes_results_and_returns_nonzero_on_failure(self) -> None:
        runner = load_batch_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands_path = root / "commands.jsonl"
            report_path = root / "batch-report.json"
            rows = [
                {"name": "success", "command": [sys.executable, "-c", "print('success stdout')"]},
                {
                    "name": "failure",
                    "command": [
                        sys.executable,
                        "-c",
                        "import sys; print('x' * 4500); print('failure stderr', file=sys.stderr); sys.exit(7)",
                    ],
                },
            ]
            commands_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    [
                        "--commands",
                        str(commands_path),
                        "--report",
                        str(report_path),
                        "--max-workers",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual([result["name"] for result in report["results"]], ["success", "failure"])
            self.assertEqual([result["returncode"] for result in report["results"]], [0, 7])
            self.assertIn("success stdout", report["results"][0]["stdout_tail"])
            self.assertLessEqual(len(report["results"][1]["stdout_tail"]), 4000)
            self.assertTrue(report["results"][1]["stdout_tail"].endswith("x" * 3999 + "\n"))
            self.assertIn("failure stderr", report["results"][1]["stderr_tail"])

    def test_non_dry_run_writes_full_logs_and_bounded_tails(self) -> None:
        runner = load_batch_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands_path = root / "commands.jsonl"
            report_path = root / "reports" / "batch-report.json"
            stdout_payload = "stdout-head-" + ("o" * (runner.OUTPUT_TAIL_CHARS + 256)) + "-stdout-tail"
            stderr_payload = "stderr-head-" + ("e" * (runner.OUTPUT_TAIL_CHARS + 256)) + "-stderr-tail"
            command = (
                "import sys; "
                f"sys.stdout.write({stdout_payload!r}); "
                f"sys.stderr.write({stderr_payload!r})"
            )
            commands_path.write_text(
                json.dumps({"name": "large-output", "command": [sys.executable, "-c", command]}) + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    [
                        "--commands",
                        str(commands_path),
                        "--report",
                        str(report_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = report["results"][0]
            stdout_log = Path(result["stdout_log"])
            stderr_log = Path(result["stderr_log"])
            self.assertTrue(stdout_log.is_file())
            self.assertTrue(stderr_log.is_file())
            self.assertEqual(stdout_log.read_text(encoding="utf-8"), stdout_payload)
            self.assertEqual(stderr_log.read_text(encoding="utf-8"), stderr_payload)
            self.assertLessEqual(len(result["stdout_tail"]), runner.OUTPUT_TAIL_CHARS)
            self.assertLessEqual(len(result["stderr_tail"]), runner.OUTPUT_TAIL_CHARS)
            self.assertNotIn("stdout-head", result["stdout_tail"])
            self.assertNotIn("stderr-head", result["stderr_tail"])
            self.assertTrue(result["stdout_tail"].endswith("-stdout-tail"))
            self.assertTrue(result["stderr_tail"].endswith("-stderr-tail"))

    def test_non_dry_run_streams_to_files_without_subprocess_run_capture(self) -> None:
        runner = load_batch_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands_path = root / "commands.jsonl"
            report_path = root / "reports" / "batch-report.json"
            commands_path.write_text(
                json.dumps({"name": "fake-large-output", "command": [sys.executable, "-c", ""]}) + "\n",
                encoding="utf-8",
            )

            class FakeProcess:
                def __init__(self, command, *, stdout, stderr, text):
                    self.command = command
                    self.text = text
                    stdout.write("a" * (runner.OUTPUT_TAIL_CHARS + 10))
                    stderr.write("b" * (runner.OUTPUT_TAIL_CHARS + 10))

                def wait(self):
                    return 0

            with (
                mock.patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=AssertionError("subprocess.run must not buffer command output"),
                ),
                mock.patch.object(runner.subprocess, "Popen", side_effect=FakeProcess) as popen,
                redirect_stdout(io.StringIO()),
            ):
                exit_code = runner.main(
                    [
                        "--commands",
                        str(commands_path),
                        "--report",
                        str(report_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            stdout_target = popen.call_args.kwargs["stdout"]
            stderr_target = popen.call_args.kwargs["stderr"]
            self.assertIsNot(stdout_target, runner.subprocess.PIPE)
            self.assertIsNot(stderr_target, runner.subprocess.PIPE)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = report["results"][0]
            self.assertEqual(len(result["stdout_tail"]), runner.OUTPUT_TAIL_CHARS)
            self.assertEqual(len(result["stderr_tail"]), runner.OUTPUT_TAIL_CHARS)
            self.assertTrue(Path(result["stdout_log"]).is_file())
            self.assertTrue(Path(result["stderr_log"]).is_file())


if __name__ == "__main__":
    raise SystemExit(unittest.main())
