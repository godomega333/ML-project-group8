#!/usr/bin/env python3
"""Run a bounded batch of experiment commands from a JSONL manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_REPORT = Path("outputs/experiments/batch-report.json")
MIN_WORKERS = 1
MAX_WORKERS = 6
OUTPUT_TAIL_CHARS = 4000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment commands with bounded concurrency.")
    parser.add_argument("--commands", type=Path, required=True, help="JSONL file with name and command rows.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Path for the batch report JSON.")
    parser.add_argument("--max-workers", type=worker_count, default=2, help="Concurrent command cap, 1 through 6.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report commands without running them.")
    return parser.parse_args(argv)


def worker_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max-workers must be an integer") from exc
    if count < MIN_WORKERS or count > MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"--max-workers must be between {MIN_WORKERS} and {MAX_WORKERS}")
    return count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = load_commands(args.commands)

    if args.dry_run:
        write_report(args.report, build_report(args.max_workers, commands, []))
        print(f"Dry run: {len(commands)} command(s) loaded")
        return 0

    results = run_commands(commands, args.max_workers, log_dir_for_report(args.report))
    write_report(args.report, build_report(args.max_workers, commands, results))
    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        print(f"Batch completed with {len(failed)} failed command(s)")
        return 1
    print(f"Batch completed successfully: {len(results)} command(s)")
    return 0


def load_commands(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"Unable to read commands file: {path}") from exc

    commands: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in {path} line {line_number}: {exc.msg}") from exc
        commands.append(validate_command_row(row, path, line_number))
    return commands


def validate_command_row(row: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SystemExit(f"Command row must be an object in {path} line {line_number}")
    if "name" not in row:
        raise SystemExit(f"Command row missing name in {path} line {line_number}")
    if "command" not in row:
        raise SystemExit(f"Command row missing command in {path} line {line_number}")
    if not isinstance(row["command"], list):
        raise SystemExit(f"Command must be a list in {path} line {line_number}")
    if not row["command"]:
        raise SystemExit(f"Command must not be empty in {path} line {line_number}")
    if not all(isinstance(part, str) for part in row["command"]):
        raise SystemExit(f"Command list entries must be strings in {path} line {line_number}")
    return row


def run_commands(commands: list[dict[str, Any]], max_workers: int, log_dir: Path) -> list[dict[str, Any]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    indexed_results: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_one_command, index, command_row, log_dir): index
            for index, command_row in enumerate(commands)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            indexed_results[index] = future.result()
    return [indexed_results[index] for index in range(len(commands))]


def run_one_command(index: int, command_row: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    command = command_row["command"]
    stdout_log = log_dir / f"{index:04d}.stdout.log"
    stderr_log = log_dir / f"{index:04d}.stderr.log"
    started = time.monotonic()
    try:
        with stdout_log.open("w", encoding="utf-8") as stdout_file, stderr_log.open(
            "w",
            encoding="utf-8",
        ) as stderr_file:
            process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, text=True)
            returncode = process.wait()
        elapsed = time.monotonic() - started
        return {
            "index": index,
            "name": command_row["name"],
            "command": command,
            "returncode": returncode,
            "elapsed_seconds": round(elapsed, 3),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stdout_tail": tail_log_file(stdout_log),
            "stderr_tail": tail_log_file(stderr_log),
        }
    except OSError as exc:
        elapsed = time.monotonic() - started
        stderr_log.write_text(str(exc), encoding="utf-8")
        stdout_log.touch()
        return {
            "index": index,
            "name": command_row["name"],
            "command": command,
            "returncode": 127,
            "elapsed_seconds": round(elapsed, 3),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stdout_tail": "",
            "stderr_tail": tail_log_file(stderr_log),
        }


def tail(value: str) -> str:
    if len(value) <= OUTPUT_TAIL_CHARS:
        return value
    return value[-OUTPUT_TAIL_CHARS:]


def tail_log_file(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - OUTPUT_TAIL_CHARS * 4))
            return handle.read().decode("utf-8", errors="replace")[-OUTPUT_TAIL_CHARS:]
    except OSError as exc:
        return tail(str(exc))


def log_dir_for_report(report_path: Path) -> Path:
    return report_path.parent / f"{report_path.stem}-logs"


def build_report(
    max_workers: int,
    commands: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "max_workers": max_workers,
        "commands": commands,
        "results": results,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
