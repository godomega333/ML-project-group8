#!/usr/bin/env python3
"""Plan and run protected boosting search experiments on the inner-tuning split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fraud_model.configs import boosting_search_configs


MIN_SEARCH_CANDIDATES = 40
MAX_SEARCH_CANDIDATES = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and run protected boosting search experiment commands.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Kaggle competition CSV files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for per-config outputs.")
    parser.add_argument("--commands", type=Path, required=True, help="JSONL command manifest to write.")
    parser.add_argument("--plan-only", action="store_true", help="Write commands without running the batch.")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent worker count for the batch runner.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_commands(args.data_dir, args.output_root)
    validate_search_size(commands)
    write_commands(args.commands, commands)

    if args.plan_only:
        print(f"Wrote {len(commands)} boosting search command(s) to {args.commands}")
        return 0

    batch_command = [
        sys.executable,
        "scripts/run_experiment_batch.py",
        "--commands",
        str(args.commands),
        "--report",
        str(args.output_root / "batch-report.json"),
        "--max-workers",
        str(args.max_workers),
    ]
    return subprocess.call(batch_command)


def build_commands(data_dir: Path, output_root: Path) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    for config in boosting_search_configs():
        config_id = config.config_id
        commands.append(
            {
                "name": config_id,
                "command": [
                    sys.executable,
                    "experiments/run_oot.py",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_root / config_id),
                    "--config-id",
                    config_id,
                    "--split-policy",
                    "inner_tuning",
                    "--run-name",
                    config_id,
                ],
            }
        )
    return commands


def validate_search_size(commands: list[dict[str, object]]) -> None:
    count = len(commands)
    if count < MIN_SEARCH_CANDIDATES or count > MAX_SEARCH_CANDIDATES:
        raise SystemExit(
            "Protected boosting search must contain "
            f"{MIN_SEARCH_CANDIDATES}-{MAX_SEARCH_CANDIDATES} candidates; found {count}"
        )


def write_commands(path: Path, commands: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(command) + "\n" for command in commands), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
