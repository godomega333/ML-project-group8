#!/usr/bin/env python3
"""Build modeling evidence package for later final-report writing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final-report modeling evidence support artifacts.")
    parser.add_argument("--final-contenders-file", type=Path, required=True)
    parser.add_argument("--kaggle-summary-file", type=Path, required=True)
    parser.add_argument("--bootstrap-file", type=Path, required=True)
    parser.add_argument("--ablation-file", type=Path, required=True)
    parser.add_argument("--lr-audit-summary-file", type=Path, required=True)
    parser.add_argument("--boosting-search-summary-file", type=Path, required=True)
    parser.add_argument("--model-selection-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memo", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contenders = pd.read_csv(args.final_contenders_file)
    kaggle = pd.read_csv(args.kaggle_summary_file)
    bootstrap = pd.read_csv(args.bootstrap_file)
    ablation = pd.read_csv(args.ablation_file)
    lr_audit = pd.read_csv(args.lr_audit_summary_file)
    boosting_search = pd.read_csv(args.boosting_search_summary_file)
    selection = pd.read_csv(args.model_selection_file)

    reject_sample_rows(contenders, "final_contenders")
    reject_sample_rows(ablation, "ablation_summary")
    reject_sample_rows(lr_audit, "lr_audit_summary")
    reject_sample_rows(boosting_search, "boosting_search_summary")
    reject_projected_overclaim(kaggle)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "final_contender_summary.csv": contenders,
        "kaggle_submission_summary.csv": kaggle,
        "bootstrap_uncertainty.csv": bootstrap,
        "ablation_summary.csv": ablation,
        "lr_audit_summary.csv": lr_audit,
        "boosting_search_summary.csv": boosting_search,
        "model_selection_decision.csv": selection,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    write_memo(args.memo, contenders, kaggle, bootstrap, ablation, lr_audit, boosting_search, selection)
    write_json(
        args.output_dir / "modeling_evidence_manifest.json",
        {
            "inputs": {
                "final_contenders_file": str(args.final_contenders_file),
                "kaggle_summary_file": str(args.kaggle_summary_file),
                "bootstrap_file": str(args.bootstrap_file),
                "ablation_file": str(args.ablation_file),
                "lr_audit_summary_file": str(args.lr_audit_summary_file),
                "boosting_search_summary_file": str(args.boosting_search_summary_file),
                "model_selection_file": str(args.model_selection_file),
            },
            "outputs": artifact_metadata(args.output_dir),
            "acceptance_gates": {
                "no_sample_rows": True,
                "ablation_rows": int(len(ablation)),
                "boosting_search_candidate_count": int(len(boosting_search)),
                "lr_audit_candidate_count": int(len(lr_audit)),
                "kaggle_observed_rows_validated": True,
            },
        },
    )
    print(f"Modeling evidence package written to {args.output_dir}")
    return 0


def reject_sample_rows(frame: pd.DataFrame, name: str) -> None:
    if "sample_rows" in frame.columns and frame["sample_rows"].notna().any():
        raise SystemExit(f"{name} contains sample_rows evidence")


def reject_projected_overclaim(kaggle: pd.DataFrame) -> None:
    observed = kaggle.loc[kaggle["observed_or_projected"].astype(str).eq("observed")]
    required = ["ref", "status", "public_score", "private_score"]
    missing = [column for column in required if column not in observed.columns or observed[column].isna().any()]
    if missing:
        raise SystemExit(f"observed Kaggle rows missing fields: {missing}")
    if not observed["status"].astype(str).eq("SubmissionStatus.COMPLETE").all():
        raise SystemExit("observed Kaggle rows must have status SubmissionStatus.COMPLETE")


def artifact_metadata(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*")):
        payload: dict[str, Any] = {
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if path.suffix == ".csv":
            payload["row_count"] = int(len(pd.read_csv(path)))
        rows.append(payload)
    return rows


def write_memo(
    path: Path,
    contenders: pd.DataFrame,
    kaggle: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ablation: pd.DataFrame,
    lr_audit: pd.DataFrame,
    boosting_search: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    final_rows = selection.loc[selection["selected_as"].astype(str).str.contains("final_boosting", case=False, na=False)]
    if final_rows.empty:
        final_line = "当前没有明确 final boosting decision row。"
    else:
        row = final_rows.iloc[0]
        final_line = (
            f"最终建模推荐使用 `{row['candidate_id']}`。"
            f"选择理由：{translate_reason(row['reason'])}"
        )

    observed = kaggle.loc[kaggle["observed_or_projected"].astype(str).eq("observed")]
    best_public = observed.sort_values("public_score", ascending=False).iloc[0]
    best_private = observed.sort_values("private_score", ascending=False).iloc[0]
    bootstrap_row = bootstrap.iloc[0]
    lines = [
        "# Final Report 建模证据备忘录",
        "",
        "本文件只整理建模证据和可写入论文的结论，不撰写 final report 正文，也不编译 LaTeX。",
        "",
        "## 最终建模建议",
        "",
        final_line,
        "",
        "## 关键证据",
        "",
        f"- Final contender 行数：{len(contenders)}",
        f"- LR audit full-scale 候选数：{len(lr_audit)}",
        f"- Boosting bounded-search full-scale 候选数：{len(boosting_search)}",
        f"- Observed Kaggle submission 行数：{len(observed)}",
        f"- Final boosting ablation 行数：{len(ablation)}",
        f"- Bootstrap repeats：{int(bootstrap_row['repeats'])}",
        "",
        "## Kaggle 排名差异",
        "",
        f"- Kaggle public 最优：`{best_public['candidate_id']}`，public `{best_public['public_score']:.6f}`。",
        f"- Kaggle private 最优：`{best_private['candidate_id']}`，private `{best_private['private_score']:.6f}`。",
        "- 最终诊断仍聚焦 local Final OOT winner；local/Kaggle 排名差异需要在论文中明确披露。",
        "",
        "## Bootstrap 结论",
        "",
        f"- 相比 LR anchor，final boosting 的 AUC delta mean 为 `{bootstrap_row['auc_delta_mean']:.6f}`，95% interval 为 `[{bootstrap_row['auc_delta_low']:.6f}, {bootstrap_row['auc_delta_high']:.6f}]`。",
        f"- AP delta mean 为 `{bootstrap_row['ap_delta_mean']:.6f}`，95% interval 为 `[{bootstrap_row['ap_delta_low']:.6f}, {bootstrap_row['ap_delta_high']:.6f}]`。",
        f"- Brier delta mean 为 `{bootstrap_row['brier_delta_mean']:.6f}`，95% interval 为 `[{bootstrap_row['brier_delta_low']:.6f}, {bootstrap_row['brier_delta_high']:.6f}]`；Brier 越低越好。",
        "",
        "## 论文写作口径",
        "",
        "- 只有 ref/status/public/private 同时存在时，Kaggle 分数才能写作 observed evidence。",
        "- Presentation 阶段的 `0.9385/0.9047` 仍是 projected-only，不能写作 observed Kaggle evidence。",
        "- Local OOT、rolling、month-gap、calibration、threshold、bootstrap、ablation 和 runtime 都是本地证据，不是 leaderboard 分数。",
        "- Final report 正文和 LaTeX 编译留到后续论文写作阶段。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def translate_reason(reason: object) -> str:
    known = {
        "Selected for focused diagnostics by primary local Final OOT ranking: highest AUC and AP among boosting candidates; Kaggle public/private remains strong.": (
            "按 primary local Final OOT 排名选择；该候选在 boosting 候选中 AUC 和 AP 最高，且 Kaggle public/private 表现仍然强于 LR anchor。"
        )
    }
    text = str(reason)
    return known.get(text, text)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
