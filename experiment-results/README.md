# Experiment Results

This directory contains compact derived result tables from the project experiments. It does not contain the raw Kaggle dataset and does not contain full intermediate model artifacts.

## `report-tables/`

Report-oriented CSV summaries:

- `model_metric_comparison.csv`: final local out-of-time comparison between boosting and logistic regression.
- `feature_profile_comparison.csv`: feature profile comparison for the final models.
- `threshold_curve.csv`: threshold-level classification metrics.
- `calibration_reliability.csv`: calibration table for the final validation predictions.
- `rolling_oot_stability.csv`: rolling chronological validation results.
- `month_gap_stability.csv`: month-gap validation results.
- `bootstrap_uncertainty.csv`: paired bootstrap uncertainty for boosting versus logistic regression.
- `ablation_summary.csv`: feature group ablation results.
- `fit_curve_summary.csv`: training-curve summary.
- `kaggle_submission_summary.csv`: observed Kaggle submission records.
- `lr_audit_summary.csv`, `boosting_search_summary.csv`, `final_contender_summary.csv`, `model_selection_decision.csv`: model search and selection summaries.

## `modeling-evidence/`

Selected evidence tables used to support final model selection and report claims:

- `final_contender_summary.csv`
- `kaggle_submission_summary.csv`
- `model_selection_decision.csv`
- `bootstrap_uncertainty.csv`
- `ablation_summary.csv`
- `lr_audit_summary.csv`
- `boosting_search_summary.csv`
- `modeling_evidence_manifest.json`

These tables are intentionally small so that model evidence can be reviewed without submitting the full `outputs/` experiment directory.
