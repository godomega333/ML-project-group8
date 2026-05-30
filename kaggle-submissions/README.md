# Kaggle Submissions

This directory contains the final Kaggle submission CSV files generated from the project code. These files are model outputs, not training data.

Each subdirectory contains:

- `submission.csv`: the file submitted to the IEEE-CIS Fraud Detection Kaggle competition.
- `config.json`: command and model configuration used to generate the submission.
- `manifest.json`: candidate identity and validation metadata.
- `runtime.json`: data loading, feature processing, training, and validation runtime metadata.

## Submitted Files

| Directory | Candidate | Kaggle Ref | Public Score | Private Score |
|---|---|---:|---:|---:|
| `submission-boosting_search_boundary_mcw20_n200_lr0_08_d6_b64/` | `boosting_search_boundary_mcw20_n200_lr0_08_d6_b64` | 53127714 | 0.925662 | 0.890898 |
| `submission-boosting_search_boundary_mcw20_n150_lr0_1_d6_b128/` | `boosting_search_boundary_mcw20_n150_lr0_1_d6_b128` | 53127736 | 0.925196 | 0.892038 |
| `submission-boosting_search_boundary_mcw20_n200_lr0_08_d6_b96/` | `boosting_search_boundary_mcw20_n200_lr0_08_d6_b96` | 53127749 | 0.926161 | 0.889969 |
| `submission-boosting_search_boundary_mcw10_n200_lr0_08_d6_b64/` | `boosting_search_boundary_mcw10_n200_lr0_08_d6_b64` | 53127756 | 0.924762 | 0.889043 |
| `submission-boosting_search_boundary_mcw10_n200_lr0_08_d6_b96/` | `boosting_search_boundary_mcw10_n200_lr0_08_d6_b96` | 53127776 | 0.926112 | 0.891415 |
| `submission-lr_l2_0_01/` | `lr_l2_0_01` | 53127779 | 0.868759 | 0.844580 |

The corresponding score ledger is also included as `experiment-results/report-tables/kaggle_submission_summary.csv`.
