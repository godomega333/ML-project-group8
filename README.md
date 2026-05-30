# Group 8 IEEE-CIS Fraud Detection Code

This package contains the source code, experiment runners, tests, and compact experiment result tables for the AI3013 Machine Learning group project on IEEE-CIS Fraud Detection.

The two submitted model families are implemented from scratch with NumPy and Pandas:

- Logistic regression with mini-batch gradient descent and L2 regularization.
- Histogram gradient boosting with decision-tree learners, shrinkage, subsampling, column sampling, class balancing, and early stopping.

No scikit-learn, TensorFlow, Keras, XGBoost, LightGBM, CatBoost, or similar model-fitting libraries are used in the submitted model implementations.

## Directory Layout

| Path | Contents |
|---|---|
| `src/fraud_model/` | Core data loading, feature engineering, model implementations, metrics, calibration, manifests, and split utilities. |
| `experiments/` | Command-line runners for local validation, search, diagnostics, ablation, reporting tables, and Kaggle submissions. |
| `scripts/` | Utility scripts for environment setup, result recording, batch execution, submission validation, and safety checks. |
| `tests/` | Unit tests for model behavior, metrics, data handling, runner contracts, and artifact validation. |
| `ieee-fraud-detection-dataset/` | Placeholder directory for the Kaggle competition CSV files. The raw dataset is not included because of size and redistribution constraints. |
| `experiment-results/` | Small derived CSV and JSON summaries used to support the report tables and figures. These are not raw data files. |
| `kaggle-submissions/` | Final Kaggle submission CSV files with lightweight config, manifest, and runtime metadata. |
| `environment.yml` | Conda environment definition. |

## Environment Setup On macOS Or Linux

From the package root:

```bash
scripts/setup_environment.sh
```

If the `mlw-project` environment already exists, the script verifies the required packages. If it does not exist, it creates the environment from `environment.yml`.

Manual setup:

```bash
conda env create -f environment.yml
conda run -n mlw-project python -c "import numpy, pandas, matplotlib, PIL; print('ok')"
```

## Environment Setup On Windows

Open Anaconda Prompt or PowerShell in the package root. If Conda is already on `PATH`, run:

```powershell
conda env create -f environment.yml
conda run -n mlw-project python -c "import numpy, pandas, matplotlib, PIL; print('ok')"
```

If the environment already exists, update it instead:

```powershell
conda env update -n mlw-project -f environment.yml
conda run -n mlw-project python -c "import numpy, pandas, matplotlib, PIL; print('ok')"
```

## Dataset Setup

Download the IEEE-CIS Fraud Detection files from Kaggle and place the five CSV files under:

```text
ieee-fraud-detection-dataset/
```

Required files:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

The code discovers these files recursively under the dataset directory, so an extracted Kaggle subfolder is acceptable as long as all required CSV files are inside it.

## Quick Code Check

Run a small deterministic demo first on macOS or Linux:

```bash
PYTHONPATH=src conda run -n mlw-project python experiments/run_demo.py \
  --data-dir ieee-fraud-detection-dataset \
  --output-dir outputs/demo \
  --sample-rows 20000 \
  --model both
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
conda run -n mlw-project python experiments/run_demo.py `
  --data-dir ieee-fraud-detection-dataset `
  --output-dir outputs/demo `
  --sample-rows 20000 `
  --model both
```

This command uses a small sample for a fast code-path check. Its metrics should not be treated as final model evidence.

Run the unit test suite on macOS or Linux:

```bash
PYTHONPATH=src conda run --no-capture-output -n mlw-project python -m unittest discover -s tests -v
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
conda run --no-capture-output -n mlw-project python -m unittest discover -s tests -v
```

## Reproduce Main Local Validation Runs

The final local validation split is chronological: earlier transactions are used for training and the final 20% by `TransactionDT` is used as the out-of-time validation window.

Final boosting contender on macOS or Linux:

```bash
PYTHONPATH=src conda run -n mlw-project python experiments/run_oot.py \
  --data-dir ieee-fraud-detection-dataset \
  --output-dir outputs/final-boosting \
  --model boosting \
  --config-id boosting_search_boundary_mcw20_n200_lr0_08_d6_b64 \
  --split-policy final_oot \
  --run-name final_boosting
```

Final boosting contender on Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
conda run -n mlw-project python experiments/run_oot.py `
  --data-dir ieee-fraud-detection-dataset `
  --output-dir outputs/final-boosting `
  --model boosting `
  --config-id boosting_search_boundary_mcw20_n200_lr0_08_d6_b64 `
  --split-policy final_oot `
  --run-name final_boosting
```

Logistic regression comparison on macOS or Linux:

```bash
PYTHONPATH=src conda run -n mlw-project python experiments/run_oot.py \
  --data-dir ieee-fraud-detection-dataset \
  --output-dir outputs/final-lr \
  --model lr \
  --config-id lr_l2_0_01 \
  --split-policy final_oot \
  --run-name final_lr
```

Logistic regression comparison on Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
conda run -n mlw-project python experiments/run_oot.py `
  --data-dir ieee-fraud-detection-dataset `
  --output-dir outputs/final-lr `
  --model lr `
  --config-id lr_l2_0_01 `
  --split-policy final_oot `
  --run-name final_lr
```

Each run writes `config.json`, `manifest.json`, `metrics.csv`, `metrics.json`, `runtime.json`, validation predictions, threshold summaries, calibration tables, and fit curves to its output directory.

## Kaggle Submission File Generation

The following command trains the selected model on the full training data and writes a local submission CSV on macOS or Linux:

```bash
PYTHONPATH=src conda run -n mlw-project python experiments/make_submission.py \
  --data-dir ieee-fraud-detection-dataset \
  --output-dir outputs/submission-final-boosting \
  --model boosting \
  --config-id boosting_search_boundary_mcw20_n200_lr0_08_d6_b64
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
conda run -n mlw-project python experiments/make_submission.py `
  --data-dir ieee-fraud-detection-dataset `
  --output-dir outputs/submission-final-boosting `
  --model boosting `
  --config-id boosting_search_boundary_mcw20_n200_lr0_08_d6_b64
```

Validate a generated submission against Kaggle's sample file:

```bash
conda run -n mlw-project python scripts/validate_submission.py \
  --submission outputs/submission-final-boosting/submission.csv \
  --sample ieee-fraud-detection-dataset/sample_submission.csv
```

Kaggle credentials are not included in this package.

The generated submission CSV files used for the final recorded Kaggle runs are included under `kaggle-submissions/`. Their public/private scores are summarized in `kaggle-submissions/README.md` and `experiment-results/report-tables/kaggle_submission_summary.csv`.

## Included Result Tables

`experiment-results/report-tables/` contains compact CSV files for model comparison, calibration, threshold behavior, rolling validation, month-gap validation, feature ablation, and Kaggle submission summaries.

`experiment-results/modeling-evidence/` contains the selected evidence package used for report-level claims, including final contender selection, search summaries, LR audit summaries, bootstrap uncertainty, ablation, and observed Kaggle submissions.

These files are derived outputs. They are included so the reported numbers can be inspected without shipping the full generated experiment artifacts.
