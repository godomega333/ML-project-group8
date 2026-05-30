# Scripts

Utility scripts for project setup, experiment bookkeeping, batch execution, and submission validation.

- `setup_environment.sh`: creates or verifies the `mlw-project` Conda environment from `environment.yml`.
- `validate_submission.py`: checks that a generated Kaggle submission matches `sample_submission.csv` in schema, row count, row order, and probability range.
- `validate_candidate_identity.py`: checks that a submission artifact matches the source validation run recorded in its manifest.
- `run_experiment_batch.py`: runs a JSONL list of experiment commands with bounded parallelism.
- `record_experiment_result.py` and `append_modeling_ledger.py`: convert experiment output directories into a Markdown ledger. These scripts are optional for this submitted code package.
- `run_kaggle_submission_candidate.py`: validates a reviewed candidate, submits it through the Kaggle CLI, polls for the score, and records the result.
- `verify_no_tracked_secrets.sh`: checks that credential-like paths are not present in tracked files.

Kaggle credentials are not stored in this package. Use the Kaggle CLI's normal local authentication flow if submission from this package is needed.
