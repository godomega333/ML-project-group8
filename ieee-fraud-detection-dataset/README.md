# Dataset Directory

The raw IEEE-CIS Fraud Detection dataset is not included in this code package because the files are large and should be downloaded from the competition source.

Download the dataset from:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

Place the following files in this directory, or in any subdirectory under it:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

The project code discovers these five files recursively under `ieee-fraud-detection-dataset/`.
