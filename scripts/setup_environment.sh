#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="mlw-project"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists. Verifying required packages..."
else
  echo "Environment ${ENV_NAME} not found. Creating from environment.yml..."
  conda env create -f environment.yml
fi

conda run -n "${ENV_NAME}" python -c '
import matplotlib
import numpy
import pandas
import PIL

print("mlw-project OK")
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("matplotlib", matplotlib.__version__)
print("pillow", PIL.__version__)
'

conda run -n "${ENV_NAME}" kaggle --version
