#!/usr/bin/env bash
set -euo pipefail

secret_path_pattern='(^|/)(kaggle\.json|access_token|\.env)$|(^|/)\.kaggle(/|$)|\.(token|cookie|session)$'

if ! command -v rg >/dev/null 2>&1; then
  echo "rg is required for secret-path verification." >&2
  exit 2
fi

if ! repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "verify_no_tracked_secrets.sh must be run inside a git repository." >&2
  exit 2
fi

if git -C "${repo_root}" ls-files | rg -i "${secret_path_pattern}"; then
  echo "Tracked credential-like path detected. Remove it from git before continuing." >&2
  exit 1
fi

echo "No tracked Kaggle credential paths detected."
