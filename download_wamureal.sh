#!/usr/bin/env bash
# Download the private WAMU real dataset using the caller's HF credentials.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash download_wamureal.sh [destination]'
  echo 'Default: <OpenWAM>/datasets/lishanyang_wamu_real'
  echo 'Authenticate first with: hf auth login'
  echo 'Alternatively set HF_TOKEN in your environment (never commit it).'
  echo 'Optional environment: OPEN_WAM_PYTHON, WAMUREAL_REPO (default WAMsU/wamu_real), WAMUREAL_MAX_WORKERS (default 8).'
  exit 0
fi
if (( $# > 1 )); then
  echo 'Usage: bash download_wamureal.sh [destination]' >&2
  exit 2
fi

PYTHON_BIN="${OPEN_WAM_PYTHON:-python3}"
DESTINATION="${1:-${ROOT_DIR}/datasets/lishanyang_wamu_real}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}. Set OPEN_WAM_PYTHON to your environment's Python." >&2
  exit 1
fi

"${PYTHON_BIN}" - "${DESTINATION}" <<'PY'
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, get_token, snapshot_download
except ImportError:
    sys.exit('Missing huggingface_hub. Install in this Python environment: '
             'python -m pip install -U huggingface_hub')

repo_id = os.environ.get('WAMUREAL_REPO', 'WAMsU/wamu_real')
token = get_token()  # HF_TOKEN or the active `hf auth login` credential.
if not token:
    sys.exit('Hugging Face login required. Run `hf auth login` or set HF_TOKEN. '
             f'Your account must have access to {repo_id}.')
try:
    workers = int(os.environ.get('WAMUREAL_MAX_WORKERS', '8'))
    if workers < 1:
        raise ValueError
except ValueError:
    sys.exit('WAMUREAL_MAX_WORKERS must be a positive integer.')

destination = Path(sys.argv[1]).expanduser().resolve()
try:
    api = HfApi(token=token)
    info = api.dataset_info(repo_id)
    print(f'Dataset: {repo_id}; private={info.private}; revision={info.sha}', flush=True)
    print(f'Destination: {destination}', flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        revision=info.sha,
        token=token,
        local_dir=str(destination),
        max_workers=workers,
    )
except Exception as exc:
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status in (401, 403, 404):
        sys.exit('Cannot access the dataset. Check the repository ID, log in with '
                 '`hf auth login`, and ask the owner to grant your account access. '
                 'If using a fine-grained token, grant it read access to this dataset.')
    # Avoid printing request details that could contain authentication data.
    sys.exit(f'Download failed ({type(exc).__name__}). Check network/disk space; '
             'rerun the same command to resume.')

print(f'Download complete: {destination}')
print('Rerunning this command reuses downloaded files and resumes interrupted transfers.')
PY
