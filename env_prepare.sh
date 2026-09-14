#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

ENV_MANAGER="${OPEN_WAM_ENV_MANAGER:-conda}"
ENV_NAME="${OPEN_WAM_CONDA_ENV:-openwam}"
PYTHON_VERSION="${OPEN_WAM_PYTHON_VERSION:-3.12}"
PYTHON_BIN="${OPEN_WAM_PYTHON:-}"

fail() { echo "[env_prepare] ERROR: $*" >&2; exit 1; }
info() { echo "[env_prepare] $*"; }

case "${1:-}" in
  --uv) ENV_MANAGER=uv ;;
  --conda) ENV_MANAGER=conda ;;
  "") ;;
  *) fail "usage: bash env_prepare.sh [--conda|--uv]" ;;
esac

if [[ "${ENV_MANAGER}" != uv && "${ENV_MANAGER}" != conda ]]; then
  fail "environment manager must be conda or uv"
fi
command -v git >/dev/null || fail "git is required"

version_ok() {
  "${1}" -c 'import sys; raise SystemExit(0 if (sys.version_info[:2] in ((3,11),(3,12))) else 1)' >/dev/null 2>&1
}

if [[ "${ENV_MANAGER}" != conda && -z "${PYTHON_BIN}" && -x .venv/bin/python ]] && version_ok .venv/bin/python; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

if [[ "${ENV_MANAGER}" == uv ]]; then
  UV_BIN="$(command -v uv || true)"
  [[ -n "${UV_BIN}" ]] || fail "uv was requested but is not installed"
  if [[ -n "${PYTHON_BIN}" ]]; then
    version_ok "${PYTHON_BIN}" || fail "OPEN_WAM_PYTHON must be Python 3.11 or 3.12"
    "${UV_BIN}" sync --frozen --python "${PYTHON_BIN}" --extra dev
  else
    if ! "${UV_BIN}" python find "${PYTHON_VERSION}" >/dev/null 2>&1; then
      info "Installing Python ${PYTHON_VERSION} through uv"
      "${UV_BIN}" python install "${PYTHON_VERSION}"
    fi
    "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}" --extra dev
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  fi
else
  command -v conda >/dev/null || fail "conda was requested but is not installed"
  if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    info "Creating conda environment ${ENV_NAME} with Python ${PYTHON_VERSION}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  CONDA_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
  version_ok "${CONDA_PYTHON}" || fail "conda environment ${ENV_NAME} is not Python 3.11/3.12"
  conda run -n "${ENV_NAME}" python -m pip install -e '.[dev]'
  PYTHON_BIN="${CONDA_PYTHON}"
fi

"${PYTHON_BIN}" -m compileall -q openwam scripts
"${PYTHON_BIN}" - <<'PY'
import torch, transformers, diffusers
import openwam
print("[env_prepare] Python/Torch/OpenWAM imports OK")
print(f"[env_prepare] torch={torch.__version__}, cuda={torch.cuda.is_available()}")
PY
info "Environment is ready. Python: ${PYTHON_BIN}"
