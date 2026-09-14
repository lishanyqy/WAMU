#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${OPEN_WAM_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python3)"
SOURCE="${OPEN_WAM_MODEL_SOURCE:-huggingface}"

ARGS=(--name "${OPEN_WAM_VIDEO_MODEL:-Wan2.2-TI2V-5B}" --source "${SOURCE}" --yes)
if [[ -n "${OPEN_WAM_MODEL_ROOT:-}" ]]; then
  ARGS+=(--root "${OPEN_WAM_MODEL_ROOT}")
fi

"${PYTHON_BIN}" scripts/download_assets/download_video_backbone.py "${ARGS[@]}"

echo "Base video model downloaded and its YAML path was updated."
echo "For a released OpenWAM finetuning checkpoint, use download_openwam_checkpoints.py separately."
