#!/usr/bin/env bash
set -euo pipefail

MODEL_KIND="${1:?pass joint or detach}"
ROOT=/home/a1905721/workspace/OpenWAM
PY=/home/a1905721/.conda/envs/fastwam/bin/python
MANIFEST="$ROOT/artifacts/franka_robotwin10s_manifest.jsonl"
WAN="$ROOT/assets/video_backbone_ckpt/Wan2.2-TI2V-5B"
LOGDIR="$ROOT/artifacts/logs/cron"
mkdir -p "$LOGDIR"

case "$MODEL_KIND" in
  joint) CKPT=/home/a1905721/modelspace/robotwin_dual_system_joint_cross_attention; OUT="$ROOT/artifacts/franka_features/openwam_joint_cross_attention"; LOCK=/tmp/openwam_joint.lock ;;
  detach) CKPT=/home/a1905721/modelspace/robotwin_dual_system_joint_cross_attention_detach; OUT="$ROOT/artifacts/franka_features/openwam_joint_cross_attention_detach"; LOCK=/tmp/openwam_detach.lock ;;
  *) echo "unknown model kind: $MODEL_KIND" >&2; exit 2 ;;
esac

exec 9>"$LOCK"
flock -n 9 || exit 0
set +u
source ~/.bashrc
set -u
module load CUDAcompat/12.2-535.161.08
cd "$ROOT"
exec "$PY" -u scripts/extract_openwam_features.py \
  --ckpt-dir "$CKPT" --wan-dir "$WAN" --manifest "$MANIFEST" \
  --output "$OUT" --device cuda:0
