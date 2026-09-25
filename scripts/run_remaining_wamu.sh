#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/apps/icl/software/CUDAcompat/12.2-535.161.08/lib:${LD_LIBRARY_PATH:-}
cd /home/a1905721/workspace/OpenWAM
PY=/home/a1905721/.conda/envs/fastwam/bin/python
run_one(){ "$PY" -u scripts/score_robotwin_mknn.py --video-a "artifacts/$1_features/$2" --reference "artifacts/$1_features/$3" --output-dir "artifacts/${1}_mknn/${2}_${3}" --device cuda:0; }
for ds in agilex franka; do
  for model in robotwin_dual_system_joint_self_attention_isolated; do
    for ref in dinov2_large depth_anything3 d4rt_encoder; do run_one "$ds" "$model" "$ref"; done
  done
done
for model in openwam_joint_cross_attention openwam_joint_cross_attention_detach; do
  for ref in dinov2_large depth_anything3 d4rt_encoder; do run_one agilex "$model" "$ref"; done
done
