#!/bin/bash
# Extract all OpenWAM variants from GT RGB (L-shape canvas: GT head + official wrist cameras),
# saving unpooled head tokens (9, 30, 8, 10, 3072) per episode (~637 GB per model).
# Usage: bash run/extract_openwam_gt_parallel.sh [jobs] [variant ...]
# Runs up to $1 checkpoints at once on one GPU (default 1; one model needs ~27 GB).
# Output goes to home first and falls back to /hpcfs once home hits its quota.
# Each job resumes by skipping shards present in either location, so it can simply be relaunched.
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
conda activate fastwam
cd /home/a1905721/workspace/OpenWAM

jobs=${1:-1}
export PRIMARY=/home/a1905721/dataspace/openwam_gt_features
export FALLBACK=/hpcfs/users/a1905721/openwam_gt_features
mkdir -p artifacts/logs/openwam_gt
run_variant() {
    v=$1
    for e in agilex franka; do
        python -u scripts/extract_openwam_features.py \
            --ckpt-dir /home/a1905721/modelspace/${v} \
            --wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
            --manifest artifacts/${e}_robotwin_gt_manifest.jsonl \
            --output ${PRIMARY}/${e}/${v} ${FALLBACK}/${e}/${v} \
            --device cuda:0 --seed 0
    done > artifacts/logs/openwam_gt/${v}.log 2>&1
    echo "finished ${v} exit=$?"
}
export -f run_variant
variants=("${@:2}")  # optional: only these variants (default: all 7)
[ ${#variants[@]} -eq 0 ] && variants=(
    robotwin_dual_system_joint_cross_attention
    robotwin_dual_system_joint_cross_attention_detach
    robotwin_dual_system_idm
    robotwin_dual_system_joint_self_attention
    robotwin_dual_system_joint_self_attention_isolated
    robotwin_dual_system_joint_self_attention_mutual
    robotwin_dual_system_joint_self_attention_video_sees_action)
printf '%s\n' "${variants[@]}" | xargs -P ${jobs} -I{} bash -c 'run_variant {}'
