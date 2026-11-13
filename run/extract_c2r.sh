#!/bin/bash
# clean2rand (wamu/clean2rand_analysis.md): extract OpenWAM features for the agilex_c2r manifest
# (250 clean train + 500 randomized test episodes) for the five clean-trained models, one at a time, lean mode.
# Resumable: finished episodes are skipped. Usage: bash run/extract_c2r.sh [model ...]
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
conda activate fastwam
cd /home/a1905721/workspace/OpenWAM
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(sft_from_scratch_robotwin_clean cross_attn_clean cross_attn_detach_robotwin_clean
                                   self_attn_detach_robotwin_clean sft_ego_robot_cotrain_robotwin_clean_action_sees_video)
mkdir -p artifacts/logs/openwam_c2r
for m in "${models[@]}"; do
    python -u scripts/extract_openwam_features.py --ckpt-dir /home/a1905721/modelspace/$m \
        --wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B --manifest artifacts/agilex_c2r_manifest.jsonl \
        --output /home/a1905721/dataspace/openwam_gt_features/agilex_c2r/$m --device cuda:0 --seed 0 --lean \
        > artifacts/logs/openwam_c2r/$m.log 2>&1
    echo "$(date '+%F %T') finished $m exit=$? files=$(ls /home/a1905721/dataspace/openwam_gt_features/agilex_c2r/$m | grep -c npz)"
done
