#!/bin/bash
# Re-extract every representation from the replayed GT RGB (lishanyang/openwam-robotwin-gt),
# so features and GT labels come from the same rendering. Old mp4-based features are kept as-is.
# Usage: bash run/extract_gt_features.sh [refs|openwam|all]   (default: all)
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM

stage=${1:-all}
embodiments="agilex franka"
modelspace=/home/a1905721/modelspace
openwam_variants="robotwin_dual_system_joint_cross_attention
robotwin_dual_system_joint_cross_attention_detach
robotwin_dual_system_idm
robotwin_dual_system_joint_self_attention
robotwin_dual_system_joint_self_attention_isolated
robotwin_dual_system_joint_self_attention_mutual
robotwin_dual_system_joint_self_attention_video_sees_action"
d4rt_ckpt=${modelspace}/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG

if [[ $stage == refs || $stage == all ]]; then
for e in $embodiments; do
    manifest=artifacts/${e}_robotwin_gt_manifest.jsonl
    pooled=artifacts/${e}_features_gt
    temporal=artifacts/${e}_temporal_features_gt

    conda deactivate; conda activate fastwam
    python -u scripts/extract_robotwin_dinov2.py --manifest $manifest --output $pooled/dinov2_large --device cuda:0
    python -u scripts/extract_robotwin_dinov2.py --manifest $manifest --output $temporal/dinov2 --device cuda:0 --keep-frames

    # videomae's cv2 needs the system gnutls; otherwise it fails to import against /lib64/libglib.
    conda deactivate; conda activate videomae; export LD_PRELOAD=/usr/lib64/libgnutls.so.30
    python -u scripts/extract_robotwin_videomaev2_features.py --manifest $manifest --model ${modelspace}/VideoMAEv2-Large --output $pooled/videomaev2_large --device cuda:0
    python -u scripts/extract_robotwin_videomaev2_features.py --manifest $manifest --model ${modelspace}/VideoMAEv2-Large --output $temporal/videomaev2 --device cuda:0 --keep-frames

    unset LD_PRELOAD; conda deactivate; conda activate depthanything
    python -u scripts/extract_robotwin_depth_features.py --manifest $manifest --model-dir ${modelspace}/depthanything --output $pooled/depth_anything3 --device cuda
    python -u scripts/extract_robotwin_depth_features.py --manifest $manifest --model-dir ${modelspace}/depthanything --output $temporal/depthv3 --device cuda --keep-frames

    conda deactivate; conda activate d4rt
    python -u scripts/extract_robotwin_d4rt_encoder.py --manifest $manifest --output $pooled/d4rt_encoder --repo ${modelspace}/d4rt/OpenD4RT_src --config ${d4rt_ckpt}/model.yaml --checkpoint ${d4rt_ckpt}/opend4rt.ckpt --device cuda
done
fi

if [[ $stage == openwam || $stage == all ]]; then
conda deactivate; conda activate fastwam
for v in $openwam_variants; do
    for e in $embodiments; do
        python -u scripts/extract_openwam_features.py \
            --ckpt-dir ${modelspace}/${v} \
            --wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
            --manifest artifacts/${e}_robotwin_gt_manifest.jsonl \
            --output artifacts/${e}_features_gt/${v} \
            --device cuda:0 --seed 0
    done
done
fi
