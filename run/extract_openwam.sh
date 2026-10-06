#!/bin/bash
openwam_variant=robotwin_dual_system_joint_self_attention_mutual
openwam_ckpt=/home/a1905721/modelspace/${openwam_variant}

openwam_variant2=robotwin_dual_system_idm
openwam_ckpt2=/home/a1905721/modelspace/${openwam_variant2}

# source ~/.bashrc
# module load CUDAcompat/12.2-535.161.08

# cd /home/a1905721/workspace/OpenWAM

# conda activate fastwam

python -u \
scripts/extract_openwam_features.py \
--ckpt-dir $openwam_ckpt \
--wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
--manifest artifacts/agilex_robotwin10s_manifest.jsonl \
--output artifacts/agilex_features/${openwam_variant} \
--device cuda:0

python -u \
scripts/extract_openwam_features.py \
--ckpt-dir $openwam_ckpt \
--wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
--manifest artifacts/franka_robotwin10s_manifest.jsonl \
--output artifacts/franka_features/${openwam_variant} \
--device cuda:0

python -u \
scripts/extract_openwam_features.py \
--ckpt-dir $openwam_ckpt2 \
--wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
--manifest artifacts/agilex_robotwin10s_manifest.jsonl \
--output artifacts/agilex_features/${openwam_variant2} \
--device cuda:0

python -u \
scripts/extract_openwam_features.py \
--ckpt-dir $openwam_ckpt2 \
--wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
--manifest artifacts/franka_robotwin10s_manifest.jsonl \
--output artifacts/franka_features/${openwam_variant2} \
--device cuda:0