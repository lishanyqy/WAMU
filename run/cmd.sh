#!/bin/bash
openwam_variant=robotwin_dual_system_joint_self_attention
openwam_ckpt=/home/a1905721/modelspace/${openwam_variant}


source ~/.bashrc
module load CUDAcompat/12.2-535.161.08

cd /home/a1905721/workspace/OpenWAM


conda activate openwam

python -u \
scripts/extract_openwam_features.py \
--ckpt-dir $openwam_ckpt \
--wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
--manifest artifacts/openwam_robotwin10s_stride4_manifest.jsonl \
--output artifacts/openwam_features/${openwam_ckpt} \
--device cuda:0


# please use 
# export PYTHONPATH=/home/a1905721/workspace/Depth-Anything-3/src:$PYTHONPATH

echo "Starting extracting depth feature"

conda activate d4rt

CUDA_VISIBLE_DEVICES=0 \
python -u \
scripts/extract_robotwin_depth_features.py \
--manifest artifacts/openwam_robotwin10s_stride4_manifest.jsonl \
--model-dir /home/a1905721/modelspace/depthanything \
--output artifacts/openwam_features/depth_anything3 \
--device cuda


source ~/.bashrc
module load CUDAcompat/12.2-535.161.08

CUDA_VISIBLE_DEVICES=0 \
nohup python -u \
scripts/extract_robotwin_dinov2.py \
--manifest artifacts/franka_robotwin10s_manifest.jsonl \
--output artifacts/franka_features/dinov2_large \
--model facebook/dinov2-large \
--device cuda:0 \
--num-frames 33 \
--frame-stride 4 \
> dino_frank.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
/home/a1905721/.conda/envs/fastwam/bin/python -u \
scripts/extract_robotwin_d4rt_encoder.py \
--manifest artifacts/franka_robotwin10s_manifest.jsonl \
--output artifacts/franka_features/d4rt_encoder \
--repo /home/a1905721/modelspace/d4rt/OpenD4RT \
--config /home/a1905721/modelspace/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
--checkpoint /home/a1905721/modelspace/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
--device cuda:0 \
--num-frames 33 > d4rt_frank.log 2>&1 &

nohup python -u \
scripts/extract_robotwin_depth_features.py \
--manifest artifacts/franka_robotwin10s_manifest.jsonl \
--model-dir /home/a1905721/modelspace/depthanything \
--output artifacts/franka_features/depth_anything3 \
--device cuda \
--frames 33 > depth_frank.log 2>&1 &


nohup python -u \
scripts/extract_robotwin_d4rt_encoder.py \
--manifest artifacts/franka_robotwin10s_manifest.jsonl \
--output artifacts/franka_features/d4rt_encoder \
--repo /home/a1905721/modelspace/d4rt/OpenD4RT \
--config /home/a1905721/modelspace/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
--checkpoint /home/a1905721/modelspace/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
--device cuda:0 \
--num-frames 33 > d4rt_frank.log 2>&1 &

