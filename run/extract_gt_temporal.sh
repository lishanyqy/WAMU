#!/bin/bash
# Per-frame (--keep-frames) reference features from the replayed GT RGB; no pooled features.
# Usage: bash run/extract_gt_temporal.sh <agilex|franka>
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM

e=$1
manifest=artifacts/${e}_robotwin_gt_manifest.jsonl
temporal=artifacts/${e}_temporal_features_gt
modelspace=/home/a1905721/modelspace

conda deactivate; conda activate fastwam
python -u scripts/extract_robotwin_dinov2.py --manifest $manifest --output $temporal/dinov2 --device cuda:0 --keep-frames

conda deactivate; conda activate videomae; export LD_PRELOAD=/usr/lib64/libgnutls.so.30
python -u scripts/extract_robotwin_videomaev2_features.py --manifest $manifest --model ${modelspace}/VideoMAEv2-Large --output $temporal/videomaev2 --device cuda:0 --keep-frames

unset LD_PRELOAD; conda deactivate; conda activate depthanything
python -u scripts/extract_robotwin_depth_features.py --manifest $manifest --model-dir ${modelspace}/depthanything --output $temporal/depthv3 --device cuda --keep-frames
