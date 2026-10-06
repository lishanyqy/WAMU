#!/bin/bash
# VideoMAEv2 (per-tubelet, all layers) and D4RT (spatial encoder memory) on aligned GT frames 1..32,
# so every pair of tubelets lines up with one OpenWAM latent (aligned frames 4k-3..4k).
# Usage: bash run/extract_gt_video_refs.sh <agilex|franka>
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM

e=$1
manifest=artifacts/${e}_robotwin_gt_manifest.jsonl
temporal=artifacts/${e}_temporal_features_gt
modelspace=/home/a1905721/modelspace
d4rt_ckpt=${modelspace}/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG

conda deactivate; conda activate videomae; export LD_PRELOAD=/usr/lib64/libgnutls.so.30
python -u scripts/extract_robotwin_videomaev2_features.py --manifest $manifest --model ${modelspace}/VideoMAEv2-Large --output $temporal/videomaev2 --device cuda:0 --keep-frames

unset LD_PRELOAD; conda deactivate; conda activate d4rt
python -u scripts/extract_robotwin_d4rt_encoder.py --manifest $manifest --output $temporal/d4rt --repo ${modelspace}/d4rt/OpenD4RT_src --config ${d4rt_ckpt}/model.yaml --checkpoint ${d4rt_ckpt}/opend4rt.ckpt --device cuda
