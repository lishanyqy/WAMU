#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /gpfs/apps/icl/software/Anaconda3/2023.03/etc/profile.d/conda.sh
module purge
module load CUDAcompat/12.2-535.161.08

# dino extraction
# conda activate fastwam
# python scripts/extract_robotwin_dinov2.py \
#   --manifest artifacts/agilex_robotwin10s_manifest.jsonl \
#   --output artifacts/agilex_temporal_features/dinov2 \
#   --model facebook/dinov2-large \
#   --device cuda:0 \
#   --num-frames 33 \
#   --frame-stride 4 \
#   --keep-frames


# python scripts/extract_robotwin_dinov2.py \
#   --manifest artifacts/franka_robotwin10s_manifest.jsonl \
#   --output artifacts/franka_temporal_features/dinov2 \
#   --model facebook/dinov2-large \
#   --device cuda:0 \
#   --num-frames 33 \
#   --frame-stride 4 \
#   --keep-frames

echo "workspace=$(pwd)"
echo "cuda=$(python -c 'import os; print(os.environ.get("CUDA_HOME", ""))' 2>/dev/null || true)"

# # depth extraction
conda activate depthanything
export PYTHONPATH=/home/a1905721/workspace/Depth-Anything-3/src:${PYTHONPATH:-}
python -c 'import torch; print("depthanything torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())'
# conda activate openwam
# conda activate 
python scripts/extract_robotwin_depth_features.py \
  --manifest artifacts/agilex_robotwin10s_manifest.jsonl \
  --model-dir /home/a1905721/modelspace/depthanything/ \
  --output artifacts/agilex_temporal_features/depthv3 \
  --device cuda \
  --frames 33 \
  --keep-frames

python scripts/extract_robotwin_depth_features.py \
  --manifest artifacts/franka_robotwin10s_manifest.jsonl \
  --model-dir /home/a1905721/modelspace/depthanything/ \
  --output artifacts/franka_temporal_features/depthv3 \
  --device cuda \
  --frames 33 \
  --keep-frames


# VideoMAEv2
conda activate videomae
python scripts/extract_robotwin_videomaev2_features.py \
  --manifest artifacts/agilex_robotwin10s_manifest.jsonl \
  --model ~/modelspace/VideoMAEv2 \
  --output artifacts/agilex_temporal_features/videomaev2 \
  --device cuda:0 \
  --frames 16 \
  --source-frames 33 \
  --stride 4 \
  --keep-frames


python scripts/extract_robotwin_videomaev2_features.py \
  --manifest artifacts/franka_robotwin10s_manifest.jsonl \
  --model ~/modelspace/VideoMAEv2-large/ \
  --output artifacts/franka_temporal_features/videomaev2 \
  --device cuda:0 \
  --frames 16 \
  --source-frames 33 \
  --stride 4 \
  --keep-frames



# d4rt
conda activate d4rt
python scripts/extract_robotwin_d4rt_encoder.py \
  --manifest artifacts/agilex_robotwin10s_manifest.jsonl \
  --output artifacts/agilex_temporal_features/d4rt \
  --repo ~/workspace/vision/Open-d4rt \
  --config ~/modelspace/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --checkpoint ~/modelspace/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --device cuda \
  --num-frames 32 \
  --keep-frames

python scripts/extract_robotwin_d4rt_encoder.py \
  --manifest artifacts/franka_robotwin10s_manifest.jsonl \
  --output artifacts/franka_temporal_features/d4rt \
  --repo ~/workspace/vision/Open-d4rt \
  --config ~/modelspace/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --checkpoint ~/modelspace/d4rt/OpenD4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --device cuda \
  --num-frames 32 \
  --keep-frames
