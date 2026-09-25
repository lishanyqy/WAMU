#!/bin/bash

# cd /home/a1905721/workspace/OpenWAM
# conda activate videomae
python -u \
scripts/extract_robotwin_videomaev2_features.py \
--manifest artifacts/agilex_robotwin10s_manifest.jsonl \
--model /home/a1905721/modelspace/VideoMAEv2-Large \
--output artifacts/agilex_features/videomaev2_large \
--device cuda:0

python -u \
scripts/extract_robotwin_videomaev2_features.py \
--manifest artifacts/franka_robotwin10s_manifest.jsonl \
--model /home/a1905721/modelspace/VideoMAEv2-Large \
--output artifacts/franka_features/videomaev2_large \
--device cuda:0