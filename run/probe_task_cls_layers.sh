#!/bin/bash
# Per-layer attention-probe task classification on unpooled tokens (10% train / 90% test, few-shot: 10 epochs).
# Usage: bash run/probe_task_cls_layers.sh [embodiment ...]   (default: agilex franka)
# Resumable: layers already in artifacts/downstream/task_cls_layers_ep10/<emb>/layers.csv are skipped.
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM
for e in ${@:-agilex franka}; do
    /home/a1905721/.conda/envs/d4rt/bin/python -u scripts/probe_task_cls_layers.py --embodiment $e --refs d4rt dinov2 --epochs 10 --output artifacts/downstream/task_cls_layers_ep10
done
