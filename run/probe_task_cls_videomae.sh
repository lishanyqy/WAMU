#!/bin/bash
# VideoMAEv2 baseline for the per-layer attention-probe task classification, on its unpooled tokens
# (recomputed on the fly; nothing stored). Runs in the videomae env and writes layers_videomaev2.csv
# so it can run alongside run/probe_task_cls_layers.sh.
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
conda deactivate; conda activate videomae
export LD_PRELOAD=/usr/lib64/libgnutls.so.30  # videomae's cv2 needs the system gnutls
cd /home/a1905721/workspace/OpenWAM
for e in ${@:-agilex franka}; do
    python -u scripts/probe_task_cls_layers.py --embodiment $e --refs videomaev2 --models --group 3 --epochs 10 --output artifacts/downstream/task_cls_layers_ep10 --csv-name layers_videomaev2.csv
done
