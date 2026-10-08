#!/bin/bash
# DA3 predicted depth at the GT grid (no features), both robots, for offline zero-shot alignment.
source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; cd /home/a1905721/workspace/OpenWAM; export OMP_NUM_THREADS=8
for e in ${@:-agilex franka}; do echo "$(date) da3 depth $e"; /home/a1905721/.conda/envs/depthanything/bin/python -u scripts/extract_robotwin_da3_features.py --embodiment $e --depth-only 2>&1 | grep --line-buffered -v "Warning\|XFORMERS\|warn"; done
echo "$(date) da3 depth done"
