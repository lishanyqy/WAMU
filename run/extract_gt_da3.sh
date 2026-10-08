#!/bin/bash
# Depth Anything 3 backbone tokens + zero-shot depth on the GT frames, both robots (resumable per episode).
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM
export OMP_NUM_THREADS=8
for e in ${@:-agilex franka}; do
    echo "$(date) da3 $e"
    /home/a1905721/.conda/envs/depthanything/bin/python -u scripts/extract_robotwin_da3_features.py --embodiment $e 2>&1 | grep --line-buffered -v "Warning\|XFORMERS\|warn"
done
echo "$(date) da3 done"
