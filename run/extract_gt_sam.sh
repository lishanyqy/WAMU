#!/bin/bash
# SAM ViT-H encoder tokens + oracle-prompted zero-shot IoU on the GT frames, both robots (resumable per episode).
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for e in ${@:-agilex franka}; do
    echo "$(date) sam $e"
    /home/a1905721/.conda/envs/fastwam/bin/python -u scripts/extract_robotwin_sam_features.py --embodiment $e 2>&1 | grep --line-buffered -v "UserWarning\|return self.preprocess"
done
echo "$(date) sam done"
