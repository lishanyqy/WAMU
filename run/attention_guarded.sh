#!/bin/bash
# Wait for >= $1 GB free GPU memory, then run scripts/attention_video_share.py with the remaining args (fastwam env).
source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; cd /home/a1905721/workspace/OpenWAM
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
need=$1; shift
freegb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print int(($2-$1)/1024)}'; }
until [ "$(freegb)" -ge "$need" ]; do sleep 60; done
echo "$(date) starting with $(freegb) GB free: $*"
for attempt in 1 2 3 4 5; do
    /home/a1905721/.conda/envs/fastwam/bin/python -u scripts/attention_video_share.py "$@" 2>&1 | grep --line-buffered -v "Warning\|warn\|INFO"
    rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] && break
    echo "$(date) attempt $attempt failed (rc=$rc); waiting for $need GB free"; sleep 120
    until [ "$(freegb)" -ge "$need" ]; do sleep 60; done
done
