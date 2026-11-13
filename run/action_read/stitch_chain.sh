#!/bin/bash
# Step 5 (model stitching) on the login node: the Slurm copies cannot be scheduled (QOSMaxCpuPerJobLimit at 200 GB).
# Waits for >= 24 GB free GPU and >= 100 GB available host memory; smoke test first, then video model M1, M2, M3.
source ~/.bashrc >/dev/null 2>&1; module load CUDAcompat/12.2-535.161.08 2>/dev/null
cd /home/a1905721/workspace/OpenWAM
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
LOG=artifacts/logs/action_read/stitch_chain.log; PY=/home/a1905721/.conda/envs/fastwam/bin/python
freegb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print int(($2-$1)/1024)}'; }
freeram() { free -g | awk '/^Mem:/ {print $7}'; }
waitres() { until [ "$(freegb)" -ge 24 ] && [ "$(freeram)" -ge 100 ]; do sleep 120; done; }
echo "$(date) stitch chain start" >> $LOG
waitres; echo "$(date) smoke" >> $LOG
$PY -u scripts/action_read/stitch.py --video-model M1 --limit-windows 2 --train-batch 2 --val-windows 2 --lrs 1e-4 > artifacts/logs/action_read/stitch_smoke.log 2>&1 \
    || { echo "$(date) smoke FAILED (see stitch_smoke.log)" >> $LOG; exit 1; }
mv artifacts/action_read/results/step5/step5_stitch_videoM1.csv artifacts/action_read/results/step5/smoke_step5_stitch_videoM1.csv
for m in M1 M2 M3; do
    waitres; echo "$(date) stitch video $m" >> $LOG
    $PY -u scripts/action_read/stitch.py --video-model $m > artifacts/logs/action_read/stitch_$m.log 2>&1; echo "$(date)   video $m exit $?" >> $LOG
done
echo "$(date) stitch chain done" >> $LOG
