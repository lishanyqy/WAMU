#!/bin/bash
# Step 3c attention enrichment for the four bridge cross-attention WAMs on the action_read OOD windows (same as M1-M3).
source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; cd /home/a1905721/workspace/OpenWAM
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
PY=/home/a1905721/.conda/envs/fastwam/bin/python; L=artifacts/logs/enrich_cross.log
F() { grep --line-buffered -v "Warning\|warn\|INFO\|normalizer\|Scene info\|instruction files\|RoboTwinDataset"; }
echo "$(date) == smoke (CA-DIR, 2 windows)" >> $L
$PY -u scripts/action_read/dump_attn_cross.py --name smokeDIR --ckpt cross_attn_direct --set OOD --limit-windows 2 2>&1 | F >> $L
grep -q "^done: 2 windows" $L || { echo "$(date) smoke failed" >> $L; exit 1; }
for spec in "CA cross_attn_clean" "VG cross_attn_detach_robotwin_clean" "DIR cross_attn_direct" "UNI cross_attention_softmax"; do set -- $spec
    echo "$(date) == $1 ($2)" >> $L
    for attempt in 1 2; do
        $PY -u scripts/action_read/dump_attn_cross.py --name $1 --ckpt $2 --set OOD 2>&1 | F >> $L
        [ -f ~/dataspace/action_read_features/$1_OOD/meta.json ] && break
    done
    $PY scripts/action_read/step3_bcd.py --part c --model $1 --set OOD 2>&1 | F >> $L
done
echo "$(date) enrich cross done" >> $L
