#!/bin/bash
# Attention share / entropy / video ablation for cross_attn_direct on the aligned rand probe test episodes (aloha-c2r).
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_c2r.log; M=cross_attn_direct; D=~/modelspace/$M
FARM=/home/a1905721/dataspace/openwam-attn-aligned; D4RT=/home/a1905721/.conda/envs/d4rt/bin/python
echo "$(date) == $M: waiting for download" >> $L
until grep -q "ALL DOWNLOADS DONE" artifacts/logs/download_direct.log && [ -f $D/checkpoint_step_30000.safetensors ] && [ -f $D/config.yaml ]; do
    pgrep -f "download_wams.sh" > /dev/null || { grep -q "ALL DOWNLOADS DONE" artifacts/logs/download_direct.log || { echo "$(date) download died" >> $L; exit 1; }; }; sleep 60; done
echo "$(date) == $M smoke (1 task)" >> $L
bash run/attention_guarded.sh 14 --ckpt-dir $D --embodiment aloha-agilex --dataset-dir $FARM --variant randomized_500 --episodes-per-task 2 --windows 1 \
    --limit-tasks 1 --action-t 0.5 --tag smoke-direct >> artifacts/logs/attention_direct_smoke.log 2>&1
$D4RT - >> $L 2>&1 <<'PY' || { echo "$(date) smoke check failed" >> /home/a1905721/workspace/OpenWAM/artifacts/logs/attention_c2r.log; exit 1; }
import numpy as np
z = np.load('artifacts/attention/cross_attn_direct_smoke-direct.npz', allow_pickle=True); a = z['abl']; c = list(z['abl_cols'])
n, na = a[:, c.index('nmse')].mean(), a[:, c.index('nmse_abl')].mean(); fv = z['frac_video'].mean(); ent = z['entropy'].mean()
print(f'smoke: kind {z["kind"]}, nmse {n:.4f} -> {na:.4f}, video share {fv:.3f}, entropy {ent:.2f}')
assert str(z['kind']) == 'cross' and n < 0.2 and na > n and 0 < fv < 1 and 0 < ent < 12
PY
echo "$(date) == $M randomized_500 (aloha-c2r)" >> $L
EMB=aloha-agilex TAG=aloha-c2r DATASET=$FARM EXTRA="--variant randomized_500 --episodes-per-task 10" MODELS=$M ACTION_T='0.25 0.5 0.75' NEED=14 bash run/attention_sweep.sh
$D4RT scripts/build_attention_report.py aloha-c2r c2r >> $L 2>&1
echo "$(date) $M attention done" >> $L
