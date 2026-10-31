#!/bin/bash
# Franka attention sweep for the four new checkpoints, each started as soon as its download has landed.
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_sweep.log
for m in sft_from_scratch_robotwin_clean cross_attn_clean cross_attn_detach_robotwin_clean cross_attn_clean_fix3072; do
    until grep -A2 "^== $m$" artifacts/logs/download_wams.log 2>/dev/null | grep -qE "^ +files:"; do sleep 60; done
    [ -f ~/modelspace/$m/checkpoint_step_30000.safetensors ] || { echo "$(date) $m: checkpoint missing after download" >> $L; continue; }
    MODELS="$m" ACTION_T='0.25 0.5 0.75' NEED=14 bash run/attention_sweep.sh
done
/home/a1905721/.conda/envs/d4rt/bin/python scripts/build_attention_report.py franka v2 >> $L 2>&1
echo "$(date) v2 franka report built" >> $L
